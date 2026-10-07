"""Compare zero-coverage gap stitching with whole-reference HMM matches.

Run with PYTHONPATH=src in the tsinfer-match-eval environment; see README.md.
Inputs are local copies of existing pipeline products; TOML paths are never followed.
Canonical alleles in the JSONL mutation records are integers (0 ancestral, 1 derived),
as in tsinfer's native match records. All interval diagnostics use site indices.
"""

import argparse
import csv
import dataclasses
import json
import logging
import math
import multiprocessing
import pathlib
import tomllib

import numpy as np
import tsinfer
import tskit
import yaml

import ch5_stitching

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetSettings:
    """Picklable settings passed to each dataset's worker initializer."""

    name: str
    input_dir: pathlib.Path
    coverage_input_dir: pathlib.Path
    ac_cutoff: int
    min_focal_ac: int
    score_tolerance: float


@dataclasses.dataclass
class FocalIntervals:
    """AC-filtered anchored associations in canonical haplotype/ancestor order."""

    offsets: np.ndarray
    left: np.ndarray
    right: np.ndarray
    nodes: np.ndarray


@dataclasses.dataclass
class WorkerContext:
    """Original reference, observations, and reusable full-model indexes."""

    settings: DatasetSettings
    reference: tskit.TreeSequence
    positions: np.ndarray
    haplotypes: np.ndarray
    focal: dict[str, np.ndarray]
    intervals: FocalIntervals
    populations: list[str]
    rho: np.ndarray
    mu: np.ndarray
    indexes: tsinfer.matching.MatcherIndexes
    log_mismatch_penalty: float
    log_switch_penalty: float


@dataclasses.dataclass
class MatchArrays:
    """Parents and canonical copied alleles on the requested site interval."""

    parents: np.ndarray
    copied: np.ndarray


@dataclasses.dataclass
class FocalMatch:
    """Transient focal arrays and the anchored interval union."""

    arrays: MatchArrays
    mask: np.ndarray


@dataclasses.dataclass
class JobResult:
    """The only two saved matches and their scalar comparison."""

    full: dict
    stitched: dict
    comparison: dict


def load_focal(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read the candidate identities and ragged panel-column lists without pickle."""
    with np.load(path, allow_pickle=False) as archive:
        focal = {name: archive[name] for name in archive.files}
    offsets = focal["offsets"]
    columns = focal["ancestor_index"]
    num_rows = len(focal["sample_id"])
    assert focal["ploidy_index"].shape == (num_rows,)
    assert offsets.shape == (num_rows + 1,)
    assert offsets[0] == 0 and offsets[-1] == len(columns)
    assert np.all(np.diff(offsets) >= 0)
    num_ancestors = len(focal["ancestor_id"])
    assert len(np.unique(focal["ancestor_id"])) == num_ancestors
    assert focal["derived_ac"].shape == (num_ancestors,)
    assert np.all(focal["derived_ac"] > 0)
    assert np.all((columns >= 0) & (columns < num_ancestors))
    return focal


def select_rows(focal: dict, requested: list[dict] | None) -> list[int]:
    """Resolve explicit haplotypes, always preserving the stored NPZ row order."""
    keys = list(zip(focal["sample_id"], focal["ploidy_index"], strict=True))
    assert len(set(keys)) == len(keys), "Duplicate focal haplotype identities"
    if requested is None:
        return list(range(len(keys)))
    requested_keys = [(item["sample_id"], item["ploidy_index"]) for item in requested]
    if len(set(requested_keys)) != len(requested_keys):
        raise ValueError("Duplicate haplotypes in the YAML selection")
    unknown = set(requested_keys) - set(keys)
    if unknown:
        raise ValueError(f"Haplotypes absent from the focal NPZ: {sorted(unknown)}")
    requested_set = set(requested_keys)
    rows = [row for row, key in enumerate(keys) if key in requested_set]
    if len(rows) == 0:
        raise ValueError("Select at least one haplotype")
    return rows


def load_focal_intervals(
    settings: DatasetSettings,
    focal: dict,
    panel,
    positions: np.ndarray,
    column_nodes: np.ndarray,
) -> FocalIntervals:
    """Join budget-zero bounds to original nodes and apply the configured AC range.

    The upstream archive stores associations in increasing ancestor-column order,
    excluding ancestors without focal seeds and those above its generation AC
    limit. Reconstruct that ordering before applying the analysis cutoff.
    """
    interval_path = settings.coverage_input_dir / "haplotype_intervals"
    interval_path /= f"{settings.name}_inferred_focal_ancestor_intervals.npz"
    with np.load(interval_path, allow_pickle=False) as archive:
        interval_offsets = archive["offsets"]
        interval_ac = archive["focal_ac"]
        interval_left = archive["left_site_index"][:, 0]
        interval_right = archive["right_site_index"][:, 0]
        generated_ac = int(archive["max_ac_cutoff"])
        assert settings.ac_cutoff <= generated_ac
        assert int(archive["num_sites"]) == len(positions)
        for field in ("sample_id", "ploidy_index"):
            assert np.array_equal(archive[field], focal[field])
    coverage_panel = tsinfer.vcz.open_store(
        settings.coverage_input_dir
        / "ancestors"
        / f"{settings.name}_inferred_ancestors.zarr"
    )
    assert np.array_equal(coverage_panel["variant_position"][:], positions)
    assert np.array_equal(coverage_panel["sample_id"][:], panel["sample_id"][:])
    focal_positions = panel["sample_focal_positions"][:]
    has_anchor = np.any(focal_positions >= 0, axis=1)
    association_columns = []
    for row in range(len(focal["sample_id"])):
        columns = focal["ancestor_index"][
            focal["offsets"][row] : focal["offsets"][row + 1]
        ]
        eligible = (focal["derived_ac"][columns] <= generated_ac) & has_anchor[columns]
        columns = columns[eligible]
        assert np.all(np.diff(columns) > 0), (
            "Interval associations require canonical ancestor order"
        )
        assert len(columns) == interval_offsets[row + 1] - interval_offsets[row]
        association_columns.extend(columns)
    association_columns = np.asarray(association_columns, dtype=np.int64)
    assert np.array_equal(focal["derived_ac"][association_columns], interval_ac)
    interval_nodes = column_nodes[association_columns]
    eligible = (interval_ac >= settings.min_focal_ac) & (
        interval_ac <= settings.ac_cutoff
    )
    # Preserve ragged row boundaries while applying the analysis AC range once.
    cumulative = np.concatenate(([0], np.cumsum(eligible)))
    interval_offsets = cumulative[interval_offsets]
    interval_left = interval_left[eligible]
    interval_right = interval_right[eligible]
    interval_nodes = interval_nodes[eligible]
    return FocalIntervals(
        interval_offsets, interval_left, interval_right, interval_nodes
    )


def load_context(settings: DatasetSettings) -> WorkerContext:
    """Validate local inputs and prepare the common model once per worker.

    Sample observations come from the masked VCZ, preserving missing calls. Panel
    columns join to original nodes through metadata, never by assuming node offsets.
    """
    base = settings.input_dir
    name = settings.name
    reference = tskit.load(base / "ancestors" / f"{name}_inferred_ancestors.trees")
    panel = tsinfer.vcz.open_store(
        base / "ancestors" / f"{name}_inferred_ancestors.zarr"
    )
    samples = tsinfer.vcz.open_store(base / "samples" / f"{name}_samples_masked.zarr")
    focal = load_focal(
        base / "focal_ancestors" / f"{name}_inferred_focal_ancestors.npz"
    )
    with (base / "configs" / f"{name}_ancestor_inference.toml").open("rb") as file:
        native_config = tomllib.load(file)
    assert native_config["match"]["path_compression"] is False
    intervals = reference.metadata["sequence_intervals"]
    assert len(intervals) == 1, "This experiment requires one inference interval"
    positions = reference.sites_position
    assert len(positions) > 0 and np.all(np.diff(positions) > 0)
    assert np.array_equal(positions, panel["variant_position"][:])
    assert np.array_equal(intervals, panel["sequence_intervals"][:])
    ancestral = np.asarray([site.ancestral_state for site in reference.sites()])
    panel_alleles = panel["variant_allele"][:]
    assert panel_alleles.shape == (len(positions), 2)
    assert np.array_equal(panel_alleles[:, 0], ancestral)
    assert np.all(panel_alleles[:, 0] != panel_alleles[:, 1])
    assert np.all(panel_alleles != "")
    for mutation in reference.mutations():
        assert mutation.derived_state in panel_alleles[mutation.site]

    panel_ids = np.asarray(panel["sample_id"][:].tolist(), dtype=str)
    assert np.array_equal(panel_ids, focal["ancestor_id"])
    ancestor_columns = {sample_id: column for column, sample_id in enumerate(panel_ids)}
    column_nodes = np.full(len(panel_ids), tskit.NULL, dtype=np.int32)
    assert reference.num_nodes == len(panel_ids) + 2
    assert reference.node(0).metadata == reference.node(1).metadata == {}
    assert reference.node(0).time > reference.node(1).time
    for node in reference.nodes():
        if node.id < 2:
            continue
        metadata = node.metadata
        assert metadata["source"] == "ancestors" and metadata["ploidy_index"] == 0
        column = ancestor_columns[str(metadata["sample_id"])]
        assert column_nodes[column] == tskit.NULL, "Duplicate ancestor metadata join"
        column_nodes[column] = node.id
    assert np.all(column_nodes >= 2), "An ancestor is absent from the reference"
    intervals = load_focal_intervals(settings, focal, panel, positions, column_nodes)
    populations = load_populations(settings, focal)

    source = next(item for item in native_config["source"] if item["name"] == "samples")
    sample_columns = tsinfer.vcz.resolve_samples_selection(
        samples, source.get("samples")
    )
    sample_ids = np.asarray(
        samples["sample_id"].oindex[sample_columns].tolist(), dtype=str
    )
    ploidy = samples["call_genotype"].shape[2]
    assert np.array_equal(focal["sample_id"], np.repeat(sample_ids, ploidy))
    assert np.array_equal(
        focal["ploidy_index"], np.tile(np.arange(ploidy), len(sample_ids))
    )
    sample_positions = samples["variant_position"][:]
    sample_rows = np.searchsorted(sample_positions, positions)
    assert np.all(sample_rows < len(sample_positions)), (
        "Reference sites absent in samples"
    )
    assert np.array_equal(sample_positions[sample_rows], positions)
    sample_alleles = samples["variant_allele"].oindex[sample_rows, :]
    assert np.all(np.count_nonzero(sample_alleles != "", axis=1) == 2)
    allowed = sample_alleles == panel_alleles[:, 0, None]
    allowed |= sample_alleles == panel_alleles[:, 1, None]
    allowed |= sample_alleles == ""
    assert np.all(allowed), "Sample and panel allele strings disagree"
    calls = samples["call_genotype"].oindex[sample_rows, sample_columns, :]
    assert np.all((calls >= -1) & (calls < sample_alleles.shape[1]))
    codes = np.where(sample_alleles == ancestral[:, None], 0, 1).astype(np.int8)
    codes[sample_alleles == ""] = -1
    lookup_calls = np.maximum(calls, 0)
    haplotypes = np.take_along_axis(codes[:, None, :], lookup_calls, axis=2)
    assert np.all(haplotypes[calls >= 0] >= 0), "Called allele is an empty string"
    haplotypes[calls < 0] = -1
    haplotypes = haplotypes.reshape(len(positions), -1)

    parameters = native_config["match"]["sources"]["samples"]
    recombination = parameters["recombination"]
    mismatch = parameters["mismatch"]
    if not 0 < recombination < 1 or not 0 < mismatch < 1:
        raise ValueError(
            "Sample recombination and mismatch must lie strictly in (0, 1)"
        )
    rho = np.full(len(positions), recombination)
    mu = np.full(len(positions), mismatch)
    num_alleles = np.full(len(positions), 2, dtype=np.uint32)
    indexes = tsinfer.matching.MatcherIndexes(
        reference, vestigial_root=False, num_alleles=num_alleles
    )
    log_mismatch_penalty = math.log(mismatch) - math.log1p(-mismatch)
    switch_baseline = -recombination + recombination / reference.num_nodes
    log_switch_penalty = math.log(recombination) - math.log(reference.num_nodes)
    log_switch_penalty -= math.log1p(switch_baseline)
    return WorkerContext(
        settings=settings,
        reference=reference,
        positions=positions,
        haplotypes=haplotypes,
        focal=focal,
        intervals=intervals,
        populations=populations,
        rho=rho,
        mu=mu,
        indexes=indexes,
        log_mismatch_penalty=log_mismatch_penalty,
        log_switch_penalty=log_switch_penalty,
    )


def initialize_worker(settings: DatasetSettings) -> None:
    """Keep a separate reusable context in each multiprocessing worker."""
    global worker_context
    worker_context = load_context(settings)


def site_parents(
    left: np.ndarray,
    right: np.ndarray,
    parents: np.ndarray,
    positions: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Convert absolute genomic path bounds to parents on [start, end)."""
    order = np.argsort(left)
    left = np.asarray(left)[order]
    right = np.asarray(right)[order]
    parents = np.asarray(parents)[order]
    assert len(left) > 0 and np.all(left < right)
    assert np.array_equal(right[:-1], left[1:]), "Path segments are not contiguous"
    starts = np.searchsorted(positions, left)
    ends = np.searchsorted(positions, right)
    assert starts[0] == start and ends[-1] == end
    result = np.full(end - start, tskit.NULL, dtype=np.int32)
    for lo, hi, parent in zip(starts, ends, parents, strict=True):
        assert start <= lo < hi <= end
        result[lo - start : hi - start] = parent
    assert np.all(result >= 0)
    return result


def run_match(
    matcher: tsinfer.matching.AncestorMatcher,
    haplotype: np.ndarray,
    start: int,
    end: int,
    buffer: np.ndarray,
    positions: np.ndarray,
) -> MatchArrays:
    """Copy requested match results before the next call overwrites native buffers."""
    left, right, parents = matcher.find_path(haplotype, start, end, buffer)
    site_path = site_parents(left, right, parents, positions, start, end)
    copied = buffer[start:end].copy()
    assert np.all((copied == 0) | (copied == 1))
    return MatchArrays(site_path, copied)


def focal_match(context: WorkerContext, haplotype: np.ndarray, row: int) -> FocalMatch:
    """Fix an interval-supported focal path; gaps are exactly its uncovered sites.

    Parent selection uses :func:`ch5_stitching.focal_parent_path`. Copied alleles
    are read from the inferred reference, rather than assumed from the interval
    generator, so any inferred/generated difference remains visible as a mutation.
    """
    start = context.intervals.offsets[row]
    end = context.intervals.offsets[row + 1]
    parents = ch5_stitching.focal_parent_path(
        len(haplotype),
        context.intervals.left[start:end],
        context.intervals.right[start:end],
        context.intervals.nodes[start:end],
    )
    covered = parents >= 0
    copied = np.full(len(haplotype), -1, dtype=np.int8)
    nodes = np.unique(parents[covered])
    if len(nodes) > 0:
        genotypes = context.reference.genotype_matrix(
            samples=nodes, isolated_as_missing=False
        )
        sites = np.flatnonzero(covered)
        columns = np.searchsorted(nodes, parents[covered])
        copied[covered] = genotypes[sites, columns]
        assert np.all((copied[covered] == 0) | (copied[covered] == 1))
    return FocalMatch(MatchArrays(parents, copied), covered)


def load_populations(settings: DatasetSettings, focal: dict) -> list[str]:
    """Join the upstream population metadata by haplotype identity."""
    path = settings.coverage_input_dir / "dataframes"
    path /= f"{settings.name}_inferred_focal_ancestor_stats.csv"
    labels = {}
    with path.open() as file:
        for record in csv.DictReader(file):
            key = (record["sample_id"], int(record["ploidy_index"]))
            population = record["population"]
            assert population != ""
            if key in labels:
                assert labels[key] == population
            labels[key] = population
    keys = zip(focal["sample_id"], focal["ploidy_index"], strict=True)
    return [labels[(str(sample), int(ploidy))] for sample, ploidy in keys]


def match_record(
    key: dict,
    result: MatchArrays,
    haplotype: np.ndarray,
    context: WorkerContext,
) -> dict:
    """Merge site-parent runs and derive mutations from the finished copied alleles.

    Site changes occur at inference-site positions; outer bounds follow the engine's
    whole-sequence convention. Reconstructing this saved path must recover every site.
    """
    assert result.parents.shape == result.copied.shape == haplotype.shape
    assert np.all(
        (result.parents >= 0) & (result.parents < context.reference.num_nodes)
    )
    assert np.all((result.copied == 0) | (result.copied == 1))
    changes = np.flatnonzero(result.parents[1:] != result.parents[:-1]) + 1
    starts = np.concatenate(([0], changes))
    bounds = np.concatenate(
        ([0.0], context.positions[changes], [context.reference.sequence_length])
    )
    path = [
        {
            "left": float(left),
            "right": float(right),
            "parent": int(result.parents[start]),
        }
        for start, left, right in zip(starts, bounds[:-1], bounds[1:], strict=True)
    ]
    reconstructed = site_parents(
        bounds[:-1],
        bounds[1:],
        result.parents[starts],
        context.positions,
        0,
        len(haplotype),
    )
    assert np.array_equal(reconstructed, result.parents)
    assert len(path) - 1 == np.count_nonzero(result.parents[1:] != result.parents[:-1])
    mismatches = (haplotype >= 0) & (haplotype != result.copied)
    mutation_sites = np.flatnonzero(mismatches)
    mutations = [
        {
            "position": float(context.positions[site]),
            "derived_state": int(haplotype[site]),
        }
        for site in mutation_sites
    ]
    assert len(mutations) == np.count_nonzero(mismatches)
    return {**key, "path": path, "mutations": mutations}


def compare_matches(
    full: dict,
    stitched: dict,
    full_arrays: MatchArrays,
    stitched_arrays: MatchArrays,
    covered: np.ndarray,
    num_focal: int,
    context: WorkerContext,
) -> dict:
    """Compare both complete paths and score them with the original full model."""
    metrics = ch5_stitching.comparison_metrics(
        full_arrays.parents, stitched_arrays.parents, covered
    )
    full_mismatches = len(full["mutations"])
    stitched_mismatches = len(stitched["mutations"])
    full_switches = len(full["path"]) - 1
    stitched_switches = len(stitched["path"]) - 1
    assert full_switches == metrics["full_switches"]
    assert stitched_switches == metrics["stitched_switches"]
    full_score = full_mismatches * context.log_mismatch_penalty
    full_score += full_switches * context.log_switch_penalty
    stitched_score = stitched_mismatches * context.log_mismatch_penalty
    stitched_score += stitched_switches * context.log_switch_penalty
    score_delta = stitched_score - full_score
    key = {
        name: full[name]
        for name in (
            "dataset",
            "source",
            "sample_id",
            "ploidy_index",
            "ac_cutoff",
            "population",
        )
    }
    comparison = {
        **key,
        "num_focal_ancestors": num_focal,
        "num_sites": len(covered),
        **metrics,
        "num_gaps": len(stitched["gap_site_intervals"]),
        "mutations_identical": full["mutations"] == stitched["mutations"],
        "full_mismatches": full_mismatches,
        "stitched_mismatches": stitched_mismatches,
        "full_score": full_score,
        "stitched_score": stitched_score,
        "score_delta": score_delta,
        "scores_equal": abs(score_delta) <= context.settings.score_tolerance,
    }
    if score_delta > 0:
        logger.warning(
            "Positive score delta %s for %s; check full-model assumptions before "
            "interpreting this result (n=%s, rho=%s, mu=%s)",
            score_delta,
            key,
            context.reference.num_nodes,
            context.rho[0],
            context.mu[0],
        )
    return comparison


def match_haplotype(row: int) -> JobResult:
    """Cover anchored focal intervals, fill zero-coverage gaps, and run the full HMM."""
    context = worker_context
    focal = context.focal
    haplotype = np.ascontiguousarray(context.haplotypes[:, row])
    focal_result = focal_match(context, haplotype, row)
    stitched_arrays = focal_result.arrays
    covered = focal_result.mask
    num_focal = context.intervals.offsets[row + 1] - context.intervals.offsets[row]
    focal_intervals = ch5_stitching.site_intervals(covered)
    gaps = ch5_stitching.site_intervals(~covered)
    coverage = np.zeros(len(haplotype), dtype=np.int8)
    for intervals in (focal_intervals, gaps):
        for start, end in intervals:
            coverage[start:end] += 1
    assert np.all(coverage == 1), "Focal coverage and gaps must partition all sites"
    buffer = np.empty(len(haplotype), dtype=np.int8)
    gap_matcher = tsinfer.matching.AncestorMatcher(
        context.indexes, context.rho, context.mu
    )
    for start, end in gaps:
        gap = run_match(
            gap_matcher, haplotype, int(start), int(end), buffer, context.positions
        )
        stitched_arrays.parents[start:end] = gap.parents
        stitched_arrays.copied[start:end] = gap.copied
    key = {
        "dataset": context.settings.name,
        "source": "samples",
        "sample_id": str(focal["sample_id"][row]),
        "ploidy_index": int(focal["ploidy_index"][row]),
        "ac_cutoff": context.settings.ac_cutoff,
        "population": context.populations[row],
    }
    stitched = match_record(key, stitched_arrays, haplotype, context)
    stitched["focal_site_intervals"] = focal_intervals.tolist()
    stitched["gap_site_intervals"] = gaps.tolist()
    full_matcher = tsinfer.matching.AncestorMatcher(
        context.indexes, context.rho, context.mu
    )
    full_arrays = run_match(
        full_matcher, haplotype, 0, len(haplotype), buffer, context.positions
    )
    full = match_record(key, full_arrays, haplotype, context)
    comparison = compare_matches(
        full, stitched, full_arrays, stitched_arrays, covered, int(num_focal), context
    )
    return JobResult(full, stitched, comparison)


def write_results(results, output_dir: pathlib.Path) -> None:
    """Stream paths and scalar cohort diagnostics in canonical NPZ order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        (output_dir / "full_matches.jsonl").open("w") as full_file,
        (output_dir / "stitched_matches.jsonl").open("w") as stitched_file,
        (output_dir / "comparison.csv").open("w", newline="") as comparison_file,
    ):
        writer = None
        for result in results:
            full_file.write(json.dumps(result.full, allow_nan=False) + "\n")
            stitched_file.write(json.dumps(result.stitched, allow_nan=False) + "\n")
            if writer is None:
                writer = csv.DictWriter(
                    comparison_file, fieldnames=list(result.comparison)
                )
                writer.writeheader()
            writer.writerow(result.comparison)
            comparison = result.comparison
            logger.info(
                "%s %s/%s: covered=%s gaps=%s agreement=%.4f extra_switch_fraction=%s",
                comparison["dataset"],
                comparison["sample_id"],
                comparison["ploidy_index"],
                comparison["focal_sites"],
                comparison["num_gaps"],
                comparison["fraction_parent_agreement"],
                comparison["extra_switch_fraction"],
            )


def run_dataset(
    settings: DatasetSettings,
    requested: list[dict] | None,
    workers: int | None,
    output_dir: pathlib.Path,
) -> None:
    """Queue whole haplotype jobs; compute the CPU default only at process launch."""
    focal_path = settings.input_dir / "focal_ancestors"
    focal_path /= f"{settings.name}_inferred_focal_ancestors.npz"
    rows = select_rows(load_focal(focal_path), requested)
    if workers is None:
        workers = multiprocessing.cpu_count()
    workers = min(workers, len(rows))
    logger.info("%s: %s haplotypes using %s workers", settings.name, len(rows), workers)
    if workers == 1:
        initialize_worker(settings)
        results = map(match_haplotype, rows)
        write_results(results, output_dir)
    else:
        processes = multiprocessing.get_context("spawn")
        with processes.Pool(
            workers, initializer=initialize_worker, initargs=(settings,)
        ) as pool:
            results = pool.imap(match_haplotype, rows, chunksize=1)
            write_results(results, output_dir)


def resolve_path(value: str, config_dir: pathlib.Path) -> pathlib.Path:
    """Expand home directories and anchor relative paths to the YAML directory."""
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open() as file:
        config = yaml.safe_load(file)
    input_dir = resolve_path(config["input_dir"], config_path.parent)
    output_dir = resolve_path(config["output_dir"], config_path.parent)
    coverage_input_dir = resolve_path(config["coverage_input_dir"], config_path.parent)
    cutoff = config["ac_cutoff"]
    minimum_ac = config["min_focal_ac"]
    if type(cutoff) is not int or cutoff <= 0:
        raise ValueError("ac_cutoff must be a positive integer")
    if type(minimum_ac) is not int or not 1 <= minimum_ac <= cutoff:
        raise ValueError("min_focal_ac must lie between 1 and ac_cutoff")
    workers = config["workers"]
    if workers is not None and (type(workers) is not int or workers <= 0):
        raise ValueError("workers must be a positive integer or null")
    tolerance = float(config["score_tolerance"])
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("score_tolerance must be finite and nonnegative")
    names = [dataset["name"] for dataset in config["datasets"]]
    if len(names) != len(set(names)):
        raise ValueError("Dataset names must be unique")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for dataset in config["datasets"]:
        settings = DatasetSettings(
            dataset["name"],
            input_dir,
            coverage_input_dir,
            cutoff,
            minimum_ac,
            tolerance,
        )
        run_dataset(
            settings, dataset["haplotypes"], workers, output_dir / settings.name
        )


if __name__ == "__main__":
    main()
