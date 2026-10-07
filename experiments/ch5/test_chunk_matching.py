"""Compare independently stitched focal/gap matches with whole-reference matches.

Run with the tsinfer-match-eval environment and --config chunk_matching_config.yaml.
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

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetSettings:
    """Picklable settings passed to each dataset's worker initializer."""

    name: str
    input_dir: pathlib.Path
    ac_cutoff: int
    score_tolerance: float


@dataclasses.dataclass
class WorkerContext:
    """Original reference, observations, and reusable full-model indexes."""

    settings: DatasetSettings
    reference: tskit.TreeSequence
    positions: np.ndarray
    ancestral: np.ndarray
    haplotypes: np.ndarray
    focal: dict[str, np.ndarray]
    column_nodes: np.ndarray
    node_columns: np.ndarray
    support_start: np.ndarray
    support_end: np.ndarray
    num_alleles: np.ndarray
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
class AcceptedMatch:
    """Transient stitched arrays and the sites retained from the reduced match."""

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
    focal = load_focal(base / "focal_ancestors" / f"{name}_inferred_focal_ancestors.npz")
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
    node_columns = np.full(reference.num_nodes, -1, dtype=np.int32)
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
        node_columns[node.id] = column
    assert np.all(column_nodes >= 2), "An ancestor is absent from the reference"
    support_start = panel["sample_start_position"][:]
    support_end = panel["sample_end_position"][:]
    assert support_start.shape == support_end.shape == panel_ids.shape
    assert np.all(support_start < support_end)

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
        raise ValueError("Sample recombination and mismatch must lie strictly in (0, 1)")
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
        ancestral=ancestral,
        haplotypes=haplotypes,
        focal=focal,
        column_nodes=column_nodes,
        node_columns=node_columns,
        support_start=support_start,
        support_end=support_end,
        num_alleles=num_alleles,
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


def site_intervals(mask: np.ndarray) -> np.ndarray:
    """Find every maximal true run, including leading and trailing intervals."""
    padded = np.pad(mask, (1, 1), constant_values=False)
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    return transitions.reshape(-1, 2)


def accepted_focal_match(
    context: WorkerContext,
    haplotype: np.ndarray,
    focal_nodes: np.ndarray,
) -> AcceptedMatch:
    """Immediately match a simplified reference and retain only exact focal evidence."""
    num_sites = len(haplotype)
    stitched = MatchArrays(
        np.full(num_sites, tskit.NULL, dtype=np.int32),
        np.full(num_sites, -1, dtype=np.int8),
    )
    accepted = np.zeros(num_sites, dtype=bool)
    if len(focal_nodes) == 0:
        return AcceptedMatch(stitched, accepted)
    assert len(np.unique(focal_nodes)) == len(focal_nodes)
    retained_nodes = np.concatenate(([0, 1], focal_nodes))
    reduced, old_to_new = context.reference.simplify(
        retained_nodes, map_nodes=True, filter_sites=False
    )
    assert np.array_equal(old_to_new[retained_nodes], np.arange(len(retained_nodes)))
    assert np.array_equal(reduced.sites_position, context.positions)
    reduced_ancestral = np.asarray([site.ancestral_state for site in reduced.sites()])
    assert np.array_equal(reduced_ancestral, context.ancestral)
    new_to_old = np.full(reduced.num_nodes, tskit.NULL, dtype=np.int32)
    surviving = np.flatnonzero(old_to_new != tskit.NULL)
    new_to_old[old_to_new[surviving]] = surviving
    assert np.all(new_to_old >= 0)
    indexes = tsinfer.matching.MatcherIndexes(
        reduced, vestigial_root=False, num_alleles=context.num_alleles
    )
    matcher = tsinfer.matching.AncestorMatcher(indexes, context.rho, context.mu)
    buffer = np.empty(num_sites, dtype=np.int8)
    reduced_match = run_match(
        matcher, haplotype, 0, num_sites, buffer, context.positions
    )
    original_parents = new_to_old[reduced_match.parents]
    eligible = np.isin(original_parents, focal_nodes)
    eligible_sites = np.flatnonzero(eligible)
    columns = context.node_columns[original_parents[eligible_sites]]
    in_support = context.positions[eligible_sites] >= context.support_start[columns]
    in_support &= context.positions[eligible_sites] < context.support_end[columns]
    exact = (haplotype >= 0) & (reduced_match.copied == haplotype)
    accepted[eligible_sites] = exact[eligible_sites] & in_support
    stitched.parents[accepted] = original_parents[accepted]
    stitched.copied[accepted] = reduced_match.copied[accepted]
    assert np.all(eligible[accepted] & exact[accepted])
    assert np.all(in_support[accepted[eligible_sites]])
    return AcceptedMatch(stitched, accepted)


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
    assert np.all((result.parents >= 0) & (result.parents < context.reference.num_nodes))
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
    accepted: np.ndarray,
    num_focal: int,
    context: WorkerContext,
) -> dict:
    """Compare both complete paths and score them with the original full model."""
    disagreement = full_arrays.parents != stitched_arrays.parents
    full_mismatches = len(full["mutations"])
    stitched_mismatches = len(stitched["mutations"])
    full_switches = len(full["path"]) - 1
    stitched_switches = len(stitched["path"]) - 1
    num_extra_switches = stitched_switches - full_switches
    extra_switch_fraction = (
        num_extra_switches / stitched_switches if stitched_switches > 0 else None
    )
    full_score = full_mismatches * context.log_mismatch_penalty
    full_score += full_switches * context.log_switch_penalty
    stitched_score = stitched_mismatches * context.log_mismatch_penalty
    stitched_score += stitched_switches * context.log_switch_penalty
    score_delta = stitched_score - full_score
    key = {
        name: full[name]
        for name in ("dataset", "source", "sample_id", "ploidy_index", "ac_cutoff")
    }
    comparison = {
        **key,
        "num_focal_ancestors": num_focal,
        "num_sites": len(accepted),
        "accepted_sites": int(np.count_nonzero(accepted)),
        "gap_sites": int(np.count_nonzero(~accepted)),
        "num_gaps": len(stitched["gap_site_intervals"]),
        "parents_identical": bool(not np.any(disagreement)),
        "fraction_parent_agreement": float(np.mean(~disagreement)),
        "accepted_disagreement_sites": int(np.count_nonzero(disagreement & accepted)),
        "gap_disagreement_sites": int(np.count_nonzero(disagreement & ~accepted)),
        "mutations_identical": full["mutations"] == stitched["mutations"],
        "full_mismatches": full_mismatches,
        "stitched_mismatches": stitched_mismatches,
        "full_switches": full_switches,
        "stitched_switches": stitched_switches,
        "num_extra_switches": num_extra_switches,
        "extra_switch_fraction": extra_switch_fraction,
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
    """Run the complete simplify → reduced match → gaps → full comparison job."""
    context = worker_context
    focal = context.focal
    haplotype = np.ascontiguousarray(context.haplotypes[:, row])
    candidates = focal["ancestor_index"][
        focal["offsets"][row] : focal["offsets"][row + 1]
    ]
    candidates = candidates[
        focal["derived_ac"][candidates] <= context.settings.ac_cutoff
    ]
    focal_nodes = context.column_nodes[candidates]
    reduced_result = accepted_focal_match(context, haplotype, focal_nodes)
    stitched_arrays = reduced_result.arrays
    accepted = reduced_result.mask
    accepted_intervals = site_intervals(accepted)
    gaps = site_intervals(~accepted)
    coverage = np.zeros(len(haplotype), dtype=np.int8)
    for intervals in (accepted_intervals, gaps):
        for start, end in intervals:
            coverage[start:end] += 1
    assert np.all(coverage == 1), "Accepted pieces and gaps must partition all sites"
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
    assert np.all(stitched_arrays.copied[accepted] == haplotype[accepted])
    key = {
        "dataset": context.settings.name,
        "source": "samples",
        "sample_id": str(focal["sample_id"][row]),
        "ploidy_index": int(focal["ploidy_index"][row]),
        "ac_cutoff": context.settings.ac_cutoff,
    }
    stitched = match_record(key, stitched_arrays, haplotype, context)
    stitched["accepted_site_intervals"] = accepted_intervals.tolist()
    stitched["gap_site_intervals"] = gaps.tolist()
    full_matcher = tsinfer.matching.AncestorMatcher(
        context.indexes, context.rho, context.mu
    )
    full_arrays = run_match(
        full_matcher, haplotype, 0, len(haplotype), buffer, context.positions
    )
    full = match_record(key, full_arrays, haplotype, context)
    comparison = compare_matches(
        full, stitched, full_arrays, stitched_arrays, accepted, len(focal_nodes), context
    )
    return JobResult(full, stitched, comparison)


def write_results(results, output_dir: pathlib.Path) -> None:
    """Stream only the two JSONL records and comparison CSV, in selected NPZ order."""
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
                "%s %s/%s: accepted=%s gaps=%s agreement=%.4f extra_switch_fraction=%s",
                comparison["dataset"],
                comparison["sample_id"],
                comparison["ploidy_index"],
                comparison["accepted_sites"],
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
    cutoff = config["ac_cutoff"]
    if type(cutoff) is not int or cutoff <= 0:
        raise ValueError("ac_cutoff must be a positive integer")
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
        settings = DatasetSettings(dataset["name"], input_dir, cutoff, tolerance)
        run_dataset(settings, dataset["haplotypes"], workers, output_dir / settings.name)


if __name__ == "__main__":
    main()
