"""Compare saved stitching with the greedy cover and focal-only HMM controls.

Run with the tsinfer-match-eval Python environment. Each queued haplotype job
reuses saved full-reference gap paths, so the controlled comparisons change
only focal parents. The selected configuration also gets independent focal
HMM matches and a simulation topology comparison against the known truth.
"""

import argparse
import dataclasses
import json
import logging
import multiprocessing
import pathlib
import tomllib

import numpy as np
import pandas as pd
import tsinfer
import tskit
import yaml
import zarr

import ch5_stitching

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetContext:
    """Read-only inputs shared by all jobs for a dataset inside one worker."""

    name: str
    reference: tskit.TreeSequence
    positions: np.ndarray
    focal: dict
    intervals: dict
    masks: dict
    matches: dict
    full: dict
    old_greedy: dict
    observations: np.ndarray
    ancestor_nodes: dict
    association_nodes: np.ndarray
    rho: np.ndarray
    mu: np.ndarray
    num_alleles: np.ndarray
    populations: dict
    settings: dict


@dataclasses.dataclass
class SitePath:
    """Original-reference copying parents and canonical alleles at sites."""

    parents: np.ndarray
    copied: np.ndarray


@dataclasses.dataclass
class HaplotypeResult:
    """Compact cohort metrics, component diagnostics, and selected run records."""

    comparisons: list[dict]
    controls: dict
    components: list[dict]
    distances: list[dict]
    matches: dict


def read_archive(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def read_matches(path: pathlib.Path) -> dict:
    records = {}
    with path.open() as file:
        for line in file:
            record = json.loads(line)
            key = (record["sample_id"], record["ploidy_index"])
            assert key not in records
            record["path"].sort(key=lambda segment: segment["left"])
            records[key] = record
    return records


def site_parents(record: dict, reference: tskit.TreeSequence) -> np.ndarray:
    segments = record["path"]
    right = np.array([segment["right"] for segment in segments])
    parents = np.array([segment["parent"] for segment in segments], dtype=np.int32)
    indices = np.searchsorted(right, reference.sites_position, side="right")
    result = parents[indices]
    assert np.count_nonzero(result[1:] != result[:-1]) == len(segments) - 1
    return result


def read_observations(
    base: pathlib.Path, name: str, reference, focal: dict
) -> np.ndarray:
    """Read each cohort once, mapping REF/ALT to the reference ancestral state."""
    samples = zarr.open_group(
        base / "samples" / f"{name}_samples_masked.zarr", mode="r"
    )
    all_sample_ids = samples["sample_id"][:].tolist()
    sample_columns = {
        sample_id: column for column, sample_id in enumerate(all_sample_ids)
    }
    selected_ids = focal["sample_id"][::2].tolist()
    columns = [sample_columns[sample_id] for sample_id in selected_ids]
    assert np.array_equal(focal["sample_id"], np.repeat(selected_ids, 2))
    assert np.array_equal(focal["ploidy_index"], np.tile([0, 1], len(columns)))
    source_positions = samples["variant_position"][:]
    rows = np.searchsorted(source_positions, reference.sites_position)
    assert np.array_equal(source_positions[rows], reference.sites_position)
    alleles = samples["variant_allele"].oindex[rows, :]
    calls = samples["call_genotype"].oindex[rows, columns, :]
    ancestral = np.asarray([site.ancestral_state for site in reference.sites()])
    codes = np.where(alleles == ancestral[:, None], 0, 1).astype(np.int8)
    codes[alleles == ""] = -1
    observations = np.take_along_axis(codes[:, None, :], np.maximum(calls, 0), axis=2)
    observations[calls < 0] = -1
    return observations.reshape(reference.num_sites, -1)


def load_context(name: str) -> DatasetContext:
    """Load once per worker; observations are memory-mapped from preparation."""
    settings = _settings["datasets"][name]
    base = _settings["input_dir"]
    reference = tskit.load(base / "ancestors" / f"{name}_inferred_ancestors.trees")
    focal = read_archive(
        base / "focal_ancestors" / f"{name}_inferred_focal_ancestors.npz"
    )
    intervals = read_archive(
        base / "haplotype_intervals" / f"{name}_inferred_focal_ancestor_intervals.npz"
    )
    full = read_matches(base / "matches" / f"{name}_inferred_samples_matches.jsonl")
    masks = {}
    matches = {}
    for ac_cutoff in _settings["pipeline"]["ac_cutoff"]:
        for budget in _settings["pipeline"]["max_mismatches"]:
            prefix = f"{name}_inferred_ac{ac_cutoff}_k{budget}"
            masks[ac_cutoff, budget] = read_archive(
                base / "focal_masks" / f"{prefix}_focal_mask.npz"
            )
            matches[ac_cutoff, budget] = read_matches(
                base / "matches" / f"{prefix}_hmm_stitched_samples_matches.jsonl"
            )
    old_name = settings["saved_greedy_name"]
    old_reference = tskit.load(
        _settings["old_input_dir"]
        / "ancestors"
        / f"{old_name}_inferred_ancestors.trees"
    )
    assert reference.tables.equals(old_reference.tables, ignore_provenance=True)
    old_greedy = read_matches(
        _settings["old_output_dir"] / old_name / "stitched_matches.jsonl"
    )
    old_comparisons = pd.read_csv(
        _settings["old_output_dir"] / old_name / "comparison.csv"
    )
    populations = {
        (row.sample_id, row.ploidy_index): row.population
        for row in old_comparisons.itertuples()
    }
    observations = np.load(
        _settings["output_dir"] / f"{name}_observations.npy", mmap_mode="r"
    )
    ancestor_nodes = {}
    for node in reference.nodes():
        if node.metadata.get("source") == "ancestors":
            ancestor_nodes[node.metadata["sample_id"]] = node.id
    # Interval archives omit candidates above their generation AC maximum.
    association_nodes = []
    for index in range(len(focal["sample_id"])):
        start, end = focal["offsets"][index : index + 2]
        columns = focal["ancestor_index"][start:end]
        columns = columns[
            focal["derived_ac"][columns] <= int(intervals["max_ac_cutoff"])
        ]
        assert (
            len(columns)
            == intervals["offsets"][index + 1] - intervals["offsets"][index]
        )
        association_nodes.extend(
            ancestor_nodes[focal["ancestor_id"][column]] for column in columns
        )
    with (base / "configs" / f"{name}_ancestor_inference.toml").open("rb") as file:
        native_config = tomllib.load(file)
    parameters = native_config["match"]["sources"]["samples"]
    return DatasetContext(
        name,
        reference,
        reference.sites_position,
        focal,
        intervals,
        masks,
        matches,
        full,
        old_greedy,
        observations,
        ancestor_nodes,
        np.asarray(association_nodes, dtype=np.int32),
        np.full(reference.num_sites, parameters["recombination"]),
        np.full(reference.num_sites, parameters["mismatch"]),
        np.full(reference.num_sites, 2, dtype=np.uint32),
        populations,
        settings,
    )


def initialize_worker(settings: dict) -> None:
    global _settings, _contexts
    _settings = settings
    _contexts = {}


@dataclasses.dataclass
class SiteRuns:
    """Covered components and gaps on the inference-site axis."""

    focal: np.ndarray
    gaps: np.ndarray


def mask_runs(mask: dict, index: int) -> SiteRuns:
    start, end = mask["offsets"][index : index + 2]
    runs = np.column_stack(
        (mask["left_site_index"][start:end], mask["right_site_index"][start:end])
    )
    gaps = mask["is_gap"][start:end]
    return SiteRuns(runs[~gaps], runs[gaps])


def coverage_array(runs: np.ndarray, num_sites: int) -> np.ndarray:
    covered = np.zeros(num_sites, dtype=bool)
    for left, right in runs:
        covered[left:right] = True
    return covered


def switch_metrics(
    parents: np.ndarray, mismatches: np.ndarray, covered: np.ndarray
) -> dict:
    changes = parents[1:] != parents[:-1]
    interior = {
        "focal": covered[1:] & covered[:-1],
        "gap": ~covered[1:] & ~covered[:-1],
        "seam": covered[1:] != covered[:-1],
    }
    counts = {
        f"{category}_switches": int(np.count_nonzero(changes & edges))
        for category, edges in interior.items()
    }
    counts["num_switches"] = int(np.count_nonzero(changes))
    assert (
        sum(counts[f"{category}_switches"] for category in interior)
        == counts["num_switches"]
    )
    counts["focal_mismatches"] = int(np.count_nonzero(mismatches & covered))
    counts["gap_mismatches"] = int(np.count_nonzero(mismatches & ~covered))
    counts["num_mismatches"] = int(np.count_nonzero(mismatches))
    counts["focal_sites"] = int(np.count_nonzero(covered))
    counts["gap_sites"] = len(covered) - counts["focal_sites"]
    return counts


def comparison_row(
    identity: dict,
    method: str,
    parents: np.ndarray,
    mismatches: np.ndarray,
    covered: np.ndarray,
    context: DatasetContext,
) -> dict:
    """Score complete paths with one common, full-reference Li-Stephens model.

    Site-constant emissions and no-switch transitions cancel against the full
    path. The transition weight uses the C matcher's actual reference node count.
    """
    metrics = switch_metrics(parents, mismatches, covered)
    key = (identity["sample_id"], identity["ploidy_index"])
    full_record = context.full[key]
    full_switches = len(full_record["path"]) - 1
    extra_switches = metrics["num_switches"] - full_switches
    metrics["num_extra_switches"] = extra_switches
    metrics["switch_increase_percentage"] = (
        100 * extra_switches / full_switches if full_switches > 0 else np.nan
    )
    rho = float(context.rho[0])
    mu = float(context.mu[0])
    switch_probability = rho / context.reference.num_nodes
    stay_probability = 1 - rho + switch_probability
    switch_penalty = np.log(switch_probability / stay_probability)
    mismatch_penalty = np.log(mu / (1 - mu))
    extra_mismatches = metrics["num_mismatches"] - len(full_record["mutations"])
    metrics["full_model_log_likelihood_delta"] = (
        extra_switches * switch_penalty + extra_mismatches * mismatch_penalty
    )
    return {**identity, "implementation": method, **metrics}


def mutation_mask(record: dict, positions: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(positions), dtype=bool)
    sites = np.searchsorted(
        positions, [mutation["position"] for mutation in record["mutations"]]
    )
    mask[sites] = True
    return mask


def match_region(
    matcher, haplotype, left: int, right: int, positions: np.ndarray
) -> SitePath:
    copied = np.empty(len(haplotype), dtype=np.int8)
    path_left, path_right, path_parent = matcher.find_path(
        haplotype, left, right, copied
    )
    parents = np.empty(right - left, dtype=np.int32)
    for start, end, node in zip(path_left, path_right, path_parent, strict=True):
        start_site = max(left, int(np.searchsorted(positions, start)))
        end_site = min(right, int(np.searchsorted(positions, end)))
        parents[start_site - left : end_site - left] = node
    return SitePath(parents, copied[left:right].copy())


def encode_record(
    key: tuple,
    parents: np.ndarray,
    mismatch: np.ndarray,
    haplotype: np.ndarray,
    context: DatasetContext,
) -> dict:
    changes = np.flatnonzero(parents[1:] != parents[:-1]) + 1
    starts = np.concatenate(([0], changes))
    bounds = np.concatenate(
        ([0], context.positions[changes], [context.reference.sequence_length])
    )
    path = [
        {"left": float(left), "right": float(right), "parent": int(parents[start])}
        for start, left, right in zip(starts, bounds[:-1], bounds[1:], strict=True)
    ]
    mutations = [
        {
            "position": float(context.positions[site]),
            "derived_state": int(haplotype[site]),
        }
        for site in np.flatnonzero(mismatch)
    ]
    return {
        "source": "samples",
        "sample_id": key[0],
        "ploidy_index": key[1],
        "path": path,
        "mutations": mutations,
    }


def distance_rows(
    identity: dict, paths: dict, covered: np.ndarray, config: dict
) -> list[dict]:
    """Count switches per eligible edge by distance to the nearest seam."""
    seam_sites = np.flatnonzero(covered[1:] != covered[:-1]) + 1
    sites = np.arange(len(covered))
    insertion = np.searchsorted(seam_sites, sites, side="right")
    distance = np.full(len(covered), np.inf)
    has_left = insertion > 0
    has_right = insertion < len(seam_sites)
    distance[has_left] = sites[has_left] - seam_sites[insertion[has_left] - 1]
    right_distance = seam_sites[insertion[has_right]] - 1 - sites[has_right]
    distance[has_right] = np.minimum(distance[has_right], right_distance)
    edge_distance = np.minimum(distance[:-1], distance[1:])
    edges = {"focal": covered[:-1] & covered[1:], "gap": ~covered[:-1] & ~covered[1:]}
    rows = []
    for region, region_edges in edges.items():
        for left, right in zip(
            config["distance_bin_edges"][:-1],
            config["distance_bin_edges"][1:],
            strict=True,
        ):
            eligible = region_edges & (edge_distance >= left) & (edge_distance < right)
            for method, parents in paths.items():
                changes = parents[1:] != parents[:-1]
                rows.append(
                    {
                        **identity,
                        "implementation": method,
                        "region": region,
                        "distance_left": left,
                        "distance_right": right,
                        "num_edges": int(np.count_nonzero(eligible)),
                        "num_switches": int(np.count_nonzero(changes & eligible)),
                    }
                )
    return rows


def analyse_haplotype(job: tuple[str, int]) -> HaplotypeResult:
    name, index = job
    if name not in _contexts:
        _contexts[name] = load_context(name)
    context = _contexts[name]
    focal = context.focal
    key = (str(focal["sample_id"][index]), int(focal["ploidy_index"][index]))
    identity = {
        "dataset": name,
        "sample_id": key[0],
        "ploidy_index": key[1],
        "population": context.populations[key],
    }
    haplotype = np.ascontiguousarray(context.observations[:, index])
    full_parents = site_parents(context.full[key], context.reference)
    full_mismatches = mutation_mask(context.full[key], context.positions)
    greedy_paths = {}
    start, end = context.intervals["offsets"][index : index + 2]
    for ac_cutoff in _settings["pipeline"]["ac_cutoff"]:
        eligible = context.intervals["focal_ac"][start:end] <= ac_cutoff
        for budget in _settings["pipeline"]["max_mismatches"]:
            parents = ch5_stitching.focal_parent_path(
                len(haplotype),
                context.intervals["left_site_index"][start:end, budget][eligible],
                context.intervals["right_site_index"][start:end, budget][eligible],
                context.association_nodes[start:end][eligible],
            )
            greedy_paths[ac_cutoff, budget] = parents
    used_nodes = np.unique(
        np.concatenate([parents[parents >= 0] for parents in greedy_paths.values()])
    )
    genotypes = context.reference.genotype_matrix(
        samples=used_nodes, isolated_as_missing=False
    )
    comparisons = []
    matches = {}
    control_rows = {}
    components = []
    distances = []
    for (ac_cutoff, budget), greedy in greedy_paths.items():
        run_identity = {**identity, "ac_cutoff": ac_cutoff, "max_mismatches": budget}
        partition = mask_runs(context.masks[ac_cutoff, budget], index)
        runs = partition.focal
        covered = coverage_array(runs, len(haplotype))
        assert np.array_equal(covered, greedy >= 0)
        saved = context.matches[ac_cutoff, budget][key]
        reduced_parents = site_parents(saved, context.reference)
        reduced_mismatches = mutation_mask(saved, context.positions)
        greedy_parents = reduced_parents.copy()
        greedy_parents[covered] = greedy[covered]
        greedy_mismatches = reduced_mismatches.copy()
        sites = np.flatnonzero(covered)
        columns = np.searchsorted(used_nodes, greedy[covered])
        copied = genotypes[sites, columns]
        greedy_mismatches[covered] = (haplotype[covered] >= 0) & (
            haplotype[covered] != copied
        )
        paths = {
            "full": full_parents,
            "greedy": greedy_parents,
            "reduced_hmm": reduced_parents,
        }
        mismatch_masks = {
            "full": full_mismatches,
            "greedy": greedy_mismatches,
            "reduced_hmm": reduced_mismatches,
        }
        for method, parents in paths.items():
            row = comparison_row(
                run_identity, method, parents, mismatch_masks[method], covered, context
            )
            comparisons.append(row)
        selected = _settings["comparison"]
        if ac_cutoff != selected["ac_cutoff"] or budget != selected["max_mismatches"]:
            continue
        old = context.old_greedy[key]
        old_coverage = coverage_array(
            np.asarray(old["focal_site_intervals"]), len(haplotype)
        )
        assert np.array_equal(covered, old_coverage)
        old_parents = site_parents(old, context.reference)
        assert np.array_equal(old_parents[covered], greedy[covered])
        old_mismatches = mutation_mask(old, context.positions)
        row = comparison_row(
            run_identity, "saved_greedy", old_parents, old_mismatches, covered, context
        )
        comparisons.append(row)
        paths["saved_greedy"] = old_parents
        # Reproduce the pipeline's site-preserving reduction, using metadata IDs.
        candidate_start, candidate_end = focal["offsets"][index : index + 2]
        columns = focal["ancestor_index"][candidate_start:candidate_end]
        eligible_columns = columns[focal["derived_ac"][columns] <= ac_cutoff]
        retained = [
            0,
            1,
            *[
                context.ancestor_nodes[focal["ancestor_id"][column]]
                for column in eligible_columns
            ],
        ]
        reduced, old_to_new = context.reference.simplify(
            retained, map_nodes=True, filter_sites=False
        )
        new_to_old = np.empty(reduced.num_nodes, dtype=np.int32)
        mapped = np.flatnonzero(old_to_new != tskit.NULL)
        new_to_old[old_to_new[mapped]] = mapped
        indexes = tsinfer.matching.MatcherIndexes(
            reduced, vestigial_root=False, num_alleles=context.num_alleles
        )
        matcher = tsinfer.matching.AncestorMatcher(indexes, context.rho, context.mu)
        whole = match_region(matcher, haplotype, 0, len(haplotype), context.positions)
        whole.parents = new_to_old[whole.parents]
        assert np.array_equal(whole.parents[covered], reduced_parents[covered])
        whole_mismatches = (haplotype >= 0) & (haplotype != whole.copied)
        assert np.array_equal(whole_mismatches[covered], reduced_mismatches[covered])
        independent = reduced_parents.copy()
        independent_mismatches = reduced_mismatches.copy()
        for component_index, (left, right) in enumerate(runs):
            local = match_region(
                matcher, haplotype, int(left), int(right), context.positions
            )
            independent[left:right] = new_to_old[local.parents]
            independent_mismatches[left:right] = (haplotype[left:right] >= 0) & (
                haplotype[left:right] != local.copied
            )
            row = {
                **run_identity,
                "component": component_index,
                "left_site": int(left),
                "right_site": int(right),
                "num_sites": int(right - left),
            }
            for method, parents in {**paths, "independent_hmm": independent}.items():
                row[f"{method}_switches"] = int(
                    np.count_nonzero(
                        parents[left + 1 : right] != parents[left : right - 1]
                    )
                )
            components.append(row)
        paths["independent_hmm"] = independent
        row = comparison_row(
            run_identity,
            "independent_hmm",
            independent,
            independent_mismatches,
            covered,
            context,
        )
        comparisons.append(row)
        independent_metrics = switch_metrics(
            independent, independent_mismatches, covered
        )
        # Verify the exact inferred alleles used by the greedy focal path survive.
        greedy_nodes = np.unique(greedy[covered])
        original_columns = np.searchsorted(used_nodes, greedy_nodes)
        original_genotypes = genotypes[:, original_columns]
        reduced_genotypes = reduced.genotype_matrix(
            samples=old_to_new[greedy_nodes], isolated_as_missing=False
        )
        preserved = int(np.count_nonzero(original_genotypes != reduced_genotypes))
        assert preserved == 0
        control_rows = {
            **run_identity,
            "num_focal_components": len(runs),
            "num_candidates": len(eligible_columns),
            "num_reduced_nodes": reduced.num_nodes,
            "genotype_differences_after_simplify": preserved,
            "global_reduced_switches": int(
                np.count_nonzero(whole.parents[1:] != whole.parents[:-1])
            ),
            "global_reduced_focal_switches": switch_metrics(
                whole.parents, whole_mismatches, covered
            )["focal_switches"],
            "global_reduced_gap_mismatches": int(
                np.count_nonzero(whole_mismatches & ~covered)
            ),
            "independent_focal_switches": independent_metrics["focal_switches"],
            "greedy_focal_switches": switch_metrics(
                greedy_parents, greedy_mismatches, covered
            )["focal_switches"],
            "independent_focal_mismatches": independent_metrics["focal_mismatches"],
            "greedy_focal_mismatches": int(
                np.count_nonzero(greedy_mismatches & covered)
            ),
        }
        distances = distance_rows(run_identity, paths, covered, _settings["plot"])
        for method, parents in paths.items():
            if method == "full":
                matches[method] = context.full[key]
            elif method == "saved_greedy":
                matches[method] = old
            else:
                mismatch = (
                    independent_mismatches
                    if method == "independent_hmm"
                    else mismatch_masks[method]
                )
                matches[method] = encode_record(
                    key, parents, mismatch, haplotype, context
                )
        if key in {
            (item["sample_id"], item["ploidy_index"])
            for item in context.settings["haplotypes"]
        }:
            archive_path = (
                _settings["output_dir"] / f"{name}_{key[0].lower()}_{key[1]}_paths.npz"
            )
            np.savez_compressed(
                archive_path,
                covered=covered,
                observed=haplotype,
                reduced_whole=whole.parents,
                independent=independent,
                greedy=greedy_parents,
                saved_greedy=old_parents,
                full=full_parents,
                reduced_stitched=reduced_parents,
            )
    return HaplotypeResult(comparisons, control_rows, components, distances, matches)


def topology_comparison(settings: dict, all_matches: dict) -> pd.DataFrame:
    """Measure marginal topology against simulated truth, over inference intervals.

    Kendall-Colijn lambda=0 measures rooted topology, not branch lengths. Sample
    ordering is explicitly joined to truth IDs, and unobserved flanks are removed.
    This single simulation is a diagnostic, not a biological accuracy claim.
    """
    rows = []
    for name, dataset in settings["datasets"].items():
        if dataset["population_source"] != "simulation":
            continue
        base = settings["input_dir"]
        reference = tskit.load(base / "ancestors" / f"{name}_inferred_ancestors.trees")
        native = tskit.load(base / "matches" / f"{name}_inferred_samples_raw.trees")
        sample_nodes = {}
        for node_id in native.samples():
            node = native.node(node_id)
            key = (node.metadata["sample_id"], node.metadata["ploidy_index"])
            sample_nodes[key] = int(node_id)
        pipeline_dataset = next(
            item for item in settings["pipeline"]["datasets"] if item["name"] == name
        )
        truth_path = pathlib.Path(pipeline_dataset["ts_path"]).expanduser()
        if not truth_path.is_absolute():
            truth_path = settings["pipeline_dir"] / truth_path
        truth = tskit.load(truth_path)
        truth_keys = []
        truth_nodes = []
        for individual in truth.individuals():
            for ploidy_index, node_id in enumerate(individual.nodes):
                truth_keys.append((f"tsk_{individual.id}", ploidy_index))
                truth_nodes.append(int(node_id))
        assert set(truth_keys) == set(sample_nodes)
        inferred_nodes = [sample_nodes[key] for key in truth_keys]
        intervals = reference.metadata["sequence_intervals"]
        truth = truth.simplify(truth_nodes)
        truth = truth.keep_intervals(intervals).trim()
        panel = zarr.open_group(
            base / "ancestors" / f"{name}_inferred_ancestors.zarr", mode="r"
        )
        alleles = panel["variant_allele"][:]
        for method in settings["topology_methods"]:
            tables = native.dump_tables()
            keep_edges = tables.edges.child < reference.num_nodes
            tables.edges.keep_rows(keep_edges)
            keep_mutations = tables.mutations.node < reference.num_nodes
            tables.mutations.keep_rows(keep_mutations)
            records = all_matches[name][method]
            for record in records:
                key = (record["sample_id"], record["ploidy_index"])
                child = sample_nodes[key]
                for segment in record["path"]:
                    tables.edges.add_row(
                        segment["left"], segment["right"], segment["parent"], child
                    )
                for mutation in record["mutations"]:
                    site = int(
                        np.searchsorted(reference.sites_position, mutation["position"])
                    )
                    state = str(alleles[site, mutation["derived_state"]])
                    tables.mutations.add_row(site=site, node=child, derived_state=state)
            tables.sort()
            inferred = tables.tree_sequence().simplify(inferred_nodes)
            inferred = inferred.keep_intervals(intervals).trim()
            distance = inferred.kc_distance(truth, lambda_=0)
            rows.append(
                {
                    "dataset": name,
                    "implementation": method,
                    "kc_topology_distance": distance,
                    "num_trees": inferred.num_trees,
                    "evaluated_bp": inferred.sequence_length,
                }
            )
    return pd.DataFrame(rows)


def run(config_path: pathlib.Path) -> None:
    config_path = config_path.resolve()
    with config_path.open() as file:
        config = yaml.safe_load(file)
    pipeline_path = (config_path.parent / config["experiment_config"]).resolve()
    with pipeline_path.open() as file:
        pipeline = yaml.safe_load(file)
    comparison_config = config["method_comparison"]
    old_path = (config_path.parent / comparison_config["saved_greedy_config"]).resolve()
    with old_path.open() as file:
        old = yaml.safe_load(file)
    input_dir = pathlib.Path(pipeline["data_dir"]).expanduser()
    if not input_dir.is_absolute():
        input_dir = pipeline_path.parent / input_dir
    settings = {
        "pipeline": pipeline,
        "pipeline_dir": pipeline_path.parent,
        "input_dir": input_dir,
        "comparison": config["comparison"],
        "datasets": {item["name"]: item for item in config["datasets"]},
        "old_input_dir": (old_path.parent / old["input_dir"]).resolve(),
        "old_output_dir": (old_path.parent / old["output_dir"]).resolve(),
        "output_dir": (config_path.parent / comparison_config["output_dir"]).resolve(),
        "plot": comparison_config,
        "topology_methods": comparison_config["topology_methods"],
    }
    assert old["ac_cutoff"] == config["comparison"]["ac_cutoff"]
    assert config["comparison"]["max_mismatches"] == 0
    assert min(pipeline["max_mismatches"]) == 0
    settings["output_dir"].mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in settings["datasets"]:
        reference = tskit.load(
            input_dir / "ancestors" / f"{name}_inferred_ancestors.trees"
        )
        focal = read_archive(
            input_dir / "focal_ancestors" / f"{name}_inferred_focal_ancestors.npz"
        )
        observations = read_observations(input_dir, name, reference, focal)
        np.save(settings["output_dir"] / f"{name}_observations.npy", observations)
        jobs.extend((name, index) for index in range(len(focal["sample_id"])))
        logger.info("Prepared %s (%s haplotypes)", name, observations.shape[1])
    comparisons = []
    controls = []
    components = []
    distances = []
    all_matches = {
        name: {method: [] for method in comparison_config["topology_methods"]}
        for name in settings["datasets"]
    }
    workers = comparison_config["workers"]
    if workers is None:
        workers = multiprocessing.cpu_count()
    pool_context = multiprocessing.get_context("spawn")
    with pool_context.Pool(
        workers, initializer=initialize_worker, initargs=(settings,)
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(analyse_haplotype, jobs), start=1
        ):
            comparisons.extend(result.comparisons)
            controls.append(result.controls)
            components.extend(result.components)
            distances.extend(result.distances)
            name = result.controls["dataset"]
            for method in all_matches[name]:
                all_matches[name][method].append(result.matches[method])
            if completed % comparison_config["progress_every"] == 0:
                logger.info("Compared %s / %s haplotypes", completed, len(jobs))
    for filename, records in {
        "comparisons": comparisons,
        "focal_controls": controls,
        "focal_components": components,
        "switch_distances": distances,
    }.items():
        pd.DataFrame(records).to_csv(
            settings["output_dir"] / f"{filename}.csv", index=False
        )
    for name, methods in all_matches.items():
        for method, records in methods.items():
            records.sort(
                key=lambda record: (record["sample_id"], record["ploidy_index"])
            )
            with (settings["output_dir"] / f"{name}_{method}_matches.jsonl").open(
                "w"
            ) as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")
    topology = topology_comparison(settings, all_matches)
    topology.to_csv(settings["output_dir"] / "simulation_topology.csv", index=False)
    logger.info("Wrote comparisons to %s", settings["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(arguments.config)
