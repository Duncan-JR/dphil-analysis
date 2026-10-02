"""Site-weighted focal-chunk coverage, independent of HMM copying paths.

Read the match-eval products with :func:`load_coverage_data`, then sweep each
haplotype with :func:`summarise_coverage`. Retain only the lowest carried focal
site's chunk for each ancestor and haplotype at each budget. All denominators
use the inferred panel's complete site axis, including uncovered sites.
"""

import dataclasses
import logging
import multiprocessing
import pathlib

import numpy as np
import polars as pl
import tskit
import zarr

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CoverageData:
    """Selected chunks and the full haplotype roster, including empty candidates."""

    dataset: str
    num_sites: int
    haplotypes: pl.DataFrame
    chunks: pl.DataFrame
    mismatch_budgets: list[int]
    first_position: int
    last_position: int


@dataclasses.dataclass
class SweepTask:
    """Only numeric arrays travel through the multiprocessing task queue."""

    haplotype_index: int
    num_sites: int
    cutoffs: np.ndarray
    focal_ac: np.ndarray
    left: np.ndarray
    right: np.ndarray


@dataclasses.dataclass
class SweepResult:
    """Coverage statistics in cutoff order for one haplotype."""

    haplotype_index: int
    records: list[dict]


def focal_ac_cutoffs(max_ac: int = 200, num_bins: int = 20) -> np.ndarray:
    """Return increasing integer thresholds, starting at 2, 3, then log spaced.

    Rounding duplicates are filled from the unused integers so there are at
    least ``num_bins`` thresholds whenever the integer range permits it.
    """
    if max_ac < 2 or num_bins < 2:
        raise ValueError("max_ac and num_bins must be >= 2")
    target = min(num_bins, max_ac - 1)
    logarithmic = np.geomspace(2, max_ac, target)
    rounded = np.rint(logarithmic).astype(np.int64)
    required = np.arange(2, min(3, max_ac) + 1)
    cutoffs = np.union1d(rounded, required)
    while len(cutoffs) < target:
        gaps = np.diff(np.log(cutoffs))
        gaps[np.diff(cutoffs) < 2] = -1
        index = int(np.argmax(gaps))
        midpoint = np.sqrt(cutoffs[index] * cutoffs[index + 1])
        value = int(np.rint(midpoint))
        value = min(max(value, cutoffs[index] + 1), cutoffs[index + 1] - 1)
        cutoffs = np.union1d(cutoffs, [value])
    return cutoffs


def _truth_populations(haplotypes: pl.DataFrame, truth_path: pathlib.Path):
    """Use the same VCF identity mapping as match-eval's TS metadata reader."""
    truth = tskit.load(truth_path)
    mapping = truth.map_to_vcf_model()
    nodes_by_id = dict(
        zip(mapping.individuals_name, mapping.individuals_nodes, strict=True)
    )
    populations = []
    for row in haplotypes.iter_rows(named=True):
        node_id = nodes_by_id[row["sample_id"]][row["ploidy_index"]]
        node = truth.node(int(node_id))
        if node.population == tskit.NULL:
            raise ValueError(f"Truth node {node_id} has no population")
        label = truth.population(node.population).metadata.get("name")
        if label is None or label == "":
            raise ValueError(f"Truth population {node.population} has no name")
        populations.append(label)
    return haplotypes.with_columns(pl.Series("population", populations))


def summarise_focal_ancestor_set_sizes(
    focal_path: pathlib.Path, cutoffs: np.ndarray
) -> pl.DataFrame:
    """Average inclusive AC-filtered candidate counts over every NPZ haplotype.

    Match-eval stores distinct ancestor indices within each haplotype's ragged
    candidate set. Count their incidences across all haplotypes, then divide by
    the full haplotype roster, including empty sets. The proportion denominator
    is every ancestor in the panel, without an AC filter. Membership depends on
    carried focal alleles and AC eligibility, independently of mismatch budget.
    """
    with np.load(focal_path, allow_pickle=False) as focal:
        num_haplotypes = len(focal["sample_id"])
        num_ancestors = len(focal["ancestor_id"])
        candidate_indices = focal["ancestor_index"]
        candidate_ac = focal["derived_ac"][candidate_indices]
    if num_haplotypes == 0 or num_ancestors == 0:
        raise ValueError("Set-size proportions require haplotypes and panel ancestors")
    sorted_ac = np.sort(candidate_ac)
    total_candidates = np.searchsorted(sorted_ac, cutoffs, side="right")
    mean_candidates = total_candidates / num_haplotypes
    proportions = mean_candidates / num_ancestors
    return pl.DataFrame(
        {
            "focal_ac": cutoffs,
            "num_haplotypes": num_haplotypes,
            "num_ancestors": num_ancestors,
            "mean_num_focal_ancestors": mean_candidates,
            "mean_proportion_included": proportions,
        }
    )


def load_coverage_data(
    data_dir: pathlib.Path,
    dataset: str,
    max_ac: int,
    max_mismatches: list[int],
    truth_ts_path: pathlib.Path | None = None,
) -> CoverageData:
    """Read only site bounds and requested AC/budget rows using Polars.

    Identities come from the focal NPZ, never from the chunk rows. Population
    labels come from the statistics CSV (which includes every haplotype), or
    from the original simulation TS via :func:`_truth_populations`. Site bounds
    are already zero-based and half-open; no BP-to-site conversion is needed.
    ``max_mismatches`` selects exact source budgets, each allowing k per side.
    Multiple focal seeds for one ancestor are reduced to the lowest carried
    focal-site index, consistently across budgets; their intervals are not merged.
    """
    if len(max_mismatches) == 0 or any(budget < 0 for budget in max_mismatches):
        raise ValueError("Select at least one nonnegative mismatch budget")
    panel_path = data_dir / "ancestors" / f"{dataset}_inferred_ancestors.zarr"
    panel = zarr.open_group(panel_path, mode="r")
    positions = panel["variant_position"]
    num_sites = positions.shape[0]
    if num_sites == 0:
        raise ValueError("Coverage requires a nonempty panel site axis")
    focal_path = data_dir / "focal_ancestors" / f"{dataset}_inferred_focal_ancestors.npz"
    with np.load(focal_path, allow_pickle=False) as focal:
        haplotypes = pl.DataFrame(
            {"sample_id": focal["sample_id"], "ploidy_index": focal["ploidy_index"]}
        )
    keys = ["sample_id", "ploidy_index"]
    if haplotypes.unique(keys).height != haplotypes.height:
        raise ValueError("Focal NPZ has duplicate haplotype identities")

    stats_path = data_dir / "dataframes" / f"{dataset}_inferred_focal_ancestor_stats.csv"
    stats = pl.scan_csv(stats_path, schema_overrides={"sample_id": pl.String})
    if "population" in stats.collect_schema():
        labels = stats.select(*keys, "population").unique().collect()
        haplotypes = haplotypes.join(labels, on=keys, how="left", validate="1:1")
    elif truth_ts_path is not None:
        haplotypes = _truth_populations(haplotypes, truth_ts_path)
    else:
        raise ValueError("Population metadata is absent; supply the original truth TS")
    invalid_labels = haplotypes.filter(
        pl.col("population").is_null() | (pl.col("population") == "")
    )
    if invalid_labels.height > 0:
        raise ValueError("Every haplotype must have a population label")
    haplotypes = haplotypes.with_row_index("haplotype_index")

    chunks_path = (
        data_dir / "dataframes" / f"{dataset}_inferred_focal_ancestor_chunks.csv"
    )
    source = pl.scan_csv(chunks_path, schema_overrides={"sample_id": pl.String})
    available = source.select("max_mismatches").unique().collect()
    missing = set(max_mismatches) - set(available["max_mismatches"].to_list())
    if len(missing) > 0:
        raise ValueError(f"Chunk CSV lacks mismatch budgets {sorted(missing)}")
    selected = source.filter(
        (pl.col("focal_ac") >= 2)
        & (pl.col("focal_ac") <= max_ac)
        & pl.col("max_mismatches").is_in(max_mismatches)
    )
    chunk_keys = [*keys, "ancestor_index", "max_mismatches"]
    selected = selected.select(
        *chunk_keys,
        "focal_site_index",
        "focal_ac",
        "left_site_index",
        "right_site_index",
    )
    selected = selected.sort("focal_site_index").unique(subset=chunk_keys, keep="first")
    chunks = selected.collect(engine="streaming")
    chunks = chunks.join(
        haplotypes.select(*keys, "haplotype_index"),
        on=keys,
        how="left",
        validate="m:1",
    )
    if chunks["haplotype_index"].null_count() > 0:
        raise ValueError("Chunk CSV contains haplotypes absent from the focal NPZ")
    invalid_bounds = chunks.filter(
        (pl.col("left_site_index") < 0)
        | (pl.col("right_site_index") > num_sites)
        | (pl.col("left_site_index") >= pl.col("right_site_index"))
    )
    if invalid_bounds.height > 0:
        raise ValueError("Chunk bounds must satisfy 0 <= left < right <= num_sites")
    logger.info(
        "Loaded %s: %d sites, %d haplotypes, %d chunks",
        dataset,
        num_sites,
        haplotypes.height,
        chunks.height,
    )
    return CoverageData(
        dataset,
        num_sites,
        haplotypes,
        chunks,
        list(max_mismatches),
        int(positions[0]),
        int(positions[-1]),
    )


def _sweep_haplotype(task: SweepTask) -> SweepResult:
    """Sweep sorted endpoints, updating cumulative AC eligibility once per chunk.

    Coverage is constant between successive endpoints. Segment lengths are
    counts of sites, so histogram weights give exact order statistics without
    allocating a per-site coverage vector. Quantiles match NumPy's default
    linear interpolation on that conceptual vector, including zero coverage.
    Time is O(C log C + B E), memory O(C + E + max coverage), for C chunks,
    B cutoffs and E distinct endpoints; neither depends on BP length.
    """
    endpoints = np.concatenate(([0, task.num_sites], task.left, task.right))
    endpoints = np.unique(endpoints)
    lengths = np.diff(endpoints)
    left_events = np.searchsorted(endpoints, task.left)
    right_events = np.searchsorted(endpoints, task.right)
    delta = np.zeros(len(endpoints), dtype=np.int64)
    stop = 0
    records = []
    for cutoff in task.cutoffs:
        next_stop = int(np.searchsorted(task.focal_ac, cutoff, side="right"))
        np.add.at(delta, left_events[stop:next_stop], 1)
        np.add.at(delta, right_events[stop:next_stop], -1)
        stop = next_stop
        coverage = np.cumsum(delta)[:-1]
        histogram = np.zeros(int(coverage.max()) + 1, dtype=np.int64)
        np.add.at(histogram, coverage, lengths)
        cumulative = np.cumsum(histogram)
        ranks = np.array([0.05, 0.95]) * (task.num_sites - 1)
        lower_ranks = np.floor(ranks).astype(np.int64)
        upper_ranks = np.ceil(ranks).astype(np.int64)
        lower = np.searchsorted(cumulative, lower_ranks, side="right")
        upper = np.searchsorted(cumulative, upper_ranks, side="right")
        quantiles = lower + (upper - lower) * (ranks - lower_ranks)
        covered_sites = task.num_sites - int(histogram[0])
        total_coverage = int(np.dot(coverage, lengths))
        records.append(
            {
                "haplotype_index": task.haplotype_index,
                "focal_ac": int(cutoff),
                "evaluated_sites": task.num_sites,
                "covered_sites": covered_sites,
                "mean_coverage": total_coverage / task.num_sites,
                "min_coverage": int(coverage.min()),
                "max_coverage": int(coverage.max()),
                "q05_coverage": float(quantiles[0]),
                "q95_coverage": float(quantiles[1]),
                "fraction_covered": covered_sites / task.num_sites,
            }
        )
    return SweepResult(task.haplotype_index, records)


def _sweep_tasks(data: CoverageData, chunks: pl.DataFrame, cutoffs: np.ndarray):
    """Slice sorted numeric columns into disjoint haplotype queue tasks."""
    indices = chunks["haplotype_index"].to_numpy()
    counts = chunks["focal_ac"].to_numpy()
    left = chunks["left_site_index"].to_numpy()
    right = chunks["right_site_index"].to_numpy()
    for index in range(data.haplotypes.height):
        start = int(np.searchsorted(indices, index, side="left"))
        stop = int(np.searchsorted(indices, index, side="right"))
        yield SweepTask(
            index,
            data.num_sites,
            cutoffs,
            counts[start:stop],
            left[start:stop],
            right[start:stop],
        )


def summarise_coverage(
    data: CoverageData,
    cutoffs: np.ndarray,
    max_mismatches: int = 0,
    workers: int | None = None,
) -> pl.DataFrame:
    """Return per-haplotype statistics for inclusive maximum focal AC thresholds.

    ``workers=None`` uses all available CPUs, capped by the haplotype count.
    Spawned workers consume independent numeric tasks through Pool queues;
    ``workers=1`` runs the same sweep locally. See :func:`_sweep_haplotype` for
    site weights and quantile definitions. Keep both ploidy copies separate.
    """
    cutoffs = np.asarray(cutoffs)
    if cutoffs.ndim != 1 or len(cutoffs) == 0:
        raise ValueError("cutoffs must be a nonempty one-dimensional array")
    if not np.issubdtype(cutoffs.dtype, np.integer):
        raise ValueError("cutoffs must be integers")
    if np.any(cutoffs < 2) or np.any(np.diff(cutoffs) <= 0):
        raise ValueError("cutoffs must be strictly increasing and >= 2")
    if max_mismatches not in data.mismatch_budgets:
        raise ValueError("Reload chunks with the requested mismatch budget")
    if workers is None:
        workers = multiprocessing.cpu_count()
    if workers < 1:
        raise ValueError("workers must be >= 1")
    workers = min(workers, data.haplotypes.height)
    chunks = data.chunks.filter(pl.col("max_mismatches") == max_mismatches)
    chunks = chunks.sort("haplotype_index", "focal_ac")
    tasks = _sweep_tasks(data, chunks, cutoffs)
    if workers == 1:
        results = list(map(_sweep_haplotype, tasks))
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers) as pool:
            results = list(pool.imap_unordered(_sweep_haplotype, tasks, chunksize=1))
    records = []
    for result in results:
        records.extend(result.records)
    frame = pl.DataFrame(records)
    frame = frame.join(data.haplotypes, on="haplotype_index", validate="m:1")
    frame = frame.with_columns(
        pl.lit(data.dataset).alias("dataset"),
        pl.lit(max_mismatches).alias("max_mismatches"),
    )
    return frame.sort("haplotype_index", "focal_ac").drop("haplotype_index")
