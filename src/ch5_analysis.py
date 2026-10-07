"""Site-weighted focal-ancestor interval coverage, independent of HMM paths.

Read raw match-eval intervals with :func:`load_coverage_data`, then sweep each
haplotype with :func:`summarise_coverage`. Generation chooses one leftmost focal
anchor per association. Denominators include the panel's complete site axis.
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
    """Raw associations and the full haplotype roster, including empty candidates."""

    dataset: str
    num_sites: int
    haplotypes: pl.DataFrame
    offsets: np.ndarray
    focal_ac: np.ndarray
    left_site_index: np.ndarray
    right_site_index: np.ndarray
    max_ac: int
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
    """Load compact intervals and select requested budget columns without repacking.

    Validate the archive roster against the focal NPZ and site count against
    the panel. Population labels come from the statistics CSV or the original
    simulation TS via :func:`_truth_populations`. Exact AC arrays are retained;
    :func:`summarise_coverage` applies inclusive cumulative analysis cutoffs.
    Requested limits must be within those used to generate the archive.
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
    archive_path = (
        data_dir
        / "haplotype_intervals"
        / f"{dataset}_inferred_focal_ancestor_intervals.npz"
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_id"]
        ploidy_indices = archive["ploidy_index"]
        if sample_ids.ndim != 1 or sample_ids.dtype.kind != "U":
            raise ValueError("Interval archive sample_id must be a Unicode roster")
        for name, expected in (
            ("sample_id", haplotypes["sample_id"].to_numpy()),
            ("ploidy_index", haplotypes["ploidy_index"].to_numpy()),
        ):
            if not np.array_equal(archive[name], expected):
                raise ValueError(
                    f"Interval archive {name} disagrees with focal NPZ roster"
                )
        for name in (
            "ploidy_index",
            "offsets",
            "focal_ac",
            "left_site_index",
            "right_site_index",
            "num_sites",
            "max_ac_cutoff",
            "max_mismatches",
        ):
            if archive[name].dtype != np.dtype("int64"):
                raise ValueError(f"Interval archive {name} must use int64")
        for name in ("num_sites", "max_ac_cutoff", "max_mismatches"):
            if archive[name].shape != ():
                raise ValueError(f"Interval archive {name} must be scalar")
        if int(archive["num_sites"]) != num_sites:
            raise ValueError("Interval archive site count disagrees with panel")
        generated_ac = int(archive["max_ac_cutoff"])
        generated_budget = int(archive["max_mismatches"])
        if generated_ac < 1 or generated_budget < 0:
            raise ValueError("Interval archive generation limits are invalid")
        if max_ac < 1 or max_ac > generated_ac:
            raise ValueError(
                "Requested AC limit exceeds interval archive generation limit"
            )
        missing = sorted(set(max_mismatches) - set(range(generated_budget + 1)))
        if len(missing) > 0:
            raise ValueError(f"Interval archive lacks mismatch budgets {missing}")
        offsets = archive["offsets"]
        counts = archive["focal_ac"]
        left = archive["left_site_index"]
        right = archive["right_site_index"]
        if counts.ndim != 1 or ploidy_indices.shape != sample_ids.shape:
            raise ValueError(
                "Interval archive association/roster dimensions are invalid"
            )
        if (
            offsets.shape != (haplotypes.height + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(counts)
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError("Interval archive has invalid ragged offsets")
        shape = (len(counts), generated_budget + 1)
        if left.shape != shape or right.shape != shape:
            raise ValueError("Interval archive bounds have invalid dimensions")
        if np.any((counts < 1) | (counts > generated_ac)):
            raise ValueError("Interval archive focal AC is outside its generation limit")
        if np.any((left < 0) | (left >= right) | (right > num_sites)):
            raise ValueError(
                "Interval archive bounds must satisfy 0 <= left < right <= num_sites"
            )
        left = left[:, max_mismatches]
        right = right[:, max_mismatches]
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

    logger.info(
        "Loaded %s: %d sites, %d haplotypes, %d associations, budgets %s",
        dataset,
        num_sites,
        haplotypes.height,
        len(counts),
        max_mismatches,
    )
    return CoverageData(
        dataset=dataset,
        num_sites=num_sites,
        haplotypes=haplotypes,
        offsets=offsets,
        focal_ac=counts,
        left_site_index=left,
        right_site_index=right,
        max_ac=max_ac,
        mismatch_budgets=list(max_mismatches),
        first_position=int(positions[0]),
        last_position=int(positions[-1]),
    )


def _sweep_haplotype(task: SweepTask) -> SweepResult:
    """Sweep sorted endpoints, updating cumulative AC eligibility once per association.

    Coverage is constant between successive endpoints. Segment lengths are
    counts of sites, so histogram weights give exact order statistics without
    allocating a per-site coverage vector. Quantiles match NumPy's default
    linear interpolation on that conceptual vector, including zero coverage.
    Time is O(C log C + B E), memory O(C + E + max coverage), for C associations,
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


def _sweep_tasks(data: CoverageData, budget_column: int, cutoffs: np.ndarray):
    """Sort ragged associations by exact AC, keeping chapter 5's AC >= 2 convention."""
    for index in range(data.haplotypes.height):
        start = data.offsets[index]
        stop = data.offsets[index + 1]
        counts = data.focal_ac[start:stop]
        eligible = np.flatnonzero(counts >= 2)
        order = np.argsort(counts[eligible], kind="stable")
        selected = eligible[order]
        left = data.left_site_index[start:stop, budget_column]
        right = data.right_site_index[start:stop, budget_column]
        yield SweepTask(
            index,
            data.num_sites,
            cutoffs,
            counts[selected],
            left[selected],
            right[selected],
        )


def summarise_coverage(
    data: CoverageData,
    cutoffs: np.ndarray,
    max_mismatches: int = 0,
    workers: int | None = None,
) -> pl.DataFrame:
    """Return per-haplotype statistics for inclusive maximum focal AC thresholds.

    As in the original chapter 5 analysis, AC=1 associations do not contribute.
    They remain available in the raw archive and loaded arrays.
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
        raise ValueError("Reload intervals with the requested mismatch budget")
    if cutoffs[-1] > data.max_ac:
        raise ValueError("Summary cutoffs exceed the loaded analysis AC limit")
    if workers is None:
        workers = multiprocessing.cpu_count()
    if workers < 1:
        raise ValueError("workers must be >= 1")
    workers = min(workers, data.haplotypes.height)
    budget_column = data.mismatch_budgets.index(max_mismatches)
    tasks = _sweep_tasks(data, budget_column, cutoffs)
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
