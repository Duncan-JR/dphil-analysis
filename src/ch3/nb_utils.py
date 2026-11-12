"""Chapter 3 exploratory preparation and analysis.

Scientific notebook settings live in notebook cells. This is rough work.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import pathlib

import numpy as np
import pandas as pd
import polars as pl
import zarr

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AnalysisConfig:
    """Settings shared by the chapter figures; lengths in input files are bp."""

    versions: tuple[str, str] = ("0.4.1", "0.5.1")
    genotype_multipliers: tuple[float, ...] = (1, 2, 5, 10, 100)
    mispol_rates: tuple[float, ...] = (0, 0.01, 0.05, 0.1)
    phase_spacings: tuple[float, ...] = (0, 200000.0, 2000000.0, 20000000.0)
    tgp_region_start: int = 0
    tgp_region_end: int = 20000000
    tgp_count_prefix: str = "variant_1kgp_all_subset"
    bp_per_mb: float = 1000000.0
    bp_per_kb: float = 1000.0
    overshoot_summary_threshold_bp: float = -10000000.0
    preprocess_chunk_rows: int = 250000
    focal_keys: tuple[str, ...] = (
        "replicate",
        "version",
        "geno_multiplier",
        "space_per_switch",
        "phase_ser",
        "mispol_rate",
        "true_focal_site",
    )
    workers: int | None = None


@dataclasses.dataclass
class OvershootHistogram:
    """Shared physical bins and profile counts used by histogram and diff plots."""

    bin_edges: np.ndarray
    counts: dict[tuple[str, float | str], np.ndarray]


@dataclasses.dataclass
class TailSteps:
    edges: np.ndarray
    fractions: np.ndarray


@dataclasses.dataclass
class ExtensionConfig:
    genotypes: tuple[tuple[int, ...], ...]
    samples: tuple[str, ...] = tuple("adbcfg")
    focal_site: int = 1
    versions: tuple[str, str] = ("0.4.1", "0.5.1")


@dataclasses.dataclass
class PerformanceConfig:
    """Benchmark ordering, labels, units and figure settings."""

    versions: tuple[str, str] = ("0.4", "0.5")
    stages: tuple[str, ...] = ("generate_ancestors", "match_ancestors", "match_samples")
    seconds_per_minute: int = 60
    bytes_per_gib: int = 1024**3


@dataclasses.dataclass
class ChapterData:
    """Focal-site comparisons and the CSV exported by :mod:`scripts.extract_zarr_validation_data`."""

    overshoots: pd.DataFrame
    spans: pd.DataFrame
    zarr_data: pd.DataFrame
    config: AnalysisConfig

    @property
    def phase_settings(self):
        """Phasing-only spacing and SER metadata from the current aggregate."""
        profiles = self.spans.loc[
            (self.spans.geno_multiplier == 0) & (self.spans.mispol_rate == 0)
        ]
        return (
            profiles[["space_per_switch", "phase_ser"]]
            .drop_duplicates()
            .sort_values("space_per_switch")
        )

    @property
    def phase_spacings(self):
        return tuple(self.phase_settings.space_per_switch)


@dataclasses.dataclass
class GenotypeSummary:
    """Pooled rates and conditional transition rates returned by summarisation."""

    rates: pd.DataFrame
    transitions: pd.DataFrame


@dataclasses.dataclass
class FocalTask:
    frame: pd.DataFrame
    config: AnalysisConfig


@dataclasses.dataclass
class FocalFrames:
    overshoots: pd.DataFrame
    spans: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class FocalPaths:
    overshoots: pathlib.Path
    spans: pathlib.Path


def _prepare_focal_chunk(task):
    """Keep both sides of every focal site and one copy of its span comparison."""
    frame = task.frame.loc[task.frame.version.isin(task.config.versions)].copy()
    columns = list(task.config.focal_keys)
    columns += [
        "inferred_node",
        "true_node",
        "allele_frequency",
        "focal_site_mispolarised",
        "true_derived_count",
        "true_allele_frequency",
    ]
    overshoots = frame.loc[:, columns + ["side", "overshoot"]]
    spans = frame.loc[frame.side == "left", columns + ["inferred_span", "true_span"]]
    return FocalFrames(overshoots, spans)


def _read_zarr_data(path, prefix):
    """Load a CSV from :func:`scripts.extract_zarr_validation_data.extract_data` for this prefix."""
    frame = pd.read_csv(
        path,
        dtype={
            "record_type": "string",
            "data_prefix": "string",
            "mispolarised": "boolean",
        },
    )
    profiles = frame.loc[frame.record_type == "profile"]
    if set(profiles.data_prefix) != {prefix}:
        raise ValueError(f"Exported Zarr data in {path} does not describe {prefix}")
    return frame


def _true_focal_frequencies(zarr_data):
    """Unflipped focal counts/frequencies from exported error-free variants."""
    truth = zarr_data.loc[zarr_data.record_type == "truth"].copy()
    truth["true_derived_count"] = truth.derived_count.astype(np.int64)
    truth["true_allele_frequency"] = truth.derived_count / truth.num_alleles
    truth["replicate"] = truth.replicate.astype(np.int64)
    return truth[
        ["replicate", "focal_position", "true_derived_count", "true_allele_frequency"]
    ]


def _focal_tasks(reader, truth, config):
    for frame in reader:
        frame = frame.merge(
            truth,
            on=["replicate", "focal_position"],
            how="left",
            validate="many_to_one",
        )
        if frame.true_derived_count.isna().any():
            raise ValueError(
                "Focal positions are missing from the exported error-free variants"
            )
        yield FocalTask(frame, config)


def _write_focal_tables(results, paths):
    """Stream prepared chunks to disk without retaining the full aggregate."""
    side_count = 0
    site_count = 0
    temporary = FocalPaths(
        paths.overshoots.with_suffix(".csv.tmp"), paths.spans.with_suffix(".csv.tmp")
    )
    with (
        temporary.overshoots.open("w") as overshoot_file,
        temporary.spans.open("w") as span_file,
    ):
        for index, result in enumerate(results):
            result.overshoots.to_csv(overshoot_file, index=False, header=index == 0)
            result.spans.to_csv(span_file, index=False, header=index == 0)
            side_count += len(result.overshoots)
            site_count += len(result.spans)
    temporary.overshoots.replace(paths.overshoots)
    temporary.spans.replace(paths.spans)
    logger.info(
        "Wrote %s focal-site sides and %s focal-site spans", side_count, site_count
    )


def preprocess_data(aggregated_path, zarr_data_path, output_folder, config=None):
    """Prepare per-focal-site comparisons for :func:`load_data` and all plots.

    Each focal site retains its own true/inferred span comparison and both
    signed overshoots within each replicate, version and error profile. Sites
    sharing an ancestor remain separate, including sites with different
    polarisation states. True focal counts/frequencies come from the CSV
    generated by :mod:`scripts.extract_zarr_validation_data`; no Zarr stores are opened.
    """
    if config is None:
        config = AnalysisConfig()
    aggregated_path = pathlib.Path(aggregated_path).expanduser()
    if not aggregated_path.name.endswith("-aggregated.csv"):
        raise ValueError("aggregated_path must end with -aggregated.csv")
    prefix = aggregated_path.name.removesuffix("-aggregated.csv")
    folder = pathlib.Path(output_folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    paths = FocalPaths(
        folder / f"{prefix}_focal_overshoots.csv", folder / f"{prefix}_focal_spans.csv"
    )
    columns = list(config.focal_keys)
    columns += [
        "inferred_node",
        "true_node",
        "side",
        "allele_frequency",
        "inferred_span",
        "true_span",
        "overshoot",
        "focal_position",
        "focal_site_mispolarised",
    ]
    reader = pd.read_csv(
        aggregated_path,
        usecols=columns,
        dtype={"version": str},
        chunksize=config.preprocess_chunk_rows,
    )
    zarr_data = _read_zarr_data(pathlib.Path(zarr_data_path).expanduser(), prefix)
    truth = _true_focal_frequencies(zarr_data)
    tasks = _focal_tasks(reader, truth, config)
    workers = config.workers
    if workers is None:
        workers = multiprocessing.cpu_count()
    if workers == 1:
        results = map(_prepare_focal_chunk, tasks)
        _write_focal_tables(results, paths)
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            results = pool.imap(_prepare_focal_chunk, tasks)
            _write_focal_tables(results, paths)
    return paths


def load_data(data_folder, data_prefix, config=None):
    """Load per-focal-site tables and PREFIX_zarr_data.csv; see `preprocess_data`."""
    if config is None:
        config = AnalysisConfig()
    folder = pathlib.Path(data_folder).expanduser()
    overshoots = pd.read_csv(folder / f"{data_prefix}_focal_overshoots.csv")
    spans = pd.read_csv(folder / f"{data_prefix}_focal_spans.csv")
    overshoots = overshoots.loc[overshoots.version.isin(config.versions)].copy()
    spans = spans.loc[spans.version.isin(config.versions)].copy()
    if set(spans.version) != set(config.versions):
        raise ValueError(f"Ancestor CSV must contain versions {config.versions}")
    zarr_data = _read_zarr_data(folder / f"{data_prefix}_zarr_data.csv", data_prefix)
    return ChapterData(overshoots, spans, zarr_data, config)


def _version_label(version):
    """Display the major/minor release while retaining full data identifiers."""
    release = ".".join(version.split(".")[:2])
    return f"tsinfer {release}"


def _no_errors(frame):
    return frame.loc[
        (frame.geno_multiplier == 0) & (frame.phase_ser == 0) & (frame.mispol_rate == 0)
    ].copy()


def _frequency_bin_edges(frequencies, num_bins=5, cutoffs=None):
    """One-significant-digit quantiles over all DAFs, or explicit bin edges.

    The outer bounds round outwards to include every observation. Tied rounded
    quantiles merge; ``cutoffs`` overrides both rounding and ``num_bins``.
    """
    if cutoffs is not None:
        edges = np.asarray(cutoffs, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("cutoffs must contain at least two edges")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("cutoffs must be finite and strictly increasing")
        return edges
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    frequencies = np.asarray(frequencies)
    boundaries = np.quantile(frequencies, np.linspace(0, 1, num_bins + 1))
    rounded = boundaries.copy()
    positive = boundaries > 0
    magnitudes = 10.0 ** np.floor(np.log10(boundaries[positive]))
    rounded[positive] = np.ceil(boundaries[positive] / magnitudes) * magnitudes
    if boundaries[0] > 0:
        magnitude = 10.0 ** np.floor(np.log10(boundaries[0]))
        rounded[0] = np.floor(boundaries[0] / magnitude) * magnitude
    edges = np.unique(rounded)
    if len(edges) < 2:
        raise ValueError("Quantile binning requires distinct frequency bounds")
    if len(edges) - 1 != num_bins:
        logger.warning(
            "Rounded/tied quantiles yield %s of %s requested bins",
            len(edges) - 1,
            num_bins,
        )
    return edges


def load_tgp_ancestor_spans(path):
    """Load full focal-site spans from the frequency-timed 1KGP benchmark.

    The external performance pipeline's empirical-span extractor exports one row per focal site.
    The benchmark uses the default frequency proxy for ancestor times, so
    ``inferred_time`` is the focal derived allele frequency, not an age.
    """
    frame = pd.read_csv(
        pathlib.Path(path).expanduser(),
        dtype={"version": str},
        usecols=[
            "version",
            "inf_focal_site",
            "focal_position",
            "inferred_time",
            "inferred_span",
        ],
    )
    return frame.rename(columns={"inferred_time": "allele_frequency"})


def summarise_ancestor_span_curves(data, tgp_spans, *, ancestor_error_bins=20):
    """Mean full spans per focal site in the error-direction figure's DAF bins.

    Retain each focal site even when it shares an ancestor with another site.
    Simulation rows come from the error-free profile in ``data.spans``; truth
    is included once using the first configured version's matched focal sites.
    Use :func:`load_tgp_ancestor_spans` for the empirical rows. All spans remain
    at their exported lengths, without clipping to a common genomic region.
    Empty bins have zero focal sites and undefined means. DAF 1 is included
    in the final bin. See :func:`plot_ancestor_span_curves`.
    """
    config = data.config
    simulation = _no_errors(data.spans)
    simulation = simulation.loc[simulation.version.isin(config.versions)].copy()
    if simulation.duplicated(["replicate", "version", "true_focal_site"]).any():
        raise ValueError("Expected one simulation span per focal site and version")
    empirical = tgp_spans.loc[tgp_spans.version.isin(config.versions)].copy()
    if empirical.duplicated(["version", "inf_focal_site"]).any():
        raise ValueError("Expected one 1KGP span per focal site and version")
    truth = simulation.loc[simulation.version == config.versions[0]].copy()
    truth["version"] = "True"
    truth["inferred_span"] = truth.true_span
    columns = ["version", "allele_frequency", "inferred_span"]
    simulated = pd.concat([simulation[columns], truth[columns]], ignore_index=True)
    simulated["dataset"] = "Simulation"
    empirical = empirical[columns].copy()
    empirical["dataset"] = "1000 Genomes Project"
    frame = pd.concat([simulated, empirical], ignore_index=True)
    if not frame.allele_frequency.between(0, 1).all():
        raise ValueError("Focal allele frequencies must be in [0, 1]")
    num_bins = ancestor_error_bins
    if num_bins < 1:
        raise ValueError("The number of frequency bins must be positive")
    edges = np.arange(num_bins + 1) / num_bins
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    frame["bin"] = pd.cut(frame.allele_frequency, cut_edges, right=False, labels=False)
    grouped = frame.groupby(["dataset", "version", "bin"]).agg(
        num_focal_sites=("inferred_span", "size"),
        mean_span_bp=("inferred_span", "mean"),
    )
    groups = [("Simulation", version) for version in (*config.versions, "True")]
    groups += [("1000 Genomes Project", version) for version in config.versions]
    index = pd.MultiIndex.from_tuples(
        [
            (dataset, version, bin_id)
            for dataset, version in groups
            for bin_id in range(num_bins)
        ],
        names=["dataset", "version", "bin"],
    )
    summary = grouped.reindex(index).reset_index()
    summary["num_focal_sites"] = summary.num_focal_sites.fillna(0).astype(int)
    summary["mean_span_mb"] = summary.mean_span_bp / config.bp_per_mb
    summary["bin_left"] = edges[summary.bin]
    summary["bin_right"] = edges[summary.bin + 1]
    summary["daf"] = (summary.bin_left + summary.bin_right) / 2
    return summary


def load_ancestor_overlap_errors(aggregated_path, config=None):
    """Read error-free whole-overlap comparisons from the aggregate CSV.

    ``num_errors`` and ``overlap_span`` describe the complete shared overlap,
    duplicated on the left/right rows. Keep only left rows, retaining one
    comparison per focal site and version as in :func:`plot_ancestor_overshoot`.
    The overlap is shared between the truth and both inferred versions.
    """
    if config is None:
        config = AnalysisConfig()
    query = pl.scan_csv(pathlib.Path(aggregated_path).expanduser())
    selected = query.filter(
        (pl.col("side") == "left")
        & (pl.col("geno_multiplier") == 0)
        & (pl.col("phase_ser") == 0)
        & (pl.col("mispol_rate") == 0)
        & pl.col("version").cast(pl.String).is_in(config.versions)
    )
    selected = selected.select(
        "replicate",
        "version",
        "true_focal_site",
        "allele_frequency",
        "num_errors",
        "overlap_span",
        "overlap_site_span",
        "num_should_be_0",
        "num_should_be_1",
    )
    table = selected.collect(engine="streaming")
    return pd.DataFrame(table.to_dict(as_series=False))


def load_demography_comparisons(aggregated_paths, config=None):
    """Load both error-free sides per focal site, tagged by model identifier.

    ``aggregated_paths`` maps model identifiers to aggregate CSV paths. Read
    only the columns needed for overshoots and whole-overlap mismatch fraction;
    exported Zarr summaries are not required. See
    :func:`plot_demography_overshoot_box` and
    :func:`summarise_demography_overlap_errors`.
    """
    if config is None:
        config = AnalysisConfig()
    frames = []
    for model, path in aggregated_paths.items():
        query = pl.scan_csv(pathlib.Path(path).expanduser())
        selected = query.filter(
            (pl.col("geno_multiplier") == 0)
            & (pl.col("phase_ser") == 0)
            & (pl.col("mispol_rate") == 0)
            & pl.col("version").cast(pl.String).is_in(config.versions)
        )
        selected = selected.select(
            "replicate",
            "version",
            "true_focal_site",
            "side",
            "overshoot",
            "num_errors",
            "overlap_span",
            "overlap_site_span",
            "num_should_be_0",
            "num_should_be_1",
        )
        table = selected.collect(engine="streaming")
        frame = pd.DataFrame(table.to_dict(as_series=False))
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _pool_overlap_errors(frame, group_columns):
    """Sum whole-comparison errors and inference sites before dividing."""
    frame = frame.copy()
    frame["error_free"] = frame.num_errors == 0
    return frame.groupby(group_columns, sort=False).agg(
        num_ancestors=("true_focal_site", "size"),
        num_error_free=("error_free", "sum"),
        num_errors=("num_errors", "sum"),
        should_be_0_errors=("num_should_be_1", "sum"),
        should_be_1_errors=("num_should_be_0", "sum"),
        overlap_bp=("overlap_span", "sum"),
        overlap_sites=("overlap_site_span", "sum"),
    )


def _overlap_error_rates(summary):
    """Add pooled mismatch fractions and zero-error proportions to summed comparisons."""
    summary = summary.copy()
    ancestor_denominator = summary.num_ancestors.replace(0, np.nan)
    overlap_denominator = summary.overlap_sites.replace(0, np.nan)
    summary["error_free_fraction"] = summary.num_error_free / ancestor_denominator
    summary["mismatch_fraction"] = summary.num_errors / overlap_denominator
    summary["should_be_0_mismatch_fraction"] = (
        summary.should_be_0_errors / overlap_denominator
    )
    summary["should_be_1_mismatch_fraction"] = (
        summary.should_be_1_errors / overlap_denominator
    )
    return summary


def summarise_demography_overlap_errors(comparisons):
    """Pool all frequencies within each model/version, counting one side once.

    Retain one whole-overlap comparison per focal site, matching
    :func:`load_ancestor_overlap_errors`. Focal-site IDs are local to a model.
    """
    frame = comparisons.loc[comparisons.side == "left"]
    keys = ["model", "replicate", "version", "true_focal_site"]
    if frame.duplicated(keys).any():
        raise ValueError("Expected one overlap comparison per focal site and version")
    summary = _pool_overlap_errors(frame, ["model", "version"]).reset_index()
    return _overlap_error_rates(summary)


def summarise_ancestor_overlap_errors(
    comparisons, config=None, num_bins=5, cutoffs=None
):
    """Pool counts and compared inference sites within each version/frequency bin.

    Mismatch fraction is the ratio of sums, never the mean of per-overlap rates.
    Zero-error proportions and counts use one whole comparison per focal site.
    Empty bins have zero counts and undefined rates. See
    :func:`load_ancestor_overlap_errors` and :func:`plot_ancestor_overlap_errors`.
    """
    if config is None:
        config = AnalysisConfig()
    keys = ["replicate", "version", "true_focal_site"]
    if comparisons.duplicated(keys).any():
        raise ValueError("Expected one overlap comparison per focal site and version")
    frequencies = comparisons.drop_duplicates(["replicate", "true_focal_site"])
    edges = _frequency_bin_edges(frequencies.allele_frequency, num_bins, cutoffs)
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    frame = comparisons.copy()
    frame["bin"] = pd.cut(frame.allele_frequency, cut_edges, right=False, labels=False)
    grouped = _pool_overlap_errors(frame, ["version", "bin"])
    index = pd.MultiIndex.from_product(
        [config.versions, range(len(edges) - 1)], names=["version", "bin"]
    )
    summary = grouped.reindex(index, fill_value=0).reset_index()
    summary["bin_left"] = edges[summary.bin]
    summary["bin_right"] = edges[summary.bin + 1]
    return _overlap_error_rates(summary)


def summarise_ancestor_error_types(
    comparisons, config=None, *, ancestor_error_bins=20, ancestor_error_version="0.5.1"
):
    """Split tsinfer 0.5 focal sites into fixed-width DAF bins across [0, 1].

    Plot rates at bin midpoints, including DAF 1 in the final bin. Empty bins
    have undefined rates. Pool each error type over compared inference sites, as in
    :func:`summarise_ancestor_overlap_errors`.

    The aggregate writer reverses its ``num_should_be_*`` names:
    ``num_should_be_0`` counts true 1 / inferred 0, and ``num_should_be_1``
    counts true 0 / inferred 1. Output names describe the actual true state.
    """
    if config is None:
        config = AnalysisConfig()
    frame = comparisons.loc[comparisons.version == ancestor_error_version].copy()
    keys = ["replicate", "true_focal_site"]
    if frame.duplicated(keys).any():
        raise ValueError("Expected one overlap comparison per focal site and version")
    num_bins = ancestor_error_bins
    if num_bins < 1:
        raise ValueError("The number of frequency bins must be positive")
    edges = np.arange(num_bins + 1) / num_bins
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    frame["bin"] = pd.cut(frame.allele_frequency, cut_edges, right=False, labels=False)
    grouped = frame.groupby("bin")
    summary = grouped.agg(
        num_ancestors=("true_focal_site", "size"),
        overlap_bp=("overlap_span", "sum"),
        overlap_sites=("overlap_site_span", "sum"),
        should_be_0_errors=("num_should_be_1", "sum"),
        should_be_1_errors=("num_should_be_0", "sum"),
    )
    summary = summary.reindex(range(num_bins), fill_value=0).reset_index()
    summary["bin_left"] = edges[:-1]
    summary["bin_right"] = edges[1:]
    summary["daf"] = (edges[:-1] + edges[1:]) / 2
    denominator = summary.overlap_sites.replace(0, np.nan)
    summary["should_be_0_mismatch_fraction"] = summary.should_be_0_errors / denominator
    summary["should_be_1_mismatch_fraction"] = summary.should_be_1_errors / denominator
    return summary


def summarise_genotype_errors(data, *, transition_bins=40):
    """Pool exported diploid error counts and bin transitions by true DAF.

    :mod:`scripts.extract_zarr_validation_data` compares each observed store to its error-free
    replicate. Counts are exported by exact true allele count, allowing
    transition_bins to change locally without access to the original stores.
    """
    config = data.config
    frame = data.zarr_data.loc[data.zarr_data.record_type == "genotype"]
    replicates = set(data.spans.replicate)
    rate_rows = []
    transition_rows = []
    edges = np.linspace(0, 1, transition_bins + 1)
    for multiplier in config.genotype_multipliers:
        selected = frame.loc[frame.geno_multiplier == multiplier]
        if set(selected.replicate) != replicates:
            raise ValueError(
                f"Missing exported genotype counts for multiplier {multiplier}"
            )
        changed = int(selected.changed.sum())
        total = int(selected.total.sum())
        rate_rows.append(
            {
                "multiplier": multiplier,
                "changed": changed,
                "total": total,
                "error_rate": changed / total,
            }
        )
        frequencies = selected.derived_count / selected.num_alleles
        for dosage, transition in [(0, "00 → 01/10"), (2, "11 → 01/10")]:
            denominator_weights = selected[f"homozygous_{dosage}"].to_numpy(
                dtype=np.int64
            )
            numerator_weights = selected[f"to_heterozygous_{dosage}"].to_numpy(
                dtype=np.int64
            )
            denominators = np.histogram(
                frequencies, edges, weights=denominator_weights
            )[0]
            numerators = np.histogram(frequencies, edges, weights=numerator_weights)[0]
            for bin_index in range(transition_bins):
                denominator = denominators[bin_index]
                numerator = numerators[bin_index]
                rate = numerator / denominator if denominator > 0 else np.nan
                transition_rows.append(
                    {
                        "multiplier": multiplier,
                        "transition": transition,
                        "daf": (edges[bin_index] + edges[bin_index + 1]) / 2,
                        "changed": numerator,
                        "total": denominator,
                        "error_rate": rate,
                    }
                )
    return GenotypeSummary(pd.DataFrame(rate_rows), pd.DataFrame(transition_rows))


def _overshoot_histogram_counts(
    frame, by, rates, config, num_bins=None, *, histogram_bins=100
):
    """Bin both versions on common Mb edges; used by histogram and diff plots."""
    frame = frame.loc[frame[by].isin(rates)]
    values = frame.overshoot.to_numpy() / config.bp_per_mb
    if num_bins is None:
        num_bins = histogram_bins
    bin_edges = np.histogram_bin_edges(values, bins=num_bins)
    counts = {}
    for version in config.versions:
        for rate in rates:
            selected = frame.loc[(frame.version == version) & (frame[by] == rate)]
            if selected.empty:
                raise ValueError(f"No overshoots for {version}, {by}={rate}")
            histogram = np.histogram(selected.overshoot / config.bp_per_mb, bin_edges)[
                0
            ]
            counts[version, rate] = histogram
    return OvershootHistogram(bin_edges, counts)


def load_combined_error_comparisons(
    aggregated_path, config=None, *, combined_error_profiles=("none", "all")
):
    """Read correctly polarised focal sites in the named `none` and `all` profiles.

    Profile names come from the aggregate rather than inferred combinations
    of error parameters. Whole-overlap metrics occur on both side rows; use
    :func:`summarise_combined_overlap_errors` to count each overlap once.
    """
    if config is None:
        config = AnalysisConfig()
    query = pl.scan_csv(pathlib.Path(aggregated_path).expanduser())
    selected = query.filter(
        pl.col("error_profile").is_in(combined_error_profiles)
        & pl.col("version").cast(pl.String).is_in(config.versions)
        & ~pl.col("focal_site_mispolarised")
    )
    columns = [
        "error_profile",
        "replicate",
        "version",
        "true_focal_site",
        "side",
        "geno_multiplier",
        "space_per_switch",
        "phase_ser",
        "mispol_rate",
        "focal_site_mispolarised",
        "num_errors",
        "overlap_span",
        "overlap_site_span",
        "num_should_be_0",
        "num_should_be_1",
        "overshoot",
    ]
    table = selected.select(columns).collect(engine="streaming")
    return pd.DataFrame(table.to_dict(as_series=False))


def summarise_combined_overlap_errors(comparisons):
    """Pool whole-overlap counts and bp separately for each profile/version."""
    frame = comparisons.loc[comparisons.side == "left"]
    keys = ["error_profile", "replicate", "version", "true_focal_site"]
    if frame.duplicated(keys).any():
        raise ValueError(
            "Expected one overlap comparison per profile, focal site and version"
        )
    summary = _pool_overlap_errors(frame, ["error_profile", "version"]).reset_index()
    return _overlap_error_rates(summary)


def _tail_steps(values):
    """Empirical fraction at or below each magnitude cutoff, counting focal sites once."""
    positions, counts = np.unique(values, return_counts=True)
    fractions = np.cumsum(counts) / len(values)
    return TailSteps(positions, fractions)


def _genotype_overshoots(data):
    frame = data.overshoots
    return frame.loc[(frame.phase_ser == 0) & (frame.mispol_rate == 0)]


def summarise_overshoots(frame, by, rates, config=None):
    """Pool signed focal-site sides for medians, IQRs and the negative tail.

    Quantiles use linear interpolation. The tail fraction counts sides strictly
    below the configured signed bp threshold; left and right are not averaged.
    """
    if config is None:
        config = AnalysisConfig()
    selected = frame.loc[frame[by].isin(rates) & frame.version.isin(config.versions)]
    grouped = selected.groupby([by, "version"], sort=False).overshoot
    summary = grouped.agg(
        num_sides="size",
        q25_bp=lambda values: values.quantile(0.25),
        median_bp="median",
        q75_bp=lambda values: values.quantile(0.75),
        fraction_below_threshold=lambda values: values.lt(
            config.overshoot_summary_threshold_bp
        ).mean(),
    )
    index = pd.MultiIndex.from_product([rates, config.versions], names=[by, "version"])
    result = summary.reindex(index).reset_index()
    if by == "space_per_switch":
        metadata = selected[[by, "phase_ser"]].drop_duplicates()
        result = result.merge(metadata, on=by, how="left", validate="many_to_one")
    return result


def _mean_overshoot_by_profile(frame, by, rates):
    """Pool focal-site means, averaging each site's two sides before pooling."""
    frame = frame.loc[frame[by].isin(rates)]
    group_keys = ["version", by]
    focal_keys = group_keys + ["replicate", "true_focal_site"]
    focal_means = frame.groupby(focal_keys).overshoot.mean()
    return focal_means.groupby(level=group_keys).mean()


def load_tgp_daf(zarr_path, config=None):
    """Load TGP counts in the configured half-open region without variant filters.

    The all-sample counts already account for polarisation; ancestral plus
    derived counts give the called denominator. No variant masks are read.
    See :func:`plot_tgp_daf` for the matching spectrum.
    """
    if config is None:
        config = AnalysisConfig()
    group = zarr.open_group(zarr_path, mode="r")
    positions = group["variant_position"][:]
    start = np.searchsorted(positions, config.tgp_region_start)
    stop = np.searchsorted(positions, config.tgp_region_end)
    region = slice(start, stop)
    prefix = config.tgp_count_prefix
    derived = group[f"{prefix}_derived_count"][region]
    ancestral = group[f"{prefix}_ancestral_count"][region]
    return pd.DataFrame(
        {
            "position": positions[region],
            "derived_count": derived,
            "num_alleles": derived + ancestral,
        }
    )


def mispolarisation_daf_counts(data):
    """Refresh segregating-site DAF counts from current focal-site polarisation.

    Every mispolarised segregating site is represented in the focal tables,
    including true singletons that become high-frequency inference sites.
    Unflipped singletons are absent there, so exported truth supplies the total
    count at each true allele count. Fixed sites retain their exported records.
    See :func:`plot_mispolarisation_daf` for polarisation and binning.
    """
    truth = data.zarr_data.loc[data.zarr_data.record_type == "truth"]
    keys = ["replicate", "derived_count", "num_alleles"]
    totals = truth.groupby(keys).size().rename("total_sites").reset_index()
    segregating = totals.loc[
        (totals.derived_count > 0) & (totals.derived_count < totals.num_alleles)
    ]
    focal = data.spans.loc[
        (data.spans.version == data.config.versions[0])
        & (data.spans.geno_multiplier == 0)
        & (data.spans.phase_ser == 0)
        & data.spans.focal_site_mispolarised
    ]
    flipped = focal.groupby(["replicate", "mispol_rate", "true_derived_count"]).size()
    flipped = flipped.rename("num_sites").reset_index()
    flipped = flipped.rename(columns={"true_derived_count": "derived_count"})
    frames = []
    for rate in data.config.mispol_rates:
        selected = flipped.loc[flipped.mispol_rate == rate, keys[:2] + ["num_sites"]]
        counts = segregating.merge(
            selected, on=keys[:2], how="left", validate="one_to_one"
        )
        counts["num_sites"] = counts.num_sites.fillna(0).astype(np.int64)
        counts["mispol_rate"] = rate
        counts["mispolarised"] = True
        unflipped = counts.copy()
        unflipped["num_sites"] = counts.total_sites - counts.num_sites
        unflipped["mispolarised"] = False
        frames.extend([counts, unflipped])
    exported = data.zarr_data.loc[data.zarr_data.record_type == "daf"]
    fixed = exported.loc[
        (exported.derived_count == 0) | (exported.derived_count == exported.num_alleles)
    ]
    frames.append(fixed)
    frequencies = pd.concat(frames, ignore_index=True)
    frequencies["mispolarised"] = frequencies.mispolarised.astype(bool)
    return frequencies


def load_mispolarisation_overlap_errors(
    aggregated_path, config=None, *, mispolarisation_error_rates=(0.01, 0.1)
):
    """Read one overlap per focal site for the baseline and pure mispol. profiles.

    Split the latter by its own focal site's polarisation, retaining sites
    separately even when they share an inferred ancestor. Error counts already
    compare truth and inference in the same observed allele encoding.
    """
    if config is None:
        config = AnalysisConfig()
    query = pl.scan_csv(pathlib.Path(aggregated_path).expanduser())
    selected = query.filter(
        (pl.col("side") == "left")
        & (pl.col("geno_multiplier") == 0)
        & (pl.col("phase_ser") == 0)
        & pl.col("mispol_rate").is_in([0.0, *mispolarisation_error_rates])
        & pl.col("version").cast(pl.String).is_in(config.versions)
    )
    focal_group = (
        pl.when(pl.col("focal_site_mispolarised"))
        .then(pl.lit("mispolarised"))
        .otherwise(pl.lit("correctly_polarised"))
    )
    rate_group = pl.concat_str(pl.col("mispol_rate"), focal_group, separator="_")
    group = (
        pl.when(pl.col("mispol_rate") == 0).then(pl.lit("none")).otherwise(rate_group)
    )
    selected = selected.select(
        group.alias("group"),
        "replicate",
        "version",
        "true_focal_site",
        "focal_position",
        "mispol_rate",
        "focal_site_mispolarised",
        "allele_frequency",
        "num_errors",
        "overlap_span",
        "overlap_site_span",
        "num_should_be_0",
        "num_should_be_1",
    )
    table = selected.collect(engine="streaming")
    return pd.DataFrame(table.to_dict(as_series=False))


def summarise_mispolarisation_overlap_errors(comparisons):
    """Pool :func:`load_mispolarisation_overlap_errors` by focal polarisation."""
    keys = ["group", "replicate", "version", "true_focal_site"]
    if comparisons.duplicated(keys).any():
        raise ValueError(
            "Expected one overlap comparison per group, focal site and version"
        )
    summary = _pool_overlap_errors(comparisons, ["group", "version"]).reset_index()
    return _overlap_error_rates(summary)


def _pool_mispolarisation_mismatch(frame, group_columns, index):
    """Pool errors with both bp and site denominators, preserving empty bins."""
    grouped = frame.groupby(group_columns, sort=False).agg(
        num_ancestors=("true_focal_site", "size"),
        num_errors=("num_errors", "sum"),
        overlap_bp=("overlap_span", "sum"),
        overlap_sites=("overlap_site_span", "sum"),
    )
    summary = grouped.reindex(index, fill_value=0)
    summary["mismatch_fraction"] = summary.num_errors / summary.overlap_sites.replace(
        0, np.nan
    )
    return summary.reset_index()


def summarise_mispolarisation_mismatch_by_daf(
    comparisons,
    true_frequencies,
    config=None,
    num_bins=None,
    *,
    ancestor_error_version="0.5.1",
    mispolarisation_mismatch_bins=10,
    mispolarisation_mismatch_groups=("none", "mispolarised", "correctly_polarised"),
    mispolarisation_mismatch_rate=0.1,
):
    """Pool error-free, mispolarised and non-mispolarised focals by true DAF.

    Consume :func:`load_mispolarisation_overlap_errors` and true frequencies
    from :func:`_true_focal_frequencies`. Match truth by replicate and physical
    focal position, including mispolarised true singletons absent from the
    error-free inference sites. Each focal contributes one whole overlap.
    Bins have equal width on [0, 1], including 1 in the final bin. Errors and compared sites are pooled before division; empty bins have undefined fractions.
    """
    if config is None:
        config = AnalysisConfig()
    if num_bins is None:
        num_bins = mispolarisation_mismatch_bins
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    selected = comparisons.loc[
        (comparisons.version == ancestor_error_version)
        & comparisons.mispol_rate.isin([0, mispolarisation_mismatch_rate])
    ].copy()
    keys = ["mispol_rate", "replicate", "true_focal_site"]
    if selected.duplicated(keys).any():
        raise ValueError("Expected one overlap comparison per profile and focal site")
    truth_keys = ["replicate", "focal_position"]
    truth = true_frequencies[truth_keys + ["true_allele_frequency"]]
    selected = selected.merge(truth, on=truth_keys, how="left", validate="many_to_one")
    if selected.true_allele_frequency.isna().any():
        raise ValueError("Missing true focal frequencies")
    selected["focal_group"] = "correctly_polarised"
    selected.loc[selected.focal_site_mispolarised, "focal_group"] = "mispolarised"
    selected.loc[selected.mispol_rate == 0, "focal_group"] = "none"
    edges = np.linspace(0, 1, num_bins + 1)
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    selected["bin"] = pd.cut(
        selected.true_allele_frequency, cut_edges, right=False, labels=False
    )
    index = pd.MultiIndex.from_product(
        [mispolarisation_mismatch_groups, range(num_bins)], names=["focal_group", "bin"]
    )
    binned = _pool_mispolarisation_mismatch(selected, ["focal_group", "bin"], index)
    binned["bin_left"] = edges[binned.bin]
    binned["bin_right"] = edges[binned.bin + 1]
    return binned


def _mispolarisation_overshoots(data, frequency):
    """Select pure profiles using unflipped true focal DAF, including 0.5 in high."""
    frame = data.overshoots
    pure = frame.loc[(frame.geno_multiplier == 0) & (frame.phase_ser == 0)]
    if frequency == "high":
        return pure.loc[pure.true_allele_frequency >= 0.5]
    if frequency == "low":
        return pure.loc[pure.true_allele_frequency < 0.5]
    if frequency == "all":
        return pure
    raise ValueError('frequency must be "high", "low" or "all"')


def _mispolarisation_low_count_overshoots(data, max_derived_count):
    """Select the shared low-count subset for histogram and diff plots."""
    frame = _mispolarisation_overshoots(data, frequency="all")
    return frame.loc[
        frame.focal_site_mispolarised & (frame.true_derived_count <= max_derived_count)
    ]


def _mispolarised_focal_sites(data, *, ancestor_error_version="0.5.1"):
    """One record per flipped focal site and rate, using the selected 0.5 version."""
    spans = data.spans
    rates = [rate for rate in data.config.mispol_rates if rate > 0]
    return spans.loc[
        (spans.version == ancestor_error_version)
        & (spans.geno_multiplier == 0)
        & (spans.phase_ser == 0)
        & spans.mispol_rate.isin(rates)
        & spans.focal_site_mispolarised
    ]


def summarise_mispolarised_focal_counts(data, *, ancestor_error_version="0.5.1"):
    """True singleton/doubleton proportions among all flipped focal sites by rate.

    Use the same cohort as panel C of :func:`plot_mispolarised_focal_overshoot_hist`.
    Each focal site is counted once, including sites sharing an ancestor.
    True counts come from exported error-free variants, before mispolarisation.
    """
    focal_sites = _mispolarised_focal_sites(
        data, ancestor_error_version=ancestor_error_version
    ).copy()
    focal_sites["true_singleton"] = focal_sites.true_derived_count == 1
    focal_sites["true_doubleton"] = focal_sites.true_derived_count == 2
    summary = focal_sites.groupby("mispol_rate").agg(
        num_mispol_focal_sites=("true_derived_count", "size"),
        num_true_singletons=("true_singleton", "sum"),
        num_true_doubletons=("true_doubleton", "sum"),
    )
    summary["proportion_true_singletons"] = (
        summary.num_true_singletons / summary.num_mispol_focal_sites
    )
    summary["proportion_true_doubletons"] = (
        summary.num_true_doubletons / summary.num_mispol_focal_sites
    )
    summary = summary.reset_index()
    return summary[
        [
            "mispol_rate",
            "num_mispol_focal_sites",
            "num_true_singletons",
            "proportion_true_singletons",
            "num_true_doubletons",
            "proportion_true_doubletons",
        ]
    ]


def _phasing_overshoots(data):
    frame = data.overshoots
    return frame.loc[(frame.geno_multiplier == 0) & (frame.mispol_rate == 0)]


@dataclasses.dataclass
class ExtensionTrace:
    haplotype: np.ndarray
    marks: dict[int, list[int]]
    removals: dict[int, int]
    start: int
    end: int


def trace_ancestor(config, version):
    """Replay the selected algorithm on complete binary data, recording conflicts.

    Indexes in the returned trace are zero-based. Younger sites are assigned 0
    even when their sample-set consensus is 1. In the 0.5.x backport both
    removal and flag updates require an older site or a derived carrier
    outside the *current* sample set.
    """
    if version not in config.versions:
        raise ValueError(f"Unknown version: {version}")
    genotypes = np.asarray(config.genotypes, dtype=np.int8)
    if not np.all((genotypes == 0) | (genotypes == 1)):
        raise ValueError("The illustration requires complete binary genotypes")
    if genotypes.shape[0] != len(config.samples):
        raise ValueError("Each sample needs one genotype row")
    counts = genotypes.sum(axis=0)
    focal = config.focal_site - 1
    initial = set(np.flatnonzero(genotypes[:, focal]))
    threshold = len(initial) // 2
    haplotype = np.full(genotypes.shape[1], -1, dtype=np.int8)
    haplotype[focal] = 1
    marks = {sample: [] for sample in initial}
    removals = {}
    boundaries = []
    directions = (range(focal - 1, -1, -1), range(focal + 1, len(counts)))
    for sites in directions:
        sample_set = initial.copy()
        disagree = set()
        last = focal
        for site in sites:
            last = site
            alleles = genotypes[:, site]
            ones = sum((alleles[sample] == 1 for sample in sample_set))
            consensus = int(2 * ones >= len(sample_set))
            older = counts[site] > counts[focal]
            haplotype[site] = consensus if older else 0
            informative = older or counts[site] > ones
            if older or (version == "0.5.1" and informative):
                removed = {
                    sample
                    for sample in sample_set & disagree
                    if alleles[sample] != consensus
                }
                for sample in removed:
                    removals[sample] = site
                sample_set -= removed
            if older or (version == "0.5.1" and informative):
                disagree = {
                    sample for sample in sample_set if alleles[sample] != consensus
                }
                for sample in disagree:
                    marks[sample].append(site)
            if len(sample_set) <= threshold:
                break
        boundaries.append(last)
    return ExtensionTrace(haplotype, marks, removals, boundaries[0], boundaries[1] + 1)


def load_performance(path, config):
    """Retain each run and derive per-run resource metrics.

    Every size must have the same repetitions for both versions and all stages.
    Peak memory is a high-water mark and cannot be summed across stages.
    """
    data = pd.read_csv(path, comment="~", dtype={"version": str})
    keys = ["num_samples", "version", "rep", "command"]
    data = data.set_index(keys).sort_index()
    if not data.index.is_unique:
        raise ValueError(
            "Expected one measurement per sample size, version, rep and stage"
        )
    for size in data.index.get_level_values("num_samples").unique():
        rows = data.xs(size, level="num_samples")
        reps = rows.index.get_level_values("rep").unique()
        expected = pd.MultiIndex.from_product(
            [config.versions, reps, config.stages], names=keys[1:]
        )
        if not rows.index.equals(expected.sort_values()):
            raise ValueError(
                "Expected both versions and all three stages for every rep"
            )
    metrics = ["elapsed_time", "max_memory", "user_time", "sys_time", "num_threads"]
    values = data[metrics].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Performance metrics must be finite and nonnegative")
    if (data.elapsed_time == 0).any():
        raise ValueError("Elapsed time must be positive")
    data["elapsed_minutes"] = data.elapsed_time / config.seconds_per_minute
    data["memory_gib"] = data.max_memory / config.bytes_per_gib
    cpu_time = data.user_time + data.sys_time
    data["parallelism"] = cpu_time / data.elapsed_time
    return data


def summarise_leftmost_overshoot_bin(frame, by, rates, *, config, histogram_bins=100):
    """Return bin boundaries, width, counts and percentages by profile/version."""
    selected = frame.loc[frame[by].isin(rates)]
    values_mb = selected.overshoot.to_numpy() / config.bp_per_mb
    edges_mb = np.histogram_bin_edges(values_mb, bins=histogram_bins)
    left_bp = edges_mb[0] * config.bp_per_mb
    right_bp = edges_mb[1] * config.bp_per_mb
    rows = []
    for rate in rates:
        for version in config.versions:
            profile = selected.loc[
                (selected[by] == rate) & (selected.version == version)
            ]
            counts = np.histogram(profile.overshoot / config.bp_per_mb, edges_mb)[0]
            rows.append(
                {
                    by: rate,
                    "version": version,
                    "left_edge_bp": left_bp,
                    "right_edge_bp": right_bp,
                    "bin_width_bp": right_bp - left_bp,
                    "num_sides_in_leftmost_bin": int(counts[0]),
                    "total_num_sides": len(profile),
                    "percentage_in_leftmost_bin": 100 * int(counts[0]) / len(profile),
                }
            )
    return pd.DataFrame(rows).set_index([by, "version"])
