"""Final chapter 3 preparation and numerical analyses.

All focal sites retain their own comparisons. Both sides contribute to signed
overshoot summaries; only left rows contribute to whole-overlap counts.
Inputs are supplied explicitly by the root Snakemake workflow.
"""

from __future__ import annotations
import dataclasses
import json
import logging
import pathlib
import numpy as np
import pandas as pd
import polars as pl
import tskit
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


@dataclasses.dataclass
class OvershootHistogram:
    """Shared physical bins and profile counts used by histogram and diff plots."""

    bin_edges: np.ndarray
    counts: dict[tuple[str, float | str], np.ndarray]


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
    """Focal-site comparisons and the CSV exported by the validation exporter."""

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
    :func:`read_comparisons`. Focal-site IDs are local to a model.
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
    :func:`read_comparisons` and :func:`plot_ancestor_overlap_errors`.
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


def summarise_genotype_errors(data, *, transition_bins=40):
    """Pool exported diploid error counts and bin transitions by true DAF.

    the validation exporter compares each observed store to its error-free
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


def summarise_mispolarisation_overlap_errors(comparisons):
    """Pool :func:`mispolarisation_comparisons` by focal polarisation."""
    keys = ["group", "replicate", "version", "true_focal_site"]
    if comparisons.duplicated(keys).any():
        raise ValueError(
            "Expected one overlap comparison per group, focal site and version"
        )
    summary = _pool_overlap_errors(comparisons, ["group", "version"]).reset_index()
    return _overlap_error_rates(summary)


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


# Prepared tables are generated by named workflow rules, never notebook caches.
COMPARISON_COLUMNS = (
    "replicate",
    "version",
    "error_profile",
    "geno_multiplier",
    "space_per_switch",
    "phase_ser",
    "mispol_rate",
    "true_focal_site",
    "inferred_node",
    "true_node",
    "side",
    "allele_frequency",
    "inferred_span",
    "true_span",
    "overshoot",
    "focal_position",
    "focal_site_mispolarised",
    "num_errors",
    "overlap_span",
    "overlap_site_span",
    "num_should_be_0",
    "num_should_be_1",
)


def prepare_comparisons(aggregate_path, validation_path, output_path, config):
    """Join unflipped truth onto every focal-side comparison and write Parquet.

    A focal site shared by several profiles remains separate in each profile.
    The many-to-one join rejects ambiguous truth and missing variant positions.
    Streaming execution avoids materialising the multi-gigabyte CSV in pandas.
    """
    prefix = pathlib.Path(aggregate_path).name.removesuffix("-aggregated.csv")
    validation = _read_zarr_data(validation_path, prefix)
    truth = _true_focal_frequencies(validation)
    if truth.duplicated(["replicate", "focal_position"]).any():
        raise ValueError("Validation truth contains duplicate replicate/position keys")
    truth_query = pl.from_pandas(truth).lazy()
    aggregate = pl.scan_csv(aggregate_path).select(COMPARISON_COLUMNS)
    aggregate = aggregate.filter(pl.col("version").is_in(config.versions))
    joined = aggregate.join(
        truth_query, on=["replicate", "focal_position"], how="left", validate="m:1"
    )
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.sink_parquet(output)
    missing = (
        pl.scan_parquet(output)
        .select(pl.col("true_derived_count").null_count())
        .collect()
        .item()
    )
    if missing != 0:
        output.unlink()
        raise ValueError(f"{missing} aggregate rows lack exported truth frequencies")


def prepare_demography(aggregate_path, output_path, config):
    """Retain both error-free sides, including whole-overlap counts, for a model."""
    query = pl.scan_csv(aggregate_path)
    selected = query.filter(
        (pl.col("geno_multiplier") == 0)
        & (pl.col("phase_ser") == 0)
        & (pl.col("mispol_rate") == 0)
        & pl.col("version").is_in(config.versions)
    )
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [name for name in COMPARISON_COLUMNS if name != "space_per_switch"]
    selected.select(columns).sink_parquet(output)


def prepare_empirical_spans(paths, output_path, config):
    """Combine the two identified external benchmark exports without deduplication."""
    frames = [load_tgp_ancestor_spans(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    if set(combined.version) != set(config.versions):
        raise ValueError(
            "Empirical span exports do not contain both configured versions"
        )
    if combined.duplicated(["version", "inf_focal_site"]).any():
        raise ValueError("Duplicate empirical focal sites in span exports")
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(combined.to_dict(orient="list")).write_parquet(output)


def export_empirical_daf(store_path, output_path, config):
    """Export a sufficient retained DAF table from explicitly declared Zarr arrays."""
    frame = load_tgp_daf(store_path, config)
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(frame).write_parquet(output)


def read_table(path, predicate=None, columns=None):
    """Read only a plot or analysis's declared selection from a prepared table."""
    query = pl.scan_parquet(path)
    if predicate is not None:
        query = query.filter(predicate)
    if columns is not None:
        query = query.select(columns)
    result = query.collect(engine="streaming")
    return pd.DataFrame(result.to_dict(as_series=False))


def error_free_predicate():
    return (
        (pl.col("geno_multiplier") == 0)
        & (pl.col("phase_ser") == 0)
        & (pl.col("mispol_rate") == 0)
    )


def read_comparisons(path, predicate=None):
    """Read comparisons without losing focal-site multiplicity or side identity."""
    return read_table(path, predicate)


def read_chapter_data(path, validation_path, config, predicate=None):
    """Construct figure data from the rule-produced table and declared validation export."""
    frame = read_comparisons(path, predicate)
    spans = frame.loc[frame.side == "left"].copy()
    validation = pd.read_csv(
        validation_path,
        dtype={
            "record_type": "string",
            "data_prefix": "string",
            "mispolarised": "boolean",
        },
    )
    return ChapterData(frame, spans, validation, config)


def read_demographies(paths):
    """Read an explicit model-to-prepared-table mapping, preserving configuration order."""
    frames = []
    for model, path in paths.items():
        frame = read_table(path, error_free_predicate())
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def mispolarisation_comparisons(frame, rates):
    """Select one whole overlap per site and label its own polarisation state."""
    selected = frame.loc[
        (frame.side == "left")
        & (frame.geno_multiplier == 0)
        & (frame.phase_ser == 0)
        & frame.mispol_rate.isin([0, *rates])
    ].copy()
    focal_group = np.where(
        selected.focal_site_mispolarised, "mispolarised", "correctly_polarised"
    )
    selected["group"] = selected.mispol_rate.astype(str) + "_" + focal_group
    selected.loc[selected.mispol_rate == 0, "group"] = "none"
    return selected


@dataclasses.dataclass
class ExtensionTrees:
    """The explicit non-unary illustrative tree sequence and its labelled nodes."""

    tree_sequence: tskit.TreeSequence
    node_ids: dict[str, int]


def build_extension_trees(extension, settings):
    """Build the three configured genealogies and verify all illustrated genotypes."""
    tables = tskit.TableCollection(settings["sequence_length"])
    tables.time_units = "uncalibrated"
    node_ids = {}
    for name in extension.samples:
        node_ids[name] = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for name, age in settings["node_ages"].items():
        node_ids[name] = tables.nodes.add_row(time=age)
    for interval, topology in zip(
        settings["intervals"], settings["topologies"], strict=True
    ):
        left, right = interval
        for parent, children in topology.items():
            for child in children:
                tables.edges.add_row(
                    left, right, parent=node_ids[parent], child=node_ids[child]
                )
    for number, node_name, age in settings["mutations"]:
        position = settings["site_positions"][number - 1]
        site = tables.sites.add_row(position=position, ancestral_state="0")
        tables.mutations.add_row(
            site=site, node=node_ids[node_name], derived_state="1", time=age
        )
    tables.sort()
    ts = tables.tree_sequence()
    if not np.array_equal(ts.genotype_matrix().T, extension.genotypes):
        raise ValueError(
            "Illustrative trees do not reproduce the configured genotype matrix"
        )
    for tree in ts.trees():
        if any(tree.num_children(node) == 1 for node in tree.nodes()):
            raise ValueError("Illustrative trees contain a unary ancestor")
    return ExtensionTrees(ts, node_ids)


@dataclasses.dataclass
class PerformanceSummary:
    """Paired per-run stage ratios and totals (sum stages before taking ratios)."""

    stages: pd.DataFrame
    totals: pd.DataFrame


def summarise_performance(frame, config):
    """Average paired old/new ratios, rather than dividing group means."""
    old, new = config.versions
    paired = frame.elapsed_time.unstack("version")
    paired["speedup"] = paired[old] / paired[new]
    stages = paired.groupby(["num_samples", "command"]).mean().reset_index()
    seconds = frame.elapsed_time.groupby(["num_samples", "version", "rep"]).sum()
    runs = seconds.unstack("version")
    runs["speedup"] = runs[old] / runs[new]
    totals = runs.groupby("num_samples").mean().reset_index()
    for version in config.versions:
        totals[f"{version}_hours"] = totals[version] / (config.seconds_per_minute**2)
    return PerformanceSummary(stages, totals)


def histogram_count_changes(frame, by, rates, config, bins):
    """Relative count changes on common physical bins, including both focal sides.

    Return 100*(new-old)/old. A bin with zero old observations is undefined,
    not zero and not a percentage-point difference between normalised curves.
    """
    binned = _overshoot_histogram_counts(frame, by, rates, config, num_bins=bins)
    old, new = config.versions
    records = []
    for rate in rates:
        before = binned.counts[old, rate]
        after = binned.counts[new, rate]
        changes = np.full(before.shape, np.nan)
        np.divide(100 * (after - before), before, out=changes, where=before > 0)
        for index in range(len(before)):
            records.append(
                {
                    by: rate,
                    "left_mb": binned.bin_edges[index],
                    "right_mb": binned.bin_edges[index + 1],
                    "old_count": int(before[index]),
                    "new_count": int(after[index]),
                    "relative_increase_pct": changes[index],
                }
            )
    return pd.DataFrame(records)


@dataclasses.dataclass
class Claim:
    """A generated TeX quantity and an auditable description of its calculation."""

    macro: str
    value: float
    tex: str
    units: str
    calculation: str
    inputs: list[str]


def _claim(
    name,
    value,
    units,
    calculation,
    inputs,
    decimals,
    *,
    thousands=False,
    floor=False,
    ceiling=False,
):
    if not np.isfinite(value):
        raise ValueError(f"Cannot emit undefined thesis claim {name}: {calculation}")
    scale = 10**decimals
    rendered_value = value
    if floor:
        rendered_value = np.floor(value * scale) / scale
    if ceiling:
        rendered_value = np.ceil(value * scale) / scale
    format_spec = f",.{decimals}f" if thousands else f".{decimals}f"
    return Claim(
        "ChThree" + name,
        float(value),
        format(rendered_value, format_spec),
        units,
        calculation,
        list(inputs),
    )


def write_statistics(paths, settings, tex_path, claims_path, checks_path):
    """Calculate chapter claims and numerical checks from declared prepared inputs.

    The JSON mapping retains unrounded values, units, selectors, denominators,
    aggregation and input paths. TeX contains presentation only; prose stays in
    the thesis repository. Reference check tables are never read as inputs.
    """
    config = AnalysisConfig(**settings["analysis"])
    selection = settings["claims"]
    precision = settings["formatting"]
    old, new = config.versions
    claims = []
    checks = {}

    def add(name, value, units, calculation, inputs, digits="integer", **kwargs):
        claims.append(
            _claim(name, value, units, calculation, inputs, precision[digits], **kwargs)
        )

    extension = ExtensionConfig(**settings["extension"])
    illustration = build_extension_trees(extension, settings["extension_trees"])
    genotypes = np.asarray(extension.genotypes)
    carriers = int(genotypes[:, extension.focal_site - 1].sum())
    add(
        "ToyFocalCarriers",
        carriers,
        "samples",
        "Derived-carrier count at the configured focal site",
        ["thesis_config.yaml"],
    )
    claims[-1].tex = selection["number_words"][carriers]
    toy_rows = []
    for version, label in zip(extension.versions, ["Old", "New"], strict=True):
        trace = trace_ancestor(extension, version)
        sample = "b" if label == "Old" else "c"
        sample_index = extension.samples.index(sample)
        removal_site = trace.removals[sample_index] + 1
        add(
            "Toy" + label + "RemovalSite",
            removal_site,
            "one-based site",
            f"Algorithm replay {version}: removal of sample {sample}",
            ["thesis_config.yaml"],
        )
        toy_rows.append(
            {
                "version": version,
                "haplotype": trace.haplotype.tolist(),
                "start": trace.start,
                "end": trace.end,
                "num_trees": illustration.tree_sequence.num_trees,
            }
        )
    checks["toy_extension"] = pd.DataFrame(toy_rows)

    comparison_inputs = [paths["comparisons"], paths["validation"]]
    baseline = read_chapter_data(
        paths["comparisons"], paths["validation"], config, error_free_predicate()
    )
    focal_rows = []
    for version, frame in baseline.spans.groupby("version"):
        focal_rows.append(
            {
                "version": version,
                "focal_sites": len(frame),
                "true_ancestors": len(
                    frame.drop_duplicates(["replicate", "true_node"])
                ),
                "inferred_ancestors": len(
                    frame.drop_duplicates(["replicate", "inferred_node"])
                ),
            }
        )
    focal = pd.DataFrame(focal_rows).set_index("version")
    checks["focal_counts"] = focal.reset_index()
    for name, column in [
        ("FocalSites", "focal_sites"),
        ("TrueAncestors", "true_ancestors"),
        ("InferredAncestors", "inferred_ancestors"),
    ]:
        add(
            name,
            focal.loc[new, column],
            "count",
            f"Error-free {new}; sites separately, ancestor IDs unique within replicate",
            comparison_inputs,
            thousands=True,
        )
    rare = baseline.overshoots.loc[
        baseline.overshoots.allele_frequency < selection["maf_threshold"]
    ]
    checks["rare_overshoot_tail"] = (
        rare.groupby("version")
        .overshoot.quantile(selection["tail_probability"])
        .reset_index()
    )
    overlap = summarise_ancestor_overlap_errors(
        baseline.spans, config, cutoffs=selection["frequency_cutoffs"]
    )
    checks["ancestor_overlap_errors"] = overlap
    indexed = overlap.set_index(["version", "bin"])
    add(
        "RecentErrorFreePercent",
        100 * indexed.loc[(new, 0), "error_free_fraction"],
        "percent of focal-site overlaps",
        f"{new}, observed DAF < {selection['maf_threshold']}; zero-error count / all focal sites",
        comparison_inputs,
    )
    oldest_bin = len(selection["frequency_cutoffs"]) - 2
    add(
        "OldestErrorFreePercent",
        100 * indexed.loc[(new, oldest_bin), "error_free_fraction"],
        "percent of focal-site overlaps",
        f"{new}, observed DAF >= {selection['frequency_cutoffs'][-2]}; zero-error count / all focal sites",
        comparison_inputs,
    )
    mismatch = overlap.pivot(index="bin", columns="version", values="mismatch_fraction")
    frequency_change = 100 * (mismatch[new] / mismatch[old] - 1)
    add(
        "FrequencyMismatchIncreasePercent",
        frequency_change.max(),
        "relative percent",
        "Maximum across configured frequency bins of 100*(new pooled mismatch / old - 1); pooled errors / compared inference sites",
        comparison_inputs,
        "two",
        ceiling=True,
    )
    checks["whole_overlap_errors"] = _overlap_error_rates(
        _pool_overlap_errors(baseline.spans, ["version"]).reset_index()
    )
    empirical = read_table(paths["empirical_spans"])
    extent = empirical.focal_position.max() - empirical.focal_position.min()
    add(
        "EmpiricalExtentMb",
        extent / config.bp_per_mb,
        "Mb",
        "Maximum minus minimum empirical inference-site position across both versions",
        [paths["empirical_spans"]],
        "one",
    )
    checks["ancestor_span_curves"] = summarise_ancestor_span_curves(
        baseline, empirical, ancestor_error_bins=settings["span_curve_bins"]
    )

    demographies = read_demographies(paths["demographies"])
    demography_inputs = list(paths["demographies"].values())
    grouped = demographies.groupby(["model", "version"]).overshoot
    quartiles = grouped.quantile(
        [
            selection["tail_probability"],
            0.01,
            0.25,
            0.5,
            0.75,
            0.99,
            1 - selection["tail_probability"],
        ]
    ).unstack()
    checks["demography_quantiles_bp"] = quartiles.reset_index()
    for version, label in [(old, "Old"), (new, "New")]:
        medians = quartiles.xs(version, level="version")[0.5] / config.bp_per_kb
        add(
            "Demography" + label + "MedianMinKb",
            medians.min(),
            "kb",
            f"Minimum across demographic models of pooled focal-side median, {version}",
            demography_inputs,
            "one",
        )
        add(
            "Demography" + label + "MedianMaxKb",
            medians.max(),
            "kb",
            f"Maximum across demographic models of pooled focal-side median, {version}",
            demography_inputs,
            "one",
        )
    demography_errors = summarise_demography_overlap_errors(demographies)
    checks["demography_overlap_errors"] = demography_errors
    with_errors = 100 * (1 - demography_errors.error_free_fraction).mean()
    add(
        "DemographyApproxErrorPercent",
        round(with_errors / selection["approximation_percent_step"])
        * selection["approximation_percent_step"],
        "percent",
        "Mean per-model/version fraction of focal overlaps with errors, rounded to nearest configured percentage step for 'roughly'",
        demography_inputs,
    )
    demography_rates = demography_errors.pivot(
        index="model", columns="version", values="mismatch_fraction"
    )
    demographic_change = 100 * (demography_rates[new] / demography_rates[old] - 1)
    add(
        "DemographyMismatchIncreasePercent",
        demographic_change.max(),
        "relative percent",
        "Maximum across models of 100*(new pooled mismatch / old - 1)",
        demography_inputs,
        "three",
        ceiling=True,
    )
    zigzag = demography_rates.loc[selection["zigzag_model"]]
    others = demography_rates.drop(selection["zigzag_model"])
    zigzag_minimum = 100 * (others.rdiv(zigzag, axis="columns") - 1).min().min()
    add(
        "ZigzagMinimumIncreasePercent",
        zigzag_minimum,
        "relative percent lower bound",
        "Minimum across other models and both versions of 100*(Zigzag pooled mismatch / other - 1), floored for 'at least'",
        demography_inputs,
        floor=True,
    )
    checks["demography_nonnegative_fraction"] = (
        grouped.apply(lambda values: values.ge(0).mean())
        .rename("fraction")
        .reset_index()
    )
    nonnegative = grouped.apply(lambda values: values.ge(0).mean())
    bound = 100 * (1 - selection["central_lower_quantile"])
    if (100 * nonnegative >= bound).sum() <= len(nonnegative) / 2:
        raise ValueError(
            "The stated nonnegative-overshoot bound no longer holds in most model/version cases"
        )
    add(
        "DemographyNonnegativeBoundPercent",
        bound,
        "percent of focal-site sides",
        "Complement of the configured lower central quantile; verified that a strict majority of model/version cases meet this bound",
        demography_inputs + ["thesis_config.yaml"],
    )
    del demographies, baseline

    genotype_predicate = (pl.col("phase_ser") == 0) & (pl.col("mispol_rate") == 0)
    genotype_predicate &= pl.col("geno_multiplier").is_in(
        selection["genotype_multipliers"]
    )
    genotype = read_chapter_data(
        paths["comparisons"], paths["validation"], config, genotype_predicate
    )
    exported = genotype.zarr_data.loc[
        (genotype.zarr_data.record_type == "genotype")
        & (
            genotype.zarr_data.geno_multiplier
            == selection["baseline_genotype_multiplier"]
        )
    ].copy()
    minor_count = np.minimum(
        exported.derived_count, exported.num_alleles - exported.derived_count
    )
    common = exported.loc[
        minor_count > selection["maf_threshold"] * exported.num_alleles
    ]
    common_rate = common.changed.sum() / common.total.sum()
    add(
        "GenotypeCommonErrorPercent",
        100 * common_rate,
        "percent of diploid genotypes",
        "Sum changed / sum total, true MAF strictly above configured threshold, lambda=1",
        [paths["validation"]],
        "three",
    )
    add(
        "GenotypePublishedRelativePercent",
        100 * (common_rate / selection["published_genotype_error_rate"] - 1),
        "relative percent",
        "100*(MAF-matched simulated diploid error rate / cited published rate - 1); published reference is configuration, not a derived result",
        [paths["validation"], "thesis_config.yaml"],
        "one",
    )
    add(
        "GenotypeAllErrorPercent",
        100 * exported.changed.sum() / exported.total.sum(),
        "percent of diploid genotypes",
        "Lambda=1; sum changed / sum total; no frequency filter",
        [paths["validation"]],
        "three",
    )
    genotype_summary = summarise_genotype_errors(
        genotype, transition_bins=settings["transition_bins"]
    )
    checks["genotype_errors"] = genotype_summary.rates
    checks["genotype_transitions"] = genotype_summary.transitions
    summary = summarise_overshoots(
        genotype.overshoots,
        "geno_multiplier",
        selection["genotype_multipliers"],
        config,
    )
    checks["genotype_overshoots"] = summary
    indexed = summary.set_index(["geno_multiplier", "version"])
    moderate = selection["baseline_genotype_multiplier"]
    extreme = selection["extreme_genotype_multiplier"]
    for version, label in [(old, "Old"), (new, "New")]:
        add(
            "Genotype" + label + "MedianKb",
            indexed.loc[(moderate, version), "median_bp"] / config.bp_per_kb,
            "kb",
            f"Lambda={moderate}, {version}; median over both focal-site sides",
            comparison_inputs,
            "two",
        )
    add(
        "GenotypeModerateSeverePercent",
        100 * indexed.loc[(moderate, new), "fraction_below_threshold"],
        "percent of focal-site sides",
        f"Lambda={moderate}, {new}; overshoot strictly below {config.overshoot_summary_threshold_bp:g} bp / all sides",
        comparison_inputs,
        "one",
    )
    add(
        "GenotypeExtremeSeverePercent",
        100 * indexed.loc[(extreme, new), "fraction_below_threshold"],
        "percent of focal-site sides",
        f"Lambda={extreme}, {new}; overshoot strictly below threshold / all sides",
        comparison_inputs,
        "one",
    )
    extreme_rows = genotype.overshoots.loc[
        (genotype.overshoots.version == new)
        & (genotype.overshoots.geno_multiplier == extreme)
    ]
    add(
        "GenotypeExtremeNonpositivePercent",
        100 * extreme_rows.overshoot.le(0).mean(),
        "percent of focal-site sides",
        f"Lambda={extreme}, {new}; overshoot <= 0 / all sides",
        comparison_inputs,
    )
    histogram = histogram_count_changes(
        genotype.overshoots,
        "geno_multiplier",
        selection["genotype_multipliers"],
        config,
        selection["histogram_bins"],
    )
    checks["genotype_histogram_changes"] = histogram
    severe_bins = (
        histogram.left_mb + histogram.right_mb
    ) / 2 < config.overshoot_summary_threshold_bp / config.bp_per_mb
    add(
        "GenotypeUndershootBinIncreasePercent",
        histogram.loc[severe_bins, "relative_increase_pct"].max(),
        "relative percent of bin count",
        "Maximum 100*(new-old)/old over bins with midpoint below the severe threshold and lambda=0,1,2,5,10; common physical edges over all selected sides; zero-old bins undefined",
        comparison_inputs,
        "one",
        ceiling=True,
    )
    negative_medians = summary.loc[summary.median_bp < 0]
    first_negative = negative_medians.groupby("version").geno_multiplier.min().max()
    add(
        "GenotypeNegativeMedianMultiplier",
        first_negative,
        "genotype multiplier",
        "Smallest tested lambda with a negative median in both versions",
        comparison_inputs,
    )
    del genotype

    phase_predicate = (pl.col("geno_multiplier") == 0) & (pl.col("mispol_rate") == 0)
    phase = read_chapter_data(
        paths["comparisons"], paths["validation"], config, phase_predicate
    )
    summary = summarise_overshoots(
        phase.overshoots, "space_per_switch", config.phase_spacings, config
    )
    checks["phasing_overshoots"] = summary
    indexed = summary.set_index(["space_per_switch", "version"])
    moderate = selection["moderate_phase_spacing"]
    extreme = selection["extreme_phase_spacing"]
    phase_metadata = phase.phase_settings.set_index("space_per_switch")
    run_metadata = (
        pd.read_csv(paths["phase_runs"])[["space_per_switch", "phase_ser"]]
        .drop_duplicates()
        .set_index("space_per_switch")
    )
    for gap in [moderate, extreme]:
        if not np.isclose(
            phase_metadata.loc[gap, "phase_ser"],
            run_metadata.loc[gap, "phase_ser"],
            rtol=1e-10,
        ):
            raise ValueError("Phase-run and aggregate SER metadata disagree")
    for gap, label in [(moderate, "Moderate"), (extreme, "Extreme")]:
        value = phase_metadata.loc[gap, "phase_ser"]
        mantissa, exponent = f"{value:.{precision['two']}e}".split("e")
        claim = _claim(
            "Phase" + label + "SER",
            value,
            "switches / adjacent heterozygous-site pairs",
            f"Aggregate phase-only profile spacing {gap:g} bp; verified against the declared phase-run export",
            comparison_inputs + [paths["phase_runs"]],
            precision["two"],
        )
        claim.tex = rf"{mantissa} \times 10^{{{int(exponent)}}}"
        claims.append(claim)
    for version, label in [(old, "Old"), (new, "New")]:
        add(
            "Phase" + label + "MedianKb",
            indexed.loc[(moderate, version), "median_bp"] / config.bp_per_kb,
            "kb",
            f"Spacing {moderate:g} bp, {version}; pooled focal-side median",
            comparison_inputs,
            "two",
        )
    add(
        "PhaseExtremeSeverePercent",
        100 * indexed.loc[(extreme, new), "fraction_below_threshold"],
        "percent of focal-site sides",
        f"Spacing {extreme:g} bp, {new}; sides strictly below severe threshold / all sides",
        comparison_inputs,
        "two",
    )
    histogram = histogram_count_changes(
        phase.overshoots,
        "phase_ser",
        sorted(phase.overshoots.phase_ser.unique()),
        config,
        selection["histogram_bins"],
    )
    checks["phasing_histogram_changes"] = histogram
    add(
        "PhaseBinIncreasePercent",
        histogram.relative_increase_pct.max(),
        "relative percent of bin count",
        "Maximum across all common physical bins and phase-only SERs of 100*(new-old)/old; zero-old bins undefined",
        comparison_inputs,
        ceiling=True,
    )
    del phase

    combined_predicate = pl.col("error_profile").is_in(
        selection["combined_profiles"]
    ) & ~pl.col("focal_site_mispolarised")
    combined = read_comparisons(paths["comparisons"], combined_predicate)
    summary = summarise_combined_overlap_errors(combined)
    checks["combined_overlap_errors"] = summary
    indexed = summary.set_index(["error_profile", "version"])
    none, all_errors = selection["combined_profiles"]
    clean_row = indexed.loc[(none, new)]
    error_row = indexed.loc[(all_errors, new)]
    add(
        "CombinedErrorFreeDropPoints",
        100 * (clean_row.error_free_fraction - error_row.error_free_fraction),
        "percentage points",
        f"{new}, correctly polarised focal sites; zero-error fraction none minus all",
        comparison_inputs,
    )
    for row, label in [(clean_row, "Baseline"), (error_row, "Error")]:
        add(
            "Combined" + label + "MismatchPercent",
            100 * row.mismatch_fraction,
            "percent of compared inference sites",
            f"{new}, correctly polarised {label} profile; sum errors / sum overlap inference sites",
            comparison_inputs,
            "three",
        )
    fold = error_row.mismatch_fraction / clean_row.mismatch_fraction
    add(
        "CombinedMismatchFold",
        fold,
        "fold",
        "Combined / none pooled mismatch fraction in updated version, correctly polarised focal sites",
        comparison_inputs,
    )
    claims[-1].tex = selection["number_words"][int(round(fold))]
    maximum = combined.loc[
        (combined.version == new) & (combined.error_profile == all_errors), "overshoot"
    ].max()
    add(
        "CombinedMaximumOvershootMb",
        maximum / config.bp_per_mb,
        "Mb",
        "Maximum signed focal-side overshoot, correctly polarised combined profile, updated version",
        comparison_inputs,
        "two",
    )
    del combined

    # The chapter excludes the lambda=100 stress panel from its quoted regimes.
    all_query = pl.scan_parquet(paths["comparisons"]).filter(
        ~pl.col("focal_site_mispolarised")
        & pl.col("geno_multiplier").is_in(selection["genotype_multipliers"])
    )
    overlap_query = all_query.filter(pl.col("side") == "left")
    pooled = (
        overlap_query.group_by(["error_profile", "version"])
        .agg(pl.col("num_errors").sum(), pl.col("overlap_site_span").sum())
        .sort(["error_profile", "version"])
        .collect()
    )
    pooled = pd.DataFrame(pooled.to_dict(as_series=False))
    pooled["mismatch_fraction"] = pooled.num_errors / pooled.overlap_site_span
    checks["all_regime_mismatch"] = pooled
    rates = pooled.pivot(
        index="error_profile", columns="version", values="mismatch_fraction"
    )
    add(
        "MaximumRegimeMismatchIncreasePercent",
        (100 * (rates[new] / rates[old] - 1)).max(),
        "relative percent",
        "Maximum over quoted pure/combined regimes (lambda <=10), correctly polarised sites; whole-overlap pooled mismatch new/old - 1",
        comparison_inputs,
        "three",
    )
    focal_keys = [
        "error_profile",
        "geno_multiplier",
        "version",
        "replicate",
        "true_focal_site",
    ]
    sides = all_query.group_by(focal_keys).agg(
        (pl.col("overshoot") < config.overshoot_summary_threshold_bp)
        .any()
        .alias("severe")
    )
    severe = (
        sides.group_by(["error_profile", "geno_multiplier", "version"])
        .agg(pl.col("severe").sum().alias("severe_focal_sites"))
        .sort(["error_profile", "geno_multiplier", "version"])
        .collect()
    )
    severe = pd.DataFrame(severe.to_dict(as_series=False))
    checks["all_regime_severe_focal_sites"] = severe
    counts = severe.pivot(
        index=["error_profile", "geno_multiplier"],
        columns="version",
        values="severe_focal_sites",
    )
    changes = 100 * (counts[new] / counts[old] - 1)
    worst = changes.idxmax()
    add(
        "MaximumSevereFocalIncreasePercent",
        changes.loc[worst],
        "relative percent of focal-site count",
        "Maximum across quoted regimes, correctly polarised sites; focal site severe if at least one side is strictly below threshold; 100*(new count / old - 1)",
        comparison_inputs,
        "two",
    )
    add(
        "AdditionalSevereFocalSites",
        counts.loc[worst, new] - counts.loc[worst, old],
        "focal sites",
        "New minus old counts in the regime attaining the maximum relative severe-focal increase",
        comparison_inputs,
    )
    add(
        "WorstSevereGenotypeMultiplier",
        worst[1],
        "genotype multiplier",
        "Genotype multiplier of regime attaining maximum relative severe-focal count increase",
        comparison_inputs,
    )

    performance_config = PerformanceConfig(**settings["performance"])
    performance = load_performance(paths["performance"], performance_config)
    old_perf, new_perf = performance_config.versions
    benchmark = summarise_performance(performance, performance_config)
    checks["performance_stages"] = benchmark.stages
    checks["performance_totals"] = benchmark.totals
    stages = benchmark.stages.set_index(["num_samples", "command"])
    totals = benchmark.totals.set_index("num_samples")
    empirical_rows = performance.loc[
        performance.prefix.eq(settings["empirical_performance_prefix"])
    ]
    empirical_samples = (
        empirical_rows.index.get_level_values("num_samples").unique().item()
    )
    simulation_rows = performance.loc[
        ~performance.prefix.eq(settings["empirical_performance_prefix"])
    ]
    largest = simulation_rows.index.get_level_values("num_samples").max()
    for name, size, stage in [
        ("LargestAncestorSpeedup", largest, "match_ancestors"),
        ("LargestSampleSpeedup", largest, "match_samples"),
        ("EmpiricalSampleSpeedup", empirical_samples, "match_samples"),
        (
            "TenThousandSampleSpeedup",
            selection["comparison_haplotypes"],
            "match_samples",
        ),
    ]:
        add(
            name,
            stages.loc[(size, stage), "speedup"],
            "fold",
            f"Haplotypes={size}, {stage}; mean of paired per-repetition old/new wall-time ratios",
            [paths["performance"]],
            "one",
        )
    for name, size in [
        ("EmpiricalTotalSpeedup", empirical_samples),
        ("TenThousandTotalSpeedup", selection["comparison_haplotypes"]),
        ("LargestTotalSpeedup", largest),
    ]:
        add(
            name,
            totals.loc[size, "speedup"],
            "fold",
            f"Haplotypes={size}; sum three stages within each run, then mean paired old/new wall-time ratios",
            [paths["performance"]],
            "one",
        )
    for version, label in [(old_perf, "Old"), (new_perf, "New")]:
        add(
            "Largest" + label + "Hours",
            totals.loc[largest, f"{version}_hours"],
            "hours",
            f"Haplotypes={largest}, version={version}; mean run total wall seconds / 3600",
            [paths["performance"]],
        )
    add(
        "ApproxLargestSampleSpeedup",
        stages.loc[(largest, "match_samples"), "speedup"],
        "fold",
        "Largest simulated sample-matching speedup rounded for 'almost' in conclusion",
        [paths["performance"]],
    )
    site_counts = empirical_rows.num_sites.unique()
    if len(site_counts) != 1:
        raise ValueError(
            "Empirical benchmark retained site counts differ between stages/runs"
        )
    add(
        "EmpiricalRetainedSites",
        site_counts.item(),
        "inference sites",
        "Empirical benchmark num_sites, verified identical across stages, repetitions and versions",
        [paths["performance"]],
        thousands=True,
    )

    # Fixed experimental settings are clearly distinguished from derived claims.
    experiment = settings["experiment"]
    for name, key, divisor, units in [
        ("SimulationMb", "sequence_length_bp", config.bp_per_mb, "Mb"),
        ("SimulationIndividuals", "diploid_individuals", 1, "diploid individuals"),
        ("SimulationHaplotypes", "diploid_individuals", 0.5, "haplotypes"),
        (
            "EmpiricalIndividuals",
            "empirical_diploid_individuals",
            1,
            "diploid individuals",
        ),
        ("EmpiricalHaplotypes", "empirical_diploid_individuals", 0.5, "haplotypes"),
        ("SevereThresholdMb", "sequence_length_bp", 2 * config.bp_per_mb, "Mb"),
    ]:
        value = (
            abs(config.overshoot_summary_threshold_bp) / config.bp_per_mb
            if name == "SevereThresholdMb"
            else experiment[key] / divisor
        )
        add(
            name,
            value,
            units,
            "Fixed experimental setting from thesis configuration",
            ["thesis_config.yaml"],
            thousands=divisor < 2,
        )
    for path in [tex_path, claims_path, checks_path]:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by dphil-analysis: snakemake ch3_statistics. Do not edit.",
        "% Claim mappings, unrounded values and numerical checks accompany this file.",
    ]
    for claim in claims:
        lines.append(f"% {claim.units}; {claim.calculation}")
        lines.append("\\newcommand{\\" + claim.macro + "}{" + claim.tex + "}")
    pathlib.Path(tex_path).write_text("\n".join(lines) + "\n")
    pathlib.Path(claims_path).write_text(
        json.dumps([dataclasses.asdict(claim) for claim in claims], indent=2) + "\n"
    )
    serialised = {
        name: json.loads(frame.to_json(orient="records"))
        for name, frame in checks.items()
    }
    pathlib.Path(checks_path).write_text(
        json.dumps(serialised, indent=2, allow_nan=False) + "\n"
    )
