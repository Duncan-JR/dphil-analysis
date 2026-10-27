"""Chapter 3 figures from ancestor-evaluation CSVs and exported Zarr summaries.

The mirrored overshoot histograms adapt the layout in tsinfer-paper's
``lib/visualisation.py``. All data access and plotting here are independent of
that repository. See ``notebooks/ch3/figure_gen.py`` for the figure sequence.
"""

import dataclasses
import logging
import multiprocessing
import pathlib

import ch3_extract_data
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl
import scipy.ndimage as ndi
import zarr

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PlotConfig:
    """Settings shared by the chapter figures; lengths in input files are bp."""

    versions: tuple[str, str] = ("0.4.1", "0.5.1")
    version_colors: tuple[str, str] = ("#1a80bb", "#ea801c")
    genotype_scaling_cmap: str = "PRGn"
    genotype_scaling_reference_color: str = "gray"
    genotype_transition_cmap: str = "RdYlBu"
    genotype_color_positions: tuple[float, float] = (0.1, 0.9)
    true_color: str = "#b8b8b8"
    genotype_multipliers: tuple[float, ...] = (1, 2, 5, 10, 100)
    transition_multipliers: tuple[float, ...] = (1, 10)
    mispol_rates: tuple[float, ...] = (0, 0.01, 0.05, 0.1)
    phase_rates: tuple[float, ...] = (0, 0.001, 0.005, 0.01)
    transition_bins: int = 40
    curve_points: int = 256
    histogram_bins: int = 100
    daf_bins: int = 200
    daf_smoothing_bins: float = 1
    daf_size: tuple[float, float] = (5, 4.5)
    tgp_color: str = "red"
    tgp_legend_label: str = "1000 Genomes Project\nn=3202"
    tgp_region_start: int = 0
    tgp_region_end: int = 20_000_000
    tgp_count_prefix: str = "variant_1kgp_all_subset"
    tgp_variant_masks: tuple[str, ...] = (
        "variant_duplicate_position_mask",
        "variant_not_biallelic_mask",
        "variant_not_snps_mask",
        "variant_bad_ancestral_mask",
        "variant_no_ancestral_allele_mask",
    )
    dpi: int = 300
    bp_per_mb: float = 1e6
    box_size: tuple[float, float] = (7, 4.5)
    ancestor_panel_spacing: float = 0.18
    ancestor_count_margin: float = 0.35
    ancestor_count_fontsize: float = 8
    ancestor_count_padding: float = 2
    ancestor_error_size: tuple[float, float] = (7, 5.4)
    ancestor_error_height_ratios: tuple[float, float, float] = (1, 1, 0.6)
    ancestor_error_bins: int = 20
    ancestor_error_version: str = "0.5.1"
    ancestor_error_type_labels: tuple[str, str] = (
        "Should be 0 (0 → 1)", "Should be 1 (1 → 0)",
    )
    histogram_size: tuple[float, float] = (7, 4.5)
    overshoot_box_size: tuple[float, float] = (8.5, 3.5)
    overshoot_tail_size: tuple[float, float] = (8.5, 4.5)
    tail_legend_width_ratio: float = 0.75
    tail_legend_gap: float = 0.04
    tail_count_height_ratio: float = 0.28
    overshoot_cmaps: tuple[str, str] = ("Blues", "Oranges")
    version_line_styles: tuple[str, str] = ("-", "--")
    zero_color: str = "#555555"
    quantile_line_style: str = "--"
    quantile_line_width: float = 1.2
    preprocess_chunk_rows: int = 250000
    focal_keys: tuple[str, ...] = (
        "replicate", "version", "geno_multiplier", "phase_ser", "mispol_rate", "true_focal_site"
    )
    overshoot_box_width_ratio: float = 0.6
    jitter_seed: int = 42
    jitter_max_points: int = 1500
    jitter_width: float = 0.32
    jitter_alpha: float = 0.15
    jitter_size: float = 5
    mean_bar_group_width: float = 0.8
    mean_bar_y_margin: float = 0.05
    curve_size: tuple[float, float] = (6, 3.5)
    transition_size: tuple[float, float] = (7, 3.5)
    phase_size: tuple[float, float] = (8, 2.8)
    phase_row_spacing: float = 0.82
    phase_individuals: int = 5
    # GRCh38 chr1 p11.1 ends at 123.4 Mb (UCSC cytoBandIdeo).
    phase_right_mb: float = 123.4
    phase_colors: tuple[str, str] = ("#4477aa", "#ee9944")
    phase_ser: float = 0.005
    workers: int | None = None


@dataclasses.dataclass
class ChapterData:
    """Focal-site comparisons and the CSV exported by :mod:`ch3_extract_data`."""

    overshoots: pd.DataFrame
    spans: pd.DataFrame
    zarr_data: pd.DataFrame
    config: PlotConfig

    @property
    def phase_rates(self):
        """Available pure phase-error profiles, including the error-free profile."""
        profiles = self.zarr_data.loc[
            (self.zarr_data.record_type == "profile")
            & (self.zarr_data.geno_multiplier == 0) & (self.zarr_data.mispol_rate == 0)
        ]
        return tuple(sorted(profiles.phase_ser.unique()))


@dataclasses.dataclass
class GenotypeSummary:
    """Pooled rates and conditional transition rates returned by summarisation."""

    rates: pd.DataFrame
    transitions: pd.DataFrame


@dataclasses.dataclass
class FocalTask:
    frame: pd.DataFrame
    config: PlotConfig


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
    columns += ["inferred_node", "true_node", "allele_frequency",
                "focal_site_mispolarised", "true_derived_count", "true_allele_frequency"]
    overshoots = frame.loc[:, columns + ["side", "overshoot"]]
    # Each focal site has exactly one left row, even across CSV chunk boundaries.
    spans = frame.loc[frame.side == "left", columns + ["inferred_span", "true_span"]]
    return FocalFrames(overshoots, spans)


def _read_zarr_data(path, prefix):
    """Load a CSV from :func:`ch3_extract_data.extract_data` for this prefix."""
    frame = pd.read_csv(
        path, dtype={"record_type": "string", "data_prefix": "string", "mispolarised": "boolean"}
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
    return truth[["replicate", "focal_position", "true_derived_count", "true_allele_frequency"]]


def _focal_tasks(reader, truth, config):
    for frame in reader:
        frame = frame.merge(truth, on=["replicate", "focal_position"], how="left",
                            validate="many_to_one")
        if frame.true_derived_count.isna().any():
            raise ValueError("Focal positions are missing from the exported error-free variants")
        yield FocalTask(frame, config)


def _write_focal_tables(results, paths):
    """Stream prepared chunks to disk without retaining the full aggregate."""
    side_count = 0
    site_count = 0
    temporary = FocalPaths(paths.overshoots.with_suffix(".csv.tmp"),
                           paths.spans.with_suffix(".csv.tmp"))
    with temporary.overshoots.open("w") as overshoot_file, temporary.spans.open("w") as span_file:
        for index, result in enumerate(results):
            result.overshoots.to_csv(overshoot_file, index=False, header=index == 0)
            result.spans.to_csv(span_file, index=False, header=index == 0)
            side_count += len(result.overshoots)
            site_count += len(result.spans)
    temporary.overshoots.replace(paths.overshoots)
    temporary.spans.replace(paths.spans)
    logger.info("Wrote %s focal-site sides and %s focal-site spans", side_count, site_count)


def preprocess_data(aggregated_path, zarr_data_path, output_folder, config=None):
    """Prepare per-focal-site comparisons for :func:`load_data` and all plots.

    Each focal site retains its own true/inferred span comparison and both
    signed overshoots within each replicate, version and error profile. Sites
    sharing an ancestor remain separate, including sites with different
    polarisation states. True focal counts/frequencies come from the CSV
    generated by :mod:`ch3_extract_data`; no Zarr stores are opened.
    """
    if config is None:
        config = PlotConfig()
    aggregated_path = pathlib.Path(aggregated_path).expanduser()
    if not aggregated_path.name.endswith("-aggregated.csv"):
        raise ValueError("aggregated_path must end with -aggregated.csv")
    prefix = aggregated_path.name.removesuffix("-aggregated.csv")
    folder = pathlib.Path(output_folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    paths = FocalPaths(folder / f"{prefix}_focal_overshoots.csv",
                      folder / f"{prefix}_focal_spans.csv")
    columns = list(config.focal_keys)
    columns += ["inferred_node", "true_node", "side", "allele_frequency",
                "inferred_span", "true_span", "overshoot",
                "focal_position", "focal_site_mispolarised"]
    reader = pd.read_csv(aggregated_path, usecols=columns,
                         dtype={"version": str}, chunksize=config.preprocess_chunk_rows)
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
        config = PlotConfig()
    folder = pathlib.Path(data_folder).expanduser()
    overshoots = pd.read_csv(folder / f"{data_prefix}_focal_overshoots.csv")
    spans = pd.read_csv(folder / f"{data_prefix}_focal_spans.csv")
    overshoots = overshoots.loc[overshoots.version.isin(config.versions)].copy()
    spans = spans.loc[spans.version.isin(config.versions)].copy()
    if set(spans.version) != set(config.versions):
        raise ValueError(f"Ancestor CSV must contain versions {config.versions}")
    zarr_data = _read_zarr_data(folder / f"{data_prefix}_zarr_data.csv", data_prefix)
    return ChapterData(overshoots, spans, zarr_data, config)


def save_figure(fig, figure_folder, name, config):
    """Display and close a figure; save a PNG only when a folder is supplied."""
    if figure_folder is not None:
        folder = pathlib.Path(figure_folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", path)
    plt.show()
    plt.close(fig)


def _version_label(version):
    """Display the major/minor release while retaining full data identifiers."""
    release = ".".join(version.split(".")[:2])
    return f"tsinfer {release}"


def _no_errors(frame):
    return frame.loc[
        (frame.geno_multiplier == 0)
        & (frame.phase_ser == 0)
        & (frame.mispol_rate == 0)
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
        logger.warning("Rounded/tied quantiles yield %s of %s requested bins",
                       len(edges) - 1, num_bins)
    return edges


def _ancestor_boxplot(data, figure_folder, variable, num_bins, cutoffs, conf=None, alpha=None, figsize=None):
    config = data.config
    spans = _no_errors(data.spans)
    true_spans = spans.drop_duplicates(["replicate", "true_focal_site"])
    edges = _frequency_bin_edges(true_spans.allele_frequency, num_bins, cutoffs)
    if variable == "overshoot":
        if not 0 < conf < 0.5:
            raise ValueError("conf must be between 0 and 0.5")
        if alpha is None:
            alpha = config.jitter_alpha
        rng = np.random.default_rng(config.jitter_seed)
    num_bins = len(edges) - 1
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(edges[-1], np.inf)
    labels = [f"{left:g}–{right:g}" for left, right in zip(edges[:-1], edges[1:])]
    if figsize is None:
        figsize = config.box_size
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": config.ancestor_panel_spacing},
    )
    groups = []
    if variable == "span":
        groups.append(("True", config.true_color, true_spans, "true_span"))
        for version, color in zip(config.versions, config.version_colors):
            groups.append((_version_label(version), color,
                           spans.loc[spans.version == version], "inferred_span"))
    else:
        frame = _no_errors(data.overshoots)
        for version, color in zip(config.versions, config.version_colors):
            groups.append((_version_label(version), color,
                           frame.loc[frame.version == version], "overshoot"))
    width = 0.8 / len(groups)
    handles = []
    for index, (label, color, subset, column) in enumerate(groups):
        bin_ids = pd.cut(subset.allele_frequency, cut_edges, labels=False, right=False)
        values = []
        positions = []
        for bin_index in range(num_bins):
            selected = subset.loc[bin_ids == bin_index, column].to_numpy()
            if len(selected) > 0:
                values.append(selected / config.bp_per_mb)
                positions.append(bin_index - 0.4 + width * (index + 0.5))
        if variable == "span":
            axes[0].boxplot(
                values, positions=positions, widths=width * 0.85,
                patch_artist=True, manage_ticks=False, showfliers=False,
                boxprops={"facecolor": color}, medianprops={"color": "black"},
            )
        else:
            for selected, position in zip(values, positions):
                _draw_overshoot_group(axes[0], selected, position, width * 0.85,
                                      color, rng, config, conf, alpha)
        if variable == "span":
            handles.append(mpatches.Patch(color=color, label=label))
        if index == 0:
            focal_rows = subset.drop_duplicates(["replicate", "true_focal_site"])
            focal_bins = pd.cut(focal_rows.allele_frequency, cut_edges,
                                   labels=False, right=False)
            counts = focal_bins.value_counts().reindex(range(num_bins), fill_value=0)
            bars = axes[1].bar(np.arange(num_bins), counts, width=width * 0.85 * 2,
                               facecolor="white", edgecolor="black")
            if variable == "span":
                axes[1].bar_label(
                    bars, labels=[f"{count:,}" for count in counts],
                    fontsize=config.ancestor_count_fontsize,
                    padding=config.ancestor_count_padding,
                )
    if variable == "span":
        axes[0].legend(handles=handles, frameon=False, loc="lower center",
                       bbox_to_anchor=(0.5, 1.01), ncol=len(groups))
    else:
        fig.set_layout_engine("constrained")
        _overshoot_legends(fig, config, conf, alpha)
    if variable == "span":
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Ancestor span (Mb)")
    else:
        _overshoot_zero_line(axes[0])
        axes[0].set_ylabel("Overshoot (Mb)")
    axes[1].set_ylabel("Count")
    if variable == "span":
        axes[1].set_yscale("log")
        lower, upper = axes[1].get_ylim()
        count_range = counts.max() / lower
        upper = lower * count_range ** (1 / (1 - config.ancestor_count_margin))
        axes[1].set_ylim(lower, upper)
        axes[1].yaxis.set_major_locator(mticker.LogLocator(numticks=3))
        axes[1].yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    else:
        axes[1].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))
        axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[1].yaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].set_xticks(np.arange(num_bins), labels, rotation=25)
    axes[1].set_xlabel("Observed focal allele frequency")
    axes[0].set_xlim(-0.6, num_bins - 0.4)
    if variable == "span":
        fig.subplots_adjust(bottom=0.18, left=0.1, right=0.98, top=0.9)
    return save_figure(fig, figure_folder, f"ancestor_{variable}_boxplot", config)


def plot_ancestor_spans(data, figure_folder=None, num_bins=5, cutoffs=None, figsize=None):
    """True and inferred spans per focal site, binned by observed focal DAF."""
    return _ancestor_boxplot(data, figure_folder, "span", num_bins, cutoffs, figsize=figsize)


def plot_ancestor_overshoot(data, figure_folder=None, num_bins=5, cutoffs=None, conf=0.01, alpha=None, figsize=None):
    """Left/right overshoots per focal site, using the span figure DAF bins."""
    return _ancestor_boxplot(data, figure_folder, "overshoot", num_bins, cutoffs, conf, alpha, figsize=figsize)


def load_ancestor_overlap_errors(aggregated_path, config=None):
    """Read error-free whole-overlap comparisons from the aggregate CSV.

    ``num_errors`` and ``overlap_span`` describe the complete shared overlap,
    duplicated on the left/right rows. Keep only left rows, retaining one
    comparison per focal site and version as in :func:`plot_ancestor_overshoot`.
    The overlap is shared between the truth and both inferred versions.
    """
    if config is None:
        config = PlotConfig()
    query = pl.scan_csv(pathlib.Path(aggregated_path).expanduser())
    selected = query.filter(
        (pl.col("side") == "left")
        & (pl.col("geno_multiplier") == 0)
        & (pl.col("phase_ser") == 0)
        & (pl.col("mispol_rate") == 0)
        & pl.col("version").cast(pl.String).is_in(config.versions)
    )
    selected = selected.select(
        "replicate", "version", "true_focal_site", "allele_frequency",
        "num_errors", "overlap_span", "num_should_be_0", "num_should_be_1",
    )
    table = selected.collect(engine="streaming")
    return pd.DataFrame(table.to_dict(as_series=False))


def summarise_ancestor_overlap_errors(comparisons, config=None, num_bins=5, cutoffs=None):
    """Pool counts and overlapping bp within each version/frequency bin.

    Error burden is the ratio of sums, never the mean of per-overlap rates.
    Zero-error proportions and counts use one whole comparison per focal site.
    Empty bins have zero counts and undefined rates. See
    :func:`load_ancestor_overlap_errors` and :func:`plot_ancestor_overlap_errors`.
    """
    if config is None:
        config = PlotConfig()
    keys = ["replicate", "version", "true_focal_site"]
    if comparisons.duplicated(keys).any():
        raise ValueError("Expected one overlap comparison per focal site and version")
    frequencies = comparisons.drop_duplicates(["replicate", "true_focal_site"])
    edges = _frequency_bin_edges(frequencies.allele_frequency, num_bins, cutoffs)
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    frame = comparisons.copy()
    frame["bin"] = pd.cut(frame.allele_frequency, cut_edges, right=False, labels=False)
    frame["error_free"] = frame.num_errors == 0
    grouped = frame.groupby(["version", "bin"]).agg(
        num_ancestors=("true_focal_site", "size"),
        num_error_free=("error_free", "sum"),
        num_errors=("num_errors", "sum"),
        overlap_bp=("overlap_span", "sum"),
    )
    index = pd.MultiIndex.from_product(
        [config.versions, range(len(edges) - 1)], names=["version", "bin"],
    )
    summary = grouped.reindex(index, fill_value=0).reset_index()
    summary["bin_left"] = edges[summary.bin]
    summary["bin_right"] = edges[summary.bin + 1]
    ancestor_denominator = summary.num_ancestors.replace(0, np.nan)
    overlap_denominator = summary.overlap_bp.replace(0, np.nan)
    summary["error_free_fraction"] = summary.num_error_free / ancestor_denominator
    summary["error_burden"] = summary.num_errors / overlap_denominator
    return summary


def plot_ancestor_overlap_errors(summary, figure_folder=None, config=None, figsize=None):
    """Compact grouped bars of error-free fractions, pooled errors/bp and counts.

    Consume :func:`summarise_ancestor_overlap_errors`. Matched versions must
    have the same per-bin comparison counts, shown once in the bottom panel.
    """
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.ancestor_error_size
    count_table = summary.pivot(index="bin", columns="version", values="num_ancestors")
    reference_counts = count_table[config.versions[0]]
    if not count_table.eq(reference_counts, axis="index").all().all():
        raise ValueError("Expected matched focal-site counts between versions")
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": config.ancestor_error_height_ratios},
    )
    positions = np.arange(len(count_table))
    width = config.mean_bar_group_width / len(config.versions)
    for version_index, (version, color) in enumerate(zip(config.versions, config.version_colors)):
        selected = summary.loc[summary.version == version].sort_values("bin")
        offsets = positions - config.mean_bar_group_width / 2 + width * (version_index + 0.5)
        axes[0].bar(offsets, selected.error_free_fraction, width=width,
                    color=color, label=_version_label(version))
        axes[1].bar(offsets, selected.error_burden, width=width, color=color)
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    axes[0].set_ylabel("Error-free\nancestors")
    axes[1].set_ylabel("Pooled error burden\n(errors/overlap bp)")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_locator(mticker.LogLocator(numticks=3))
    axes[1].yaxis.set_minor_formatter(mticker.NullFormatter())
    axes[2].bar(positions, reference_counts, width=config.mean_bar_group_width,
                facecolor="white", edgecolor="black")
    axes[2].set_ylabel("Count")
    axes[2].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))
    axes[2].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    labels = [f"{row.bin_left:g}–{row.bin_right:g}" for row in selected.itertuples()]
    axes[2].set_xticks(positions, labels, rotation=25)
    axes[2].set_xlabel("Observed focal allele frequency")
    axes[2].set_xlim(-0.6, len(positions) - 0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(config.versions), frameon=False)
    return save_figure(fig, figure_folder, "ancestor_overlap_errors", config)


def summarise_ancestor_error_types(comparisons, config=None):
    """Split tsinfer 0.5 focal sites into fixed-width DAF bins across [0, 1].

    Plot rates at bin midpoints, including DAF 1 in the final bin. Empty bins
    have undefined rates. Pool each error type over total overlap bp, as in
    :func:`summarise_ancestor_overlap_errors`.

    The aggregate writer reverses its ``num_should_be_*`` names:
    ``num_should_be_0`` counts true 1 / inferred 0, and ``num_should_be_1``
    counts true 0 / inferred 1. Output names describe the actual true state.
    """
    if config is None:
        config = PlotConfig()
    frame = comparisons.loc[comparisons.version == config.ancestor_error_version].copy()
    keys = ["replicate", "true_focal_site"]
    if frame.duplicated(keys).any():
        raise ValueError("Expected one overlap comparison per focal site and version")
    num_bins = config.ancestor_error_bins
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
        should_be_0_errors=("num_should_be_1", "sum"),
        should_be_1_errors=("num_should_be_0", "sum"),
    )
    summary = summary.reindex(range(num_bins), fill_value=0).reset_index()
    summary["bin_left"] = edges[:-1]
    summary["bin_right"] = edges[1:]
    summary["daf"] = (edges[:-1] + edges[1:]) / 2
    denominator = summary.overlap_bp.replace(0, np.nan)
    summary["should_be_0_burden"] = summary.should_be_0_errors / denominator
    summary["should_be_1_burden"] = summary.should_be_1_errors / denominator
    return summary


def plot_ancestor_error_types(summary, figure_folder=None, config=None, figsize=None):
    """Two directional error-burden curves from :func:`summarise_ancestor_error_types`."""
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.curve_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    colors = plt.get_cmap(config.genotype_transition_cmap)(config.genotype_color_positions)
    columns = ["should_be_0_burden", "should_be_1_burden"]
    for column, label, color in zip(columns, config.ancestor_error_type_labels, colors):
        positive_rates = summary[column].where(summary[column] > 0)
        ax.plot(summary.daf, positive_rates, color=color, label=label)
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Derived allele frequency")
    ax.set_ylabel("Pooled error burden\n(errors/overlap bp)")
    ax.text(0.04, 0.96, _version_label(config.ancestor_error_version),
            transform=ax.transAxes, va="top")
    ax.legend(frameon=False)
    return save_figure(fig, figure_folder, "ancestor_overlap_error_types", config)


def summarise_genotype_errors(data):
    """Pool exported diploid error counts and bin transitions by true DAF.

    :mod:`ch3_extract_data` compares each observed store to its error-free
    replicate. Counts are exported by exact true allele count, allowing
    transition_bins to change locally without access to the original stores.
    """
    config = data.config
    frame = data.zarr_data.loc[data.zarr_data.record_type == "genotype"]
    replicates = set(data.spans.replicate)
    rate_rows = []
    transition_rows = []
    edges = np.linspace(0, 1, config.transition_bins + 1)
    for multiplier in config.genotype_multipliers:
        selected = frame.loc[frame.geno_multiplier == multiplier]
        if set(selected.replicate) != replicates:
            raise ValueError(f"Missing exported genotype counts for multiplier {multiplier}")
        changed = int(selected.changed.sum())
        total = int(selected.total.sum())
        rate_rows.append({"multiplier": multiplier, "changed": changed,
                          "total": total, "error_rate": changed / total})
        frequencies = selected.derived_count / selected.num_alleles
        for dosage, transition in [(0, "00 → 01/10"), (2, "11 → 01/10")]:
            denominator_weights = selected[f"homozygous_{dosage}"].to_numpy(dtype=np.int64)
            numerator_weights = selected[f"to_heterozygous_{dosage}"].to_numpy(dtype=np.int64)
            denominators = np.histogram(frequencies, edges, weights=denominator_weights)[0]
            numerators = np.histogram(frequencies, edges, weights=numerator_weights)[0]
            for bin_index in range(config.transition_bins):
                denominator = denominators[bin_index]
                numerator = numerators[bin_index]
                rate = numerator / denominator if denominator > 0 else np.nan
                transition_rows.append({
                    "multiplier": multiplier, "transition": transition,
                    "daf": (edges[bin_index] + edges[bin_index + 1]) / 2,
                    "changed": numerator, "total": denominator, "error_rate": rate,
                })
    return GenotypeSummary(pd.DataFrame(rate_rows), pd.DataFrame(transition_rows))


def plot_genotype_error_scaling(summary, figure_folder=None, config=None, figsize=None):
    """Observed diploid genotype error rate versus linear scaling from 1×."""
    if config is None:
        config = PlotConfig()
    rates = summary.rates.sort_values("multiplier")
    base = rates.loc[rates.multiplier == 1, "error_rate"].item()
    if figsize is None:
        figsize = config.curve_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    colors = plt.get_cmap(config.genotype_scaling_cmap)(config.genotype_color_positions)
    ax.plot(rates.multiplier, rates.error_rate, "o-", color=colors[0],
            label="Observed")
    linear_x = np.geomspace(rates.multiplier.min(), rates.multiplier.max(), config.curve_points)
    linear_y = base * linear_x
    ax.plot(linear_x, linear_y, "--",
            color=config.genotype_scaling_reference_color, label="Linear scaling from 1×")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(rates.multiplier, [f"{value:g}×" for value in rates.multiplier])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: f"{value * 100:.2g}%"))
    ax.set_xlabel("Error multiplier")
    ax.set_ylabel("Diploid genotype error rate")
    ax.legend(frameon=False)
    return save_figure(fig, figure_folder, "genotype_error_scaling", config)


def plot_genotype_error_transitions(summary, figure_folder=None, config=None, figsize=None):
    """Conditional homozygote-to-heterozygote rates by true DAF at 1× and 10×."""
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.transition_size
    fig, axes = plt.subplots(1, len(config.transition_multipliers),
                             figsize=figsize, sharey=True,
                             squeeze=False, layout="constrained")
    colors = plt.get_cmap(config.genotype_transition_cmap)(config.genotype_color_positions)
    for ax, multiplier in zip(axes[0], config.transition_multipliers):
        subset = summary.transitions.loc[summary.transitions.multiplier == multiplier]
        for (label, group), color in zip(subset.groupby("transition", sort=False),
                                         colors):
            positive_rates = group.error_rate.where(group.error_rate > 0)
            ax.plot(group.daf, positive_rates, color=color, label=label)
        ax.set_xlim(0, 1)
        ax.set_yscale("log")
        ax.set_xlabel("True derived allele frequency")
        ax.text(0.04, 0.96, f"{multiplier:g}×", transform=ax.transAxes, va="top")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: f"{value * 100:.2g}%"))
    selected = summary.transitions.loc[
        summary.transitions.multiplier.isin(config.transition_multipliers)
    ]
    positive_rates = selected.loc[selected.error_rate > 0, "error_rate"]
    axes[0, 0].set_ylim(positive_rates.min() / 1.2, positive_rates.max() * 1.2)
    axes[0, 0].set_ylabel("Homozygote-to-heterozygote error rate")
    axes[0, -1].legend(frameon=False)
    return save_figure(fig, figure_folder, "genotype_error_transitions", config)


def _overshoot_colors(rates, cmap_name):
    colors = plt.get_cmap(cmap_name)(np.linspace(0.25, 0.8, len(rates)))
    return dict(zip(rates, colors))


def _overshoot_histogram(data, frame, by, rates, legend_title,
                         name, figure_folder, reverse_order=False, figsize=None, legend_fontsize=None,
                         mirror_top_annotations=False):
    config = data.config
    frame = frame.loc[frame[by].isin(rates)].copy()
    values = frame.overshoot.to_numpy() / config.bp_per_mb
    # Shared edges preserve directly comparable counts between all profiles.
    bin_edges = np.histogram_bin_edges(values, bins=config.histogram_bins)
    counts = {}
    max_count = 0
    for version in config.versions:
        for rate in rates:
            selected = frame.loc[(frame.version == version) & (frame[by] == rate)]
            if selected.empty:
                raise ValueError(f"No overshoots for {version}, {by}={rate}")
            histogram = np.histogram(selected.overshoot / config.bp_per_mb, bin_edges)[0]
            counts[version, rate] = histogram
            max_count = max(max_count, histogram.max())
    if figsize is None:
        figsize = config.histogram_size
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"hspace": 0})
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for row, (ax, version, cmap_name) in enumerate(zip(axes, config.versions,
                                                      config.overshoot_cmaps)):
        palette = _overshoot_colors(rates, cmap_name)
        for rate in reversed(rates):
            histogram = counts[version, rate].copy()
            if reverse_order:
                histogram[centers >= 0] = 0
            ax.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), align="edge",
                   color=palette[rate], linewidth=0, bottom=0)
        if reverse_order:
            for rate in rates:
                histogram = counts[version, rate].copy()
                histogram[centers < 0] = 0
                ax.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), align="edge",
                       color=palette[rate], linewidth=0, bottom=0)
        ax.set_yscale("log")
        ax.set_ylim(0.8, max_count * 1.2)
        max_power = int(np.floor(np.log10(max_count)))
        ax.set_yticks([10 ** power for power in range(1, max_power + 1)])
        ax.axvline(0, color=config.zero_color, linestyle="--", linewidth=1)
        ax.set_ylabel("Count")
        if row == 1:
            ax.invert_yaxis()
        handles = []
        for rate in rates:
            label = f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}"
            handles.append(mpatches.Patch(color=palette[rate], label=label))
        legend_location = "upper right" if row == 0 else "lower right"
        version_x = 0.02
        version_alignment = "left"
        if row == 0 and mirror_top_annotations:
            legend_location = "upper left"
            version_x = 0.98
            version_alignment = "right"
        ax.legend(handles=handles, title=legend_title,
                  fontsize=legend_fontsize, title_fontsize=legend_fontsize,
                  loc=legend_location)
        ax.text(version_x, 0.96 if row == 0 else 0.04, _version_label(version),
                transform=ax.transAxes, va="top" if row == 0 else "bottom",
                ha=version_alignment,
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "pad": 4})
    axes[1].set_xlim(bin_edges[0], bin_edges[-1])
    axes[1].set_xlabel("Overshoot (Mb)")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.98)
    return save_figure(fig, figure_folder, f"{name}_hist", config)


def _draw_overshoot_group(ax, values, position, box_width, color, rng, config, conf, alpha):
    """White boxes, sampled central/all tail dots with one opacity, quantile caps."""
    ax.boxplot(
        [values], positions=[position], widths=box_width,
        patch_artist=True, manage_ticks=False, showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"}, capprops={"color": "black"},
        medianprops={"color": "black", "linewidth": 1.2}, zorder=3,
    )
    low, high = np.quantile(values, [conf, 1 - conf])
    outliers = (values < low) | (values > high)
    central = values[~outliers]
    sample_size = min(len(central), config.jitter_max_points)
    sampled = rng.choice(central, size=sample_size, replace=False)
    sampled = np.concatenate([sampled, values[outliers]])
    jitter = rng.uniform(-config.jitter_width / 2, config.jitter_width / 2, len(sampled))
    ax.scatter(position + jitter, sampled, s=config.jitter_size, alpha=alpha,
               color=color, edgecolors="none", rasterized=True, zorder=2)
    ax.hlines([low, high], position - config.jitter_width / 2,
              position + config.jitter_width / 2, color="black",
              linestyle=config.quantile_line_style, linewidth=config.quantile_line_width,
              zorder=5)


def _overshoot_zero_line(ax):
    """Match the border, above jitter (2) and below boxes (3)."""
    ax.axhline(0, color="black", linestyle="-",
               linewidth=ax.spines["left"].get_linewidth(), zorder=2.5)


def _overshoot_legends(fig, config, conf, alpha):
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", color=color,
                      alpha=alpha, markeredgecolor="none", label=_version_label(version))
        for version, color in zip(config.versions, config.version_colors)
    ]
    handles.append(mlines.Line2D([], [], color="black", linestyle=config.quantile_line_style,
                                 label=f"{conf * 100:g}% tail quantiles"))
    fig.legend(handles=handles, frameon=False, loc="outside right center")


def _overshoot_boxplot(data, frame, by, rates, label, name, figure_folder, conf, alpha, figsize=None):
    """Grouped boxes of focal-site sides; see `_draw_overshoot_group`."""
    if not 0 < conf < 0.5:
        raise ValueError("conf must be between 0 and 0.5")
    config = data.config
    if alpha is None:
        alpha = config.jitter_alpha
    rng = np.random.default_rng(config.jitter_seed)
    if figsize is None:
        figsize = config.overshoot_box_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    group_width = 0.8 / len(config.versions)
    box_width = config.jitter_width * config.overshoot_box_width_ratio
    for version_index, (version, color) in enumerate(zip(config.versions, config.version_colors)):
        for index, rate in enumerate(rates):
            subset = frame.loc[(frame.version == version) & (frame[by] == rate)]
            if subset.empty:
                raise ValueError(f"No overshoots for {version}, {by}={rate}")
            values = subset.overshoot.to_numpy() / config.bp_per_mb
            position = index - 0.4 + group_width * (version_index + 0.5)
            _draw_overshoot_group(ax, values, position, box_width, color, rng, config, conf, alpha)
    _overshoot_zero_line(ax)
    ax.set_ylabel("Overshoot (Mb)")
    ax.set_xlabel(label)
    labels = [f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}" for rate in rates]
    ax.set_xticks(range(len(rates)), labels)
    ax.set_xlim(-0.5, len(rates) - 0.5)
    ax.grid(axis="y", alpha=0.15)
    _overshoot_legends(fig, config, conf, alpha)
    return save_figure(fig, figure_folder, f"{name}_box", config)


@dataclasses.dataclass
class TailSteps:
    edges: np.ndarray
    fractions: np.ndarray


def _tail_steps(values):
    """Empirical fraction at or below each magnitude cutoff, counting focal sites once."""
    positions, counts = np.unique(values, return_counts=True)
    fractions = np.cumsum(counts) / len(values)
    return TailSteps(positions, fractions)


def _overshoot_tails(
    data, frame, by, rates, label, name, figure_folder, figsize=None, fraction_above=True,
    show_counts=False,
):
    """Overshoot magnitude fractions with independent panel x ranges.

    Each focal site contributes its larger negative and positive side magnitudes,
    both floored at zero. All focal sites supply the denominator, including
    those with no overshoot of that sign. With ``fraction_above=True``, count
    focal sites strictly above the magnitude cutoff; False counts focal sites
    at or below it (see `_tail_steps`).
    ``show_counts`` adds grouped bars of nonzero focal-site counts below each sign.
    """
    config = data.config
    if figsize is None:
        figsize = config.overshoot_tail_size
    rows = 2 if show_counts else 1
    height_ratios = [1, config.tail_count_height_ratio] if show_counts else [1]
    fig = plt.figure(figsize=figsize, layout="constrained")
    grid = fig.add_gridspec(
        rows, 3, width_ratios=[1, 1, config.tail_legend_width_ratio],
        height_ratios=height_ratios,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    legend_ax = fig.add_subplot(grid[:, 2])
    legend_ax.set_axis_off()
    version_handles = []
    site_counts = np.zeros((2, len(config.versions), len(rates)), dtype=int)
    min_fraction = 1
    keys = ["replicate", "true_focal_site"]
    for version_index, (version, cmap_name, line_style) in enumerate(zip(
        config.versions, config.overshoot_cmaps, config.version_line_styles
    )):
        palette = _overshoot_colors(rates, cmap_name)
        handles = []
        for rate_index, rate in enumerate(rates):
            subset = frame.loc[(frame.version == version) & (frame[by] == rate)]
            if subset.empty:
                raise ValueError(f"No overshoots for {version}, {by}={rate}")
            focal_sites = subset.groupby(keys).overshoot.agg(["min", "max"])
            num_sites = len(focal_sites)
            min_fraction = min(min_fraction, 1 / num_sites)
            negative = -focal_sites["min"].clip(upper=0).to_numpy() / config.bp_per_mb
            positive = focal_sites["max"].clip(lower=0).to_numpy() / config.bp_per_mb
            category = f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}"
            for sign_index, (ax, values) in enumerate(zip(axes, (negative, positive))):
                site_counts[sign_index, version_index, rate_index] = np.count_nonzero(values > 0)
                steps = _tail_steps(values)
                # Include zero so both magnitude curves start at the common origin.
                cutoffs = np.union1d(steps.edges, [0])
                indexes = np.searchsorted(steps.edges, cutoffs, side="right")
                fractions = np.concatenate(([0], steps.fractions))[indexes]
                if fraction_above:
                    fractions = 1 - fractions
                ax.step(cutoffs, fractions, where="post",
                        color=palette[rate], linestyle=line_style)
            handles.append(mlines.Line2D([], [], color=palette[rate], linestyle=line_style,
                                         label=category))
        version_handles.append(handles)
    for ax, sign in zip(axes, ("Negative", "Positive")):
        ax.set_xlabel(f"{sign} overshoot (Mb)")
        ax.set_yscale("log")
        ax.margins(x=0.02)
        ax.set_ylim(min_fraction * 0.8, 1)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda value, position: f"{value * 100:g}%"
        ))
        ax.grid(axis="y", alpha=0.15)
    if show_counts:
        positions = np.arange(len(rates))
        width = config.mean_bar_group_width / len(config.versions)
        categories = [f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}" for rate in rates]
        count_axes = [fig.add_subplot(grid[1, index]) for index in range(2)]
        for sign_index, ax in enumerate(count_axes):
            for version_index, (version, color) in enumerate(zip(config.versions, config.version_colors)):
                offsets = positions - config.mean_bar_group_width / 2 + width * (version_index + 0.5)
                ax.bar(offsets, site_counts[sign_index, version_index], width=width,
                       color=color, label=_version_label(version))
            ax.set_xticks(positions, categories)
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    direction = "above" if fraction_above else "below"
    separator = "\n" if show_counts else " "
    axes[0].set_ylabel(f"Fraction of focal sites{separator}{direction} cutoff")
    for index, (version, handles) in enumerate(zip(config.versions, version_handles)):
        offset = config.tail_legend_gap / 2
        anchor_y = 0.5 + offset if index == 0 else 0.5 - offset
        location = "lower center" if index == 0 else "upper center"
        legend = legend_ax.legend(
            handles=handles, title=f"{label} ({_version_label(version)})", loc=location,
            bbox_to_anchor=(0.5, anchor_y), borderaxespad=0, frameon=True,
        )
        if index == 0:
            legend_ax.add_artist(legend)
    if show_counts:
        legend_ax.add_artist(legend)
        handles, labels = count_axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, title="Focal site counts", frameon=False,
                         fontsize="small", loc="lower center")
    return save_figure(fig, figure_folder, f"{name}_tails", config)


def _genotype_overshoots(data):
    frame = data.overshoots
    return frame.loc[(frame.phase_ser == 0) & (frame.mispol_rate == 0)]


def _overshoot_mean(data, frame, by, rates, label, name, figure_folder, figsize=None):
    """Grouped mean signed overshoot by error profile, coloured by version.

    Each focal site contributes its mean left/right overshoot. Focal sites
    are pooled across replicates, as in `_overshoot_histogram`.
    """
    config = data.config
    frame = frame.loc[frame[by].isin(rates)]
    group_keys = ["version", by]
    focal_keys = group_keys + ["replicate", "true_focal_site"]
    focal_means = frame.groupby(focal_keys).overshoot.mean()
    means = focal_means.groupby(level=group_keys).mean()
    if figsize is None:
        figsize = config.curve_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    positions = np.arange(len(rates))
    width = config.mean_bar_group_width / len(config.versions)
    for index, (version, color) in enumerate(zip(config.versions, config.version_colors)):
        values = means.loc[version].reindex(rates)
        offsets = positions - config.mean_bar_group_width / 2 + width * (index + 0.5)
        ax.bar(offsets, values / config.bp_per_mb, width=width,
               color=color, label=_version_label(version))
    labels = [f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}" for rate in rates]
    ax.set_xticks(positions, labels)
    ax.set_xlabel(label)
    ax.set_ylabel("Mean overshoot (Mb)")
    ax.axhline(0, color="black", linestyle="--",
               linewidth=ax.spines["left"].get_linewidth())
    ax.use_sticky_edges = False
    ax.margins(y=config.mean_bar_y_margin)
    ax.legend(frameon=False)
    return save_figure(fig, figure_folder, f"{name}_mean", config)


def plot_genotype_overshoot_mean(data, figure_folder=None, figsize=None):
    """Grouped mean overshoot by genotype error; see `_overshoot_mean`."""
    return _overshoot_mean(
        data, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Error multiplier", "genotype_error_overshoot", figure_folder,
        figsize=figsize,
    )


def plot_genotype_overshoot_hist(data, figure_folder=None, figsize=None, legend_fontsize=None):
    """Mirrored histograms for pure genotype-error profiles."""
    return _overshoot_histogram(
        data, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Error multiplier", "genotype_error_overshoot", figure_folder,
        reverse_order=True,
        figsize=figsize, legend_fontsize=legend_fontsize,
    )


def plot_genotype_overshoot_box(data, figure_folder=None, conf=0.01, alpha=None, figsize=None):
    """Grouped boxplots for pure genotype-error profiles; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Error multiplier", "genotype_error_overshoot", figure_folder, conf, alpha,
        figsize=figsize,
    )


def plot_genotype_overshoot_tails(data, figure_folder=None, figsize=None, fraction_above=True):
    """Above/below magnitude cutoff fractions by genotype-error multiplier; see `_overshoot_tails`."""
    return _overshoot_tails(
        data, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Error multiplier", "genotype_error_overshoot", figure_folder,
        figsize=figsize, fraction_above=fraction_above,
    )


def load_tgp_daf(zarr_path, config=None):
    """Load polarised TGP SNP counts for the configured half-open region.

    Stored masks mark excluded variants. In addition to duplicate positions,
    non-biallelic sites and indels, exclude unknown ancestral states and states
    absent from the site's alleles. The all-sample counts already account for
    polarisation; ancestral plus derived counts give the called denominator.
    No allele-count filter is applied, retaining singletons as in the simulation.
    See :func:`plot_tgp_daf` for the matching spectrum.
    """
    if config is None:
        config = PlotConfig()
    group = zarr.open_group(zarr_path, mode="r")
    positions = group["variant_position"][:]
    start = np.searchsorted(positions, config.tgp_region_start)
    stop = np.searchsorted(positions, config.tgp_region_end)
    region = slice(start, stop)
    retained = np.ones(stop - start, dtype=bool)
    for name in config.tgp_variant_masks:
        excluded = group[name][region].astype(bool)
        retained &= ~excluded
    prefix = config.tgp_count_prefix
    derived = group[f"{prefix}_derived_count"][region][retained]
    ancestral = group[f"{prefix}_ancestral_count"][region][retained]
    return pd.DataFrame({
        "position": positions[region][retained],
        "derived_count": derived,
        "num_alleles": derived + ancestral,
    })


def _format_daf_axes(ax):
    """Common axes for the simulation and TGP spectra."""
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Derived allele frequency")
    ax.set_ylabel("Site count")


def _plot_daf_counts(ax, frequencies, config, weights=None, **kwargs):
    """Shared binning and smoothing for :func:`plot_mispolarisation_daf` and TGP."""
    edges = np.linspace(0, 1, config.daf_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = np.histogram(frequencies, edges, weights=weights)[0]
    smoothed_counts = ndi.gaussian_filter1d(
        counts.astype(float), config.daf_smoothing_bins, mode="reflect"
    )
    lines = ax.plot(centers, smoothed_counts, **kwargs)
    return lines[0]


def plot_tgp_daf(variants, figure_folder=None, config=None, figsize=None):
    """Plot :func:`load_tgp_daf` counts with the simulation's bins and styling."""
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.daf_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    _format_daf_axes(ax)
    frequencies = variants.derived_count.to_numpy() / variants.num_alleles.to_numpy()
    _plot_daf_counts(
        ax, frequencies, config, color=config.tgp_color, label=config.tgp_legend_label,
    )
    ax.legend(loc="upper right", frameon=False)
    return save_figure(fig, figure_folder, "tgp_daf", config)


def plot_mispolarisation_daf(data, figure_folder=None, figsize=None):
    """Overlay smoothed site counts per DAF bin after flipping mispolarised sites.

    The stored allele counts precede mispolarisation. The mask is True for
    correctly polarised sites; its complement selects the DAFs to reverse.
    Fixed-width histograms and Gaussian smoothing avoid KDE evaluation costs
    growing with the number of sites. Counts are pooled across replicates,
    including singletons (which the error generator does not mispolarise).
    See :func:`plot_tgp_daf` for the empirical comparison with the same bins,
    smoothing and line width.
    """
    config = data.config
    replicates = set(data.spans.replicate)
    frequencies = data.zarr_data.loc[data.zarr_data.record_type == "daf"]
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(config.mispol_rates)))
    if figsize is None:
        figsize = config.daf_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    _format_daf_axes(ax)
    for rate, color in zip(config.mispol_rates, colors):
        selected = frequencies.loc[frequencies.mispol_rate == rate]
        if set(selected.replicate) != replicates:
            raise ValueError(f"Missing exported allele-frequency counts for mispol rate {rate}")
        daf = selected.derived_count.to_numpy() / selected.num_alleles.to_numpy()
        mispolarised = selected.mispolarised.to_numpy(dtype=bool)
        daf[mispolarised] = 1 - daf[mispolarised]
        weights = selected.num_sites.to_numpy(dtype=np.int64)
        _plot_daf_counts(ax, daf, config, weights=weights, color=color, label=f"{rate:g}")
    ax.legend(title="Simulation:\nMispol. rate", loc="upper right", frameon=False)
    return save_figure(fig, figure_folder, "mispolarisation_daf", config)


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


def plot_mispolarisation_overshoot_hist(data, figure_folder=None, frequency="all", figsize=None, legend_fontsize=None):
    """Mirrored histograms at high (≥ 0.5), low (< 0.5), or all true DAFs."""
    return _overshoot_histogram(
        data, _mispolarisation_overshoots(data, frequency), "mispol_rate",
        data.config.mispol_rates, "Mispol. rate",
        f"mispolarisation_overshoot_{frequency}_frequency", figure_folder,
        figsize=figsize, legend_fontsize=legend_fontsize,
    )


def plot_mispolarisation_overshoot_box(data, figure_folder=None, frequency="all", conf=0.01, alpha=None, figsize=None):
    """Grouped mispolarisation boxplots; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data, _mispolarisation_overshoots(data, frequency), "mispol_rate",
        data.config.mispol_rates, "Mispolarisation rate",
        f"mispolarisation_overshoot_{frequency}_frequency", figure_folder, conf, alpha,
        figsize=figsize,
    )


def plot_mispolarisation_overshoot_tails(
    data, figure_folder=None, frequency="all", figsize=None, fraction_above=True, show_counts=False
):
    """Above/below magnitude cutoff fractions by mispolarisation rate; see `_overshoot_tails`."""
    return _overshoot_tails(
        data, _mispolarisation_overshoots(data, frequency), "mispol_rate",
        data.config.mispol_rates, "Mispol. rate",
        f"mispolarisation_overshoot_{frequency}_frequency", figure_folder,
        figsize=figsize, fraction_above=fraction_above, show_counts=show_counts,
    )


def plot_mispolarisation_low_count_overshoot_hist(data, figure_folder=None, max_derived_count=5, figsize=None, legend_fontsize=None):
    """Only mispolarised focal sites with true derived count at most the bound."""
    frame = _mispolarisation_overshoots(data, frequency="all")
    selected = frame.loc[
        frame.focal_site_mispolarised & (frame.true_derived_count <= max_derived_count)
    ]
    rates = [rate for rate in data.config.mispol_rates if rate > 0]
    return _overshoot_histogram(
        data, selected, "mispol_rate", rates, "Mispol. rate",
        f"mispolarisation_overshoot_derived_count_le_{max_derived_count}", figure_folder,
        figsize=figsize, legend_fontsize=legend_fontsize, mirror_top_annotations=True,
    )


def _phasing_overshoots(data):
    frame = data.overshoots
    return frame.loc[(frame.geno_multiplier == 0) & (frame.mispol_rate == 0)]


def plot_phasing_overshoot_mean(data, figure_folder=None, figsize=None):
    """Grouped mean overshoot by switch error rate; see `_overshoot_mean`."""
    return _overshoot_mean(
        data, _phasing_overshoots(data), "phase_ser", data.config.phase_rates,
        "Switch error rate", "phasing_error_overshoot", figure_folder,
        figsize=figsize,
    )


def plot_phasing_overshoot_hist(data, figure_folder=None, figsize=None, legend_fontsize=None):
    """Mirrored histograms for pure phase-switch profiles."""
    return _overshoot_histogram(
        data, _phasing_overshoots(data), "phase_ser", data.config.phase_rates,
        "Switch error rate", "phasing_error_overshoot", figure_folder, reverse_order=True,
        figsize=figsize, legend_fontsize=legend_fontsize,
    )


def plot_phasing_overshoot_box(data, figure_folder=None, conf=0.01, alpha=None, figsize=None):
    """Grouped phase-switch boxplots; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data, _phasing_overshoots(data), "phase_ser", data.config.phase_rates,
        "Switch error rate", "phasing_error_overshoot", figure_folder, conf, alpha,
        figsize=figsize,
    )


def plot_phasing_overshoot_tails(data, figure_folder=None, figsize=None, fraction_above=True):
    """Above/below magnitude cutoff fractions by switch error rate; see `_overshoot_tails`."""
    return _overshoot_tails(
        data, _phasing_overshoots(data), "phase_ser", data.config.phase_rates,
        "Switch error rate", "phasing_error_overshoot", figure_folder,
        figsize=figsize, fraction_above=fraction_above,
    )


def _phase_runs(positions, phase, length):
    """Return physical runs; a switch starts at its saved right-hand variant."""
    changes = np.flatnonzero(phase[1:] != phase[:-1]) + 1
    indices = np.concatenate(([0], changes))
    boundaries = np.concatenate(([0], positions[changes], [length]))
    widths = np.diff(boundaries)
    return boundaries[:-1], widths, phase[indices]


def _draw_phase_pair(ax, lefts, widths, states, y, config):
    for haplotype in range(2):
        for state in (False, True):
            selected = states == state
            runs = np.column_stack((lefts[selected], widths[selected]))
            color = config.phase_colors[haplotype ^ int(state)]
            ax.broken_barh(runs, (y + haplotype * 0.3, 0.24),
                           facecolors=color, edgecolors="none")


def plot_phase_switch_illustration(phasing_folder, data_prefix, figure_folder=None, config=None, figsize=None):
    """Aligned true/error haplotype pairs for the first individuals on chr1's p arm.

    Colours follow the saved true haplotype identity even at homozygous sites.
    """
    if config is None:
        config = PlotConfig()
    folder = pathlib.Path(phasing_folder).expanduser()
    stores = ch3_extract_data.discover_stores(folder / "zarr_vcfs", data_prefix)
    truth_store = ch3_extract_data.find_store(stores, 0)
    observed_store = ch3_extract_data.find_store(stores, 0, phase_ser=config.phase_ser)
    truth = zarr.open_group(truth_store.path, mode="r")
    observed = zarr.open_group(observed_store.path, mode="r")
    positions = observed["variant_position"][:]
    if not np.array_equal(positions, truth["variant_position"][:]):
        raise ValueError("Phasing illustration stores have different variant positions")
    if not np.array_equal(truth["sample_id"][:], observed["sample_id"][:]):
        raise ValueError("Phasing illustration stores have different sample order")
    phase = observed["call_genotype_phase"][:].astype(bool)
    length = observed["contig_length"][0]
    individuals = min(config.phase_individuals, phase.shape[1])
    right_mb = min(config.phase_right_mb, length / config.bp_per_mb)
    if figsize is None:
        figsize = config.phase_size
    fig, overview = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    for index in range(individuals):
        pair_lefts, pair_widths, pair_states = _phase_runs(positions, phase[:, index], length)
        y = (individuals - index - 1) * config.phase_row_spacing
        _draw_phase_pair(overview[0], np.array([0]), np.array([length / config.bp_per_mb]),
                         np.array([False]), y, config)
        _draw_phase_pair(overview[1], pair_lefts / config.bp_per_mb,
                         pair_widths / config.bp_per_mb, pair_states, y, config)
    for ax in overview:
        ax.set_xlim(0, right_mb)
        top = (individuals - 1) * config.phase_row_spacing + 0.95
        ax.set_ylim(-0.1, top)
        row_centers = np.arange(individuals) * config.phase_row_spacing + 0.27
        ax.set_yticks(row_centers,
                      [str(index + 1) for index in reversed(range(individuals))])
        ax.set_xlabel("Chromosome 1 p arm (Mb)")
    overview[0].set_ylabel("Individual (two haplotypes each)")
    for ax, label in zip(overview, ("Correct phase", f"SER = {config.phase_ser:g}")):
        ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top")
    handles = [mpatches.Patch(color=color, label=f"True haplotype {index + 1}")
               for index, color in enumerate(config.phase_colors)]
    fig.legend(handles=handles, loc="outside right center", frameon=False)
    return save_figure(fig, figure_folder, "phase_switch_illustration", config)
