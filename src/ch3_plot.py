"""Chapter 3 figures from ancestor-evaluation CSVs and exported Zarr summaries.

The mirrored overshoot histograms adapt the layout in tsinfer-paper's
``lib/visualisation.py``. All data access and plotting here are independent of
that repository. See ``notebooks/ch3/figure_gen.py`` for the figure sequence.
"""

import dataclasses
import logging
import multiprocessing
import pathlib

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
    phase_spacings: tuple[float, ...] = (0, 2e5, 2e6, 2e7)
    transition_bins: int = 40
    curve_points: int = 256
    histogram_bins: int = 100
    daf_bins: int = 200
    daf_smoothing_bins: float = 1
    daf_size: tuple[float, float] = (5, 4.5)
    combined_daf_size: tuple[float, float] = (7, 4.5)
    simulation_daf_legend_label: str = "Simulation (n=12,000)"
    combined_tgp_legend_label: str = "1000 Genomes\nProject (n=6404)"
    combined_daf_legend_x: float = 0.96
    combined_daf_simulation_legend_y: float = 0.97
    combined_daf_tgp_legend_y: float = 0.65
    combined_daf_legend_fontsize: float = 10
    tgp_color: str = "red"
    tgp_legend_label: str = "1000 Genomes Project\nn=3202"
    tgp_region_start: int = 0
    tgp_region_end: int = 20_000_000
    tgp_count_prefix: str = "variant_1kgp_all_subset"
    dpi: int = 300
    bp_per_mb: float = 1e6
    bp_per_kb: float = 1e3
    box_size: tuple[float, float] = (7, 4.5)
    ancestor_panel_spacing: float = 0.18
    ancestor_count_margin: float = 0.35
    ancestor_count_fontsize: float = 8
    ancestor_count_padding: float = 2
    ancestor_error_size: tuple[float, float] = (7, 5.4)
    combined_error_size: tuple[float, float] = (11, 5.6)
    combined_error_width_ratios: tuple[float, float] = (0.4, 0.6)
    combined_error_profiles: tuple[str, str] = ("none", "all")
    combined_error_labels: tuple[str, str] = ("No errors", "Combined errors")
    combined_error_legend_labels: tuple[str, str] = ("No errors", "Combined\nerrors")
    ancestor_error_height_ratios: tuple[float, float, float] = (1, 1, 0.6)
    ancestor_error_bins: int = 20
    ancestor_error_y_log: bool = True
    ancestor_error_log_subs: tuple[float, ...] = (1, 2, 5)
    ancestor_error_log_numticks: int = 6
    ancestor_error_hatch: str = "///"
    ancestor_error_legend_fontsize: float = 9
    ancestor_error_label_fontsize: float = 8
    ancestor_error_label_padding: float = 3
    ancestor_error_label_decimals: int = 3
    ancestor_error_change_decimals: int = 2
    ancestor_mismatch_fraction_label_margin: float = 0.2
    ancestor_error_count_margin: float = 0.4
    ancestor_error_version: str = "0.5.1"
    ancestor_error_type_labels: tuple[str, str] = (
        "True = 0, Observed = 1", "True = 1, Observed = 0",
    )
    mispolarisation_error_rates: tuple[float, ...] = (0.01, 0.1)
    mispolarisation_error_size: tuple[float, float] = (9.5, 6)
    mispolarisation_error_log_subs: tuple[float, ...] = (1,)
    mispolarisation_error_group_fontsize: float = 9
    mispolarisation_error_bracket_y: float = -0.5
    mispolarisation_error_bracket_depth: float = 0.06
    mispolarisation_error_rate_label_y: float = -0.62
    mispolarisation_mismatch_rate: float = 0.1
    mispolarisation_mismatch_bins: int = 10
    mispolarisation_mismatch_size: tuple[float, float] = (10, 5)
    mispolarisation_mismatch_height_ratios: tuple[float, float] = (4, 1.2)
    mispolarisation_mismatch_groups: tuple[str, ...] = ("none", "mispolarised", "correctly_polarised")
    mispolarisation_mismatch_labels: tuple[str, ...] = ("No error", "Mispol focal sites", "Non-mispol focal sites")
    mispolarisation_mismatch_colors: tuple[str, ...] = ("#777777", "#ea801c", "#6a3d9a")
    mispolarisation_mismatch_y_margin: float = 0.15
    histogram_size: tuple[float, float] = (7, 4.5)
    histogram_line_width: float = 1.2
    mispolarised_focal_size: tuple[float, float] = (9, 4.5)
    mispolarised_focal_width_ratios: tuple[float, float] = (0.58, 0.42)
    mispolarised_focal_dac_title: str = "True DAC spectrum\n(mispol. focal sites)"
    mispolarised_focal_dac_marker_max_count: int = 10
    mispolarised_focal_dac_marker: str = "o"
    mispolarised_focal_dac_marker_size: float = 4
    mispolarised_focal_dac_y_margin: float = 0.15
    histogram_labelled_legend_anchors: tuple[tuple[float, float], tuple[float, float]] = (
        (0.04, 1), (0.98, 1),
    )
    overshoot_diff_size: tuple[float, float] = (8.5, 5)
    overshoot_diff_linthresh_pct: float = 1
    overshoot_diff_line_width: float = 1.2
    overshoot_diff_legend_fontsize: float = 10
    overshoot_diff_panel_labels: tuple[str, str] = ("A", "B")
    overshoot_diff_panel_label_position: tuple[float, float] = (0.02, 0.96)
    overshoot_diff_panel_label_fontsize: float = 12
    overshoot_diff_cmap: str = "viridis"
    overshoot_diff_color_range: tuple[float, float] = (0, 0.8)
    overshoot_diff_panel_gap: float = 0.15
    overshoot_summary_size: tuple[float, float] = (8.5, 5)
    overshoot_summary_threshold_bp: float = -10e6
    overshoot_summary_interval_width: float = 5
    overshoot_summary_marker_size: float = 5
    overshoot_summary_bar_height: float = 0.65
    overshoot_summary_fraction_margin: float = 0.1
    overshoot_summary_fraction_ticks: int = 4
    overshoot_summary_percent_decimals: int = 1
    overshoot_summary_label_fontsize: float = 9
    overshoot_scatter_size: tuple[float, float] = (5, 4.5)
    overshoot_scatter_marker_size: float = 5
    overshoot_scatter_label_fontsize: float = 10
    overshoot_scatter_change_fontsize: float = 10
    overshoot_scatter_label_padding: float = 3
    overshoot_scatter_guide_depth: float = 0.05
    phase_scatter_guide_depth: float = 0.2
    phase_ser_decimals: int = 2
    phase_scatter_change_offset: tuple[float, float] = (-5, 4)
    overshoot_scatter_vertical_dash: tuple[int, tuple[int, int]] = (0, (2, 3))
    overshoot_scatter_zero_dash: tuple[int, tuple[int, int]] = (0, (6, 3))
    overshoot_scatter_change_offset: tuple[float, float] = (-5, 4)
    genotype_scatter_error_change_offset: tuple[float, float] = (-5, -4)
    genotype_scatter_connector_width: float = 0.8
    overshoot_scatter_x_margin: float = 0.15
    overshoot_scatter_y_margin: float = 0.18
    overshoot_scatter_tick_decimals: int = 2
    overshoot_scatter_correlation_decimals: int = 3
    overshoot_box_size: tuple[float, float] = (8.5, 3.5)
    demography_quantile_size: tuple[float, float] = (8, 4.5)
    demography_quantile_probabilities: tuple[float, ...] = (0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999)
    demography_quantile_center_limits_kb: tuple[float, float] = (-25, 250)
    demography_quantile_tail_limits_mb: tuple[float, float] = (-20, 20)
    demography_quantile_interval_width: float = 5
    demography_quantile_marker_size: float = 5
    demography_quantile_cap_size: float = 8
    demography_quantile_label_fontsize: float = 9
    overshoot_tail_size: tuple[float, float] = (8.5, 4.5)
    tail_legend_width_ratio: float = 0.75
    tail_legend_gap: float = 0.04
    tail_count_height_ratio: float = 0.28
    overshoot_cmaps: tuple[str, str] = ("Blues", "Oranges")
    version_line_styles: tuple[str, str] = ("-", "--")
    zero_color: str = "#555555"
    quantile_line_style: str = "--"
    quantile_secondary_line_style: str = "-."
    quantile_secondary_color: str = "#555555"
    quantile_line_width: float = 1.2
    preprocess_chunk_rows: int = 250000
    focal_keys: tuple[str, ...] = (
        "replicate", "version", "geno_multiplier", "space_per_switch", "phase_ser", "mispol_rate", "true_focal_site"
    )
    overshoot_box_width_ratio: float = 0.6
    overshoot_linthresh_bp: float = 1000
    jitter_seed: int = 42
    jitter_max_points: int | None = 1500
    jitter_sample_interval_mb: tuple[float, float] = (-1, 1)
    jitter_width: float = 0.32
    jitter_alpha: float = 0.15
    jitter_size: float = 5
    mean_bar_group_width: float = 0.8
    mean_bar_y_margin: float = 0.05
    mean_change_decimals: int = 1
    mean_change_fontsize: float = 8
    mean_change_padding: float = 4
    curve_size: tuple[float, float] = (6, 3.5)
    span_curve_size: tuple[float, float] = (4.5, 4.8)
    span_curve_height_ratios: tuple[float, float] = (3, 0.6)
    span_curve_true_color: str = "#8b0000"
    span_curve_title_fontsize: float = 10
    span_curve_legend_fontsize: float = 11
    span_curve_limit_factor: float = 1.2
    span_curve_line_width: float = 1.8
    transition_size: tuple[float, float] = (7, 3.5)
    phase_size: tuple[float, float] = (8.5, 4.5)
    phase_row_spacing: float = 0.82
    phase_group_gap: float = 0.45
    phase_label_fontsize: float = 9
    phase_right_mb: float = 20
    phase_colors: tuple[str, str] = ("#4477aa", "#ee9944")
    workers: int | None = None


@dataclasses.dataclass
class ChapterData:
    """Focal-site comparisons and the CSV exported by :mod:`ch3_extract_data`."""

    overshoots: pd.DataFrame
    spans: pd.DataFrame
    zarr_data: pd.DataFrame
    config: PlotConfig

    @property
    def phase_settings(self):
        """Phasing-only spacing and SER metadata from the current aggregate."""
        profiles = self.spans.loc[
            (self.spans.geno_multiplier == 0) & (self.spans.mispol_rate == 0)
        ]
        return profiles[["space_per_switch", "phase_ser"]].drop_duplicates().sort_values("space_per_switch")

    @property
    def phase_spacings(self):
        return tuple(self.phase_settings.space_per_switch)


@dataclasses.dataclass
class GenotypeSummary:
    """Pooled rates and conditional transition rates returned by summarisation."""

    rates: pd.DataFrame
    transitions: pd.DataFrame


@dataclasses.dataclass
class OvershootHistogram:
    """Shared physical bins and profile counts used by histogram and diff plots."""

    bin_edges: np.ndarray
    counts: dict[tuple[str, float | str], np.ndarray]


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


def _ancestor_boxplot(
    data, figure_folder, variable, num_bins, cutoffs, conf=None, alpha=None,
    figsize=None, y_log=False,
):
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
        if y_log:
            axes[0].set_yscale("log")
        axes[0].set_ylabel("Ancestor span (Mb)")
    else:
        if y_log:
            _symlog_overshoot_axis(axes[0], config)
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
    if variable == "overshoot":
        for ax, label in zip(axes, config.overshoot_diff_panel_labels):
            ax.text(*config.overshoot_diff_panel_label_position, label,
                    transform=ax.transAxes, va="top", fontweight="bold",
                    fontsize=config.overshoot_diff_panel_label_fontsize,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    if variable == "span":
        fig.subplots_adjust(bottom=0.18, left=0.1, right=0.98, top=0.9)
    return save_figure(fig, figure_folder, f"ancestor_{variable}_boxplot", config)


def plot_ancestor_spans(
    data, figure_folder=None, num_bins=5, cutoffs=None, figsize=None, y_log=True,
):
    """True and inferred spans per focal site, binned by observed focal DAF."""
    return _ancestor_boxplot(
        data, figure_folder, "span", num_bins, cutoffs,
        figsize=figsize, y_log=y_log,
    )


def plot_ancestor_overshoot(
    data, figure_folder=None, num_bins=5, cutoffs=None, conf=0.01, alpha=None,
    figsize=None, y_log=False,
):
    """Left/right overshoots in DAF bins; ``y_log=True`` uses a symlog axis."""
    return _ancestor_boxplot(
        data, figure_folder, "overshoot", num_bins, cutoffs, conf, alpha,
        figsize=figsize, y_log=y_log,
    )


def load_tgp_ancestor_spans(path):
    """Load full focal-site spans from the frequency-timed 1KGP benchmark.

    ``scripts/ch3_anc_perf/extract_spans.py`` exports one row per focal site.
    The benchmark uses the default frequency proxy for ancestor times, so
    ``inferred_time`` is the focal derived allele frequency, not an age.
    """
    frame = pd.read_csv(
        pathlib.Path(path).expanduser(), dtype={"version": str},
        usecols=["version", "inf_focal_site", "focal_position", "inferred_time", "inferred_span"],
    )
    return frame.rename(columns={"inferred_time": "allele_frequency"})


def summarise_ancestor_span_curves(data, tgp_spans):
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
    num_bins = config.ancestor_error_bins
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
        [(dataset, version, bin_id) for dataset, version in groups
         for bin_id in range(num_bins)],
        names=["dataset", "version", "bin"],
    )
    summary = grouped.reindex(index).reset_index()
    summary["num_focal_sites"] = summary.num_focal_sites.fillna(0).astype(int)
    summary["mean_span_mb"] = summary.mean_span_bp / config.bp_per_mb
    summary["bin_left"] = edges[summary.bin]
    summary["bin_right"] = edges[summary.bin + 1]
    summary["daf"] = (summary.bin_left + summary.bin_right) / 2
    return summary


def plot_ancestor_span_curves(
    summary, dataset, title=None, figure_folder=None, config=None, figsize=None, y_log=True,
):
    """One dataset's mean-span curves, with a short focal-count panel below.

    Consume :func:`summarise_ancestor_span_curves`; ``dataset`` must be
    ``Simulation`` or ``1000 Genomes Project``. Counts must match between
    versions (and truth). All lines are solid, with dark red simulation truth.
    Use the complete summary for both calls to share mean-span axis limits.
    """
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.span_curve_size
    if title is None:
        title = dataset
    filenames = {
        "Simulation": "ancestor_span_frequency_simulation",
        "1000 Genomes Project": "ancestor_span_frequency_tgp",
    }
    if dataset not in filenames:
        raise ValueError("dataset must be Simulation or 1000 Genomes Project")
    selected = summary.loc[summary.dataset == dataset]
    count_table = selected.pivot(index="bin", columns="version", values="num_focal_sites")
    reference = count_table.iloc[:, 0]
    if not count_table.eq(reference, axis="index").all().all():
        raise ValueError(f"Expected matched focal-site counts within {dataset}")
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": config.span_curve_height_ratios},
    )
    versions = list(config.versions)
    colors = list(config.version_colors)
    if dataset == "Simulation":
        versions.append("True")
        colors.append(config.span_curve_true_color)
    for version, color in zip(versions, colors):
        rows = selected.loc[selected.version == version].sort_values("bin")
        label = "True" if version == "True" else _version_label(version)
        axes[0].plot(
            rows.daf, rows.mean_span_mb, color=color, linestyle="-",
            linewidth=config.span_curve_line_width, label=label,
        )
    bins = selected.drop_duplicates("bin").sort_values("bin")
    widths = bins.bin_right - bins.bin_left
    axes[1].bar(bins.daf, reference, width=widths, facecolor="white", edgecolor="black")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Count")
    axes[1].yaxis.set_major_locator(mticker.LogLocator(numticks=2))
    axes[1].yaxis.set_minor_formatter(mticker.NullFormatter())
    if y_log:
        axes[0].set_yscale("log")
        positive_means = summary.loc[summary.mean_span_mb > 0, "mean_span_mb"]
        lower = positive_means.min() / config.span_curve_limit_factor
        upper = positive_means.max() * config.span_curve_limit_factor
        axes[0].set_ylim(lower, upper)
    axes[0].set_title(title, fontsize=config.span_curve_title_fontsize)
    axes[0].set_ylabel("Mean ancestor span (Mb)")
    axes[0].legend(
        frameon=False, loc="upper left",
        fontsize=config.span_curve_legend_fontsize,
    )
    axes[1].set_xlabel("Derived focal allele frequency")
    axes[1].set_xlim(0, 1)
    return save_figure(fig, figure_folder, filenames[dataset], config)

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
        "num_errors", "overlap_span", "overlap_site_span", "num_should_be_0", "num_should_be_1",
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
        config = PlotConfig()
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
            "replicate", "version", "true_focal_site", "side", "overshoot",
            "num_errors", "overlap_span", "overlap_site_span", "num_should_be_0", "num_should_be_1",
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
    summary["should_be_0_mismatch_fraction"] = summary.should_be_0_errors / overlap_denominator
    summary["should_be_1_mismatch_fraction"] = summary.should_be_1_errors / overlap_denominator
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


def summarise_ancestor_overlap_errors(comparisons, config=None, num_bins=5, cutoffs=None):
    """Pool counts and compared inference sites within each version/frequency bin.

    Mismatch fraction is the ratio of sums, never the mean of per-overlap rates.
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
    grouped = _pool_overlap_errors(frame, ["version", "bin"])
    index = pd.MultiIndex.from_product(
        [config.versions, range(len(edges) - 1)], names=["version", "bin"],
    )
    summary = grouped.reindex(index, fill_value=0).reset_index()
    summary["bin_left"] = edges[summary.bin]
    summary["bin_right"] = edges[summary.bin + 1]
    return _overlap_error_rates(summary)


def plot_ancestor_overlap_errors(
    summary, figure_folder=None, config=None, figsize=None, *,
    group_by="bin", group_labels=None, show_counts=True,
    name="ancestor_overlap_errors", show_mismatch_change=False,
    panel_labels=(), group_xlabel="Demographic model", figure=None, split_error_types=True,
):
    """Grouped fractions and counts, with pooled mismatch fraction stacked by direction.

    Consume :func:`summarise_ancestor_overlap_errors`. Matched versions must
    have the same per-group comparison counts. ``show_counts=False`` omits
    the bottom count panel. For categorical groups, pass their column as
    ``group_by`` and an ordered identifier-to-label mapping as
    ``group_labels``; no frequency binning is performed by this plot.
    ``show_mismatch_change`` labels the second version's pooled mismatch fraction
    with its signed percentage change relative to the first version.
    ``panel_labels`` adds letters in axes coordinates. ``figure`` can supply
    a SubFigure for composition; its parent handles saving and display.
    Set ``config.ancestor_error_y_log=False`` for a zero-based linear
    mismatch-fraction axis, allowing the stacked error directions to be compared by height.
    ``split_error_types=False`` shows total mismatch fraction as solid bars and omits
    the error-direction legend.
    """
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.ancestor_error_size
    count_table = summary.pivot(index=group_by, columns="version", values="num_ancestors")
    if group_labels is not None:
        count_table = count_table.reindex(group_labels)
    reference_counts = count_table[config.versions[0]]
    if not count_table.eq(reference_counts, axis="index").all().all():
        raise ValueError("Expected matched focal-site counts between versions")
    panel_count = 3 if show_counts else 2
    height_ratios = config.ancestor_error_height_ratios[:panel_count]
    fig = plt.figure(figsize=figsize, layout="constrained") if figure is None else figure
    axes = fig.subplots(
        panel_count, 1, sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    positions = np.arange(len(count_table))
    width = config.mean_bar_group_width / len(config.versions)
    for version_index, (version, color) in enumerate(zip(config.versions, config.version_colors)):
        selected = summary.loc[summary.version == version].set_index(group_by)
        selected = selected.reindex(count_table.index)
        offsets = positions - config.mean_bar_group_width / 2 + width * (version_index + 0.5)
        bars = axes[0].bar(
            offsets, selected.error_free_fraction, width=width,
            color=color, label=_version_label(version),
        )
        proportion_labels = [
            f"{value:.{config.ancestor_error_label_decimals}f}"
            if np.isfinite(value) else ""
            for value in selected.error_free_fraction
        ]
        axes[0].bar_label(
            bars, labels=proportion_labels, color=color,
            fontsize=config.ancestor_error_label_fontsize,
            padding=config.ancestor_error_label_padding,
        )
        if split_error_types:
            lower_fraction = selected.should_be_1_mismatch_fraction
            upper_fraction = selected.should_be_0_mismatch_fraction
            axes[1].bar(offsets, lower_fraction, width=width,
                        facecolor=color, edgecolor=color)
            fraction_bars = axes[1].bar(
                offsets, upper_fraction, width=width,
                bottom=lower_fraction, facecolor="white",
                edgecolor=color, hatch=config.ancestor_error_hatch,
            )
        else:
            total_fraction = selected.mismatch_fraction
            fraction_bars = axes[1].bar(offsets, total_fraction, width=width,
                                      facecolor=color, edgecolor=color)
        if show_mismatch_change and version_index == 1:
            reference_fraction = summary.loc[summary.version == config.versions[0]]
            reference_fraction = reference_fraction.set_index(group_by).mismatch_fraction
            reference_fraction = reference_fraction.reindex(count_table.index)
            denominator = reference_fraction.where(reference_fraction > 0)
            change = (selected.mismatch_fraction / denominator - 1) * 100
            change_labels = [
                f"{value:+.{config.ancestor_error_change_decimals}f}%"
                if np.isfinite(value) else ""
                for value in change
            ]
            axes[1].bar_label(
                fraction_bars, labels=change_labels, color=color,
                fontsize=config.ancestor_error_label_fontsize,
                padding=config.ancestor_error_label_padding,
            )
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    axes[0].set_ylabel("Proportion of\nerror-free focal sites")
    axes[1].set_ylabel("Pooled mismatch fraction\n(mismatches/site)")
    if show_mismatch_change:
        axes[1].margins(y=config.ancestor_mismatch_fraction_label_margin)
    if config.ancestor_error_y_log:
        axes[1].set_yscale("log")
        axes[1].yaxis.set_major_locator(
            mticker.LogLocator(subs=config.ancestor_error_log_subs,
                               numticks=config.ancestor_error_log_numticks)
        )
        axes[1].yaxis.set_major_formatter(
            mticker.LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf))
        )
        axes[1].yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        axes[1].set_ylim(bottom=0)
        axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    if show_counts:
        axes[2].bar(positions, reference_counts, width=config.mean_bar_group_width,
                    facecolor="white", edgecolor="black")
        axes[2].set_ylabel("Count")
        axes[2].margins(y=config.ancestor_error_count_margin)
        axes[2].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))
        axes[2].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    if group_labels is None:
        labels = [f"{row.bin_left:g}–{row.bin_right:g}" for row in selected.itertuples()]
        axes[-1].set_xlabel("Observed focal allele frequency")
        rotation = 25
    else:
        labels = [group_labels[group] for group in count_table.index]
        axes[-1].set_xlabel(group_xlabel)
        rotation = 0
    axes[-1].set_xticks(positions, labels, rotation=rotation)
    axes[-1].set_xlim(-0.6, len(positions) - 0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(config.versions), frameon=False)
    if split_error_types:
        error_handles = [
            mpatches.Patch(facecolor="white", edgecolor="black", hatch=config.ancestor_error_hatch),
            mpatches.Patch(facecolor="black", edgecolor="black"),
        ]
        axes[1].legend(error_handles, config.ancestor_error_type_labels, frameon=False,
                       loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2,
                       fontsize=config.ancestor_error_legend_fontsize)
    for ax, label in zip(axes, panel_labels):
        ax.text(*config.overshoot_diff_panel_label_position, label,
                transform=ax.transAxes, va="top", fontweight="bold",
                fontsize=config.overshoot_diff_panel_label_fontsize,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    if figure is not None:
        return fig
    return save_figure(fig, figure_folder, name, config)


def summarise_ancestor_error_types(comparisons, config=None):
    """Split tsinfer 0.5 focal sites into fixed-width DAF bins across [0, 1].

    Plot rates at bin midpoints, including DAF 1 in the final bin. Empty bins
    have undefined rates. Pool each error type over compared inference sites, as in
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


def plot_ancestor_error_types(summary, figure_folder=None, config=None, figsize=None):
    """Two directional mismatch-fraction curves from :func:`summarise_ancestor_error_types`."""
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.curve_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    colors = plt.get_cmap(config.genotype_transition_cmap)(config.genotype_color_positions)
    columns = ["should_be_0_mismatch_fraction", "should_be_1_mismatch_fraction"]
    for column, label, color in zip(columns, config.ancestor_error_type_labels, colors):
        positive_rates = summary[column].where(summary[column] > 0)
        ax.plot(summary.daf, positive_rates, color=color, label=label)
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Derived focal allele frequency")
    ax.set_ylabel("Pooled mismatch fraction\n(mismatches/site)")
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


def _profile_label(rate, by, config):
    if by == "error_profile":
        labels = dict(zip(config.combined_error_profiles, config.combined_error_labels))
        return labels[rate]
    if by == "phase_ser":
        if rate == 0:
            return "No errors"
        mantissa, exponent = f"{rate:.{config.phase_ser_decimals}e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    if by == "space_per_switch":
        return "No errors" if rate == 0 else f"{rate / config.bp_per_mb:g}"
    return f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}"


def _overshoot_colors(rates, cmap_name, color_range=None):
    if color_range is None:
        color_range = (0.25, 0.8)
    colors = plt.get_cmap(cmap_name)(np.linspace(*color_range, len(rates)))
    return dict(zip(rates, colors))


def _overshoot_histogram_counts(frame, by, rates, config, num_bins=None):
    """Bin both versions on common Mb edges; used by histogram and diff plots."""
    frame = frame.loc[frame[by].isin(rates)]
    values = frame.overshoot.to_numpy() / config.bp_per_mb
    # Shared edges preserve directly comparable counts between all profiles.
    if num_bins is None:
        num_bins = config.histogram_bins
    bin_edges = np.histogram_bin_edges(values, bins=num_bins)
    counts = {}
    for version in config.versions:
        for rate in rates:
            selected = frame.loc[(frame.version == version) & (frame[by] == rate)]
            if selected.empty:
                raise ValueError(f"No overshoots for {version}, {by}={rate}")
            histogram = np.histogram(selected.overshoot / config.bp_per_mb, bin_edges)[0]
            counts[version, rate] = histogram
    return OvershootHistogram(bin_edges, counts)


def _overshoot_histogram(data, frame, by, rates, legend_title,
                         name, figure_folder, reverse_order=False, figsize=None, legend_fontsize=None,
                         mirrored=True, panel_labels=False, lines=False, figure=None):
    """Draw shared-bin counts; mirrored=False gives separated upward panels."""
    config = data.config
    binned = _overshoot_histogram_counts(frame, by, rates, config)
    bin_edges = binned.bin_edges
    counts = binned.counts
    max_count = max(histogram.max() for histogram in counts.values())
    if figsize is None:
        figsize = config.histogram_size
    panel_gap = 0 if mirrored else config.overshoot_diff_panel_gap
    if figure is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={"hspace": panel_gap})
    else:
        fig = figure
        axes = fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": panel_gap})
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for row, (ax, version, cmap_name) in enumerate(zip(axes, config.versions,
                                                      config.overshoot_cmaps)):
        palette = _overshoot_colors(rates, cmap_name)
        for rate in rates if lines else reversed(rates):
            histogram = counts[version, rate].copy()
            if reverse_order:
                histogram[centers >= 0] = 0
            if lines:
                ax.stairs(histogram, bin_edges, baseline=None, color=palette[rate],
                          linewidth=config.histogram_line_width)
            else:
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
        if row == 1 and mirrored:
            ax.invert_yaxis()
        handles = []
        for rate in rates:
            label = _profile_label(rate, by, config)
            if lines:
                handle = mlines.Line2D([], [], color=palette[rate], label=label,
                                      linewidth=config.histogram_line_width)
            else:
                handle = mpatches.Patch(color=palette[rate], label=label)
            handles.append(handle)
        annotations_at_top = row == 0 or not mirrored
        legend_location = "upper right" if annotations_at_top else "lower right"
        legend_anchor = None
        title = legend_title
        if panel_labels:
            legend_location = "upper left" if row == 0 else "upper right"
            legend_anchor = config.histogram_labelled_legend_anchors[row]
            version_heading = _version_label(version).replace(" ", r"\ ")
            title = rf"$\mathbf{{{version_heading}}}$" + f"\n{legend_title}"
        ax.legend(handles=handles, title=title,
                  fontsize=legend_fontsize, title_fontsize=legend_fontsize,
                  loc=legend_location, bbox_to_anchor=legend_anchor)
        if not panel_labels:
            ax.text(0.02, 0.96 if annotations_at_top else 0.04, _version_label(version),
                    transform=ax.transAxes, va="top" if annotations_at_top else "bottom",
                    ha="left", fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "#cccccc", "pad": 4})
        if panel_labels:
            ax.text(*config.overshoot_diff_panel_label_position,
                    config.overshoot_diff_panel_labels[row], transform=ax.transAxes,
                    ha="left", va="top", fontweight="bold",
                    fontsize=config.overshoot_diff_panel_label_fontsize)
    axes[1].set_xlim(bin_edges[0], bin_edges[-1])
    axes[1].set_xlabel("Overshoot (Mb)")
    if figure is not None:
        return fig
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.98)
    return save_figure(fig, figure_folder, f"{name}_hist", config)


def _overshoot_diff(config, frame, by, rates, legend_title, name, figure_folder,
                    figsize=None, legend_fontsize=None, num_bins=None, figure=None):
    """Step counts above signed percentage count changes on identical Mb bins.

    Reuse :func:`_overshoot_histogram_counts`; ``num_bins=None`` uses the
    histogram configuration, while an explicit value overrides the bin count.
    Percentage changes are 100 * (new count - old count) / old count; bins
    with zero old count are undefined and left as gaps. The count axis is log;
    the change axis is symmetric log to retain negative changes and zero.
    """
    binned = _overshoot_histogram_counts(frame, by, rates, config, num_bins=num_bins)
    old_version, new_version = config.versions
    if figsize is None:
        figsize = config.overshoot_diff_size
    fig = plt.figure(figsize=figsize, layout="constrained") if figure is None else figure
    axes = fig.subplots(2, 1, sharex=True)
    palette = _overshoot_colors(rates, config.overshoot_diff_cmap,
                                config.overshoot_diff_color_range)
    if figure is None:
        fig.get_layout_engine().set(hspace=config.overshoot_diff_panel_gap)
    for rate in rates:
        old_counts = binned.counts[old_version, rate]
        new_counts = binned.counts[new_version, rate]
        changes = np.full(old_counts.shape, np.nan)
        np.divide(100 * (new_counts - old_counts), old_counts,
                  out=changes, where=old_counts > 0)
        label = _profile_label(rate, by, config)
        visible_counts = np.where(old_counts > 0, old_counts, np.nan)
        axes[0].stairs(visible_counts, binned.bin_edges, baseline=None,
                       color=palette[rate], label=label,
                       linewidth=config.overshoot_diff_line_width)
        axes[1].stairs(changes, binned.bin_edges, baseline=None,
                       color=palette[rate], label=label,
                       linewidth=config.overshoot_diff_line_width)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"Observation count\n({_version_label(old_version)})")
    axes[1].set_yscale("symlog", linthresh=config.overshoot_diff_linthresh_pct)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    axes[1].set_ylabel("Percentage increase in\ntsinfer 0.5")
    axes[1].axhline(0, color=config.zero_color, linestyle="--", linewidth=1)
    for ax, panel_label in zip(axes, config.overshoot_diff_panel_labels):
        ax.axvline(0, color=config.zero_color, linestyle="--", linewidth=1)
        ax.text(*config.overshoot_diff_panel_label_position, panel_label,
                transform=ax.transAxes, ha="left", va="top", fontweight="bold",
                fontsize=config.overshoot_diff_panel_label_fontsize,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    if legend_fontsize is None:
        legend_fontsize = config.overshoot_diff_legend_fontsize
    handles = []
    labels = []
    for rate in rates:
        handles.append(mlines.Line2D([], [], color=palette[rate]))
        labels.append(_profile_label(rate, by, config))
    wrapped_title = legend_title.replace(" ", "\n", 1)
    if by == "error_profile":
        wrapped_title = legend_title
        labels = config.combined_error_legend_labels
    legend = fig.legend(handles, labels, title=wrapped_title, loc="outside right center",
                        fontsize=legend_fontsize, title_fontsize=legend_fontsize, frameon=False)
    if by == "error_profile":
        legend.get_title().set_fontweight("bold")
    axes[1].set_xlim(binned.bin_edges[0], binned.bin_edges[-1])
    axes[1].set_xlabel("Overshoot (Mb)")
    if figure is not None:
        return fig
    return save_figure(fig, figure_folder, f"{name}_diff", config)


def load_combined_error_comparisons(aggregated_path, config=None):
    """Read correctly polarised focal sites in the named `none` and `all` profiles.

    Profile names come from the aggregate rather than inferred combinations
    of error parameters. Whole-overlap metrics occur on both side rows; use
    :func:`summarise_combined_overlap_errors` to count each overlap once.
    """
    if config is None:
        config = PlotConfig()
    query = pl.scan_csv(pathlib.Path(aggregated_path).expanduser())
    selected = query.filter(
        pl.col("error_profile").is_in(config.combined_error_profiles)
        & pl.col("version").cast(pl.String).is_in(config.versions)
        & ~pl.col("focal_site_mispolarised")
    )
    columns = [
        "error_profile", "replicate", "version", "true_focal_site", "side",
        "geno_multiplier", "space_per_switch", "phase_ser", "mispol_rate",
        "focal_site_mispolarised",
        "num_errors", "overlap_span", "overlap_site_span", "num_should_be_0", "num_should_be_1",
        "overshoot",
    ]
    table = selected.select(columns).collect(engine="streaming")
    return pd.DataFrame(table.to_dict(as_series=False))


def summarise_combined_overlap_errors(comparisons):
    """Pool whole-overlap counts and bp separately for each profile/version."""
    frame = comparisons.loc[comparisons.side == "left"]
    keys = ["error_profile", "replicate", "version", "true_focal_site"]
    if frame.duplicated(keys).any():
        raise ValueError("Expected one overlap comparison per profile, focal site and version")
    summary = _pool_overlap_errors(frame, ["error_profile", "version"]).reset_index()
    return _overlap_error_rates(summary)


def plot_combined_overlap_errors(
    summary, figure_folder=None, config=None, figsize=None, *, figure=None,
):
    """Two accuracy panels, pooled into two named error-profile groups."""
    if config is None:
        config = PlotConfig()
    config = dataclasses.replace(config, ancestor_error_y_log=False)
    labels = dict(zip(config.combined_error_profiles, config.combined_error_labels))
    return plot_ancestor_overlap_errors(
        summary, figure_folder, config, figsize,
        group_by="error_profile", group_labels=labels, group_xlabel="Error profile",
        name="combined_error_overlap_errors", show_mismatch_change=True,
        show_counts=False, panel_labels=("A", "B"), figure=figure,
    )


def plot_combined_overshoot_diff(
    comparisons, figure_folder=None, config=None, figsize=None, *,
    num_bins=None, legend_fontsize=None, figure=None,
):
    """Usual shared-bin counts and signed changes for `none` versus `all`."""
    if config is None:
        config = PlotConfig()
    return _overshoot_diff(
        config, comparisons, "error_profile", config.combined_error_profiles,
        "Error profile", "combined_error_overshoot", figure_folder,
        figsize=figsize, num_bins=num_bins, legend_fontsize=legend_fontsize,
        figure=figure,
    )


def plot_combined_errors(
    comparisons, figure_folder=None, config=None, figsize=None, *, num_bins=None,
):
    """Four-panel thesis figure: 40% accuracy (A–B), 60% overshoot (C–D).

    The standalone counterparts are :func:`plot_combined_overlap_errors`
    and :func:`plot_combined_overshoot_diff`.
    """
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.combined_error_size
    fig = plt.figure(figsize=figsize, layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=config.combined_error_width_ratios)
    summary = summarise_combined_overlap_errors(comparisons)
    plot_combined_overlap_errors(summary, config=config, figure=left)
    right_config = dataclasses.replace(config, overshoot_diff_panel_labels=("C", "D"))
    plot_combined_overshoot_diff(
        comparisons, config=right_config, figure=right, num_bins=num_bins,
    )
    return save_figure(fig, figure_folder, "combined_errors", config)


def _draw_overshoot_group(ax, values, position, box_width, color, rng, config, conf, alpha):
    """Sample central dots unless ``jitter_max_points=None``; keep one opacity."""
    ax.boxplot(
        [values], positions=[position], widths=box_width,
        patch_artist=True, manage_ticks=False, showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"}, capprops={"color": "black"},
        medianprops={"color": "black", "linewidth": 1.2}, zorder=3,
    )
    low, high = np.quantile(values, [conf, 1 - conf])
    if config.jitter_max_points is None:
        sampled = values
    else:
        sample_low, sample_high = config.jitter_sample_interval_mb
        within_interval = (values >= sample_low) & (values <= sample_high)
        central = values[within_interval]
        sample_size = min(len(central), config.jitter_max_points)
        sampled = rng.choice(central, size=sample_size, replace=False)
        sampled = np.concatenate([sampled, values[~within_interval]])
    jitter = rng.uniform(-config.jitter_width / 2, config.jitter_width / 2, len(sampled))
    ax.scatter(position + jitter, sampled, s=config.jitter_size, alpha=alpha,
               color=color, edgecolors="none", rasterized=True, zorder=2)
    ax.hlines([low, high], position - config.jitter_width / 2,
              position + config.jitter_width / 2, color="black",
              linestyle=config.quantile_line_style, linewidth=config.quantile_line_width,
              zorder=5)


def _symlog_overshoot_axis(ax, config):
    """Use signed decade ticks and a linear region that includes zero."""
    linthresh = config.overshoot_linthresh_bp / config.bp_per_mb
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())


def _overshoot_zero_line(ax):
    """Match the border, above jitter (2) and below boxes (3)."""
    ax.axhline(0, color="black", linestyle="-",
               linewidth=ax.spines["left"].get_linewidth(), zorder=2.5)


def _overshoot_legends(fig, config, conf, alpha, conf_secondary=None):
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", color=color,
                      alpha=alpha, markeredgecolor="none", label=_version_label(version))
        for version, color in zip(config.versions, config.version_colors)
    ]
    handles.append(mlines.Line2D([], [], color="black", linestyle=config.quantile_line_style,
                                 label=f"{conf * 100:g}% tail quantiles"))
    if conf_secondary is not None:
        handles.append(mlines.Line2D(
            [], [], color=config.quantile_secondary_color,
            linestyle=config.quantile_secondary_line_style,
            label=f"{conf_secondary * 100:g}% tail quantiles",
        ))
    fig.legend(handles=handles, frameon=False, loc="outside right center")


def _overshoot_boxplot(
    config, frame, by, rates, label, name, figure_folder, conf, alpha,
    figsize=None, group_labels=None,
    y_log=False, conf_secondary=None,
):
    """Grouped boxes of focal-site sides; see `_draw_overshoot_group`.

    ``y_log=True`` uses a symmetric log axis, preserving signed values
    and zeros. The central linear region is set in :class:`PlotConfig`.
    Quartiles, whiskers and tail quantiles precede the axis transformation.
    """
    if not 0 < conf < 0.5:
        raise ValueError("conf must be between 0 and 0.5")
    if conf_secondary is not None and not 0 < conf_secondary < 0.5:
        raise ValueError("conf_secondary must be between 0 and 0.5")
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
            if conf_secondary is not None:
                quantiles = np.quantile(values, [conf_secondary, 1 - conf_secondary])
                ax.hlines(
                    quantiles, position - config.jitter_width / 2,
                    position + config.jitter_width / 2,
                    color=config.quantile_secondary_color,
                    linestyle=config.quantile_secondary_line_style,
                    linewidth=config.quantile_line_width, zorder=5,
                )
    if y_log:
        _symlog_overshoot_axis(ax, config)
    _overshoot_zero_line(ax)
    ax.set_ylabel("Overshoot (Mb)")
    ax.set_xlabel(label)
    if group_labels is None:
        labels = [_profile_label(rate, by, config) for rate in rates]
    else:
        labels = [group_labels[rate] for rate in rates]
    ax.set_xticks(range(len(rates)), labels)
    ax.set_xlim(-0.5, len(rates) - 0.5)
    ax.grid(axis="y", alpha=0.15)
    _overshoot_legends(fig, config, conf, alpha, conf_secondary)
    return save_figure(fig, figure_folder, f"{name}_box", config)


@dataclasses.dataclass
class TailSteps:
    edges: np.ndarray
    fractions: np.ndarray


def plot_demography_overshoot_box(
    comparisons, model_labels, figure_folder=None, config=None,
    conf=0.01, alpha=None, figsize=None,
    y_log=False, conf_secondary=0.01,
):
    """Plot both error-free sides, grouped by model, using the usual boxes.

    ``model_labels`` maps model identifiers to display labels in plot order.
    Load comparisons with :func:`load_demography_comparisons`.
    Set ``y_log=True`` for a signed log axis that retains zero values.
    ``conf_secondary`` adds a second pair of tail quantile caps and legend
    entry without changing the point sample. Set it to None to omit
    the additional caps.
    """
    if config is None:
        config = PlotConfig()
    return _overshoot_boxplot(
        config, comparisons, "model", list(model_labels), "Demographic model",
        "demography_overshoot", figure_folder, conf, alpha, figsize=figsize, y_log=y_log,
        group_labels=model_labels, conf_secondary=conf_secondary,
    )


def plot_demography_overshoot_quantiles(
    comparisons, model_labels, figure_folder=None, config=None, figsize=None,
):
    """Rows of demographic models, with central and tail quantiles in two columns.

    Use error-free focal-site sides from :func:`load_demography_comparisons`.
    All observations contribute to the quantiles; no observations are sampled.
    The central column shows medians and IQRs in kb. The tail column shows
    1st--99th percentile intervals inside 0.1st--99.9th percentile intervals
    in Mb. Both columns use linear axes shared across demographic models.
    """
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.demography_quantile_size
    fig, axes = plt.subplots(
        len(model_labels), 2, figsize=figsize, sharex="col", squeeze=False,
        layout="constrained",
    )
    positions = np.arange(len(config.versions))[::-1]
    for row, (model, label) in enumerate(model_labels.items()):
        central, tails = axes[row]
        for version, color, position in zip(config.versions, config.version_colors, positions):
            selected = comparisons.loc[
                (comparisons.model == model) & (comparisons.version == version)
            ]
            if selected.empty:
                raise ValueError(f"No overshoots for {version}, model={model}")
            quantiles = np.quantile(selected.overshoot, config.demography_quantile_probabilities)
            outer_low, inner_low, q25, median, q75, inner_high, outer_high = quantiles
            central.plot(
                np.array([q25, q75]) / config.bp_per_kb, [position, position],
                color=color, linewidth=config.demography_quantile_interval_width,
                solid_capstyle="butt",
            )
            central.plot(
                median / config.bp_per_kb, position, marker="o", color=color,
                markeredgecolor="white", markersize=config.demography_quantile_marker_size,
            )
            tails.plot(
                np.array([outer_low, outer_high]) / config.bp_per_mb, [position, position],
                color=color, linewidth=config.quantile_line_width, marker="|",
                markersize=config.demography_quantile_cap_size,
            )
            tails.plot(
                np.array([inner_low, inner_high]) / config.bp_per_mb, [position, position],
                color=color, linewidth=config.demography_quantile_interval_width,
                solid_capstyle="butt",
            )
        central.set_ylabel(label, rotation=0, ha="right", va="center",
                           fontsize=config.demography_quantile_label_fontsize)
        central.set_xlim(config.demography_quantile_center_limits_kb)
        tails.set_xlim(config.demography_quantile_tail_limits_mb)
        for ax in (central, tails):
            ax.axvline(0, color=config.zero_color, linewidth=config.quantile_line_width,
                       linestyle=config.quantile_line_style, zorder=0)
            ax.set_yticks([])
            ax.set_ylim(-0.5, len(config.versions) - 0.5)
            ax.tick_params(axis="x", labelsize=config.demography_quantile_label_fontsize)
            ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0, 0].set_title("A: Median and IQR", loc="left")
    axes[0, 1].set_title("B: Tail quantiles", loc="left")
    axes[-1, 0].set_xlabel("Overshoot (kb)")
    axes[-1, 1].set_xlabel("Overshoot (Mb)")
    version_handles = [
        mlines.Line2D([], [], color=color, linewidth=config.demography_quantile_interval_width,
                      label=_version_label(version))
        for version, color in zip(config.versions, config.version_colors)
    ]
    fig.legend(handles=version_handles, loc="outside upper center", ncols=len(config.versions),
               frameon=False)
    return save_figure(fig, figure_folder, "demography_overshoot_quantiles", config)


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
            category = _profile_label(rate, by, config)
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
        categories = [_profile_label(rate, by, config) for rate in rates]
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


def summarise_overshoots(frame, by, rates, config=None):
    """Pool signed focal-site sides for medians, IQRs and the negative tail.

    Quantiles use linear interpolation. The tail fraction counts sides strictly
    below the configured signed bp threshold; left and right are not averaged.
    """
    if config is None:
        config = PlotConfig()
    selected = frame.loc[frame[by].isin(rates) & frame.version.isin(config.versions)]
    grouped = selected.groupby([by, "version"], sort=False).overshoot
    summary = grouped.agg(
        num_sides="size",
        q25_bp=lambda values: values.quantile(0.25),
        median_bp="median",
        q75_bp=lambda values: values.quantile(0.75),
        fraction_below_threshold=lambda values: values.lt(config.overshoot_summary_threshold_bp).mean(),
    )
    index = pd.MultiIndex.from_product([rates, config.versions], names=[by, "version"])
    return summary.reindex(index).reset_index()


def plot_overshoot_summary(
    summary, by, rates, regime_labels, name, figure_folder=None, config=None,
):
    """Matched rows of median/IQR intervals and left-aligned negative-tail bars.

    Pass :func:`summarise_overshoots` output. Both panels use the same version
    ordering and colours; their linear x axes are shared across error regimes.
    """
    if config is None:
        config = PlotConfig()
    fig, axes = plt.subplots(
        len(rates), 2, figsize=config.overshoot_summary_size, sharex="col",
        squeeze=False, layout="constrained",
    )
    positions = np.arange(len(config.versions))[::-1]
    fraction_max = summary.fraction_below_threshold.max()
    fraction_limit = fraction_max * (1 + config.overshoot_summary_fraction_margin)
    indexed = summary.set_index([by, "version"])
    for row, rate in enumerate(rates):
        central, tail = axes[row]
        for version, color, position in zip(config.versions, config.version_colors, positions):
            values = indexed.loc[(rate, version)]
            interval = np.array([values.q25_bp, values.q75_bp]) / config.bp_per_mb
            central.plot(
                interval, [position, position], color=color,
                linewidth=config.overshoot_summary_interval_width, solid_capstyle="butt",
            )
            central.plot(
                values.median_bp / config.bp_per_mb, position, marker="o", color=color,
                markeredgecolor="white", markersize=config.overshoot_summary_marker_size,
            )
            tail.barh(
                position, values.fraction_below_threshold, left=0, color=color,
                height=config.overshoot_summary_bar_height,
            )
        central.axvline(0, color=config.zero_color, linewidth=config.quantile_line_width,
                        linestyle=config.quantile_line_style, zorder=0)
        regime_label = regime_labels[rate]
        if by == "space_per_switch" and rate != 0:
            ser = summary.loc[summary[by] == rate, "phase_ser"].iloc[0]
            coefficient, exponent = f"{ser:.2e}".split("e")
            regime_label += "\n" + rf"SER = ${coefficient} \times 10^{{{int(exponent)}}}$"
        central.set_ylabel(regime_label, rotation=0, ha="right", va="center",
                           fontsize=config.overshoot_summary_label_fontsize)
        tail.set_xlim(0, fraction_limit)
        tail.xaxis.set_major_locator(mticker.MaxNLocator(nbins=config.overshoot_summary_fraction_ticks))
        tail.xaxis.set_major_formatter(mticker.PercentFormatter(
            xmax=1, decimals=config.overshoot_summary_percent_decimals,
        ))
        for ax in (central, tail):
            ax.set_yticks([])
            ax.set_ylim(-0.5, len(config.versions) - 0.5)
            ax.tick_params(axis="x", labelsize=config.overshoot_summary_label_fontsize)
            ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0, 0].set_title("A: Median and IQR", loc="left")
    threshold_mb = config.overshoot_summary_threshold_bp / config.bp_per_mb
    axes[0, 1].set_title(f"B: Overshoot < {threshold_mb:g} Mb", loc="left")
    axes[-1, 0].set_xlabel("Overshoot (Mb)")
    axes[-1, 1].set_xlabel("Fraction of focal-site sides")
    handles = [
        mlines.Line2D([], [], color=color, linewidth=config.overshoot_summary_interval_width,
                      label=_version_label(version))
        for version, color in zip(config.versions, config.version_colors)
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(config.versions),
               frameon=False)
    return save_figure(fig, figure_folder, name, config)


def _mean_overshoot_by_profile(frame, by, rates):
    """Pool focal-site means, averaging each site's two sides before pooling."""
    frame = frame.loc[frame[by].isin(rates)]
    group_keys = ["version", by]
    focal_keys = group_keys + ["replicate", "true_focal_site"]
    focal_means = frame.groupby(focal_keys).overshoot.mean()
    return focal_means.groupby(level=group_keys).mean()


def _overshoot_difference_label(difference_bp, config):
    """Format a signed mean difference using a compact physical-length unit."""
    if abs(difference_bp) >= config.bp_per_mb:
        value = difference_bp / config.bp_per_mb
        unit = "Mb"
    elif abs(difference_bp) >= config.bp_per_kb:
        value = difference_bp / config.bp_per_kb
        unit = "kb"
    else:
        value = difference_bp
        unit = "bp"
    number = f"{value:+.{config.mean_change_decimals}f}"
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return f"{number}{unit}"


def _overshoot_scatter(data, frame, by, rates, x_rates, xlabel, name, figure_folder,
                       figsize=None, x_log=False):
    """Connect mean signed overshoots against error rates, with Pearson r.

    Use the same means as `_overshoot_mean`. Vertical guides label the error
    profiles below zero. Orange labels give the signed difference in mean
    overshoot (0.5 minus 0.4), using compact bp/kb/Mb units. The x-axis is linear and retains
    the error-free profile at zero. With x_log=True, correlate against log10(x)
    and use a logarithmic axis; supplied x values must then be positive.
    """
    config = data.config
    means = _mean_overshoot_by_profile(frame, by, rates)
    x_values = x_rates.reindex(rates)
    x_values = x_values.sort_values()
    correlation_x = np.log10(x_values) if x_log else x_values
    if figsize is None:
        figsize = config.overshoot_scatter_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    plotted_means = means.unstack("version").reindex(x_values.index) / config.bp_per_mb
    mean_range = plotted_means.max().max() - plotted_means.min().min()
    guide_depth = config.overshoot_scatter_guide_depth
    if by == "space_per_switch":
        guide_depth = config.phase_scatter_guide_depth
    guide_label_y = -guide_depth * mean_range
    change_offset = (config.phase_scatter_change_offset if by == "space_per_switch"
                     else config.overshoot_scatter_change_offset)
    for index, (version, color) in enumerate(zip(config.versions, config.version_colors)):
        values = means.loc[version].reindex(x_values.index) / config.bp_per_mb
        correlation = np.corrcoef(correlation_x, values)[0, 1]
        label = f"{_version_label(version)} (r = {correlation:.{config.overshoot_scatter_correlation_decimals}f})"
        ax.plot(x_values, values, "o-", color=color, markerfacecolor=color,
                markeredgecolor=color, markersize=config.overshoot_scatter_marker_size,
                label=label, zorder=3)
        if index == 1:
            reference = means.loc[config.versions[0]].reindex(x_values.index)
            differences = means.loc[version].reindex(x_values.index) - reference
            connector = None
            if by == "geno_multiplier":
                connector = {
                    "arrowstyle": "-", "color": color,
                    "linewidth": config.genotype_scatter_connector_width,
                    "relpos": (1, 0.5), "shrinkA": 0,
                    "shrinkB": config.overshoot_scatter_marker_size / 2,
                }
            for (rate, x_value), value, difference in zip(x_values.items(), values, differences):
                point_offset = change_offset
                vertical_alignment = "bottom"
                if by == "geno_multiplier" and rate >= 1:
                    point_offset = config.genotype_scatter_error_change_offset
                    vertical_alignment = "top"
                ax.annotate(
                    _overshoot_difference_label(difference, config), xy=(x_value, value),
                    xytext=point_offset, textcoords="offset points",
                    color=color, fontsize=config.overshoot_scatter_change_fontsize,
                    ha="right", va=vertical_alignment,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                    arrowprops=connector,
                )
    for rate, x_value in x_values.items():
        label = f"{rate:g}X" if by == "geno_multiplier" else _profile_label(rate, by, config)
        if by == "space_per_switch":
            label += " Mb"
        ax.axvline(x_value, color=config.zero_color, linestyle=config.overshoot_scatter_vertical_dash,
                   linewidth=config.overshoot_diff_line_width, zorder=0)
        ax.annotate(label, xy=(x_value, guide_label_y),
                    xytext=(config.overshoot_scatter_label_padding, 0),
                    textcoords="offset points", color=config.zero_color,
                    fontsize=config.overshoot_scatter_label_fontsize,
                    va="top", ha="left")
    ax.axhline(0, color="black", linestyle=config.overshoot_scatter_zero_dash,
               linewidth=config.overshoot_diff_line_width, zorder=1)
    if x_log:
        ax.set_xscale("log")
    ax.set_xticks(x_values.to_numpy())
    if by == "space_per_switch":
        labels = [_profile_label(value, "phase_ser", config) for value in x_values]
        ax.set_xticklabels(labels)
    else:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(
            xmax=1, decimals=config.overshoot_scatter_tick_decimals,
        ))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.margins(x=config.overshoot_scatter_x_margin, y=config.overshoot_scatter_y_margin)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean overshoot (Mb)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    return save_figure(fig, figure_folder, f"{name}_scatter", config)


def _overshoot_mean(data, frame, by, rates, label, name, figure_folder, figsize=None):
    """Grouped mean signed overshoot by error profile, coloured by version.

    Each focal site contributes its mean left/right overshoot. Focal sites
    are pooled across replicates, as in `_overshoot_histogram`. Labels below
    the second version's bars give 100 * (new mean - old mean) / abs(old mean),
    so a decrease remains negative when the reference mean is negative.
    """
    config = data.config
    means = _mean_overshoot_by_profile(frame, by, rates)
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
        if index == 1:
            reference = means.loc[config.versions[0]].reindex(rates)
            changes = 100 * (values - reference) / reference.abs()
            for offset, value, change in zip(offsets, values, changes):
                ax.annotate(
                    f"{change:+.{config.mean_change_decimals}f}%",
                    xy=(offset, min(value / config.bp_per_mb, 0)),
                    xytext=(0, -config.mean_change_padding),
                    textcoords="offset points", ha="center", va="top",
                    color=color, fontsize=config.mean_change_fontsize,
                )
    labels = [_profile_label(rate, by, config) for rate in rates]
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


def plot_genotype_overshoot_scatter(data, figure_folder=None, figsize=None):
    """Mean overshoot versus measured pooled diploid error rate, as in scaling."""
    exported = data.zarr_data.loc[data.zarr_data.record_type == "genotype"]
    pooled = exported.groupby("geno_multiplier")[["changed", "total"]].sum()
    x_rates = pooled.changed / pooled.total
    # The error-free profile has no exported genotype-error counts.
    x_rates.loc[0] = 0
    return _overshoot_scatter(
        data, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        x_rates, "Diploid genotype error rate", "genotype_error_overshoot", figure_folder,
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


def plot_genotype_overshoot_diff(data, figure_folder=None, figsize=None, legend_fontsize=None, num_bins=None):
    """Genotype counts and percentage changes; see `_overshoot_diff`."""
    return _overshoot_diff(
        data.config, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Error multiplier", "genotype_error_overshoot", figure_folder,
        figsize=figsize, legend_fontsize=legend_fontsize, num_bins=num_bins,
    )


def plot_genotype_overshoot_box(
    data, figure_folder=None, conf=0.01, alpha=None, figsize=None, y_log=False,
):
    """Grouped boxplots for pure genotype-error profiles; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data.config, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Error multiplier", "genotype_error_overshoot", figure_folder, conf, alpha,
        figsize=figsize, y_log=y_log,
    )


def plot_genotype_overshoot_tails(data, figure_folder=None, figsize=None, fraction_above=True):
    """Above/below magnitude cutoff fractions by genotype-error multiplier; see `_overshoot_tails`."""
    return _overshoot_tails(
        data, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Error multiplier", "genotype_error_overshoot", figure_folder,
        figsize=figsize, fraction_above=fraction_above,
    )


def load_tgp_daf(zarr_path, config=None):
    """Load TGP counts in the configured half-open region without variant filters.

    The all-sample counts already account for polarisation; ancestral plus
    derived counts give the called denominator. No variant masks are read.
    See :func:`plot_tgp_daf` for the matching spectrum.
    """
    if config is None:
        config = PlotConfig()
    group = zarr.open_group(zarr_path, mode="r")
    positions = group["variant_position"][:]
    start = np.searchsorted(positions, config.tgp_region_start)
    stop = np.searchsorted(positions, config.tgp_region_end)
    region = slice(start, stop)
    prefix = config.tgp_count_prefix
    derived = group[f"{prefix}_derived_count"][region]
    ancestral = group[f"{prefix}_ancestral_count"][region]
    return pd.DataFrame({
        "position": positions[region],
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
        & (data.spans.geno_multiplier == 0) & (data.spans.phase_ser == 0)
        & data.spans.focal_site_mispolarised
    ]
    flipped = focal.groupby(["replicate", "mispol_rate", "true_derived_count"]).size()
    flipped = flipped.rename("num_sites").reset_index()
    flipped = flipped.rename(columns={"true_derived_count": "derived_count"})
    frames = []
    for rate in data.config.mispol_rates:
        selected = flipped.loc[flipped.mispol_rate == rate, keys[:2] + ["num_sites"]]
        counts = segregating.merge(selected, on=keys[:2], how="left", validate="one_to_one")
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


def plot_mispolarisation_daf(data, figure_folder=None, figsize=None, frequencies=None, tgp_variants=None):
    """Overlay smoothed site counts per DAF bin after flipping mispolarised sites.

    The stored allele counts precede mispolarisation. The mask is True for
    correctly polarised sites; its complement selects the DAFs to reverse.
    Fixed-width histograms and Gaussian smoothing avoid KDE evaluation costs
    growing with the number of sites. Counts are pooled across replicates,
    including singletons, which are eligible for mispolarisation. Pass counts
    from :func:`mispolarisation_daf_counts` to use current aggregate polarisation.
    See :func:`plot_tgp_daf` for the empirical comparison with the same bins,
    smoothing and line width. Pass ``tgp_variants`` to overlay the empirical
    spectrum on the same axes with separate simulation and empirical legends.
    """
    config = data.config
    replicates = set(data.spans.replicate)
    if frequencies is None:
        frequencies = data.zarr_data.loc[data.zarr_data.record_type == "daf"]
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(config.mispol_rates)))
    if figsize is None:
        figsize = config.daf_size if tgp_variants is None else config.combined_daf_size
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
    if tgp_variants is None:
        ax.legend(title="Simulation:\nMispol. rate", loc="upper right", frameon=False)
    else:
        simulation_legend = ax.legend(
            title=f"{config.simulation_daf_legend_label}\nMispol. rate", loc="upper right",
            bbox_to_anchor=(config.combined_daf_legend_x, config.combined_daf_simulation_legend_y),
            frameon=False, alignment="left", borderaxespad=0,
            fontsize=config.combined_daf_legend_fontsize,
            title_fontsize=config.combined_daf_legend_fontsize,
        )
        ax.add_artist(simulation_legend)
        tgp_frequencies = tgp_variants.derived_count.to_numpy() / tgp_variants.num_alleles.to_numpy()
        empirical_line = _plot_daf_counts(
            ax, tgp_frequencies, config, color=config.tgp_color,
            label=config.combined_tgp_legend_label,
        )
        ax.legend(
            handles=[empirical_line], loc="upper right", frameon=False,
            bbox_to_anchor=(config.combined_daf_legend_x, config.combined_daf_tgp_legend_y),
            borderaxespad=0, fontsize=config.combined_daf_legend_fontsize,
        )
    return save_figure(fig, figure_folder, "mispolarisation_daf", config)


def load_mispolarisation_overlap_errors(aggregated_path, config=None):
    """Read one overlap per focal site for the baseline and pure mispol. profiles.

    Split the latter by its own focal site's polarisation, retaining sites
    separately even when they share an inferred ancestor. Error counts already
    compare truth and inference in the same observed allele encoding.
    """
    if config is None:
        config = PlotConfig()
    query = pl.scan_csv(pathlib.Path(aggregated_path).expanduser())
    selected = query.filter(
        (pl.col("side") == "left")
        & (pl.col("geno_multiplier") == 0)
        & (pl.col("phase_ser") == 0)
        & pl.col("mispol_rate").is_in([0.0, *config.mispolarisation_error_rates])
        & pl.col("version").cast(pl.String).is_in(config.versions)
    )
    focal_group = (
        pl.when(pl.col("focal_site_mispolarised")).then(pl.lit("mispolarised"))
        .otherwise(pl.lit("correctly_polarised"))
    )
    rate_group = pl.concat_str(pl.col("mispol_rate"), focal_group, separator="_")
    group = pl.when(pl.col("mispol_rate") == 0).then(pl.lit("none")).otherwise(rate_group)
    selected = selected.select(
        group.alias("group"), "replicate", "version", "true_focal_site", "focal_position",
        "mispol_rate", "focal_site_mispolarised", "allele_frequency", "num_errors", "overlap_span",
        "overlap_site_span",
        "num_should_be_0", "num_should_be_1",
    )
    table = selected.collect(engine="streaming")
    return pd.DataFrame(table.to_dict(as_series=False))


def summarise_mispolarisation_overlap_errors(comparisons):
    """Pool :func:`load_mispolarisation_overlap_errors` by focal polarisation."""
    keys = ["group", "replicate", "version", "true_focal_site"]
    if comparisons.duplicated(keys).any():
        raise ValueError("Expected one overlap comparison per group, focal site and version")
    summary = _pool_overlap_errors(comparisons, ["group", "version"]).reset_index()
    return _overlap_error_rates(summary)


def plot_mispolarisation_overlap_errors(summary, figure_folder=None, config=None, figsize=None):
    """Accuracy and count panels with paired focal-state groups under each rate."""
    if config is None:
        config = PlotConfig()
    config = dataclasses.replace(
        config, ancestor_error_log_subs=config.mispolarisation_error_log_subs,
        ancestor_error_y_log=True,
    )
    if figsize is None:
        figsize = config.mispolarisation_error_size
    labels = {"none": "No mispol."}
    for rate in config.mispolarisation_error_rates:
        labels[f"{rate:g}_correctly_polarised"] = "Non-mispol.\nfocal site"
        labels[f"{rate:g}_mispolarised"] = "Mispol.\nfocal site"
    fig = plt.figure(figsize=figsize, layout="constrained")
    plot_ancestor_overlap_errors(
        summary, config=config,
        group_by="group", group_labels=labels, group_xlabel="",
        show_mismatch_change=True, panel_labels=("A", "B", "C"), figure=fig,
        split_error_types=False,
    )
    ax = fig.axes[-1]
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        mticker.LogLocator(subs=config.mispolarisation_error_log_subs,
                           numticks=config.ancestor_error_log_numticks)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelsize=config.mispolarisation_error_group_fontsize)
    y = config.mispolarisation_error_bracket_y
    depth = config.mispolarisation_error_bracket_depth
    for index, rate in enumerate(config.mispolarisation_error_rates):
        left = 1 + 2 * index - config.mean_bar_group_width / 2
        right = 2 + 2 * index + config.mean_bar_group_width / 2
        ax.plot([left, left, right, right], [y + depth, y, y, y + depth],
                transform=ax.get_xaxis_transform(), color="black", linewidth=1,
                clip_on=False)
        ax.text((left + right) / 2, config.mispolarisation_error_rate_label_y,
                f"Mispol. rate = {rate:g}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=config.mispolarisation_error_group_fontsize)
    return save_figure(fig, figure_folder, "mispolarisation_overlap_errors", config)


def _pool_mispolarisation_mismatch(frame, group_columns, index):
    """Pool errors with both bp and site denominators, preserving empty bins."""
    grouped = frame.groupby(group_columns, sort=False).agg(
        num_ancestors=("true_focal_site", "size"),
        num_errors=("num_errors", "sum"),
        overlap_bp=("overlap_span", "sum"),
        overlap_sites=("overlap_site_span", "sum"),
    )
    summary = grouped.reindex(index, fill_value=0)
    summary["mismatch_fraction"] = summary.num_errors / summary.overlap_sites.replace(0, np.nan)
    return summary.reset_index()


def summarise_mispolarisation_mismatch_by_daf(comparisons, true_frequencies, config=None, num_bins=None):
    """Pool error-free, mispolarised and non-mispolarised focals by true DAF.

    Consume :func:`load_mispolarisation_overlap_errors` and true frequencies
    from :func:`_true_focal_frequencies`. Match truth by replicate and physical
    focal position, including mispolarised true singletons absent from the
    error-free inference sites. Each focal contributes one whole overlap.
    Bins have equal width on [0, 1], including 1 in the final bin. Errors and compared sites are pooled before division; empty bins have undefined fractions.
    """
    if config is None:
        config = PlotConfig()
    if num_bins is None:
        num_bins = config.mispolarisation_mismatch_bins
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    selected = comparisons.loc[
        (comparisons.version == config.ancestor_error_version)
        & comparisons.mispol_rate.isin([0, config.mispolarisation_mismatch_rate])
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
    selected["bin"] = pd.cut(selected.true_allele_frequency, cut_edges, right=False, labels=False)
    index = pd.MultiIndex.from_product(
        [config.mispolarisation_mismatch_groups, range(num_bins)], names=["focal_group", "bin"],
    )
    binned = _pool_mispolarisation_mismatch(selected, ["focal_group", "bin"], index)
    binned["bin_left"] = edges[binned.bin]
    binned["bin_right"] = edges[binned.bin + 1]
    return binned


def plot_mispolarisation_mismatch_by_daf(
    summary, figure_folder=None, config=None, figsize=None, *, y_scale="log",
):
    """Grouped mismatch-fraction bars in A and focal-site counts in B, on shared DAF bins.

    Use :func:`summarise_mispolarisation_mismatch_by_daf`. Fractions use
    compared inference sites and default to log y;
    the count panel is always logarithmic. Colours follow the same group
    order in both panels: no error, mispol focal sites, non-mispol focal sites.
    """
    if config is None:
        config = PlotConfig()
    if figsize is None:
        figsize = config.mispolarisation_mismatch_size
    if y_scale not in ("linear", "log"):
        raise ValueError('y_scale must be "linear" or "log"')
    fraction_column = "mismatch_fraction"
    groups = config.mispolarisation_mismatch_groups
    fig, (fraction_ax, count_ax) = plt.subplots(
        2, 1, sharex=True, figsize=figsize, layout="constrained",
        gridspec_kw={"height_ratios": config.mispolarisation_mismatch_height_ratios},
    )
    width = config.mean_bar_group_width / len(groups)
    for position, (group, color, label) in enumerate(zip(
        groups, config.mispolarisation_mismatch_colors, config.mispolarisation_mismatch_labels,
    )):
        binned = summary.loc[summary.focal_group == group]
        offsets = binned.bin - config.mean_bar_group_width / 2 + width * (position + 0.5)
        fraction_ax.bar(offsets, binned[fraction_column], width=width, color=color, label=label)
        count_ax.bar(offsets, binned.num_ancestors, width=width,
                     facecolor="white", edgecolor=color)
    fraction_ax.set_ylabel("Pooled mismatch fraction\n(mismatches/site)")
    fraction_ax.set_yscale(y_scale)
    if y_scale == "linear":
        fraction_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        maximum = summary[fraction_column].max()
        fraction_ax.set_ylim(0, maximum * (1 + config.mispolarisation_mismatch_y_margin))
    else:
        fraction_ax.margins(y=config.mispolarisation_mismatch_y_margin)
    count_ax.set_ylabel("Count")
    count_ax.set_yscale("log")
    count_ax.yaxis.set_major_locator(mticker.LogLocator(numticks=3))
    count_ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    bins = summary.loc[summary.focal_group == groups[0]]
    bin_labels = [f"{row.bin_left:g}–{row.bin_right:g}" for row in bins.itertuples()]
    count_ax.set_xticks(bins.bin, bin_labels, rotation=45, ha="right")
    count_ax.set_xlim(-0.6, len(bins) - 0.4)
    count_ax.set_xlabel("True focal derived allele frequency")
    for ax, label in zip((fraction_ax, count_ax), ("A", "B")):
        ax.text(*config.overshoot_diff_panel_label_position, label, transform=ax.transAxes,
                va="top", fontweight="bold", fontsize=config.overshoot_diff_panel_label_fontsize,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    handles, labels = fraction_ax.get_legend_handles_labels()
    title = f"{_version_label(config.ancestor_error_version)}; mispol. rate = {config.mispolarisation_mismatch_rate:g}"
    fig.legend(handles, labels, title=title, loc="outside upper center", ncol=len(groups), frameon=False)
    return save_figure(fig, figure_folder, "mispolarisation_mismatch_by_daf", config)


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


def plot_mispolarisation_overshoot_scatter(
    data, figure_folder=None, frequency="all", figsize=None,
):
    """Mean signed overshoot versus mispolarisation rate; see `_overshoot_scatter`."""
    rates = data.config.mispol_rates
    x_rates = pd.Series(rates, index=rates)
    return _overshoot_scatter(
        data, _mispolarisation_overshoots(data, frequency), "mispol_rate", rates,
        x_rates, "Mispolarisation rate", f"mispolarisation_overshoot_{frequency}_frequency",
        figure_folder, figsize=figsize,
    )


def plot_mispolarisation_overshoot_diff(
    data, figure_folder=None, frequency="all", figsize=None, legend_fontsize=None, num_bins=None,
    focal_site_mispolarised=None,
):
    """True-DAF-selected count changes, optionally selecting focal polarisation."""
    selected = _mispolarisation_overshoots(data, frequency)
    name = f"mispolarisation_overshoot_{frequency}_frequency"
    if focal_site_mispolarised is not None:
        selected = selected.loc[selected.focal_site_mispolarised == focal_site_mispolarised]
        state = "flipped" if focal_site_mispolarised else "correct"
        name = f"mispolarisation_overshoot_{state}_focal"
        if frequency != "all":
            name += f"_{frequency}_frequency"
    rates = data.config.mispol_rates
    if focal_site_mispolarised is True:
        rates = [rate for rate in rates if rate > 0]
    return _overshoot_diff(
        data.config, selected, "mispol_rate",
        rates, "Mispol. rate", name, figure_folder,
        figsize=figsize, legend_fontsize=legend_fontsize, num_bins=num_bins,
    )


def plot_mispolarisation_overshoot_box(
    data, figure_folder=None, frequency="all", conf=0.01, alpha=None,
    figsize=None, y_log=False,
):
    """Grouped mispolarisation boxplots; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data.config, _mispolarisation_overshoots(data, frequency), "mispol_rate",
        data.config.mispol_rates, "Mispolarisation rate",
        f"mispolarisation_overshoot_{frequency}_frequency", figure_folder, conf, alpha,
        figsize=figsize, y_log=y_log,
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


def _mispolarisation_low_count_overshoots(data, max_derived_count):
    """Select the shared low-count subset for histogram and diff plots."""
    frame = _mispolarisation_overshoots(data, frequency="all")
    return frame.loc[
        frame.focal_site_mispolarised & (frame.true_derived_count <= max_derived_count)
    ]


def _mispolarised_focal_sites(data):
    """One record per flipped focal site and rate, using the selected 0.5 version."""
    spans = data.spans
    rates = [rate for rate in data.config.mispol_rates if rate > 0]
    return spans.loc[
        (spans.version == data.config.ancestor_error_version) & (spans.geno_multiplier == 0)
        & (spans.phase_ser == 0) & spans.mispol_rate.isin(rates)
        & spans.focal_site_mispolarised
    ]


def summarise_mispolarised_focal_counts(data):
    """True singleton/doubleton proportions among all flipped focal sites by rate.

    Use the same cohort as panel C of :func:`plot_mispolarised_focal_overshoot_hist`.
    Each focal site is counted once, including sites sharing an ancestor.
    True counts come from exported error-free variants, before mispolarisation.
    """
    focal_sites = _mispolarised_focal_sites(data).copy()
    focal_sites["true_singleton"] = focal_sites.true_derived_count == 1
    focal_sites["true_doubleton"] = focal_sites.true_derived_count == 2
    summary = focal_sites.groupby("mispol_rate").agg(
        num_mispol_focal_sites=("true_derived_count", "size"),
        num_true_singletons=("true_singleton", "sum"),
        num_true_doubletons=("true_doubleton", "sum"),
    )
    summary["proportion_true_singletons"] = summary.num_true_singletons / summary.num_mispol_focal_sites
    summary["proportion_true_doubletons"] = summary.num_true_doubletons / summary.num_mispol_focal_sites
    summary = summary.reset_index()
    return summary[[
        "mispol_rate", "num_mispol_focal_sites", "num_true_singletons",
        "proportion_true_singletons", "num_true_doubletons", "proportion_true_doubletons",
    ]]


def plot_mispolarised_focal_overshoot_hist(
    data, figure_folder=None, figsize=None, legend_fontsize=None,
):
    """Narrow overshoot panels beside a discrete true-DAC spectrum for flipped focals.

    Overshoots retain all mispolarised focal sites at each positive rate.
    DAC curves use one focal-site record from the selected 0.5 version at each
    rate across the full observed DAC range. Straight segments join occupied
    integer bin centres, with markers only up to the configured maximum AC.
    Curves use the 0.5 rate palette, largest rate first.
    True counts come from exported error-free variants; both axes are logarithmic.
    """
    config = data.config
    if figsize is None:
        figsize = config.mispolarised_focal_size
    frame = _mispolarisation_overshoots(data, frequency="all")
    selected = frame.loc[frame.focal_site_mispolarised]
    rates = [rate for rate in config.mispol_rates if rate > 0]
    fig = plt.figure(figsize=figsize, layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=config.mispolarised_focal_width_ratios)
    _overshoot_histogram(
        data, selected, "mispol_rate", rates, "Mispol. rate",
        "mispolarisation_overshoot_flipped_focal", figure_folder,
        figsize=figsize, legend_fontsize=legend_fontsize,
        mirrored=False, panel_labels=True, lines=True, figure=left,
    )
    focal_sites = _mispolarised_focal_sites(data)
    palette = _overshoot_colors(rates, config.overshoot_cmaps[1])
    ax = right.subplots()
    plot_rates = sorted(rates, reverse=True)
    for rate in plot_rates:
        selected = focal_sites.loc[focal_sites.mispol_rate == rate]
        counts = selected.true_derived_count.value_counts().sort_index()
        marker_indices = np.flatnonzero(counts.index <= config.mispolarised_focal_dac_marker_max_count)
        ax.plot(counts.index, counts, color=palette[rate], label=f"{rate:g}",
                marker=config.mispolarised_focal_dac_marker,
                markevery=marker_indices,
                markersize=config.mispolarised_focal_dac_marker_size,
                linewidth=config.histogram_line_width)
    ax.set(xscale="log", yscale="log", xlabel="True derived allele count",
           ylabel="Focal-site count", title=config.mispolarised_focal_dac_title)
    ax.margins(y=config.mispolarised_focal_dac_y_margin)
    ax.legend(title="Mispol. rate", loc="upper center", frameon=False, fontsize=legend_fontsize)
    ax.text(*config.overshoot_diff_panel_label_position, "C", transform=ax.transAxes,
            va="top", fontweight="bold", fontsize=config.overshoot_diff_panel_label_fontsize,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    return save_figure(fig, figure_folder, "mispolarisation_overshoot_flipped_focal_hist", config)


def plot_mispolarisation_low_count_overshoot_scatter(
    data, figure_folder=None, max_derived_count=5, figsize=None,
):
    """Low-count mispolarised means against rate; see `_overshoot_scatter`."""
    selected = _mispolarisation_low_count_overshoots(data, max_derived_count)
    rates = [rate for rate in data.config.mispol_rates if rate > 0]
    x_rates = pd.Series(rates, index=rates)
    return _overshoot_scatter(
        data, selected, "mispol_rate", rates, x_rates, "Mispolarisation rate",
        f"mispolarisation_overshoot_derived_count_le_{max_derived_count}", figure_folder,
        figsize=figsize,
    )


def plot_mispolarisation_low_count_overshoot_diff(
    data, figure_folder=None, max_derived_count=5, figsize=None, legend_fontsize=None, num_bins=None,
):
    """Low-count mispolarised focal-site changes; see `_overshoot_diff`."""
    selected = _mispolarisation_low_count_overshoots(data, max_derived_count)
    rates = [rate for rate in data.config.mispol_rates if rate > 0]
    return _overshoot_diff(
        data.config, selected, "mispol_rate", rates, "Mispol. rate",
        f"mispolarisation_overshoot_derived_count_le_{max_derived_count}", figure_folder,
        figsize=figsize, legend_fontsize=legend_fontsize, num_bins=num_bins,
    )


def _phasing_overshoots(data):
    frame = data.overshoots
    return frame.loc[(frame.geno_multiplier == 0) & (frame.mispol_rate == 0)]


def plot_phasing_overshoot_mean(data, figure_folder=None, figsize=None):
    """Grouped mean overshoot by mean switch gap; see `_overshoot_mean`."""
    return _overshoot_mean(
        data, _phasing_overshoots(data), "space_per_switch", data.config.phase_spacings,
        "Mean gap (Mb)", "phasing_error_overshoot", figure_folder,
        figsize=figsize,
    )


def plot_phasing_overshoot_hist(data, figure_folder=None, figsize=None, legend_fontsize=None):
    """Mirrored histograms for pure phase-switch profiles."""
    return _overshoot_histogram(
        data, _phasing_overshoots(data), "space_per_switch", data.config.phase_spacings,
        "Mean gap (Mb)", "phasing_error_overshoot", figure_folder, reverse_order=True,
        figsize=figsize, legend_fontsize=legend_fontsize,
    )


def plot_phasing_overshoot_scatter(data, figure_folder=None, figsize=None, x_log=True):
    """Mean overshoot versus fitted SER; omit the zero-rate no-switch profile."""
    rates = tuple(gap for gap in data.config.phase_spacings if gap > 0)
    x_rates = data.phase_settings.set_index("space_per_switch").phase_ser
    return _overshoot_scatter(
        data, _phasing_overshoots(data), "space_per_switch", rates, x_rates,
        "Switch error rate", "phasing_error_overshoot", figure_folder,
        figsize=figsize, x_log=x_log,
    )


def plot_phasing_overshoot_diff(data, figure_folder=None, figsize=None, legend_fontsize=None, num_bins=None):
    """Switch-error counts and percentage changes; see `_overshoot_diff`."""
    # Assign colours by increasing error rate, rather than increasing switch spacing.
    rates = tuple(sorted(data.phase_settings.phase_ser))
    return _overshoot_diff(
        data.config, _phasing_overshoots(data), "phase_ser", rates,
        "Switch error rate", "phasing_error_overshoot", figure_folder,
        figsize=figsize, legend_fontsize=legend_fontsize, num_bins=num_bins,
    )


def plot_phasing_overshoot_box(
    data, figure_folder=None, conf=0.01, alpha=None, figsize=None, y_log=False,
):
    """Grouped phase-switch boxplots; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data.config, _phasing_overshoots(data), "space_per_switch", data.config.phase_spacings,
        "Mean gap (Mb)", "phasing_error_overshoot", figure_folder, conf, alpha,
        figsize=figsize, y_log=y_log,
    )


def plot_phasing_overshoot_tails(data, figure_folder=None, figsize=None, fraction_above=True):
    """Above/below magnitude cutoff fractions by mean switch gap; see `_overshoot_tails`."""
    return _overshoot_tails(
        data, _phasing_overshoots(data), "space_per_switch", data.config.phase_spacings,
        "Mean gap (Mb)", "phasing_error_overshoot", figure_folder,
        figsize=figsize, fraction_above=fraction_above,
    )


def _draw_phase_pair(ax, lefts, widths, states, y, config):
    for haplotype in range(2):
        for state in (False, True):
            selected = states == state
            runs = np.column_stack((lefts[selected], widths[selected]))
            color = config.phase_colors[haplotype ^ int(state)]
            ax.broken_barh(runs, (y + haplotype * 0.3, 0.24),
                           facecolors=color, edgecolors="none")


def plot_phase_switch_illustration(phase_runs_path, figure_folder=None, config=None, figsize=None):
    """Compare true phase with six exported rows from the 20 Mb simulation."""
    if config is None:
        config = PlotConfig()
    runs = pd.read_csv(phase_runs_path)
    rows = runs[["plot_row", "individual", "space_per_switch", "phase_ser"]].drop_duplicates().sort_values("plot_row")
    settings = rows.space_per_switch.drop_duplicates().to_list()
    left = runs.left.min() / config.bp_per_mb
    right = min(config.phase_right_mb, runs.right.max() / config.bp_per_mb)
    y_positions = []
    if figsize is None:
        figsize = config.phase_size
    fig, overview = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    for index, row in enumerate(rows.itertuples(index=False)):
        setting = settings.index(row.space_per_switch)
        y = ((len(rows) - index - 1) * config.phase_row_spacing
             + (len(settings) - setting - 1) * config.phase_group_gap)
        y_positions.append(y)
        selected = runs.loc[runs.plot_row == row.plot_row].sort_values("left")
        starts = selected.left.to_numpy() / config.bp_per_mb
        widths = (selected.right - selected.left).to_numpy() / config.bp_per_mb
        _draw_phase_pair(overview[0], np.array([left]), np.array([right - left]),
                         np.array([False]), y, config)
        _draw_phase_pair(overview[1], starts, widths, selected.phase.to_numpy(), y, config)
        if index == 0 or row.space_per_switch != rows.iloc[index - 1].space_per_switch:
            ser_label = _profile_label(row.phase_ser, "phase_ser", config)
            label = (f"SER = {ser_label} "
                     f"({row.space_per_switch / config.bp_per_mb:g} Mb mean gap)")
            overview[1].text(0.03, y + 0.65, label,
                             transform=overview[1].get_yaxis_transform(),
                             fontsize=config.phase_label_fontsize, va="bottom")
    for ax, title in zip(overview, ("Correct phase", "Simulated phase")):
        ax.set_xlim(left, right)
        ax.set_ylim(-0.1, max(y_positions) + 1.1)
        ax.set_yticks(np.asarray(y_positions) + 0.27, rows.individual.astype(str))
        ax.set_xlabel("Chromosome 20 position (Mb)")
        ax.set_title(title, fontsize=config.phase_label_fontsize)
    overview[0].set_ylabel("Individual ID")
    handles = [mpatches.Patch(color=color, label=f"True haplotype {index + 1}")
               for index, color in enumerate(config.phase_colors)]
    fig.legend(handles=handles, loc="outside right center", frameon=False)
    return save_figure(fig, figure_folder, "phase_switch_illustration", config)
