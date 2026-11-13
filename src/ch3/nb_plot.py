"""Chapter 3 exploratory figures and styling.

Scientific notebook settings live in notebook cells. This is rough work.
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
import xml.etree.ElementTree as ET

import IPython.display as ipd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from . import nb_utils

logger = logging.getLogger(__name__)


def save_figure(fig, figure_folder, name, config, *, dpi=300):
    """Display and close a figure; save a PNG only when a folder is supplied."""
    if figure_folder is not None:
        folder = pathlib.Path(figure_folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", path)
    plt.show()
    plt.close(fig)


def _ancestor_boxplot(
    data,
    figure_folder,
    variable,
    num_bins,
    cutoffs,
    conf=None,
    alpha=None,
    figsize=None,
    y_log=False,
    *,
    ancestor_count_fontsize=8,
    ancestor_count_margin=0.35,
    ancestor_count_padding=2,
    ancestor_panel_spacing=0.18,
    box_size=(7, 4.5),
    dpi=300,
    jitter_alpha=0.15,
    jitter_max_points=1500,
    jitter_sample_interval_mb=(-1, 1),
    jitter_seed=42,
    jitter_size=5,
    jitter_width=0.32,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_linthresh_bp=1000,
    quantile_line_style="--",
    quantile_line_width=1.2,
    quantile_secondary_color="#555555",
    quantile_secondary_line_style="-.",
    true_color="#b8b8b8",
    version_colors=("#1a80bb", "#ea801c"),
):
    config = data.config
    spans = nb_utils._no_errors(data.spans)
    true_spans = spans.drop_duplicates(["replicate", "true_focal_site"])
    edges = nb_utils._frequency_bin_edges(
        true_spans.allele_frequency, num_bins, cutoffs
    )
    if variable == "overshoot":
        if not 0 < conf < 0.5:
            raise ValueError("conf must be between 0 and 0.5")
        if alpha is None:
            alpha = jitter_alpha
        rng = np.random.default_rng(jitter_seed)
    num_bins = len(edges) - 1
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(edges[-1], np.inf)
    labels = [f"{left:g}–{right:g}" for left, right in zip(edges[:-1], edges[1:])]
    if figsize is None:
        figsize = box_size
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": ancestor_panel_spacing},
    )
    groups = []
    if variable == "span":
        groups.append(("True", true_color, true_spans, "true_span"))
        for version, color in zip(config.versions, version_colors):
            groups.append(
                (
                    nb_utils._version_label(version),
                    color,
                    spans.loc[spans.version == version],
                    "inferred_span",
                )
            )
    else:
        frame = nb_utils._no_errors(data.overshoots)
        for version, color in zip(config.versions, version_colors):
            groups.append(
                (
                    nb_utils._version_label(version),
                    color,
                    frame.loc[frame.version == version],
                    "overshoot",
                )
            )
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
                values,
                positions=positions,
                widths=width * 0.85,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                boxprops={"facecolor": color},
                medianprops={"color": "black"},
            )
        else:
            for selected, position in zip(values, positions):
                _draw_overshoot_group(
                    axes[0],
                    selected,
                    position,
                    width * 0.85,
                    color,
                    rng,
                    config,
                    conf,
                    alpha,
                    jitter_max_points=jitter_max_points,
                    jitter_sample_interval_mb=jitter_sample_interval_mb,
                    jitter_size=jitter_size,
                    jitter_width=jitter_width,
                    quantile_line_style=quantile_line_style,
                    quantile_line_width=quantile_line_width,
                )
        if variable == "span":
            handles.append(mpatches.Patch(color=color, label=label))
        if index == 0:
            focal_rows = subset.drop_duplicates(["replicate", "true_focal_site"])
            focal_bins = pd.cut(
                focal_rows.allele_frequency, cut_edges, labels=False, right=False
            )
            counts = focal_bins.value_counts().reindex(range(num_bins), fill_value=0)
            bars = axes[1].bar(
                np.arange(num_bins),
                counts,
                width=width * 0.85 * 2,
                facecolor="white",
                edgecolor="black",
            )
            if variable == "span":
                axes[1].bar_label(
                    bars,
                    labels=[f"{count:,}" for count in counts],
                    fontsize=ancestor_count_fontsize,
                    padding=ancestor_count_padding,
                )
    if variable == "span":
        axes[0].legend(
            handles=handles,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(groups),
        )
    else:
        fig.set_layout_engine("constrained")
        _overshoot_legends(
            fig,
            config,
            conf,
            alpha,
            quantile_line_style=quantile_line_style,
            quantile_secondary_color=quantile_secondary_color,
            quantile_secondary_line_style=quantile_secondary_line_style,
            version_colors=version_colors,
        )
    if variable == "span":
        if y_log:
            axes[0].set_yscale("log")
        axes[0].set_ylabel("Ancestor span (Mb)")
    else:
        if y_log:
            _symlog_overshoot_axis(
                axes[0], config, overshoot_linthresh_bp=overshoot_linthresh_bp
            )
        _overshoot_zero_line(axes[0])
        axes[0].set_ylabel("Overshoot (Mb)")
    axes[1].set_ylabel("Count")
    if variable == "span":
        axes[1].set_yscale("log")
        lower, upper = axes[1].get_ylim()
        count_range = counts.max() / lower
        upper = lower * count_range ** (1 / (1 - ancestor_count_margin))
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
        for ax, label in zip(axes, overshoot_diff_panel_labels):
            ax.text(
                *overshoot_diff_panel_label_position,
                label,
                transform=ax.transAxes,
                va="top",
                fontweight="bold",
                fontsize=overshoot_diff_panel_label_fontsize,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            )
    if variable == "span":
        fig.subplots_adjust(bottom=0.18, left=0.1, right=0.98, top=0.9)
    return save_figure(
        fig, figure_folder, f"ancestor_{variable}_boxplot", config, dpi=dpi
    )


def plot_ancestor_spans(
    data,
    figure_folder=None,
    num_bins=5,
    cutoffs=None,
    figsize=None,
    y_log=True,
    *,
    ancestor_count_fontsize=8,
    ancestor_count_margin=0.35,
    ancestor_count_padding=2,
    ancestor_panel_spacing=0.18,
    box_size=(7, 4.5),
    dpi=300,
    jitter_alpha=0.15,
    jitter_max_points=1500,
    jitter_sample_interval_mb=(-1, 1),
    jitter_seed=42,
    jitter_size=5,
    jitter_width=0.32,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_linthresh_bp=1000,
    quantile_line_style="--",
    quantile_line_width=1.2,
    quantile_secondary_color="#555555",
    quantile_secondary_line_style="-.",
    true_color="#b8b8b8",
    version_colors=("#1a80bb", "#ea801c"),
):
    """True and inferred spans per focal site, binned by observed focal DAF."""
    return _ancestor_boxplot(
        data,
        figure_folder,
        "span",
        num_bins,
        cutoffs,
        figsize=figsize,
        y_log=y_log,
        ancestor_count_fontsize=ancestor_count_fontsize,
        ancestor_count_margin=ancestor_count_margin,
        ancestor_count_padding=ancestor_count_padding,
        ancestor_panel_spacing=ancestor_panel_spacing,
        box_size=box_size,
        dpi=dpi,
        jitter_alpha=jitter_alpha,
        jitter_max_points=jitter_max_points,
        jitter_sample_interval_mb=jitter_sample_interval_mb,
        jitter_seed=jitter_seed,
        jitter_size=jitter_size,
        jitter_width=jitter_width,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        overshoot_linthresh_bp=overshoot_linthresh_bp,
        quantile_line_style=quantile_line_style,
        quantile_line_width=quantile_line_width,
        quantile_secondary_color=quantile_secondary_color,
        quantile_secondary_line_style=quantile_secondary_line_style,
        true_color=true_color,
        version_colors=version_colors,
    )


def plot_ancestor_overshoot(
    data,
    figure_folder=None,
    num_bins=5,
    cutoffs=None,
    conf=0.01,
    alpha=None,
    figsize=None,
    y_log=False,
    *,
    ancestor_count_fontsize=8,
    ancestor_count_margin=0.35,
    ancestor_count_padding=2,
    ancestor_panel_spacing=0.18,
    box_size=(7, 4.5),
    dpi=300,
    jitter_alpha=0.15,
    jitter_max_points=1500,
    jitter_sample_interval_mb=(-1, 1),
    jitter_seed=42,
    jitter_size=5,
    jitter_width=0.32,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_linthresh_bp=1000,
    quantile_line_style="--",
    quantile_line_width=1.2,
    quantile_secondary_color="#555555",
    quantile_secondary_line_style="-.",
    true_color="#b8b8b8",
    version_colors=("#1a80bb", "#ea801c"),
):
    """Left/right overshoots in DAF bins; ``y_log=True`` uses a symlog axis."""
    return _ancestor_boxplot(
        data,
        figure_folder,
        "overshoot",
        num_bins,
        cutoffs,
        conf,
        alpha,
        figsize=figsize,
        y_log=y_log,
        ancestor_count_fontsize=ancestor_count_fontsize,
        ancestor_count_margin=ancestor_count_margin,
        ancestor_count_padding=ancestor_count_padding,
        ancestor_panel_spacing=ancestor_panel_spacing,
        box_size=box_size,
        dpi=dpi,
        jitter_alpha=jitter_alpha,
        jitter_max_points=jitter_max_points,
        jitter_sample_interval_mb=jitter_sample_interval_mb,
        jitter_seed=jitter_seed,
        jitter_size=jitter_size,
        jitter_width=jitter_width,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        overshoot_linthresh_bp=overshoot_linthresh_bp,
        quantile_line_style=quantile_line_style,
        quantile_line_width=quantile_line_width,
        quantile_secondary_color=quantile_secondary_color,
        quantile_secondary_line_style=quantile_secondary_line_style,
        true_color=true_color,
        version_colors=version_colors,
    )


def plot_ancestor_span_curves(
    summary,
    dataset,
    title=None,
    figure_folder=None,
    config=None,
    figsize=None,
    y_log=True,
    *,
    dpi=300,
    span_curve_height_ratios=(3, 0.6),
    span_curve_legend_fontsize=11,
    span_curve_limit_factor=1.2,
    span_curve_line_width=1.8,
    span_curve_size=(4.5, 4.8),
    span_curve_title_fontsize=10,
    span_curve_true_color="#8b0000",
    version_colors=("#1a80bb", "#ea801c"),
):
    """One dataset's mean-span curves, with a short focal-count panel below.

    Consume :func:`nb_utils.summarise_ancestor_span_curves`; ``dataset`` must be
    ``Simulation`` or ``1000 Genomes Project``. Counts must match between
    versions (and truth). All lines are solid, with dark red simulation truth.
    Use the complete summary for both calls to share mean-span axis limits.
    """
    if config is None:
        config = nb_utils.AnalysisConfig()
    if figsize is None:
        figsize = span_curve_size
    if title is None:
        title = dataset
    filenames = {
        "Simulation": "ancestor_span_frequency_simulation",
        "1000 Genomes Project": "ancestor_span_frequency_tgp",
    }
    if dataset not in filenames:
        raise ValueError("dataset must be Simulation or 1000 Genomes Project")
    selected = summary.loc[summary.dataset == dataset]
    count_table = selected.pivot(
        index="bin", columns="version", values="num_focal_sites"
    )
    reference = count_table.iloc[:, 0]
    if not count_table.eq(reference, axis="index").all().all():
        raise ValueError(f"Expected matched focal-site counts within {dataset}")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": span_curve_height_ratios},
    )
    versions = list(config.versions)
    colors = list(version_colors)
    if dataset == "Simulation":
        versions.append("True")
        colors.append(span_curve_true_color)
    for version, color in zip(versions, colors):
        rows = selected.loc[selected.version == version].sort_values("bin")
        label = "True" if version == "True" else nb_utils._version_label(version)
        axes[0].plot(
            rows.daf,
            rows.mean_span_mb,
            color=color,
            linestyle="-",
            linewidth=span_curve_line_width,
            label=label,
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
        lower = positive_means.min() / span_curve_limit_factor
        upper = positive_means.max() * span_curve_limit_factor
        axes[0].set_ylim(lower, upper)
    axes[0].set_title(title, fontsize=span_curve_title_fontsize)
    axes[0].set_ylabel("Mean ancestor span (Mb)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=span_curve_legend_fontsize)
    axes[1].set_xlabel("Derived focal allele frequency")
    axes[1].set_xlim(0, 1)
    return save_figure(fig, figure_folder, filenames[dataset], config, dpi=dpi)


def plot_ancestor_overlap_errors(
    summary,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    group_by="bin",
    group_labels=None,
    show_counts=True,
    name="ancestor_overlap_errors",
    show_mismatch_change=False,
    panel_labels=(),
    group_xlabel="Demographic model",
    figure=None,
    split_error_types=True,
    ancestor_error_change_decimals=2,
    ancestor_error_count_margin=0.4,
    ancestor_error_hatch="///",
    ancestor_error_height_ratios=(1, 1, 0.6),
    ancestor_error_label_decimals=3,
    ancestor_error_label_fontsize=8,
    ancestor_error_label_padding=3,
    ancestor_error_legend_fontsize=9,
    ancestor_error_log_numticks=6,
    ancestor_error_log_subs=(1, 2, 5),
    ancestor_error_size=(7, 5.4),
    ancestor_error_type_labels=("True = 0, Observed = 1", "True = 1, Observed = 0"),
    ancestor_error_y_log=True,
    ancestor_mismatch_fraction_label_margin=0.2,
    dpi=300,
    mean_bar_group_width=0.8,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    version_colors=("#1a80bb", "#ea801c"),
):
    """Grouped fractions and counts, with pooled mismatch fraction stacked by direction.

    Consume :func:`nb_utils.summarise_ancestor_overlap_errors`. Matched versions must
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
        config = nb_utils.AnalysisConfig()
    if figsize is None:
        figsize = ancestor_error_size
    count_table = summary.pivot(
        index=group_by, columns="version", values="num_ancestors"
    )
    if group_labels is not None:
        count_table = count_table.reindex(group_labels)
    reference_counts = count_table[config.versions[0]]
    if not count_table.eq(reference_counts, axis="index").all().all():
        raise ValueError("Expected matched focal-site counts between versions")
    panel_count = 3 if show_counts else 2
    height_ratios = ancestor_error_height_ratios[:panel_count]
    fig = (
        plt.figure(figsize=figsize, layout="constrained") if figure is None else figure
    )
    axes = fig.subplots(
        panel_count, 1, sharex=True, gridspec_kw={"height_ratios": height_ratios}
    )
    positions = np.arange(len(count_table))
    width = mean_bar_group_width / len(config.versions)
    for version_index, (version, color) in enumerate(
        zip(config.versions, version_colors)
    ):
        selected = summary.loc[summary.version == version].set_index(group_by)
        selected = selected.reindex(count_table.index)
        offsets = positions - mean_bar_group_width / 2 + width * (version_index + 0.5)
        bars = axes[0].bar(
            offsets,
            selected.error_free_fraction,
            width=width,
            color=color,
            label=nb_utils._version_label(version),
        )
        proportion_labels = [
            f"{value:.{ancestor_error_label_decimals}f}" if np.isfinite(value) else ""
            for value in selected.error_free_fraction
        ]
        axes[0].bar_label(
            bars,
            labels=proportion_labels,
            color=color,
            fontsize=ancestor_error_label_fontsize,
            padding=ancestor_error_label_padding,
        )
        if split_error_types:
            lower_fraction = selected.should_be_1_mismatch_fraction
            upper_fraction = selected.should_be_0_mismatch_fraction
            axes[1].bar(
                offsets, lower_fraction, width=width, facecolor=color, edgecolor=color
            )
            fraction_bars = axes[1].bar(
                offsets,
                upper_fraction,
                width=width,
                bottom=lower_fraction,
                facecolor="white",
                edgecolor=color,
                hatch=ancestor_error_hatch,
            )
        else:
            total_fraction = selected.mismatch_fraction
            fraction_bars = axes[1].bar(
                offsets, total_fraction, width=width, facecolor=color, edgecolor=color
            )
        if show_mismatch_change and version_index == 1:
            reference_fraction = summary.loc[summary.version == config.versions[0]]
            reference_fraction = reference_fraction.set_index(
                group_by
            ).mismatch_fraction
            reference_fraction = reference_fraction.reindex(count_table.index)
            denominator = reference_fraction.where(reference_fraction > 0)
            change = (selected.mismatch_fraction / denominator - 1) * 100
            change_labels = [
                f"{value:+.{ancestor_error_change_decimals}f}%"
                if np.isfinite(value)
                else ""
                for value in change
            ]
            axes[1].bar_label(
                fraction_bars,
                labels=change_labels,
                color=color,
                fontsize=ancestor_error_label_fontsize,
                padding=ancestor_error_label_padding,
            )
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    axes[0].set_ylabel("Proportion of\nerror-free focal sites")
    axes[1].set_ylabel("Pooled mismatch fraction\n(mismatches/site)")
    if show_mismatch_change:
        axes[1].margins(y=ancestor_mismatch_fraction_label_margin)
    if ancestor_error_y_log:
        axes[1].set_yscale("log")
        axes[1].yaxis.set_major_locator(
            mticker.LogLocator(
                subs=ancestor_error_log_subs, numticks=ancestor_error_log_numticks
            )
        )
        axes[1].yaxis.set_major_formatter(
            mticker.LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf))
        )
        axes[1].yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        axes[1].set_ylim(bottom=0)
        axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    if show_counts:
        axes[2].bar(
            positions,
            reference_counts,
            width=mean_bar_group_width,
            facecolor="white",
            edgecolor="black",
        )
        axes[2].set_ylabel("Count")
        axes[2].margins(y=ancestor_error_count_margin)
        axes[2].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))
        axes[2].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    if group_labels is None:
        labels = [
            f"{row.bin_left:g}–{row.bin_right:g}" for row in selected.itertuples()
        ]
        axes[-1].set_xlabel("Observed focal allele frequency")
        rotation = 25
    else:
        labels = [group_labels[group] for group in count_table.index]
        axes[-1].set_xlabel(group_xlabel)
        rotation = 0
    axes[-1].set_xticks(positions, labels, rotation=rotation)
    axes[-1].set_xlim(-0.6, len(positions) - 0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(config.versions),
        frameon=False,
    )
    if split_error_types:
        error_handles = [
            mpatches.Patch(
                facecolor="white", edgecolor="black", hatch=ancestor_error_hatch
            ),
            mpatches.Patch(facecolor="black", edgecolor="black"),
        ]
        axes[1].legend(
            error_handles,
            ancestor_error_type_labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=2,
            fontsize=ancestor_error_legend_fontsize,
        )
    for ax, label in zip(axes, panel_labels):
        ax.text(
            *overshoot_diff_panel_label_position,
            label,
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
            fontsize=overshoot_diff_panel_label_fontsize,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    if figure is not None:
        return fig
    return save_figure(fig, figure_folder, name, config, dpi=dpi)


def plot_ancestor_error_types(
    summary,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    ancestor_error_type_labels=("True = 0, Observed = 1", "True = 1, Observed = 0"),
    curve_size=(6, 3.5),
    dpi=300,
    genotype_color_positions=(0.1, 0.9),
    genotype_transition_cmap="RdYlBu",
):
    """Two directional mismatch-fraction curves from :func:`nb_utils.summarise_ancestor_error_types`."""
    if config is None:
        config = nb_utils.AnalysisConfig()
    if figsize is None:
        figsize = curve_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    colors = plt.get_cmap(genotype_transition_cmap)(genotype_color_positions)
    columns = ["should_be_0_mismatch_fraction", "should_be_1_mismatch_fraction"]
    for column, label, color in zip(columns, ancestor_error_type_labels, colors):
        positive_rates = summary[column].where(summary[column] > 0)
        ax.plot(summary.daf, positive_rates, color=color, label=label)
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Derived focal allele frequency")
    ax.set_ylabel("Pooled mismatch fraction\n(mismatches/site)")
    ax.legend(frameon=False)
    return save_figure(
        fig, figure_folder, "ancestor_overlap_error_types", config, dpi=dpi
    )


def plot_genotype_error_scaling(
    summary,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    curve_points=256,
    curve_size=(6, 3.5),
    dpi=300,
    genotype_color_positions=(0.1, 0.9),
    genotype_scaling_cmap="PRGn",
    genotype_scaling_reference_color="gray",
):
    """Observed diploid genotype error rate versus linear scaling from 1×."""
    if config is None:
        config = nb_utils.AnalysisConfig()
    rates = summary.rates.sort_values("multiplier")
    base = rates.loc[rates.multiplier == 1, "error_rate"].item()
    if figsize is None:
        figsize = curve_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    colors = plt.get_cmap(genotype_scaling_cmap)(genotype_color_positions)
    ax.plot(rates.multiplier, rates.error_rate, "o-", color=colors[0], label="Observed")
    linear_x = np.geomspace(
        rates.multiplier.min(), rates.multiplier.max(), curve_points
    )
    linear_y = base * linear_x
    ax.plot(
        linear_x,
        linear_y,
        "--",
        color=genotype_scaling_reference_color,
        label="Linear scaling from 1×",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(rates.multiplier, [f"{value:g}×" for value in rates.multiplier])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, position: f"{value * 100:.2g}%")
    )
    ax.set_xlabel("Error multiplier")
    ax.set_ylabel("Diploid genotype error rate")
    ax.legend(frameon=False)
    return save_figure(fig, figure_folder, "genotype_error_scaling", config, dpi=dpi)


def plot_genotype_error_transitions(
    summary,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    dpi=300,
    genotype_color_positions=(0.1, 0.9),
    genotype_transition_cmap="RdYlBu",
    transition_multipliers=(1, 10),
    transition_size=(7, 3.5),
):
    """Conditional homozygote-to-heterozygote rates by true DAF at 1× and 10×."""
    if config is None:
        config = nb_utils.AnalysisConfig()
    if figsize is None:
        figsize = transition_size
    fig, axes = plt.subplots(
        1,
        len(transition_multipliers),
        figsize=figsize,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    colors = plt.get_cmap(genotype_transition_cmap)(genotype_color_positions)
    for ax, multiplier in zip(axes[0], transition_multipliers):
        subset = summary.transitions.loc[summary.transitions.multiplier == multiplier]
        for (label, group), color in zip(
            subset.groupby("transition", sort=False), colors
        ):
            positive_rates = group.error_rate.where(group.error_rate > 0)
            ax.plot(group.daf, positive_rates, color=color, label=label)
        ax.set_xlim(0, 1)
        ax.set_yscale("log")
        ax.set_xlabel("True derived allele frequency")
        ax.text(0.04, 0.96, f"{multiplier:g}×", transform=ax.transAxes, va="top")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, position: f"{value * 100:.2g}%")
        )
    selected = summary.transitions.loc[
        summary.transitions.multiplier.isin(transition_multipliers)
    ]
    positive_rates = selected.loc[selected.error_rate > 0, "error_rate"]
    axes[0, 0].set_ylim(positive_rates.min() / 1.2, positive_rates.max() * 1.2)
    axes[0, 0].set_ylabel("Homozygote-to-heterozygote error rate")
    axes[0, -1].legend(frameon=False)
    return save_figure(
        fig, figure_folder, "genotype_error_transitions", config, dpi=dpi
    )


def _profile_label(
    rate,
    by,
    config,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    phase_ser_decimals=2,
):
    if by == "error_profile":
        labels = dict(zip(combined_error_profiles, combined_error_labels))
        return labels[rate]
    if by == "phase_ser":
        if rate == 0:
            return "No errors"
        mantissa, exponent = f"{rate:.{phase_ser_decimals}e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    if by == "space_per_switch":
        return "No errors" if rate == 0 else f"{rate / config.bp_per_mb:g}"
    return f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}"


def _overshoot_colors(rates, cmap_name, color_range=None):
    if color_range is None:
        color_range = (0.25, 0.8)
    colors = plt.get_cmap(cmap_name)(np.linspace(*color_range, len(rates)))
    return dict(zip(rates, colors))


def _overshoot_histogram(
    data,
    frame,
    by,
    rates,
    legend_title,
    name,
    figure_folder,
    reverse_order=False,
    figsize=None,
    legend_fontsize=None,
    mirrored=True,
    panel_labels=False,
    lines=False,
    figure=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    histogram_labelled_legend_anchors=((0.04, 1), (0.98, 1)),
    histogram_line_width=1.2,
    histogram_size=(7, 4.5),
    overshoot_cmaps=("Blues", "Oranges"),
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """Draw shared-bin counts; mirrored=False gives separated upward panels."""
    config = data.config
    binned = nb_utils._overshoot_histogram_counts(
        frame, by, rates, config, histogram_bins=histogram_bins
    )
    bin_edges = binned.bin_edges
    counts = binned.counts
    max_count = max((histogram.max() for histogram in counts.values()))
    if figsize is None:
        figsize = histogram_size
    panel_gap = 0 if mirrored else overshoot_diff_panel_gap
    if figure is None:
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": panel_gap}
        )
    else:
        fig = figure
        axes = fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": panel_gap})
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for row, (ax, version, cmap_name) in enumerate(
        zip(axes, config.versions, overshoot_cmaps)
    ):
        palette = _overshoot_colors(rates, cmap_name)
        for rate in rates if lines else reversed(rates):
            histogram = counts[version, rate].copy()
            if reverse_order:
                histogram[centers >= 0] = 0
            if lines:
                ax.stairs(
                    histogram,
                    bin_edges,
                    baseline=None,
                    color=palette[rate],
                    linewidth=histogram_line_width,
                )
            else:
                ax.bar(
                    bin_edges[:-1],
                    histogram,
                    width=np.diff(bin_edges),
                    align="edge",
                    color=palette[rate],
                    linewidth=0,
                    bottom=0,
                )
        if reverse_order:
            for rate in rates:
                histogram = counts[version, rate].copy()
                histogram[centers < 0] = 0
                ax.bar(
                    bin_edges[:-1],
                    histogram,
                    width=np.diff(bin_edges),
                    align="edge",
                    color=palette[rate],
                    linewidth=0,
                    bottom=0,
                )
        ax.set_yscale("log")
        ax.set_ylim(0.8, max_count * 1.2)
        max_power = int(np.floor(np.log10(max_count)))
        ax.set_yticks([10**power for power in range(1, max_power + 1)])
        ax.axvline(0, color=zero_color, linestyle="--", linewidth=1)
        ax.set_ylabel("Count")
        if row == 1 and mirrored:
            ax.invert_yaxis()
        handles = []
        for rate in rates:
            label = _profile_label(
                rate,
                by,
                config,
                combined_error_labels=combined_error_labels,
                combined_error_profiles=combined_error_profiles,
                phase_ser_decimals=phase_ser_decimals,
            )
            if lines:
                handle = mlines.Line2D(
                    [],
                    [],
                    color=palette[rate],
                    label=label,
                    linewidth=histogram_line_width,
                )
            else:
                handle = mpatches.Patch(color=palette[rate], label=label)
            handles.append(handle)
        annotations_at_top = row == 0 or not mirrored
        legend_location = "upper right" if annotations_at_top else "lower right"
        legend_anchor = None
        title = legend_title
        if panel_labels:
            legend_location = "upper left" if row == 0 else "upper right"
            legend_anchor = histogram_labelled_legend_anchors[row]
            version_heading = nb_utils._version_label(version).replace(" ", "\\ ")
            title = f"$\\mathbf{{{version_heading}}}$" + f"\n{legend_title}"
        ax.legend(
            handles=handles,
            title=title,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            loc=legend_location,
            bbox_to_anchor=legend_anchor,
        )
        if not panel_labels:
            ax.text(
                0.02,
                0.96 if annotations_at_top else 0.04,
                nb_utils._version_label(version),
                transform=ax.transAxes,
                va="top" if annotations_at_top else "bottom",
                ha="left",
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "pad": 4},
            )
        if panel_labels:
            ax.text(
                *overshoot_diff_panel_label_position,
                overshoot_diff_panel_labels[row],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=overshoot_diff_panel_label_fontsize,
            )
    axes[1].set_xlim(bin_edges[0], bin_edges[-1])
    axes[1].set_xlabel("Overshoot (Mb)")
    if figure is not None:
        return fig
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.98)
    return save_figure(fig, figure_folder, f"{name}_hist", config, dpi=dpi)


def _overshoot_diff(
    config,
    frame,
    by,
    rates,
    legend_title,
    name,
    figure_folder,
    figsize=None,
    legend_fontsize=None,
    num_bins=None,
    figure=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_legend_labels=("No errors", "Combined\nerrors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    overshoot_diff_cmap="viridis",
    overshoot_diff_color_range=(0, 0.8),
    overshoot_diff_legend_fontsize=10,
    overshoot_diff_line_width=1.2,
    overshoot_diff_linthresh_pct=1,
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_diff_size=(8.5, 5),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """Step counts above signed percentage count changes on identical Mb bins.

    Reuse :func:`nb_utils._overshoot_histogram_counts`; ``num_bins=None`` uses the
    histogram configuration, while an explicit value overrides the bin count.
    Percentage changes are 100 * (new count - old count) / old count; bins
    with zero old count are undefined and left as gaps. The count axis is log;
    the change axis is symmetric log to retain negative changes and zero.
    """
    binned = nb_utils._overshoot_histogram_counts(
        frame, by, rates, config, num_bins=num_bins, histogram_bins=histogram_bins
    )
    old_version, new_version = config.versions
    if figsize is None:
        figsize = overshoot_diff_size
    fig = (
        plt.figure(figsize=figsize, layout="constrained") if figure is None else figure
    )
    axes = fig.subplots(2, 1, sharex=True)
    palette = _overshoot_colors(rates, overshoot_diff_cmap, overshoot_diff_color_range)
    if figure is None:
        fig.get_layout_engine().set(hspace=overshoot_diff_panel_gap)
    for rate in rates:
        old_counts = binned.counts[old_version, rate]
        new_counts = binned.counts[new_version, rate]
        changes = np.full(old_counts.shape, np.nan)
        np.divide(
            100 * (new_counts - old_counts),
            old_counts,
            out=changes,
            where=old_counts > 0,
        )
        label = _profile_label(
            rate,
            by,
            config,
            combined_error_labels=combined_error_labels,
            combined_error_profiles=combined_error_profiles,
            phase_ser_decimals=phase_ser_decimals,
        )
        visible_counts = np.where(old_counts > 0, old_counts, np.nan)
        axes[0].stairs(
            visible_counts,
            binned.bin_edges,
            baseline=None,
            color=palette[rate],
            label=label,
            linewidth=overshoot_diff_line_width,
        )
        axes[1].stairs(
            changes,
            binned.bin_edges,
            baseline=None,
            color=palette[rate],
            label=label,
            linewidth=overshoot_diff_line_width,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"Observation count\n({nb_utils._version_label(old_version)})")
    axes[1].set_yscale("symlog", linthresh=overshoot_diff_linthresh_pct)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    axes[1].set_ylabel("Percentage increase in\ntsinfer 0.5")
    axes[1].axhline(0, color=zero_color, linestyle="--", linewidth=1)
    for ax, panel_label in zip(axes, overshoot_diff_panel_labels):
        ax.axvline(0, color=zero_color, linestyle="--", linewidth=1)
        ax.text(
            *overshoot_diff_panel_label_position,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=overshoot_diff_panel_label_fontsize,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    if legend_fontsize is None:
        legend_fontsize = overshoot_diff_legend_fontsize
    handles = []
    labels = []
    for rate in rates:
        handles.append(mlines.Line2D([], [], color=palette[rate]))
        labels.append(
            _profile_label(
                rate,
                by,
                config,
                combined_error_labels=combined_error_labels,
                combined_error_profiles=combined_error_profiles,
                phase_ser_decimals=phase_ser_decimals,
            )
        )
    wrapped_title = legend_title.replace(" ", "\n", 1)
    if by == "error_profile":
        wrapped_title = legend_title
        labels = combined_error_legend_labels
    legend = fig.legend(
        handles,
        labels,
        title=wrapped_title,
        loc="outside right center",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        frameon=False,
    )
    if by == "error_profile":
        legend.get_title().set_fontweight("bold")
    axes[1].set_xlim(binned.bin_edges[0], binned.bin_edges[-1])
    axes[1].set_xlabel("Overshoot (Mb)")
    if figure is not None:
        return fig
    return save_figure(fig, figure_folder, f"{name}_diff", config, dpi=dpi)


def plot_combined_overlap_errors(
    summary,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    figure=None,
    ancestor_error_change_decimals=2,
    ancestor_error_count_margin=0.4,
    ancestor_error_hatch="///",
    ancestor_error_height_ratios=(1, 1, 0.6),
    ancestor_error_label_decimals=3,
    ancestor_error_label_fontsize=8,
    ancestor_error_label_padding=3,
    ancestor_error_legend_fontsize=9,
    ancestor_error_log_numticks=6,
    ancestor_error_log_subs=(1, 2, 5),
    ancestor_error_size=(7, 5.4),
    ancestor_error_type_labels=("True = 0, Observed = 1", "True = 1, Observed = 0"),
    ancestor_error_y_log=True,
    ancestor_mismatch_fraction_label_margin=0.2,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    mean_bar_group_width=0.8,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    version_colors=("#1a80bb", "#ea801c"),
):
    """Two accuracy panels, pooled into two named error-profile groups."""
    if config is None:
        config = nb_utils.AnalysisConfig()
    ancestor_error_y_log = False
    labels = dict(zip(combined_error_profiles, combined_error_labels))
    return plot_ancestor_overlap_errors(
        summary,
        figure_folder,
        config,
        figsize,
        group_by="error_profile",
        group_labels=labels,
        group_xlabel="Error profile",
        name="combined_error_overlap_errors",
        show_mismatch_change=True,
        show_counts=False,
        panel_labels=("A", "B"),
        figure=figure,
        ancestor_error_change_decimals=ancestor_error_change_decimals,
        ancestor_error_count_margin=ancestor_error_count_margin,
        ancestor_error_hatch=ancestor_error_hatch,
        ancestor_error_height_ratios=ancestor_error_height_ratios,
        ancestor_error_label_decimals=ancestor_error_label_decimals,
        ancestor_error_label_fontsize=ancestor_error_label_fontsize,
        ancestor_error_label_padding=ancestor_error_label_padding,
        ancestor_error_legend_fontsize=ancestor_error_legend_fontsize,
        ancestor_error_log_numticks=ancestor_error_log_numticks,
        ancestor_error_log_subs=ancestor_error_log_subs,
        ancestor_error_size=ancestor_error_size,
        ancestor_error_type_labels=ancestor_error_type_labels,
        ancestor_error_y_log=ancestor_error_y_log,
        ancestor_mismatch_fraction_label_margin=ancestor_mismatch_fraction_label_margin,
        dpi=dpi,
        mean_bar_group_width=mean_bar_group_width,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        version_colors=version_colors,
    )


def plot_combined_overshoot_diff(
    comparisons,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    num_bins=None,
    legend_fontsize=None,
    figure=None,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_legend_labels=("No errors", "Combined\nerrors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    overshoot_diff_cmap="viridis",
    overshoot_diff_color_range=(0, 0.8),
    overshoot_diff_legend_fontsize=10,
    overshoot_diff_line_width=1.2,
    overshoot_diff_linthresh_pct=1,
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_diff_size=(8.5, 5),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """Usual shared-bin counts and signed changes for `none` versus `all`."""
    if config is None:
        config = nb_utils.AnalysisConfig()
    return _overshoot_diff(
        config,
        comparisons,
        "error_profile",
        combined_error_profiles,
        "Error profile",
        "combined_error_overshoot",
        figure_folder,
        figsize=figsize,
        num_bins=num_bins,
        legend_fontsize=legend_fontsize,
        figure=figure,
        combined_error_labels=combined_error_labels,
        combined_error_legend_labels=combined_error_legend_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        overshoot_diff_cmap=overshoot_diff_cmap,
        overshoot_diff_color_range=overshoot_diff_color_range,
        overshoot_diff_legend_fontsize=overshoot_diff_legend_fontsize,
        overshoot_diff_line_width=overshoot_diff_line_width,
        overshoot_diff_linthresh_pct=overshoot_diff_linthresh_pct,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        overshoot_diff_size=overshoot_diff_size,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )


def plot_combined_errors(
    comparisons,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    num_bins=None,
    ancestor_error_change_decimals=2,
    ancestor_error_count_margin=0.4,
    ancestor_error_hatch="///",
    ancestor_error_height_ratios=(1, 1, 0.6),
    ancestor_error_label_decimals=3,
    ancestor_error_label_fontsize=8,
    ancestor_error_label_padding=3,
    ancestor_error_legend_fontsize=9,
    ancestor_error_log_numticks=6,
    ancestor_error_log_subs=(1, 2, 5),
    ancestor_error_size=(7, 5.4),
    ancestor_error_type_labels=("True = 0, Observed = 1", "True = 1, Observed = 0"),
    ancestor_error_y_log=True,
    ancestor_mismatch_fraction_label_margin=0.2,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_legend_labels=("No errors", "Combined\nerrors"),
    combined_error_profiles=("none", "all"),
    combined_error_size=(11, 5.6),
    combined_error_width_ratios=(0.4, 0.6),
    dpi=300,
    histogram_bins=100,
    mean_bar_group_width=0.8,
    overshoot_diff_cmap="viridis",
    overshoot_diff_color_range=(0, 0.8),
    overshoot_diff_legend_fontsize=10,
    overshoot_diff_line_width=1.2,
    overshoot_diff_linthresh_pct=1,
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_diff_size=(8.5, 5),
    phase_ser_decimals=2,
    version_colors=("#1a80bb", "#ea801c"),
    zero_color="#555555",
):
    """Four-panel thesis figure: 40% accuracy (A–B), 60% overshoot (C–D).

    The standalone counterparts are :func:`plot_combined_overlap_errors`
    and :func:`plot_combined_overshoot_diff`.
    """
    if config is None:
        config = nb_utils.AnalysisConfig()
    if figsize is None:
        figsize = combined_error_size
    fig = plt.figure(figsize=figsize, layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=combined_error_width_ratios)
    summary = nb_utils.summarise_combined_overlap_errors(comparisons)
    plot_combined_overlap_errors(
        summary,
        config=config,
        figure=left,
        ancestor_error_change_decimals=ancestor_error_change_decimals,
        ancestor_error_count_margin=ancestor_error_count_margin,
        ancestor_error_hatch=ancestor_error_hatch,
        ancestor_error_height_ratios=ancestor_error_height_ratios,
        ancestor_error_label_decimals=ancestor_error_label_decimals,
        ancestor_error_label_fontsize=ancestor_error_label_fontsize,
        ancestor_error_label_padding=ancestor_error_label_padding,
        ancestor_error_legend_fontsize=ancestor_error_legend_fontsize,
        ancestor_error_log_numticks=ancestor_error_log_numticks,
        ancestor_error_log_subs=ancestor_error_log_subs,
        ancestor_error_size=ancestor_error_size,
        ancestor_error_type_labels=ancestor_error_type_labels,
        ancestor_error_y_log=ancestor_error_y_log,
        ancestor_mismatch_fraction_label_margin=ancestor_mismatch_fraction_label_margin,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        mean_bar_group_width=mean_bar_group_width,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        version_colors=version_colors,
    )
    plot_combined_overshoot_diff(
        comparisons,
        config=config,
        figure=right,
        num_bins=num_bins,
        combined_error_labels=combined_error_labels,
        combined_error_legend_labels=combined_error_legend_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        overshoot_diff_cmap=overshoot_diff_cmap,
        overshoot_diff_color_range=overshoot_diff_color_range,
        overshoot_diff_legend_fontsize=overshoot_diff_legend_fontsize,
        overshoot_diff_line_width=overshoot_diff_line_width,
        overshoot_diff_linthresh_pct=overshoot_diff_linthresh_pct,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=("C", "D"),
        overshoot_diff_size=overshoot_diff_size,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )
    return save_figure(fig, figure_folder, "combined_errors", config, dpi=dpi)


def _draw_overshoot_group(
    ax,
    values,
    position,
    box_width,
    color,
    rng,
    config,
    conf,
    alpha,
    *,
    jitter_max_points=1500,
    jitter_sample_interval_mb=(-1, 1),
    jitter_size=5,
    jitter_width=0.32,
    quantile_line_style="--",
    quantile_line_width=1.2,
):
    """Sample central dots unless ``jitter_max_points=None``; keep one opacity."""
    ax.boxplot(
        [values],
        positions=[position],
        widths=box_width,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black", "linewidth": 1.2},
        zorder=3,
    )
    low, high = np.quantile(values, [conf, 1 - conf])
    if jitter_max_points is None:
        sampled = values
    else:
        sample_low, sample_high = jitter_sample_interval_mb
        within_interval = (values >= sample_low) & (values <= sample_high)
        central = values[within_interval]
        sample_size = min(len(central), jitter_max_points)
        sampled = rng.choice(central, size=sample_size, replace=False)
        sampled = np.concatenate([sampled, values[~within_interval]])
    jitter = rng.uniform(-jitter_width / 2, jitter_width / 2, len(sampled))
    ax.scatter(
        position + jitter,
        sampled,
        s=jitter_size,
        alpha=alpha,
        color=color,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    ax.hlines(
        [low, high],
        position - jitter_width / 2,
        position + jitter_width / 2,
        color="black",
        linestyle=quantile_line_style,
        linewidth=quantile_line_width,
        zorder=5,
    )


def _symlog_overshoot_axis(ax, config, *, overshoot_linthresh_bp=1000):
    """Use signed decade ticks and a linear region that includes zero."""
    linthresh = overshoot_linthresh_bp / config.bp_per_mb
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())


def _overshoot_zero_line(ax):
    """Match the border, above jitter (2) and below boxes (3)."""
    ax.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=ax.spines["left"].get_linewidth(),
        zorder=2.5,
    )


def _overshoot_legends(
    fig,
    config,
    conf,
    alpha,
    conf_secondary=None,
    *,
    quantile_line_style="--",
    quantile_secondary_color="#555555",
    quantile_secondary_line_style="-.",
    version_colors=("#1a80bb", "#ea801c"),
):
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=color,
            alpha=alpha,
            markeredgecolor="none",
            label=nb_utils._version_label(version),
        )
        for version, color in zip(config.versions, version_colors)
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle=quantile_line_style,
            label=f"{conf * 100:g}% tail quantiles",
        )
    )
    if conf_secondary is not None:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=quantile_secondary_color,
                linestyle=quantile_secondary_line_style,
                label=f"{conf_secondary * 100:g}% tail quantiles",
            )
        )
    fig.legend(handles=handles, frameon=False, loc="outside right center")


def _overshoot_boxplot(
    config,
    frame,
    by,
    rates,
    label,
    name,
    figure_folder,
    conf,
    alpha,
    figsize=None,
    group_labels=None,
    y_log=False,
    conf_secondary=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    jitter_alpha=0.15,
    jitter_max_points=1500,
    jitter_sample_interval_mb=(-1, 1),
    jitter_seed=42,
    jitter_size=5,
    jitter_width=0.32,
    overshoot_box_size=(8.5, 3.5),
    overshoot_box_width_ratio=0.6,
    overshoot_linthresh_bp=1000,
    phase_ser_decimals=2,
    quantile_line_style="--",
    quantile_line_width=1.2,
    quantile_secondary_color="#555555",
    quantile_secondary_line_style="-.",
    version_colors=("#1a80bb", "#ea801c"),
):
    """Grouped boxes of focal-site sides; see `_draw_overshoot_group`.

    ``y_log=True`` uses a symmetric log axis, preserving signed values
    and zeros. The central linear region is set in :class:`nb_utils.FigurePlotConfig`.
    Quartiles, whiskers and tail quantiles precede the axis transformation.
    """
    if not 0 < conf < 0.5:
        raise ValueError("conf must be between 0 and 0.5")
    if conf_secondary is not None and (not 0 < conf_secondary < 0.5):
        raise ValueError("conf_secondary must be between 0 and 0.5")
    if alpha is None:
        alpha = jitter_alpha
    rng = np.random.default_rng(jitter_seed)
    if figsize is None:
        figsize = overshoot_box_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    group_width = 0.8 / len(config.versions)
    box_width = jitter_width * overshoot_box_width_ratio
    for version_index, (version, color) in enumerate(
        zip(config.versions, version_colors)
    ):
        for index, rate in enumerate(rates):
            subset = frame.loc[(frame.version == version) & (frame[by] == rate)]
            if subset.empty:
                raise ValueError(f"No overshoots for {version}, {by}={rate}")
            values = subset.overshoot.to_numpy() / config.bp_per_mb
            position = index - 0.4 + group_width * (version_index + 0.5)
            _draw_overshoot_group(
                ax,
                values,
                position,
                box_width,
                color,
                rng,
                config,
                conf,
                alpha,
                jitter_max_points=jitter_max_points,
                jitter_sample_interval_mb=jitter_sample_interval_mb,
                jitter_size=jitter_size,
                jitter_width=jitter_width,
                quantile_line_style=quantile_line_style,
                quantile_line_width=quantile_line_width,
            )
            if conf_secondary is not None:
                quantiles = np.quantile(values, [conf_secondary, 1 - conf_secondary])
                ax.hlines(
                    quantiles,
                    position - jitter_width / 2,
                    position + jitter_width / 2,
                    color=quantile_secondary_color,
                    linestyle=quantile_secondary_line_style,
                    linewidth=quantile_line_width,
                    zorder=5,
                )
    if y_log:
        _symlog_overshoot_axis(
            ax, config, overshoot_linthresh_bp=overshoot_linthresh_bp
        )
    _overshoot_zero_line(ax)
    ax.set_ylabel("Overshoot (Mb)")
    ax.set_xlabel(label)
    if group_labels is None:
        labels = [
            _profile_label(
                rate,
                by,
                config,
                combined_error_labels=combined_error_labels,
                combined_error_profiles=combined_error_profiles,
                phase_ser_decimals=phase_ser_decimals,
            )
            for rate in rates
        ]
    else:
        labels = [group_labels[rate] for rate in rates]
    ax.set_xticks(range(len(rates)), labels)
    ax.set_xlim(-0.5, len(rates) - 0.5)
    ax.grid(axis="y", alpha=0.15)
    _overshoot_legends(
        fig,
        config,
        conf,
        alpha,
        conf_secondary,
        quantile_line_style=quantile_line_style,
        quantile_secondary_color=quantile_secondary_color,
        quantile_secondary_line_style=quantile_secondary_line_style,
        version_colors=version_colors,
    )
    return save_figure(fig, figure_folder, f"{name}_box", config, dpi=dpi)


def plot_demography_overshoot_box(
    comparisons,
    model_labels,
    figure_folder=None,
    config=None,
    conf=0.01,
    alpha=None,
    figsize=None,
    y_log=False,
    conf_secondary=0.01,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    jitter_alpha=0.15,
    jitter_max_points=1500,
    jitter_sample_interval_mb=(-1, 1),
    jitter_seed=42,
    jitter_size=5,
    jitter_width=0.32,
    overshoot_box_size=(8.5, 3.5),
    overshoot_box_width_ratio=0.6,
    overshoot_linthresh_bp=1000,
    phase_ser_decimals=2,
    quantile_line_style="--",
    quantile_line_width=1.2,
    quantile_secondary_color="#555555",
    quantile_secondary_line_style="-.",
    version_colors=("#1a80bb", "#ea801c"),
):
    """Plot both error-free sides, grouped by model, using the usual boxes.

    ``model_labels`` maps model identifiers to display labels in plot order.
    Load comparisons with :func:`nb_utils.load_demography_comparisons`.
    Set ``y_log=True`` for a signed log axis that retains zero values.
    ``conf_secondary`` adds a second pair of tail quantile caps and legend
    entry without changing the point sample. Set it to None to omit
    the additional caps.
    """
    if config is None:
        config = nb_utils.AnalysisConfig()
    return _overshoot_boxplot(
        config,
        comparisons,
        "model",
        list(model_labels),
        "Demographic model",
        "demography_overshoot",
        figure_folder,
        conf,
        alpha,
        figsize=figsize,
        y_log=y_log,
        group_labels=model_labels,
        conf_secondary=conf_secondary,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        jitter_alpha=jitter_alpha,
        jitter_max_points=jitter_max_points,
        jitter_sample_interval_mb=jitter_sample_interval_mb,
        jitter_seed=jitter_seed,
        jitter_size=jitter_size,
        jitter_width=jitter_width,
        overshoot_box_size=overshoot_box_size,
        overshoot_box_width_ratio=overshoot_box_width_ratio,
        overshoot_linthresh_bp=overshoot_linthresh_bp,
        phase_ser_decimals=phase_ser_decimals,
        quantile_line_style=quantile_line_style,
        quantile_line_width=quantile_line_width,
        quantile_secondary_color=quantile_secondary_color,
        quantile_secondary_line_style=quantile_secondary_line_style,
        version_colors=version_colors,
    )


def plot_demography_overshoot_quantiles(
    comparisons,
    model_labels,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    demography_quantile_cap_size=8,
    demography_quantile_center_limits_kb=(-25, 250),
    demography_quantile_interval_width=5,
    demography_quantile_label_fontsize=9,
    demography_quantile_marker_size=5,
    demography_quantile_probabilities=(0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999),
    demography_quantile_size=(8, 4.5),
    demography_quantile_tail_limits_mb=(-20, 20),
    dpi=300,
    quantile_line_style="--",
    quantile_line_width=1.2,
    version_colors=("#1a80bb", "#ea801c"),
    zero_color="#555555",
):
    """Rows of demographic models, with central and tail quantiles in two columns.

    Use error-free focal-site sides from :func:`nb_utils.load_demography_comparisons`.
    All observations contribute to the quantiles; no observations are sampled.
    The central column shows medians and IQRs in kb. The tail column shows
    1st--99th percentile intervals inside 0.1st--99.9th percentile intervals
    in Mb. Both columns use linear axes shared across demographic models.
    """
    if config is None:
        config = nb_utils.AnalysisConfig()
    if figsize is None:
        figsize = demography_quantile_size
    fig, axes = plt.subplots(
        len(model_labels),
        2,
        figsize=figsize,
        sharex="col",
        squeeze=False,
        layout="constrained",
    )
    positions = np.arange(len(config.versions))[::-1]
    for row, (model, label) in enumerate(model_labels.items()):
        central, tails = axes[row]
        for version, color, position in zip(config.versions, version_colors, positions):
            selected = comparisons.loc[
                (comparisons.model == model) & (comparisons.version == version)
            ]
            if selected.empty:
                raise ValueError(f"No overshoots for {version}, model={model}")
            quantiles = np.quantile(
                selected.overshoot, demography_quantile_probabilities
            )
            outer_low, inner_low, q25, median, q75, inner_high, outer_high = quantiles
            central.plot(
                np.array([q25, q75]) / config.bp_per_kb,
                [position, position],
                color=color,
                linewidth=demography_quantile_interval_width,
                solid_capstyle="butt",
            )
            central.plot(
                median / config.bp_per_kb,
                position,
                marker="o",
                color=color,
                markeredgecolor="white",
                markersize=demography_quantile_marker_size,
            )
            tails.plot(
                np.array([outer_low, outer_high]) / config.bp_per_mb,
                [position, position],
                color=color,
                linewidth=quantile_line_width,
                marker="|",
                markersize=demography_quantile_cap_size,
            )
            tails.plot(
                np.array([inner_low, inner_high]) / config.bp_per_mb,
                [position, position],
                color=color,
                linewidth=demography_quantile_interval_width,
                solid_capstyle="butt",
            )
        central.set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=demography_quantile_label_fontsize,
        )
        central.set_xlim(demography_quantile_center_limits_kb)
        tails.set_xlim(demography_quantile_tail_limits_mb)
        for ax in (central, tails):
            ax.axvline(
                0,
                color=zero_color,
                linewidth=quantile_line_width,
                linestyle=quantile_line_style,
                zorder=0,
            )
            ax.set_yticks([])
            ax.set_ylim(-0.5, len(config.versions) - 0.5)
            ax.tick_params(axis="x", labelsize=demography_quantile_label_fontsize)
            ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0, 0].set_title("A: Median and IQR", loc="left")
    axes[0, 1].set_title("B: Tail quantiles", loc="left")
    axes[-1, 0].set_xlabel("Overshoot (kb)")
    axes[-1, 1].set_xlabel("Overshoot (Mb)")
    version_handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            linewidth=demography_quantile_interval_width,
            label=nb_utils._version_label(version),
        )
        for version, color in zip(config.versions, version_colors)
    ]
    fig.legend(
        handles=version_handles,
        loc="outside upper center",
        ncols=len(config.versions),
        frameon=False,
    )
    return save_figure(
        fig, figure_folder, "demography_overshoot_quantiles", config, dpi=dpi
    )


def _overshoot_tails(
    data,
    frame,
    by,
    rates,
    label,
    name,
    figure_folder,
    figsize=None,
    fraction_above=True,
    show_counts=False,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    mean_bar_group_width=0.8,
    overshoot_cmaps=("Blues", "Oranges"),
    overshoot_tail_size=(8.5, 4.5),
    phase_ser_decimals=2,
    tail_count_height_ratio=0.28,
    tail_legend_gap=0.04,
    tail_legend_width_ratio=0.75,
    version_colors=("#1a80bb", "#ea801c"),
    version_line_styles=("-", "--"),
):
    """Overshoot magnitude fractions with independent panel x ranges.

    Each focal site contributes its larger negative and positive side magnitudes,
    both floored at zero. All focal sites supply the denominator, including
    those with no overshoot of that sign. With ``fraction_above=True``, count
    focal sites strictly above the magnitude cutoff; False counts focal sites
    at or below it (see `nb_utils._tail_steps`).
    ``show_counts`` adds grouped bars of nonzero focal-site counts below each sign.
    """
    config = data.config
    if figsize is None:
        figsize = overshoot_tail_size
    rows = 2 if show_counts else 1
    height_ratios = [1, tail_count_height_ratio] if show_counts else [1]
    fig = plt.figure(figsize=figsize, layout="constrained")
    grid = fig.add_gridspec(
        rows,
        3,
        width_ratios=[1, 1, tail_legend_width_ratio],
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
    for version_index, (version, cmap_name, line_style) in enumerate(
        zip(config.versions, overshoot_cmaps, version_line_styles)
    ):
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
            category = _profile_label(
                rate,
                by,
                config,
                combined_error_labels=combined_error_labels,
                combined_error_profiles=combined_error_profiles,
                phase_ser_decimals=phase_ser_decimals,
            )
            for sign_index, (ax, values) in enumerate(zip(axes, (negative, positive))):
                site_counts[sign_index, version_index, rate_index] = np.count_nonzero(
                    values > 0
                )
                steps = nb_utils._tail_steps(values)
                cutoffs = np.union1d(steps.edges, [0])
                indexes = np.searchsorted(steps.edges, cutoffs, side="right")
                fractions = np.concatenate(([0], steps.fractions))[indexes]
                if fraction_above:
                    fractions = 1 - fractions
                ax.step(
                    cutoffs,
                    fractions,
                    where="post",
                    color=palette[rate],
                    linestyle=line_style,
                )
            handles.append(
                mlines.Line2D(
                    [], [], color=palette[rate], linestyle=line_style, label=category
                )
            )
        version_handles.append(handles)
    for ax, sign in zip(axes, ("Negative", "Positive")):
        ax.set_xlabel(f"{sign} overshoot (Mb)")
        ax.set_yscale("log")
        ax.margins(x=0.02)
        ax.set_ylim(min_fraction * 0.8, 1)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, position: f"{value * 100:g}%")
        )
        ax.grid(axis="y", alpha=0.15)
    if show_counts:
        positions = np.arange(len(rates))
        width = mean_bar_group_width / len(config.versions)
        categories = [
            _profile_label(
                rate,
                by,
                config,
                combined_error_labels=combined_error_labels,
                combined_error_profiles=combined_error_profiles,
                phase_ser_decimals=phase_ser_decimals,
            )
            for rate in rates
        ]
        count_axes = [fig.add_subplot(grid[1, index]) for index in range(2)]
        for sign_index, ax in enumerate(count_axes):
            for version_index, (version, color) in enumerate(
                zip(config.versions, version_colors)
            ):
                offsets = (
                    positions - mean_bar_group_width / 2 + width * (version_index + 0.5)
                )
                ax.bar(
                    offsets,
                    site_counts[sign_index, version_index],
                    width=width,
                    color=color,
                    label=nb_utils._version_label(version),
                )
            ax.set_xticks(positions, categories)
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    direction = "above" if fraction_above else "below"
    separator = "\n" if show_counts else " "
    axes[0].set_ylabel(f"Fraction of focal sites{separator}{direction} cutoff")
    for index, (version, handles) in enumerate(zip(config.versions, version_handles)):
        offset = tail_legend_gap / 2
        anchor_y = 0.5 + offset if index == 0 else 0.5 - offset
        location = "lower center" if index == 0 else "upper center"
        legend = legend_ax.legend(
            handles=handles,
            title=f"{label} ({nb_utils._version_label(version)})",
            loc=location,
            bbox_to_anchor=(0.5, anchor_y),
            borderaxespad=0,
            frameon=True,
        )
        if index == 0:
            legend_ax.add_artist(legend)
    if show_counts:
        legend_ax.add_artist(legend)
        handles, labels = count_axes[0].get_legend_handles_labels()
        legend_ax.legend(
            handles,
            labels,
            title="Focal site counts",
            frameon=False,
            fontsize="small",
            loc="lower center",
        )
    return save_figure(fig, figure_folder, f"{name}_tails", config, dpi=dpi)


def plot_overshoot_summary(
    summary,
    by,
    rates,
    regime_labels,
    name,
    figure_folder=None,
    config=None,
    *,
    dpi=300,
    overshoot_summary_bar_height=0.65,
    overshoot_summary_fraction_margin=0.1,
    overshoot_summary_fraction_ticks=4,
    overshoot_summary_interval_width=5,
    overshoot_summary_label_fontsize=9,
    overshoot_summary_marker_size=5,
    overshoot_summary_percent_decimals=1,
    overshoot_summary_size=(8.5, 5),
    quantile_line_style="--",
    quantile_line_width=1.2,
    version_colors=("#1a80bb", "#ea801c"),
    zero_color="#555555",
):
    """Matched rows of median/IQR intervals and left-aligned negative-tail bars.

    Pass :func:`nb_utils.summarise_overshoots` output. Both panels use the same version
    ordering and colours; their linear x axes are shared across error regimes.
    """
    if config is None:
        config = nb_utils.AnalysisConfig()
    fig, axes = plt.subplots(
        len(rates),
        2,
        figsize=overshoot_summary_size,
        sharex="col",
        squeeze=False,
        layout="constrained",
    )
    positions = np.arange(len(config.versions))[::-1]
    fraction_max = summary.fraction_below_threshold.max()
    fraction_limit = fraction_max * (1 + overshoot_summary_fraction_margin)
    indexed = summary.set_index([by, "version"])
    for row, rate in enumerate(rates):
        central, tail = axes[row]
        for version, color, position in zip(config.versions, version_colors, positions):
            values = indexed.loc[rate, version]
            interval = np.array([values.q25_bp, values.q75_bp]) / config.bp_per_mb
            central.plot(
                interval,
                [position, position],
                color=color,
                linewidth=overshoot_summary_interval_width,
                solid_capstyle="butt",
            )
            central.plot(
                values.median_bp / config.bp_per_mb,
                position,
                marker="o",
                color=color,
                markeredgecolor="white",
                markersize=overshoot_summary_marker_size,
            )
            tail.barh(
                position,
                values.fraction_below_threshold,
                left=0,
                color=color,
                height=overshoot_summary_bar_height,
            )
        central.axvline(
            0,
            color=zero_color,
            linewidth=quantile_line_width,
            linestyle=quantile_line_style,
            zorder=0,
        )
        regime_label = regime_labels[rate]
        if by == "space_per_switch" and rate != 0:
            ser = summary.loc[summary[by] == rate, "phase_ser"].iloc[0]
            coefficient, exponent = f"{ser:.2e}".split("e")
            regime_label += (
                "\n" + f"SER = ${coefficient} \\times 10^{{{int(exponent)}}}$"
            )
        central.set_ylabel(
            regime_label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=overshoot_summary_label_fontsize,
        )
        tail.set_xlim(0, fraction_limit)
        tail.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=overshoot_summary_fraction_ticks)
        )
        tail.xaxis.set_major_formatter(
            mticker.PercentFormatter(
                xmax=1, decimals=overshoot_summary_percent_decimals
            )
        )
        for ax in (central, tail):
            ax.set_yticks([])
            ax.set_ylim(-0.5, len(config.versions) - 0.5)
            ax.tick_params(axis="x", labelsize=overshoot_summary_label_fontsize)
            ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0, 0].set_title("A: Median and IQR", loc="left")
    threshold_mb = config.overshoot_summary_threshold_bp / config.bp_per_mb
    axes[0, 1].set_title(f"B: Overshoot < {threshold_mb:g} Mb", loc="left")
    axes[-1, 0].set_xlabel("Overshoot (Mb)")
    axes[-1, 1].set_xlabel("Fraction of focal-site sides")
    handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            linewidth=overshoot_summary_interval_width,
            label=nb_utils._version_label(version),
        )
        for version, color in zip(config.versions, version_colors)
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncols=len(config.versions),
        frameon=False,
    )
    return save_figure(fig, figure_folder, name, config, dpi=dpi)


def _overshoot_difference_label(difference_bp, config, *, mean_change_decimals=1):
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
    number = f"{value:+.{mean_change_decimals}f}"
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return f"{number}{unit}"


def _overshoot_scatter(
    data,
    frame,
    by,
    rates,
    x_rates,
    xlabel,
    name,
    figure_folder,
    figsize=None,
    x_log=False,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    genotype_scatter_connector_width=0.8,
    genotype_scatter_error_change_offset=(-5, -4),
    mean_change_decimals=1,
    overshoot_diff_line_width=1.2,
    overshoot_scatter_change_fontsize=10,
    overshoot_scatter_change_offset=(-5, 4),
    overshoot_scatter_correlation_decimals=3,
    overshoot_scatter_guide_depth=0.05,
    overshoot_scatter_label_fontsize=10,
    overshoot_scatter_label_padding=3,
    overshoot_scatter_marker_size=5,
    overshoot_scatter_size=(5, 4.5),
    overshoot_scatter_tick_decimals=2,
    overshoot_scatter_vertical_dash=(0, (2, 3)),
    overshoot_scatter_x_margin=0.15,
    overshoot_scatter_y_margin=0.18,
    overshoot_scatter_zero_dash=(0, (6, 3)),
    phase_scatter_change_offset=(-5, 4),
    phase_scatter_guide_depth=0.2,
    phase_ser_decimals=2,
    version_colors=("#1a80bb", "#ea801c"),
    zero_color="#555555",
):
    """Connect mean signed overshoots against error rates, with Pearson r.

    Pool focal-site means with `nb_utils._mean_overshoot_by_profile`. Vertical guides label the error
    profiles below zero. Orange labels give the signed difference in mean
    overshoot (0.5 minus 0.4), using compact bp/kb/Mb units. The x-axis is linear and retains
    the error-free profile at zero. With x_log=True, correlate against log10(x)
    and use a logarithmic axis; supplied x values must then be positive.
    """
    config = data.config
    means = nb_utils._mean_overshoot_by_profile(frame, by, rates)
    x_values = x_rates.reindex(rates)
    x_values = x_values.sort_values()
    correlation_x = np.log10(x_values) if x_log else x_values
    if figsize is None:
        figsize = overshoot_scatter_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    plotted_means = means.unstack("version").reindex(x_values.index) / config.bp_per_mb
    mean_range = plotted_means.max().max() - plotted_means.min().min()
    guide_depth = overshoot_scatter_guide_depth
    if by == "space_per_switch":
        guide_depth = phase_scatter_guide_depth
    guide_label_y = -guide_depth * mean_range
    change_offset = (
        phase_scatter_change_offset
        if by == "space_per_switch"
        else overshoot_scatter_change_offset
    )
    for index, (version, color) in enumerate(zip(config.versions, version_colors)):
        values = means.loc[version].reindex(x_values.index) / config.bp_per_mb
        correlation = np.corrcoef(correlation_x, values)[0, 1]
        label = f"{nb_utils._version_label(version)} (r = {correlation:.{overshoot_scatter_correlation_decimals}f})"
        ax.plot(
            x_values,
            values,
            "o-",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=overshoot_scatter_marker_size,
            label=label,
            zorder=3,
        )
        if index == 1:
            reference = means.loc[config.versions[0]].reindex(x_values.index)
            differences = means.loc[version].reindex(x_values.index) - reference
            connector = None
            if by == "geno_multiplier":
                connector = {
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": genotype_scatter_connector_width,
                    "relpos": (1, 0.5),
                    "shrinkA": 0,
                    "shrinkB": overshoot_scatter_marker_size / 2,
                }
            for (rate, x_value), value, difference in zip(
                x_values.items(), values, differences
            ):
                point_offset = change_offset
                vertical_alignment = "bottom"
                if by == "geno_multiplier" and rate >= 1:
                    point_offset = genotype_scatter_error_change_offset
                    vertical_alignment = "top"
                ax.annotate(
                    _overshoot_difference_label(
                        difference, config, mean_change_decimals=mean_change_decimals
                    ),
                    xy=(x_value, value),
                    xytext=point_offset,
                    textcoords="offset points",
                    color=color,
                    fontsize=overshoot_scatter_change_fontsize,
                    ha="right",
                    va=vertical_alignment,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                    arrowprops=connector,
                )
    for rate, x_value in x_values.items():
        label = (
            f"{rate:g}X"
            if by == "geno_multiplier"
            else _profile_label(
                rate,
                by,
                config,
                combined_error_labels=combined_error_labels,
                combined_error_profiles=combined_error_profiles,
                phase_ser_decimals=phase_ser_decimals,
            )
        )
        if by == "space_per_switch":
            label += " Mb"
        ax.axvline(
            x_value,
            color=zero_color,
            linestyle=overshoot_scatter_vertical_dash,
            linewidth=overshoot_diff_line_width,
            zorder=0,
        )
        ax.annotate(
            label,
            xy=(x_value, guide_label_y),
            xytext=(overshoot_scatter_label_padding, 0),
            textcoords="offset points",
            color=zero_color,
            fontsize=overshoot_scatter_label_fontsize,
            va="top",
            ha="left",
        )
    ax.axhline(
        0,
        color="black",
        linestyle=overshoot_scatter_zero_dash,
        linewidth=overshoot_diff_line_width,
        zorder=1,
    )
    if x_log:
        ax.set_xscale("log")
    ax.set_xticks(x_values.to_numpy())
    if by == "space_per_switch":
        labels = [
            _profile_label(
                value,
                "phase_ser",
                config,
                combined_error_labels=combined_error_labels,
                combined_error_profiles=combined_error_profiles,
                phase_ser_decimals=phase_ser_decimals,
            )
            for value in x_values
        ]
        ax.set_xticklabels(labels)
    else:
        ax.xaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1, decimals=overshoot_scatter_tick_decimals)
        )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.margins(x=overshoot_scatter_x_margin, y=overshoot_scatter_y_margin)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean overshoot (Mb)")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    return save_figure(fig, figure_folder, f"{name}_scatter", config, dpi=dpi)


def plot_genotype_overshoot_hist(
    data,
    figure_folder=None,
    figsize=None,
    legend_fontsize=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    histogram_labelled_legend_anchors=((0.04, 1), (0.98, 1)),
    histogram_line_width=1.2,
    histogram_size=(7, 4.5),
    overshoot_cmaps=("Blues", "Oranges"),
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """Mirrored histograms for pure genotype-error profiles."""
    return _overshoot_histogram(
        data,
        nb_utils._genotype_overshoots(data),
        "geno_multiplier",
        data.config.genotype_multipliers,
        "Error multiplier",
        "genotype_error_overshoot",
        figure_folder,
        reverse_order=True,
        figsize=figsize,
        legend_fontsize=legend_fontsize,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        histogram_labelled_legend_anchors=histogram_labelled_legend_anchors,
        histogram_line_width=histogram_line_width,
        histogram_size=histogram_size,
        overshoot_cmaps=overshoot_cmaps,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )


def plot_genotype_overshoot_diff(
    data,
    figure_folder=None,
    figsize=None,
    legend_fontsize=None,
    num_bins=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_legend_labels=("No errors", "Combined\nerrors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    overshoot_diff_cmap="viridis",
    overshoot_diff_color_range=(0, 0.8),
    overshoot_diff_legend_fontsize=10,
    overshoot_diff_line_width=1.2,
    overshoot_diff_linthresh_pct=1,
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_diff_size=(8.5, 5),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """Genotype counts and percentage changes; see `_overshoot_diff`."""
    return _overshoot_diff(
        data.config,
        nb_utils._genotype_overshoots(data),
        "geno_multiplier",
        data.config.genotype_multipliers,
        "Error multiplier",
        "genotype_error_overshoot",
        figure_folder,
        figsize=figsize,
        legend_fontsize=legend_fontsize,
        num_bins=num_bins,
        combined_error_labels=combined_error_labels,
        combined_error_legend_labels=combined_error_legend_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        overshoot_diff_cmap=overshoot_diff_cmap,
        overshoot_diff_color_range=overshoot_diff_color_range,
        overshoot_diff_legend_fontsize=overshoot_diff_legend_fontsize,
        overshoot_diff_line_width=overshoot_diff_line_width,
        overshoot_diff_linthresh_pct=overshoot_diff_linthresh_pct,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        overshoot_diff_size=overshoot_diff_size,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )


def plot_genotype_overshoot_box(
    data,
    figure_folder=None,
    conf=0.01,
    alpha=None,
    figsize=None,
    y_log=False,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    jitter_alpha=0.15,
    jitter_max_points=1500,
    jitter_sample_interval_mb=(-1, 1),
    jitter_seed=42,
    jitter_size=5,
    jitter_width=0.32,
    overshoot_box_size=(8.5, 3.5),
    overshoot_box_width_ratio=0.6,
    overshoot_linthresh_bp=1000,
    phase_ser_decimals=2,
    quantile_line_style="--",
    quantile_line_width=1.2,
    quantile_secondary_color="#555555",
    quantile_secondary_line_style="-.",
    version_colors=("#1a80bb", "#ea801c"),
):
    """Grouped boxplots for pure genotype-error profiles; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data.config,
        nb_utils._genotype_overshoots(data),
        "geno_multiplier",
        data.config.genotype_multipliers,
        "Error multiplier",
        "genotype_error_overshoot",
        figure_folder,
        conf,
        alpha,
        figsize=figsize,
        y_log=y_log,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        jitter_alpha=jitter_alpha,
        jitter_max_points=jitter_max_points,
        jitter_sample_interval_mb=jitter_sample_interval_mb,
        jitter_seed=jitter_seed,
        jitter_size=jitter_size,
        jitter_width=jitter_width,
        overshoot_box_size=overshoot_box_size,
        overshoot_box_width_ratio=overshoot_box_width_ratio,
        overshoot_linthresh_bp=overshoot_linthresh_bp,
        phase_ser_decimals=phase_ser_decimals,
        quantile_line_style=quantile_line_style,
        quantile_line_width=quantile_line_width,
        quantile_secondary_color=quantile_secondary_color,
        quantile_secondary_line_style=quantile_secondary_line_style,
        version_colors=version_colors,
    )


def plot_genotype_overshoot_tails(
    data,
    figure_folder=None,
    figsize=None,
    fraction_above=True,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    mean_bar_group_width=0.8,
    overshoot_cmaps=("Blues", "Oranges"),
    overshoot_tail_size=(8.5, 4.5),
    phase_ser_decimals=2,
    tail_count_height_ratio=0.28,
    tail_legend_gap=0.04,
    tail_legend_width_ratio=0.75,
    version_colors=("#1a80bb", "#ea801c"),
    version_line_styles=("-", "--"),
):
    """Above/below magnitude cutoff fractions by genotype-error multiplier; see `_overshoot_tails`."""
    return _overshoot_tails(
        data,
        nb_utils._genotype_overshoots(data),
        "geno_multiplier",
        data.config.genotype_multipliers,
        "Error multiplier",
        "genotype_error_overshoot",
        figure_folder,
        figsize=figsize,
        fraction_above=fraction_above,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        mean_bar_group_width=mean_bar_group_width,
        overshoot_cmaps=overshoot_cmaps,
        overshoot_tail_size=overshoot_tail_size,
        phase_ser_decimals=phase_ser_decimals,
        tail_count_height_ratio=tail_count_height_ratio,
        tail_legend_gap=tail_legend_gap,
        tail_legend_width_ratio=tail_legend_width_ratio,
        version_colors=version_colors,
        version_line_styles=version_line_styles,
    )


def _format_daf_axes(ax):
    """Common axes for the simulation and TGP spectra."""
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Derived allele frequency")
    ax.set_ylabel("Site count")


def _plot_daf_counts(
    ax,
    frequencies,
    config,
    weights=None,
    *,
    daf_bins=200,
    daf_smoothing_bins=1,
    **kwargs,
):
    """Shared binning and smoothing for :func:`plot_mispolarisation_daf` and TGP."""
    edges = np.linspace(0, 1, daf_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = np.histogram(frequencies, edges, weights=weights)[0]
    smoothed_counts = ndi.gaussian_filter1d(
        counts.astype(float), daf_smoothing_bins, mode="reflect"
    )
    lines = ax.plot(centers, smoothed_counts, **kwargs)
    return lines[0]


def plot_tgp_daf(
    variants,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    daf_bins=200,
    daf_size=(5, 4.5),
    daf_smoothing_bins=1,
    dpi=300,
    tgp_color="red",
    tgp_legend_label="1000 Genomes Project\nn=3202",
):
    """Plot :func:`nb_utils.load_tgp_daf` counts with the simulation's bins and styling."""
    if config is None:
        config = nb_utils.AnalysisConfig()
    if figsize is None:
        figsize = daf_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    _format_daf_axes(ax)
    frequencies = variants.derived_count.to_numpy() / variants.num_alleles.to_numpy()
    _plot_daf_counts(
        ax,
        frequencies,
        config,
        color=tgp_color,
        label=tgp_legend_label,
        daf_bins=daf_bins,
        daf_smoothing_bins=daf_smoothing_bins,
    )
    ax.legend(loc="upper right", frameon=False)
    return save_figure(fig, figure_folder, "tgp_daf", config, dpi=dpi)


def plot_mispolarisation_daf(
    data,
    figure_folder=None,
    figsize=None,
    frequencies=None,
    tgp_variants=None,
    *,
    combined_daf_legend_fontsize=10,
    combined_daf_legend_x=0.96,
    combined_daf_simulation_legend_y=0.97,
    combined_daf_size=(7, 4.5),
    combined_daf_tgp_legend_y=0.65,
    combined_tgp_legend_label="1000 Genomes\nProject (n=6404)",
    daf_bins=200,
    daf_size=(5, 4.5),
    daf_smoothing_bins=1,
    dpi=300,
    simulation_daf_legend_label="Simulation (n=12,000)",
    tgp_color="red",
):
    """Overlay smoothed site counts per DAF bin after flipping mispolarised sites.

    The stored allele counts precede mispolarisation. The mask is True for
    correctly polarised sites; its complement selects the DAFs to reverse.
    Fixed-width histograms and Gaussian smoothing avoid KDE evaluation costs
    growing with the number of sites. Counts are pooled across replicates,
    including singletons, which are eligible for mispolarisation. Pass counts
    from :func:`nb_utils.mispolarisation_daf_counts` to use current aggregate polarisation.
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
        figsize = daf_size if tgp_variants is None else combined_daf_size
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    _format_daf_axes(ax)
    for rate, color in zip(config.mispol_rates, colors):
        selected = frequencies.loc[frequencies.mispol_rate == rate]
        if set(selected.replicate) != replicates:
            raise ValueError(
                f"Missing exported allele-frequency counts for mispol rate {rate}"
            )
        daf = selected.derived_count.to_numpy() / selected.num_alleles.to_numpy()
        mispolarised = selected.mispolarised.to_numpy(dtype=bool)
        daf[mispolarised] = 1 - daf[mispolarised]
        weights = selected.num_sites.to_numpy(dtype=np.int64)
        _plot_daf_counts(
            ax,
            daf,
            config,
            weights=weights,
            color=color,
            label=f"{rate:g}",
            daf_bins=daf_bins,
            daf_smoothing_bins=daf_smoothing_bins,
        )
    if tgp_variants is None:
        ax.legend(title="Simulation:\nMispol. rate", loc="upper right", frameon=False)
    else:
        simulation_legend = ax.legend(
            title=f"{simulation_daf_legend_label}\nMispol. rate",
            loc="upper right",
            bbox_to_anchor=(combined_daf_legend_x, combined_daf_simulation_legend_y),
            frameon=False,
            alignment="left",
            borderaxespad=0,
            fontsize=combined_daf_legend_fontsize,
            title_fontsize=combined_daf_legend_fontsize,
        )
        ax.add_artist(simulation_legend)
        tgp_frequencies = (
            tgp_variants.derived_count.to_numpy() / tgp_variants.num_alleles.to_numpy()
        )
        empirical_line = _plot_daf_counts(
            ax,
            tgp_frequencies,
            config,
            color=tgp_color,
            label=combined_tgp_legend_label,
            daf_bins=daf_bins,
            daf_smoothing_bins=daf_smoothing_bins,
        )
        ax.legend(
            handles=[empirical_line],
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(combined_daf_legend_x, combined_daf_tgp_legend_y),
            borderaxespad=0,
            fontsize=combined_daf_legend_fontsize,
        )
    return save_figure(fig, figure_folder, "mispolarisation_daf", config, dpi=dpi)


def plot_mispolarisation_overlap_errors(
    summary,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    ancestor_error_change_decimals=2,
    ancestor_error_count_margin=0.4,
    ancestor_error_hatch="///",
    ancestor_error_height_ratios=(1, 1, 0.6),
    ancestor_error_label_decimals=3,
    ancestor_error_label_fontsize=8,
    ancestor_error_label_padding=3,
    ancestor_error_legend_fontsize=9,
    ancestor_error_log_numticks=6,
    ancestor_error_log_subs=(1, 2, 5),
    ancestor_error_size=(7, 5.4),
    ancestor_error_type_labels=("True = 0, Observed = 1", "True = 1, Observed = 0"),
    ancestor_error_y_log=True,
    ancestor_mismatch_fraction_label_margin=0.2,
    dpi=300,
    mean_bar_group_width=0.8,
    mispolarisation_error_bracket_depth=0.06,
    mispolarisation_error_bracket_y=-0.5,
    mispolarisation_error_group_fontsize=9,
    mispolarisation_error_log_subs=(1,),
    mispolarisation_error_rate_label_y=-0.62,
    mispolarisation_error_rates=(0.01, 0.1),
    mispolarisation_error_size=(9.5, 6),
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    version_colors=("#1a80bb", "#ea801c"),
):
    """Accuracy and count panels with paired focal-state groups under each rate."""
    if config is None:
        config = nb_utils.AnalysisConfig()
    ancestor_error_log_subs = mispolarisation_error_log_subs
    ancestor_error_y_log = True
    if figsize is None:
        figsize = mispolarisation_error_size
    labels = {"none": "No mispol."}
    for rate in mispolarisation_error_rates:
        labels[f"{rate:g}_correctly_polarised"] = "Non-mispol.\nfocal site"
        labels[f"{rate:g}_mispolarised"] = "Mispol.\nfocal site"
    fig = plt.figure(figsize=figsize, layout="constrained")
    plot_ancestor_overlap_errors(
        summary,
        config=config,
        group_by="group",
        group_labels=labels,
        group_xlabel="",
        show_mismatch_change=True,
        panel_labels=("A", "B", "C"),
        figure=fig,
        split_error_types=False,
        ancestor_error_change_decimals=ancestor_error_change_decimals,
        ancestor_error_count_margin=ancestor_error_count_margin,
        ancestor_error_hatch=ancestor_error_hatch,
        ancestor_error_height_ratios=ancestor_error_height_ratios,
        ancestor_error_label_decimals=ancestor_error_label_decimals,
        ancestor_error_label_fontsize=ancestor_error_label_fontsize,
        ancestor_error_label_padding=ancestor_error_label_padding,
        ancestor_error_legend_fontsize=ancestor_error_legend_fontsize,
        ancestor_error_log_numticks=ancestor_error_log_numticks,
        ancestor_error_log_subs=ancestor_error_log_subs,
        ancestor_error_size=ancestor_error_size,
        ancestor_error_type_labels=ancestor_error_type_labels,
        ancestor_error_y_log=ancestor_error_y_log,
        ancestor_mismatch_fraction_label_margin=ancestor_mismatch_fraction_label_margin,
        dpi=dpi,
        mean_bar_group_width=mean_bar_group_width,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        version_colors=version_colors,
    )
    ax = fig.axes[-1]
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        mticker.LogLocator(
            subs=mispolarisation_error_log_subs, numticks=ancestor_error_log_numticks
        )
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelsize=mispolarisation_error_group_fontsize)
    y = mispolarisation_error_bracket_y
    depth = mispolarisation_error_bracket_depth
    for index, rate in enumerate(mispolarisation_error_rates):
        left = 1 + 2 * index - mean_bar_group_width / 2
        right = 2 + 2 * index + mean_bar_group_width / 2
        ax.plot(
            [left, left, right, right],
            [y + depth, y, y, y + depth],
            transform=ax.get_xaxis_transform(),
            color="black",
            linewidth=1,
            clip_on=False,
        )
        ax.text(
            (left + right) / 2,
            mispolarisation_error_rate_label_y,
            f"Mispol. rate = {rate:g}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=mispolarisation_error_group_fontsize,
        )
    return save_figure(
        fig, figure_folder, "mispolarisation_overlap_errors", config, dpi=dpi
    )


def plot_mispolarisation_mismatch_by_daf(
    summary,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    y_scale="log",
    ancestor_error_version="0.5.1",
    dpi=300,
    mean_bar_group_width=0.8,
    mispolarisation_mismatch_colors=("#777777", "#ea801c", "#6a3d9a"),
    mispolarisation_mismatch_groups=("none", "mispolarised", "correctly_polarised"),
    mispolarisation_mismatch_height_ratios=(4, 1.2),
    mispolarisation_mismatch_labels=(
        "No error",
        "Mispol focal sites",
        "Non-mispol focal sites",
    ),
    mispolarisation_mismatch_rate=0.1,
    mispolarisation_mismatch_size=(10, 5),
    mispolarisation_mismatch_y_margin=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
):
    """Grouped mismatch-fraction bars in A and focal-site counts in B, on shared DAF bins.

    Use :func:`nb_utils.summarise_mispolarisation_mismatch_by_daf`. Fractions use
    compared inference sites and default to log y;
    the count panel is always logarithmic. Colours follow the same group
    order in both panels: no error, mispol focal sites, non-mispol focal sites.
    """
    if config is None:
        config = nb_utils.AnalysisConfig()
    if figsize is None:
        figsize = mispolarisation_mismatch_size
    if y_scale not in ("linear", "log"):
        raise ValueError('y_scale must be "linear" or "log"')
    fraction_column = "mismatch_fraction"
    groups = mispolarisation_mismatch_groups
    fig, (fraction_ax, count_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize,
        layout="constrained",
        gridspec_kw={"height_ratios": mispolarisation_mismatch_height_ratios},
    )
    width = mean_bar_group_width / len(groups)
    for position, (group, color, label) in enumerate(
        zip(groups, mispolarisation_mismatch_colors, mispolarisation_mismatch_labels)
    ):
        binned = summary.loc[summary.focal_group == group]
        offsets = binned.bin - mean_bar_group_width / 2 + width * (position + 0.5)
        fraction_ax.bar(
            offsets, binned[fraction_column], width=width, color=color, label=label
        )
        count_ax.bar(
            offsets,
            binned.num_ancestors,
            width=width,
            facecolor="white",
            edgecolor=color,
        )
    fraction_ax.set_ylabel("Pooled mismatch fraction\n(mismatches/site)")
    fraction_ax.set_yscale(y_scale)
    if y_scale == "linear":
        fraction_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        maximum = summary[fraction_column].max()
        fraction_ax.set_ylim(0, maximum * (1 + mispolarisation_mismatch_y_margin))
    else:
        fraction_ax.margins(y=mispolarisation_mismatch_y_margin)
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
        ax.text(
            *overshoot_diff_panel_label_position,
            label,
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
            fontsize=overshoot_diff_panel_label_fontsize,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    handles, labels = fraction_ax.get_legend_handles_labels()
    title = f"{nb_utils._version_label(ancestor_error_version)}; mispol. rate = {mispolarisation_mismatch_rate:g}"
    fig.legend(
        handles,
        labels,
        title=title,
        loc="outside upper center",
        ncol=len(groups),
        frameon=False,
    )
    return save_figure(
        fig, figure_folder, "mispolarisation_mismatch_by_daf", config, dpi=dpi
    )


def plot_mispolarisation_overshoot_hist(
    data,
    figure_folder=None,
    frequency="all",
    figsize=None,
    legend_fontsize=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    histogram_labelled_legend_anchors=((0.04, 1), (0.98, 1)),
    histogram_line_width=1.2,
    histogram_size=(7, 4.5),
    overshoot_cmaps=("Blues", "Oranges"),
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """Mirrored histograms at high (≥ 0.5), low (< 0.5), or all true DAFs."""
    return _overshoot_histogram(
        data,
        nb_utils._mispolarisation_overshoots(data, frequency),
        "mispol_rate",
        data.config.mispol_rates,
        "Mispol. rate",
        f"mispolarisation_overshoot_{frequency}_frequency",
        figure_folder,
        figsize=figsize,
        legend_fontsize=legend_fontsize,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        histogram_labelled_legend_anchors=histogram_labelled_legend_anchors,
        histogram_line_width=histogram_line_width,
        histogram_size=histogram_size,
        overshoot_cmaps=overshoot_cmaps,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )


def plot_mispolarisation_overshoot_scatter(
    data,
    figure_folder=None,
    frequency="all",
    figsize=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    genotype_scatter_connector_width=0.8,
    genotype_scatter_error_change_offset=(-5, -4),
    mean_change_decimals=1,
    overshoot_diff_line_width=1.2,
    overshoot_scatter_change_fontsize=10,
    overshoot_scatter_change_offset=(-5, 4),
    overshoot_scatter_correlation_decimals=3,
    overshoot_scatter_guide_depth=0.05,
    overshoot_scatter_label_fontsize=10,
    overshoot_scatter_label_padding=3,
    overshoot_scatter_marker_size=5,
    overshoot_scatter_size=(5, 4.5),
    overshoot_scatter_tick_decimals=2,
    overshoot_scatter_vertical_dash=(0, (2, 3)),
    overshoot_scatter_x_margin=0.15,
    overshoot_scatter_y_margin=0.18,
    overshoot_scatter_zero_dash=(0, (6, 3)),
    phase_scatter_change_offset=(-5, 4),
    phase_scatter_guide_depth=0.2,
    phase_ser_decimals=2,
    version_colors=("#1a80bb", "#ea801c"),
    zero_color="#555555",
):
    """Mean signed overshoot versus mispolarisation rate; see `_overshoot_scatter`."""
    rates = data.config.mispol_rates
    x_rates = pd.Series(rates, index=rates)
    return _overshoot_scatter(
        data,
        nb_utils._mispolarisation_overshoots(data, frequency),
        "mispol_rate",
        rates,
        x_rates,
        "Mispolarisation rate",
        f"mispolarisation_overshoot_{frequency}_frequency",
        figure_folder,
        figsize=figsize,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        genotype_scatter_connector_width=genotype_scatter_connector_width,
        genotype_scatter_error_change_offset=genotype_scatter_error_change_offset,
        mean_change_decimals=mean_change_decimals,
        overshoot_diff_line_width=overshoot_diff_line_width,
        overshoot_scatter_change_fontsize=overshoot_scatter_change_fontsize,
        overshoot_scatter_change_offset=overshoot_scatter_change_offset,
        overshoot_scatter_correlation_decimals=overshoot_scatter_correlation_decimals,
        overshoot_scatter_guide_depth=overshoot_scatter_guide_depth,
        overshoot_scatter_label_fontsize=overshoot_scatter_label_fontsize,
        overshoot_scatter_label_padding=overshoot_scatter_label_padding,
        overshoot_scatter_marker_size=overshoot_scatter_marker_size,
        overshoot_scatter_size=overshoot_scatter_size,
        overshoot_scatter_tick_decimals=overshoot_scatter_tick_decimals,
        overshoot_scatter_vertical_dash=overshoot_scatter_vertical_dash,
        overshoot_scatter_x_margin=overshoot_scatter_x_margin,
        overshoot_scatter_y_margin=overshoot_scatter_y_margin,
        overshoot_scatter_zero_dash=overshoot_scatter_zero_dash,
        phase_scatter_change_offset=phase_scatter_change_offset,
        phase_scatter_guide_depth=phase_scatter_guide_depth,
        phase_ser_decimals=phase_ser_decimals,
        version_colors=version_colors,
        zero_color=zero_color,
    )


def plot_mispolarisation_overshoot_diff(
    data,
    figure_folder=None,
    frequency="all",
    figsize=None,
    legend_fontsize=None,
    num_bins=None,
    focal_site_mispolarised=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_legend_labels=("No errors", "Combined\nerrors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    overshoot_diff_cmap="viridis",
    overshoot_diff_color_range=(0, 0.8),
    overshoot_diff_legend_fontsize=10,
    overshoot_diff_line_width=1.2,
    overshoot_diff_linthresh_pct=1,
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_diff_size=(8.5, 5),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """True-DAF-selected count changes, optionally selecting focal polarisation."""
    selected = nb_utils._mispolarisation_overshoots(data, frequency)
    name = f"mispolarisation_overshoot_{frequency}_frequency"
    if focal_site_mispolarised is not None:
        selected = selected.loc[
            selected.focal_site_mispolarised == focal_site_mispolarised
        ]
        state = "flipped" if focal_site_mispolarised else "correct"
        name = f"mispolarisation_overshoot_{state}_focal"
        if frequency != "all":
            name += f"_{frequency}_frequency"
    rates = data.config.mispol_rates
    if focal_site_mispolarised is True:
        rates = [rate for rate in rates if rate > 0]
    return _overshoot_diff(
        data.config,
        selected,
        "mispol_rate",
        rates,
        "Mispol. rate",
        name,
        figure_folder,
        figsize=figsize,
        legend_fontsize=legend_fontsize,
        num_bins=num_bins,
        combined_error_labels=combined_error_labels,
        combined_error_legend_labels=combined_error_legend_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        overshoot_diff_cmap=overshoot_diff_cmap,
        overshoot_diff_color_range=overshoot_diff_color_range,
        overshoot_diff_legend_fontsize=overshoot_diff_legend_fontsize,
        overshoot_diff_line_width=overshoot_diff_line_width,
        overshoot_diff_linthresh_pct=overshoot_diff_linthresh_pct,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        overshoot_diff_size=overshoot_diff_size,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )


def plot_mispolarisation_overshoot_box(
    data,
    figure_folder=None,
    frequency="all",
    conf=0.01,
    alpha=None,
    figsize=None,
    y_log=False,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    jitter_alpha=0.15,
    jitter_max_points=1500,
    jitter_sample_interval_mb=(-1, 1),
    jitter_seed=42,
    jitter_size=5,
    jitter_width=0.32,
    overshoot_box_size=(8.5, 3.5),
    overshoot_box_width_ratio=0.6,
    overshoot_linthresh_bp=1000,
    phase_ser_decimals=2,
    quantile_line_style="--",
    quantile_line_width=1.2,
    quantile_secondary_color="#555555",
    quantile_secondary_line_style="-.",
    version_colors=("#1a80bb", "#ea801c"),
):
    """Grouped mispolarisation boxplots; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data.config,
        nb_utils._mispolarisation_overshoots(data, frequency),
        "mispol_rate",
        data.config.mispol_rates,
        "Mispolarisation rate",
        f"mispolarisation_overshoot_{frequency}_frequency",
        figure_folder,
        conf,
        alpha,
        figsize=figsize,
        y_log=y_log,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        jitter_alpha=jitter_alpha,
        jitter_max_points=jitter_max_points,
        jitter_sample_interval_mb=jitter_sample_interval_mb,
        jitter_seed=jitter_seed,
        jitter_size=jitter_size,
        jitter_width=jitter_width,
        overshoot_box_size=overshoot_box_size,
        overshoot_box_width_ratio=overshoot_box_width_ratio,
        overshoot_linthresh_bp=overshoot_linthresh_bp,
        phase_ser_decimals=phase_ser_decimals,
        quantile_line_style=quantile_line_style,
        quantile_line_width=quantile_line_width,
        quantile_secondary_color=quantile_secondary_color,
        quantile_secondary_line_style=quantile_secondary_line_style,
        version_colors=version_colors,
    )


def plot_mispolarisation_overshoot_tails(
    data,
    figure_folder=None,
    frequency="all",
    figsize=None,
    fraction_above=True,
    show_counts=False,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    mean_bar_group_width=0.8,
    overshoot_cmaps=("Blues", "Oranges"),
    overshoot_tail_size=(8.5, 4.5),
    phase_ser_decimals=2,
    tail_count_height_ratio=0.28,
    tail_legend_gap=0.04,
    tail_legend_width_ratio=0.75,
    version_colors=("#1a80bb", "#ea801c"),
    version_line_styles=("-", "--"),
):
    """Above/below magnitude cutoff fractions by mispolarisation rate; see `_overshoot_tails`."""
    return _overshoot_tails(
        data,
        nb_utils._mispolarisation_overshoots(data, frequency),
        "mispol_rate",
        data.config.mispol_rates,
        "Mispol. rate",
        f"mispolarisation_overshoot_{frequency}_frequency",
        figure_folder,
        figsize=figsize,
        fraction_above=fraction_above,
        show_counts=show_counts,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        mean_bar_group_width=mean_bar_group_width,
        overshoot_cmaps=overshoot_cmaps,
        overshoot_tail_size=overshoot_tail_size,
        phase_ser_decimals=phase_ser_decimals,
        tail_count_height_ratio=tail_count_height_ratio,
        tail_legend_gap=tail_legend_gap,
        tail_legend_width_ratio=tail_legend_width_ratio,
        version_colors=version_colors,
        version_line_styles=version_line_styles,
    )


def plot_mispolarised_focal_overshoot_hist(
    data,
    figure_folder=None,
    figsize=None,
    legend_fontsize=None,
    *,
    ancestor_error_version="0.5.1",
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    histogram_labelled_legend_anchors=((0.04, 1), (0.98, 1)),
    histogram_line_width=1.2,
    histogram_size=(7, 4.5),
    mispolarised_focal_dac_marker="o",
    mispolarised_focal_dac_marker_max_count=10,
    mispolarised_focal_dac_marker_size=4,
    mispolarised_focal_dac_title="True DAC spectrum\n(mispol. focal sites)",
    mispolarised_focal_dac_y_margin=0.15,
    mispolarised_focal_size=(9, 4.5),
    mispolarised_focal_width_ratios=(0.58, 0.42),
    overshoot_cmaps=("Blues", "Oranges"),
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    phase_ser_decimals=2,
    zero_color="#555555",
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
        figsize = mispolarised_focal_size
    frame = nb_utils._mispolarisation_overshoots(data, frequency="all")
    selected = frame.loc[frame.focal_site_mispolarised]
    rates = [rate for rate in config.mispol_rates if rate > 0]
    fig = plt.figure(figsize=figsize, layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=mispolarised_focal_width_ratios)
    _overshoot_histogram(
        data,
        selected,
        "mispol_rate",
        rates,
        "Mispol. rate",
        "mispolarisation_overshoot_flipped_focal",
        figure_folder,
        figsize=figsize,
        legend_fontsize=legend_fontsize,
        mirrored=False,
        panel_labels=True,
        lines=True,
        figure=left,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        histogram_labelled_legend_anchors=histogram_labelled_legend_anchors,
        histogram_line_width=histogram_line_width,
        histogram_size=histogram_size,
        overshoot_cmaps=overshoot_cmaps,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )
    focal_sites = nb_utils._mispolarised_focal_sites(
        data, ancestor_error_version=ancestor_error_version
    )
    palette = _overshoot_colors(rates, overshoot_cmaps[1])
    ax = right.subplots()
    plot_rates = sorted(rates, reverse=True)
    for rate in plot_rates:
        selected = focal_sites.loc[focal_sites.mispol_rate == rate]
        counts = selected.true_derived_count.value_counts().sort_index()
        marker_indices = np.flatnonzero(
            counts.index <= mispolarised_focal_dac_marker_max_count
        )
        ax.plot(
            counts.index,
            counts,
            color=palette[rate],
            label=f"{rate:g}",
            marker=mispolarised_focal_dac_marker,
            markevery=marker_indices,
            markersize=mispolarised_focal_dac_marker_size,
            linewidth=histogram_line_width,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="True derived allele count",
        ylabel="Focal-site count",
        title=mispolarised_focal_dac_title,
    )
    ax.margins(y=mispolarised_focal_dac_y_margin)
    ax.legend(
        title="Mispol. rate",
        loc="upper center",
        frameon=False,
        fontsize=legend_fontsize,
    )
    ax.text(
        *overshoot_diff_panel_label_position,
        "C",
        transform=ax.transAxes,
        va="top",
        fontweight="bold",
        fontsize=overshoot_diff_panel_label_fontsize,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    return save_figure(
        fig,
        figure_folder,
        "mispolarisation_overshoot_flipped_focal_hist",
        config,
        dpi=dpi,
    )


def plot_mispolarisation_low_count_overshoot_scatter(
    data,
    figure_folder=None,
    max_derived_count=5,
    figsize=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    genotype_scatter_connector_width=0.8,
    genotype_scatter_error_change_offset=(-5, -4),
    mean_change_decimals=1,
    overshoot_diff_line_width=1.2,
    overshoot_scatter_change_fontsize=10,
    overshoot_scatter_change_offset=(-5, 4),
    overshoot_scatter_correlation_decimals=3,
    overshoot_scatter_guide_depth=0.05,
    overshoot_scatter_label_fontsize=10,
    overshoot_scatter_label_padding=3,
    overshoot_scatter_marker_size=5,
    overshoot_scatter_size=(5, 4.5),
    overshoot_scatter_tick_decimals=2,
    overshoot_scatter_vertical_dash=(0, (2, 3)),
    overshoot_scatter_x_margin=0.15,
    overshoot_scatter_y_margin=0.18,
    overshoot_scatter_zero_dash=(0, (6, 3)),
    phase_scatter_change_offset=(-5, 4),
    phase_scatter_guide_depth=0.2,
    phase_ser_decimals=2,
    version_colors=("#1a80bb", "#ea801c"),
    zero_color="#555555",
):
    """Low-count mispolarised means against rate; see `_overshoot_scatter`."""
    selected = nb_utils._mispolarisation_low_count_overshoots(data, max_derived_count)
    rates = [rate for rate in data.config.mispol_rates if rate > 0]
    x_rates = pd.Series(rates, index=rates)
    return _overshoot_scatter(
        data,
        selected,
        "mispol_rate",
        rates,
        x_rates,
        "Mispolarisation rate",
        f"mispolarisation_overshoot_derived_count_le_{max_derived_count}",
        figure_folder,
        figsize=figsize,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        genotype_scatter_connector_width=genotype_scatter_connector_width,
        genotype_scatter_error_change_offset=genotype_scatter_error_change_offset,
        mean_change_decimals=mean_change_decimals,
        overshoot_diff_line_width=overshoot_diff_line_width,
        overshoot_scatter_change_fontsize=overshoot_scatter_change_fontsize,
        overshoot_scatter_change_offset=overshoot_scatter_change_offset,
        overshoot_scatter_correlation_decimals=overshoot_scatter_correlation_decimals,
        overshoot_scatter_guide_depth=overshoot_scatter_guide_depth,
        overshoot_scatter_label_fontsize=overshoot_scatter_label_fontsize,
        overshoot_scatter_label_padding=overshoot_scatter_label_padding,
        overshoot_scatter_marker_size=overshoot_scatter_marker_size,
        overshoot_scatter_size=overshoot_scatter_size,
        overshoot_scatter_tick_decimals=overshoot_scatter_tick_decimals,
        overshoot_scatter_vertical_dash=overshoot_scatter_vertical_dash,
        overshoot_scatter_x_margin=overshoot_scatter_x_margin,
        overshoot_scatter_y_margin=overshoot_scatter_y_margin,
        overshoot_scatter_zero_dash=overshoot_scatter_zero_dash,
        phase_scatter_change_offset=phase_scatter_change_offset,
        phase_scatter_guide_depth=phase_scatter_guide_depth,
        phase_ser_decimals=phase_ser_decimals,
        version_colors=version_colors,
        zero_color=zero_color,
    )


def plot_mispolarisation_low_count_overshoot_diff(
    data,
    figure_folder=None,
    max_derived_count=5,
    figsize=None,
    legend_fontsize=None,
    num_bins=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_legend_labels=("No errors", "Combined\nerrors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    overshoot_diff_cmap="viridis",
    overshoot_diff_color_range=(0, 0.8),
    overshoot_diff_legend_fontsize=10,
    overshoot_diff_line_width=1.2,
    overshoot_diff_linthresh_pct=1,
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_diff_size=(8.5, 5),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """Low-count mispolarised focal-site changes; see `_overshoot_diff`."""
    selected = nb_utils._mispolarisation_low_count_overshoots(data, max_derived_count)
    rates = [rate for rate in data.config.mispol_rates if rate > 0]
    return _overshoot_diff(
        data.config,
        selected,
        "mispol_rate",
        rates,
        "Mispol. rate",
        f"mispolarisation_overshoot_derived_count_le_{max_derived_count}",
        figure_folder,
        figsize=figsize,
        legend_fontsize=legend_fontsize,
        num_bins=num_bins,
        combined_error_labels=combined_error_labels,
        combined_error_legend_labels=combined_error_legend_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        overshoot_diff_cmap=overshoot_diff_cmap,
        overshoot_diff_color_range=overshoot_diff_color_range,
        overshoot_diff_legend_fontsize=overshoot_diff_legend_fontsize,
        overshoot_diff_line_width=overshoot_diff_line_width,
        overshoot_diff_linthresh_pct=overshoot_diff_linthresh_pct,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        overshoot_diff_size=overshoot_diff_size,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )


def plot_phasing_overshoot_hist(
    data,
    figure_folder=None,
    figsize=None,
    legend_fontsize=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    histogram_labelled_legend_anchors=((0.04, 1), (0.98, 1)),
    histogram_line_width=1.2,
    histogram_size=(7, 4.5),
    overshoot_cmaps=("Blues", "Oranges"),
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """Mirrored histograms for pure phase-switch profiles."""
    return _overshoot_histogram(
        data,
        nb_utils._phasing_overshoots(data),
        "space_per_switch",
        data.config.phase_spacings,
        "Mean gap (Mb)",
        "phasing_error_overshoot",
        figure_folder,
        reverse_order=True,
        figsize=figsize,
        legend_fontsize=legend_fontsize,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        histogram_labelled_legend_anchors=histogram_labelled_legend_anchors,
        histogram_line_width=histogram_line_width,
        histogram_size=histogram_size,
        overshoot_cmaps=overshoot_cmaps,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )


def plot_phasing_overshoot_diff(
    data,
    figure_folder=None,
    figsize=None,
    legend_fontsize=None,
    num_bins=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_legend_labels=("No errors", "Combined\nerrors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    histogram_bins=100,
    overshoot_diff_cmap="viridis",
    overshoot_diff_color_range=(0, 0.8),
    overshoot_diff_legend_fontsize=10,
    overshoot_diff_line_width=1.2,
    overshoot_diff_linthresh_pct=1,
    overshoot_diff_panel_gap=0.15,
    overshoot_diff_panel_label_fontsize=12,
    overshoot_diff_panel_label_position=(0.02, 0.96),
    overshoot_diff_panel_labels=("A", "B"),
    overshoot_diff_size=(8.5, 5),
    phase_ser_decimals=2,
    zero_color="#555555",
):
    """Switch-error counts and percentage changes; see `_overshoot_diff`."""
    rates = tuple(sorted(data.phase_settings.phase_ser))
    return _overshoot_diff(
        data.config,
        nb_utils._phasing_overshoots(data),
        "phase_ser",
        rates,
        "Switch error rate",
        "phasing_error_overshoot",
        figure_folder,
        figsize=figsize,
        legend_fontsize=legend_fontsize,
        num_bins=num_bins,
        combined_error_labels=combined_error_labels,
        combined_error_legend_labels=combined_error_legend_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        histogram_bins=histogram_bins,
        overshoot_diff_cmap=overshoot_diff_cmap,
        overshoot_diff_color_range=overshoot_diff_color_range,
        overshoot_diff_legend_fontsize=overshoot_diff_legend_fontsize,
        overshoot_diff_line_width=overshoot_diff_line_width,
        overshoot_diff_linthresh_pct=overshoot_diff_linthresh_pct,
        overshoot_diff_panel_gap=overshoot_diff_panel_gap,
        overshoot_diff_panel_label_fontsize=overshoot_diff_panel_label_fontsize,
        overshoot_diff_panel_label_position=overshoot_diff_panel_label_position,
        overshoot_diff_panel_labels=overshoot_diff_panel_labels,
        overshoot_diff_size=overshoot_diff_size,
        phase_ser_decimals=phase_ser_decimals,
        zero_color=zero_color,
    )


def plot_phasing_overshoot_box(
    data,
    figure_folder=None,
    conf=0.01,
    alpha=None,
    figsize=None,
    y_log=False,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    jitter_alpha=0.15,
    jitter_max_points=1500,
    jitter_sample_interval_mb=(-1, 1),
    jitter_seed=42,
    jitter_size=5,
    jitter_width=0.32,
    overshoot_box_size=(8.5, 3.5),
    overshoot_box_width_ratio=0.6,
    overshoot_linthresh_bp=1000,
    phase_ser_decimals=2,
    quantile_line_style="--",
    quantile_line_width=1.2,
    quantile_secondary_color="#555555",
    quantile_secondary_line_style="-.",
    version_colors=("#1a80bb", "#ea801c"),
):
    """Grouped phase-switch boxplots; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data.config,
        nb_utils._phasing_overshoots(data),
        "space_per_switch",
        data.config.phase_spacings,
        "Mean gap (Mb)",
        "phasing_error_overshoot",
        figure_folder,
        conf,
        alpha,
        figsize=figsize,
        y_log=y_log,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        jitter_alpha=jitter_alpha,
        jitter_max_points=jitter_max_points,
        jitter_sample_interval_mb=jitter_sample_interval_mb,
        jitter_seed=jitter_seed,
        jitter_size=jitter_size,
        jitter_width=jitter_width,
        overshoot_box_size=overshoot_box_size,
        overshoot_box_width_ratio=overshoot_box_width_ratio,
        overshoot_linthresh_bp=overshoot_linthresh_bp,
        phase_ser_decimals=phase_ser_decimals,
        quantile_line_style=quantile_line_style,
        quantile_line_width=quantile_line_width,
        quantile_secondary_color=quantile_secondary_color,
        quantile_secondary_line_style=quantile_secondary_line_style,
        version_colors=version_colors,
    )


def plot_phasing_overshoot_tails(
    data,
    figure_folder=None,
    figsize=None,
    fraction_above=True,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    mean_bar_group_width=0.8,
    overshoot_cmaps=("Blues", "Oranges"),
    overshoot_tail_size=(8.5, 4.5),
    phase_ser_decimals=2,
    tail_count_height_ratio=0.28,
    tail_legend_gap=0.04,
    tail_legend_width_ratio=0.75,
    version_colors=("#1a80bb", "#ea801c"),
    version_line_styles=("-", "--"),
):
    """Above/below magnitude cutoff fractions by mean switch gap; see `_overshoot_tails`."""
    return _overshoot_tails(
        data,
        nb_utils._phasing_overshoots(data),
        "space_per_switch",
        data.config.phase_spacings,
        "Mean gap (Mb)",
        "phasing_error_overshoot",
        figure_folder,
        figsize=figsize,
        fraction_above=fraction_above,
        combined_error_labels=combined_error_labels,
        combined_error_profiles=combined_error_profiles,
        dpi=dpi,
        mean_bar_group_width=mean_bar_group_width,
        overshoot_cmaps=overshoot_cmaps,
        overshoot_tail_size=overshoot_tail_size,
        phase_ser_decimals=phase_ser_decimals,
        tail_count_height_ratio=tail_count_height_ratio,
        tail_legend_gap=tail_legend_gap,
        tail_legend_width_ratio=tail_legend_width_ratio,
        version_colors=version_colors,
        version_line_styles=version_line_styles,
    )


def _draw_phase_pair(
    ax, lefts, widths, states, y, config, *, phase_colors=("#4477aa", "#ee9944")
):
    for haplotype in range(2):
        for state in (False, True):
            selected = states == state
            runs = np.column_stack((lefts[selected], widths[selected]))
            color = phase_colors[haplotype ^ int(state)]
            ax.broken_barh(
                runs, (y + haplotype * 0.3, 0.24), facecolors=color, edgecolors="none"
            )


def plot_phase_switch_illustration(
    phase_runs_path,
    figure_folder=None,
    config=None,
    figsize=None,
    *,
    combined_error_labels=("No errors", "Combined errors"),
    combined_error_profiles=("none", "all"),
    dpi=300,
    phase_colors=("#4477aa", "#ee9944"),
    phase_group_gap=0.45,
    phase_label_fontsize=9,
    phase_right_mb=20,
    phase_row_spacing=0.82,
    phase_ser_decimals=2,
    phase_size=(8.5, 4.5),
):
    """Compare true phase with six exported rows from the 20 Mb simulation."""
    if config is None:
        config = nb_utils.AnalysisConfig()
    runs = pd.read_csv(phase_runs_path)
    rows = (
        runs[["plot_row", "individual", "space_per_switch", "phase_ser"]]
        .drop_duplicates()
        .sort_values("plot_row")
    )
    settings = rows.space_per_switch.drop_duplicates().to_list()
    left = runs.left.min() / config.bp_per_mb
    right = min(phase_right_mb, runs.right.max() / config.bp_per_mb)
    y_positions = []
    if figsize is None:
        figsize = phase_size
    fig, overview = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    for index, row in enumerate(rows.itertuples(index=False)):
        setting = settings.index(row.space_per_switch)
        y = (len(rows) - index - 1) * phase_row_spacing + (
            len(settings) - setting - 1
        ) * phase_group_gap
        y_positions.append(y)
        selected = runs.loc[runs.plot_row == row.plot_row].sort_values("left")
        starts = selected.left.to_numpy() / config.bp_per_mb
        widths = (selected.right - selected.left).to_numpy() / config.bp_per_mb
        _draw_phase_pair(
            overview[0],
            np.array([left]),
            np.array([right - left]),
            np.array([False]),
            y,
            config,
            phase_colors=phase_colors,
        )
        _draw_phase_pair(
            overview[1],
            starts,
            widths,
            selected.phase.to_numpy(),
            y,
            config,
            phase_colors=phase_colors,
        )
        if index == 0 or row.space_per_switch != rows.iloc[index - 1].space_per_switch:
            ser_label = _profile_label(
                row.phase_ser,
                "phase_ser",
                config,
                combined_error_labels=combined_error_labels,
                combined_error_profiles=combined_error_profiles,
                phase_ser_decimals=phase_ser_decimals,
            )
            label = f"SER = {ser_label} ({row.space_per_switch / config.bp_per_mb:g} Mb mean gap)"
            overview[1].text(
                0.03,
                y + 0.65,
                label,
                transform=overview[1].get_yaxis_transform(),
                fontsize=phase_label_fontsize,
                va="bottom",
            )
    for ax, title in zip(overview, ("Correct phase", "Simulated phase")):
        ax.set_xlim(left, right)
        ax.set_ylim(-0.1, max(y_positions) + 1.1)
        ax.set_yticks(np.asarray(y_positions) + 0.27, rows.individual.astype(str))
        ax.set_xlabel("Chromosome 20 position (Mb)")
        ax.set_title(title, fontsize=phase_label_fontsize)
    overview[0].set_ylabel("Individual ID")
    handles = [
        mpatches.Patch(color=color, label=f"True haplotype {index + 1}")
        for index, color in enumerate(phase_colors)
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False)
    return save_figure(fig, figure_folder, "phase_switch_illustration", config, dpi=dpi)


def _circle(
    ax,
    x,
    y,
    allele,
    color,
    edge,
    config,
    radius=None,
    *,
    allele_fontsize=10.8,
    allele_text_offset=0.038,
    circle_radius=0.205,
    focal_color="#cc3838",
    ink_color="#253642",
):
    """A flat allele symbol keeps the small genotype matrix easy to read."""
    if radius is None:
        radius = circle_radius
    ax.add_patch(
        mpatches.Circle(
            (x, y),
            radius,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.3 if edge == focal_color else 0.7,
            zorder=5,
        )
    )
    ax.text(
        x,
        y + allele_text_offset,
        str(allele),
        ha="center",
        va="center",
        fontsize=allele_fontsize,
        color=ink_color,
        zorder=6,
    )


def _first_conflict(
    ax,
    x,
    y,
    color,
    config,
    *,
    bar_height=0.52,
    conflict_line_width=1.8,
    first_conflict_dashes=3,
):
    """Equal solid segments at both ends, shared by the matrix and legend."""
    boundaries = np.linspace(
        y - bar_height / 2, y + bar_height / 2, 2 * first_conflict_dashes
    )
    for start, end in zip(boundaries[::2], boundaries[1::2]):
        (line,) = ax.plot(
            [x, x],
            [start, end],
            color=color,
            lw=conflict_line_width,
            solid_capstyle="butt",
            zorder=8,
        )
        clip = mpatches.Rectangle(
            (x - 0.15, y - bar_height / 2), 0.3, bar_height, transform=ax.transData
        )
        line.set_clip_path(clip)


def _ragged(ax, x, y, color, config, *, bar_height=0.52, conflict_line_width=1.8):
    ys = np.linspace(y - bar_height / 2, y + bar_height / 2, 9)
    xs = x + np.tile([-0.03, 0.03], 5)[: len(ys)]
    (line,) = ax.plot(
        xs, ys, color=color, lw=conflict_line_width, solid_capstyle="butt", zorder=8
    )
    clip = mpatches.Rectangle(
        (x - 0.15, y - bar_height / 2), 0.3, bar_height, transform=ax.transData
    )
    line.set_clip_path(clip)


def _bar(
    ax, left, right, y, color, config, edge="none", hatch=None, *, bar_height=0.52
):
    patch = mpatches.Rectangle(
        (left, y - bar_height / 2),
        right - left,
        bar_height,
        facecolor=color,
        edgecolor=edge,
        linewidth=0.5,
        hatch=hatch,
        zorder=1,
    )
    ax.add_patch(patch)


def _site_bracket(
    ax, sites, label, color, config, *, legend_fontsize=10, site_spacing=1.4
):
    left = min(sites) * site_spacing - 0.31
    right = max(sites) * site_spacing + 0.31
    middle = (left + right) / 2
    y = -1.21
    vertices = [
        (left, y + 0.12),
        (left, y + 0.02),
        (left + 0.06, y),
        (left + 0.15, y),
        (middle - 0.15, y),
        (middle - 0.04, y),
        (middle, y - 0.12),
        (middle + 0.04, y),
        (middle + 0.15, y),
        (right - 0.15, y),
        (right - 0.06, y),
        (right, y + 0.02),
        (right, y + 0.12),
    ]
    codes = [mpath.Path.MOVETO] + [mpath.Path.CURVE4] * 12
    ax.add_patch(
        mpatches.PathPatch(
            mpath.Path(vertices, codes), facecolor="none", edgecolor=color, linewidth=1
        )
    )
    ax.text(
        middle,
        -1.48,
        label,
        color=color,
        ha="center",
        va="bottom",
        fontsize=legend_fontsize,
    )


def _legend(
    ax,
    x,
    config,
    *,
    allele_fontsize=10.8,
    allele_text_offset=0.038,
    bar_height=0.52,
    circle_radius=0.205,
    conflict_color="#ff2626",
    conflict_line_width=1.8,
    contained_color="#8167a6",
    first_conflict_dashes=3,
    focal_color="#cc3838",
    ink_color="#253642",
    legend_fontsize=10,
    outside_color="#d6dfe5",
    sample_color="#748a98",
    version_colors=("#1a80bb", "#ea801c"),
):
    fontsize = legend_fontsize
    ink = ink_color
    ax.text(
        x, -0.05, "Sample set ($S$)", fontsize=fontsize, fontweight="bold", color=ink
    )
    entries = (
        (0.45, sample_color, "Retained (both versions)"),
        (1.0, version_colors[0], "Retained (0.4 only)"),
        (1.55, outside_color, "Outside / removed"),
    )
    for y, color, label in entries:
        _bar(ax, x, x + 0.42, y, color, config, bar_height=bar_height)
        ax.text(x + 0.58, y, label, va="center", color=ink, fontsize=fontsize)
    ax.text(
        x, 2.4, "Consensus conflicts", fontsize=fontsize, fontweight="bold", color=ink
    )
    for y in (2.9, 3.45):
        _bar(ax, x, x + 0.42, y, sample_color, config, bar_height=bar_height)
    _first_conflict(
        ax,
        x + 0.21,
        2.9,
        conflict_color,
        config,
        bar_height=bar_height,
        conflict_line_width=conflict_line_width,
        first_conflict_dashes=first_conflict_dashes,
    )
    ax.text(
        x + 0.58, 2.9, "First disagreement", va="center", fontsize=fontsize, color=ink
    )
    _ragged(
        ax,
        x + 0.21,
        3.45,
        conflict_color,
        config,
        bar_height=bar_height,
        conflict_line_width=conflict_line_width,
    )
    ax.text(x + 0.58, 3.45, "Second: remove", va="center", fontsize=fontsize, color=ink)
    ax.text(
        x,
        4.1,
        "Red: both versions",
        va="center",
        fontsize=fontsize,
        color=conflict_color,
    )
    ax.text(
        x,
        4.6,
        "Orange: 0.5 only",
        va="center",
        fontsize=fontsize,
        color=version_colors[1],
    )
    ax.text(
        x, 5.35, "Highlighted sites", fontsize=fontsize, fontweight="bold", color=ink
    )
    sites = (
        (5.9, focal_color, "Focal Site"),
        (6.45, contained_color, "Within-clade doubleton"),
        (7.0, version_colors[1], "Conflicting-clade doubleton"),
    )
    for y, color, label in sites:
        _circle(
            ax,
            x + 0.21,
            y,
            1,
            color,
            color,
            config,
            radius=0.17,
            allele_fontsize=allele_fontsize,
            allele_text_offset=allele_text_offset,
            circle_radius=circle_radius,
            focal_color=focal_color,
            ink_color=ink_color,
        )
        ax.text(x + 0.58, y, label, va="center", color=ink, fontsize=fontsize)


def plot_ancestor_extension(
    config,
    figure_folder=None,
    *,
    allele_fontsize=10.8,
    allele_text_offset=0.038,
    bar_height=0.52,
    circle_radius=0.205,
    conflict_color="#ff2626",
    conflict_line_width=1.8,
    contained_color="#8167a6",
    dpi=300,
    figsize=(9.0, 4.9),
    first_conflict_dashes=3,
    focal_color="#cc3838",
    focal_ring_width=0.065,
    ink_color="#253642",
    label_fontsize=11,
    legend_fontsize=10,
    legend_width=6.3,
    neutral_allele_color="#d7dce0",
    older_allele_color="#ead4a5",
    outside_color="#d6dfe5",
    outside_gap=0.22,
    row_spacing=0.76,
    sample_color="#748a98",
    site_spacing=1.4,
    version_colors=("#1a80bb", "#ea801c"),
):
    """Display the toy comparison and optionally export PNG and editable SVG.

    Arrows mean that the stopping criterion has not been reached within the
    displayed window. The stop site itself is included in the inferred 0.5
    ancestor. The eight-site example removes b in both versions, then c only
    in 0.5, reaching the halving threshold and leaving 0.4 two sites longer.
    """
    genotypes = np.asarray(config.genotypes, dtype=np.int8)
    counts = genotypes.sum(axis=0)
    old = nb_utils.trace_ancestor(config, config.versions[0])
    new = nb_utils.trace_ancestor(config, config.versions[1])
    focal = config.focal_site - 1
    initial = set(np.flatnonzero(genotypes[:, focal]))
    contained = []
    shared = []
    for site in np.flatnonzero(counts == 2):
        carriers = set(np.flatnonzero(genotypes[:, site]))
        if carriers <= initial:
            contained.append(site + 1)
        elif carriers & initial:
            shared.append(site + 1)
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    site_positions = np.arange(1, genotypes.shape[1] + 1) * site_spacing
    left_x = 0.5 * site_spacing
    last_x = (genotypes.shape[1] + 0.5) * site_spacing
    ys = np.arange(len(config.samples), dtype=float) * row_spacing
    ys[len(initial) :] += outside_gap
    axis_y = ys[-1] + 0.88
    ancestor_ys = (axis_y + 1.06, axis_y + 1.93)
    ax.set(xlim=(-2.55, last_x + legend_width), ylim=(ancestor_ys[-1] + 0.63, -2.15))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.12,
        -0.76,
        "Allele count",
        ha="right",
        va="center",
        color=ink_color,
        fontsize=label_fontsize,
    )
    for site, (x, count) in enumerate(zip(site_positions, counts), start=1):
        color = ink_color
        if site == config.focal_site:
            color = focal_color
        elif site in contained:
            color = contained_color
        elif site in shared:
            color = version_colors[1]
        ax.text(
            x,
            -0.76,
            str(count),
            color=color,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold" if count > counts[focal] else "normal",
        )
    _site_bracket(
        ax,
        [config.focal_site],
        "Focal",
        focal_color,
        config,
        legend_fontsize=legend_fontsize,
        site_spacing=site_spacing,
    )
    _site_bracket(
        ax,
        contained,
        "Within\nclade",
        contained_color,
        config,
        legend_fontsize=legend_fontsize,
        site_spacing=site_spacing,
    )
    _site_bracket(
        ax,
        shared,
        "Conflicting\nclade",
        version_colors[1],
        config,
        legend_fontsize=legend_fontsize,
        site_spacing=site_spacing,
    )
    for sample, (name, row, y) in enumerate(zip(config.samples, genotypes, ys)):
        _bar(ax, left_x, last_x, y, outside_color, config, bar_height=bar_height)
        if sample in initial:
            stop = new.removals.get(sample)
            right = last_x if stop is None else (stop + 1.5) * site_spacing
            _bar(ax, left_x, right, y, sample_color, config, bar_height=bar_height)
            if sample in new.removals and sample not in old.removals:
                _bar(
                    ax,
                    right,
                    last_x,
                    y,
                    version_colors[0],
                    config,
                    bar_height=bar_height,
                )
        ax.text(
            0.18,
            y,
            name,
            ha="right",
            va="center",
            fontsize=label_fontsize + 1,
            fontstyle="italic",
            color=focal_color if sample in initial else ink_color,
            fontweight="bold" if sample in initial else "normal",
        )
        for site, allele in enumerate(row):
            face = neutral_allele_color if allele else "#edf0f2"
            edge = "#aab4bb"
            if allele:
                if site == focal:
                    face, edge = (focal_color, focal_color)
                elif site + 1 in contained:
                    face, edge = (contained_color, contained_color)
                elif site + 1 in shared:
                    face, edge = (version_colors[1], version_colors[1])
                elif counts[site] > counts[focal]:
                    face, edge = (older_allele_color, "#b59a66")
            _circle(
                ax,
                site_positions[site],
                y,
                allele,
                face,
                edge,
                config,
                allele_fontsize=allele_fontsize,
                allele_text_offset=allele_text_offset,
                circle_radius=circle_radius,
                focal_color=focal_color,
                ink_color=ink_color,
            )
        if sample in new.removals:
            specific = sample not in old.removals
            trace = new if specific else old
            removal = trace.removals[sample]
            color = version_colors[1] if specific else conflict_color
            first = max((site for site in trace.marks[sample] if site < removal))
            _first_conflict(
                ax,
                (first + 1.5) * site_spacing,
                y,
                color,
                config,
                bar_height=bar_height,
                conflict_line_width=conflict_line_width,
                first_conflict_dashes=first_conflict_dashes,
            )
            _ragged(
                ax,
                (removal + 1.5) * site_spacing,
                y,
                color,
                config,
                bar_height=bar_height,
                conflict_line_width=conflict_line_width,
            )
    ax.text(
        -0.9,
        (ys[0] + ys[-1]) / 2,
        "Sample haplotype",
        rotation=90,
        ha="center",
        va="center",
        fontsize=label_fontsize,
        color=ink_color,
    )
    ax.plot([left_x, last_x], [axis_y, axis_y], lw=0.8, color="#8e9aa2")
    ax.text(
        0.12,
        axis_y + 0.19,
        "Site",
        ha="right",
        va="center",
        fontsize=label_fontsize,
        color=ink_color,
    )
    for site, x in enumerate(site_positions, start=1):
        ax.plot([x, x], [axis_y - 0.06, axis_y + 0.08], color="#8e9aa2", lw=0.8)
        ax.text(
            x,
            axis_y + 0.3,
            str(site),
            ha="center",
            va="center",
            fontsize=legend_fontsize,
            color=ink_color,
        )
    for trace, y, version, color in zip(
        (old, new), ancestor_ys, config.versions, version_colors
    ):
        stop_x = (trace.end + 0.4) * site_spacing
        _bar(ax, left_x, stop_x, y, color, config, bar_height=bar_height)
        if trace.end == genotypes.shape[1]:
            ax.add_patch(
                mpatches.Polygon(
                    [
                        [stop_x, y - bar_height / 2],
                        [stop_x + 0.32, y],
                        [stop_x, y + bar_height / 2],
                    ],
                    facecolor=color,
                    edgecolor="none",
                    zorder=1,
                )
            )
        else:
            ax.plot([stop_x, stop_x], [y - 0.34, y + 0.34], color=color, lw=1.5)
        short_version = ".".join(version.split(".")[:2])
        ax.text(
            -0.04,
            y,
            f"Inferred ancestor\n({short_version})",
            ha="right",
            va="center",
            fontsize=label_fontsize - 0.5,
            color=color,
            linespacing=1.35,
        )
        for site in range(trace.start, trace.end):
            if site == focal:
                ax.add_patch(
                    mpatches.Circle(
                        (site_positions[site], y),
                        circle_radius + focal_ring_width,
                        facecolor=focal_color,
                        edgecolor="none",
                        zorder=3,
                    )
                )
            _circle(
                ax,
                site_positions[site],
                y,
                trace.haplotype[site],
                "#ffffff",
                "#ffffff",
                config,
                allele_fontsize=allele_fontsize,
                allele_text_offset=allele_text_offset,
                circle_radius=circle_radius,
                focal_color=focal_color,
                ink_color=ink_color,
            )
    legend_x = last_x + 0.62
    _legend(
        ax,
        legend_x,
        config,
        allele_fontsize=allele_fontsize,
        allele_text_offset=allele_text_offset,
        bar_height=bar_height,
        circle_radius=circle_radius,
        conflict_color=conflict_color,
        conflict_line_width=conflict_line_width,
        contained_color=contained_color,
        first_conflict_dashes=first_conflict_dashes,
        focal_color=focal_color,
        ink_color=ink_color,
        legend_fontsize=legend_fontsize,
        outside_color=outside_color,
        sample_color=sample_color,
        version_colors=version_colors,
    )
    if figure_folder is not None:
        folder = pathlib.Path(figure_folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            folder / "ancestor_extension_comparison.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.show()
    plt.close(fig)
    return fig


def _repetition_points(
    ax,
    positions,
    values,
    color,
    config,
    edgecolor="white",
    linewidth=0.35,
    *,
    repetition_alpha=0.7,
    repetition_marker_size=13,
):
    """Use reproducible horizontal jitter so repeated values remain visible."""
    ax.scatter(
        positions,
        values,
        s=repetition_marker_size,
        color=color,
        alpha=repetition_alpha,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=4,
    )


def plot_speedup(
    data,
    config,
    *,
    dpi=300,
    group_width=0.8,
    jitter_seed=42,
    jitter_width=0.32,
    repetition_alpha=0.7,
    repetition_marker_size=13,
    sample_labels={},
    speedup_figsize=(7.5, 3.5),
    speedup_top_margin=0.1,
    stage_colors=("#4477aa", "#228833", "#ccbb44"),
    stage_labels=("Generate ancestors", "Match ancestors", "Match samples"),
):
    """Mean paired per-run speedups, with one jittered point per repetition."""
    elapsed = data.elapsed_time.unstack("version")
    speedup = elapsed[config.versions[0]] / elapsed[config.versions[1]]
    means = speedup.groupby(["num_samples", "command"]).mean()
    sizes = means.index.get_level_values("num_samples").unique()
    x = np.arange(len(sizes))
    width = group_width / len(config.stages)
    rng = np.random.default_rng(jitter_seed)
    fig, ax = plt.subplots(figsize=speedup_figsize, dpi=dpi, layout="constrained")
    for index, (stage, label, color) in enumerate(
        zip(config.stages, stage_labels, stage_colors)
    ):
        offset = (index - (len(config.stages) - 1) / 2) * width
        values = means.xs(stage, level="command").reindex(sizes)
        bars = ax.bar(x + offset, values, width=width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f×", fontsize=8, padding=3)
        for position, size in zip(x + offset, sizes):
            reps = speedup.xs((size, stage), level=("num_samples", "command"))
            jitter = rng.uniform(-jitter_width, jitter_width, len(reps))
            _repetition_points(
                ax,
                position + width * jitter,
                reps,
                "black",
                config,
                repetition_alpha=repetition_alpha,
                repetition_marker_size=repetition_marker_size,
            )
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    sample_labels = [sample_labels.get(size, f"{size:,}") for size in sizes]
    ax.set_xticks(x, sample_labels)
    ax.set_xlabel("Sample size (haplotypes)")
    ax.set_ylabel("Speedup (tsinfer 0.4 / tsinfer 0.5)")
    ax.set_ylim(0, max(1, speedup.max()) * (1 + speedup_top_margin))
    ax.legend(frameon=False, loc="upper left")
    return fig


def plot_elapsed(
    data,
    config,
    *,
    dpi=300,
    elapsed_bar_width=0.65,
    figsize=(9.0, 4.9),
    jitter_seed=42,
    jitter_width=0.32,
    repetition_alpha=0.7,
    repetition_marker_size=13,
    sample_labels={},
    sample_titles={},
    stage_colors=("#4477aa", "#228833", "#ccbb44"),
    stage_labels=("Generate ancestors", "Match ancestors", "Match samples"),
    version_labels=("tsinfer 0.4", "tsinfer 0.5"),
):
    """Mean stage stacks with per-run cumulative times at each stack boundary.

    Total speedup is the mean of paired per-run total-time ratios.
    """
    sizes = data.index.get_level_values("num_samples").unique()
    rng = np.random.default_rng(jitter_seed)
    fig, axes = plt.subplots(
        1, len(sizes), figsize=figsize, dpi=dpi, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for ax, size in zip(axes, sizes):
        runs = data.xs(size, level="num_samples").elapsed_minutes.unstack("command")
        runs = runs.reindex(columns=config.stages)
        cumulative = runs.cumsum(axis=1)
        means = runs.groupby("version").mean().reindex(config.versions)
        positions = np.arange(len(config.versions))
        bottom = np.zeros(len(config.versions))
        jitter_positions = {}
        for position, version in zip(positions, config.versions):
            n = len(runs.xs(version, level="version"))
            jitter = rng.uniform(-jitter_width, jitter_width, n)
            jitter_positions[version] = position + elapsed_bar_width * jitter
        for stage, label, color in zip(config.stages, stage_labels, stage_colors):
            values = means[stage].to_numpy()
            ax.bar(
                positions,
                values,
                bottom=bottom,
                label=label,
                color=color,
                width=elapsed_bar_width,
            )
            bottom += values
            for version in config.versions:
                reps = cumulative.xs(version, level="version")[stage]
                _repetition_points(
                    ax,
                    jitter_positions[version],
                    reps,
                    color,
                    config,
                    edgecolor="black",
                    linewidth=0.6,
                    repetition_alpha=repetition_alpha,
                    repetition_marker_size=repetition_marker_size,
                )
        totals = runs.sum(axis=1).unstack("version")
        speedup = (totals[config.versions[0]] / totals[config.versions[1]]).mean()
        for position, total in zip(positions, bottom):
            annotation = f"{total:,.1f}"
            if position == 1:
                annotation += f"\n({speedup:.2f}×)"
            highest = totals[config.versions[position]].max()
            ax.annotate(
                annotation,
                (position, highest),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        sample_label = sample_labels.get(size, f"{size:,}")
        ax.set_title(sample_titles.get(size, f"{sample_label}\nhaplotypes"))
        ax.set_xticks(positions, version_labels, rotation=20)
        ax.set_ylim(0, totals.to_numpy().max() * 1.18)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    axes[0].set_ylabel("Elapsed time (minutes)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(config.stages),
        frameon=False,
    )
    return fig


def plot_resources(
    data,
    metric,
    config,
    reference_data=None,
    *,
    dpi=300,
    figsize=(9.0, 4.9),
    jitter_seed=42,
    parallelism_reference_cores=192,
    reference_markers=("^", "^"),
    repetition_alpha=0.7,
    repetition_marker_size=13,
    resource_jitter_log10=0.025,
    resource_range_capsize=4,
    resource_range_linewidth=0.65,
    stage_labels=("Generate ancestors", "Match ancestors", "Match samples"),
    version_colors=("#1a80bb", "#ea801c"),
    version_labels=("tsinfer 0.4", "tsinfer 0.5"),
):
    """Mean resource curves with repetition variability.

    Memory retains jittered individual runs on log-log axes. Parallelism uses
    capped min–max error bars on a shared linear y scale, without jitter.
    Isolated triangles mark mean 1KGP resources, using the same error bars.
    """
    ylabel = {
        "memory_gib": "Peak resident memory (GiB)",
        "parallelism": "Average active CPU cores",
    }[metric]
    fig, axes = plt.subplots(
        1,
        len(config.stages),
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    sizes = data.index.get_level_values("num_samples").unique()
    rng = np.random.default_rng(jitter_seed)
    for ax, stage, stage_label in zip(axes, config.stages, stage_labels):
        ax.set_xscale("log")
        for version, version_label, color in zip(
            config.versions, version_labels, version_colors
        ):
            values = data.xs((version, stage), level=("version", "command"))
            means = values[metric].groupby("num_samples").mean().reindex(sizes)
            label = (
                version_label if reference_data is None else f"{version_label} (sim)"
            )
            if metric == "parallelism":
                ranges = values[metric].groupby("num_samples").agg(["min", "max"])
                ranges = ranges.reindex(sizes)
                errors = np.vstack([means - ranges["min"], ranges["max"] - means])
                ax.errorbar(
                    sizes,
                    means.to_numpy(dtype=float),
                    yerr=errors,
                    marker="o",
                    color=color,
                    label=label,
                    capsize=resource_range_capsize,
                    elinewidth=resource_range_linewidth,
                    capthick=resource_range_linewidth,
                )
            else:
                ax.plot(sizes, means, marker="o", color=color, label=label)
                run_sizes = values.index.get_level_values("num_samples").to_numpy()
                jitter = rng.uniform(
                    -resource_jitter_log10, resource_jitter_log10, len(values)
                )
                _repetition_points(
                    ax,
                    run_sizes * 10**jitter,
                    values[metric],
                    color,
                    config,
                    repetition_alpha=repetition_alpha,
                    repetition_marker_size=repetition_marker_size,
                )
        if reference_data is not None:
            for version, label, color, marker in zip(
                config.versions, version_labels, version_colors, reference_markers
            ):
                reference = reference_data.xs(
                    (version, stage), level=("version", "command")
                )
                means = reference[metric].groupby("num_samples").mean()
                if metric == "parallelism":
                    ranges = (
                        reference[metric].groupby("num_samples").agg(["min", "max"])
                    )
                    errors = np.vstack([means - ranges["min"], ranges["max"] - means])
                    ax.errorbar(
                        means.index,
                        means.to_numpy(dtype=float),
                        yerr=errors,
                        marker=marker,
                        linestyle="none",
                        color=color,
                        label=f"{label} (1KGP)",
                        zorder=3,
                        capsize=resource_range_capsize,
                        elinewidth=resource_range_linewidth,
                        capthick=resource_range_linewidth,
                    )
                else:
                    ax.scatter(
                        means.index,
                        means,
                        color=color,
                        marker=marker,
                        label=f"{label} (1KGP)",
                        zorder=3,
                    )
                    run_sizes = reference.index.get_level_values(
                        "num_samples"
                    ).to_numpy()
                    jitter = rng.uniform(
                        -resource_jitter_log10, resource_jitter_log10, len(reference)
                    )
                    _repetition_points(
                        ax,
                        run_sizes * 10**jitter,
                        reference[metric],
                        color,
                        config,
                        repetition_alpha=repetition_alpha,
                        repetition_marker_size=repetition_marker_size,
                    )
        if metric == "parallelism":
            ax.axhline(
                parallelism_reference_cores,
                color="gray",
                linestyle=":",
                label=f"{parallelism_reference_cores} cores",
            )
            ax.set_ylim(bottom=0)
        else:
            ax.set_yscale("log")
        ax.set_xticks(sizes, [f"{size:,}" for size in sizes], rotation=30)
        ax.set_xlabel("Sample size (haplotypes)")
        ax.set_title(stage_label)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    reference_label = f"{parallelism_reference_cores} cores"
    if reference_label in labels:
        index = labels.index(reference_label)
        handles.append(handles.pop(index))
        labels.append(labels.pop(index))
    fig.legend(
        handles, labels, loc="outside upper center", ncol=len(labels), frameon=False
    )
    return fig


def plot_buffering_frequency_overshoots(
    multiplier,
    *,
    buffering_config,
    buffering_figure_folder,
    buffering_overshoots,
    config,
    dpi=300,
    jitter_alpha=0.15,
    jitter_seed=42,
    mean_bar_group_width=0.8,
    frequency_cutoffs=(0, 0.001, 0.003, 0.02, 0.5, 1),
    box_size=(9, 4.5),
    version_colors=("#1a80bb", "#ea801c", "#ff0000"),
    tail_quantile=0.01,
    box_y_log=False,
):
    """Draw the usual overshoot layout for one selected genotype-error profile."""
    frame = buffering_overshoots.loc[
        buffering_overshoots.geno_multiplier == multiplier
    ].copy()
    edges = np.array(frequency_cutoffs, dtype=float)
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(edges[-1], np.inf)
    frame["frequency_bin"] = pd.cut(
        frame.allele_frequency, cut_edges, labels=False, right=False
    )
    num_bins = len(edges) - 1
    counts = (
        frame.loc[frame.side == "left"]
        .groupby(["frequency_bin", "version"], observed=True)
        .size()
        .unstack("version")
    )
    counts = counts.reindex(range(num_bins), fill_value=0)
    reference_counts = counts[config.versions[0]]
    assert counts.eq(reference_counts, axis="index").all().all()
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=box_size,
        layout="constrained",
        gridspec_kw={"height_ratios": [4, 1]},
    )
    rng = np.random.default_rng(jitter_seed)
    width = mean_bar_group_width / len(buffering_config.versions)
    for index, (version, color) in enumerate(
        zip(buffering_config.versions, version_colors)
    ):
        version_rows = frame.loc[frame.version == version]
        for bin_index in range(num_bins):
            rows = version_rows.loc[version_rows.frequency_bin == bin_index]
            if rows.empty:
                continue
            values = rows.overshoot.to_numpy() / config.bp_per_mb
            position = bin_index - mean_bar_group_width / 2
            position = position + width * (index + 0.5)
            _draw_overshoot_group(
                axes[0],
                values,
                position,
                width * 0.85,
                color,
                rng,
                buffering_config,
                tail_quantile,
                jitter_alpha,
                jitter_max_points=1500,
                jitter_sample_interval_mb=(-1, 1),
                jitter_size=5,
                jitter_width=0.32,
                quantile_line_style="--",
                quantile_line_width=1.2,
            )
    _overshoot_legends(
        fig,
        buffering_config,
        tail_quantile,
        jitter_alpha,
        quantile_line_style="--",
        quantile_secondary_color="#555555",
        quantile_secondary_line_style="-.",
        version_colors=version_colors,
    )
    if box_y_log:
        _symlog_overshoot_axis(axes[0], buffering_config, overshoot_linthresh_bp=1000)
    _overshoot_zero_line(axes[0])
    axes[0].set_ylabel("Overshoot (Mb)")
    axes[0].set_title(f"Genotype error multiplier: {multiplier:g}×")
    axes[1].bar(
        np.arange(num_bins),
        reference_counts,
        width=width * 0.85 * 2,
        facecolor="white",
        edgecolor="black",
    )
    axes[1].set_ylabel("Count")
    axes[1].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))
    axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    labels = [f"{left:g}–{right:g}" for left, right in zip(edges[:-1], edges[1:])]
    axes[1].set_xticks(np.arange(num_bins), labels, rotation=25)
    axes[1].set_xlabel("Observed focal allele frequency")
    axes[0].set_xlim(-0.6, num_bins - 0.4)
    fig.savefig(
        buffering_figure_folder / f"overshoot_frequency_genotype_{multiplier:g}x.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(fig)


def plot_buffering_negative_lines(
    metric,
    ylabel,
    proportion=False,
    *,
    buffering_config,
    buffering_figure_folder,
    buffering_line_labels,
    buffering_negative_summary,
    buffering_settings,
    dpi=300,
    line_figsize=(12, 3.8),
    version_colors=("#1a80bb", "#ea801c", "#ff0000"),
    line_marker="o",
    line_line_width=1.8,
    line_tick_rotation=35,
):
    """Compare methods across frequency bins in three genotype-error panels."""
    fig, axes = plt.subplots(
        1,
        len(buffering_settings["box_multipliers"]),
        sharex=True,
        sharey=True,
        figsize=line_figsize,
        layout="constrained",
    )
    bins = np.arange(len(buffering_line_labels))
    for ax, multiplier in zip(axes, buffering_settings["box_multipliers"]):
        profile = buffering_negative_summary.loc[
            buffering_negative_summary.geno_multiplier == multiplier
        ]
        for version, color in zip(buffering_config.versions, version_colors):
            rows = profile.loc[profile.version == version]
            rows = rows.set_index("frequency_bin").reindex(bins)
            label = f"tsinfer {'.'.join(version.split('.')[:2])}"
            ax.plot(
                bins,
                rows[metric],
                color=color,
                label=label,
                marker=line_marker,
                linewidth=line_line_width,
            )
        ax.set_title(f"{multiplier:g}× genotype errors")
        ax.set_xticks(
            bins, buffering_line_labels, rotation=line_tick_rotation, ha="right"
        )
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel(ylabel)
    if proportion:
        axes[0].set_ylim(0, 1)
        axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    else:
        axes[0].set_ylim(bottom=0)
    fig.supxlabel("Observed focal allele frequency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(
        buffering_figure_folder / f"{metric}.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(fig)


def plot_ancestor_errors_by_genotype_error_multiplier(
    *,
    buffering_bar_width,
    buffering_config,
    buffering_error_summary,
    buffering_figure_folder,
    buffering_group_labels,
    buffering_multipliers,
    buffering_positions,
    fontsize=8,
    fontsize_2=9,
    dpi=300,
    error_size=(9, 5.4),
    version_colors=("#1a80bb", "#ea801c", "#ff0000"),
):
    """Ancestor errors by genotype error multiplier. Presentation choices are supplied by the notebook."""
    buffering_error_figure, buffering_error_axes = plt.subplots(
        2, 1, sharex=True, figsize=error_size, layout="constrained"
    )
    for index, (version, color) in enumerate(
        zip(buffering_config.versions, version_colors)
    ):
        rows = buffering_error_summary.loc[buffering_error_summary.version == version]
        rows = rows.set_index("geno_multiplier").reindex(buffering_multipliers)
        offsets = buffering_positions - 0.8 / 2
        offsets = offsets + buffering_bar_width * (index + 0.5)
        label = f"tsinfer {'.'.join(version.split('.')[:2])}"
        bars = buffering_error_axes[0].bar(
            offsets,
            rows.error_free_fraction,
            width=buffering_bar_width,
            color=color,
            label=label,
        )
        buffering_error_axes[0].bar_label(
            bars, fmt=f"%.{3}f", color=color, fontsize=fontsize, padding=3
        )
        buffering_error_axes[1].bar(
            offsets,
            rows.should_be_1_mismatch_fraction,
            width=buffering_bar_width,
            facecolor=color,
            edgecolor=color,
        )
        buffering_error_axes[1].bar(
            offsets,
            rows.should_be_0_mismatch_fraction,
            bottom=rows.should_be_1_mismatch_fraction,
            width=buffering_bar_width,
            facecolor="white",
            edgecolor=color,
            hatch="///",
        )
    buffering_error_axes[0].set_ylim(0, 1)
    buffering_error_axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    buffering_error_axes[0].set_ylabel("Proportion of\nerror-free ancestors")
    buffering_error_axes[1].set_ylabel("Pooled mismatch fraction")
    buffering_error_axes[1].set_yscale("log")
    buffering_error_axes[1].set_xticks(
        buffering_positions, buffering_group_labels.values()
    )
    buffering_error_axes[1].set_xlabel("Genotype error multiplier")
    buffering_method_handles, buffering_method_labels = buffering_error_axes[
        0
    ].get_legend_handles_labels()
    buffering_error_figure.legend(
        buffering_method_handles,
        buffering_method_labels,
        loc="outside upper center",
        ncol=len(buffering_config.versions),
        frameon=False,
    )
    buffering_direction_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///"),
        mpatches.Patch(facecolor="black", edgecolor="black"),
    ]
    buffering_error_axes[1].legend(
        buffering_direction_handles,
        ("True = 0, Observed = 1", "True = 1, Observed = 0"),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        fontsize=fontsize_2,
    )
    buffering_error_figure.savefig(
        buffering_figure_folder / "ancestor_errors_by_genotype_multiplier.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    return buffering_error_figure


def close_figure(figure):
    """Release an already-displayed notebook figure."""
    plt.close(figure)


def plot_extension_tree(
    extension_ts,
    extension_node_ids,
    extension_config,
    *,
    figure_folder=None,
    svg_id="ancestor_extension_trees",
    ink_color="#253642",
    muted_color="#66747d",
    background_color="#f1f4f6",
    font_size=14.5,
    sample_font_size=15.8,
    mutation_font_size=14.8,
    site_font_size=13.8,
    node_colors={"F": "#cc3838", "E": "#8167a6", "G": ("#1a80bb", "#ea801c")[1]},
    label_radius=11,
    descendant_colors={"F": "#cc3838", "E": "#cc3838", "G": ("#1a80bb", "#ea801c")[1]},
    older_color="#b59a66",
    size=(680, 300),
    axis_gap=14,
    plot_interval=(0, 100),
    y_ticks=(0, 2, 4, 6, 8, 10),
    symbol_size=6,
    canvas_color="white",
    site_label_offset=-16,
):
    extension_svg_id = svg_id
    extension_ink = ink_color
    extension_muted = muted_color
    extension_background = background_color
    extension_style = [
        f"#{extension_svg_id} text {{font-family: Arial, sans-serif; fill: {extension_ink}; font-size: {font_size}px;}}",
        f"#{extension_svg_id} .edge {{stroke: {extension_muted}; stroke-width: 1.4;}}",
        f"#{extension_svg_id} .node > .sym {{fill: {extension_muted}; stroke: white; stroke-width: 1;}}",
        f"#{extension_svg_id} .sample > .lab {{font-style: italic; font-size: {sample_font_size}px;}}",
        f"#{extension_svg_id} .mut .sym {{stroke-width: 2;}}",
        f"#{extension_svg_id} .mut .lab {{font-size: {mutation_font_size}px;}}",
        f"#{extension_svg_id} .background path {{fill: {extension_background}; fill-opacity: 1;}}",
        f"#{extension_svg_id} .background path:nth-child(even) {{fill: white;}}",
        f"#{extension_svg_id} .site > .lab {{font-size: {site_font_size}px; text-anchor: middle; font-weight: bold;}}",
    ]
    extension_node_attrs = {}
    for name, color in node_colors.items():
        node_id = extension_node_ids[name]
        extension_node_attrs[node_id] = {"r": label_radius}
        extension_style.extend(
            [
                f"#{extension_svg_id} .n{node_id} > .sym {{fill: white; stroke: {color}; stroke-width: 1.8;}}",
                f"#{extension_svg_id} .n{node_id} > .lab {{fill: {color}; font-weight: bold; text-anchor: middle; dominant-baseline: central;}}",
            ]
        )
    for name, color in descendant_colors.items():
        node_id = extension_node_ids[name]
        extension_style.extend(
            [
                f"#{extension_svg_id} .n{node_id} .node > .edge {{stroke: {color}; stroke-width: 2;}}",
                f"#{extension_svg_id} .n{node_id} .sample > .lab {{fill: {color}; font-weight: bold;}}",
            ]
        )
    extension_color_by_node = {
        extension_node_ids[name]: color for name, color in node_colors.items()
    }
    for mutation in extension_ts.mutations():
        color = extension_color_by_node.get(mutation.node, older_color)
        extension_style.extend(
            [
                f"#{extension_svg_id} .m{mutation.id} > .sym {{stroke: {color};}}",
                f"#{extension_svg_id} .m{mutation.id} > .lab {{fill: {color};}}",
                f"#{extension_svg_id} .site.s{mutation.site} > .lab {{fill: {color};}}",
            ]
        )
    extension_node_labels = {
        extension_node_ids[name]: name for name in extension_config.samples
    }
    for name in node_colors:
        extension_node_labels[extension_node_ids[name]] = name
    extension_mutation_labels = {}
    for mutation in extension_ts.mutations():
        extension_mutation_labels[mutation.id] = str(mutation.site + 1)
    extension_tree_svg = extension_ts.draw_svg(
        size=size,
        canvas_size=(size[0], size[1] + axis_gap),
        x_lim=plot_interval,
        x_label="Genome position (bp)",
        time_scale="time",
        y_axis=True,
        y_gridlines=False,
        y_label="Age (generations ago)",
        y_ticks=y_ticks,
        node_labels=extension_node_labels,
        node_attrs=extension_node_attrs,
        mutation_labels=extension_mutation_labels,
        symbol_size=symbol_size,
        root_svg_attributes={"id": extension_svg_id},
        style="\n".join(extension_style),
        preamble=f'<rect width="100%" height="100%" fill="{canvas_color}"/>',
    )
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    extension_svg_root = ET.fromstring(extension_tree_svg)
    extension_x_axis = extension_svg_root.find(
        ".//{http://www.w3.org/2000/svg}g[@class='x-axis']"
    )
    extension_axis_shift = f"translate(0 {axis_gap})"
    extension_x_axis.set("transform", extension_axis_shift)
    extension_background_group = extension_svg_root.find(
        ".//{http://www.w3.org/2000/svg}g[@class='background']"
    )
    extension_background_group.set("transform", extension_axis_shift)
    extension_circle_classes = {f"n{node_id}" for node_id in extension_node_attrs}
    for group in extension_svg_root.iter("{http://www.w3.org/2000/svg}g"):
        group_classes = set(group.get("class", "").split())
        if group_classes.isdisjoint(extension_circle_classes):
            continue
        label = group.find("{http://www.w3.org/2000/svg}text")
        label.set("transform", "translate(0 0)")
    for site in extension_ts.sites():
        site_group = extension_svg_root.find(
            f".//{{http://www.w3.org/2000/svg}}g[@class='site s{site.id}']"
        )
        site_label = ET.SubElement(
            site_group,
            "{http://www.w3.org/2000/svg}text",
            {"class": "lab", "x": "0", "y": str(site_label_offset)},
        )
        site_label.text = str(site.id + 1)
    extension_tree_svg = ET.tostring(extension_svg_root, encoding="unicode")
    if figure_folder is not None:
        subprocess.run(
            [
                "rsvg-convert",
                "--format=pdf",
                "--output",
                str(pathlib.Path(figure_folder) / "ancestor_extension_trees.pdf"),
            ],
            input=extension_tree_svg,
            text=True,
            check=True,
        )
    ipd.display(ipd.SVG(extension_tree_svg))
    return extension_tree_svg


def show_figure():
    """Display the active notebook figure."""
    plt.show()


def save_notebook_figure(
    figure, path, *, dpi=150, bbox_inches="tight", facecolor="white"
):
    """Save a local exploratory figure at its explicit notebook destination."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
