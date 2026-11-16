"""Chapter 5 notebook figures. Presentation choices are explicit keyword arguments."""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import IPython.display as ipython_display
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mpl_toolkits.axes_grid1 as axes_grid1
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
import scipy.stats as stats

from . import nb_utils

logger = logging.getLogger(__name__)


def _cohort_dataset_cutoff(data, dataset, cutoff):
    """Use each cohort's saved setting in a single-setting panel row."""
    return dataset["runs"][0]["af_cutoff"] if cutoff is None else cutoff


def _cohort_dataset_label(data, dataset, *, cohort_labels={}):
    cohort_labels = {
        row.dataset: cohort_labels.get(row.dataset, f"{row.num_samples:,}")
        for row in data.cohorts.itertuples()
    }
    return cohort_labels[dataset["name"]]


def _cohort_method_label(
    data,
    method,
    *,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
):
    if method_colors is not None:
        return method_labels[method]
    return "Full HMM" if method == "full_hmm" else "Stitched"


def _cohort_dataset_title(
    data,
    dataset,
    cutoff,
    *,
    cohort_labels={},
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
):
    samples = len(dataset["cohort"]) // 2
    return f"{_cohort_dataset_label(data, dataset, cohort_labels=cohort_labels)} · {_cohort_cutoff_label(data, cutoff, samples, method_colors=method_colors)}"


def _cohort_cohort_curve(
    data,
    ax,
    curve,
    metric,
    color,
    label,
    sem=False,
    *,
    line_width=1.7,
    marker="o",
    marker_size=5,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    real_data_marker="o",
):
    """Connect cohorts by method, or separate real data in AF comparisons."""
    if method_colors is not None:
        values = curve.sort_values("num_samples")
        sem_column = (
            "sem" if metric == "mean" else metric.removesuffix("_mean") + "_sem"
        )
        yerr = values[sem_column] if sem else None
        ax.errorbar(
            values.num_samples,
            values[metric],
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=marker_size,
            linewidth=line_width,
            capsize=3,
            label=label,
        )
        return
    truth = data.cohorts.set_index("dataset").has_truth
    simulated = curve.dataset.map(truth)
    for is_simulation in [True, False]:
        values = curve.loc[simulated.eq(is_simulation)].sort_values("num_samples")
        if len(values) == 0:
            continue
        sem_column = (
            "sem" if metric == "mean" else metric.removesuffix("_mean") + "_sem"
        )
        yerr = values[sem_column] if sem else None
        marker = "o" if is_simulation else real_data_marker
        ax.errorbar(
            values.num_samples,
            values[metric],
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=marker_size,
            linewidth=line_width,
            capsize=3,
            linestyle="-" if is_simulation or not truth.any() else "none",
            label=label,
        )
        label = None


def _cohort_cutoff_label(
    data,
    cutoff: float | None,
    samples: int | None = None,
    *,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
) -> str:
    if cutoff is None:
        return "Cohort-specific AF" if method_colors is None else ""
    if samples is None:
        sizes = []
        for dataset, size in zip(data.datasets, data.sample_sizes, strict=True):
            if any((run["af_cutoff"] == cutoff for run in dataset["runs"])):
                sizes.append(size)
    else:
        sizes = [samples]
    counts = [str(round(2 * int(size) * cutoff)) for size in sizes]
    return f"AF ≤ {cutoff:g} [AC {' / '.join(counts)}]"


def _cohort_panels(
    data, rows=1, columns=2, title=None, *, panel_height=3.6, panel_width=4.4
):
    figure, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        layout="constrained",
        figsize=(columns * panel_width, rows * panel_height),
    )
    if title is not None:
        figure.suptitle(title)
    for ax in axes.flat:
        ax.tick_params(axis="y", labelleft=True)
    return (figure, axes)


def _cohort_sample_axis(
    data,
    ax,
    datasets=None,
    *,
    cohort_labels={},
    sample_tick_alignments={
        "tgp_chr20p_n3202": "right",
        "out_of_africa_n5000_28_1mbp_error": "left",
    },
    sample_tick_rotation=0,
):
    cohort_labels = {
        row.dataset: cohort_labels.get(row.dataset, f"{row.num_samples:,}")
        for row in data.cohorts.itertuples()
    }
    sample_labels = [cohort_labels[name] for name in data.cohorts.dataset]
    if datasets is None:
        selected = np.ones(len(data.cohorts), dtype=bool)
    else:
        selected = data.cohorts.dataset.isin(datasets)
    ax.set_xscale("log")
    labels = np.asarray(sample_labels)[selected]
    ax.set_xticks(data.sample_sizes[selected], labels=labels)
    rotation = sample_tick_rotation
    alignments = sample_tick_alignments
    names = data.cohorts.loc[selected, "dataset"]
    for label, name in zip(ax.get_xticklabels(), names, strict=True):
        label.set_rotation(rotation)
        default_alignment = "center" if rotation == 0 else "right"
        label.set_horizontalalignment(alignments.get(name, default_alignment))
    ax.minorticks_off()
    ax.set_xlabel("Samples (diploid individuals)")


def _cohort_zoom_y(data, axes, lower=None, upper=None):
    axes = list(axes)
    lows, highs = zip(*(ax.get_ylim() for ax in axes), strict=True)
    minimum = min(lows)
    maximum = max(highs)
    if lower is not None:
        minimum = max(lower, minimum)
    if upper is not None:
        maximum = min(upper, maximum)
    for ax in axes:
        ax.set_ylim(minimum, maximum)


def save_cohort_figure(data, figure, name, *, export_dpi=180):
    root = data.config["cache_dir"]
    figure.savefig(root / f"{name}.png", dpi=export_dpi, bbox_inches="tight")
    figure.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(figure)


def plot_scaling(
    data,
    table,
    metrics,
    title,
    include_full=False,
    sem=False,
    reference=None,
    log_y=False,
    zoom=False,
    *,
    cohort_labels={},
    cutoff_colors=["#0072B2", "#D55E00"],
    full_color="#555555",
    line_width=1.7,
    marker="o",
    marker_size=5,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
    range_margin=0.08,
    real_data_marker="o",
    sample_tick_alignments={
        "tgp_chr20p_n3202": "right",
        "out_of_africa_n5000_28_1mbp_error": "left",
    },
    sample_tick_rotation=0,
):
    figure, axes = _cohort_panels(
        data,
        columns=len(metrics),
        title=title,
        panel_height=panel_height,
        panel_width=panel_width,
    )
    for ax, (metric, ylabel) in zip(axes.flat, metrics.items(), strict=True):
        _cohort_comparison_curves(
            data,
            ax,
            table,
            metric,
            include_full=include_full,
            sem=sem,
            cutoff_colors=cutoff_colors,
            full_color=full_color,
            line_width=line_width,
            marker=marker,
            marker_size=marker_size,
            method_colors=method_colors,
            method_labels=method_labels,
            real_data_marker=real_data_marker,
        )
        _cohort_sample_axis(
            data,
            ax,
            cohort_labels=cohort_labels,
            sample_tick_alignments=sample_tick_alignments,
            sample_tick_rotation=sample_tick_rotation,
        )
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        elif zoom:
            ax.margins(y=range_margin)
        elif not zoom:
            ax.set_ylim(bottom=0)
        if reference is not None:
            ax.axhline(reference, color="gray", linestyle="--", linewidth=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=len(labels), fontsize="small"
    )
    return figure


def _cohort_comparison_curves(
    data,
    ax,
    table,
    metric,
    include_full=False,
    sem=False,
    *,
    cutoff_colors=["#0072B2", "#D55E00"],
    full_color="#555555",
    line_width=1.7,
    marker="o",
    marker_size=5,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    real_data_marker="o",
):
    """Draw either one curve per method or the archive's AF comparisons."""
    colors = (
        dict(zip(data.cutoffs, cutoff_colors, strict=True))
        if method_colors is None
        else dict.fromkeys(data.cutoffs, method_colors["greedy"])
    )
    if method_colors is not None:
        for method, color in method_colors.items():
            if method == "full_hmm" and (not include_full):
                continue
            selected = (
                table.method.eq(method)
                if "method" in table
                else table.af_cutoff.notna()
            )
            curve = table.loc[selected]
            _cohort_cohort_curve(
                data,
                ax,
                curve,
                metric,
                color,
                _cohort_method_label(
                    data,
                    method,
                    method_colors=method_colors,
                    method_labels=method_labels,
                ),
                sem=sem,
                line_width=line_width,
                marker=marker,
                marker_size=marker_size,
                method_colors=method_colors,
                real_data_marker=real_data_marker,
            )
        return
    curves = [
        (
            cutoff,
            colors[cutoff],
            _cohort_cutoff_label(data, cutoff, method_colors=method_colors),
        )
        for cutoff in data.cutoffs
    ]
    if include_full:
        curves.insert(0, (None, full_color, "Full HMM"))
    for cutoff, color, label in curves:
        selected = (
            table.method.eq("full_hmm")
            if cutoff is None
            else table.af_cutoff.eq(cutoff)
        )
        _cohort_cohort_curve(
            data,
            ax,
            table.loc[selected],
            metric,
            color,
            label,
            sem=sem,
            line_width=line_width,
            marker=marker,
            marker_size=marker_size,
            method_colors=method_colors,
            real_data_marker=real_data_marker,
        )


def plot_coverage(
    data,
    table,
    *,
    cohort_labels={},
    line_width=1.7,
    marker="o",
    marker_size=5,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    panel_height=3.6,
    panel_width=4.4,
    population_colors={"African": "#009E73", "Non-African": "#CC79A7"},
    range_margin=0.08,
    real_data_marker="o",
    sample_tick_alignments={
        "tgp_chr20p_n3202": "right",
        "out_of_africa_n5000_28_1mbp_error": "left",
    },
    sample_tick_rotation=0,
):
    figure, axes = _cohort_panels(
        data,
        columns=len(data.panel_cutoffs),
        title="Physical focal coverage by population",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    for ax, cutoff in zip(axes.flat, data.panel_cutoffs, strict=True):
        for population, color in population_colors.items():
            selected = table.population_group.eq(population)
            if cutoff is not None:
                selected &= table.af_cutoff.eq(cutoff)
            curve = table.loc[selected]
            _cohort_cohort_curve(
                data,
                ax,
                curve,
                "mean",
                color,
                population,
                sem=True,
                line_width=line_width,
                marker=marker,
                marker_size=marker_size,
                method_colors=method_colors,
                real_data_marker=real_data_marker,
            )
        _cohort_sample_axis(
            data,
            ax,
            cohort_labels=cohort_labels,
            sample_tick_alignments=sample_tick_alignments,
            sample_tick_rotation=sample_tick_rotation,
        )
        ax.set(
            title=_cohort_cutoff_label(data, cutoff, method_colors=method_colors),
            ylabel="Physical focal coverage (%)",
        )
        ax.margins(y=range_margin)
        ax.legend()
    _cohort_zoom_y(data, axes.flat)
    return figure


def plot_focal_lengths(
    data,
    lengths,
    unit,
    num_bins,
    *,
    cohort_labels={},
    cutoff_colors=["#0072B2", "#D55E00"],
    histogram_alpha=0.25,
    kde_grid_size=256,
    length_display_quantile=0.99,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    panel_height=3.6,
    panel_width=4.4,
):
    colors = (
        dict(zip(data.cutoffs, cutoff_colors, strict=True))
        if method_colors is None
        else dict.fromkeys(data.cutoffs, method_colors["greedy"])
    )
    figure, axes = _cohort_panels(
        data,
        rows=len(data.panel_cutoffs),
        columns=len(data.datasets),
        title=f"Retained focal lengths · {unit}",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    source = lengths.loc[~lengths.is_gap & lengths.unit.eq(unit)]
    for row, panel_cutoff in enumerate(data.panel_cutoffs):
        pooled = (
            source
            if panel_cutoff is None
            else source.loc[source.af_cutoff.eq(panel_cutoff)]
        )
        upper = nb_utils.perf_scaling_weighted_quantile(pooled, length_display_quantile)
        bins = np.linspace(0, upper, num_bins + 1)
        grid = np.linspace(0, upper, kde_grid_size)
        for col, dataset in enumerate(data.datasets):
            cutoff = _cohort_dataset_cutoff(data, dataset, panel_cutoff)
            ax = axes[row, col]
            values = pooled.loc[pooled.dataset.eq(dataset["name"])]
            total = values["count"].sum()
            tail = values.loc[values.length.gt(upper), "count"].sum() / total
            ax.hist(
                values.length,
                bins=bins,
                weights=values["count"],
                color=colors[cutoff],
                alpha=histogram_alpha,
                label=f"{total:,} regions; tail {tail:.1%}",
            )
            density = nb_utils.perf_scaling_length_density(values, grid)
            ax.plot(grid, density * total * (bins[1] - bins[0]), color=colors[cutoff])
            minimum = (
                dataset["runs"][0]["min_focal_region_size"]
                if unit == "sites"
                else dataset["runs"][0]["min_region_size_kb"]
            )
            label = (
                f"{minimum:,} site minimum"
                if unit == "sites"
                else f"{minimum:g} kb mean equivalent"
            )
            ax.axvline(minimum, linestyle="--", color="gray", label=label)
            ax.set(
                title=_cohort_dataset_title(
                    data,
                    dataset,
                    cutoff,
                    cohort_labels=cohort_labels,
                    method_colors=method_colors,
                ),
                xlabel=f"Focal length ({unit})",
                ylabel="Count per bin",
                xlim=(0, upper),
                ylim=(0, None),
            )
            ax.legend(fontsize="small")
    return figure


def plot_gap_lengths(
    data,
    lengths,
    *,
    cohort_labels={},
    cutoff_colors=["#0072B2", "#D55E00"],
    kde_grid_size=256,
    length_display_quantile=0.99,
    line_width=1.7,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    panel_height=3.6,
    panel_width=4.4,
):
    colors = (
        dict(zip(data.cutoffs, cutoff_colors, strict=True))
        if method_colors is None
        else dict.fromkeys(data.cutoffs, method_colors["greedy"])
    )
    figure, axes = _cohort_panels(
        data,
        columns=len(data.datasets),
        title="Gap length distributions",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    source = lengths.loc[lengths.is_gap & lengths.unit.eq("kb")]
    upper = nb_utils.perf_scaling_weighted_quantile(source, length_display_quantile)
    grid = np.linspace(0, upper, kde_grid_size)
    for col, dataset in enumerate(data.datasets):
        ax = axes[0, col]
        for run in dataset["runs"]:
            cutoff = run["af_cutoff"]
            color = colors[cutoff]
            values = source.loc[
                source.dataset.eq(dataset["name"]) & source.af_cutoff.eq(cutoff)
            ]
            median = nb_utils.perf_scaling_weighted_quantile(values, 0.5)
            density = nb_utils.perf_scaling_length_density(values, grid)
            tail = (
                values.loc[values.length.gt(upper), "count"].sum()
                / values["count"].sum()
            )
            label = f"median {median:.1f} kb; tail {tail:.1%}"
            if method_colors is None:
                label = f"{_cohort_cutoff_label(data, cutoff, data.sample_sizes[col], method_colors=method_colors)}\n{label}"
            ax.plot(grid, density, color=color, label=label)
            ax.axvline(median, color=color, linestyle="--", linewidth=line_width)
        title = (
            _cohort_dataset_label(data, dataset, cohort_labels=cohort_labels)
            if method_colors is None
            else _cohort_dataset_title(
                data,
                dataset,
                cutoff,
                cohort_labels=cohort_labels,
                method_colors=method_colors,
            )
        )
        ax.set(
            title=title,
            xlabel="Gap length (kb)",
            ylabel="Density per kb",
            xlim=(0, upper),
            ylim=(0, None),
        )
        ax.legend(fontsize="small")
    _cohort_zoom_y(data, axes.flat, lower=0)
    return figure


def plot_one_site_gaps(
    data,
    lengths,
    *,
    bar_width=0.65,
    cohort_labels={},
    cutoff_colors=["#0072B2", "#D55E00"],
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    panel_height=3.6,
    panel_width=4.4,
    range_margin=0.08,
):
    """Count one-inference-site gaps, annotating their share of all gaps."""
    colors = (
        dict(zip(data.cutoffs, cutoff_colors, strict=True))
        if method_colors is None
        else dict.fromkeys(data.cutoffs, method_colors["greedy"])
    )
    gaps = lengths.loc[lengths.is_gap & lengths.unit.eq("sites")]
    keys = ["dataset", "af_cutoff"]
    totals = gaps.groupby(keys)["count"].sum().rename("total_gaps")
    singles = gaps.loc[gaps.length.eq(1)].groupby(keys)["count"].sum()
    singles = singles.reindex(totals.index, fill_value=0).rename("one_site_gaps")
    counts = pd.concat([totals, singles], axis=1).reset_index()
    counts["one_site_percentage"] = 100 * counts.one_site_gaps / counts.total_gaps
    counts.to_csv(data.config["cache_dir"] / "one_site_gap_counts.csv", index=False)
    figure, axes = _cohort_panels(
        data,
        columns=len(data.datasets),
        title="Number of one-site gaps · labels show percentage of all gaps",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    for col, dataset in enumerate(data.datasets):
        ax = axes[0, col]
        samples = data.sample_sizes[col]
        table = (
            counts.loc[counts.dataset.eq(dataset["name"])]
            .set_index("af_cutoff")
            .loc[data.cutoffs]
        )
        bars = ax.bar(
            np.arange(len(data.cutoffs)),
            table.one_site_gaps,
            color=[colors[cutoff] for cutoff in data.cutoffs],
            width=bar_width,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in table.one_site_percentage],
            padding=3,
        )
        minimum_kb = dataset["runs"][0]["min_region_size_kb"]
        minimum_sites = dataset["runs"][0]["min_focal_region_size"]
        labels = [
            _cohort_cutoff_label(data, cutoff, samples, method_colors=method_colors)
            for cutoff in data.cutoffs
        ]
        ax.set(
            title=f"{_cohort_dataset_label(data, dataset, cohort_labels=cohort_labels)} · minimum {minimum_kb:g} kb ({minimum_sites:,} sites)",
            xticks=np.arange(len(data.cutoffs)),
            xticklabels=labels,
            xlabel="AF cutoff",
            ylabel="Number of one-site gaps",
        )
        ax.ticklabel_format(axis="y", style="plain")
        ax.margins(y=range_margin * 2)
    _cohort_zoom_y(data, axes.flat, lower=0)
    return figure


def plot_switch_composition(
    data,
    summary,
    *,
    bar_width=0.65,
    cohort_labels={},
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
    switch_categories=["focal", "gap", "seam"],
    switch_colors=["#009E73", "#E69F00", "#7B61A8"],
    switch_labels=["Within focal coverage", "Within gaps", "At gap seams"],
):
    figure, axes = _cohort_panels(
        data,
        rows=len(data.panel_cutoffs),
        columns=len(data.datasets),
        title="Switch composition · each cohort uses its saved focal mask",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    for row, panel_cutoff in enumerate(data.panel_cutoffs):
        for col, dataset in enumerate(data.datasets):
            cutoff = _cohort_dataset_cutoff(data, dataset, panel_cutoff)
            ax = axes[row, col]
            values = summary.loc[
                summary.dataset.eq(dataset["name"]) & summary.af_cutoff.eq(cutoff)
            ].iloc[0]
            bottom = np.zeros(2)
            for category, color, label in zip(
                switch_categories, switch_colors, switch_labels, strict=True
            ):
                height = np.array(
                    [
                        values[f"full_{category}_switches_mean"],
                        values[f"stitched_{category}_switches_mean"],
                    ]
                )
                ax.bar(
                    [0, 1],
                    height,
                    bottom=bottom,
                    color=color,
                    label=label,
                    width=bar_width,
                )
                bottom += height
            ax.set(
                title=_cohort_dataset_title(
                    data,
                    dataset,
                    cutoff,
                    cohort_labels=cohort_labels,
                    method_colors=method_colors,
                ),
                xticks=[0, 1],
                xticklabels=[
                    "Full HMM",
                    _cohort_method_label(
                        data,
                        "greedy",
                        method_colors=method_colors,
                        method_labels=method_labels,
                    ),
                ],
                ylabel="Mean switches per haplotype",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    return figure


def plot_gap_parents(
    data,
    table,
    weight,
    *,
    bar_width=0.65,
    gap_parent_categories={
        "older": {"label": "Older than cutoff", "color": "#0072B2"},
        "recent_nonfocal": {
            "label": "Recent, not focal for haplotype",
            "color": "#E69F00",
        },
        "focal": {"label": "Focal candidate for haplotype", "color": "#009E73"},
    },
    gap_parent_label_minimum_percentage=5,
    gap_parent_weightings={
        "span_bp": "Gap sequence in bp",
        "distinct_parents": "Distinct gap parents",
    },
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    panel_height=3.6,
    panel_width=4.4,
):
    """Compare parent categories in identical gaps for the two matchers."""
    methods = {"full_hmm": "Full HMM", "greedy": "Stitched"}
    ylabel = gap_parent_weightings[weight]
    figure, axes = _cohort_panels(
        data,
        rows=len(data.cutoffs),
        columns=len(methods),
        title=f"Parents used in gaps · {ylabel}",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    x = np.arange(len(data.sample_sizes))
    for row, cutoff in enumerate(data.cutoffs):
        for col, (method, method_label) in enumerate(methods.items()):
            ax = axes[row, col]
            bottom = np.zeros(len(x))
            for category, settings in gap_parent_categories.items():
                curve = table.loc[
                    table.method.eq(method)
                    & table.af_cutoff.eq(cutoff)
                    & table.category.eq(category)
                ].set_index("num_samples")
                heights = curve.loc[
                    data.sample_sizes, f"{weight}_percentage_mean"
                ].to_numpy()
                ax.bar(
                    x,
                    heights,
                    bottom=bottom,
                    color=settings["color"],
                    width=bar_width,
                    label=settings["label"],
                )
                for index, height in enumerate(heights):
                    if height >= gap_parent_label_minimum_percentage:
                        ax.text(
                            x[index],
                            bottom[index] + height / 2,
                            f"{height:.1f}%",
                            ha="center",
                            va="center",
                            fontsize="small",
                        )
                bottom += heights
            np.testing.assert_allclose(bottom, 100)
            ax.set(
                title=f"{method_label} · {_cohort_cutoff_label(data, cutoff, method_colors=method_colors)}",
                xticks=x,
                xticklabels=[f"{n:,}" for n in data.sample_sizes],
                xlabel="Samples (diploid individuals)",
                ylabel="Mean within-haplotype share (%)",
                ylim=(0, 100),
                yticks=np.arange(0, 101, 20),
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=len(labels), fontsize="small"
    )
    return figure


def plot_extra_switch_distributions(
    data,
    *,
    cohort_labels={},
    cutoff_colors=["#0072B2", "#D55E00"],
    kde_grid_size=256,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
):
    colors = (
        dict(zip(data.cutoffs, cutoff_colors, strict=True))
        if method_colors is None
        else dict.fromkeys(data.cutoffs, method_colors["greedy"])
    )
    path = (
        data.config["data_dir"] / data.config["manifest"]["tables"]["haplotype_results"]
    )
    metrics = {
        "num_extra_switches": "Extra switches vs Full HMM",
        "switch_increase_percentage": "Increase in switches (%)",
    }
    table = pd.read_csv(path, usecols=["dataset", "af_cutoff", *metrics])
    figure, axes = _cohort_panels(
        data,
        rows=len(metrics),
        columns=len(data.datasets),
        title="Extra-switch distributions",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    for row, (metric, label) in enumerate(metrics.items()):
        for col, dataset in enumerate(data.datasets):
            ax = axes[row, col]
            for run in dataset["runs"]:
                cutoff = run["af_cutoff"]
                color = colors[cutoff]
                selected = table.dataset.eq(dataset["name"]) & table.af_cutoff.eq(
                    cutoff
                )
                values = table.loc[selected, metric].dropna().to_numpy()
                curve_label = _cohort_cutoff_label(
                    data, cutoff, data.sample_sizes[col], method_colors=method_colors
                )
                if method_colors is not None:
                    curve_label = method_labels["greedy"]
                if np.ptp(values) == 0:
                    ax.axvline(values[0], color=color, label=f"{curve_label}: constant")
                else:
                    grid = np.linspace(values.min(), values.max(), kde_grid_size)
                    ax.plot(
                        grid,
                        stats_density(values, grid),
                        color=color,
                        label=curve_label,
                    )
            title = (
                _cohort_dataset_label(data, dataset, cohort_labels=cohort_labels)
                if method_colors is None
                else _cohort_dataset_title(
                    data,
                    dataset,
                    cutoff,
                    cohort_labels=cohort_labels,
                    method_colors=method_colors,
                )
            )
            ax.set(title=title, xlabel=label, ylabel="Density", ylim=(0, None))
            ax.legend(fontsize="small")
    return figure


def plot_age_correlations(
    data,
    table,
    frequency_bin="all",
    metrics=None,
    *,
    cohort_labels={},
    cutoff_colors=["#0072B2", "#D55E00"],
    full_color="#555555",
    line_width=1.7,
    marker="o",
    marker_size=5,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
    real_data_marker="o",
    sample_tick_alignments={
        "tgp_chr20p_n3202": "right",
        "out_of_africa_n5000_28_1mbp_error": "left",
    },
    sample_tick_rotation=0,
):
    if metrics is None:
        metrics = {
            "pearson_r": "Pearson r",
            "median_absolute_error_generations": "Median absolute error (generations)",
        }
    references = ["truth", "full_hmm"] if data.cohorts.has_truth.any() else ["full_hmm"]
    figure, axes = _cohort_panels(
        data,
        rows=len(references),
        columns=len(metrics),
        title=f"Mutation ages · {frequency_bin}",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    for row, reference in enumerate(references):
        selected = table.loc[
            table.reference_method.eq(reference) & table.frequency_bin.eq(frequency_bin)
        ]
        for col, (metric, ylabel) in enumerate(metrics.items()):
            ax = axes[row, col]
            _cohort_comparison_curves(
                data,
                ax,
                selected,
                metric,
                include_full=reference == "truth",
                cutoff_colors=cutoff_colors,
                full_color=full_color,
                line_width=line_width,
                marker=marker,
                marker_size=marker_size,
                method_colors=method_colors,
                method_labels=method_labels,
                real_data_marker=real_data_marker,
            )
            _cohort_sample_axis(
                data,
                ax,
                datasets=selected.dataset.unique(),
                cohort_labels=cohort_labels,
                sample_tick_alignments=sample_tick_alignments,
                sample_tick_rotation=sample_tick_rotation,
            )
            reference_label = "Truth" if reference == "truth" else "Full HMM"
            ax.set(title=f"Compared with {reference_label}", ylabel=ylabel)
            if metric in ["pearson_r", "spearman_r"]:
                low, high = ax.get_ylim()
                ax.set_ylim(max(-1, low), min(1, high))
            else:
                ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=len(labels), fontsize="small"
    )
    return figure


def plot_age_density(
    data,
    correlations,
    reference,
    *,
    age_hexbin_gridsize=65,
    age_maximum_quantile=0.999,
    age_minimum=1,
    cohort_labels={},
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
):
    datasets = [
        dataset
        for dataset in data.datasets
        if reference != "truth" or dataset["truth"] is not None
    ]
    cases = [("greedy", cutoff) for cutoff in data.panel_cutoffs]
    if reference == "truth":
        cases.insert(0, ("full_hmm", None))
    extra_methods = []
    if method_colors is not None:
        extra_methods = [
            method for method in method_colors if method not in {"full_hmm", "greedy"}
        ]
    cases.extend((method, None) for method in extra_methods)
    reference_label = "True" if reference == "truth" else "Full HMM"
    figure, axes = _cohort_panels(
        data,
        rows=len(cases),
        columns=len(datasets),
        title=f"Mutation-age pairs · compared with {reference_label} ages",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    tables = []
    maximum = age_minimum
    for dataset in datasets:
        path = data.config["cache_dir"] / "age_pairs" / f"{dataset['name']}.parquet"
        stored = pl.read_parquet(path)
        table = pd.DataFrame({name: stored[name].to_numpy() for name in stored.columns})
        tables.append(table)
        columns = [
            reference,
            "full_hmm",
            *[f"af_{run['af_cutoff']}" for run in dataset["runs"]],
            *extra_methods,
        ]
        quantiles = table[columns].quantile(age_maximum_quantile)
        maximum = max(maximum, quantiles.max())
    minimum = age_minimum
    extent = (
        np.log10(minimum),
        np.log10(maximum),
        np.log10(minimum),
        np.log10(maximum),
    )
    collections = []
    for row, (method, panel_cutoff) in enumerate(cases):
        for col, dataset in enumerate(datasets):
            cutoff = _cohort_dataset_cutoff(data, dataset, panel_cutoff)
            key = f"af_{cutoff}" if method == "greedy" else method
            ax = axes[row, col]
            table = tables[col]
            selected = table[reference].ge(minimum) & table[key].ge(minimum)
            drawn = table.loc[selected]
            artist = ax.hexbin(
                drawn[reference],
                drawn[key],
                xscale="log",
                yscale="log",
                extent=extent,
                gridsize=age_hexbin_gridsize,
                mincnt=1,
                bins="log",
                cmap="viridis",
            )
            collections.append(artist)
            ax.plot(
                [minimum, maximum],
                [minimum, maximum],
                color="red",
                linestyle="--",
                linewidth=1,
            )
            samples = len(dataset["cohort"]) // 2
            label = (
                "Full HMM"
                if method == "full_hmm"
                else _cohort_cutoff_label(
                    data, cutoff, samples, method_colors=method_colors
                )
            )
            if method_colors is not None and method == "greedy":
                label = f"{method_labels[method]} · {label}"
            if method in extra_methods:
                label = method_labels[method]
            ax.set(
                title=f"{_cohort_dataset_label(data, dataset, cohort_labels=cohort_labels)} · {label}",
                xlim=(minimum, maximum),
                ylim=(minimum, maximum),
                xlabel=f"{reference_label} mutation age (generations)",
                ylabel="Estimated mutation age (generations)",
            )
            scores = correlations.loc[
                correlations.dataset.eq(dataset["name"])
                & correlations.reference_method.eq(reference)
                & correlations.frequency_bin.eq("all")
            ]
            scores = scores.loc[scores.method.eq(method)]
            if method == "greedy":
                scores = scores.loc[scores.af_cutoff.eq(cutoff)]
            result = scores.iloc[0]
            ax.text(
                0.04,
                0.96,
                f"r = {result.pearson_r:.3f}\nN = {result.matched_mutations:,}",
                transform=ax.transAxes,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
    upper = max((artist.get_array().max() for artist in collections))
    for artist in collections:
        artist.set_norm(mcolors.LogNorm(vmin=1, vmax=upper))
    figure.colorbar(collections[0], ax=axes, label="Mutations per hexagon", shrink=0.8)
    return figure


def _cohort_bar_axis(data, ax, ylabel, *, cohort_labels={}):
    cohort_labels = {
        row.dataset: cohort_labels.get(row.dataset, f"{row.num_samples:,}")
        for row in data.cohorts.itertuples()
    }
    sample_labels = [cohort_labels[name] for name in data.cohorts.dataset]
    ax.set(
        xticks=np.arange(len(data.sample_sizes)),
        xticklabels=sample_labels,
        xlabel="Samples (diploid individuals)",
        ylabel=ylabel,
        ylim=(0, None),
    )


def _cohort_stage_stack(
    data, ax, table, metric, settings, *, bar_width=0.65, stage_label_margin=0.22
):
    bottom = np.zeros(len(data.sample_sizes))
    for stage, style in settings.items():
        selected = table.loc[table.stage.eq(stage)].set_index("num_samples")
        heights = selected.loc[data.sample_sizes, metric].to_numpy()
        ax.bar(
            np.arange(len(bottom)),
            heights,
            bottom=bottom,
            color=style["color"],
            label=style["label"],
            width=bar_width,
        )
        bottom += heights
    ax.set_ylim(0, bottom.max() * (1 + stage_label_margin))
    return bottom


def plot_wall_stages(
    data,
    breakdown,
    *,
    bar_width=0.65,
    cohort_labels={},
    panel_height=3.6,
    panel_width=4.4,
    stage_label_margin=0.22,
    wall_stages={
        "input_setup": {"label": "Load inputs", "color": "#56B4E9"},
        "reference_catalog": {"label": "Reference catalog", "color": "#009E73"},
        "coordinator_process_startup": {"label": "Process startup", "color": "#E69F00"},
        "coordinator_packing_completion": {
            "label": "Pack reference (coordinator)",
            "color": "#CC79A7",
        },
        "coordinator_worker_readiness": {
            "label": "Wait for worker readiness",
            "color": "#F0E442",
        },
        "coordinator_matching_drain": {
            "label": "Load samples, match, assemble",
            "color": "#0072B2",
        },
        "coordinator_shutdown": {"label": "Worker shutdown", "color": "#D55E00"},
        "finalise_output": {"label": "Finalise output", "color": "#999999"},
        "other": {"label": "Uninstrumented wall time", "color": "#DDDDDD"},
    },
):
    figure, axes = _cohort_panels(
        data,
        columns=1,
        title="Greedy pipeline · sequential wall-time stages",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    figure.set_size_inches(panel_width * 2, panel_height)
    ax = axes[0, 0]
    bottom = _cohort_stage_stack(
        data,
        ax,
        breakdown.wall,
        "wall_s",
        wall_stages,
        bar_width=bar_width,
        stage_label_margin=stage_label_margin,
    )
    _cohort_bar_axis(data, ax, "End-to-end wall time (s)", cohort_labels=cohort_labels)
    for x, value in enumerate(bottom):
        ax.annotate(
            f"{value:.1f} s",
            (x, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")
    return figure


def plot_worker_stages(
    data,
    breakdown,
    *,
    bar_width=0.65,
    cohort_labels={},
    panel_height=3.6,
    panel_width=4.4,
    stage_label_margin=0.22,
    worker_stages={
        "reference_packing": {"label": "Pack reference (workers)", "color": "#CC79A7"},
        "worker_setup": {"label": "Worker setup", "color": "#E69F00"},
        "candidate_extension_cover": {
            "label": "Extend candidates and cover",
            "color": "#009E73",
        },
        "gap_native_find_path": {
            "label": "Gap HMM: native find_path",
            "color": "#0072B2",
        },
        "gap_path_conversion": {"label": "Gap path conversion", "color": "#56B4E9"},
        "gap_mutation_extraction": {
            "label": "Gap mutation extraction",
            "color": "#D55E00",
        },
        "gap_final_concatenation": {
            "label": "Final gap concatenation",
            "color": "#F0E442",
        },
        "gap_other": {"label": "Other work inside gap matching", "color": "#999999"},
    },
):
    figure, axes = _cohort_panels(
        data,
        columns=2,
        title="Greedy worker-stage time · mean per matched haplotype",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    metrics = {
        "service_s": "Worker elapsed time\nper haplotype (ms)",
        "thread_cpu_s": "Worker CPU time\nper haplotype (ms)",
    }
    for ax, (metric, label) in zip(axes.flat, metrics.items(), strict=True):
        selected = breakdown.worker.loc[breakdown.worker.metric.eq(metric)]
        bottom = _cohort_stage_stack(
            data,
            ax,
            selected,
            "ms_per_haplotype",
            worker_stages,
            bar_width=bar_width,
            stage_label_margin=stage_label_margin,
        )
        _cohort_bar_axis(data, ax, label, cohort_labels=cohort_labels)
        ax.set_title(
            "Worker elapsed time" if metric == "service_s" else "Worker CPU time"
        )
        for x, value in enumerate(bottom):
            ax.annotate(
                f"{value:.1f} ms",
                (x, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize="small")
    return figure


def plot_recurrence(
    data,
    table,
    *,
    bar_width=0.65,
    cohort_labels={},
    cutoff_colors=["#0072B2", "#D55E00"],
    full_color="#555555",
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
    stage_label_margin=0.22,
    truth_color="#009E73",
):
    colors = (
        dict(zip(data.cutoffs, cutoff_colors, strict=True))
        if method_colors is None
        else dict.fromkeys(data.cutoffs, method_colors["greedy"])
    )
    figure, axes = _cohort_panels(
        data,
        columns=2,
        title="Recurrent mutations at saved inference sites",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    cases = [
        ("truth", None, "Truth", truth_color),
        ("full_hmm", None, "Full HMM", full_color),
    ]
    if method_colors is not None:
        for method, color in method_colors.items():
            if method == "full_hmm":
                continue
            cases.append((method, None, method_labels[method], color))
    else:
        for cutoff in data.cutoffs:
            label = (
                "Stitched" if len(data.cutoffs) == 1 else f"Stitched · AF ≤ {cutoff:g}"
            )
            cases.append(("greedy", cutoff, label, colors[cutoff]))
    metrics = {
        "recurrent_sites": "Number of recurrent sites",
        "mean_mutations_per_recurrent_site": "Mean mutations per recurrent site",
    }
    width = bar_width / len(cases)
    x = np.arange(len(data.sample_sizes))
    for ax, (metric, ylabel) in zip(axes.flat, metrics.items(), strict=True):
        for index, (method, cutoff, label, color) in enumerate(cases):
            selected = table.method.eq(method)
            if cutoff is not None:
                selected &= table.af_cutoff.eq(cutoff)
            values = (
                table.loc[selected].set_index("num_samples").reindex(data.sample_sizes)
            )
            positions = x + (index - (len(cases) - 1) / 2) * width
            bars = ax.bar(
                positions, values[metric], width=width, color=color, label=label
            )
            if metric == "recurrent_sites":
                labels = [
                    f"({value:.1f}%)" if np.isfinite(value) else ""
                    for value in values.recurrent_site_percentage
                ]
                ax.bar_label(
                    bars, labels=labels, padding=3, fontsize="x-small", rotation=90
                )
        _cohort_bar_axis(data, ax, ylabel, cohort_labels=cohort_labels)
        ax.margins(y=stage_label_margin)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=len(labels), fontsize="small"
    )
    return figure


def plot_root_totals(
    data,
    summary,
    *,
    cohort_labels={},
    grouped_bar_width=0.32,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
    root_edge_categories={
        "seam": {"label": "Touches a focal/gap seam", "color": "#7B61A8"},
        "gap_interior": {"label": "Gap interior", "color": "#E69F00"},
        "focal": {"label": "Focal only", "color": "#009E73"},
    },
):
    """Group Full HMM and stitched stacks by size in two metric panels."""
    cohort_labels = {
        row.dataset: cohort_labels.get(row.dataset, f"{row.num_samples:,}")
        for row in data.cohorts.itertuples()
    }
    sample_labels = [cohort_labels[name] for name in data.cohorts.dataset]
    metrics = {
        "edge_count_per_haplotype": "Root edges per haplotype",
        "span_kb_per_haplotype": "Root-copying span\n(kb per haplotype)",
    }
    figure, axes = _cohort_panels(
        data,
        columns=2,
        title="Direct ultimate-root → sample copying",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    x = np.arange(len(data.sample_sizes))
    methods = ["full_hmm", "greedy"]
    width = grouped_bar_width
    for ax, (metric, ylabel) in zip(axes.flat, metrics.items(), strict=True):
        for index, method in enumerate(methods):
            positions = x + (index - (len(methods) - 1) / 2) * width
            bottom = np.zeros(len(x))
            for category, style in root_edge_categories.items():
                mask = summary.method.eq(method) & summary.category.eq(category)
                selected = summary.loc[mask].set_index("num_samples")
                heights = selected.loc[data.sample_sizes, metric].to_numpy()
                ax.bar(
                    positions,
                    heights,
                    bottom=bottom,
                    color=style["color"],
                    width=width,
                    label=style["label"] if index == 0 else None,
                    edgecolor="white",
                    hatch="//" if index == 0 else None,
                )
                bottom += heights
        ax.set(
            xticks=x,
            xticklabels=sample_labels,
            ylabel=ylabel,
            xlabel="Samples (diploid individuals)",
            ylim=(0, None),
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for method in methods:
        hatch = "//" if method == "full_hmm" else None
        handles.append(patches.Patch(facecolor="white", edgecolor="gray", hatch=hatch))
        labels.append(
            _cohort_method_label(
                data, method, method_colors=method_colors, method_labels=method_labels
            )
        )
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(root_edge_categories),
        fontsize="small",
    )
    return figure


def _conditioning_panels(
    config,
    rows,
    columns,
    title,
    *,
    dpi=100,
    panel_height=3.6,
    panel_width=4.4,
    provisional_note="Greedy results are provisional",
):
    figure, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(columns * panel_width, rows * panel_height),
        dpi=dpi,
        layout="constrained",
    )
    figure.suptitle(f"{title}\n{provisional_note}")
    return (figure, axes)


def save_conditioning_figure(config, figure, name, *, export_dpi=180):
    directory = config["output_dir"] / "figures"
    directory.mkdir(exist_ok=True)
    figure.savefig(directory / f"{name}.png", dpi=export_dpi)
    figure.savefig(directory / f"{name}.pdf")
    return figure


def plot_conditioned_recurrence(
    config,
    table,
    *,
    bar_headroom_fraction=0.22,
    bar_width=0.65,
    dpi=100,
    full_color="#555555",
    greedy_color="#0072B2",
    panel_height=3.6,
    panel_width=4.4,
    provisional_note="Greedy results are provisional",
    truth_color="#009E73",
):
    """Reproduce pt11's two recurrent-site quantities for each dataset."""
    figure, axes = _conditioning_panels(
        config,
        len(config["datasets"]),
        2,
        "Recurrent mutations at inference sites",
        dpi=dpi,
        panel_height=panel_height,
        panel_width=panel_width,
        provisional_note=provisional_note,
    )
    methods = {
        "full_hmm": ("Full HMM", full_color),
        "greedy": ("Greedy\nconditioned gaps", greedy_color),
        "truth": ("Truth", truth_color),
    }
    metrics = {
        "recurrent_sites": "Number of recurrent sites",
        "mean_mutations_per_recurrent_site": "Mean mutation records per recurrent site",
    }
    for row, dataset in enumerate(config["datasets"]):
        selected = table.loc[table.dataset.eq(dataset["name"])]
        labels = [methods[value][0] for value in selected.method]
        colors = [methods[value][1] for value in selected.method]
        for ax, (metric, ylabel) in zip(axes[row], metrics.items(), strict=True):
            bars = ax.bar(labels, selected[metric], color=colors, width=bar_width)
            if metric == "recurrent_sites":
                annotations = [
                    f"{count:,}\n({percentage:.1f}%)"
                    for count, percentage in zip(
                        selected.recurrent_sites,
                        selected.recurrent_site_percentage,
                        strict=True,
                    )
                ]
            else:
                annotations = [
                    f"{value:.2f}" if np.isfinite(value) else "undefined"
                    for value in selected[metric]
                ]
            ax.bar_label(bars, labels=annotations, padding=4)
            ax.margins(y=bar_headroom_fraction)
            ax.set(title=dataset["label"], ylabel=ylabel, ylim=(0, None))
    return figure


def plot_conditioned_error_sites(
    config,
    table,
    *,
    dpi=100,
    error_color="#D55E00",
    greedy_color="#0072B2",
    panel_height=3.6,
    panel_width=4.4,
    provisional_note="Greedy results are provisional",
):
    """Show exact set counts; circle areas are deliberately not proportional."""
    figure, axes = _conditioning_panels(
        config,
        1,
        len(table),
        "Simulation: recurrent sites and injected genotype errors",
        dpi=dpi,
        panel_height=panel_height,
        panel_width=panel_width,
        provisional_note=provisional_note,
    )
    for ax, (_, record) in zip(axes.flat, table.iterrows(), strict=True):
        ax.add_patch(
            matplotlib.patches.Circle(
                (0.38, 0.54), 0.26, facecolor=greedy_color, alpha=0.3
            )
        )
        ax.add_patch(
            matplotlib.patches.Circle(
                (0.67, 0.54), 0.26, facecolor=error_color, alpha=0.3
            )
        )
        ax.text(
            0.24, 0.54, f"{record.recurrent_only_sites:,}", ha="center", va="center"
        )
        ax.text(0.525, 0.54, f"{record.overlap_sites:,}", ha="center", va="center")
        ax.text(0.8, 0.54, f"{record.error_only_sites:,}", ha="center", va="center")
        ax.text(0.31, 0.86, "Recurrent sites", ha="center")
        ax.text(0.75, 0.86, "Sites with ≥1 error", ha="center")
        total = record.recurrent_only_sites + record.overlap_sites
        fraction = 100 * record.overlap_sites / total if total > 0 else np.nan
        ax.text(
            0.52,
            0.19,
            f"{fraction:.1f}% of recurrent sites contain an injected error\nNeither: {record.neither_sites:,} of {record.num_sites:,} inference sites\nSchematic areas; counts are exact",
            ha="center",
            va="top",
            fontsize="small",
        )
        label = (
            "Full HMM" if record.method == "full_hmm" else "Greedy (conditioned gaps)"
        )
        ax.set(title=label, xlim=(0, 1.05), ylim=(0, 1), aspect="equal")
        ax.axis("off")
    return figure


def plot_conditioned_recurrent_mutation_errors(
    config,
    table,
    *,
    bar_width=0.65,
    dpi=100,
    error_color="#D55E00",
    other_color="#B8B8B8",
    panel_height=3.6,
    panel_width=4.4,
    provisional_note="Greedy results are provisional",
    truth_color="#009E73",
):
    """Keep unassignable reference mutations separate from sample-call overlap."""
    figure, axes = _conditioning_panels(
        config,
        1,
        1,
        "Simulation: mutation records at recurrent sites",
        dpi=dpi,
        panel_height=panel_height,
        panel_width=panel_width,
        provisional_note=provisional_note,
    )
    ax = axes[0, 0]
    labels = [
        "Full HMM" if method == "full_hmm" else "Greedy\nconditioned gaps"
        for method in table.method
    ]
    categories = {
        "recurrent_sample_mutations_at_error_calls": (
            "Sample mutation at an injected-error call",
            error_color,
        ),
        "recurrent_sample_mutations_at_other_calls": (
            "Sample mutation without an injected genotype error",
            other_color,
        ),
        "recurrent_reference_mutations": (
            "Reference mutation: no individual call assignment",
            truth_color,
        ),
    }
    bottom = np.zeros(len(table))
    for column, (label, color) in categories.items():
        heights = table[column].to_numpy()
        bars = ax.bar(
            labels, heights, bottom=bottom, width=bar_width, label=label, color=color
        )
        annotations = [f"{value:,}" if value > 0 else "" for value in heights]
        ax.bar_label(bars, labels=annotations, label_type="center", fontsize="small")
        bottom += heights
    ax.set(ylabel="Mutation records", ylim=(0, None))
    figure.legend(loc="outside lower center", fontsize="small")
    return figure


def plot_conditioned_copying_events(
    config,
    table,
    *,
    bar_width=0.65,
    dpi=100,
    greedy_color="#0072B2",
    panel_height=3.6,
    panel_width=4.4,
    provisional_note="Greedy results are provisional",
    switch_color="#E69F00",
):
    """Stack mean switches above mean sample mutations, per diploid person."""
    figure, axes = _conditioning_panels(
        config,
        1,
        len(config["datasets"]),
        "Copying events per diploid sample",
        dpi=dpi,
        panel_height=panel_height,
        panel_width=panel_width,
        provisional_note=provisional_note,
    )
    for ax, dataset in zip(axes.flat, config["datasets"], strict=True):
        selected = table.loc[table.dataset.eq(dataset["name"])]
        labels = [
            "Full HMM" if value == "full_hmm" else "Greedy\nconditioned gaps"
            for value in selected.method
        ]
        bottom = np.zeros(len(selected))
        for metric, label, color in (
            ("mutations", "Sample mutations", greedy_color),
            ("switches", "Copying switches", switch_color),
        ):
            values = selected[metric].to_numpy()
            bars = ax.bar(
                labels, values, bottom=bottom, color=color, width=bar_width, label=label
            )
            ax.bar_label(
                bars, labels=[f"{value:.2f}" for value in values], label_type="center"
            )
            bottom += values
        ax.set(
            title=dataset["label"],
            ylabel="Mean count per diploid individual",
            ylim=(0, None),
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2)
    return figure


def correlation_figure(
    results,
    config,
    experiment,
    split_frequency=False,
    *,
    curve_colors=["#0072B2", "#D55E00"],
    curve_label="{value} mismatches allowed",
    curve_parameter="max_mismatches",
    dataset_linestyles={"sim_n300_10mbp_error": "-", "tgp_n300_10mbp": "--"},
    line_width=1.5,
    marker_size=4,
    panel_height=4,
    panel_width=7,
    x_label="AC cutoff",
    x_parameter="ac_cutoff",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Place truth comparisons left and Full HMM comparisons right in each row."""
    x_parameter = x_parameter
    curve_parameter = curve_parameter
    x_values = sorted(experiment[x_parameter])
    curve_values = sorted(experiment[curve_parameter])
    palette = curve_colors
    colors = dict(zip(curve_values, palette, strict=True))
    groups = (
        config["dating"]["frequency_bins"]["labels"] if split_frequency else ["all"]
    )
    fig, axes = plt.subplots(
        len(groups),
        2,
        squeeze=False,
        layout="constrained",
        figsize=(panel_width * 2, panel_height * len(groups)),
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for row, group in enumerate(groups):
        table = results.loc[results.frequency_bin.eq(group)]
        truth = table.loc[table.reference_method.eq("true_arg")]
        baseline = truth.loc[truth.comparison_method.eq("full_hmm"), "pearson_r"].item()
        ax = axes[row, 0]
        for value in curve_values:
            observations = truth.loc[truth[curve_parameter].eq(value)].sort_values(
                x_parameter
            )
            y = np.concatenate(([baseline], observations.pearson_r.to_numpy()))
            ax.plot(
                np.arange(len(y)),
                y,
                color=colors[value],
                label=curve_label.format(value=value),
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
            )
        prefix = f"{group} · " if split_frequency else ""
        ax.set(
            title=f"{prefix}Simulation: inferred vs true ages",
            ylabel="Pearson r",
            xlabel=x_label,
            xticks=np.arange(len(x_values) + 1),
            xticklabels=["Full HMM", *[str(value) for value in x_values]],
        )
        ax.legend()
        ax = axes[row, 1]
        reference = table.loc[table.reference_method.eq("full_hmm")]
        for dataset in config["datasets"]:
            name = dataset["name"]
            if name not in config["dating"]["compare_full_datasets"]:
                continue
            for value in curve_values:
                selected = reference.dataset.eq(name) & reference[curve_parameter].eq(
                    value
                )
                observations = reference.loc[selected].sort_values(x_parameter)
                label = curve_label.format(value=value)
                ax.plot(
                    observations[x_parameter],
                    observations.pearson_r,
                    color=colors[value],
                    linestyle=dataset_linestyles[name],
                    label=f"{dataset['label']} · {label}",
                    marker="o",
                    markersize=marker_size,
                    linewidth=line_width,
                )
        ax.set(
            title=f"{prefix}Stitched vs Full HMM ages",
            ylabel="Pearson r",
            xlabel=x_label,
            xticks=x_values,
        )
        ax.legend()
    title = (
        "Mutation-age correlations by derived allele frequency"
        if split_frequency
        else "Mutation-age correlations"
    )
    fig.suptitle(title)
    return fig


def stitched_age_density_figure(
    pairs,
    config,
    experiment,
    dataset,
    *,
    age_scale="log",
    bins=80,
    cmap="viridis",
    colorbar_shrink=0.7,
    column_label="AC {value:g}",
    curve_label="{value} mismatches allowed",
    curve_parameter="max_mismatches",
    identity_color="#D55E00",
    panel_size=3.6,
    x_parameter="ac_cutoff",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Arrange Full HMM versus stitched densities by the two grid parameters."""
    column_parameter = x_parameter
    row_parameter = curve_parameter
    column_values = sorted(experiment[column_parameter])
    row_values = sorted(experiment[row_parameter])
    selected = pairs.loc[pairs.dataset.eq(dataset["name"])]
    lower = selected[["full_hmm_age", "stitched_age"]].min().min()
    upper = selected[["full_hmm_age", "stitched_age"]].max().max()
    if age_scale == "log":
        edges = np.geomspace(lower, upper, bins + 1)
    else:
        edges = np.linspace(0, upper, bins + 1)
    cases = []
    histograms = []
    for row_value in row_values:
        for column_value in column_values:
            mask = selected[row_parameter].eq(row_value)
            mask &= selected[column_parameter].eq(column_value)
            case = selected.loc[mask]
            counts, _, _ = np.histogram2d(
                case.full_hmm_age, case.stitched_age, bins=edges
            )
            cases.append(case)
            histograms.append(counts)
    maximum = max((counts.max() for counts in histograms))
    norm = mcolors.LogNorm(vmin=1, vmax=maximum)
    fig, axes = plt.subplots(
        len(row_values),
        len(column_values),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
        figsize=(panel_size * len(column_values), panel_size * len(row_values)),
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for index, ax in enumerate(axes.flat):
        row, column = divmod(index, len(column_values))
        case = cases[index]
        counts = histograms[index]
        image = ax.pcolormesh(
            edges,
            edges,
            np.ma.masked_equal(counts.T, 0),
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        ax.plot(
            [edges[0], edges[-1]],
            [edges[0], edges[-1]],
            color=identity_color,
            linestyle="--",
            linewidth=1,
        )
        correlation = stats.pearsonr(case.full_hmm_age, case.stitched_age).statistic
        column_label = column_label.format(value=column_values[column])
        ax.set(
            title=f"{column_label}\nr = {correlation:.3f}; n = {len(case):,}",
            xscale=age_scale,
            yscale=age_scale,
            xlim=(edges[0], edges[-1]),
            ylim=(edges[0], edges[-1]),
            aspect="equal",
        )
        if column == 0:
            row_label = curve_label.format(value=row_values[row])
            ax.set_ylabel(f"{row_label}\nStitched mutation age (generations)")
        if row == len(row_values) - 1:
            ax.set_xlabel("Full HMM mutation age (generations)")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Mutations per bin (log colour scale)",
        shrink=colorbar_shrink,
    )
    scale = age_scale
    fig.suptitle(
        f"{dataset['label']}: Full HMM vs stitched mutation ages · all frequencies\nCommon mutations, axes and colour scale · {scale} age axes"
    )
    return fig


def stats_density(values, grid):
    """Evaluate unreflected KDE for signed extra-switch distributions."""
    counts = pd.Series(values).value_counts().sort_index()
    table = pd.DataFrame({"length": counts.index, "count": counts.to_numpy()})
    weights = table["count"].to_numpy(dtype=float)
    observations = table.length.to_numpy(dtype=float)
    total = weights.sum()
    mean = np.average(observations, weights=weights)
    variance = np.sum(weights * (observations - mean) ** 2) / (total - 1)
    bandwidth = np.sqrt(variance) * total ** (-1 / 5)
    weighted_variance = np.cov(observations, aweights=weights).item()
    kde = stats.gaussian_kde(
        observations, weights=weights, bw_method=bandwidth / np.sqrt(weighted_variance)
    )
    return kde(grid)


def plot_root_edge_totals(
    data,
    summary,
    metric,
    ylabel,
    *,
    bar_width=0.65,
    cohort_labels={},
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
    root_edge_categories={
        "seam": {"label": "Touches a focal/gap seam", "color": "#7B61A8"},
        "gap_interior": {"label": "Gap interior", "color": "#E69F00"},
        "focal": {"label": "Focal only", "color": "#009E73"},
    },
):
    """Stack the same location classes for Full HMM and stitched whole edges."""
    figure, axes = _cohort_panels(
        data,
        rows=len(data.panel_cutoffs),
        columns=len(data.datasets),
        title="Direct ultimate-root → sample copying · observed extent",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    for row, panel_cutoff in enumerate(data.panel_cutoffs):
        for col, dataset in enumerate(data.datasets):
            cutoff = _cohort_dataset_cutoff(data, dataset, panel_cutoff)
            ax = axes[row, col]
            values = summary.loc[
                summary.dataset.eq(dataset["name"]) & summary.af_cutoff.eq(cutoff)
            ]
            bottom = np.zeros(2)
            for category, style in root_edge_categories.items():
                selected = values.loc[values.category.eq(category)].set_index("method")
                heights = selected.loc[["full_hmm", "greedy"], metric].to_numpy()
                ax.bar(
                    [0, 1],
                    heights,
                    bottom=bottom,
                    color=style["color"],
                    width=bar_width,
                    label=style["label"],
                )
                bottom += heights
            ax.set(
                title=_cohort_dataset_title(
                    data,
                    dataset,
                    cutoff,
                    cohort_labels=cohort_labels,
                    method_colors=method_colors,
                ),
                xticks=[0, 1],
                xticklabels=[
                    "Full HMM",
                    _cohort_method_label(
                        data,
                        "greedy",
                        method_colors=method_colors,
                        method_labels=method_labels,
                    ),
                ],
                ylabel=ylabel,
                ylim=(0, None),
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=len(labels), fontsize="small"
    )
    return figure


def plot_root_edge_differences(
    data,
    summary,
    *,
    bar_width=0.65,
    cohort_labels={},
    full_color="#555555",
    marker_size=5,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
    root_edge_categories={
        "seam": {"label": "Touches a focal/gap seam", "color": "#7B61A8"},
        "gap_interior": {"label": "Gap interior", "color": "#E69F00"},
        "focal": {"label": "Focal only", "color": "#009E73"},
    },
):
    """Show category-wise signed differences; their sum is the net excess."""
    sample_labels = [
        cohort_labels.get(row.dataset, f"{row.num_samples:,}")
        for row in data.cohorts.itertuples()
    ]
    metrics = {
        "edge_count_per_haplotype": "Additional root edges per haplotype",
        "span_kb_per_haplotype": "Additional root-copying span\n(kb per haplotype)",
    }
    title = f"{_cohort_method_label(data, 'greedy', method_colors=method_colors, method_labels=method_labels)} − Full HMM · signed root-copying differences"
    figure, axes = _cohort_panels(
        data,
        rows=len(data.panel_cutoffs),
        columns=len(metrics),
        title=title,
        panel_height=panel_height,
        panel_width=panel_width,
    )
    x = np.arange(len(data.sample_sizes))
    for row, panel_cutoff in enumerate(data.panel_cutoffs):
        source = (
            summary
            if panel_cutoff is None
            else summary.loc[summary.af_cutoff.eq(panel_cutoff)]
        )
        for col, (metric, ylabel) in enumerate(metrics.items()):
            ax = axes[row, col]
            positive = np.zeros(len(x))
            negative = np.zeros(len(x))
            net = np.zeros(len(x))
            for category, style in root_edge_categories.items():
                selected = source.loc[source.category.eq(category)]
                pivot = selected.pivot(
                    index="num_samples", columns="method", values=metric
                )
                heights = (
                    (pivot.greedy - pivot.full_hmm).loc[data.sample_sizes].to_numpy()
                )
                bottom = np.where(heights >= 0, positive, negative)
                ax.bar(
                    x,
                    heights,
                    bottom=bottom,
                    color=style["color"],
                    width=bar_width,
                    label=style["label"],
                )
                positive += np.maximum(heights, 0)
                negative += np.minimum(heights, 0)
                net += heights
            ax.scatter(
                x,
                net,
                color=full_color,
                marker="D",
                s=marker_size**2,
                zorder=3,
                label="Net difference",
            )
            ax.axhline(0, color=full_color, linewidth=1)
            ax.set(
                title=_cohort_cutoff_label(
                    data, panel_cutoff, method_colors=method_colors
                ),
                xticks=x,
                xticklabels=sample_labels,
                xlabel="Samples (diploid individuals)",
                ylabel=ylabel,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=len(labels), fontsize="small"
    )
    return figure


def plot_root_edge_spans(
    data,
    *,
    cohort_labels={},
    full_color="#555555",
    line_width=1.7,
    method_colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    panel_height=3.6,
    panel_width=4.4,
    root_edge_categories={
        "seam": {"label": "Touches a focal/gap seam", "color": "#7B61A8"},
        "gap_interior": {"label": "Gap interior", "color": "#E69F00"},
        "focal": {"label": "Focal only", "color": "#009E73"},
    },
    root_edge_histogram_alpha=0.65,
    root_edge_histogram_bins=55,
):
    """Use log-spaced bins over the complete observed span range, including tails."""
    frames = []
    for dataset in data.datasets:
        path = data.config["cache_dir"] / "root_edges" / f"{dataset['name']}.csv"
        frames.append(pd.read_csv(path))
    edges = pd.concat(frames, ignore_index=True)
    minimum = edges.span_bp.min()
    maximum = edges.span_bp.max()
    bins = np.geomspace(minimum, maximum, root_edge_histogram_bins + 1)
    figure, axes = _cohort_panels(
        data,
        rows=len(data.panel_cutoffs),
        columns=len(data.datasets),
        title="Root → sample edge spans · Full HMM and stitched locations",
        panel_height=panel_height,
        panel_width=panel_width,
    )
    for row, panel_cutoff in enumerate(data.panel_cutoffs):
        for col, dataset in enumerate(data.datasets):
            cutoff = _cohort_dataset_cutoff(data, dataset, panel_cutoff)
            ax = axes[row, col]
            selected = edges.loc[
                edges.dataset.eq(dataset["name"]) & edges.af_cutoff.eq(cutoff)
            ]
            samples = 2 * data.sample_sizes[col]
            baseline = selected.loc[selected.method.eq("full_hmm"), "span_bp"]
            grouped = []
            colors = []
            labels = []
            for category, style in root_edge_categories.items():
                values = selected.loc[
                    selected.method.eq("greedy") & selected.category.eq(category),
                    "span_bp",
                ]
                grouped.append(values.to_numpy())
                colors.append(style["color"])
                labels.append(
                    _cohort_method_label(
                        data,
                        "greedy",
                        method_colors=method_colors,
                        method_labels=method_labels,
                    )
                    + ": "
                    + style["label"]
                )
            weights = [np.full(len(values), 1 / samples) for values in grouped]
            ax.hist(
                grouped,
                bins=bins,
                weights=weights,
                stacked=True,
                color=colors,
                alpha=root_edge_histogram_alpha,
                label=labels,
            )
            ax.hist(
                baseline,
                bins=bins,
                weights=np.full(len(baseline), 1 / samples),
                histtype="step",
                color=full_color,
                linewidth=line_width,
                label="Full HMM total",
            )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Observed edge span (bp)",
                ylabel="Edges per haplotype per bin",
                title=_cohort_dataset_title(
                    data,
                    dataset,
                    cutoff,
                    cohort_labels=cohort_labels,
                    method_colors=method_colors,
                ),
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize="small")
    return figure


def plot_population_coverage(
    budget, *, cutoffs, dataset_names, max_ac, population_colors, population_means
):
    """Plot population means in two rows, with simulation left and 1KGP right."""
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7), sharex=True, sharey="row", layout="constrained"
    )
    metrics = ["fraction_covered", "mean_coverage"]
    ylabels = ["Fraction of sites covered", "Mean site coverage"]
    for column, label in enumerate(dataset_names):
        means = population_means[budget][label]
        populations = means["population"].unique().sort().to_list()
        for row, (metric, ylabel) in enumerate(zip(metrics, ylabels, strict=True)):
            ax = axes[row, column]
            for population in populations:
                values = means.filter(pl.col("population") == population)
                ax.plot(
                    values["focal_ac"].to_numpy(),
                    values[metric].to_numpy(),
                    marker="o",
                    markersize=3,
                    label=population,
                    color=population_colors.get(population),
                )
            ax.set_xscale("log")
            ax.set_xticks([2, 3, 5, 10, 20, 50, 100, max_ac])
            ax.set_xticklabels(["2", "3", "5", "10", "20", "50", "100", str(max_ac)])
            ax.set_xlim(cutoffs[0], cutoffs[-1])
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Maximum focal AC")
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(label)
                ax.legend(title="Population")
    axes[0, 0].set_ylim(0, 1.02)
    axes[1, 0].set_ylim(bottom=0)
    fig.suptitle(f"Focal ancestor coverage: max_mismatches = {budget}")
    plt.show()
    return fig


def plot_population_coverage_histograms(
    budget,
    *,
    coverage_by_budget,
    coverage_histogram_bins,
    dataset_names,
    histogram_ac_cutoffs,
    population_colors,
):
    """Compare per-haplotype coverage distributions at each requested AC cutoff."""
    fig, axes = plt.subplots(
        len(histogram_ac_cutoffs),
        2,
        figsize=(12, 10),
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    for column, label in enumerate(dataset_names):
        coverage = coverage_by_budget[budget][label]
        populations = coverage["population"].unique().sort().to_list()
        for row, cutoff in enumerate(histogram_ac_cutoffs):
            ax = axes[row, column]
            selected = coverage.filter(pl.col("focal_ac") == cutoff)
            for population in populations:
                samples = selected.filter(pl.col("population") == population)
                fractions = samples["fraction_covered"].to_numpy()
                ax.hist(
                    fractions,
                    bins=coverage_histogram_bins,
                    histtype="step",
                    linewidth=1.7,
                    label=population,
                    color=population_colors.get(population),
                )
            ax.set_title(f"{label}: maximum focal AC ≤ {cutoff}")
            ax.set_xlabel("Fraction of sites covered")
            ax.set_ylabel("Number of haplotypes")
            ax.set_xlim(0, 1)
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.25)
            ax.legend(title="Population")
    fig.suptitle(f"Sample coverage distributions: max_mismatches = {budget}")
    plt.show()
    return fig


def plot_cumulative_focal_ancestors(
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    num_haploid_samples: int,
    ac_cutoffs: list[int],
) -> plt.Axes:
    """Plot cumulative inferred ancestors by focal AF, marking count cutoffs.

    The dataframe has one row per ancestor and a derived_ac column from the
    focal-ancestor NPZ used for coverage evaluation. Divide by the total number
    of haplotypes to put both datasets on the same count-based frequency axis.
    This equals called-haplotype AF when there are no missing calls.
    Cutoffs use the same conversion, so the curve counts eligible ancestors
    at each dashed line, including ancestors equal to the cutoff.
    """
    focal_frequencies = dataframe["derived_ac"] / num_haploid_samples
    counts_by_frequency = focal_frequencies.value_counts().sort_index()
    cumulative_counts = counts_by_frequency.cumsum()
    x = np.concatenate(([1 / num_haploid_samples], counts_by_frequency.index, [1]))
    y = np.concatenate(([0], cumulative_counts, [len(dataframe)]))
    ax.step(x, y, where="post")
    for cutoff in ac_cutoffs:
        cutoff_frequency = cutoff / num_haploid_samples
        ax.axvline(cutoff_frequency, linestyle="--", color="0.5", linewidth=1)
        ax.text(
            cutoff_frequency,
            0.98,
            f"AC ≤ {cutoff}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            color="0.4",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlim(1 / num_haploid_samples, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Focal allele frequency")
    ax.set_ylabel("Cumulative count of inferred ancestors")
    return ax


def plot_focal_match_stats(
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    cutoff: int,
    score_bins: np.ndarray,
    coverage_bins: np.ndarray,
    population_colors: dict[str, str],
    title: str,
) -> plt.Axes:
    """Plot one cutoff's score/coverage association with marginal histograms.

    The dataframe contains one row per haplotype and cutoff. Use common bin
    edges across calls to compare distributions at different cutoffs. The
    histograms are attached to ax and share its corresponding coordinate axis.
    Population colours identify points, within-population fits and stacked
    histograms; the dashed grey line shows the pooled fit.
    """
    cutoff_rows = dataframe.loc[dataframe["ac_cutoff"] == cutoff]
    scores = cutoff_rows["path_log_likelihood"].to_numpy()
    coverage = cutoff_rows["fraction_covered_sites"].to_numpy()
    pooled_correlation_label = "n/a"
    if len(scores) > 1 and np.ptp(scores) > 0:
        regression = scipy.stats.linregress(scores, coverage)
        if np.isfinite(regression.rvalue):
            pooled_correlation_label = f"{regression.rvalue:.3f}"
        fitted_scores = np.array([scores.min(), scores.max()])
        fitted_coverage = regression.intercept + regression.slope * fitted_scores
        ax.plot(
            fitted_scores, fitted_coverage, color="0.5", linestyle="--", linewidth=1.5
        )
    score_groups = []
    coverage_groups = []
    histogram_colors = []
    for population, color in population_colors.items():
        population_rows = cutoff_rows.loc[cutoff_rows["population"] == population]
        if population_rows.empty:
            continue
        population_scores = population_rows["path_log_likelihood"].to_numpy()
        population_coverage = population_rows["fraction_covered_sites"].to_numpy()
        population_fit = None
        correlation_label = "n/a"
        if len(population_scores) > 1 and np.ptp(population_scores) > 0:
            population_fit = scipy.stats.linregress(
                population_scores, population_coverage
            )
            if np.isfinite(population_fit.rvalue):
                correlation_label = f"{population_fit.rvalue:.3f}"
        ax.scatter(
            population_scores,
            population_coverage,
            s=18,
            alpha=0.6,
            color=color,
            label=f"{population}: r = {correlation_label}, n = {len(population_rows)}",
        )
        if population_fit is not None:
            fitted_scores = np.array([population_scores.min(), population_scores.max()])
            fitted_coverage = (
                population_fit.intercept + population_fit.slope * fitted_scores
            )
            ax.plot(fitted_scores, fitted_coverage, color=color, linewidth=2)
        score_groups.append(population_scores)
        coverage_groups.append(population_coverage)
        histogram_colors.append(color)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(
        0.04,
        0.2,
        f"Pooled r = {pooled_correlation_label}\nn = {len(cutoff_rows)} haplotypes",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.set_xlabel("Path log likelihood (higher is better)")
    ax.set_ylabel("Fraction of covered inference sites")
    ax.set_xlim(score_bins[0], score_bins[-1])
    ax.set_ylim(coverage_bins[0], coverage_bins[-1])
    divider = axes_grid1.make_axes_locatable(ax)
    score_ax = divider.append_axes("top", size="25%", pad=0.12, sharex=ax)
    coverage_ax = divider.append_axes("right", size="25%", pad=0.12, sharey=ax)
    score_ax.hist(
        score_groups, bins=score_bins, stacked=True, color=histogram_colors, alpha=0.7
    )
    coverage_ax.hist(
        coverage_groups,
        bins=coverage_bins,
        orientation="horizontal",
        stacked=True,
        color=histogram_colors,
        alpha=0.7,
    )
    score_ax.tick_params(axis="x", labelbottom=False)
    coverage_ax.tick_params(axis="y", labelleft=False)
    score_ax.set_ylabel("Haplotypes")
    coverage_ax.set_xlabel("Haplotypes")
    score_ax.set_title(title)
    return ax


def plot_intervals(
    intervals,
    title,
    *,
    config,
    plot_figure_width=10,
    plot_minimum_figure_height=2.4,
    plot_row_height=0.45,
    plot_tick_step=5,
):
    """Show input ancestors with closed left and open right endpoints."""
    height = max(plot_minimum_figure_height, plot_row_height * len(intervals) + 1.2)
    fig, ax = plt.subplots(figsize=(plot_figure_width, height))
    for row, (left, right) in enumerate(intervals):
        ax.hlines(row, left, right, color="tab:blue", linewidth=3)
        ax.plot(left, row, "o", color="tab:blue", markersize=7)
        ax.plot(
            right,
            row,
            "o",
            color="tab:blue",
            markerfacecolor="white",
            markersize=7,
            markeredgewidth=1.5,
        )
    rows = range(len(intervals))
    labels = [
        f"{row}: [{left}, {right})" for row, (left, right) in enumerate(intervals)
    ]
    ax.set_yticks(rows, labels)
    ax.set_ylim(max(len(intervals) - 0.5, 0.5), -0.5)
    ax.set_xlim(0, config["num_sites"])
    ticks = range(0, config["num_sites"] + 1, plot_tick_step)
    ax.set_xticks(ticks)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlabel("Site boundary (filled left endpoint; open right endpoint)")
    ax.set_ylabel("Ancestor row")
    ax.set_title(title)
    if len(intervals) == 0:
        ax.text(0.5, 0.5, "No candidate intervals", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    plt.show()


def draw_strip(
    ax: plt.Axes,
    edges: np.ndarray,
    values: np.ndarray,
    source: bool = False,
    *,
    category_cmap,
    category_norm,
    source_cmap,
    source_norm,
) -> None:
    """Draw site categories using their conventional genomic display bins."""
    cmap = source_cmap if source else category_cmap
    norm = source_norm if source else category_norm
    ax.pcolormesh(
        edges,
        [0, 1],
        values[None, :],
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    ax.set_yticks([])


def plot_overview(
    dataset: nb_utils.DatasetDiagnostics,
    *,
    category_cmap,
    category_norm,
    coverage_handles,
    plot_full_color="#222222",
    plot_genome_unit_bp=1000000,
    plot_genome_unit_label="Mb",
    plot_overview_figsize=[16, 7],
    plot_stitched_color="#6A3D9A",
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> None:
    """Align provenance, both parent paths, and agreement over the inference span."""
    case = dataset.primary
    edges = dataset.site_edges / plot_genome_unit_bp
    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=plot_overview_figsize,
        gridspec_kw={"height_ratios": [0.5, 2, 2, 0.5]},
        layout="constrained",
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    draw_strip(
        axes[0],
        edges,
        (~case.accepted).astype(int),
        source=True,
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    for ax, parents, label, color in zip(
        axes[1:3],
        (case.full_parents, case.stitched_parents),
        ("Full HMM", "Stitched"),
        (plot_full_color, plot_stitched_color),
        strict=True,
    ):
        ax.stairs(parents, edges, baseline=None, color=color, linewidth=0.8)
        ax.set_ylabel(f"{label}\nOriginal node ID")
        ax.grid(alpha=0.2)
    parent_min = min(case.full_parents.min(), case.stitched_parents.min())
    parent_max = max(case.full_parents.max(), case.stitched_parents.max())
    for ax in axes[1:3]:
        ax.set_ylim(parent_min - 1, parent_max + 1)
    draw_strip(
        axes[3],
        edges,
        case.categories,
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[3].set_ylabel("Agreement", rotation=0, ha="right")
    axes[3].set_xlabel(f"Genomic position ({plot_genome_unit_label})")
    axes[3].set_xlim(edges[0], edges[-1])
    fig.suptitle(
        f"{dataset.label}: {case.label} | agreement {case.summary['fraction_parent_agreement']:.1%} | switches {case.summary['full_switches']} → {case.summary['stitched_switches']} | extra switch prop. {case.summary['extra_switch_fraction']:.1%}"
    )
    fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
    plt.show()


def plot_detail(
    dataset: nb_utils.DatasetDiagnostics,
    window: nb_utils.DetailWindow,
    *,
    category_cmap,
    category_norm,
    coverage_handles,
    plot_detail_figsize=[16, 7],
    plot_detail_unit_bp=1000,
    plot_detail_unit_label="kb",
    plot_full_color="#222222",
    plot_parent_row_height_in=0.24,
    plot_source_colors=["#B8DED4", "#FFE3A3"],
    plot_stitched_color="#6A3D9A",
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> None:
    """Plot both paths on shared categorical parent rows and annotate the target."""
    case = dataset.primary
    start, end = (window.start, window.end)
    positions = dataset.positions[start:end] / plot_detail_unit_bp
    edges = dataset.site_edges[start : end + 1] / plot_detail_unit_bp
    full = case.full_parents[start:end]
    stitched = case.stitched_parents[start:end]
    nodes = np.union1d(full, stitched)
    full_rows = np.searchsorted(nodes, full)
    stitched_rows = np.searchsorted(nodes, stitched)
    figsize = list(plot_detail_figsize)
    figsize[1] = max(figsize[1], len(nodes) * plot_parent_row_height_in + 2)
    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [0.4, 4, 0.4, 0.6]},
        layout="constrained",
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    draw_strip(
        axes[0],
        edges,
        (~case.accepted[start:end]).astype(int),
        source=True,
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    source = (~case.accepted[start:end]).astype(int)
    changes = np.flatnonzero(source[1:] != source[:-1]) + 1
    run_bounds = np.concatenate(([0], changes, [len(source)]))
    for lo, hi in zip(run_bounds[:-1], run_bounds[1:], strict=True):
        axes[1].axvspan(
            edges[lo],
            edges[hi],
            color=plot_source_colors[source[lo]],
            alpha=0.5,
            linewidth=0,
        )
    axes[1].stairs(
        full_rows,
        edges,
        baseline=None,
        color=plot_full_color,
        linestyle="--",
        linewidth=1.5,
        label="Full HMM",
    )
    axes[1].stairs(
        stitched_rows,
        edges,
        baseline=None,
        color=plot_stitched_color,
        linewidth=1.5,
        label="Stitched",
    )
    axes[1].scatter(
        positions, full_rows, color=plot_full_color, s=12, marker="o", zorder=3
    )
    axes[1].scatter(
        positions, stitched_rows, color=plot_stitched_color, s=16, marker="x", zorder=4
    )
    labels = []
    for node_id in nodes:
        metadata = dataset.reference.node(int(node_id)).metadata
        identity = metadata.get("sample_id", "synthetic root")
        labels.append(f"{identity} (node {node_id})")
    axes[1].set(
        yticks=np.arange(len(nodes)),
        yticklabels=labels,
        ylim=(-0.7, len(nodes) - 0.3),
        ylabel="Copying parent",
    )
    axes[1].tick_params(axis="y", labelsize="small")
    axes[1].grid(axis="y", alpha=0.2)
    path_handles = axes[1].get_legend_handles_labels()[0]
    draw_strip(
        axes[2],
        edges,
        case.categories[start:end],
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[2].set_ylabel("Agreement", rotation=0, ha="right")
    for parents, level, color in zip(
        (case.full_parents, case.stitched_parents),
        (1, 0),
        (plot_full_color, plot_stitched_color),
        strict=True,
    ):
        switches = np.flatnonzero(parents[1:] != parents[:-1]) + 1
        switches = switches[(switches >= start) & (switches < end)]
        x = dataset.positions[switches] / plot_detail_unit_bp
        axes[3].scatter(x, np.full(len(x), level), marker="|", color=color, s=100)
    axes[3].set(
        yticks=[0, 1],
        yticklabels=["Stitched", "Full"],
        ylim=(-0.7, 1.7),
        xlabel=f"Genomic position ({plot_detail_unit_label}); ticks mark parent switches",
    )
    for boundary in (window.target_start, window.target_end):
        if start <= boundary <= end:
            x = dataset.site_edges[boundary] / plot_detail_unit_bp
            for ax in axes:
                ax.axvline(x, color="0.3", linestyle=":", linewidth=1)
    axes[-1].set_xlim(edges[0], edges[-1])
    fraction = np.mean(~case.disagreement[start:end])
    target_fraction = np.mean(
        ~case.disagreement[window.target_start : window.target_end]
    )
    fig.suptitle(
        f"{dataset.label}: {case.label} — {window.kind}\nTarget sites [{window.target_start}, {window.target_end}), agreement {target_fraction:.1%}; displayed [{start}, {end}), agreement {fraction:.1%}"
    )
    fig.legend(
        handles=coverage_handles + path_handles, loc="outside lower center", ncol=4
    )
    plt.show()


def path_diagnostics_plot_overview(
    dataset: nb_utils.PathDiagnosticsDatasetDiagnostics,
    *,
    category_cmap,
    category_norm,
    coverage_handles,
    plot_full_color="#222222",
    plot_genome_unit_bp=1000000,
    plot_genome_unit_label="Mb",
    plot_overview_figsize=[16, 7],
    plot_stitched_color="#6A3D9A",
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> plt.Figure:
    """Align provenance, both parent paths, and agreement over the inference span."""
    case = dataset.primary
    edges = dataset.site_edges / plot_genome_unit_bp
    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=plot_overview_figsize,
        gridspec_kw={"height_ratios": [0.5, 2, 2, 0.5]},
        layout="constrained",
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    draw_strip(
        axes[0],
        edges,
        (~case.covered).astype(int),
        source=True,
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    for ax, parents, label, color in zip(
        axes[1:3],
        (case.full_parents, case.stitched_parents),
        ("Full HMM", "Stitched"),
        (plot_full_color, plot_stitched_color),
        strict=True,
    ):
        ax.stairs(parents, edges, baseline=None, color=color, linewidth=0.8)
        ax.set_ylabel(f"{label}\nOriginal node ID")
        ax.grid(alpha=0.2)
    parent_min = min(case.full_parents.min(), case.stitched_parents.min())
    parent_max = max(case.full_parents.max(), case.stitched_parents.max())
    for ax in axes[1:3]:
        ax.set_ylim(parent_min - 1, parent_max + 1)
    draw_strip(
        axes[3],
        edges,
        case.categories,
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[3].set_ylabel("Agreement", rotation=0, ha="right")
    axes[3].set_xlabel(f"Genomic position ({plot_genome_unit_label})")
    axes[3].set_xlim(edges[0], edges[-1])
    fig.suptitle(
        f"{dataset.label}: {case.label} | agreement {case.summary['fraction_parent_agreement']:.1%} | switches {case.summary['full_switches']} → {case.summary['stitched_switches']} | extra switches {case.summary['num_extra_switches']}"
    )
    fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
    return fig


def path_diagnostics_plot_detail(
    dataset: nb_utils.PathDiagnosticsDatasetDiagnostics,
    window: nb_utils.DetailWindow,
    *,
    site_axis: bool = False,
    category_cmap,
    category_norm,
    coverage_handles,
    plot_detail_figsize=[16, 7],
    plot_detail_unit_bp=1000,
    plot_detail_unit_label="kb",
    plot_full_color="#222222",
    plot_parent_row_height_in=0.24,
    plot_source_colors=["#B8DED4", "#FFE3A3"],
    plot_stitched_color="#6A3D9A",
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> plt.Figure:
    """Plot both paths on shared categorical parent rows and annotate the target."""
    case = dataset.primary
    start, end = (window.start, window.end)
    positions = dataset.positions[start:end] / plot_detail_unit_bp
    edges = dataset.site_edges[start : end + 1] / plot_detail_unit_bp
    if site_axis:
        positions = np.arange(start, end)
        edges = np.arange(start, end + 1)
    full = case.full_parents[start:end]
    stitched = case.stitched_parents[start:end]
    nodes = np.union1d(full, stitched)
    full_rows = np.searchsorted(nodes, full)
    stitched_rows = np.searchsorted(nodes, stitched)
    figsize = list(plot_detail_figsize)
    figsize[1] = max(figsize[1], len(nodes) * plot_parent_row_height_in + 2)
    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [0.4, 4, 0.4, 0.6]},
        layout="constrained",
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    draw_strip(
        axes[0],
        edges,
        (~case.covered[start:end]).astype(int),
        source=True,
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    source = (~case.covered[start:end]).astype(int)
    changes = np.flatnonzero(source[1:] != source[:-1]) + 1
    run_bounds = np.concatenate(([0], changes, [len(source)]))
    for lo, hi in zip(run_bounds[:-1], run_bounds[1:], strict=True):
        axes[1].axvspan(
            edges[lo],
            edges[hi],
            color=plot_source_colors[source[lo]],
            alpha=0.5,
            linewidth=0,
        )
    axes[1].stairs(
        full_rows,
        edges,
        baseline=None,
        color=plot_full_color,
        linestyle="--",
        linewidth=1.5,
        label="Full HMM",
    )
    axes[1].stairs(
        stitched_rows,
        edges,
        baseline=None,
        color=plot_stitched_color,
        linewidth=1.5,
        label="Stitched",
    )
    axes[1].scatter(
        positions, full_rows, color=plot_full_color, s=12, marker="o", zorder=3
    )
    axes[1].scatter(
        positions, stitched_rows, color=plot_stitched_color, s=16, marker="x", zorder=4
    )
    labels = []
    for node_id in nodes:
        metadata = dataset.reference.node(int(node_id)).metadata
        identity = metadata.get("sample_id", "synthetic root")
        labels.append(f"{identity} (node {node_id})")
    axes[1].set(
        yticks=np.arange(len(nodes)),
        yticklabels=labels,
        ylim=(-0.7, len(nodes) - 0.3),
        ylabel="Copying parent",
    )
    axes[1].tick_params(axis="y", labelsize="small")
    axes[1].grid(axis="y", alpha=0.2)
    path_handles = axes[1].get_legend_handles_labels()[0]
    draw_strip(
        axes[2],
        edges,
        case.categories[start:end],
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[2].set_ylabel("Agreement", rotation=0, ha="right")
    for parents, level, color in zip(
        (case.full_parents, case.stitched_parents),
        (1, 0),
        (plot_full_color, plot_stitched_color),
        strict=True,
    ):
        switches = np.flatnonzero(parents[1:] != parents[:-1]) + 1
        switches = switches[(switches >= start) & (switches < end)]
        x = switches if site_axis else dataset.positions[switches] / plot_detail_unit_bp
        axes[3].scatter(x, np.full(len(x), level), marker="|", color=color, s=100)
    position_label = f"Genomic position ({plot_detail_unit_label})"
    if site_axis:
        position_label = "Inference-site index"
    axes[3].set(
        yticks=[0, 1],
        yticklabels=["Stitched", "Full"],
        ylim=(-0.7, 1.7),
        xlabel=f"{position_label}; ticks mark parent switches",
    )
    for boundary in (window.target_start, window.target_end):
        if start <= boundary <= end:
            x = (
                boundary
                if site_axis
                else dataset.site_edges[boundary] / plot_detail_unit_bp
            )
            for ax in axes:
                ax.axvline(x, color="0.3", linestyle=":", linewidth=1)
    axes[-1].set_xlim(edges[0], edges[-1])
    fraction = np.mean(~case.disagreement[start:end])
    target_fraction = np.mean(
        ~case.disagreement[window.target_start : window.target_end]
    )
    fig.suptitle(
        f"{dataset.label}: {case.label} — {window.kind}\nTarget sites [{window.target_start}, {window.target_end}), agreement {target_fraction:.1%}; displayed [{start}, {end}), agreement {fraction:.1%}"
    )
    fig.legend(
        handles=coverage_handles + path_handles, loc="outside lower center", ncol=4
    )
    return fig


@dataclasses.dataclass
class CaseSummaryFigure:
    """Summary figure and its switch-count table."""

    figure: plt.Figure
    switches: pd.DataFrame


def plot_case_summary(
    datasets: list[nb_utils.PathDiagnosticsDatasetDiagnostics],
    title: str,
    *,
    plot_category_colors=["#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
    plot_category_labels=[
        "Focal / agree",
        "Focal / differ",
        "Gap fill / agree",
        "Gap fill / differ",
    ],
    plot_cohort_figsize=[16, 8],
    plot_switch_colors=["#009E73", "#E69F00", "#7B61A8"],
    plot_switch_labels=["Within focal coverage", "Within gaps", "At focal/gap seams"],
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> CaseSummaryFigure:
    """Compare site compositions and switch-type counts for selected haplotypes."""
    fig, axes = plt.subplots(
        2,
        len(datasets),
        figsize=plot_cohort_figsize,
        layout="constrained",
        squeeze=False,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    switch_rows = []
    for column, dataset in enumerate(datasets):
        y = np.arange(len(dataset.cases))
        accumulated = np.zeros(len(y))
        for category, (label, color) in enumerate(
            zip(plot_category_labels, plot_category_colors, strict=True)
        ):
            fractions = np.array(
                [np.mean(case.categories == category) for case in dataset.cases]
            )
            axes[0, column].barh(
                y, fractions, left=accumulated, color=color, label=label
            )
            accumulated += fractions
        axes[0, column].set(
            yticks=y,
            yticklabels=[case.label for case in dataset.cases],
            xlim=(0, 1),
            xlabel="Fraction of inference sites",
            title=dataset.label,
        )
        axes[0, column].invert_yaxis()
        axes[0, column].legend(
            fontsize="small", loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2
        )
        for offset, method in zip((-0.2, 0.2), ("full", "stitched"), strict=True):
            parents = [getattr(case, f"{method}_parents") for case in dataset.cases]
            counts = np.array(
                [
                    nb_utils.path_diagnostics_switch_breakdown(case, path)
                    for case, path in zip(dataset.cases, parents, strict=True)
                ]
            )
            accumulated = np.zeros(len(y))
            for category, color in enumerate(plot_switch_colors):
                axes[1, column].barh(
                    y + offset,
                    counts[:, category],
                    left=accumulated,
                    height=0.35,
                    color=color,
                    alpha=1 if method == "stitched" else 0.5,
                )
                accumulated += counts[:, category]
            for case, values in zip(dataset.cases, counts, strict=True):
                switch_rows.append(
                    {
                        "dataset": dataset.label,
                        "haplotype": case.label,
                        "method": method,
                        **dict(zip(plot_switch_labels, values, strict=True)),
                    }
                )
        axes[1, column].set(
            yticks=y,
            yticklabels=[case.label for case in dataset.cases],
            xlabel="Parent switches (upper: full HMM; lower: stitched)",
        )
        axes[1, column].invert_yaxis()
        handles = [
            mpatches.Patch(color=color, label=label)
            for color, label in zip(plot_switch_colors, plot_switch_labels, strict=True)
        ]
        axes[1, column].legend(handles=handles, fontsize="small")
    fig.suptitle(title)
    switch_table = pd.DataFrame(switch_rows)
    return CaseSummaryFigure(fig, switch_table)


def plot_case_tracks(
    datasets: list[nb_utils.PathDiagnosticsDatasetDiagnostics],
    title: str,
    *,
    category_cmap,
    category_norm,
    coverage_handles,
    plot_genome_unit_bp=1000000,
    plot_genome_unit_label="Mb",
    plot_tracks_figsize=[16, 7],
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> plt.Figure:
    """Align provenance and parent agreement for all selected haplotypes."""
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=plot_tracks_figsize,
        layout="constrained",
        squeeze=False,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        edges = dataset.site_edges / plot_genome_unit_bp
        for index, case in enumerate(dataset.cases):
            y = index * 2
            ax.pcolormesh(
                edges,
                [y, y + 0.7],
                (~case.covered)[None, :],
                cmap=source_cmap,
                norm=source_norm,
                shading="flat",
                rasterized=True,
            )
            ax.pcolormesh(
                edges,
                [y + 0.8, y + 1.5],
                case.categories[None, :],
                cmap=category_cmap,
                norm=category_norm,
                shading="flat",
                rasterized=True,
            )
        ax.set(
            yticks=np.arange(len(dataset.cases)) * 2 + 0.75,
            yticklabels=[case.label for case in dataset.cases],
            xlabel=f"Genomic position ({plot_genome_unit_label})",
            title=dataset.label,
        )
        ax.set_xlim(edges[0], edges[-1])
        ax.invert_yaxis()
    fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
    fig.suptitle(title)
    return fig


def plot_population_histograms(
    metric: str,
    xlabel: str,
    title: str,
    *,
    integer_bins: bool = False,
    bounded_percentage: bool = False,
    config,
    plot_full_color="#222222",
    plot_population_results_count_bin_width=1,
    plot_population_results_density_tick_count=5,
    plot_population_results_histogram_alpha=0.3,
    plot_population_results_histogram_bins=30,
    plot_population_results_histogram_figsize=[14, 10],
    plot_population_results_kde_bandwidth="scott",
    plot_population_results_kde_grid_size=512,
    plot_population_results_line_width=2,
    plot_population_results_point_alpha=0.6,
    plot_population_results_point_size=12,
    plot_population_results_point_strip_depth=0.15,
    plot_population_results_population_colors={
        "YRI": "tab:blue",
        "AFR": "tab:blue",
        "CEU": "tab:orange",
        "EUR": "tab:orange",
        "CHB": "tab:gray",
        "EAS": "tab:gray",
    },
    plot_population_results_random_seed=42,
    plot_population_results_range_padding=0.05,
    population_comparisons,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> plt.Figure:
    """Show distributions, every defined haplotype, KDEs, and population means.

    Bins and axes are shared by populations within each dataset. KDEs are
    reflected at zero and, for agreement percentages, at 100 to keep density
    inside the metric's support. Zero denominators remain missing observations.
    """
    fig, axes = plt.subplots(
        len(config["datasets"][0]["populations"]),
        len(config["datasets"]),
        figsize=plot_population_results_histogram_figsize,
        layout="constrained",
        sharex="col",
        sharey="col",
        squeeze=False,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    rng = np.random.default_rng(plot_population_results_random_seed)
    for column, settings in enumerate(config["datasets"]):
        dataset_table = population_comparisons.loc[
            population_comparisons.dataset.eq(settings["name"])
        ]
        dataset_values = dataset_table[metric].dropna().to_numpy()
        lower = min(0, dataset_values.min())
        upper = dataset_values.max()
        if bounded_percentage:
            lower, upper = (0, 100)
        if integer_bins:
            bin_width = plot_population_results_count_bin_width
            bins = np.arange(lower - bin_width / 2, upper + bin_width, bin_width)
        else:
            if not bounded_percentage:
                upper += max(upper - lower, 1) * plot_population_results_range_padding
            bins = np.linspace(lower, upper, plot_population_results_histogram_bins + 1)
        grid = np.linspace(lower, upper, plot_population_results_kde_grid_size)
        density_max = 0
        for row, population in enumerate(settings["populations"]):
            ax = axes[row, column]
            color = plot_population_results_population_colors[population]
            observations = dataset_table.loc[
                dataset_table.population.eq(population), metric
            ]
            values = observations.dropna().to_numpy()
            excluded = observations.isna().sum()
            densities, _ = np.histogram(values, bins=bins, density=True)
            density_max = max(density_max, densities.max())
            ax.hist(
                values,
                bins=bins,
                density=True,
                color=color,
                alpha=plot_population_results_histogram_alpha,
                edgecolor="white",
            )
            if len(values) > 1 and np.ptp(values) > 0:
                kde = stats.gaussian_kde(
                    values, bw_method=plot_population_results_kde_bandwidth
                )
                density = kde(grid)
                if lower == 0:
                    density += kde(-grid)
                if bounded_percentage:
                    density += kde(200 - grid)
                ax.plot(
                    grid,
                    density,
                    color=color,
                    linewidth=plot_population_results_line_width,
                )
                density_max = max(density_max, density.max())
            mean = values.mean()
            ax.axvline(
                mean,
                color=color,
                linestyle="--",
                linewidth=plot_population_results_line_width,
            )
            annotation_x = 0.02 if bounded_percentage else 0.98
            annotation_alignment = "left" if bounded_percentage else "right"
            ax.text(
                annotation_x,
                0.95,
                f"{population}: n = {len(values)}; excluded = {excluded}\nMean = {mean:.2f}",
                transform=ax.transAxes,
                ha=annotation_alignment,
                va="top",
            )
            ax.set_ylabel("Density")
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title(settings["label"])
            if row == len(settings["populations"]) - 1:
                ax.set_xlabel(xlabel)
        strip_depth = density_max * plot_population_results_point_strip_depth
        for row, population in enumerate(settings["populations"]):
            ax = axes[row, column]
            color = plot_population_results_population_colors[population]
            values = dataset_table.loc[dataset_table.population.eq(population), metric]
            values = values.dropna().to_numpy()
            point_y = -rng.uniform(strip_depth / 4, strip_depth * 3 / 4, len(values))
            ax.scatter(
                values,
                point_y,
                s=plot_population_results_point_size,
                alpha=plot_population_results_point_alpha,
                color=color,
                linewidths=0,
            )
            ax.axhline(0, color=plot_full_color, linewidth=0.5)
            density_upper = density_max * 1.25
            tick_locator = mticker.MaxNLocator(
                nbins=plot_population_results_density_tick_count
            )
            density_ticks = tick_locator.tick_values(0, density_upper)
            density_ticks = density_ticks[
                (density_ticks >= 0) & (density_ticks <= density_upper)
            ]
            ax.set_yticks(density_ticks)
            ax.set(xlim=(bins[0], bins[-1]), ylim=(-strip_depth, density_upper))
    fig.suptitle(f"{title}\nSolid: kernel density; dashed: mean; points: haplotypes")
    return fig


def plot_gap_property_histograms(
    metric: str,
    xlabel: str,
    title: str,
    *,
    integer_bins: bool = False,
    config,
    gap_population_handles,
    plot_full_color="#222222",
    plot_population_results_count_bin_width=1,
    plot_population_results_density_tick_count=5,
    plot_population_results_gap_histogram_color="tab:gray",
    plot_population_results_gap_histogram_figsize=[14, 5],
    plot_population_results_histogram_alpha=0.3,
    plot_population_results_histogram_bins=30,
    plot_population_results_kde_bandwidth="scott",
    plot_population_results_kde_grid_size=512,
    plot_population_results_line_width=2,
    plot_population_results_point_alpha=0.6,
    plot_population_results_point_size=12,
    plot_population_results_point_strip_depth=0.15,
    plot_population_results_population_colors={
        "YRI": "tab:blue",
        "AFR": "tab:blue",
        "CEU": "tab:orange",
        "EUR": "tab:orange",
        "CHB": "tab:gray",
        "EAS": "tab:gray",
    },
    plot_population_results_random_seed=42,
    plot_population_results_range_padding=0.05,
    population_comparisons,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> plt.Figure:
    """Pool haplotype gap properties by dataset, retaining population point colours."""
    fig, axes = plt.subplots(
        1,
        len(config["datasets"]),
        figsize=plot_population_results_gap_histogram_figsize,
        layout="constrained",
        squeeze=False,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    rng = np.random.default_rng(plot_population_results_random_seed)
    for ax, settings in zip(axes[0], config["datasets"], strict=True):
        dataset_table = population_comparisons.loc[
            population_comparisons.dataset.eq(settings["name"])
        ]
        observations = dataset_table[metric]
        values = observations.dropna().to_numpy()
        excluded = observations.isna().sum()
        upper = values.max()
        if integer_bins:
            bin_width = plot_population_results_count_bin_width
            bins = np.arange(-bin_width / 2, upper + bin_width, bin_width)
        else:
            upper += upper * plot_population_results_range_padding
            bins = np.linspace(0, upper, plot_population_results_histogram_bins + 1)
        grid = np.linspace(0, upper, plot_population_results_kde_grid_size)
        densities, _ = np.histogram(values, bins=bins, density=True)
        density_max = densities.max()
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=plot_population_results_gap_histogram_color,
            alpha=plot_population_results_histogram_alpha,
            edgecolor="white",
        )
        if len(values) > 1 and np.ptp(values) > 0:
            kde = stats.gaussian_kde(
                values, bw_method=plot_population_results_kde_bandwidth
            )
            density = kde(grid) + kde(-grid)
            ax.plot(
                grid,
                density,
                color=plot_full_color,
                linewidth=plot_population_results_line_width,
            )
            density_max = max(density_max, density.max())
        mean = values.mean()
        ax.axvline(
            mean,
            color=plot_full_color,
            linestyle="--",
            linewidth=plot_population_results_line_width,
        )
        ax.text(
            0.98,
            0.95,
            f"n = {len(values)}; excluded = {excluded}\nMean = {mean:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        strip_depth = density_max * plot_population_results_point_strip_depth
        for population in settings["populations"]:
            population_values = dataset_table.loc[
                dataset_table.population.eq(population), metric
            ]
            population_values = population_values.dropna().to_numpy()
            point_y = -rng.uniform(
                strip_depth / 4, strip_depth * 3 / 4, len(population_values)
            )
            ax.scatter(
                population_values,
                point_y,
                color=plot_population_results_population_colors[population],
                s=plot_population_results_point_size,
                alpha=plot_population_results_point_alpha,
                linewidths=0,
            )
        density_upper = density_max * 1.25
        tick_locator = mticker.MaxNLocator(
            nbins=plot_population_results_density_tick_count
        )
        density_ticks = tick_locator.tick_values(0, density_upper)
        density_ticks = density_ticks[
            (density_ticks >= 0) & (density_ticks <= density_upper)
        ]
        ax.set_yticks(density_ticks)
        ax.set(
            title=settings["label"],
            xlabel=xlabel,
            ylabel="Density",
            xlim=(bins[0], bins[-1]),
            ylim=(-strip_depth, density_upper),
        )
        ax.axhline(0, color=plot_full_color, linewidth=0.5)
        ax.grid(axis="y", alpha=0.2)
    fig.legend(handles=gap_population_handles, loc="outside lower center", ncol=3)
    fig.suptitle(f"{title}\nSolid: kernel density; dashed: mean; points: haplotypes")
    return fig


def plot_outlier_switch_genome(
    dataset: nb_utils.PathDiagnosticsDatasetDiagnostics,
    analysis: nb_utils.OutlierPathAnalysis,
    *,
    category_cmap,
    category_norm,
    coverage_handles,
    plot_full_color="#222222",
    plot_genome_unit_bp=1000000,
    plot_genome_unit_label="Mb",
    plot_outliers_genome_figsize=[16, 8],
    plot_population_results_point_size=12,
    plot_population_results_population_colors={
        "YRI": "tab:blue",
        "AFR": "tab:blue",
        "CEU": "tab:orange",
        "EUR": "tab:orange",
        "CHB": "tab:gray",
        "EAS": "tab:gray",
    },
    plot_stitched_color="#6A3D9A",
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> plt.Figure:
    """Locate gaps, parent differences, switch gains/losses, and cumulative excess."""
    case = dataset.primary
    edges = dataset.site_edges / plot_genome_unit_bp
    fig, axes = plt.subplots(
        5,
        1,
        sharex=True,
        figsize=plot_outliers_genome_figsize,
        gridspec_kw={"height_ratios": [0.3, 0.3, 1, 1, 1]},
        layout="constrained",
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    draw_strip(
        axes[0],
        edges,
        (~case.covered).astype(int),
        source=True,
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    draw_strip(
        axes[1],
        edges,
        case.categories,
        category_cmap=category_cmap,
        category_norm=category_norm,
        source_cmap=source_cmap,
        source_norm=source_norm,
    )
    axes[1].set_ylabel("Agreement", rotation=0, ha="right")
    full_changes = case.full_parents[1:] != case.full_parents[:-1]
    stitched_changes = case.stitched_parents[1:] != case.stitched_parents[:-1]
    delta = stitched_changes.astype(int) - full_changes.astype(int)
    x = dataset.positions[1:] / plot_genome_unit_bp
    for selected, level, color, label in (
        (delta > 0, 1, plot_stitched_color, "Stitched switch only"),
        (delta < 0, -1, plot_full_color, "Full switch only"),
    ):
        selected_count = np.count_nonzero(selected)
        axes[2].scatter(
            x[selected],
            np.full(selected_count, level),
            marker="|",
            s=100,
            color=color,
            label=label,
        )
    axes[2].set(
        yticks=[-1, 1], yticklabels=["Full only", "Stitched only"], ylim=(-1.5, 1.5)
    )
    axes[2].set_ylabel("Switch\npositions")
    axes[3].step(x, np.cumsum(delta), where="post", color=plot_stitched_color)
    axes[3].set_ylabel("Cumulative\nextra switches")
    gap_midpoints = (analysis.gaps.start_bp + analysis.gaps.end_bp) / 2
    axes[4].scatter(
        gap_midpoints / plot_genome_unit_bp,
        analysis.gaps.extra_gap_and_seam_switches,
        color=plot_population_results_population_colors[case.summary["population"]],
        s=plot_population_results_point_size,
    )
    axes[4].set_ylabel("Gap + seams\nnet extra switches")
    for ax in axes[2:]:
        ax.axhline(0, color="0.5", linewidth=0.5)
        ax.grid(alpha=0.2)
    axes[-1].set(
        xlim=(edges[0], edges[-1]),
        xlabel=f"Genomic position ({plot_genome_unit_label})",
    )
    fig.suptitle(
        f"{dataset.label}: {case.label} ({case.summary['population']}) — where switches change"
    )
    fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
    return fig


def plot_outlier_gap_properties(
    dataset: nb_utils.PathDiagnosticsDatasetDiagnostics,
    analysis: nb_utils.OutlierPathAnalysis,
    *,
    config,
    plot_outliers_gap_figsize=[16, 6],
    plot_population_results_point_alpha=0.6,
    plot_population_results_population_colors={
        "YRI": "tab:blue",
        "AFR": "tab:blue",
        "CEU": "tab:orange",
        "EUR": "tab:orange",
        "CHB": "tab:gray",
        "EAS": "tab:gray",
    },
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
) -> plt.Figure:
    """Show gap length against its local switch excess and parent agreement."""
    case = dataset.primary
    gaps = analysis.gaps
    fig, axes = plt.subplots(
        1, 2, figsize=plot_outliers_gap_figsize, layout="constrained", dpi=dpi
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    color = plot_population_results_population_colors[case.summary["population"]]
    for ax, metric, ylabel in (
        (
            axes[0],
            "extra_gap_and_seam_switches",
            "Net extra switches in gap + adjacent seams",
        ),
        (axes[1], "agreement_fraction", "Parent agreement within gap"),
    ):
        ax.scatter(
            gaps.num_sites,
            gaps[metric],
            color=color,
            alpha=plot_population_results_point_alpha,
        )
        ax.set(xscale="log", xlabel="Sites in gap (log scale)", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0].axhline(0, color="0.5", linewidth=0.5)
    axes[1].set_ylim(-0.05, 1.05)
    ranked = gaps.sort_values("extra_switches_per_site", ascending=False)
    labelled = ranked.head(config["outliers"]["gaps_to_display"])
    for (num_sites, extra), group in labelled.groupby(
        ["num_sites", "extra_gap_and_seam_switches"]
    ):
        label = ", ".join((f"g{gap}" for gap in group.gap))
        axes[0].annotate(
            label,
            (num_sites, extra),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize="small",
        )
    fig.suptitle(
        f"{dataset.label}: {case.label} ({case.summary['population']}) — individual gaps"
    )
    return fig


@dataclasses.dataclass
class Panels:
    """A figure and its dataset-by-mismatch axes."""

    figure: plt.Figure
    axes: np.ndarray


def _parameter_panels(
    title: str,
    by_curve: bool = False,
    *,
    config,
    curve_values,
    plot_panel_height=4,
    plot_panel_width=7,
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Keep dataset columns and, where requested, mismatch rows consistent."""
    nrows = len(curve_values) if by_curve else 1
    figsize = (plot_panel_width * len(config["datasets"]), plot_panel_height * nrows)
    fig, axes = plt.subplots(
        nrows,
        len(config["datasets"]),
        figsize=figsize,
        squeeze=False,
        sharey="row",
        layout="constrained",
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for row in range(nrows):
        for column, settings in enumerate(config["datasets"]):
            title_text = settings["label"]
            if by_curve:
                title_text += " · " + curve_label.format(value=curve_values[row])
            axes[row, column].set_title(title_text)
    fig.suptitle(title)
    return Panels(fig, axes)


def _hourglass_line(
    ax,
    table,
    metric: str,
    color: str,
    label: str,
    *,
    x_values,
    plot_hourglass_alpha=0.3,
    plot_hourglass_width=0.12,
    plot_line_width=1.5,
    plot_marker_size=4,
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
):
    """Plot means and compact hourglass patches spanning mean ± one SEM.

    This uses the same haplotype SEM calculation as the earlier hyperparameter
    error bars. The two triangles meet at the mean; their tips mark ±SEM.
    """
    summary = table.groupby(x_parameter)[metric].agg(["mean", "sem", "count"])
    ax.plot(
        summary.index,
        summary["mean"],
        color=color,
        label=label,
        marker="o",
        markersize=plot_marker_size,
        linewidth=plot_line_width,
    )
    for x_value, values in summary.iterrows():
        mean = values["mean"]
        sem = values["sem"]
        if np.isfinite(sem):
            width = plot_hourglass_width
            vertices = [
                (x_value - width, mean - sem),
                (x_value + width, mean - sem),
                (x_value, mean),
                (x_value + width, mean + sem),
                (x_value - width, mean + sem),
                (x_value, mean),
            ]
            patch = mpatches.Polygon(
                vertices, facecolor=color, edgecolor=color, alpha=plot_hourglass_alpha
            )
            ax.add_patch(patch)
    ax.set_xticks(x_values)
    ax.set_xlabel(x_label)
    ax.legend()


def plot_parameter_metric(
    metric: str,
    title: str,
    ylabel: str,
    *,
    comparisons,
    config,
    x_values,
    curve_values,
    curve_colors,
    plot_hourglass_alpha=0.3,
    plot_hourglass_width=0.12,
    plot_line_width=1.5,
    plot_marker_size=4,
    plot_panel_height=4,
    plot_panel_width=7,
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Plot haplotype metric means and SEMs across a saved parameter grid."""
    panels = _parameter_panels(
        title,
        config=config,
        curve_values=curve_values,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        x_parameter=x_parameter,
        curve_parameter=curve_parameter,
        x_label=x_label,
        curve_label=curve_label,
        x_value_label=x_value_label,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    fig = panels.figure
    axes = panels.axes
    for column, settings in enumerate(config["datasets"]):
        ax = axes[0, column]
        dataset = comparisons.loc[comparisons.dataset.eq(settings["name"])]
        for curve_value in curve_values:
            table = dataset.loc[dataset[curve_parameter].eq(curve_value)]
            _hourglass_line(
                ax,
                table,
                metric,
                curve_colors[curve_value],
                curve_label.format(value=curve_value),
                x_values=x_values,
                plot_hourglass_alpha=plot_hourglass_alpha,
                plot_hourglass_width=plot_hourglass_width,
                plot_line_width=plot_line_width,
                plot_marker_size=plot_marker_size,
                x_parameter=x_parameter,
                curve_parameter=curve_parameter,
                x_label=x_label,
                curve_label=curve_label,
                x_value_label=x_value_label,
            )
        ax.set_ylabel(ylabel)
    return fig


def plot_parameter_coverage(
    region: str,
    title: str,
    *,
    comparisons,
    config,
    x_values,
    curve_values,
    plot_line_width=1.5,
    plot_marker_size=4,
    plot_panel_height=4,
    plot_panel_width=7,
    plot_population_colors={"African": "#0072B2", "Non-African": "#E69F00"},
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Average physical window coverage within analogous population groups."""
    panels = _parameter_panels(
        title,
        by_curve=True,
        config=config,
        curve_values=curve_values,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        x_parameter=x_parameter,
        curve_parameter=curve_parameter,
        x_label=x_label,
        curve_label=curve_label,
        x_value_label=x_value_label,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    fig = panels.figure
    axes = panels.axes
    metric = f"{region}_coverage_percentage"
    for row, curve_value in enumerate(curve_values):
        for column, settings in enumerate(config["datasets"]):
            selected = comparisons.dataset.eq(settings["name"])
            selected &= comparisons[curve_parameter].eq(curve_value)
            table = comparisons.loc[selected]
            ax = axes[row, column]
            for group, color in plot_population_colors.items():
                group_table = table.loc[table.population_group.eq(group)]
                means = group_table.groupby(x_parameter)[metric].mean()
                ax.plot(
                    means.index,
                    means,
                    color=color,
                    label=group,
                    marker="o",
                    markersize=plot_marker_size,
                    linewidth=plot_line_width,
                )
            ax.set(
                xlabel=x_label,
                ylabel=f"Genome covered by {region} regions (%)",
                xticks=x_values,
                ylim=(0, 100),
            )
            ax.legend()
    return fig


def plot_parameter_length_distribution(
    region: str,
    title: str,
    *,
    x_colors,
    config,
    x_values,
    lengths,
    curve_values,
    plot_histogram_alpha=0.16,
    plot_histogram_bins=40,
    plot_kde_bandwidth="scott",
    plot_kde_grid_size=512,
    plot_length_display_quantile=0.99,
    plot_line_width=1.5,
    plot_panel_height=4,
    plot_panel_width=7,
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Plot common-bin interval histograms and reflected KDE count curves."""
    panels = _parameter_panels(
        title,
        by_curve=True,
        config=config,
        curve_values=curve_values,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        x_parameter=x_parameter,
        curve_parameter=curve_parameter,
        x_label=x_label,
        curve_label=curve_label,
        x_value_label=x_value_label,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    fig = panels.figure
    axes = panels.axes
    for row, curve_value in enumerate(curve_values):
        selected = lengths.region.eq(region) & lengths[curve_parameter].eq(curve_value)
        row_table = lengths.loc[selected]
        upper = row_table.num_sites.quantile(plot_length_display_quantile)
        bins = np.linspace(0, upper + 1, plot_histogram_bins + 1)
        bin_width = bins[1] - bins[0]
        grid = np.linspace(0, bins[-1], plot_kde_grid_size)
        for column, settings in enumerate(config["datasets"]):
            ax = axes[row, column]
            dataset = row_table.loc[row_table.dataset.eq(settings["name"])]
            for x_value in x_values:
                values = dataset.loc[
                    dataset[x_parameter].eq(x_value), "num_sites"
                ].to_numpy()
                color = x_colors[x_value]
                tail_count = np.count_nonzero(values > bins[-1])
                label = x_value_label.format(value=x_value)
                ax.hist(values, bins=bins, color=color, alpha=plot_histogram_alpha)
                if len(values) > 1 and np.ptp(values) > 0:
                    kde = stats.gaussian_kde(values, bw_method=plot_kde_bandwidth)
                    density = kde(grid) + kde(-grid)
                    counts = density * len(values) * bin_width
                    ax.plot(
                        grid,
                        counts,
                        color=color,
                        label=label,
                        linewidth=plot_line_width,
                    )
                elif len(values) > 0:
                    ax.axvline(values[0], color=color, label=label)
            ax.set(
                xlabel=f"{region.capitalize()} region length (number of sites)",
                ylabel="Count",
                xlim=(0, bins[-1]),
                ylim=(0, None),
            )
            ax.legend(title=x_label)
    return fig


def plot_parameter_switch_composition(
    *,
    comparisons,
    config,
    x_values,
    curve_values,
    plot_bar_width=0.65,
    plot_full_stitched_gap_bar_widths=1,
    plot_group_label_y=-0.2,
    plot_panel_height=4,
    plot_panel_width=7,
    plot_switch_categories=["focal", "gap", "seam"],
    plot_switch_colors=["#009E73", "#E69F00", "#7B61A8"],
    plot_switch_labels=["Within focal coverage", "Within gaps", "At gap seams"],
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
    baseline_partition,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Show raw mean switch counts, with one explicitly partitioned full bar."""
    panels = _parameter_panels(
        "Switch composition changes",
        by_curve=True,
        config=config,
        curve_values=curve_values,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        x_parameter=x_parameter,
        curve_parameter=curve_parameter,
        x_label=x_label,
        curve_label=curve_label,
        x_value_label=x_value_label,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    fig = panels.figure
    axes = panels.axes
    baseline_cutoff = baseline_partition
    assert baseline_cutoff in x_values
    x = _parameter_bar_positions(
        x_values=x_values,
        plot_bar_width=plot_bar_width,
        plot_full_stitched_gap_bar_widths=plot_full_stitched_gap_bar_widths,
        x_parameter=x_parameter,
        curve_parameter=curve_parameter,
        x_label=x_label,
        curve_label=curve_label,
        x_value_label=x_value_label,
    )
    stitched_x = x[1:]
    for row, curve_value in enumerate(curve_values):
        for column, settings in enumerate(config["datasets"]):
            selected = comparisons.dataset.eq(settings["name"])
            selected &= comparisons[curve_parameter].eq(curve_value)
            table = comparisons.loc[selected]
            baseline = table.loc[table[x_parameter].eq(baseline_cutoff)]
            ax = axes[row, column]
            bottom = np.zeros(len(x))
            for category, label, color in zip(
                plot_switch_categories,
                plot_switch_labels,
                plot_switch_colors,
                strict=True,
            ):
                full_mean = baseline[f"full_{category}_switches"].mean()
                stitched_means = table.groupby(x_parameter)[
                    f"stitched_{category}_switches"
                ].mean()
                heights = np.concatenate(
                    ([full_mean], stitched_means.loc[x_values].to_numpy())
                )
                ax.bar(
                    x,
                    heights,
                    bottom=bottom,
                    width=plot_bar_width,
                    color=color,
                    label=label,
                )
                bottom += heights
            full_label = f"Full HMM\n(AC {baseline_cutoff} partition)"
            ax.set(
                xticks=x,
                xticklabels=[full_label] + [str(x_value) for x_value in x_values],
                ylabel="Mean switches per haplotype",
                ylim=(0, None),
            )
            ax.text(
                stitched_x.mean(),
                plot_group_label_y,
                "Stitched HMM (AC cutoff)",
                transform=ax.get_xaxis_transform(),
                ha="center",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    return fig


def save_figure(fig, filename: str, *, output_dir):
    """Export each displayed figure beside its underlying cohort tables."""
    fig.savefig(output_dir / f"{filename}.png")
    plt.show()
    plt.close(fig)


def plot_parameter_metric_rows(
    table,
    metrics: dict,
    title: str,
    *,
    config,
    x_values,
    curve_values,
    curve_colors,
    plot_line_width=1.5,
    plot_marker_size=4,
    plot_panel_height=4,
    plot_panel_width=7,
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Use metric rows and dataset columns with mismatch-coloured curves."""
    figsize = (
        plot_panel_width * len(config["datasets"]),
        plot_panel_height * len(metrics),
    )
    fig, axes = plt.subplots(
        len(metrics),
        len(config["datasets"]),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey="row",
        layout="constrained",
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for row, (metric, ylabel) in enumerate(metrics.items()):
        for column, settings in enumerate(config["datasets"]):
            ax = axes[row, column]
            dataset_table = table.loc[table.dataset.eq(settings["name"])]
            for curve_value in curve_values:
                selected = dataset_table[curve_parameter].eq(curve_value)
                observations = dataset_table.loc[selected].sort_values(x_parameter)
                ax.plot(
                    observations[x_parameter],
                    observations[metric],
                    color=curve_colors[curve_value],
                    label=curve_label.format(value=curve_value),
                    marker="o",
                    markersize=plot_marker_size,
                    linewidth=plot_line_width,
                )
            ax.set(
                xlabel=x_label, ylabel=ylabel, xticks=x_values, title=settings["label"]
            )
            ax.legend()
    fig.suptitle(title)
    return Panels(fig, axes)


def plot_parameter_statistic_differences(
    *,
    config,
    x_values,
    curve_values,
    curve_colors,
    plot_line_width=1.5,
    plot_marker_size=4,
    plot_panel_height=4,
    plot_panel_width=7,
    ts_differences,
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Plot signed raw edge/tree count changes relative to Full HMM."""
    metrics = {
        "num_edges_difference": "Edges: stitched HMM − Full HMM",
        "num_trees_difference": "Trees: stitched HMM − Full HMM",
    }
    panels = plot_parameter_metric_rows(
        ts_differences,
        metrics,
        "Comparison of TS statistics",
        config=config,
        x_values=x_values,
        curve_values=curve_values,
        curve_colors=curve_colors,
        plot_line_width=plot_line_width,
        plot_marker_size=plot_marker_size,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        x_parameter=x_parameter,
        curve_parameter=curve_parameter,
        x_label=x_label,
        curve_label=curve_label,
        x_value_label=x_value_label,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    for ax in panels.axes.flat:
        ax.axhline(0, color="gray", linestyle="--", linewidth=plot_line_width)
    fig = panels.figure
    return fig


def plot_parameter_arg_metrics(
    *,
    arg_comparisons,
    config,
    x_values,
    curve_values,
    curve_colors,
    plot_line_width=1.5,
    plot_marker_size=4,
    plot_panel_height=4,
    plot_panel_width=7,
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Plot ARF and TPR against Full HMM for both datasets."""
    metrics = {"arf": "ARF", "tpr": "TPR"}
    panels = plot_parameter_metric_rows(
        arg_comparisons,
        metrics,
        "Comparing ARG metrics",
        config=config,
        x_values=x_values,
        curve_values=curve_values,
        curve_colors=curve_colors,
        plot_line_width=plot_line_width,
        plot_marker_size=plot_marker_size,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        x_parameter=x_parameter,
        curve_parameter=curve_parameter,
        x_label=x_label,
        curve_label=curve_label,
        x_value_label=x_value_label,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    return panels.figure


def plot_parameter_truth_arg_metrics(
    *,
    x_values,
    curve_values,
    curve_colors,
    plot_line_width=1.5,
    plot_marker_size=4,
    plot_panel_height=4,
    plot_panel_width=7,
    truth_comparisons,
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
    baseline_x=1,
    stitched_method="stitched_hmm",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Show Full HMM and stitched simulation ARF/TPR against biological truth."""
    figsize = (plot_panel_width * 2, plot_panel_height)
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained", dpi=dpi)
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    full = truth_comparisons.loc[truth_comparisons.comparison_method.eq("full_hmm")]
    full_cutoff = baseline_x
    metrics = {"arf": "ARF", "tpr": "TPR"}
    for ax, (metric, ylabel) in zip(axes, metrics.items(), strict=True):
        baseline = full[metric].item()
        for curve_value in curve_values:
            selected = truth_comparisons.comparison_method.eq(stitched_method)
            selected &= truth_comparisons[curve_parameter].eq(curve_value)
            observations = truth_comparisons.loc[selected].sort_values(x_parameter)
            x = np.concatenate(([full_cutoff], observations[x_parameter].to_numpy()))
            y = np.concatenate(([baseline], observations[metric].to_numpy()))
            ax.plot(
                x,
                y,
                color=curve_colors[curve_value],
                marker="o",
                markersize=plot_marker_size,
                linewidth=plot_line_width,
                label=curve_label.format(value=curve_value),
            )
        ticks = [full_cutoff, *x_values]
        tick_labels = [
            f"{full_cutoff}\n(Full HMM)",
            *[str(x_value) for x_value in x_values],
        ]
        ax.set(
            xlabel=x_label,
            ylabel=ylabel,
            xticks=ticks,
            xticklabels=tick_labels,
            title="Simulation vs true ARG",
        )
        ax.legend()
    fig.suptitle("ARG metrics against the true tree sequence")
    return fig


def panels(
    columns=1,
    rows=1,
    *,
    plot_panel_height=4.7,
    plot_panel_width=7,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Use the same panel sizing across the notebook."""
    fig, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        layout="constrained",
        figsize=(plot_panel_width * columns, plot_panel_height * rows),
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    return Panels(fig, axes)


def benchmark_save_figure(fig, name, *, output_dir, plot_export_dpi=200):
    """Keep displayed plots and reusable PNG/PDF exports identical."""
    fig.savefig(output_dir / f"{name}.png", dpi=plot_export_dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.show()


def phase_legend(
    ax,
    settings,
    title=None,
    order=None,
    *,
    plot_phase_legend_edgecolor="#333333",
    plot_phase_legend_facecolor="#F5F5F5",
    plot_phase_legend_font_size=11,
    plot_phase_legend_handleheight=1.2,
    plot_phase_legend_handlelength=2.4,
    plot_phase_legend_labelspacing=0.6,
    **kwargs,
):
    """List stack phases top to bottom, with optional phase-specific colours."""
    if order is None:
        order = list(reversed(settings))
    handles = []
    for phase in order:
        values = settings[phase]
        handle = mpatches.Patch(
            facecolor=values.get("legend_facecolor", plot_phase_legend_facecolor),
            edgecolor=plot_phase_legend_edgecolor,
            hatch=values["hatch"],
            label=values["label"],
        )
        handles.append(handle)
    ax.legend(
        handles=handles,
        title=title,
        fontsize=plot_phase_legend_font_size,
        handlelength=plot_phase_legend_handlelength,
        handleheight=plot_phase_legend_handleheight,
        labelspacing=plot_phase_legend_labelspacing,
        **kwargs,
    )


def distribution(
    ax,
    metric,
    xlabel,
    *,
    colors,
    comparisons,
    labels,
    plot_histogram_alpha=0.16,
    plot_histogram_bins=80,
    plot_kde_bandwidth="scott",
    plot_kde_grid_size=512,
    plot_line_width=1.7,
    stitched_methods,
):
    """Overlay fine count histograms and KDEs on the same count scale."""
    pooled = comparisons[metric].dropna().to_numpy()
    bins = np.linspace(pooled.min() - 0.5, pooled.max() + 0.5, plot_histogram_bins + 1)
    grid = np.linspace(bins[0], bins[-1], plot_kde_grid_size)
    bin_width = bins[1] - bins[0]
    for method in stitched_methods:
        table = comparisons.loc[comparisons.method.eq(method)]
        values = table[metric].dropna().to_numpy()
        color = colors[method]
        ax.hist(values, bins=bins, color=color, alpha=plot_histogram_alpha)
        kde = stats.gaussian_kde(values, bw_method=plot_kde_bandwidth)
        counts = kde(grid) * len(values) * bin_width
        ax.plot(
            grid, counts, color=color, linewidth=plot_line_width, label=labels[method]
        )
        mean = values.mean()
        ax.axvline(
            mean,
            color=color,
            linestyle="--",
            linewidth=plot_line_width,
            label=f"{labels[method]} mean: {mean:.2f}",
        )
        excluded = table[metric].isna().sum()
        if excluded > 0:
            ax.text(
                0.98,
                0.9 - 0.08 * stitched_methods.index(method),
                f"{labels[method]}: {excluded} undefined",
                ha="right",
                transform=ax.transAxes,
                color=color,
            )
    ax.set(xlabel=xlabel, ylabel="Haplotype count", xlim=(bins[0], bins[-1]))
    ax.legend(fontsize="small")


def parameter_grid_panel_grid(
    title,
    row_labels,
    sharey="row",
    by_cutoff=False,
    *,
    datasets,
    plot_panel_height=3.5,
    plot_panel_width=5,
    populations,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Always keep sample sizes in columns and the chosen quantity in rows."""
    fig, axes = plt.subplots(
        len(row_labels),
        len(datasets),
        squeeze=False,
        sharey=sharey,
        layout="constrained",
        figsize=(plot_panel_width * len(datasets), plot_panel_height * len(row_labels)),
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for row, label in enumerate(row_labels):
        for col, dataset in enumerate(datasets):
            cohort = populations.loc[populations.dataset.eq(dataset["name"])]
            n = cohort.sample_id.nunique()
            row_label = nb_utils.cutoff_label(label, dataset) if by_cutoff else label
            axes[row, col].set_title(f"{n:,} individuals\n{row_label}")
            axes[row, col].tick_params(
                axis="y", which="both", left=True, labelleft=True
            )
    fig.suptitle(title)
    return Panels(fig, axes)


def zoom_row(axes, *, plot_range_margin=0.08):
    """Zoom to all displayed curves/SEMs across AFs and sample sizes, with margin."""
    limits = np.array([ax.dataLim.intervaly for ax in axes])
    finite_limits = limits[np.isfinite(limits).all(axis=1)]
    if len(finite_limits) == 0:
        return
    lower = finite_limits[:, 0].min()
    upper = finite_limits[:, 1].max()
    margin = (upper - lower) * plot_range_margin
    if margin == 0:
        margin = max(abs(lower), 1) * plot_range_margin
    for ax in axes:
        ax.set_ylim(lower - margin, upper + margin)


def parameter_grid_hourglass_line(
    ax,
    table,
    metric,
    color,
    label,
    *,
    minimum_sizes,
    plot_hourglass_alpha=0.3,
    plot_hourglass_width=0.07,
    plot_line_width=1.5,
    plot_marker_size=4,
):
    """Reuse pt6's mean ± one haplotype SEM glyphs with categorical positions."""
    table = table.sort_values("min_focal_region_size")
    means = table[f"{metric}_mean"].to_numpy()
    sems = table[f"{metric}_sem"].to_numpy()
    x = np.arange(len(table))
    ax.plot(
        x,
        means,
        color=color,
        label=label,
        marker="o",
        markersize=plot_marker_size,
        linewidth=plot_line_width,
    )
    for position, mean, sem in zip(x, means, sems, strict=True):
        width = plot_hourglass_width
        vertices = [
            (position - width, mean - sem),
            (position + width, mean - sem),
            (position, mean),
            (position + width, mean + sem),
            (position - width, mean + sem),
            (position, mean),
        ]
        ax.add_patch(
            mpatches.Polygon(
                vertices, facecolor=color, edgecolor=color, alpha=plot_hourglass_alpha
            )
        )
    minimum_axis(ax, minimum_sizes=minimum_sizes)


def metric_lines(
    table,
    metrics,
    title,
    summary_statistics=False,
    baseline=False,
    *,
    af_cutoffs,
    cutoff_colors,
    datasets,
    minimum_sizes,
    plot_hourglass_alpha=0.3,
    plot_hourglass_width=0.07,
    plot_line_width=1.5,
    plot_marker_size=4,
    plot_panel_height=3.5,
    plot_panel_width=5,
    plot_range_margin=0.08,
    populations,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Metric rows and sample-size columns; one line per AF cutoff."""
    panels = parameter_grid_panel_grid(
        title,
        list(metrics.values()),
        datasets=datasets,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        populations=populations,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    for row, (metric, ylabel) in enumerate(metrics.items()):
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            selected = table.loc[table.dataset.eq(dataset["name"])]
            full = (
                selected.loc[selected.comparison_method.eq("full_hmm")]
                if baseline
                else None
            )
            has_curves = False
            for cutoff in af_cutoffs:
                curve = selected.loc[selected.af_cutoff.eq(cutoff)].sort_values(
                    "min_focal_region_size"
                )
                if curve.empty:
                    continue
                has_curves = True
                color = cutoff_colors[cutoff]
                label = nb_utils.cutoff_label(cutoff, dataset)
                if summary_statistics:
                    parameter_grid_hourglass_line(
                        ax,
                        curve,
                        metric,
                        color,
                        label,
                        minimum_sizes=minimum_sizes,
                        plot_hourglass_alpha=plot_hourglass_alpha,
                        plot_hourglass_width=plot_hourglass_width,
                        plot_line_width=plot_line_width,
                        plot_marker_size=plot_marker_size,
                    )
                else:
                    curve = curve.set_index("min_focal_region_size").reindex(
                        minimum_sizes
                    )
                    y = curve[metric].to_numpy()
                    if baseline:
                        full_value = full[metric].item() if not full.empty else np.nan
                        y = np.concatenate(([full_value], y))
                    ax.plot(
                        np.arange(len(y)),
                        y,
                        color=color,
                        label=label,
                        marker="o",
                        markersize=plot_marker_size,
                        linewidth=plot_line_width,
                    )
            if baseline and (not has_curves) and (not full.empty):
                ax.plot(
                    0,
                    full[metric].item(),
                    marker="o",
                    color="gray",
                    linestyle="none",
                    label="Full HMM",
                )
            minimum_axis(ax, baseline, minimum_sizes=minimum_sizes)
            ax.set_ylabel(ylabel)
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend()
        zoom_row(panels.axes[row], plot_range_margin=plot_range_margin)
    return panels.figure


def parameter_grid_save_figure(fig, name, *, output_dir):
    """Keep exported figures with plot caches and embed each in the notebook."""
    fig.savefig(output_dir / f"{name}.png")
    plt.show()
    plt.close(fig)


def minimum_size_kb_plot(
    *,
    config,
    datasets,
    minimum_colors,
    minimum_sizes,
    output_dir,
    plot_bar_width=0.65,
    plot_panel_height=3.5,
    plot_panel_width=5,
    plot_range_margin=0.08,
    populations,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    rows = []
    for dataset in datasets:
        with np.load(config["data_dir"] / dataset["sites"]) as site_data:
            positions = site_data["positions"]
        span_kb = (positions[-1] - positions[0]) / 1000
        mean_kb_per_site = span_kb / len(positions)
        for minimum in minimum_sizes:
            rows.append(
                {
                    "dataset": dataset["name"],
                    "num_sites": len(positions),
                    "site_span_kb": span_kb,
                    "min_focal_region_size": minimum,
                    "mean_min_region_size_kb": minimum * mean_kb_per_site,
                }
            )
    equivalents = pd.DataFrame(rows)
    equivalents.to_csv(output_dir / "minimum_size_kb.csv", index=False)
    panels = parameter_grid_panel_grid(
        "Mean physical equivalent of site-based minimum sizes",
        [""],
        datasets=datasets,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        populations=populations,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    for col, dataset in enumerate(datasets):
        ax = panels.axes[0, col]
        table = equivalents.loc[equivalents.dataset.eq(dataset["name"])]
        values = table["mean_min_region_size_kb"].to_numpy()
        bars = ax.bar(
            np.arange(len(minimum_sizes)),
            values,
            color=[minimum_colors[minimum] for minimum in minimum_sizes],
            width=plot_bar_width,
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3)
        minimum_axis(ax, minimum_sizes=minimum_sizes)
        ax.set_ylabel("Mean minimum region size (kb)")
        ax.margins(y=plot_range_margin * 2)
    return panels.figure


def focal_coverage(
    *,
    af_cutoffs,
    datasets,
    minimum_sizes,
    plot_panel_height=3.5,
    plot_panel_width=5,
    plot_population_colors={"African": "#0072B2", "Non-African": "#E69F00"},
    plot_range_margin=0.08,
    population_coverage,
    populations,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    panels = parameter_grid_panel_grid(
        "Focal region coverage",
        af_cutoffs,
        by_cutoff=True,
        datasets=datasets,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        populations=populations,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    for row, cutoff in enumerate(af_cutoffs):
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            selected = population_coverage.dataset.eq(
                dataset["name"]
            ) & population_coverage.af_cutoff.eq(cutoff)
            table = population_coverage.loc[selected]
            for group, color in plot_population_colors.items():
                curve = table.loc[table.population_group.eq(group)].sort_values(
                    "min_focal_region_size"
                )
                ax.plot(
                    np.arange(len(curve)),
                    curve["mean"],
                    color=color,
                    label=group,
                    marker="o",
                )
            minimum_axis(ax, minimum_sizes=minimum_sizes)
            ax.set_ylabel("Physical focal coverage (%)")
            ax.legend()
        zoom_row(panels.axes[row], plot_range_margin=plot_range_margin)
    return panels.figure


def focal_lengths(
    num_bins,
    *,
    af_cutoffs,
    cutoff_colors,
    datasets,
    lengths,
    minimum_colors,
    minimum_sizes,
    plot_baseline_minimum=1,
    plot_histogram_alpha=0.25,
    plot_kde_grid_size=512,
    plot_length_display_quantile=0.99,
    plot_panel_height=3.5,
    plot_panel_width=5,
    populations,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    panels = parameter_grid_panel_grid(
        "Focal region lengths · minimum size 1",
        af_cutoffs,
        sharey=False,
        by_cutoff=True,
        datasets=datasets,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        populations=populations,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    for row, cutoff in enumerate(af_cutoffs):
        selected = lengths.af_cutoff.eq(cutoff) & ~lengths.is_gap
        selected &= lengths.min_focal_region_size.eq(plot_baseline_minimum)
        table = lengths.loc[selected]
        quantile = nb_utils.paramsearch_weighted_quantile(
            table, plot_length_display_quantile
        )
        upper = max(quantile, max(minimum_sizes)) + 1
        bins = np.linspace(0, upper, num_bins + 1)
        grid = np.linspace(0, upper, plot_kde_grid_size)
        bin_width = bins[1] - bins[0]
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            values = table.loc[table.dataset.eq(dataset["name"])]
            total = values["count"].sum()
            tail = values.loc[values.num_sites.gt(upper), "count"].sum()
            color = cutoff_colors[cutoff]
            ax.hist(
                values.num_sites,
                bins=bins,
                weights=values["count"],
                color=color,
                alpha=plot_histogram_alpha,
                label=f"{total:,} regions; tail {tail:,}",
            )
            density = nb_utils.paramsearch_length_density(values, grid)
            ax.plot(grid, density * total * bin_width, color=color)
            for minimum, marker_color in minimum_colors.items():
                ax.axvline(
                    minimum, color=marker_color, linestyle="--", label=f"Min {minimum}"
                )
            ax.set(
                xlabel="Focal length (inference sites)",
                ylabel="Count per bin",
                xlim=(0, upper),
                ylim=(0, None),
            )
            ax.legend(fontsize="small")
    return panels.figure


def gap_lengths(
    *,
    af_cutoffs,
    datasets,
    lengths,
    minimum_colors,
    plot_kde_grid_size=512,
    plot_length_display_quantile=0.99,
    plot_line_width=1.5,
    plot_panel_height=3.5,
    plot_panel_width=5,
    plot_range_margin=0.08,
    populations,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    panels = parameter_grid_panel_grid(
        "Gap length distributions",
        af_cutoffs,
        by_cutoff=True,
        datasets=datasets,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        populations=populations,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    for row, cutoff in enumerate(af_cutoffs):
        table = lengths.loc[lengths.af_cutoff.eq(cutoff) & lengths.is_gap]
        upper = (
            nb_utils.paramsearch_weighted_quantile(table, plot_length_display_quantile)
            + 1
        )
        grid = np.linspace(0, upper, plot_kde_grid_size)
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            for minimum, color in minimum_colors.items():
                selected = table.dataset.eq(
                    dataset["name"]
                ) & table.min_focal_region_size.eq(minimum)
                values = table.loc[selected]
                median = nb_utils.paramsearch_weighted_quantile(values, 0.5)
                density = nb_utils.paramsearch_length_density(values, grid)
                total = values["count"].sum()
                tail = values.loc[values.num_sites.gt(upper), "count"].sum() / total
                ax.plot(
                    grid,
                    density,
                    color=color,
                    label=f"Min {minimum}; median {median:g}; tail {tail:.1%}",
                )
                ax.axvline(
                    median, color=color, linestyle="--", linewidth=plot_line_width
                )
            ax.set(
                xlabel="Gap length (inference sites)",
                ylabel="Density per inference site",
                xlim=(0, upper),
                ylim=(0, None),
            )
            ax.legend(fontsize="small")
        density_upper = max((ax.dataLim.ymax for ax in panels.axes[row]))
        for ax in panels.axes[row]:
            ax.set_ylim(0, density_upper * (1 + plot_range_margin))
    return panels.figure


def one_site_gaps(
    *,
    af_cutoffs,
    datasets,
    grid_keys,
    lengths,
    minimum_colors,
    minimum_sizes,
    output_dir,
    plot_bar_width=0.65,
    plot_panel_height=3.5,
    plot_panel_width=5,
    plot_range_margin=0.08,
    populations,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    gaps = lengths.loc[lengths.is_gap]
    totals = gaps.groupby(grid_keys)["count"].sum().rename("total_gaps")
    singles = gaps.loc[gaps.num_sites.eq(1)].groupby(grid_keys)["count"].sum()
    singles = singles.reindex(totals.index, fill_value=0).rename("one_site_gaps")
    counts = pd.concat([totals, singles], axis=1).reset_index()
    counts["one_site_percentage"] = 100 * counts.one_site_gaps / counts.total_gaps
    counts.to_csv(output_dir / "one_site_gap_counts.csv", index=False)
    panels = parameter_grid_panel_grid(
        "Number of one-site gaps · labels show percentage of all gaps",
        af_cutoffs,
        by_cutoff=True,
        datasets=datasets,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        populations=populations,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    for row, cutoff in enumerate(af_cutoffs):
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            selected = counts.dataset.eq(dataset["name"]) & counts.af_cutoff.eq(cutoff)
            table = (
                counts.loc[selected]
                .set_index("min_focal_region_size")
                .loc[minimum_sizes]
            )
            bars = ax.bar(
                np.arange(len(minimum_sizes)),
                table.one_site_gaps,
                color=[minimum_colors[minimum] for minimum in minimum_sizes],
                width=plot_bar_width,
            )
            ax.bar_label(
                bars,
                labels=[f"{value:.1f}%" for value in table.one_site_percentage],
                padding=3,
            )
            minimum_axis(ax, minimum_sizes=minimum_sizes)
            ax.set_ylabel("Number of one-site gaps")
            ax.ticklabel_format(axis="y", style="plain")
            ax.margins(y=plot_range_margin * 2)
    return panels.figure


def parameter_grid_switch_composition(
    *,
    af_cutoffs,
    datasets,
    minimum_sizes,
    plot_bar_width=0.65,
    plot_baseline_minimum=1,
    plot_panel_height=3.5,
    plot_panel_width=5,
    plot_switch_categories=["focal", "gap", "seam"],
    plot_switch_colors=["#009E73", "#E69F00", "#7B61A8"],
    plot_switch_labels=["Within focal coverage", "Within gaps", "At gap seams"],
    populations,
    summary,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    panels = parameter_grid_panel_grid(
        "Switch composition",
        af_cutoffs,
        by_cutoff=True,
        datasets=datasets,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        populations=populations,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    x = np.concatenate(([0], np.arange(len(minimum_sizes)) + 2 * plot_bar_width))
    for row, cutoff in enumerate(af_cutoffs):
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            selected = summary.dataset.eq(dataset["name"]) & summary.af_cutoff.eq(
                cutoff
            )
            table = summary.loc[selected].sort_values("min_focal_region_size")
            baseline = table.loc[table.min_focal_region_size.eq(plot_baseline_minimum)]
            bottom = np.zeros(len(x))
            for category, color, label in zip(
                plot_switch_categories,
                plot_switch_colors,
                plot_switch_labels,
                strict=True,
            ):
                full = baseline[f"full_{category}_switches_mean"].item()
                heights = np.concatenate(
                    ([full], table[f"stitched_{category}_switches_mean"].to_numpy())
                )
                ax.bar(
                    x,
                    heights,
                    bottom=bottom,
                    width=plot_bar_width,
                    color=color,
                    label=label,
                )
                bottom += heights
            ax.set(
                xticks=x,
                xticklabels=["Full HMM", *[str(size) for size in minimum_sizes]],
                xlabel="Minimum focal size (inference sites)",
                ylabel="Mean switches per haplotype",
            )
    handles, labels = panels.axes[0, 0].get_legend_handles_labels()
    panels.figure.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    return panels.figure


def mutation_age_plots(
    table,
    groups,
    statistic="pearson_r",
    *,
    cutoff_colors,
    datasets,
    minimum_sizes,
    plot_panel_height=3.5,
    plot_panel_width=5,
    plot_range_margin=0.08,
    populations,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Reference/bin rows and sample-size columns with consistent AF colours."""
    references = {
        "true_arg": "Inferred vs true ages",
        "full_hmm": "Stitched vs Full HMM ages",
    }
    row_labels = [
        f"{label} · {group}" for group in groups for label in references.values()
    ]
    statistic_label = {"pearson_r": "Pearson r", "spearman_rho": "Spearman ρ"}[
        statistic
    ]
    panels = parameter_grid_panel_grid(
        f"Mutation-age correlations · {statistic_label}",
        row_labels,
        datasets=datasets,
        plot_panel_height=plot_panel_height,
        plot_panel_width=plot_panel_width,
        populations=populations,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    for group_index, group in enumerate(groups):
        for reference_index, (reference, label) in enumerate(references.items()):
            row = 2 * group_index + reference_index
            table_group = table.loc[
                table.frequency_bin.eq(group) & table.reference_method.eq(reference)
            ]
            baseline = reference == "true_arg"
            for col, dataset in enumerate(datasets):
                ax = panels.axes[row, col]
                selected = table_group.loc[table_group.dataset.eq(dataset["name"])]
                for cutoff, color in cutoff_colors.items():
                    curve = selected.loc[selected.af_cutoff.eq(cutoff)].sort_values(
                        "min_focal_region_size"
                    )
                    y = curve[statistic].to_numpy()
                    if baseline:
                        full = selected.loc[
                            selected.comparison_method.eq("full_hmm"), statistic
                        ].item()
                        y = np.concatenate(([full], y))
                    ax.plot(
                        np.arange(len(y)),
                        y,
                        color=color,
                        label=nb_utils.cutoff_label(cutoff, dataset),
                        marker="o",
                    )
                minimum_axis(ax, baseline, minimum_sizes=minimum_sizes)
                n = selected.matched_mutations.iloc[0]
                ax.set_ylabel(f"{statistic_label} ({n:,} mutations)")
                ax.legend()
            zoom_row(panels.axes[row], plot_range_margin=plot_range_margin)
    return panels.figure


def _parameter_bar_positions(
    *,
    x_values,
    plot_bar_width=0.65,
    plot_full_stitched_gap_bar_widths=1,
    x_parameter="ac_cutoff",
    curve_parameter="max_mismatches",
    x_label="AC cutoff",
    curve_label="{value} mismatches allowed",
    x_value_label="AC {value}",
) -> np.ndarray:
    """Leave one configured bar width of clear space after the Full HMM bar."""
    bar_width = plot_bar_width
    clear_gap = bar_width * plot_full_stitched_gap_bar_widths
    first_stitched = bar_width + clear_gap
    stitched_x = np.arange(len(x_values)) + first_stitched
    return np.concatenate(([0], stitched_x))


def minimum_axis(ax, baseline=False, *, minimum_sizes):
    """Use equally spaced labelled sizes, keeping Full HMM distinct from min 1."""
    labels = [str(value) for value in minimum_sizes]
    if baseline:
        labels = ["Full HMM", *labels]
    ax.set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        xlabel="Minimum focal size (inference sites; equally spaced)",
    )


def plot_mean_proportion_of_ancestors_included(
    *,
    ancestor_set_sizes,
    cutoffs,
    dataset_names,
    max_ac,
    figsize=(12, 4),
    marker="o",
    markersize=3,
    alpha=0.25,
):
    """Mean proportion of ancestors included. Presentation choices are supplied by the notebook."""
    ancestor_set_figure, axes = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True, layout="constrained"
    )
    for ax, label in zip(axes, dataset_names, strict=True):
        sizes = ancestor_set_sizes[label]
        ax.plot(
            sizes["focal_ac"].to_numpy(),
            sizes["mean_proportion_included"].to_numpy(),
            marker=marker,
            markersize=markersize,
        )
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 5, 10, 20, 50, 100, max_ac])
        ax.set_xticklabels(["2", "3", "5", "10", "20", "50", "100", str(max_ac)])
        ax.set_xlim(cutoffs[0], cutoffs[-1])
        ax.set_xlabel("Maximum focal AC")
        ax.set_ylabel("Mean proportion of ancestors included")
        ax.set_title(label)
        ax.grid(alpha=alpha)
    axes[0].set_ylim(bottom=0)
    plt.show()


def plot_cumulative_inferred_ancestors_by_focal_frequency(
    *,
    ac_cutoffs,
    dataset_names,
    focal_ancestors_by_dataset,
    haplotype_counts,
    figsize=(14, 5),
):
    """Cumulative inferred ancestors by focal frequency. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True, layout="constrained"
    )
    for ax, label in zip(axes, dataset_names, strict=True):
        plot_cumulative_focal_ancestors(
            ax, focal_ancestors_by_dataset[label], haplotype_counts[label], ac_cutoffs
        )
        ax.set_title(label)
    plt.show()


def plot_focal_coverage_across_haploid_samples(
    *,
    ac_cutoffs,
    dataset_names,
    focal_stats_by_dataset,
    num_haploid_samples,
    figsize=(14, 5),
):
    """Focal coverage across haploid samples. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True, layout="constrained"
    )
    frequency_labels = [f"{cutoff / num_haploid_samples:g}" for cutoff in ac_cutoffs]
    for ax, label in zip(axes, dataset_names, strict=True):
        stats = focal_stats_by_dataset[label]
        coverage_groups = []
        for cutoff in ac_cutoffs:
            cutoff_rows = stats.loc[stats["ac_cutoff"] == cutoff]
            coverage_groups.append(cutoff_rows["fraction_covered_sites"].to_numpy())
        ax.boxplot(coverage_groups, tick_labels=frequency_labels)
        ax.set_xlabel("Cumulative focal frequency cutoff")
        ax.set_ylabel("Fraction of covered inference sites per haplotype")
        ax.set_ylim(0, 1)
        ax.set_title(label)
    plt.show()


def plot_match_score_and_focal_coverage_at_each_frequency_cutoff(
    *,
    ac_cutoffs,
    combined_stats,
    dataset_names,
    focal_stats_by_dataset,
    num_haploid_samples,
    population_names,
):
    """Match score and focal coverage at each frequency cutoff. Presentation choices are supplied by the notebook."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    population_colors = {
        population: color_cycle[index % len(color_cycle)]
        for index, population in enumerate(population_names)
    }
    score_range = combined_stats["path_log_likelihood"].agg(["min", "max"])
    score_padding = (score_range["max"] - score_range["min"]) * 0.05
    score_bins = np.linspace(
        score_range["min"] - score_padding, score_range["max"] + score_padding, 21
    )
    coverage_bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(
        len(ac_cutoffs),
        2,
        figsize=(15, 6 * len(ac_cutoffs)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.08, right=0.97, bottom=0.03, top=0.98, wspace=0.4, hspace=0.5
    )
    for row, cutoff in enumerate(ac_cutoffs):
        frequency = cutoff / num_haploid_samples
        for column, label in enumerate(dataset_names):
            plot_focal_match_stats(
                axes[row, column],
                focal_stats_by_dataset[label],
                cutoff,
                score_bins,
                coverage_bins,
                population_colors,
                title=f"{label}: focal frequency ≤ {frequency:g} (AC ≤ {cutoff})",
            )
            axes[row, column].tick_params(labelbottom=True, labelleft=True)
    plt.show()


@dataclasses.dataclass
class DiagnosingStitchedFocalAndGapMatchesResult:
    category_cmap: object
    category_norm: object
    source_cmap: object
    source_norm: object


def make_path_colormaps():
    """Diagnosing stitched focal and gap matches. Presentation choices are supplied by the notebook."""
    category_cmap = mcolors.ListedColormap(["#009E73", "#CC79A7", "#56B4E9", "#D55E00"])
    category_norm = mcolors.BoundaryNorm(np.arange(5) - 0.5, category_cmap.N)
    source_cmap = mcolors.ListedColormap(["#B8DED4", "#FFE3A3"])
    source_norm = mcolors.BoundaryNorm(np.arange(3) - 0.5, source_cmap.N)
    return DiagnosingStitchedFocalAndGapMatchesResult(
        category_cmap, category_norm, source_cmap, source_norm
    )


@dataclasses.dataclass
class CoverageAgreementAndSwitchesAcrossTheSavedHaplotypesResult:
    handles: object
    switch_rows: object


def plot_coverage_agreement_and_switches_across_the_saved_haplotypes(
    *,
    datasets,
    experiment_config,
    figsize=[16, 8],
    xlim=(0, 1),
    xlabel="Fraction of inference sites",
    fontsize="small",
    height=0.35,
    xlabel_2="Parent switches (upper: full HMM; lower: stitched)",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Coverage, agreement, and switches across the saved haplotypes. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        2, len(datasets), figsize=figsize, layout="constrained", squeeze=False, dpi=dpi
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    switch_rows = []
    for column, dataset in enumerate(datasets):
        y = np.arange(len(dataset.cases))
        accumulated = np.zeros(len(y))
        for category, (label, color) in enumerate(
            zip(
                [
                    "Focal / agree",
                    "Focal / differ",
                    "Gap fill / agree",
                    "Gap fill / differ",
                ],
                ["#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
                strict=True,
            )
        ):
            fractions = np.array(
                [np.mean(case.categories == category) for case in dataset.cases]
            )
            axes[0, column].barh(
                y, fractions, left=accumulated, color=color, label=label
            )
            accumulated += fractions
        axes[0, column].set(
            yticks=y,
            yticklabels=[case.label for case in dataset.cases],
            xlim=xlim,
            xlabel=xlabel,
            title=dataset.label,
        )
        axes[0, column].invert_yaxis()
        axes[0, column].legend(
            fontsize=fontsize, loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2
        )
        for offset, method in zip((-0.2, 0.2), ("full", "stitched"), strict=True):
            parents = [getattr(case, f"{method}_parents") for case in dataset.cases]
            counts = np.array(
                [
                    nb_utils.switch_breakdown(case, path)
                    for case, path in zip(dataset.cases, parents, strict=True)
                ]
            )
            accumulated = np.zeros(len(y))
            for category, color in enumerate(["#009E73", "#E69F00", "#7B61A8"]):
                axes[1, column].barh(
                    y + offset,
                    counts[:, category],
                    left=accumulated,
                    height=height,
                    color=color,
                    alpha=1 if method == "stitched" else 0.5,
                )
                accumulated += counts[:, category]
            for case, values in zip(dataset.cases, counts, strict=True):
                switch_rows.append(
                    {
                        "dataset": dataset.label,
                        "haplotype": case.label,
                        "method": method,
                        **dict(
                            zip(
                                [
                                    "Within focal coverage",
                                    "Within gaps",
                                    "At focal/gap seams",
                                ],
                                values,
                                strict=True,
                            )
                        ),
                    }
                )
        axes[1, column].set(
            yticks=y,
            yticklabels=[case.label for case in dataset.cases],
            xlabel=xlabel_2,
        )
        axes[1, column].invert_yaxis()
        handles = [
            mpatches.Patch(color=color, label=label)
            for color, label in zip(
                ["#009E73", "#E69F00", "#7B61A8"],
                ["Within focal coverage", "Within gaps", "At focal/gap seams"],
                strict=True,
            )
        ]
        axes[1, column].legend(handles=handles, fontsize=fontsize)
    fig.suptitle(f"Saved haplotypes at focal AC ≤ {experiment_config['ac_cutoff']}")
    plt.show()
    return CoverageAgreementAndSwitchesAcrossTheSavedHaplotypesResult(
        handles, switch_rows
    )


def plot_extra_switch_fraction(
    *,
    datasets,
    figsize=[16, 4],
    color="#6A3D9A",
    ylim=(0, 1),
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Extra switch fraction. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharey=True,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        fractions = [case.summary["extra_switch_fraction"] for case in dataset.cases]
        bars = ax.bar(np.arange(len(fractions)), fractions, color=color)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in fractions], padding=3)
        ax.set(
            xticks=np.arange(len(fractions)),
            xticklabels=[case.label for case in dataset.cases],
            ylim=ylim,
            title=dataset.label,
        )
    axes[0, 0].set_ylabel("Net extra switches / stitched switches")
    plt.show()


def plot_where_retained_focal_pieces_and_gap_fills_agree_with_the_full(
    *,
    category_cmap,
    category_norm,
    datasets,
    source_cmap,
    source_norm,
    figsize=[16, 7],
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Where retained focal pieces and gap fills agree with the full path. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1, len(datasets), figsize=figsize, layout="constrained", squeeze=False, dpi=dpi
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        edges = dataset.site_edges / 1000000
        for index, case in enumerate(dataset.cases):
            y = index * 2
            ax.pcolormesh(
                edges,
                [y, y + 0.7],
                (~case.accepted)[None, :],
                cmap=source_cmap,
                norm=source_norm,
                shading="flat",
                rasterized=True,
            )
            ax.pcolormesh(
                edges,
                [y + 0.8, y + 1.5],
                case.categories[None, :],
                cmap=category_cmap,
                norm=category_norm,
                shading="flat",
                rasterized=True,
            )
        ax.set(
            yticks=np.arange(len(dataset.cases)) * 2 + 0.75,
            yticklabels=[case.label for case in dataset.cases],
            xlabel=f"Genomic position ({'Mb'})",
            title=dataset.label,
        )
        ax.set_xlim(edges[0], edges[-1])
        ax.invert_yaxis()
    coverage_handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(
            ["#B8DED4", "#FFE3A3"] + ["#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
            ["Focal coverage", "Full-reference gap fill"]
            + [
                "Focal / agree",
                "Focal / differ",
                "Gap fill / agree",
                "Gap fill / differ",
            ],
            strict=True,
        )
    ]
    fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
    fig.suptitle("Each haplotype: provenance above, parent agreement below")
    plt.show()
    return coverage_handles


def plot_does_disagreement_concentrate_near_focal_gap_seams(
    *,
    bins,
    datasets,
    distance_rows,
    figsize=[16, 4],
    marker="o",
    ylim=(0, 1),
    xlabel="Distance to nearest seam (inference-site steps)",
    alpha=0.2,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Does disagreement concentrate near focal/gap seams?. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharey=True,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        for focal, label, color in zip(
            (True, False),
            ["Focal coverage", "Full-reference gap fill"],
            ("#CC79A7", "#D55E00"),
            strict=True,
        ):
            totals = np.zeros(len(bins) - 1, dtype=int)
            differences = np.zeros(len(bins) - 1, dtype=int)
            for case in dataset.cases:
                distance = nb_utils.seam_distances(case.accepted)
                selected = case.accepted == focal
                totals += np.histogram(distance[selected], bins=bins)[0]
                differences += np.histogram(
                    distance[selected & case.disagreement], bins=bins
                )[0]
            rate = np.divide(
                differences, totals, out=np.full(len(totals), np.nan), where=totals > 0
            )
            labels = [
                f"{lo}–{hi - 1}" for lo, hi in zip(bins[:-1], bins[1:], strict=True)
            ]
            ax.plot(np.arange(len(rate)), rate, marker=marker, label=label, color=color)
            for lo, hi, count, difference, fraction in zip(
                bins[:-1], bins[1:], totals, differences, rate, strict=True
            ):
                distance_rows.append(
                    {
                        "dataset": dataset.label,
                        "source": label,
                        "distance_start": lo,
                        "distance_end": hi,
                        "num_sites": count,
                        "disagree_sites": difference,
                        "disagreement_fraction": fraction,
                    }
                )
        ax.set(
            xticks=np.arange(len(labels)),
            xticklabels=labels,
            ylim=ylim,
            xlabel=xlabel,
            title=dataset.label,
        )
        ax.tick_params(axis="x", labelrotation=60)
        ax.grid(alpha=alpha)
        ax.legend()
    axes[0, 0].set_ylabel("Fraction of sites with different parents")
    plt.show()


def plot_gap_determination(
    *,
    datasets,
    experiment_config,
    gap_comparisons,
    figsize=(12, 4),
    width=0.35,
    ylabel="Number of maximal gap intervals",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Gap determination. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1, len(datasets), figsize=figsize, layout="constrained", squeeze=False, dpi=dpi
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        results = gap_comparisons[dataset.name]
        x = np.arange(len(results))
        for offset, field, label, color in zip(
            (-0.2, 0.2),
            ("hmm_num_gaps", "zero_coverage_num_gaps"),
            ["HMM gaps", "Zero-coverage gaps"],
            ["#0072B2", "#E69F00"],
            strict=True,
        ):
            counts = [result.summary[field] for result in results]
            bars = ax.bar(x + offset, counts, width=width, color=color, label=label)
            ax.bar_label(bars, padding=3)
        ax.set(
            xticks=x,
            xticklabels=[result.case.label for result in results],
            title=dataset.label,
            ylabel=ylabel,
        )
        ax.margins(y=0.15)
        ax.legend()
    fig.suptitle(
        f"Gap counts for the same haplotypes, AC ≤ {experiment_config['ac_cutoff']}, zero mismatches"
    )
    plt.show()


def plot_gap_determination_2(
    *,
    datasets,
    gap_comparisons,
    figsize=(12, 4),
    width=0.35,
    ylabel="Total number of gap intervals across saved haplotypes",
    title="Total gaps: HMM extraction versus zero-mismatch coverage",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Gap determination. Presentation choices are supplied by the notebook."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=dpi)
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    x = np.arange(len(datasets))
    for offset, field, label, color in zip(
        (-0.2, 0.2),
        ("hmm_num_gaps", "zero_coverage_num_gaps"),
        ["HMM gaps", "Zero-coverage gaps"],
        ["#0072B2", "#E69F00"],
        strict=True,
    ):
        counts = [
            sum((result.summary[field] for result in gap_comparisons[dataset.name]))
            for dataset in datasets
        ]
        bars = ax.bar(x + offset, counts, width=width, label=label, color=color)
        ax.bar_label(bars, padding=3)
    ax.set(
        xticks=x,
        xticklabels=[
            f"{dataset.label}\n{len(dataset.cases)} saved haplotypes"
            for dataset in datasets
        ],
        ylabel=ylabel,
        title=title,
    )
    ax.margins(y=0.15)
    ax.legend()
    plt.show()


def plot_gap_determination_3(
    *,
    datasets,
    gap_comparisons,
    figsize=(12, 5),
    xlim=(0, 1),
    xlabel="Fraction of inference sites",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Gap determination. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharex=True,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        results = gap_comparisons[dataset.name]
        y = np.arange(len(results))
        left = np.zeros(len(y))
        for field, label, color in zip(
            (
                "overlap_sites",
                "hmm_only_gap_sites",
                "zero_only_gap_sites",
                "neither_gap_sites",
            ),
            ["Both gaps", "HMM only", "Zero-coverage only", "Neither gap"],
            ["#7B61A8", "#0072B2", "#E69F00", "#DDDDDD"],
            strict=True,
        ):
            fractions = [
                result.summary[field] / result.summary["num_sites"]
                for result in results
            ]
            ax.barh(y, fractions, left=left, label=label, color=color)
            left += fractions
        ax.set(
            yticks=y,
            yticklabels=[result.case.label for result in results],
            xlim=xlim,
            xlabel=xlabel,
            title=dataset.label,
        )
        ax.invert_yaxis()
    handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(
            ["#7B61A8", "#0072B2", "#E69F00", "#DDDDDD"],
            ["Both gaps", "HMM only", "Zero-coverage only", "Neither gap"],
            strict=True,
        )
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    plt.show()
    return handles


def plot_gap_determination_4(
    *,
    datasets,
    gap_comparisons,
    handles,
    figsize=(12, 6),
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Gap determination. Presentation choices are supplied by the notebook."""
    overlap_cmap = mcolors.ListedColormap(["#7B61A8", "#0072B2", "#E69F00", "#DDDDDD"])
    overlap_norm = mcolors.BoundaryNorm(np.arange(5) - 0.5, overlap_cmap.N)
    for dataset in datasets:
        result = next(
            (
                item
                for item in gap_comparisons[dataset.name]
                if item.case is dataset.primary
            )
        )
        edges = dataset.site_edges / 1000000
        fig, axes = plt.subplots(
            3, 1, sharex=True, figsize=figsize, layout="constrained", dpi=dpi
        )
        for axis in fig.axes:
            axis.spines["top"].set_visible(show_top_spine)
            axis.spines["right"].set_visible(show_right_spine)
        for ax, mask, label, color in zip(
            axes[:2],
            (~result.case.accepted, result.coverage == 0),
            ["HMM gaps", "Zero-coverage gaps"],
            ["#0072B2", "#E69F00"],
            strict=True,
        ):
            cmap = mcolors.ListedColormap(
                [["#7B61A8", "#0072B2", "#E69F00", "#DDDDDD"][-1], color]
            )
            ax.pcolormesh(
                edges, [0, 1], mask[None, :], cmap=cmap, vmin=0, vmax=1, rasterized=True
            )
            ax.set_yticks([])
            ax.set_ylabel(label, rotation=0, ha="right", va="center")
        axes[2].pcolormesh(
            edges,
            [0, 1],
            result.categories[None, :],
            cmap=overlap_cmap,
            norm=overlap_norm,
            rasterized=True,
        )
        axes[2].set(
            yticks=[], xlim=(edges[0], edges[-1]), xlabel=f"Genomic position ({'Mb'})"
        )
        axes[2].set_ylabel("Overlap", rotation=0, ha="right", va="center")
        fig.suptitle(
            f"{dataset.label}: {result.case.label} | gap-site IoU {result.summary['gap_site_iou']:.1%} | {result.summary['exact_gap_bound_matches']} exactly matching gap bounds"
        )
        fig.legend(handles=handles, loc="outside lower center", ncol=4)
        fig.canvas.draw()
        plt.show()


def plot_performance_against_sample_size(
    *, figure, workers, color="gray", linestyle=":", linewidth=1
):
    """Performance against sample size. Presentation choices are supplied by the notebook."""
    figure.axes[0].axhline(1, color=color, linestyle=linestyle, linewidth=linewidth)
    figure.axes[1].axhline(
        workers, color=color, linestyle=linestyle, linewidth=linewidth
    )
    figure.axes[1].set_ylim(top=workers * (1 + 0.08))
    figure.axes[1].set_title(f"{workers} matching workers (dotted line)")


def plot_performance_against_sample_size_2(
    *, figure, performance, workers, color="gray", linestyle=":", linewidth=1
):
    """Performance against sample size. Presentation choices are supplied by the notebook."""
    figure.axes[0].axhline(1, color=color, linestyle=linestyle, linewidth=linewidth)
    figure.axes[1].axhline(
        workers, color=color, linestyle=linestyle, linewidth=linewidth
    )
    maximum_cores = max(workers, performance.effective_cores.max())
    figure.axes[1].set_ylim(top=maximum_cores * (1 + 0.08))
    figure.axes[1].set_title(f"{workers} matching workers (dotted line)")


def plot_report_age_correlations(
    *,
    axes,
    figure,
    method_colors,
    report_ages,
    report_data,
    markersize=5,
    linewidth=1.7,
    age_metrics={
        "pearson_r": "Pearson correlation",
        "spearman_r": "Spearman rank correlation",
    },
    age_titles={
        "pearson_r": "Pearson correlation",
        "spearman_r": "Spearman rank correlation",
    },
    marker="o",
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
):
    """Figures for the supervisor update (11 November 2026). Presentation choices are supplied by the notebook."""
    for ax, (metric, ylabel) in zip(axes.flat, age_metrics.items(), strict=True):
        for method, label in method_labels.items():
            curve = report_ages.loc[report_ages.method.eq(method)].sort_values(
                "num_samples"
            )
            ax.plot(
                curve.num_samples,
                curve[metric],
                color=method_colors[method],
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                label=label,
            )
        _cohort_sample_axis(
            report_data,
            ax,
            datasets=report_ages.dataset.unique(),
            cohort_labels={
                "out_of_africa_n50_28_1mbp_error": "50\nOOA",
                "out_of_africa_n500_28_1mbp_error": "500\nOOA",
                "out_of_africa_n5000_28_1mbp_error": "5,000\nOOA",
                "out_of_africa_n50000_28_1mbp_error": "50,000\nOOA",
                "tgp_chr20p_n3202": "3,202\n1KGP",
            },
            sample_tick_alignments={
                "tgp_chr20p_n3202": "right",
                "out_of_africa_n5000_28_1mbp_error": "left",
            },
            sample_tick_rotation=0,
        )
        ax.set(ylabel=ylabel, title=age_titles[metric])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels))


def plot_report_speedup_parallelism(
    *,
    config,
    cores_ax,
    dataset_labels,
    method_colors,
    positions,
    report_performance,
    speedup_ax,
    stitched,
    width=0.65,
    ylabel="Wall-time speedup over Full HMM",
    ylim=(0, None),
    markersize=5,
    linewidth=1.7,
    color="#555555",
    ylabel_2="Average active CPU cores",
    greedy_color="#0072B2",
    marker="o",
    method_labels={"full_hmm": "Full HMM", "greedy": "Greedy HMM"},
    parallelism_title="Parallelism",
    performance_xlabel="Dataset (diploid individuals)",
    speedup_title="Wall-time speed-up",
    worker_reference_label="{workers} threads",
    worker_reference_linestyle="--",
):
    """Figures for the supervisor update (11 November 2026). Presentation choices are supplied by the notebook."""
    bars = speedup_ax.bar(positions, stitched.speedup, color=greedy_color, width=width)
    speedup_ax.bar_label(
        bars, labels=[f"{value:.2f}×" for value in stitched.speedup], padding=3
    )
    speedup_ax.set(
        xticks=positions,
        xticklabels=dataset_labels,
        xlabel=performance_xlabel,
        ylabel=ylabel,
        title=speedup_title,
        ylim=ylim,
    )
    speedup_ax.margins(y=0.22)
    for method, label in method_labels.items():
        curve = report_performance.loc[
            report_performance.method.eq(method)
        ].sort_values("num_samples")
        cores_ax.plot(
            positions,
            curve.effective_cores,
            color=method_colors[method],
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            label=label,
        )
    workers = config["experiment"]["performance"]["worker_counts"][0]
    cores_ax.axhline(
        workers,
        color=color,
        linestyle=worker_reference_linestyle,
        linewidth=linewidth,
        label=worker_reference_label.format(workers=workers),
    )
    maximum_cores = max(workers, report_performance.effective_cores.max())
    cores_ax.set(
        xticks=positions,
        xticklabels=dataset_labels,
        xlabel=performance_xlabel,
        ylabel=ylabel_2,
        title=parallelism_title,
        ylim=(0, maximum_cores * (1 + 0.08)),
    )
    cores_ax.legend()


def plot_extra_switches_vs_hyperparameters(
    *,
    hyperparameter_metrics,
    requested_mismatch_settings,
    switch_counts,
    figsize=[12, 8],
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Extra switches vs hyperparameters. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True, dpi=dpi)
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    available_cutoffs = sorted(switch_counts.ac_cutoff.unique())
    for column, mismatch_count in enumerate(requested_mismatch_settings):
        for row, (metric, ylabel) in enumerate(hyperparameter_metrics):
            ax = axes[row, column]
            ax.text(
                0.5,
                0.5,
                "No results with a recorded\nmismatch allowance",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{mismatch_count} mismatches allowed per side")
            ax.set_ylabel(ylabel)
            ax.set_xticks(available_cutoffs)
    for ax in axes[-1]:
        ax.set_xlabel("AC cutoff")
    fig.suptitle("Extra switches by mismatch allowance")
    fig.tight_layout()
    plt.show()


def plot_extra_switches_vs_hyperparameters_2(
    *,
    hyperparameter_metrics,
    switch_counts,
    figsize=[12, 5],
    marker="o",
    fontsize="small",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Extra switches vs hyperparameters. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, (metric, ylabel) in zip(axes, hyperparameter_metrics, strict=True):
        summary = switch_counts.groupby("ac_cutoff")[metric].agg(["mean", "sem"])
        ax.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["sem"],
            marker=marker,
            capsize=3,
        )
        ax.set_xlabel("AC cutoff")
        ax.set_ylabel(ylabel)
        ax.set_xticks(summary.index)
        if metric == "switch_increase_percentage":
            undefined_count = switch_counts[metric].isna().sum()
            ax.text(
                0.98,
                0.96,
                f"{undefined_count} undefined increases excluded",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=fontsize,
            )
    fig.suptitle("Current saved configuration (mean ± SEM)")
    fig.tight_layout()
    plt.show()


def make_path_legend():
    """Extra switches vs hyperparameters. Presentation choices are supplied by the notebook."""
    coverage_handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(
            ["#B8DED4", "#FFE3A3"] + ["#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
            ["Focal coverage", "Full-reference gap fill"]
            + [
                "Focal / agree",
                "Focal / differ",
                "Gap fill / agree",
                "Gap fill / differ",
            ],
            strict=True,
        )
    ]
    return coverage_handles


def plot_extra_switches_by_population(*, fig, population_output_dir):
    """Extra switches by population. Presentation choices are supplied by the notebook."""
    fig.savefig(population_output_dir / "population_extra_switch_histograms.png")
    plt.show()


def plot_percentage_increase_in_switches_by_population(*, fig, population_output_dir):
    """Percentage increase in switches by population. Presentation choices are supplied by the notebook."""
    fig.savefig(population_output_dir / "population_switch_increase_histograms.png")
    plt.show()


def plot_gap_properties(
    *,
    config,
    population_comparisons,
    population_output_dir,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Gap properties. Presentation choices are supplied by the notebook."""
    gap_population_handles = [
        mpatches.Patch(
            color={
                "YRI": "tab:blue",
                "AFR": "tab:blue",
                "CEU": "tab:orange",
                "EUR": "tab:orange",
                "CHB": "tab:gray",
                "EAS": "tab:gray",
            }[population],
            label=label,
        )
        for population, label in (
            ("YRI", "YRI / AFR"),
            ("CEU", "CEU / EUR"),
            ("CHB", "CHB / EAS"),
        )
    ]
    fig = plot_gap_property_histograms(
        "num_gaps",
        "Number of gaps",
        "Number of gaps per haplotype",
        integer_bins=True,
        config=config,
        gap_population_handles=gap_population_handles,
        plot_full_color="#222222",
        plot_population_results_count_bin_width=1,
        plot_population_results_density_tick_count=5,
        plot_population_results_gap_histogram_color="tab:gray",
        plot_population_results_gap_histogram_figsize=[14, 5],
        plot_population_results_histogram_alpha=0.3,
        plot_population_results_histogram_bins=30,
        plot_population_results_kde_bandwidth="scott",
        plot_population_results_kde_grid_size=512,
        plot_population_results_line_width=2,
        plot_population_results_point_alpha=0.6,
        plot_population_results_point_size=12,
        plot_population_results_point_strip_depth=0.15,
        plot_population_results_population_colors={
            "YRI": "tab:blue",
            "AFR": "tab:blue",
            "CEU": "tab:orange",
            "EUR": "tab:orange",
            "CHB": "tab:gray",
            "EAS": "tab:gray",
        },
        plot_population_results_random_seed=42,
        plot_population_results_range_padding=0.05,
        population_comparisons=population_comparisons,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    fig.savefig(population_output_dir / "population_number_of_gaps_histograms.png")
    plt.show()
    return gap_population_handles


def plot_gap_properties_2(*, fig, population_output_dir):
    """Gap properties. Presentation choices are supplied by the notebook."""
    fig.savefig(population_output_dir / "population_mean_sites_per_gap_histograms.png")
    plt.show()


def plot_gap_properties_3(
    *,
    config,
    gap_population_handles,
    population_comparisons,
    population_output_dir,
    figsize=[14, 9],
    s=12,
    alpha=0.6,
    linewidths=0,
    ylabel="Mean sites per gap",
    alpha_2=0.2,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Gap properties. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        2,
        len(config["datasets"]),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharey="col",
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for column, settings in enumerate(config["datasets"]):
        dataset_table = population_comparisons.loc[
            population_comparisons.dataset.eq(settings["name"])
        ]
        defined_table = dataset_table.dropna(subset=["mean_sites_per_gap"])
        excluded = len(dataset_table) - len(defined_table)
        for row, (metric, xlabel) in enumerate(
            (
                ("num_focal_ancestors", "Number of focal ancestors"),
                ("num_gaps", "Number of gaps"),
            )
        ):
            ax = axes[row, column]
            for population in settings["populations"]:
                observations = defined_table.loc[
                    defined_table.population.eq(population)
                ]
                ax.scatter(
                    observations[metric],
                    observations.mean_sites_per_gap,
                    color={
                        "YRI": "tab:blue",
                        "AFR": "tab:blue",
                        "CEU": "tab:orange",
                        "EUR": "tab:orange",
                        "CHB": "tab:gray",
                        "EAS": "tab:gray",
                    }[population],
                    s=s,
                    alpha=alpha,
                    linewidths=linewidths,
                )
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.text(
                0.98,
                0.95,
                f"n = {len(defined_table)}; excluded = {excluded}",
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
            if defined_table[metric].nunique() == 1:
                ax.set_xticks(defined_table[metric].unique())
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
            ax.grid(alpha=alpha_2)
            if row == 0:
                ax.set_title(settings["label"])
    fig.legend(handles=gap_population_handles, loc="outside lower center", ncol=3)
    fig.suptitle("Gap length and focal ancestor availability")
    fig.savefig(population_output_dir / "population_gap_properties_scatter.png")
    plt.show()


def plot_gap_properties_4(
    *,
    config,
    gap_population_handles,
    gap_predictors,
    population_comparisons,
    population_output_dir,
    figsize=[16, 8],
    s=12,
    alpha=0.6,
    linewidths=0,
    alpha_2=0.2,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Gap properties. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        len(config["datasets"]),
        len(gap_predictors),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharey="row",
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for row, settings in enumerate(config["datasets"]):
        row_label = ["Simulation", "1KGP chr20"][row]
        dataset_table = population_comparisons.loc[
            population_comparisons.dataset.eq(settings["name"])
        ]
        defined_table = dataset_table.dropna(subset=["mean_sites_per_gap"])
        excluded = len(dataset_table) - len(defined_table)
        for column, (metric, xlabel) in enumerate(gap_predictors):
            ax = axes[row, column]
            for population in settings["populations"]:
                observations = defined_table.loc[
                    defined_table.population.eq(population)
                ]
                ax.scatter(
                    observations[metric],
                    observations.num_extra_switches,
                    color={
                        "YRI": "tab:blue",
                        "AFR": "tab:blue",
                        "CEU": "tab:orange",
                        "EUR": "tab:orange",
                        "CHB": "tab:gray",
                        "EAS": "tab:gray",
                    }[population],
                    s=s,
                    alpha=alpha,
                    linewidths=linewidths,
                )
            ax.set_xlabel(xlabel)
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
            if column == 0:
                ax.set_ylabel("Extra switches")
            if column == 0:
                ax.set_title(f"{row_label} · {xlabel}")
            elif row == 0:
                ax.set_title(xlabel)
            ax.text(
                0.98,
                0.95,
                f"n = {len(defined_table)}; excluded = {excluded}",
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
            ax.grid(alpha=alpha_2)
    fig.legend(handles=gap_population_handles, loc="outside lower center", ncol=3)
    fig.suptitle("Extra switches and gap properties by population")
    fig.savefig(population_output_dir / "population_extra_switch_gap_scatter.png")
    plt.show()


def plot_decomposing_switches_by_type(
    *,
    config,
    population_output_dir,
    population_switch_categories,
    population_switch_means,
    figsize=[14, 6],
    height=0.32,
    xlim=(0, 100),
    xlabel="Mean proportion of switches (%)",
    alpha=0.2,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Decomposing switches by type. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1,
        len(config["datasets"]),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharex=True,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, settings in zip(axes[0], config["datasets"], strict=True):
        populations = settings["populations"]
        y = np.arange(len(populations))
        for method_index, method in enumerate(("full", "stitched")):
            offset = (2 * method_index - 1) * 0.2
            centers = y + offset
            keys = [
                (settings["name"], population, method) for population in populations
            ]
            means = population_switch_means.loc[keys]
            left = np.zeros(len(populations))
            for category, color in zip(
                population_switch_categories,
                ["#009E73", "#E69F00", "#7B61A8"],
                strict=True,
            ):
                widths = means[category].to_numpy() * 100
                ax.barh(centers, widths, left=left, height=height, color=color)
                left += widths
        method_ticks = np.column_stack((y - 0.2, y + 0.2)).ravel()
        method_labels = [
            f"{population} · {method}"
            for population in populations
            for method in ("Full", "Stitched")
        ]
        ax.set(
            yticks=method_ticks,
            yticklabels=method_labels,
            xlim=xlim,
            xlabel=xlabel,
            title=settings["label"],
            xticks=[0, 20, 40, 60, 80, 100],
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=alpha)
    population_switch_handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(
            ["#009E73", "#E69F00", "#7B61A8"],
            ["Within focal coverage", "Within gaps", "At focal/gap seams"],
            strict=True,
        )
    ]
    fig.legend(handles=population_switch_handles, loc="outside lower center", ncol=3)
    fig.suptitle("Switch composition by population: mean of haplotype proportions")
    fig.savefig(population_output_dir / "population_switch_compositions.png")
    plt.show()


def plot_decomposing_switches_by_type_2(
    *,
    config,
    population_output_dir,
    population_switch_categories,
    population_switch_means,
    population_switch_proportions,
    figsize=[14, 7],
    color="#222222",
    color_2="lightgray",
    linewidth=0.7,
    fontsize="small",
    s=14,
    alpha=0.2,
    linewidths=0,
    s_2=90,
    xlim=(-0.15, 1.15),
    linestyle="none",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Decomposing switches by type. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1,
        len(config["datasets"]),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    triangle_height = np.sqrt(3) / 2
    for ax, settings in zip(axes[0], config["datasets"], strict=True):
        ax.plot([0, 1, 0.5, 0], [0, 0, triangle_height, 0], color=color)
        for fraction in [0.2, 0.4, 0.6, 0.8]:
            for category in range(len(population_switch_categories)):
                endpoints = np.zeros((2, len(population_switch_categories)))
                endpoints[:, category] = fraction
                other_categories = [
                    index
                    for index in range(len(population_switch_categories))
                    if index != category
                ]
                endpoints[0, other_categories[0]] = 1 - fraction
                endpoints[1, other_categories[1]] = 1 - fraction
                coordinates = nb_utils.ternary_coordinates(endpoints)
                ax.plot(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    color=color_2,
                    linewidth=linewidth,
                    zorder=0,
                )
            ax.text(
                fraction,
                -0.025,
                f"{1 - fraction:.0%}",
                ha="center",
                va="top",
                fontsize=fontsize,
            )
            ax.text(
                0.5 * fraction - 0.02,
                triangle_height * fraction,
                f"{fraction:.0%}",
                ha="right",
                fontsize=fontsize,
            )
            ax.text(
                1 - 0.5 * fraction + 0.02,
                triangle_height * fraction,
                f"{1 - fraction:.0%}",
                ha="left",
                fontsize=fontsize,
            )
        for method in ("full", "stitched"):
            for population, label_offset in zip(
                settings["populations"], [[-24, -18], [0, 20], [26, -18]], strict=True
            ):
                color = {
                    "YRI": "tab:blue",
                    "AFR": "tab:blue",
                    "CEU": "tab:orange",
                    "EUR": "tab:orange",
                    "CHB": "tab:gray",
                    "EAS": "tab:gray",
                }[population]
                selected = population_switch_proportions.dataset.eq(settings["name"])
                selected &= population_switch_proportions.method.eq(method)
                selected &= population_switch_proportions.population.eq(population)
                proportions = population_switch_proportions.loc[
                    selected, population_switch_categories
                ]
                proportions = proportions.dropna().to_numpy()
                coordinates = nb_utils.ternary_coordinates(proportions)
                ax.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    color=color,
                    marker={"full": "o", "stitched": "^"}[method],
                    s=s,
                    alpha=alpha,
                    linewidths=linewidths,
                )
                mean = population_switch_means.loc[
                    settings["name"], population, method
                ].to_numpy()
                coordinate = nb_utils.ternary_coordinates(mean[np.newaxis, :])[0]
                ax.scatter(
                    coordinate[0],
                    coordinate[1],
                    color=color,
                    marker={"full": "o", "stitched": "^"}[method],
                    s=s_2,
                    edgecolors="white",
                    zorder=3,
                )
                ax.annotate(
                    population,
                    coordinate,
                    xytext=label_offset,
                    textcoords="offset points",
                    color=color,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7},
                )
        ax.text(0, -0.07, "Within focal\ncoverage", ha="center", va="top")
        ax.text(1, -0.07, "Within gaps", ha="center", va="top")
        ax.text(
            0.5, triangle_height + 0.03, "At focal/gap seams", ha="center", va="bottom"
        )
        ax.set(
            title=settings["label"],
            aspect="equal",
            xlim=xlim,
            ylim=(-0.16, triangle_height + 0.1),
        )
        ax.set_axis_off()
    population_method_handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            linestyle=linestyle,
            marker={"full": "o", "stitched": "^"}[method],
            label=label,
        )
        for method, label in (("full", "Full HMM"), ("stitched", "Stitched HMM"))
    ]
    population_color_handles = [
        mpatches.Patch(
            color={
                "YRI": "tab:blue",
                "AFR": "tab:blue",
                "CEU": "tab:orange",
                "EUR": "tab:orange",
                "CHB": "tab:gray",
                "EAS": "tab:gray",
            }[population],
            label=label,
        )
        for population, label in (
            ("YRI", "YRI / AFR"),
            ("CEU", "CEU / EUR"),
            ("CHB", "CHB / EAS"),
        )
    ]
    fig.legend(
        handles=population_color_handles + population_method_handles,
        loc="outside lower center",
        ncol=5,
    )
    fig.suptitle("Switch compositions: haplotypes and population means (large markers)")
    fig.savefig(population_output_dir / "population_switch_ternary.png")
    plt.show()


def plot_parent_agreement_proportions(
    *,
    config,
    population_agreement_columns,
    population_agreement_means,
    population_output_dir,
    figsize=[14, 6],
    xlim=(0, 100),
    xlabel="Mean proportion of all inference sites (%)",
    alpha=0.2,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Parent agreement proportions. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1,
        len(config["datasets"]),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharex=True,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, settings in zip(axes[0], config["datasets"], strict=True):
        populations = settings["populations"]
        y = np.arange(len(populations))
        keys = [(settings["name"], population) for population in populations]
        means = population_agreement_means.loc[keys]
        left = np.zeros(len(populations))
        for category, color in zip(
            population_agreement_columns,
            ["#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
            strict=True,
        ):
            widths = means[category].to_numpy() * 100
            ax.barh(y, widths, left=left, height=0.32 * 2, color=color)
            left += widths
        ax.set(
            yticks=y,
            yticklabels=populations,
            xlim=xlim,
            xlabel=xlabel,
            title=settings["label"],
            xticks=[0, 20, 40, 60, 80, 100],
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=alpha)
    population_agreement_handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(
            ["#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
            ["Focal / agree", "Focal / disagree", "Gaps / agree", "Gaps / disagree"],
            strict=True,
        )
    ]
    fig.legend(handles=population_agreement_handles, loc="outside lower center", ncol=4)
    fig.suptitle("Parent agreement proportions by population")
    fig.savefig(population_output_dir / "population_parent_agreement_strips.png")
    plt.show()


def plot_parent_agreement_in_focal_regions(*, fig, population_output_dir):
    """Parent agreement in focal regions. Presentation choices are supplied by the notebook."""
    fig.savefig(population_output_dir / "population_focal_agreement_histograms.png")
    plt.show()


def plot_parent_agreement_in_gaps(*, fig, population_output_dir):
    """Parent agreement in gaps. Presentation choices are supplied by the notebook."""
    fig.savefig(population_output_dir / "population_gap_agreement_histograms.png")
    plt.show()


def plot_summary_plots(
    *,
    category_cmap,
    category_norm,
    coverage_handles,
    outlier_datasets,
    outlier_output_dir,
    outlier_summary,
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Summary plots. Presentation choices are supplied by the notebook."""
    plt.show()
    ipython_display.display(outlier_summary.switches)
    outlier_summary.switches.to_csv(
        outlier_output_dir / "outlier_switch_counts.csv", index=False
    )
    fig = plot_case_tracks(
        outlier_datasets,
        "Outliers: provenance above, parent agreement below",
        category_cmap=category_cmap,
        category_norm=category_norm,
        coverage_handles=coverage_handles,
        plot_genome_unit_bp=1000000,
        plot_genome_unit_label="Mb",
        plot_tracks_figsize=[16, 7],
        source_cmap=source_cmap,
        source_norm=source_norm,
        dpi=dpi,
        show_top_spine=show_top_spine,
        show_right_spine=show_right_spine,
    )
    fig.savefig(outlier_output_dir / "outlier_provenance_and_agreement.png")
    plt.show()


def plot_summary_plots_2(
    *,
    category_cmap,
    category_norm,
    coverage_handles,
    outlier_case_datasets,
    outlier_output_dir,
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Summary plots. Presentation choices are supplied by the notebook."""
    for dataset in outlier_case_datasets:
        case = dataset.primary
        prefix = f"{dataset.name}_{case.summary['sample_id'].lower()}_{case.summary['ploidy_index']}"
        fig = path_diagnostics_plot_overview(
            dataset,
            category_cmap=category_cmap,
            category_norm=category_norm,
            coverage_handles=coverage_handles,
            plot_full_color="#222222",
            plot_genome_unit_bp=1000000,
            plot_genome_unit_label="Mb",
            plot_overview_figsize=[16, 7],
            plot_stitched_color="#6A3D9A",
            source_cmap=source_cmap,
            source_norm=source_norm,
            dpi=dpi,
            show_top_spine=show_top_spine,
            show_right_spine=show_right_spine,
        )
        fig.savefig(outlier_output_dir / f"{prefix}_whole_span_paths.png")
        plt.show()


def plot_deep_dive(
    *,
    outlier_analyses,
    outlier_datasets,
    outlier_output_dir,
    figsize=[16, 8],
    height=0.32,
    xlabel="Parent switches",
    alpha=0.2,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Deep dive. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1,
        len(outlier_datasets),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, dataset in zip(axes[0], outlier_datasets, strict=True):
        y = np.arange(len(dataset.cases))
        for method_index, method in enumerate(("full", "stitched")):
            centers = y + (2 * method_index - 1) * 0.2
            left = np.zeros(len(dataset.cases))
            for kind, color in zip(
                [
                    "Identical switch",
                    "Shifted parent sequence",
                    "Changed parent sequence",
                    "Unanchored difference",
                ],
                ["tab:gray", "tab:blue", "tab:orange", "tab:red"],
                strict=True,
            ):
                widths = []
                for case in dataset.cases:
                    table = outlier_analyses[dataset.name, case.label].switch_counts
                    widths.append(int(table.loc[table.kind.eq(kind), method].iloc[0]))
                widths = np.array(widths)
                ax.barh(centers, widths, left=left, color=color, height=height)
                left += widths
        method_ticks = np.column_stack((y - 0.2, y + 0.2)).ravel()
        method_labels = [
            f"{case.label} · {method}"
            for case in dataset.cases
            for method in ("Full", "Stitched")
        ]
        ax.set(
            yticks=method_ticks,
            yticklabels=method_labels,
            xlabel=xlabel,
            title=dataset.label,
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=alpha)
    transition_handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(
            ["tab:gray", "tab:blue", "tab:orange", "tab:red"],
            [
                "Identical switch",
                "Shifted parent sequence",
                "Changed parent sequence",
                "Unanchored difference",
            ],
            strict=True,
        )
    ]
    fig.legend(handles=transition_handles, loc="outside lower center", ncol=2)
    fig.suptitle("Outliers: switch position changes and parent sequence changes")
    fig.savefig(outlier_output_dir / "outlier_parent_transition_partition.png")
    plt.show()


def plot_deep_dive_2(
    *,
    category_cmap,
    category_norm,
    config,
    coverage_handles,
    outlier_analyses,
    outlier_case_datasets,
    outlier_output_dir,
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Deep dive. Presentation choices are supplied by the notebook."""
    for dataset in outlier_case_datasets:
        case = dataset.primary
        key = (dataset.name, case.label)
        analysis = outlier_analyses[key]
        prefix = f"{dataset.name}_{case.summary['sample_id'].lower()}_{case.summary['ploidy_index']}"
        print(f"{dataset.label}: {case.label} ({case.summary['population']})")
        gap_columns = [
            "gap",
            "start_site",
            "end_site",
            "num_sites",
            "agreement_fraction",
            "full_gap_and_seam_switches",
            "stitched_gap_and_seam_switches",
            "extra_gap_and_seam_switches",
            "extra_interior_switches",
            "left_flank_parent_agrees",
            "right_flank_parent_agrees",
        ]
        ranked_gaps = analysis.gaps.sort_values(
            ["extra_switches_per_site", "differ_sites"], ascending=[False, False]
        )
        print("Gaps ranked by net extra switches per site (including their seams):")
        displayed_gaps = ranked_gaps[gap_columns].head(
            config["outliers"]["gaps_to_display"]
        )
        ipython_display.display(displayed_gaps.round(3))
        print(
            "Disagreement regions ranked by net extra switches, then number of sites:"
        )
        ranked_regions = analysis.divergences.sort_values(
            ["extra_switches", "num_sites"], ascending=[False, False]
        )
        region_columns = [
            "difference",
            "start_site",
            "end_site",
            "num_sites",
            "gap_sites",
            "kind",
            "full_switches",
            "stitched_switches",
            "extra_switches",
            "full_parent_sequence",
            "stitched_parent_sequence",
        ]
        displayed_regions = ranked_regions[region_columns].head(
            config["outliers"]["divergences_to_display"]
        )
        ipython_display.display(displayed_regions)
        fig = plot_outlier_switch_genome(
            dataset,
            analysis,
            category_cmap=category_cmap,
            category_norm=category_norm,
            coverage_handles=coverage_handles,
            plot_full_color="#222222",
            plot_genome_unit_bp=1000000,
            plot_genome_unit_label="Mb",
            plot_outliers_genome_figsize=[16, 8],
            plot_population_results_point_size=12,
            plot_population_results_population_colors={
                "YRI": "tab:blue",
                "AFR": "tab:blue",
                "CEU": "tab:orange",
                "EUR": "tab:orange",
                "CHB": "tab:gray",
                "EAS": "tab:gray",
            },
            plot_stitched_color="#6A3D9A",
            source_cmap=source_cmap,
            source_norm=source_norm,
            dpi=dpi,
            show_top_spine=show_top_spine,
            show_right_spine=show_right_spine,
        )
        fig.savefig(outlier_output_dir / f"{prefix}_switch_distribution.png")
        plt.show()
        fig = plot_outlier_gap_properties(
            dataset,
            analysis,
            config=config,
            plot_outliers_gap_figsize=[16, 6],
            plot_population_results_point_alpha=0.6,
            plot_population_results_population_colors={
                "YRI": "tab:blue",
                "AFR": "tab:blue",
                "CEU": "tab:orange",
                "EUR": "tab:orange",
                "CHB": "tab:gray",
                "EAS": "tab:gray",
            },
            dpi=dpi,
            show_top_spine=show_top_spine,
            show_right_spine=show_right_spine,
        )
        fig.savefig(outlier_output_dir / f"{prefix}_gap_properties.png")
        plt.show()


def plot_deep_dive_3(
    *,
    category_cmap,
    category_norm,
    config,
    coverage_handles,
    outlier_case_datasets,
    outlier_output_dir,
    outlier_window_rows,
    outlier_windows,
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Deep dive. Presentation choices are supplied by the notebook."""
    for dataset in outlier_case_datasets:
        case = dataset.primary
        key = (dataset.name, case.label)
        prefix = f"{dataset.name}_{case.summary['sample_id'].lower()}_{case.summary['ploidy_index']}"
        rows = []
        for index, window in enumerate(outlier_windows[key]):
            target = case.disagreement[window.target_start : window.target_end]
            row = {
                "dataset": dataset.name,
                "haplotype": case.label,
                **dataclasses.asdict(window),
                "target_agreement": float(np.mean(~target)),
                "display_start_bp": dataset.positions[window.start],
                "display_end_bp": dataset.site_edges[window.end],
            }
            rows.append(row)
        outlier_window_rows.extend(rows)
        print(f"Detailed windows: {dataset.label}, {case.label}")
        ipython_display.display(pd.DataFrame(rows).round(3))
        for index, window in enumerate(outlier_windows[key]):
            site_axis = (
                config["outliers"]["short_gap_site_axis"]
                and window.kind == "Short gap with extra switches"
            )
            fig = path_diagnostics_plot_detail(
                dataset,
                window,
                site_axis=site_axis,
                category_cmap=category_cmap,
                category_norm=category_norm,
                coverage_handles=coverage_handles,
                plot_detail_figsize=[16, 7],
                plot_detail_unit_bp=1000,
                plot_detail_unit_label="kb",
                plot_full_color="#222222",
                plot_parent_row_height_in=0.24,
                plot_source_colors=["#B8DED4", "#FFE3A3"],
                plot_stitched_color="#6A3D9A",
                source_cmap=source_cmap,
                source_norm=source_norm,
                dpi=dpi,
                show_top_spine=show_top_spine,
                show_right_spine=show_right_spine,
            )
            fig.savefig(outlier_output_dir / f"{prefix}_detail_{index + 1}.png")
            plt.show()


def plot_deep_dive_4(
    *,
    config,
    outlier_datasets,
    outlier_output_dir,
    outlier_selection,
    figsize=[16, 9],
    marker="o",
    ylim=(0, 1),
    alpha=0.2,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Deep dive. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        len(config["outliers"]["criteria"]),
        len(outlier_datasets),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharey=True,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    outlier_distance_rows = []
    bins = np.array(config["distance_bin_edges"])
    distance_labels = [
        f"{lo}–{hi - 1}" for lo, hi in zip(bins[:-1], bins[1:], strict=True)
    ]
    for column, dataset in enumerate(outlier_datasets):
        for row, criterion in enumerate(config["outliers"]["criteria"]):
            selected = outlier_selection.loc[
                outlier_selection.dataset.eq(dataset.name)
                & outlier_selection.criterion.eq(criterion["label"])
            ].iloc[0]
            case = next(
                (
                    case
                    for case in dataset.cases
                    if case.summary["sample_id"] == selected.sample_id
                    and case.summary["ploidy_index"] == selected.ploidy_index
                )
            )
            distance = nb_utils.path_diagnostics_seam_distances(case.covered)
            ax = axes[row, column]
            for focal, label, color in (
                (True, "Focal regions", "#CC79A7"),
                (False, "Gaps", "#D55E00"),
            ):
                selected_sites = case.covered == focal
                totals = np.histogram(distance[selected_sites], bins=bins)[0]
                differences = np.histogram(
                    distance[selected_sites & case.disagreement], bins=bins
                )[0]
                rate = np.divide(
                    differences,
                    totals,
                    out=np.full(len(totals), np.nan),
                    where=totals > 0,
                )
                ax.plot(
                    np.arange(len(rate)), rate, marker=marker, color=color, label=label
                )
                for lo, hi, count, difference in zip(
                    bins[:-1], bins[1:], totals, differences, strict=True
                ):
                    outlier_distance_rows.append(
                        {
                            "dataset": dataset.name,
                            "haplotype": case.label,
                            "source": label,
                            "distance_start": lo,
                            "distance_end": hi,
                            "num_sites": int(count),
                            "disagree_sites": int(difference),
                        }
                    )
            ax.set(
                xticks=np.arange(len(distance_labels)),
                xticklabels=distance_labels,
                ylim=ylim,
                title=f"{case.label} ({case.summary['population']}) — {criterion['label']}",
            )
            ax.tick_params(axis="x", labelrotation=60)
            ax.grid(alpha=alpha)
            if column == 0:
                ax.set_ylabel("Parent disagreement fraction")
            if row == len(config["outliers"]["criteria"]) - 1:
                ax.set_xlabel("Distance from nearest seam (site steps)")
            if row == 0:
                ax.legend()
    fig.suptitle("Outliers: disagreement inside focal regions and gaps")
    fig.savefig(outlier_output_dir / "outlier_disagreement_by_seam_distance.png")
    plt.show()
    return outlier_distance_rows


def plot_extra_switch_fraction_2(
    *,
    datasets,
    figsize=[16, 4],
    color="#6A3D9A",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Extra switch fraction. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharey=True,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        fractions = [case.summary["extra_switch_fraction"] for case in dataset.cases]
        bars = ax.bar(np.arange(len(fractions)), fractions, color=color)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in fractions], padding=3)
        ax.set(
            xticks=np.arange(len(fractions)),
            xticklabels=[case.label for case in dataset.cases],
            title=dataset.label,
        )
    axes[0, 0].set_ylabel("Net extra switches / stitched switches")
    plt.show()


def plot_whole_span_paths_for_the_selected_simulated_and_real(
    *,
    category_cmap,
    category_norm,
    coverage_handles,
    datasets,
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Whole-span paths for the selected simulated and real haplotypes. Presentation choices are supplied by the notebook."""
    for dataset in datasets:
        path_diagnostics_plot_overview(
            dataset,
            category_cmap=category_cmap,
            category_norm=category_norm,
            coverage_handles=coverage_handles,
            plot_full_color="#222222",
            plot_genome_unit_bp=1000000,
            plot_genome_unit_label="Mb",
            plot_overview_figsize=[16, 7],
            plot_stitched_color="#6A3D9A",
            source_cmap=source_cmap,
            source_norm=source_norm,
            dpi=dpi,
            show_top_spine=show_top_spine,
            show_right_spine=show_right_spine,
        )
        plt.show()


def plot_detailed_parent_paths_covered_focal_pieces_gap_fills(
    *,
    category_cmap,
    category_norm,
    coverage_handles,
    datasets,
    detail_windows,
    source_cmap,
    source_norm,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Detailed parent paths: covered focal pieces, gap fills, agreement, and difference. Presentation choices are supplied by the notebook."""
    for dataset in datasets:
        for window in detail_windows[dataset.name]:
            path_diagnostics_plot_detail(
                dataset,
                window,
                category_cmap=category_cmap,
                category_norm=category_norm,
                coverage_handles=coverage_handles,
                plot_detail_figsize=[16, 7],
                plot_detail_unit_bp=1000,
                plot_detail_unit_label="kb",
                plot_full_color="#222222",
                plot_parent_row_height_in=0.24,
                plot_source_colors=["#B8DED4", "#FFE3A3"],
                plot_stitched_color="#6A3D9A",
                source_cmap=source_cmap,
                source_norm=source_norm,
                dpi=dpi,
                show_top_spine=show_top_spine,
                show_right_spine=show_right_spine,
            )
            plt.show()


def plot_does_disagreement_concentrate_near_focal_gap_seams_2(
    *,
    bins,
    datasets,
    distance_rows,
    figsize=[16, 4],
    marker="o",
    ylim=(0, 1),
    xlabel="Distance to nearest seam (inference-site steps)",
    alpha=0.2,
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Does disagreement concentrate near focal/gap seams?. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
        sharey=True,
        dpi=dpi,
    )
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        for focal, label, color in zip(
            (True, False),
            ["Focal coverage", "Full-reference gap fill"],
            ("#CC79A7", "#D55E00"),
            strict=True,
        ):
            totals = np.zeros(len(bins) - 1, dtype=int)
            differences = np.zeros(len(bins) - 1, dtype=int)
            for case in dataset.cases:
                distance = nb_utils.path_diagnostics_seam_distances(case.covered)
                selected = case.covered == focal
                totals += np.histogram(distance[selected], bins=bins)[0]
                differences += np.histogram(
                    distance[selected & case.disagreement], bins=bins
                )[0]
            rate = np.divide(
                differences, totals, out=np.full(len(totals), np.nan), where=totals > 0
            )
            labels = [
                f"{lo}–{hi - 1}" for lo, hi in zip(bins[:-1], bins[1:], strict=True)
            ]
            ax.plot(np.arange(len(rate)), rate, marker=marker, label=label, color=color)
            for lo, hi, count, difference, fraction in zip(
                bins[:-1], bins[1:], totals, differences, rate, strict=True
            ):
                distance_rows.append(
                    {
                        "dataset": dataset.label,
                        "source": label,
                        "distance_start": lo,
                        "distance_end": hi,
                        "num_sites": count,
                        "disagree_sites": difference,
                        "disagreement_fraction": fraction,
                    }
                )
        ax.set(
            xticks=np.arange(len(labels)),
            xticklabels=labels,
            ylim=ylim,
            xlabel=xlabel,
            title=dataset.label,
        )
        ax.tick_params(axis="x", labelrotation=60)
        ax.grid(alpha=alpha)
        ax.legend()
    axes[0, 0].set_ylabel("Fraction of sites with different parents")
    plt.show()


def plot_benchmark_correctness_panels(
    *,
    axes,
    colors,
    comparisons,
    labels,
    methods,
    stitched_methods,
    width=0.65,
    title="Switch composition",
    ylabel="Mean switches per haplotype",
    fontsize="small",
    title_2="Parent agreement with Full HMM",
    ylabel_2="Mean agreement (%)",
    ylim=(0, 110),
    baseline_partition_method,
):
    """Correctness. Presentation choices are supplied by the notebook."""
    axes[0, 0].set_title("Extra switches")
    distribution(
        axes[0, 1],
        "switch_increase_percentage",
        "Increase in switches (%)",
        colors=colors,
        comparisons=comparisons,
        labels=labels,
        plot_histogram_alpha=0.16,
        plot_histogram_bins=80,
        plot_kde_bandwidth="scott",
        plot_kde_grid_size=512,
        plot_line_width=1.7,
        stitched_methods=stitched_methods,
    )
    axes[0, 1].set_title("Percent increase in switches")
    ax = axes[1, 0]
    baseline = comparisons.loc[comparisons.method.eq(baseline_partition_method)]
    for index, method in enumerate(methods):
        table = (
            baseline
            if method == "full_hmm"
            else comparisons.loc[comparisons.method.eq(method)]
        )
        prefix = "full" if method == "full_hmm" else "stitched"
        bottom = 0
        for category, values in {
            "focal": {"label": "Within focal coverage", "alpha": 1, "hatch": ""},
            "gap": {"label": "Within gaps", "alpha": 0.65, "hatch": "//"},
            "seam": {"label": "At gap seams", "alpha": 0.35, "hatch": "xx"},
        }.items():
            height = table[f"{prefix}_{category}_switches"].mean()
            ax.bar(
                index,
                height,
                bottom=bottom,
                width=width,
                color=colors[method],
                alpha=values["alpha"],
                hatch=values["hatch"],
                edgecolor="white",
            )
            bottom += height
        ax.annotate(
            f"{bottom:.2f}",
            (index, bottom),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
        )
    ax.set(
        title=title,
        ylabel=ylabel,
        xticks=np.arange(len(methods)),
        xticklabels=[labels[method] for method in methods],
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    phase_legend(
        ax,
        {
            "focal": {"label": "Within focal coverage", "alpha": 1, "hatch": ""},
            "gap": {"label": "Within gaps", "alpha": 0.65, "hatch": "//"},
            "seam": {"label": "At gap seams", "alpha": 0.35, "hatch": "xx"},
        },
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(
            {
                "focal": {"label": "Within focal coverage", "alpha": 1, "hatch": ""},
                "gap": {"label": "Within gaps", "alpha": 0.65, "hatch": "//"},
                "seam": {"label": "At gap seams", "alpha": 0.35, "hatch": "xx"},
            }
        ),
        plot_phase_legend_edgecolor="#333333",
        plot_phase_legend_facecolor="#F5F5F5",
        plot_phase_legend_font_size=11,
        plot_phase_legend_handleheight=1.2,
        plot_phase_legend_handlelength=2.4,
        plot_phase_legend_labelspacing=0.6,
    )
    ax = axes[1, 1]
    regions = ["focal", "gap"]
    positions = np.arange(len(regions))
    width = 0.3
    for index, method in enumerate(stitched_methods):
        table = comparisons.loc[comparisons.method.eq(method)]
        means = [table[f"{region}_agreement_percentage"].mean() for region in regions]
        x = positions + (index - (len(stitched_methods) - 1) / 2) * width
        ax.bar(x, means, width=width, color=colors[method], label=labels[method])
        for location, mean in zip(x, means, strict=True):
            ax.annotate(
                f"{mean:.2f}%",
                (location, mean),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=fontsize,
            )
    ax.set(
        title=title_2,
        ylabel=ylabel_2,
        xticks=positions,
        xticklabels=["Focal regions", "Gaps"],
        ylim=ylim,
    )
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(loc="lower right")


def plot_reference_wall_time(
    *,
    ax,
    colors,
    labels,
    methods,
    reference_phases,
    selected,
    width=0.65,
    title="From ancestor reference to written outputs",
    ylabel="Wall time (s)",
):
    """Performance. Presentation choices are supplied by the notebook."""
    for index, method in enumerate(methods):
        table = reference_phases.loc[reference_phases.method.eq(method)]
        means = table.groupby("phase").wall_s.mean()
        bottom = 0
        for phase, values in {
            "full_pipeline": {
                "label": "Full pipeline (Full HMM)",
                "legend_facecolor": "#0072B2",
                "alpha": 1,
                "hatch": "",
            },
            "matching": {"label": "Matching", "alpha": 1, "hatch": ""},
            "focal_preparation": {
                "label": "Focal preparation",
                "alpha": 0.65,
                "hatch": "//",
            },
            "other": {
                "label": "Loading, setup and output",
                "alpha": 0.35,
                "hatch": "xx",
            },
        }.items():
            if phase not in means:
                continue
            height = means.loc[phase]
            ax.bar(
                index,
                height,
                bottom=bottom,
                width=width,
                color=colors[method],
                alpha=values["alpha"],
                hatch=values["hatch"],
                edgecolor="white",
            )
            bottom += height
        times = (
            selected.loc[selected.method.eq(method)]
            .sort_values("repeat")
            .wall_s.to_numpy()
        )
        trial_dots(
            ax,
            index,
            times,
            colors[method],
            plot_repeat_marker_size=20,
            plot_repeat_offsets=[-0.08, 0, 0.08],
        )
        ax.annotate(
            f"{bottom:.1f} s",
            (index, times.max()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )
    ax.set(
        title=title,
        ylabel=ylabel,
        xticks=np.arange(len(methods)),
        xticklabels=[labels[method] for method in methods],
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)


def plot_matching_time_panels(
    *,
    axes,
    colors,
    full_mean,
    labels,
    methods,
    output_dir,
    resident,
    stages,
    stitched_methods,
    tick_labels,
    width=0.65,
    title="Resident matching throughput",
    ylabel="Wall time / matched haplotypes (ms)",
    title_2="Recorded worker time per haplotype",
    ylabel_2="Worker elapsed time (ms)",
    ax,
    worker_phases={
        "focal_match": {
            "label": "Focal matching / greedy mapping",
            "alpha": 1,
            "hatch": "",
        },
        "gap_match": {"label": "Gap HMM", "alpha": 0.65, "hatch": "//"},
        "overhead": {
            "label": "Simplification, setup and stitching",
            "alpha": 0.35,
            "hatch": "xx",
        },
    },
):
    """Performance. Presentation choices are supplied by the notebook."""
    for index, method in enumerate(methods):
        values = (
            resident.loc[resident.method.eq(method)]
            .sort_values("repeat")
            .wall_ms_per_haplotype.to_numpy()
        )
        mean = values.mean()
        speedup = full_mean / mean
        tick_labels.append(f"{labels[method]}\n({speedup:.2f}× Full HMM speed)")
        ax.bar(index, mean, width=width, color=colors[method])
        trial_dots(
            ax,
            index,
            values,
            colors[method],
            plot_repeat_marker_size=20,
            plot_repeat_offsets=[-0.08, 0, 0.08],
        )
        ax.annotate(
            f"{mean:.2f} ms",
            (index, values.max()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )
    ax.set(
        title=title,
        ylabel=ylabel,
        xticks=np.arange(len(methods)),
        xticklabels=tick_labels,
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax = axes[0, 1]
    worker = stages.loc[stages.scope.eq("worker")]
    worker = worker.merge(
        resident[["run_id", "method", "matched_haplotypes"]], on="run_id"
    )
    worker["ms_per_haplotype"] = 1000 * worker.service_s / worker.matched_haplotypes
    worker["phase"] = worker.stage.where(
        worker.stage.isin(["focal_match", "gap_match"]), "overhead"
    )
    per_run = (
        worker.groupby(["run_id", "method", "phase"])
        .ms_per_haplotype.sum()
        .reset_index()
    )
    per_run.to_csv(output_dir / "worker_service_times.csv", index=False)
    for index, method in enumerate(stitched_methods):
        table = per_run.loc[per_run.method.eq(method)]
        means = table.groupby("phase").ms_per_haplotype.mean()
        bottom = 0
        for phase, values in worker_phases.items():
            height = means.get(phase, 0)
            ax.bar(
                index,
                height,
                bottom=bottom,
                width=width,
                color=colors[method],
                alpha=values["alpha"],
                hatch=values["hatch"],
                edgecolor="white",
            )
            bottom += height
        matching_mean = means.loc[["focal_match", "gap_match"]].sum()
        ax.annotate(
            f"Matching: {matching_mean:.1f} ms",
            (index, bottom),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )
    ax.set(
        title=title_2,
        ylabel=ylabel_2,
        xticks=np.arange(len(stitched_methods)),
        xticklabels=[labels[method] for method in stitched_methods],
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
    return ax


def plot_per_sample_data(
    *,
    config,
    labels,
    output_dir,
    repository_dir,
    resident,
    results,
    sample_runs,
    sample_settings,
    stitched_methods,
    worker_tables,
    figsize=(14, 6),
    s=18,
    alpha=0.6,
    xlabel="Focal coverage of sample (%)",
    ylabel="Recorded worker time per haplotype (ms)",
    xlim=(0, 100),
    ylim=(0, None),
    ylabel_2="Samples",
    xlabel_2="Samples",
    title="Population",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Per-sample data. Presentation choices are supplied by the notebook."""
    if len(worker_tables) != len(sample_runs):
        ipython_display.display(
            ipython_display.Markdown(
                "**Per-sample timings unavailable:** the saved benchmark retains only worker-stage totals across all haplotypes. Per-haplotype timings must be collected before these plots can be drawn."
            )
        )
    else:
        inputs = results.manifest["datasets"][config["dataset"]]["inputs"]
        population_path = nb_utils.resolve_path(
            sample_settings["population_metadata"], repository_dir
        )
        sample_data = nb_utils.per_sample_data(
            pd.concat(worker_tables, ignore_index=True),
            pathlib.Path(inputs["mask"]),
            pathlib.Path(inputs["reference"]),
            population_path,
            sample_settings,
            resident,
        )
        sample_data.to_csv(
            output_dir / "per_sample_matching_performance.csv", index=False
        )
        fig = plt.figure(figsize=figsize, layout="constrained", dpi=dpi)
        for axis in fig.axes:
            axis.spines["top"].set_visible(show_top_spine)
            axis.spines["right"].set_visible(show_right_spine)
        fig.suptitle("Per-sample matching performance")
        subfigures = fig.subfigures(1, len(stitched_methods))
        population_handles = []
        for method, subfigure in zip(stitched_methods, subfigures, strict=True):
            subfigure.suptitle(labels[method])
            ratio = 4
            grid = subfigure.add_gridspec(
                2, 2, width_ratios=[ratio, 1], height_ratios=[1, ratio]
            )
            ax = subfigure.add_subplot(grid[1, 0])
            top = subfigure.add_subplot(grid[0, 0], sharex=ax)
            right = subfigure.add_subplot(grid[1, 1], sharey=ax)
            table = sample_data.loc[sample_data.method.eq(method)]
            assert set(table.population) <= set(
                {"CEU": "#0072B2", "CHB": "#D55E00", "YRI": "#009E73"}
            )
            x_bins = np.linspace(0, 100, 25 + 1)
            y_bins = np.linspace(0, table.worker_time_ms.max(), 25 + 1)
            for population, color in {
                "CEU": "#0072B2",
                "CHB": "#D55E00",
                "YRI": "#009E73",
            }.items():
                group = table.loc[table.population.eq(population)]
                points = ax.scatter(
                    group.focal_coverage_percentage,
                    group.worker_time_ms,
                    color=color,
                    s=s,
                    alpha=alpha,
                    label=population,
                )
                top.hist(
                    group.focal_coverage_percentage,
                    bins=x_bins,
                    color=color,
                    alpha=alpha,
                )
                right.hist(
                    group.worker_time_ms,
                    bins=y_bins,
                    orientation="horizontal",
                    color=color,
                    alpha=alpha,
                )
                if method == stitched_methods[0]:
                    population_handles.append(points)
            ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
            top.set(ylabel=ylabel_2)
            right.set(xlabel=xlabel_2)
            top.tick_params(labelbottom=False)
            right.tick_params(labelleft=False)
        fig.legend(
            handles=population_handles,
            title=title,
            loc="outside lower center",
            ncol=len(population_handles),
        )
        benchmark_save_figure(
            fig,
            "per_sample_matching_performance",
            output_dir=output_dir,
            plot_export_dpi=200,
        )


def plot_memory_parallelism_panels(
    *,
    axes,
    colors,
    labels,
    methods,
    mode_count,
    resident,
    runs,
    width,
    fontsize="small",
    title="Peak process memory",
    ylabel="Whole-trial peak RSS (GiB)",
    width_1=0.65,
    color="#777777",
    linestyle="--",
    linewidth=1.7,
    title_2="CPU utilisation during resident matching",
    ylabel_2="Average CPU cores used",
    ax,
):
    """Performance. Presentation choices are supplied by the notebook."""
    for mode_index, (mode, settings) in enumerate(
        {
            "from_reference": {"label": "From reference", "alpha": 1, "hatch": ""},
            "resident_match": {
                "label": "Resident matching",
                "alpha": 0.45,
                "hatch": "//",
            },
        }.items()
    ):
        selected = runs.loc[runs["mode"].eq(mode)]
        for index, method in enumerate(methods):
            values = (
                selected.loc[selected.method.eq(method)]
                .sort_values("repeat")
                .trial_peak_rss_bytes.to_numpy()
            )
            values = values / 1024**3
            position = index + (mode_index - (mode_count - 1) / 2) * width
            ax.bar(
                position,
                values.mean(),
                width=width,
                color=colors[method],
                alpha=settings["alpha"],
                hatch=settings["hatch"],
                edgecolor="white",
            )
            trial_dots(
                ax,
                position,
                values,
                colors[method],
                plot_repeat_marker_size=20,
                plot_repeat_offsets=[-0.08, 0, 0.08],
            )
            ax.annotate(
                f"{values.mean():.1f}",
                (position, values.max()),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=fontsize,
            )
    ax.set(
        title=title,
        ylabel=ylabel,
        xticks=np.arange(len(methods)),
        xticklabels=[labels[method] for method in methods],
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    phase_legend(
        ax,
        {
            "from_reference": {"label": "From reference", "alpha": 1, "hatch": ""},
            "resident_match": {
                "label": "Resident matching",
                "alpha": 0.45,
                "hatch": "//",
            },
        },
        order=list(
            {
                "from_reference": {"label": "From reference", "alpha": 1, "hatch": ""},
                "resident_match": {
                    "label": "Resident matching",
                    "alpha": 0.45,
                    "hatch": "//",
                },
            }
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        plot_phase_legend_edgecolor="#333333",
        plot_phase_legend_facecolor="#F5F5F5",
        plot_phase_legend_font_size=11,
        plot_phase_legend_handleheight=1.2,
        plot_phase_legend_handlelength=2.4,
        plot_phase_legend_labelspacing=0.6,
    )
    ax = axes[0, 1]
    workers = resident.actual_workers.iloc[0]
    for index, method in enumerate(methods):
        values = (
            resident.loc[resident.method.eq(method)]
            .sort_values("repeat")
            .effective_cores.to_numpy()
        )
        ax.bar(index, values.mean(), width=width_1, color=colors[method])
        trial_dots(
            ax,
            index,
            values,
            colors[method],
            plot_repeat_marker_size=20,
            plot_repeat_offsets=[-0.08, 0, 0.08],
        )
        ax.annotate(
            f"{values.mean():.2f}",
            (index, values.max()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )
    ax.axhline(
        workers,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=f"{workers} matching workers",
    )
    ax.set(
        title=title_2,
        ylabel=ylabel_2,
        xticks=np.arange(len(methods)),
        xticklabels=[labels[method] for method in methods],
        ylim=(0, max(workers, resident.effective_cores.max()) * 1.2),
    )
    ax.legend(loc="lower right")


def plot_per_sample_data_2(
    *,
    labels,
    sample_data,
    stitched_methods,
    figsize=(14, 6),
    s=18,
    alpha=0.6,
    xlabel="Focal coverage of sample (%)",
    ylabel="Recorded worker time per haplotype (ms)",
    xlim=(0, 100),
    ylim=(0, None),
    ylabel_2="Samples",
    xlabel_2="Samples",
    title="Population",
    dpi=100,
    show_top_spine=True,
    show_right_spine=True,
):
    """Per-sample data. Presentation choices are supplied by the notebook."""
    fig = plt.figure(figsize=figsize, layout="constrained", dpi=dpi)
    for axis in fig.axes:
        axis.spines["top"].set_visible(show_top_spine)
        axis.spines["right"].set_visible(show_right_spine)
    fig.suptitle("Per-sample matching performance")
    subfigures = fig.subfigures(1, len(stitched_methods))
    population_handles = []
    for method, subfigure in zip(stitched_methods, subfigures, strict=True):
        subfigure.suptitle(labels[method])
        ratio = 4
        grid = subfigure.add_gridspec(
            2, 2, width_ratios=[ratio, 1], height_ratios=[1, ratio]
        )
        ax = subfigure.add_subplot(grid[1, 0])
        top = subfigure.add_subplot(grid[0, 0], sharex=ax)
        right = subfigure.add_subplot(grid[1, 1], sharey=ax)
        table = sample_data.loc[sample_data.method.eq(method)]
        assert set(table.population) <= set(
            {"CEU": "#0072B2", "CHB": "#D55E00", "YRI": "#009E73"}
        )
        x_bins = np.linspace(0, 100, 25 + 1)
        y_bins = np.linspace(0, table.worker_time_ms.max(), 25 + 1)
        for population, color in {
            "CEU": "#0072B2",
            "CHB": "#D55E00",
            "YRI": "#009E73",
        }.items():
            group = table.loc[table.population.eq(population)]
            points = ax.scatter(
                group.focal_coverage_percentage,
                group.worker_time_ms,
                color=color,
                s=s,
                alpha=alpha,
                label=population,
            )
            top.hist(
                group.focal_coverage_percentage, bins=x_bins, color=color, alpha=alpha
            )
            right.hist(
                group.worker_time_ms,
                bins=y_bins,
                orientation="horizontal",
                color=color,
                alpha=alpha,
            )
            if method == stitched_methods[0]:
                population_handles.append(points)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        top.set(ylabel=ylabel_2)
        right.set(xlabel=xlabel_2)
        top.tick_params(labelbottom=False)
        right.tick_params(labelleft=False)
    fig.legend(
        handles=population_handles,
        title=title,
        loc="outside lower center",
        ncol=len(population_handles),
    )
    return fig


def plot_whole_region_results(
    *,
    artifact_dir,
    colors,
    mode_labels,
    mode_order,
    whole,
    whole_names,
    figsize=(11, 4.1),
    width=0.23,
    ylim=(0, None),
    fontsize=9,
    dpi=180,
):
    """Whole-region results. Presentation choices are supplied by the notebook."""
    figure, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    cohort_labels = [
        "n = 320\n640 haplotypes",
        "n = 1,012\n64 haplotypes",
        "n = 3,202\n64 haplotypes",
    ]
    x = np.arange(len(whole_names))
    for ax, metric, scale, title, ylabel in zip(
        axes,
        ["root_edges", "root_span_bp"],
        [1, 1000],
        ["Root-to-sample edges", "Observed root-copying span"],
        ["Edges per tested haplotype", "kb per tested haplotype"],
        strict=True,
    ):
        for offset, mode in enumerate(mode_order):
            table = (
                whole.loc[whole["mode"].eq(mode)].set_index("dataset").loc[whole_names]
            )
            values = table[metric] / table.haplotypes / scale
            ax.bar(
                x + (offset - 1) * 0.23,
                values,
                width=width,
                color=colors[mode],
                label=mode_labels[mode],
            )
        ax.set(
            title=title, ylabel=ylabel, xticks=x, xticklabels=cohort_labels, ylim=ylim
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=fontsize)
    figure.savefig(artifact_dir / "whole_region_comparison.png", dpi=dpi)
    plt.show()


def plot_small_real_data_controls_with_pre_existing_root_edges(
    *,
    artifact_dir,
    colors,
    mode_labels,
    mode_order,
    window_names,
    windows,
    figsize=(10, 4.3),
    width=0.23,
    ylim=(0, None),
    fontsize=9,
    dpi=180,
):
    """Small real-data controls with pre-existing root edges. Presentation choices are supplied by the notebook."""
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(window_names))
    for ax, metric, scale, title, ylabel in zip(
        axes,
        ["root_edges", "root_span_bp"],
        [1, 1000],
        ["Root-edge counts · 64 haplotypes", "Root-copying span · 64 haplotypes"],
        ["Edges", "Summed span (kb)"],
        strict=True,
    ):
        for offset, mode in enumerate(mode_order):
            table = (
                windows.loc[windows["mode"].eq(mode)]
                .set_index("dataset")
                .loc[window_names]
            )
            ax.bar(
                x + (offset - 1) * 0.23,
                table[metric] / scale,
                width=width,
                color=colors[mode],
                label=mode_labels[mode],
            )
        ax.set(
            title=title,
            ylabel=ylabel,
            xticks=x,
            xticklabels=["1–2 Mb", "10–11 Mb"],
            ylim=ylim,
        )
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=fontsize
    )
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(artifact_dir / "small_window_comparison.png", dpi=dpi)
    plt.show()


def show_figure():
    """Display the active notebook figure."""
    plt.show()


def trial_dots(
    ax,
    x,
    values,
    color,
    *,
    plot_repeat_marker_size=20,
    plot_repeat_offsets=(-0.08, 0, 0.08),
):
    """Draw each saved repetition with consistent horizontal spacing."""
    offsets = np.linspace(
        min(plot_repeat_offsets), max(plot_repeat_offsets), len(values)
    )
    ax.scatter(
        x + offsets,
        values,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        s=plot_repeat_marker_size,
        zorder=5,
    )


def save_notebook_figure(
    figure, path, *, dpi=150, bbox_inches="tight", facecolor="white"
):
    """Save a local exploratory figure at its explicit notebook destination."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)


def resize_figure(figure, *, figsize):
    """Set a displayed figure's physical dimensions explicitly."""
    figure.set_size_inches(*figsize)


def plot_conditioning_review_paths(
    paired,
    summary,
    *,
    cohort_labels,
    figsize=(14, 9),
    dpi=120,
    full_color="#555555",
    method_colors,
    method_labels,
    bins=60,
    region_colors={"focal": "#009E73", "gap": "#E69F00", "seam": "#CC79A7"},
):
    """Show each greedy arm against its verified common Full HMM baseline."""
    figure, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, layout="constrained")
    datasets = list(cohort_labels)
    x = np.arange(len(datasets))
    methods = list(method_colors)
    width = 0.8 / (len(methods) + 1)
    full = (
        summary.loc[summary.conditioning.eq(methods[0])]
        .set_index("dataset")
        .loc[datasets]
    )
    axes[0, 0].bar(
        x - width,
        full.sample_mutations_full,
        width=width,
        color=full_color,
        label="Full HMM",
    )
    for index, method in enumerate(methods):
        values = (
            summary.loc[summary.conditioning.eq(method)]
            .set_index("dataset")
            .loc[datasets]
        )
        color = method_colors[method]
        label = method_labels[method]
        style = "-" if index == 0 else "--"
        axes[0, 0].bar(
            x + index * width,
            values.sample_mutations_greedy,
            width=width,
            color=color,
            label=label,
        )
        position = x + (index - (len(methods) - 1) / 2) * 0.36
        positive = np.zeros(len(datasets))
        negative = np.zeros(len(datasets))
        for region, region_color in region_colors.items():
            difference = (
                values[f"stitched_{region}_switches"] - values[f"full_{region}_switches"]
            )
            bottom = np.where(difference >= 0, positive, negative)
            axes[0, 1].bar(
                position,
                difference,
                bottom=bottom,
                width=0.34,
                color=region_color,
                hatch="" if index == 0 else "//",
                label=f"{label} · {region}",
            )
            positive += np.maximum(difference, 0)
            negative += np.minimum(difference, 0)
        for metric, marker, component in [
            ("delta_switch_log_odds", "o", "Transition"),
            ("delta_mismatch_log_odds", "s", "Emission"),
        ]:
            axes[1, 0].plot(
                x,
                -values[metric],
                marker=marker,
                linestyle=style,
                color=color,
                label=f"{label} · {component}",
            )
        for cohort_index, dataset in enumerate(datasets):
            selected = paired.loc[
                paired.dataset.eq(dataset) & paired.conditioning.eq(method)
            ]
            axes[1, 1].hist(
                -selected.delta_log_path_odds,
                bins=bins,
                density=True,
                histtype="step",
                linestyle=style,
                color=f"C{cohort_index}",
                label=f"{cohort_labels[dataset]} · {label}",
            )
    axes[0, 0].set(
        ylabel="Mean terminal mutations / haplotype", title="Sample mismatch events"
    )
    axes[0, 1].set(
        ylabel="Mean extra parent switches / haplotype", title="Greedy minus Full HMM"
    )
    axes[1, 0].set(
        ylabel="Mean extra negative log path weight",
        title="Cost of the saved greedy paths",
    )
    axes[1, 1].set(
        xlabel="Full HMM − greedy log path weight",
        ylabel="Haplotype density",
        title="Paired path-weight losses",
    )
    for ax in axes.flat:
        ax.legend(fontsize="x-small")
    for ax in axes.flat[:3]:
        ax.set_xticks(x, [cohort_labels[name] for name in datasets])
        ax.tick_params(axis="x", labelsize="small")
        ax.axhline(0, color="0.8", linewidth=0.6)
    return figure


def plot_conditioning_review_performance(
    runs,
    *,
    cohort_labels,
    figsize=(12, 7),
    dpi=120,
    full_color="#555555",
    greedy_color="#0072B2",
):
    """Keep latest end-to-end and resident timers separate in categorical panels."""
    figure, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, layout="constrained")
    datasets = list(cohort_labels)
    x = np.arange(len(datasets))
    for row, mode in enumerate(["from_reference", "resident_match"]):
        selected = runs.loc[runs["mode"].eq(mode)]
        for method, label, color in [
            ("full_hmm", "Full HMM", full_color),
            ("greedy", "Conditioned greedy", greedy_color),
        ]:
            values = (
                selected.loc[selected.method.eq(method)]
                .set_index("dataset")
                .loc[datasets]
            )
            axes[row, 0].plot(x, values.wall_s, "o-", color=color, label=label)
        axes[row, 0].set_yscale("log")
        axes[row, 0].set(ylabel="Wall time (s)", title=mode.replace("_", " ").title())
        axes[row, 0].legend(fontsize="small")
        greedy = (
            selected.loc[selected.method.eq("greedy")]
            .set_index("dataset")
            .loc[datasets]
        )
        axes[row, 1].bar(x, greedy.speedup, color=greedy_color)
        axes[row, 1].axhline(1, color=full_color, linestyle="--", linewidth=1)
        axes[row, 1].set(
            ylabel="Full HMM / greedy wall time", title="Speedup (>1 favours greedy)"
        )
        for ax in axes[row]:
            ax.set_xticks(x, [cohort_labels[name] for name in datasets])
            ax.tick_params(axis="x", labelsize="small")
    return figure


def plot_conditioning_failure_overview(
    data,
    *,
    figsize=(16, 10),
    dpi=120,
    map_limits_mb=(26.1, 30.4),
    analysis_end_bp,
    full_color="#222222",
    floor_color="#0072B2",
    impossible_color="#B2182B",
    focal_color="#B8DED4",
    gap_color="#FFE3A3",
):
    """Locate the true map plateau, saved paths and contradictory hard endpoints."""
    figure, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        dpi=dpi,
        layout="constrained",
        gridspec_kw={"height_ratios": [1, 1.5, 1.3]},
    )
    start, end = data.request["start"], data.request["end"]
    initial, final = data.request["initial_state"], data.request["final_state"]
    positions = data.positions
    first, last = positions[start], positions[end - 1]
    facts = data.facts
    ax = axes[0]
    ax.plot(
        data.map_table.position_bp / 1e6,
        data.map_table.map_cm,
        color=full_color,
        linewidth=1.5,
    )
    ax.axvspan(
        facts["map_plateau_start_bp"] / 1e6,
        facts["map_plateau_end_bp"] / 1e6,
        color=gap_color,
        alpha=0.6,
        label="Source-map rate = 0",
    )
    ax.axvspan(
        first / 1e6,
        last / 1e6,
        color=impossible_color,
        alpha=0.4,
        label="Captured failing window",
    )
    ax.axvline(
        analysis_end_bp / 1e6,
        color="0.4",
        linestyle="--",
        label="Analysis ends at 28.1 Mb",
    )
    selected = data.map_table.position_bp.between(
        map_limits_mb[0] * 1e6, map_limits_mb[1] * 1e6
    )
    values = data.map_table.loc[selected, "map_cm"]
    margin = max(float(values.max() - values.min()) * 0.15, 0.015)
    ax.set(
        xlim=map_limits_mb,
        ylim=(values.min() - margin, values.max() + margin),
        xlabel="Chromosome 20 position (Mb)",
        ylabel="Cumulative map (cM)",
        title="The supplied map has an exact 3.86 Mb plateau; this is not exponential-rounding underflow",
    )
    ax.legend(loc="upper left", fontsize="small", ncol=3)
    nodes = sorted(data.parent_labels)
    rows = {node: index for index, node in enumerate(nodes)}
    a, b = data.context_start, data.context_end
    edges = np.r_[positions[a:b], positions[b]] / 1e6
    ax = axes[1]
    ax.axvspan(edges[0], positions[start + 1] / 1e6, color=focal_color, alpha=0.6)
    ax.axvspan(positions[start + 1] / 1e6, last / 1e6, color=gap_color, alpha=0.6)
    ax.axvspan(last / 1e6, edges[-1], color=focal_color, alpha=0.6)
    for name, color, style, label in [
        ("full_zero", full_color, "--", "Saved Full HMM · original zero map"),
        ("greedy_floor", floor_color, "-", "Saved conditioned greedy · r floor 1e−20"),
    ]:
        path = np.array([rows[node] for node in data.parents[name][a:b]])
        ax.stairs(
            path,
            edges,
            baseline=None,
            color=color,
            linestyle=style,
            linewidth=2,
            label=label,
        )
        ax.scatter(positions[a:b] / 1e6, path, s=9, color=color, zorder=3)
    ax.scatter(
        [first / 1e6, last / 1e6],
        [rows[initial], rows[final]],
        s=100,
        facecolors="none",
        edgecolors=impossible_color,
        linewidths=2,
        label="Hard endpoint pins",
        zorder=5,
    )
    for x in [first, last]:
        ax.axvline(x / 1e6, color=impossible_color, linestyle=":", linewidth=1)
    ax.set(
        yticks=range(len(nodes)),
        yticklabels=[data.parent_labels[node] for node in nodes],
        xlim=(edges[0], edges[-1]),
        ylim=(-0.5, len(nodes) - 0.5),
        xlabel="Chromosome 20 position (Mb); each dot is an inference site",
        title="Same actual haplotype: tsk_0 / haplotype 0 · fixed focal runs (green), residual gap (yellow)",
    )
    ax.grid(axis="y", alpha=0.2)
    ax = axes[2]
    left_mm = int(
        data.replay_summary.set_index("case").loc["left_only_zero", "mismatches"]
    )
    right_mm = int(
        data.replay_summary.set_index("case").loc["right_only_zero", "mismatches"]
    )
    ax.plot([0, 1], [1, 1], color=full_color, linewidth=2)
    ax.plot([0, 1], [0, 0], color="0.7", linewidth=2)
    ax.scatter([0], [1], color=floor_color, s=110, zorder=3)
    ax.scatter(
        [1], [0], marker="x", s=180, color=impossible_color, linewidths=3, zorder=3
    )
    ax.annotate(
        "",
        xy=(0.67, 0),
        xytext=(0.36, 1),
        arrowprops={
            "arrowstyle": "->",
            "color": impossible_color,
            "linestyle": "--",
            "linewidth": 2,
        },
    )
    ax.text(
        0.5,
        0.5,
        "Forbidden parent change: T(L,R) = r/n = 0",
        color=impossible_color,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none"},
    )
    ax.text(
        0.05,
        1.12,
        f"Stay on L: {left_mm} mismatches are allowed, but final state is L",
        va="bottom",
    )
    ax.text(
        0.05,
        -0.16,
        f"Stay on R: {right_mm} mismatches are allowed, but initial state is R",
        va="top",
    )
    ax.text(1.01, 0, "Required R\nis unreachable", color=impossible_color, va="center")
    ax.set(
        xlim=(-0.03, 1.25),
        ylim=(-0.4, 1.55),
        yticks=[0, 1],
        yticklabels=[f"R = {final}", f"L = {initial}"],
        xticks=[0, 1],
        xticklabels=[f"Pinned start\nsite {start:,}", f"Pinned end\nsite {end - 1:,}"],
        title="Impossible zero-map constraint: positive mismatch probability cannot create a parent transition",
    )
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize="small")
    return figure


def plot_conditioning_failure_alleles(
    data,
    *,
    start,
    end,
    columns=43,
    figsize=(17, 10),
    dpi=120,
    full_color="#222222",
    floor_color="#0072B2",
    impossible_color="#B2182B",
    allele_colors=("#F6F6F6", "#F2CE72", "#D9D9D9"),
    mismatch_edge_color="#D73027",
    allele_fontsize=9,
    coordinate_fontsize=7,
):
    """Wrap every real allele into readable blocks aligned with categorical paths.

    Binary 0/1 are the simulation's literal allele labels. Red frames mark
    disagreement with the captured sample. No alleles or path are invented
    for the impossible two-pin run: the dotted red line is a one-pin replay.
    """
    blocks = list(range(start, end, columns))
    figure, axes = plt.subplots(
        2 * len(blocks),
        1,
        figsize=figsize,
        dpi=dpi,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.1, 2.5] * len(blocks)},
    )
    nodes = sorted(data.parent_labels)
    rows = {node: index for index, node in enumerate(nodes)}
    cmap = mcolors.ListedColormap(allele_colors)
    initial = data.request["initial_state"]
    final = data.request["final_state"]
    for block, left in enumerate(blocks):
        right = min(left + columns, end)
        sites = np.arange(left, right)
        x = np.arange(len(sites))
        edges = np.arange(len(sites) + 1) - 0.5
        path_ax, allele_ax = axes[2 * block : 2 * block + 2]
        for name, color, style, label in [
            ("full_zero", full_color, "--", "Full HMM · zero map"),
            ("greedy_floor", floor_color, "-", "Conditioned · floor 1e−20"),
            (
                "left_only_zero",
                impossible_color,
                ":",
                "Zero map · left pin only (right pin violated)",
            ),
        ]:
            values = np.array([rows[node] for node in data.parents[name][left:right]])
            offset = (
                0.1
                if name == "greedy_floor"
                else -0.1
                if name == "left_only_zero"
                else 0
            )
            path_ax.stairs(
                values + offset,
                edges,
                baseline=None,
                color=color,
                linestyle=style,
                linewidth=1.8,
                label=label,
            )
        for site, node in [
            (data.request["start"], initial),
            (data.request["end"] - 1, final),
        ]:
            if left <= site < right:
                path_ax.scatter(
                    site - left,
                    rows[node],
                    facecolors="none",
                    edgecolors=impossible_color,
                    linewidths=2,
                    s=95,
                    zorder=5,
                )
        path_ax.set(
            yticks=range(len(nodes)),
            yticklabels=[str(node) for node in nodes],
            ylim=(len(nodes) - 0.5, -0.5),
            xlim=(-0.5, len(sites) - 0.5),
            xticks=[],
            ylabel="Parent node",
            title=f"Inference sites {left:,}–{right - 1:,}; all alleles below use equal-width site columns",
        )
        path_ax.grid(axis="y", alpha=0.2)
        labels = ["Actual sample tsk_0 / hap 0"]
        values = [data.target[left:right]]
        for node in nodes:
            labels.append(data.parent_labels[node])
            values.append(data.reference_alleles[node][left:right])
        for name, label in [
            ("full_zero", "Full HMM copied alleles"),
            ("greedy_floor", "Floored greedy copied alleles"),
        ]:
            labels.append(label)
            values.append(data.copied[name][left:right])
        table = np.array(values)
        plotted = np.where(table < 0, 2, table)
        allele_ax.imshow(
            plotted, aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=2
        )
        for row in range(len(labels)):
            for col, site in enumerate(sites):
                state = table[row, col]
                allele_ax.text(
                    col,
                    row,
                    "·" if state < 0 else str(int(state)),
                    ha="center",
                    va="center",
                    fontsize=allele_fontsize,
                )
                if row > 0 and state >= 0 and state != data.target[site]:
                    rectangle = mpatches.Rectangle(
                        (col - 0.46, row - 0.46),
                        0.92,
                        0.92,
                        fill=False,
                        edgecolor=mismatch_edge_color,
                        linewidth=1.3,
                    )
                    allele_ax.add_patch(rectangle)
        for site in [data.request["start"], data.request["end"] - 1]:
            if left <= site < right:
                allele_ax.axvspan(
                    site - left - 0.5, site - left + 0.5, color=floor_color, alpha=0.12
                )
        ticks = np.arange(0, len(sites), 3)
        coordinates = [f"{sites[i]:,}\n{int(data.positions[sites[i]]):,}" for i in ticks]
        allele_ax.set(
            yticks=range(len(labels)),
            yticklabels=labels,
            xticks=ticks,
            xticklabels=coordinates,
            xlabel="Inference-site index / physical position (bp)",
        )
        allele_ax.tick_params(axis="x", labelsize=coordinate_fontsize, rotation=90)
        allele_ax.tick_params(axis="y", labelsize="small")
        allele_ax.set_xticks(np.arange(-0.5, len(sites), 1), minor=True)
        allele_ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        allele_ax.grid(which="minor", color="0.8", linewidth=0.4)
        allele_ax.tick_params(which="minor", bottom=False, left=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize="small")
    figure.suptitle(
        "Actual binary haplotypes: 0 / 1 are allele states; red frames are sample mismatches\n"
        "Blue-tinted columns and open circles mark hard endpoint pins; no feasible two-pin zero-map path exists"
    )
    return figure
