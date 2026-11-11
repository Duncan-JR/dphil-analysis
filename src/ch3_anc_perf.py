"""Plots for the three-stage benchmarks in the chapter 3 notebook."""

import dataclasses
import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PlotConfig:
    """Benchmark ordering, labels, units and figure settings."""

    versions: tuple[str, str] = ("0.4", "0.5")
    version_labels: tuple[str, str] = ("tsinfer 0.4", "tsinfer 0.5")
    version_colors: tuple[str, str] = ("#1a80bb", "#ea801c")
    stages: tuple[str, ...] = (
        "generate_ancestors", "match_ancestors", "match_samples",
    )
    stage_labels: tuple[str, ...] = (
        "Generate ancestors", "Match ancestors", "Match samples",
    )
    stage_colors: tuple[str, ...] = ("#4477aa", "#228833", "#ccbb44")
    seconds_per_minute: int = 60
    bytes_per_gib: int = 1024**3
    group_width: float = 0.8
    sample_labels: dict[int, str] = dataclasses.field(default_factory=dict)
    sample_titles: dict[int, str] = dataclasses.field(default_factory=dict)
    elapsed_bar_width: float = 0.65
    jitter_seed: int = 42
    jitter_width: float = 0.18
    repetition_marker_size: float = 13
    repetition_alpha: float = 0.7
    resource_jitter_log10: float = 0.025
    resource_range_capsize: float = 4
    resource_range_linewidth: float = 0.65
    parallelism_reference_cores: int = 192
    reference_markers: tuple[str, str] = ("^", "^")
    speedup_figsize: tuple[float, float] = (7.5, 3.5)
    speedup_top_margin: float = 0.10
    figsize: tuple[float, float] = (10, 3.8)
    dpi: int = 150


def load_performance(path, config):
    """Retain each run and derive per-run resource metrics.

    Every size must have the same repetitions for both versions and all stages.
    Peak memory is a high-water mark and cannot be summed across stages.
    """
    data = pd.read_csv(path, comment="~", dtype={"version": str})
    keys = ["num_samples", "version", "rep", "command"]
    data = data.set_index(keys).sort_index()
    if not data.index.is_unique:
        raise ValueError("Expected one measurement per sample size, version, rep and stage")
    for size in data.index.get_level_values("num_samples").unique():
        rows = data.xs(size, level="num_samples")
        reps = rows.index.get_level_values("rep").unique()
        expected = pd.MultiIndex.from_product(
            [config.versions, reps, config.stages], names=keys[1:],
        )
        if not rows.index.equals(expected.sort_values()):
            raise ValueError("Expected both versions and all three stages for every rep")
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


def _repetition_points(ax, positions, values, color, config, edgecolor="white", linewidth=0.35):
    """Use reproducible horizontal jitter so repeated values remain visible."""
    ax.scatter(
        positions, values, s=config.repetition_marker_size, color=color,
        alpha=config.repetition_alpha, edgecolors=edgecolor, linewidths=linewidth,
        zorder=4,
    )


def plot_speedup(data, config):
    """Mean paired per-run speedups, with one jittered point per repetition."""
    elapsed = data.elapsed_time.unstack("version")
    speedup = elapsed[config.versions[0]] / elapsed[config.versions[1]]
    means = speedup.groupby(["num_samples", "command"]).mean()
    sizes = means.index.get_level_values("num_samples").unique()
    x = np.arange(len(sizes))
    width = config.group_width / len(config.stages)
    rng = np.random.default_rng(config.jitter_seed)
    fig, ax = plt.subplots(figsize=config.speedup_figsize, dpi=config.dpi, layout="constrained")
    for index, (stage, label, color) in enumerate(
        zip(config.stages, config.stage_labels, config.stage_colors)
    ):
        offset = (index - (len(config.stages) - 1) / 2) * width
        values = means.xs(stage, level="command").reindex(sizes)
        bars = ax.bar(x + offset, values, width=width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f×", fontsize=8, padding=3)
        for position, size in zip(x + offset, sizes):
            reps = speedup.xs((size, stage), level=("num_samples", "command"))
            jitter = rng.uniform(-config.jitter_width, config.jitter_width, len(reps))
            _repetition_points(ax, position + width * jitter, reps, "black", config)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    sample_labels = [config.sample_labels.get(size, f"{size:,}") for size in sizes]
    ax.set_xticks(x, sample_labels)
    ax.set_xlabel("Sample size (haplotypes)")
    ax.set_ylabel("Speedup (tsinfer 0.4 / tsinfer 0.5)")
    ax.set_ylim(0, max(1, speedup.max()) * (1 + config.speedup_top_margin))
    ax.legend(frameon=False, loc="upper left")
    return fig


def plot_elapsed(data, config):
    """Mean stage stacks with per-run cumulative times at each stack boundary.

    Total speedup is the mean of paired per-run total-time ratios.
    """
    sizes = data.index.get_level_values("num_samples").unique()
    rng = np.random.default_rng(config.jitter_seed)
    fig, axes = plt.subplots(
        1, len(sizes), figsize=config.figsize, dpi=config.dpi, layout="constrained",
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
            jitter = rng.uniform(-config.jitter_width, config.jitter_width, n)
            jitter_positions[version] = position + config.elapsed_bar_width * jitter
        for stage, label, color in zip(
            config.stages, config.stage_labels, config.stage_colors
        ):
            values = means[stage].to_numpy()
            ax.bar(
                positions, values, bottom=bottom, label=label, color=color,
                width=config.elapsed_bar_width,
            )
            bottom += values
            for version in config.versions:
                reps = cumulative.xs(version, level="version")[stage]
                _repetition_points(
                    ax, jitter_positions[version], reps, color, config, edgecolor="black", linewidth=0.6,
                )
        totals = runs.sum(axis=1).unstack("version")
        speedup = (totals[config.versions[0]] / totals[config.versions[1]]).mean()
        for position, total in zip(positions, bottom):
            annotation = f"{total:,.1f}"
            if position == 1:
                annotation += f"\n({speedup:.2f}×)"
            # Place the label above every individual total as well as the mean.
            highest = totals[config.versions[position]].max()
            ax.annotate(
                annotation, (position, highest), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=8,
            )
        sample_label = config.sample_labels.get(size, f"{size:,}")
        ax.set_title(config.sample_titles.get(size, f"{sample_label}\nhaplotypes"))
        ax.set_xticks(positions, config.version_labels, rotation=20)
        ax.set_ylim(0, totals.to_numpy().max() * 1.18)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    axes[0].set_ylabel("Elapsed time (minutes)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(config.stages), frameon=False)
    return fig


def plot_resources(data, metric, config, reference_data=None):
    """Mean resource curves with repetition variability.

    Memory retains jittered individual runs on log-log axes. Parallelism uses
    capped min–max error bars on a shared linear y scale, without jitter.
    Isolated triangles mark mean 1KGP resources, using the same error bars.
    """
    ylabel = {"memory_gib": "Peak resident memory (GiB)", "parallelism": "Average active CPU cores"}[metric]
    fig, axes = plt.subplots(
        1, len(config.stages), figsize=config.figsize, dpi=config.dpi,
        sharex=True, sharey=True, layout="constrained",
    )
    sizes = data.index.get_level_values("num_samples").unique()
    rng = np.random.default_rng(config.jitter_seed)
    for ax, stage, stage_label in zip(axes, config.stages, config.stage_labels):
        ax.set_xscale("log")
        for version, version_label, color in zip(
            config.versions, config.version_labels, config.version_colors
        ):
            values = data.xs((version, stage), level=("version", "command"))
            means = values[metric].groupby("num_samples").mean().reindex(sizes)
            label = version_label if reference_data is None else f"{version_label} (sim)"
            if metric == "parallelism":
                ranges = values[metric].groupby("num_samples").agg(["min", "max"])
                ranges = ranges.reindex(sizes)
                errors = np.vstack([means - ranges["min"], ranges["max"] - means])
                ax.errorbar(
                    sizes, means.to_numpy(dtype=float), yerr=errors, marker="o", color=color, label=label,
                    capsize=config.resource_range_capsize,
                    elinewidth=config.resource_range_linewidth,
                    capthick=config.resource_range_linewidth,
                )
            else:
                ax.plot(sizes, means, marker="o", color=color, label=label)
                run_sizes = values.index.get_level_values("num_samples").to_numpy()
                jitter = rng.uniform(-config.resource_jitter_log10, config.resource_jitter_log10, len(values))
                _repetition_points(ax, run_sizes * 10**jitter, values[metric], color, config)
        if reference_data is not None:
            for version, label, color, marker in zip(
                config.versions, config.version_labels, config.version_colors,
                config.reference_markers,
            ):
                reference = reference_data.xs((version, stage), level=("version", "command"))
                means = reference[metric].groupby("num_samples").mean()
                if metric == "parallelism":
                    ranges = reference[metric].groupby("num_samples").agg(["min", "max"])
                    errors = np.vstack([means - ranges["min"], ranges["max"] - means])
                    ax.errorbar(
                        means.index, means.to_numpy(dtype=float), yerr=errors, marker=marker, linestyle="none",
                        color=color, label=f"{label} (1KGP)", zorder=3,
                        capsize=config.resource_range_capsize,
                        elinewidth=config.resource_range_linewidth,
                        capthick=config.resource_range_linewidth,
                    )
                else:
                    ax.scatter(
                        means.index, means, color=color, marker=marker,
                        label=f"{label} (1KGP)", zorder=3,
                    )
                    run_sizes = reference.index.get_level_values("num_samples").to_numpy()
                    jitter = rng.uniform(-config.resource_jitter_log10, config.resource_jitter_log10, len(reference))
                    _repetition_points(ax, run_sizes * 10**jitter, reference[metric], color, config)
        if metric == "parallelism":
            ax.axhline(config.parallelism_reference_cores, color="gray", linestyle=":",
                       label=f"{config.parallelism_reference_cores} cores")
            ax.set_ylim(bottom=0)
        else:
            ax.set_yscale("log")
        ax.set_xticks(sizes, [f"{size:,}" for size in sizes], rotation=30)
        ax.set_xlabel("Sample size (haplotypes)")
        ax.set_title(stage_label)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    reference_label = f"{config.parallelism_reference_cores} cores"
    if reference_label in labels:
        index = labels.index(reference_label)
        handles.append(handles.pop(index))
        labels.append(labels.pop(index))
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    return fig
