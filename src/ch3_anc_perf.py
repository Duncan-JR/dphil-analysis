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

    versions: tuple[str, str] = ("0.4.1", "0.5.1")
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
    figsize: tuple[float, float] = (10, 3.8)
    dpi: int = 150


def load_performance(path, config):
    """Load one measurement per size/version/stage and derive resource metrics.

    ``~`` marks an editor footer in the supplied aggregate CSV. Peak memory is
    a high-water mark, not a quantity that can be summed across stages.
    """
    data = pd.read_csv(path, comment="~")
    keys = ["num_samples", "version", "command"]
    data = data.set_index(keys, verify_integrity=True).sort_index()
    sizes = data.index.get_level_values("num_samples").unique()
    expected = pd.MultiIndex.from_product(
        [sizes, config.versions, config.stages], names=keys,
    )
    if not data.index.sort_values().equals(expected.sort_values()):
        raise ValueError("Expected both versions and all three stages at each sample size")
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


def plot_speedup(data, config):
    """Group elapsed-time ratios by sample size, with one bar per stage."""
    elapsed = data.elapsed_time.unstack("version")
    speedup = elapsed[config.versions[0]] / elapsed[config.versions[1]]
    sizes = speedup.index.get_level_values("num_samples").unique()
    x = np.arange(len(sizes))
    width = config.group_width / len(config.stages)
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, layout="constrained")
    for index, (stage, label, color) in enumerate(
        zip(config.stages, config.stage_labels, config.stage_colors)
    ):
        offset = (index - (len(config.stages) - 1) / 2) * width
        values = speedup.xs(stage, level="command").reindex(sizes)
        bars = ax.bar(x + offset, values, width=width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f×", fontsize=8, padding=3)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"{size:,}" for size in sizes])
    ax.set_xlabel("Sample size (haplotypes)")
    ax.set_ylabel("Speedup (0.4 elapsed / 0.5 elapsed)")
    ax.set_ylim(0, speedup.max() * 1.25)
    ax.legend(frameon=False, loc="upper left")
    return fig


def plot_elapsed(data, config):
    """Two additive stage stacks per size, with a local linear time scale."""
    sizes = data.index.get_level_values("num_samples").unique()
    fig, axes = plt.subplots(
        1, len(sizes), figsize=config.figsize, dpi=config.dpi, layout="constrained",
    )
    for ax, size in zip(np.atleast_1d(axes), sizes):
        bottom = np.zeros(len(config.versions))
        for stage, label, color in zip(
            config.stages, config.stage_labels, config.stage_colors
        ):
            times = data.xs((size, stage), level=("num_samples", "command"))
            values = times.elapsed_minutes.reindex(config.versions).to_numpy()
            ax.bar(config.versions, values, bottom=bottom, label=label, color=color)
            bottom += values
        for version, total in zip(config.versions, bottom):
            ax.annotate(
                f"{total:,.1f}", (version, total), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=8,
            )
        ax.set_title(f"{size:,} haplotypes")
        ax.set_xlabel("tsinfer version")
        ax.set_ylim(0, bottom.max() * 1.18)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    axes[0].set_ylabel("Elapsed time (minutes)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(config.stages), frameon=False)
    return fig


def plot_resources(data, metric, config):
    """Compare versions over sample size in separate panels for each stage.

    Memory uses log-log axes; average active CPU cores uses a log sample axis
    and a shared linear scale. CPU time includes user and system time.
    """
    ylabel = {"memory_gib": "Peak resident memory (GiB)", "parallelism": "Average active CPU cores"}[metric]
    fig, axes = plt.subplots(
        1, len(config.stages), figsize=config.figsize, dpi=config.dpi,
        sharex=True, sharey=True, layout="constrained",
    )
    sizes = data.index.get_level_values("num_samples").unique()
    for ax, stage, label in zip(axes, config.stages, config.stage_labels):
        for version, color in zip(config.versions, config.version_colors):
            values = data.xs((version, stage), level=("version", "command"))
            values = values.reindex(sizes)
            ax.plot(sizes, values[metric], marker="o", color=color, label=f"tsinfer {version}")
        if metric == "parallelism":
            threads = data.xs(stage, level="command").num_threads.unstack("version")
            for index, version in enumerate(config.versions):
                reference = "Configured threads" if index == 0 else "_nolegend_"
                ax.plot(sizes, threads[version], color="gray", linestyle=":", label=reference)
            ax.set_ylim(bottom=0)
        else:
            ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{size:,}" for size in sizes], rotation=30)
        ax.set_xlabel("Sample size (haplotypes)")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    return fig
