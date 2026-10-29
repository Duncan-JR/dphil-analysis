"""Replay benchmark ancestor grouping without loading genotypes or running HMMs.

The matched ancestor TS retains each original ancestor's time, ID and covered
interval. These are sufficient to replay the legacy hybrid linesweep/epoch
scheduler. The original standalone grouping module is read from a pinned local
Git tag; the installed tsinfer matcher is never imported or invoked.
"""

import dataclasses
import json
import logging
import multiprocessing
import pathlib
import queue
import re
import subprocess
import sys
import traceback
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tskit

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AnalysisConfig:
    """Source revision and settings for the benchmark scheduler replay."""

    tsinfer_source: pathlib.Path = pathlib.Path("~/work/tsinfer")
    grouping_ref: str = "0.5.1"
    epoch_cutoff_factor: int = 500
    workers: int | None = None
    queue_timeout: float = 1
    grid_size: tuple[float, float] = (10, 7)
    point_size: float = 5
    point_alpha: float = 0.6


@dataclasses.dataclass
class AncestorInputs:
    start: np.ndarray
    end: np.ndarray
    time: np.ndarray
    position: np.ndarray
    tree_stats: dict


@dataclasses.dataclass
class GroupingResult:
    groups: dict
    epoch_first_group: int


@dataclasses.dataclass
class AnalysisTask:
    folder: pathlib.Path
    num_samples: int
    version: str
    analysis_config: AnalysisConfig


@dataclasses.dataclass
class AnalysisResult:
    groups: pd.DataFrame
    summary: dict


def load_ancestor_inputs(path):
    """Recover original ancestor intervals from their complete copying paths.

    Original ancestors are the initial sample-flagged nodes; later nodes are
    path-compression ancestors. Only the virtual root (ID 0) lacks a copying
    path, and its interval spans all inference sites. Endpoint mapping matches
    the legacy Matcher's position map, including position 0 at the left edge.
    """
    ts = tskit.load(path)
    num_ancestors = ts.num_samples
    if not np.array_equal(ts.samples(), np.arange(num_ancestors)):
        raise ValueError("Expected original ancestors to be the initial sample nodes")
    left = np.full(ts.num_nodes, np.inf)
    right = np.full(ts.num_nodes, -np.inf)
    np.minimum.at(left, ts.edges_child, ts.edges_left)
    np.maximum.at(right, ts.edges_child, ts.edges_right)
    left = left[:num_ancestors]
    right = right[:num_ancestors]
    left[0] = 0
    right[0] = ts.sequence_length
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Each nonvirtual ancestor must have a copying path")
    positions = np.append(ts.sites_position, ts.sequence_length)
    positions[0] = 0
    start = np.searchsorted(positions, left)
    end = np.searchsorted(positions, right)
    if not np.array_equal(positions[start], left) or not np.array_equal(positions[end], right):
        raise ValueError("Ancestor endpoints must map to inference-site boundaries")
    stats = {
        "num_ancestors": num_ancestors,
        "num_inference_sites": ts.num_sites,
        "ancestor_ts_nodes": ts.num_nodes,
        "ancestor_ts_edges": ts.num_edges,
        "pc_ancestors": ts.num_nodes - num_ancestors,
        "sequence_length": ts.sequence_length,
    }
    return AncestorInputs(start, end, ts.nodes_time[:num_ancestors], positions, stats)


def load_legacy_grouping(config):
    """Load the standalone grouping code at the benchmark's legacy revision.

    This module only imports NumPy/Numba, so no legacy Zarr environment or
    compiled matcher is required. Both benchmark versions use this scheduler.
    """
    command = [
        "git", "-C", str(config.tsinfer_source.expanduser()),
        "show", f"{config.grouping_ref}:tsinfer/ancestors.py",
    ]
    source = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    module = types.ModuleType("ch3_legacy_ancestors")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{config.grouping_ref}:tsinfer/ancestors.py", "exec"), module.__dict__)
    return module


def replay_grouping(inputs, config, grouping):
    """Replay AncestorMatcher.group_by_linesweep's hybrid grouping exactly.

    Linesweep only covers the older, small epochs. Young epochs beyond the
    original median-size cutoff retain their time grouping. Preserve the
    legacy group-ID gap at that boundary and remove the virtual-root task.
    """
    breaks = np.flatnonzero(inputs.time[1:] != inputs.time[:-1]) + 1
    epoch_start = np.r_[0, breaks]
    epoch_end = np.r_[breaks, len(inputs.time)]
    sizes = epoch_end - epoch_start
    cutoff = config.epoch_cutoff_factor * np.median(sizes)
    sizes[:len(sizes) // 2] = 0
    large = np.flatnonzero(sizes > cutoff)
    first_large = int(large[0]) if len(large) > 0 else len(sizes)
    stop = int(epoch_start[first_large]) if first_large < len(sizes) else len(inputs.time)
    groups = grouping.group_ancestors_by_linesweep(
        inputs.start[:stop], inputs.end[:stop], inputs.time[:stop],
    )
    first_epoch_group = len(groups) + 1
    for index in range(first_large, len(sizes)):
        group_id = first_epoch_group + index - first_large
        groups[group_id] = np.arange(epoch_start[index], epoch_end[index])
    groups[0].remove(0)
    return GroupingResult(groups, first_epoch_group)


def load_group_timings(path):
    """Read complete group sizes and elapsed times from an ancestor-match log."""
    text = pathlib.Path(path).read_text()
    starts = re.findall(r"Starting group (\d+) of \d+ with (\d+) ancestors", text)
    ends = re.findall(r"Finished group (\d+) of \d+ in ([\d.]+) seconds", text)
    sizes = pd.Series({int(group): int(size) for group, size in starts}, name="size")
    times = pd.Series({int(group): float(seconds) for group, seconds in ends}, name="seconds")
    if set(sizes.index) != set(times.index):
        raise ValueError(f"Incomplete group timing log: {path}")
    result = pd.concat([sizes, times], axis=1)
    result.index.name = "group"
    return result


def analyse_task(task, grouping):
    """Replay groups, validate every size against the log, and summarise work."""
    inputs = load_ancestor_inputs(task.folder / "ancestors.trees")
    replay = replay_grouping(inputs, task.analysis_config, grouping)
    rows = []
    spans = inputs.end - inputs.start
    bp = inputs.position[inputs.end] - inputs.position[inputs.start]
    for group_id, ids in replay.groups.items():
        rows.append({
            "group": group_id,
            "size": len(ids),
            "total_sites": spans[ids].sum(),
            "max_sites": spans[ids].max() if len(ids) > 0 else 0,
            "total_bp": bp[ids].sum(),
            "scheduler": "epoch" if group_id >= replay.epoch_first_group else "linesweep",
        })
    groups = pd.DataFrame(rows).set_index("group")
    logged = load_group_timings(task.folder / "match_ancestors.log")
    pd.testing.assert_series_equal(groups['size'], logged['size'])
    groups["seconds"] = logged.seconds
    groups["num_samples"] = task.num_samples
    groups["version"] = task.version
    summary = dict(inputs.tree_stats)
    summary.update(
        num_samples=task.num_samples,
        version=task.version,
        num_groups=len(groups),
        median_group_size=groups['size'].median(),
        mean_ancestor_sites=spans[2:].mean(),
        median_ancestor_sites=np.median(spans[2:]),
        total_ancestor_sites=spans[2:].sum(),
        mean_ancestor_bp=bp[2:].mean(),
        median_ancestor_bp=np.median(bp[2:]),
        group_seconds=groups.seconds.sum(),
        linesweep_seconds=groups.loc[groups.scheduler == "linesweep", "seconds"].sum(),
        epoch_seconds=groups.loc[groups.scheduler == "epoch", "seconds"].sum(),
    )
    inferred = tskit.load(task.folder / "inferred.trees")
    summary.update(
        inferred_nodes=inferred.num_nodes,
        inferred_edges=inferred.num_edges,
        inferred_mutations=inferred.num_mutations,
    )
    return AnalysisResult(groups.reset_index(), summary)


def analysis_worker(tasks, results, config):
    """Consume dataset/version tasks from the shared multiprocessing queue."""
    try:
        grouping = load_legacy_grouping(config)
        while True:
            task = tasks.get()
            if task is None:
                break
            result = analyse_task(task, grouping)
            results.put(result)
    except Exception:
        results.put(traceback.format_exc())


def preprocess_results(folder, plot_config, analysis_config):
    """Replay all completed runs in parallel and cache compact analysis CSVs.

    Available CPU cores consume independent size/version jobs. No HMM is run.
    Call explicitly to refresh after changing the benchmark inputs or revision.
    """
    folder = pathlib.Path(folder)
    perf = pd.read_csv(folder / "perf_combined.csv", comment="~")
    runs = perf[["num_samples", "version"]].drop_duplicates()
    tasks = []
    for run in runs.itertuples(index=False):
        task_folder = folder / run.version / f"n{run.num_samples}"
        tasks.append(AnalysisTask(task_folder, run.num_samples, run.version, analysis_config))
    worker_count = analysis_config.workers
    if worker_count is None:
        worker_count = multiprocessing.cpu_count()
    worker_count = min(worker_count, len(tasks))
    context = multiprocessing.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    workers = [
        context.Process(target=analysis_worker, args=(task_queue, result_queue, analysis_config))
        for _ in range(worker_count)
    ]
    for task in tasks:
        task_queue.put(task)
    for _ in workers:
        task_queue.put(None)
    for worker in workers:
        worker.start()
    completed = []
    try:
        while len(completed) < len(tasks):
            try:
                result = result_queue.get(timeout=analysis_config.queue_timeout)
            except queue.Empty:
                failures = [worker.exitcode for worker in workers if worker.exitcode not in (None, 0)]
                if len(failures) > 0 or not any(worker.is_alive() for worker in workers):
                    raise RuntimeError(f"Grouping workers stopped before finishing: {failures}")
                continue
            if isinstance(result, str):
                raise RuntimeError(result)
            completed.append(result)
            logger.info("Replayed tsinfer %s, n=%s", result.summary['version'], result.summary['num_samples'])
    finally:
        for worker in workers:
            if len(completed) < len(tasks) and worker.is_alive():
                worker.terminate()
            worker.join()
        task_queue.close()
        result_queue.close()
    group_frames = [result.groups for result in completed]
    groups = pd.concat(group_frames, ignore_index=True)
    groups = groups.sort_values(["num_samples", "version", "group"])
    summaries = [result.summary for result in completed]
    summary = pd.DataFrame(summaries).sort_values(["num_samples", "version"])
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    groups.to_csv(output / "ancestor_groups.csv", index=False)
    summary.to_csv(output / "ancestor_summary.csv", index=False)
    metadata = dataclasses.asdict(analysis_config)
    metadata['tsinfer_source'] = str(analysis_config.tsinfer_source.expanduser())
    (output / "replay_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summarise_results(groups, summary, perf, plot_config)


@dataclasses.dataclass
class BenchmarkAnalysis:
    groups: pd.DataFrame
    summary: pd.DataFrame


def summarise_results(groups, summary, perf, config):
    """Combine structural summaries, group timing and stage resource metrics."""
    keys = ["num_samples", "version"]
    summary = summary.set_index(keys)
    timings = perf.set_index(keys + ["command"])
    summary["num_threads"] = timings.xs("match_ancestors", level="command").num_threads
    for stage in config.stages:
        values = timings.xs(stage, level="command")
        cpu = values.user_time + values.sys_time
        summary[f"{stage}_cpu_seconds"] = cpu
        summary[f"{stage}_elapsed_seconds"] = values.elapsed_time
        summary[f"{stage}_mean_cores"] = cpu / values.elapsed_time
    for key, row in summary.iterrows():
        size, version = key
        selected = groups.loc[(groups.num_samples == size) & (groups.version == version)]
        threads = timings.loc[(size, version, "match_ancestors"), "num_threads"]
        small = selected['size'] < threads
        summary.loc[key, "underfull_group_fraction"] = small.mean()
        small_seconds = selected.loc[small, "seconds"].sum()
        summary.loc[key, "underfull_time_fraction"] = small_seconds / selected.seconds.sum()
    return BenchmarkAnalysis(groups, summary.reset_index())


def load_analysis(folder, config):
    """Load the small replay caches; see :func:`preprocess_results` to refresh."""
    folder = pathlib.Path(folder)
    groups = pd.read_csv(folder / "analysis/ancestor_groups.csv")
    summary = pd.read_csv(folder / "analysis/ancestor_summary.csv")
    perf = pd.read_csv(folder / "perf_combined.csv", comment="~")
    return summarise_results(groups, summary, perf, config)


def plot_linesweep(analysis, config, analysis_config):
    """Four sample-size panels comparing the actual hybrid scheduling groups."""
    sizes = sorted(analysis.groups.num_samples.unique())
    if len(sizes) != 4:
        raise ValueError("The 2 × 2 linesweep grid requires four sample sizes")
    fig, axes = plt.subplots(
        2, 2, figsize=analysis_config.grid_size, dpi=config.dpi, layout="constrained",
    )
    for ax, size in zip(axes.flat, sizes):
        for version, label, color in zip(config.versions, config.version_labels, config.version_colors):
            rows = analysis.groups.loc[
                (analysis.groups.num_samples == size) & (analysis.groups.version == version)
            ]
            rows = rows.sort_values('group')
            nonempty = rows.loc[rows['size'] > 0]
            ax.scatter(
                nonempty['group'], nonempty['size'], label=label, color=color,
                s=analysis_config.point_size, alpha=analysis_config.point_alpha, rasterized=True,
            )
        sample_summary = analysis.summary.loc[analysis.summary.num_samples == size]
        threads = sample_summary.num_threads.unique()
        if len(threads) != 1:
            raise ValueError("Expected the same configured threads for both versions")
        ax.axhline(threads[0], color="gray", linestyle=":", linewidth=1, label=f"{threads[0]} threads")
        ax.set_title(f"{size:,} haplotypes")
        ax.set_yscale("log")
        ax.set_xlabel("Matching group (oldest → youngest)")
        ax.set_ylabel("Ancestors per group")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    return fig


def plot_work_reduction(analysis, config):
    """Contrast saved CPU work with saved elapsed time for each matching stage."""
    summary = analysis.summary.set_index(["num_samples", "version"])
    fig, axes = plt.subplots(
        1, len(config.stages) - 1, figsize=config.figsize, dpi=config.dpi,
        sharey=True, layout="constrained",
    )
    for ax, stage, label in zip(axes, config.stages[1:], config.stage_labels[1:]):
        for metric, marker, metric_label in [
            ("cpu_seconds", "o", "CPU-time reduction"),
            ("elapsed_seconds", "s", "Elapsed-time speedup"),
        ]:
            values = summary[f"{stage}_{metric}"].unstack("version")
            reduction = values[config.versions[0]] / values[config.versions[1]]
            ax.plot(reduction.index, reduction, marker=marker, label=metric_label)
        ax.axhline(1, color="gray", linestyle=":")
        ax.set_title(label)
        ax.set_xscale("log")
        ax.set_xticks(values.index, [f"{size:,}" for size in values.index])
        ax.set_xlabel("Sample size (haplotypes)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Ratio (tsinfer 0.4 / tsinfer 0.5)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    return fig
