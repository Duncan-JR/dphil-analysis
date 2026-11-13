"""Chapter 3 workflow orchestration; scientific methods live in thesis_utils.

The root Snakefile supplies all paths. Independent figures run in spawned
processes using queues, up to the available CPU count and allocated threads.
Figure identifiers are independent of their configured output filenames.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import pathlib
import shutil
import subprocess
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from . import thesis_plot, thesis_utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FigureTask:
    name: str
    output: str
    inputs: dict
    settings: dict


@dataclasses.dataclass
class FigureResult:
    name: str
    error: str | None = None


def _plot_figure(task):
    """Prepare each final figure and call its focused renderer."""
    settings = task.settings
    paths = task.inputs
    name = task.name
    config = thesis_utils.AnalysisConfig(**settings["analysis"])
    kwargs = settings["plot_kwargs"].get(name, {}).copy()
    shared_renderers = {
        "demography_overlap_errors": thesis_plot.plot_ancestor_overlap_errors,
        "ancestor_span_simulation": thesis_plot.plot_ancestor_span_curves,
        "ancestor_span_tgp": thesis_plot.plot_ancestor_span_curves,
        "genotype_overshoot_summary": thesis_plot.plot_overshoot_summary,
        "phasing_overshoot_summary": thesis_plot.plot_overshoot_summary,
    }
    if name in shared_renderers:
        function = shared_renderers[name]
    else:
        function = getattr(thesis_plot, f"plot_{name}")

    if name in ("ancestor_extension", "extension_tree"):
        return function()
    if name.startswith("demography_"):
        comparisons = thesis_utils.read_demographies(paths["demographies"])
        if name == "demography_overshoot_quantiles":
            return function(comparisons, config=config)
        summary = thesis_utils.summarise_demography_overlap_errors(comparisons)
        return function(
            summary, config=config, group_labels=thesis_plot.demography_labels, **kwargs
        )
    if name in ("speedup", "elapsed", "resources"):
        performance_config = thesis_utils.PerformanceConfig()
        frame = thesis_utils.load_performance(paths["performance"], performance_config)
        if name == "resources":
            empirical = frame.prefix.eq(settings["empirical_performance_prefix"])
            return function(
                frame.loc[~empirical],
                "parallelism",
                performance_config,
                reference_data=frame.loc[empirical],
            )
        return function(frame, performance_config)
    if name == "phase_switch_illustration":
        return function(paths["phase_runs"], config=config)
    if name == "combined_errors":
        profiles = settings["claims"]["combined_profiles"]
        predicate = pl.col("error_profile").is_in(profiles) & ~pl.col(
            "focal_site_mispolarised"
        )
        frame = thesis_utils.read_comparisons(paths["comparisons"], predicate)
        return function(frame, config=config)
    if name.startswith("genotype_overshoot"):
        config.genotype_multipliers = settings["claims"]["genotype_multipliers"]
        predicate = (pl.col("phase_ser") == 0) & (pl.col("mispol_rate") == 0)
        predicate &= pl.col("geno_multiplier").is_in(config.genotype_multipliers)
    elif (
        name.startswith("mispolarisation_")
        or name == "mispolarised_focal_overshoot_hist"
    ):
        predicate = (pl.col("geno_multiplier") == 0) & (pl.col("phase_ser") == 0)
    elif name.startswith("phasing_"):
        predicate = (pl.col("geno_multiplier") == 0) & (pl.col("mispol_rate") == 0)
    else:
        predicate = thesis_utils.error_free_predicate()
    data = thesis_utils.read_chapter_data(
        paths["comparisons"], paths["validation"], config, predicate
    )
    if name.startswith("ancestor_span_"):
        empirical = thesis_utils.read_table(paths["empirical_spans"])
        summary = thesis_utils.summarise_ancestor_span_curves(data, empirical)
        if "title" in kwargs:
            extent = empirical.focal_position.max() - empirical.focal_position.min()
            kwargs["title"] = kwargs["title"].format(extent_mb=extent / 1e6)
        return function(summary, config=config, **kwargs)
    if name == "ancestor_overlap_errors":
        summary = thesis_utils.summarise_ancestor_overlap_errors(
            data.spans, config, cutoffs=settings["claims"]["frequency_cutoffs"]
        )
        return function(summary, config=config)
    if name == "genotype_error_scaling":
        summary = thesis_utils.summarise_genotype_errors(data)
        return function(summary, config=config)
    if name.endswith("overshoot_summary"):
        by = kwargs.pop("by", "geno_multiplier")
        if by == "geno_multiplier":
            rates = config.genotype_multipliers
            labels = {rate: rf"$\lambda = {rate:g}$" for rate in rates}
        else:
            rates = [0, *sorted(gap for gap in config.phase_spacings if gap != 0)[::-1]]
            labels = {
                gap: "No switches" if gap == 0 else f"{gap / 1e6:g} Mb" for gap in rates
            }
        summary = thesis_utils.summarise_overshoots(data.overshoots, by, rates, config)
        return function(summary, by, rates, labels, config=config, **kwargs)
    if name == "mispolarisation_daf":
        empirical = thesis_utils.read_table(paths["empirical_daf"])
        counts = thesis_utils.mispolarisation_daf_counts(data)
        return function(data, frequencies=counts, tgp_variants=empirical)
    if name == "mispolarisation_overlap_errors":
        comparisons = thesis_utils.mispolarisation_comparisons(data.spans)
        summary = thesis_utils.summarise_mispolarisation_overlap_errors(comparisons)
        return function(summary, config=config)
    return function(data)


def _figure_worker(tasks, results):
    while True:
        task = tasks.get()
        if task is None:
            return
        try:
            output = pathlib.Path(task.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            if task.name == "pipeline_illustration":
                shutil.copyfile(task.inputs["pipeline_illustration"], output)
            else:
                figure = _plot_figure(task)
                if output.suffix == ".pdf":
                    subprocess.run(
                        ["rsvg-convert", "--format=pdf", "--output", str(output)],
                        input=figure,
                        text=True,
                        check=True,
                    )
                else:
                    figure.savefig(
                        output, dpi="figure", bbox_inches="tight", facecolor="white"
                    )
                    plt.close(figure)
            results.put(FigureResult(task.name))
        except Exception:
            results.put(FigureResult(task.name, traceback.format_exc()))
        finally:
            plt.close("all")


def build_figures(inputs, outputs, settings, threads):
    """Generate the configured final figures without upstream inference."""
    destinations = dict(zip(settings["figures"], outputs, strict=True))
    context = multiprocessing.get_context("spawn")
    tasks = context.Queue()
    results = context.Queue()
    worker_count = min(multiprocessing.cpu_count(), threads, len(destinations))
    workers = [
        context.Process(target=_figure_worker, args=(tasks, results))
        for _ in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    for name, output in destinations.items():
        tasks.put(FigureTask(name, str(output), inputs, settings))
    for _ in workers:
        tasks.put(None)
    failures = []
    for _ in destinations:
        result = results.get()
        if result.error is not None:
            failures.append(f"{result.name}:\n{result.error}")
        else:
            logger.info("Generated %s", result.name)
    for worker in workers:
        worker.join()
    tasks.close()
    results.close()
    if failures:
        raise RuntimeError(
            "Chapter 3 figure generation failed:\n" + "\n".join(failures)
        )
