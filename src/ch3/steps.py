"""Chapter 3 workflow orchestration; scientific methods live in thesis_utils.

The root Snakefile supplies all paths. Independent figures run in spawned
processes using queues, up to the available CPU count and allocated threads.
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
    """Load each figure's required selection and call its focused renderer."""
    settings = task.settings
    paths = task.inputs
    config = thesis_utils.AnalysisConfig(**settings["analysis"])
    specification = settings["figures"][task.name]
    function = getattr(thesis_plot, specification["function"])
    kwargs = specification["kwargs"].copy()
    name = task.name
    if name.startswith("ancestor_extension"):
        extension = thesis_utils.ExtensionConfig(**settings["extension"])
        if name.endswith(".pdf"):
            trees = thesis_utils.build_extension_trees(
                extension, settings["extension_trees"]
            )
            return function(trees.tree_sequence, trees.node_ids, extension, **kwargs)
        return function(extension, **kwargs)
    if name.startswith("demography_"):
        comparisons = thesis_utils.read_demographies(paths["demographies"])
        labels = settings["demography_labels"]
        if name == "demography_overshoot_box.png":
            return function(comparisons, labels, config=config, **kwargs)
        summary = thesis_utils.summarise_demography_overlap_errors(comparisons)
        return function(summary, config=config, group_labels=labels, **kwargs)
    if name.startswith("inference_performance_"):
        performance_config = thesis_utils.PerformanceConfig(**settings["performance"])
        frame = thesis_utils.load_performance(paths["performance"], performance_config)
        if name.endswith("parallelism.png"):
            empirical = frame.prefix.eq(settings["empirical_performance_prefix"])
            return function(
                frame.loc[~empirical],
                "parallelism",
                performance_config,
                reference_data=frame.loc[empirical],
                **kwargs,
            )
        return function(frame, performance_config, **kwargs)
    if name == "phase_switch_illustration.png":
        return function(paths["phase_runs"], config=config, **kwargs)
    if name == "combined_errors.png":
        profiles = kwargs["combined_error_profiles"]
        predicate = pl.col("error_profile").is_in(profiles) & ~pl.col(
            "focal_site_mispolarised"
        )
        frame = thesis_utils.read_comparisons(paths["comparisons"], predicate)
        return function(frame, config=config, **kwargs)
    if name.startswith("genotype_error_overshoot"):
        config.genotype_multipliers = settings["claims"]["genotype_multipliers"]
        predicate = (pl.col("phase_ser") == 0) & (pl.col("mispol_rate") == 0)
        predicate &= pl.col("geno_multiplier").is_in(config.genotype_multipliers)
    elif name.startswith("mispolarisation_"):
        predicate = (pl.col("geno_multiplier") == 0) & (pl.col("phase_ser") == 0)
    elif name.startswith("phasing_"):
        predicate = (pl.col("geno_multiplier") == 0) & (pl.col("mispol_rate") == 0)
    else:
        predicate = thesis_utils.error_free_predicate()
    data = thesis_utils.read_chapter_data(
        paths["comparisons"], paths["validation"], config, predicate
    )
    if name.startswith("ancestor_span_frequency_"):
        empirical = thesis_utils.read_table(paths["empirical_spans"])
        summary = thesis_utils.summarise_ancestor_span_curves(
            data, empirical, ancestor_error_bins=settings["span_curve_bins"]
        )
        dataset = (
            "Simulation" if name.endswith("simulation.png") else "1000 Genomes Project"
        )
        if dataset == "1000 Genomes Project":
            extent = empirical.focal_position.max() - empirical.focal_position.min()
            kwargs["title"] = settings["empirical_span_title"].format(
                extent_mb=extent / config.bp_per_mb
            )
        return function(summary, dataset, config=config, **kwargs)
    if name == "ancestor_overlap_errors.png":
        summary = thesis_utils.summarise_ancestor_overlap_errors(
            data.spans, config, cutoffs=settings["claims"]["frequency_cutoffs"]
        )
        return function(summary, config=config, **kwargs)
    if name == "genotype_error_scaling.png":
        summary = thesis_utils.summarise_genotype_errors(
            data, transition_bins=settings["transition_bins"]
        )
        return function(summary, config=config, **kwargs)
    if name.endswith("overshoot_summary.png"):
        if name.startswith("genotype"):
            by = "geno_multiplier"
            rates = config.genotype_multipliers
            labels = {rate: rf"$\lambda = {rate:g}$" for rate in rates}
        else:
            by = "space_per_switch"
            rates = [0, *sorted(gap for gap in config.phase_spacings if gap != 0)[::-1]]
            labels = {
                gap: "No switches" if gap == 0 else f"{gap / config.bp_per_mb:g} Mb"
                for gap in rates
            }
        summary = thesis_utils.summarise_overshoots(data.overshoots, by, rates, config)
        return function(summary, by, rates, labels, config=config, **kwargs)
    if name == "mispolarisation_daf.png":
        empirical = thesis_utils.read_table(paths["empirical_daf"])
        counts = thesis_utils.mispolarisation_daf_counts(data)
        return function(data, frequencies=counts, tgp_variants=empirical, **kwargs)
    if name == "mispolarisation_overlap_errors.png":
        comparisons = thesis_utils.mispolarisation_comparisons(
            data.spans, kwargs["mispolarisation_error_rates"]
        )
        summary = thesis_utils.summarise_mispolarisation_overlap_errors(comparisons)
        return function(summary, config=config, **kwargs)
    return function(data, **kwargs)


def _figure_worker(tasks, results):
    while True:
        task = tasks.get()
        if task is None:
            return
        try:
            figure = _plot_figure(task)
            output = pathlib.Path(task.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix == ".pdf":
                subprocess.run(
                    ["rsvg-convert", "--format=pdf", "--output", str(output)],
                    input=figure,
                    text=True,
                    check=True,
                )
            else:
                dpi = task.settings["figures"][task.name]["kwargs"].get(
                    "dpi", task.settings["export_dpi"]
                )
                if task.name.startswith("inference_performance_"):
                    dpi = task.settings["export_dpi"]
                figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
                plt.close(figure)
            results.put(FigureResult(task.name))
        except Exception:
            results.put(FigureResult(task.name, traceback.format_exc()))
        finally:
            plt.close("all")


def build_figures(inputs, outputs, settings, threads):
    """Generate all individually declared final figures without upstream inference."""
    illustration = pathlib.Path(outputs["anc_eval_pipe.png"])
    illustration.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inputs["pipeline_illustration"], illustration)
    context = multiprocessing.get_context("spawn")
    tasks = context.Queue()
    results = context.Queue()
    worker_count = min(multiprocessing.cpu_count(), threads, len(settings["figures"]))
    workers = [
        context.Process(target=_figure_worker, args=(tasks, results))
        for _ in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    for name in settings["figures"]:
        tasks.put(FigureTask(name, str(outputs[name]), inputs, settings))
    for _ in workers:
        tasks.put(None)
    failures = []
    for _ in settings["figures"]:
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
