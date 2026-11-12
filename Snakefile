"""Maintained thesis outputs. Run from the analysis repository root."""
import pathlib

from ch3 import steps, thesis_utils

configfile: "thesis_config.yaml"

CH3 = config["chapter3"]
INPUTS = CH3["inputs"]
THESIS = pathlib.Path(config["thesis_root"])
WORK = pathlib.Path(config["work_root"])
SOURCES = ["src/ch3/thesis_utils.py", "src/ch3/thesis_plot.py", "src/ch3/steps.py", "pyproject.toml", "uv.lock"]
PREPARATION_SOURCES = ["src/ch3/thesis_utils.py", "pyproject.toml", "uv.lock"]
FIGURE_NAMES = [
    "ancestor_extension_trees.pdf", "ancestor_extension_comparison.png", "anc_eval_pipe.png",
    "demography_overshoot_box.png", "demography_overlap_errors.png",
    "ancestor_span_frequency_simulation.png", "ancestor_span_frequency_tgp.png",
    "ancestor_overshoot_boxplot.png", "ancestor_overlap_errors.png",
    "genotype_error_scaling.png", "genotype_error_overshoot_summary.png", "genotype_error_overshoot_diff.png",
    "mispolarisation_daf.png", "mispolarisation_overshoot_correct_focal_diff.png",
    "mispolarisation_overshoot_flipped_focal_hist.png", "mispolarisation_overlap_errors.png",
    "phase_switch_illustration.png", "phasing_error_overshoot_summary.png", "phasing_error_overshoot_diff.png",
    "combined_errors.png", "inference_performance_speedup.png", "inference_performance_elapsed.png",
    "inference_performance_parallelism.png",
]
if set(CH3["figures"]) != set(FIGURE_NAMES) - {"anc_eval_pipe.png"}:
    raise ValueError("Chapter 3 figure configuration must match the individually declared thesis outputs")
FIGURES = [str(THESIS / "figures/ch3" / name) for name in FIGURE_NAMES]
STATISTICS = [str(THESIS / "statistics" / name) for name in ("ch3.tex", "ch3_claims.json", "ch3_checks.json")]
COMPARISONS = str(WORK / "comparisons.parquet")
EMPIRICAL_SPANS = str(WORK / "empirical_spans.parquet")
EMPIRICAL_DAF = str(WORK / "empirical_daf.parquet")
DEMOGRAPHIES = {"OutOfAfrica_3G09": COMPARISONS}
DEMOGRAPHIES.update({model: str(WORK / f"{model}.parquet") for model in INPUTS["demographies"]})

# Enumerate every metadata/chunk file in the consumed arrays at DAG construction.
# Other arrays (including enormous call_genotype) are not read by the exporter.
STORE = pathlib.Path(INPUTS["empirical_store"])
COUNT_PREFIX = CH3["analysis"]["tgp_count_prefix"]
ARRAY_NAMES = ["variant_position", f"{COUNT_PREFIX}_derived_count", f"{COUNT_PREFIX}_ancestral_count"]
STORE_FILES = [str(path) for path in STORE.iterdir() if path.is_file() and path.name in {"zarr.json", ".zgroup", ".zattrs", ".zmetadata"}] if STORE.exists() else []
for array in ARRAY_NAMES:
    directory = STORE / array
    if not directory.exists():
        raise FileNotFoundError(f"Required chapter 3 DAF array is missing: {directory}. Restore the input identified in plans/thesis_inputs.json.")
    STORE_FILES.extend(str(path) for path in sorted(directory.rglob("*")) if path.is_file())


def prepared_inputs():
    return {
        "comparisons": COMPARISONS, "demographies": DEMOGRAPHIES,
        "empirical_spans": EMPIRICAL_SPANS, "empirical_daf": EMPIRICAL_DAF,
        "validation": INPUTS["validation"], "phase_runs": INPUTS["phase_runs"],
        "performance": INPUTS["performance"], "pipeline_illustration": INPUTS["pipeline_illustration"],
    }


rule all:
    input: FIGURES + STATISTICS


rule ch3_prepare_comparisons:
    input:
        aggregate=INPUTS["aggregate"], validation=INPUTS["validation"], source=PREPARATION_SOURCES
    output: COMPARISONS
    params: analysis=CH3["analysis"]
    run:
        thesis_utils.prepare_comparisons(input.aggregate, input.validation, output[0], thesis_utils.AnalysisConfig(**params.analysis))


rule ch3_prepare_demography:
    input:
        aggregate=lambda wildcards: INPUTS["demographies"][wildcards.model], source=PREPARATION_SOURCES
    output: str(WORK / "{model}.parquet")
    wildcard_constraints: model="|".join(INPUTS["demographies"])
    params: analysis=CH3["analysis"]
    run:
        thesis_utils.prepare_demography(input.aggregate, output[0], thesis_utils.AnalysisConfig(**params.analysis))


rule ch3_prepare_empirical_spans:
    input: spans=INPUTS["empirical_spans"], source=PREPARATION_SOURCES
    output: EMPIRICAL_SPANS
    params: analysis=CH3["analysis"]
    run:
        thesis_utils.prepare_empirical_spans(input.spans, output[0], thesis_utils.AnalysisConfig(**params.analysis))


rule ch3_prepare_empirical_daf:
    input: store=str(STORE), arrays=STORE_FILES, source=PREPARATION_SOURCES
    output: EMPIRICAL_DAF
    params: analysis=CH3["analysis"]
    run:
        thesis_utils.export_empirical_daf(input.store, output[0], thesis_utils.AnalysisConfig(**params.analysis))


rule ch3_plots:
    input:
        comparisons=COMPARISONS, demographies=list(DEMOGRAPHIES.values()),
        empirical_spans=EMPIRICAL_SPANS, empirical_daf=EMPIRICAL_DAF,
        validation=INPUTS["validation"], phase_runs=INPUTS["phase_runs"],
        performance=INPUTS["performance"], illustration=INPUTS["pipeline_illustration"], source=SOURCES
    output: FIGURES
    threads: workflow.cores
    params: settings=CH3
    run:
        outputs = dict(zip(FIGURE_NAMES, output, strict=True))
        steps.build_figures(prepared_inputs(), outputs, params.settings, threads)


rule ch3_statistics:
    input:
        comparisons=COMPARISONS, demographies=list(DEMOGRAPHIES.values()),
        empirical_spans=EMPIRICAL_SPANS, empirical_daf=EMPIRICAL_DAF,
        validation=INPUTS["validation"], phase_runs=INPUTS["phase_runs"],
        performance=INPUTS["performance"], source=PREPARATION_SOURCES
    output: STATISTICS
    params: settings=CH3
    run:
        thesis_utils.write_statistics(prepared_inputs(), params.settings, output[0], output[1], output[2])
