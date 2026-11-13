"""Maintained thesis outputs. Run from the analysis repository root."""

import pathlib

from ch3 import steps, thesis_utils

configfile: "thesis_config.yaml"

thesis_root = pathlib.Path(config["thesis_root"])
work_root = pathlib.Path(config["work_root"])
statistics_dir = pathlib.Path(config["statistics_dir"])


rule all:
    input:
        expand(thesis_root / "figures/ch3/{figure}", figure=config["chapter3"]["figures"].values()),
        statistics_dir / "ch3.tex",
        statistics_dir / "ch3_claims.json",
        statistics_dir / "ch3_checks.json",


rule ch3_prepare_comparisons:
    input:
        aggregate="data/ch3/OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1-aggregated.csv",
        validation="data/ch3/OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1_zarr_data.csv",
        source=[
            "src/ch3/thesis_utils.py", "pyproject.toml", "uv.lock",
            "data/general/dependencies/tsls-0.1.dev121.tar.gz",
        ],
    output:
        work_root / "comparisons.parquet",
    params:
        analysis=config["chapter3"]["analysis"],
    run:
        thesis_utils.prepare_comparisons(
            input.aggregate, input.validation, output[0],
            thesis_utils.AnalysisConfig(**params.analysis),
        )


rule ch3_prepare_demography:
    input:
        aggregate="data/ch3/demographies/{model}-chr20-L0-R2e7-n6000-s1-aggregated.csv",
        source=[
            "src/ch3/thesis_utils.py", "pyproject.toml", "uv.lock",
            "data/general/dependencies/tsls-0.1.dev121.tar.gz",
        ],
    output:
        work_root / "{model}.parquet",
    wildcard_constraints:
        model="|".join(config["chapter3"]["demographies"]),
    params:
        analysis=config["chapter3"]["analysis"],
    run:
        thesis_utils.prepare_demography(
            input.aggregate, output[0], thesis_utils.AnalysisConfig(**params.analysis),
        )


rule ch3_prepare_empirical_spans:
    input:
        spans=["data/ch3/tgp_ancestor_spans_0.4.1.csv", "data/ch3/tgp_ancestor_spans.csv"],
        source=[
            "src/ch3/thesis_utils.py", "pyproject.toml", "uv.lock",
            "data/general/dependencies/tsls-0.1.dev121.tar.gz",
        ],
    output:
        work_root / "empirical_spans.parquet",
    params:
        analysis=config["chapter3"]["analysis"],
    run:
        thesis_utils.prepare_empirical_spans(
            input.spans, output[0], thesis_utils.AnalysisConfig(**params.analysis),
        )


rule ch3_prepare_empirical_daf:
    input:
        store="data/general/zarr_vcfs/tgp/chr20/data.zarr",
        arrays=lambda wildcards: thesis_utils.empirical_daf_inputs(
            "data/general/zarr_vcfs/tgp/chr20/data.zarr",
            config["chapter3"]["analysis"]["tgp_count_prefix"],
        ),
        source=[
            "src/ch3/thesis_utils.py", "pyproject.toml", "uv.lock",
            "data/general/dependencies/tsls-0.1.dev121.tar.gz",
        ],
    output:
        work_root / "empirical_daf.parquet",
    params:
        analysis=config["chapter3"]["analysis"],
    run:
        thesis_utils.export_empirical_daf(
            input.store, output[0], thesis_utils.AnalysisConfig(**params.analysis),
        )


rule ch3_plots:
    input:
        comparisons=work_root / "comparisons.parquet",
        demographies=expand(work_root / "{model}.parquet", model=config["chapter3"]["demographies"]),
        empirical_spans=work_root / "empirical_spans.parquet",
        empirical_daf=work_root / "empirical_daf.parquet",
        validation="data/ch3/OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1_zarr_data.csv",
        phase_runs="data/ch3/OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1_phase_runs.csv",
        performance="data/ch3/performance/perf_combined.csv",
        pipeline_illustration="data/ch3/thesis_assets/anc_eval_pipe.png",
        source=[
            "src/ch3/thesis_utils.py", "src/ch3/thesis_plot.py", "src/ch3/steps.py",
            "pyproject.toml", "uv.lock", "data/general/dependencies/tsls-0.1.dev121.tar.gz",
        ],
    output:
        expand(thesis_root / "figures/ch3/{figure}", figure=config["chapter3"]["figures"].values()),
    threads: workflow.cores
    params:
        settings=config["chapter3"],
    run:
        paths = dict(input.items())
        paths["demographies"] = dict(zip(params.settings["demographies"], input.demographies, strict=True))
        paths["demographies"]["OutOfAfrica_3G09"] = input.comparisons
        steps.build_figures(paths, output, params.settings, threads)


rule ch3_statistics:
    input:
        comparisons=work_root / "comparisons.parquet",
        demographies=expand(work_root / "{model}.parquet", model=config["chapter3"]["demographies"]),
        empirical_spans=work_root / "empirical_spans.parquet",
        empirical_daf=work_root / "empirical_daf.parquet",
        validation="data/ch3/OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1_zarr_data.csv",
        phase_runs="data/ch3/OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1_phase_runs.csv",
        performance="data/ch3/performance/perf_combined.csv",
        source=[
            "src/ch3/thesis_utils.py", "pyproject.toml", "uv.lock",
            "data/general/dependencies/tsls-0.1.dev121.tar.gz",
        ],
    output:
        tex=statistics_dir / "ch3.tex",
        claims=statistics_dir / "ch3_claims.json",
        checks=statistics_dir / "ch3_checks.json",
    params:
        settings=config["chapter3"],
    run:
        paths = dict(input.items())
        paths["demographies"] = dict(zip(params.settings["demographies"], input.demographies, strict=True))
        paths["demographies"]["OutOfAfrica_3G09"] = input.comparisons
        thesis_utils.write_statistics(paths, params.settings, output.tex, output.claims, output.checks)
