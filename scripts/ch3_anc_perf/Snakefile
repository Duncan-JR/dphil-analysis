"""Run inside an exclusive allocation; every job reserves the whole allocation."""

import pathlib
import shlex

configfile: "config.yaml"

output_dir = config["output_dir"]
config_path = workflow.configfiles[0]
threads_per_job = config["resources"]["threads"]
memory_per_job = config["resources"]["mem_mb"]
time_per_job = config["resources"]["time_min"]

shell.prefix(
    "set -euo pipefail; "
    f"export OMP_NUM_THREADS={config['helper_threads']} "
    f"OPENBLAS_NUM_THREADS={config['helper_threads']} "
    f"MKL_NUM_THREADS={config['helper_threads']} "
    f"NUMEXPR_NUM_THREADS={config['helper_threads']}; "
)


def zarr_path(wildcards):
    num_samples = int(wildcards.num_samples)
    n = num_samples // 2
    pattern = pathlib.Path(config["zarr_pattern"].format(n=n))
    paths = list(pattern.parent.glob(pattern.name))
    if len(paths) != 1:
        raise ValueError(f"Expected one dataset for n={n}, found {len(paths)}: {pattern}")
    return str(paths[0])


def runner(version):
    installation = config["versions"][version]
    return (
        f"PYTHONPATH={shlex.quote(installation)} uv run "
        f"--project {shlex.quote(config['runtime_project'])} --no-sync "
        "python anc_perf.py"
    )


rule all:
    input:
        f"{output_dir}/perf_combined.csv",


rule generate_ancestors:
    input:
        zarr=zarr_path,
    output:
        f"{output_dir}/{{version}}/n{{num_samples}}/ancestors.ancestors",
    log:
        f"{output_dir}/{{version}}/n{{num_samples}}/generate_ancestors.log",
    threads: threads_per_job
    resources:
        mem_mb=memory_per_job,
        time_min=time_per_job,
    params:
        runner=lambda wildcards: runner(wildcards.version),
        config=config_path,
    shell:
        "{params.runner} generate_ancestors --config {params.config:q} "
        "--zarr {input.zarr:q} --output {output:q} --version {wildcards.version} "
        "--num-samples {wildcards.num_samples} --threads {threads} > {log:q} 2>&1"


rule match_ancestors:
    input:
        zarr=zarr_path,
        ancestors=f"{output_dir}/{{version}}/n{{num_samples}}/ancestors.ancestors",
    output:
        f"{output_dir}/{{version}}/n{{num_samples}}/ancestors.trees",
    log:
        f"{output_dir}/{{version}}/n{{num_samples}}/match_ancestors.log",
    threads: threads_per_job
    resources:
        mem_mb=memory_per_job,
        time_min=time_per_job,
    params:
        runner=lambda wildcards: runner(wildcards.version),
        config=config_path,
    shell:
        "{params.runner} match_ancestors --config {params.config:q} "
        "--zarr {input.zarr:q} --input {input.ancestors:q} --output {output:q} "
        "--version {wildcards.version} --num-samples {wildcards.num_samples} "
        "--threads {threads} > {log:q} 2>&1"


rule match_samples:
    input:
        zarr=zarr_path,
        ancestors=f"{output_dir}/{{version}}/n{{num_samples}}/ancestors.trees",
    output:
        f"{output_dir}/{{version}}/n{{num_samples}}/inferred.trees",
    log:
        f"{output_dir}/{{version}}/n{{num_samples}}/match_samples.log",
    threads: threads_per_job
    resources:
        mem_mb=memory_per_job,
        time_min=time_per_job,
    params:
        runner=lambda wildcards: runner(wildcards.version),
        config=config_path,
    shell:
        "{params.runner} match_samples --config {params.config:q} "
        "--zarr {input.zarr:q} --input {input.ancestors:q} --output {output:q} "
        "--version {wildcards.version} --num-samples {wildcards.num_samples} "
        "--threads {threads} > {log:q} 2>&1"


rule extract_performance:
    input:
        f"{output_dir}/{{version}}/n{{num_samples}}/inferred.trees",
    output:
        f"{output_dir}/{{version}}/n{{num_samples}}/perf.csv",
    threads: threads_per_job
    resources:
        mem_mb=memory_per_job,
    params:
        runner=lambda wildcards: runner(wildcards.version),
        config=config_path,
    shell:
        "{params.runner} extract --config {params.config:q} --input {input:q} "
        "--output {output:q} --version {wildcards.version} "
        "--num-samples {wildcards.num_samples} --threads {threads}"


rule aggregate_version:
    input:
        lambda wildcards: expand(
            f"{output_dir}/{{version}}/n{{num_samples}}/perf.csv",
            version=wildcards.version,
            num_samples=config["datasets"],
        ),
    output:
        f"{output_dir}/perf_{{version}}.csv",
    threads: threads_per_job
    resources:
        mem_mb=memory_per_job,
    params:
        runner=lambda wildcards: runner(wildcards.version),
        config=config_path,
    shell:
        "{params.runner} aggregate --config {params.config:q} "
        "--input {input:q} --output {output:q}"


rule aggregate_all:
    input:
        expand(f"{output_dir}/perf_{{version}}.csv", version=config["versions"]),
    output:
        f"{output_dir}/perf_combined.csv",
    threads: threads_per_job
    resources:
        mem_mb=memory_per_job,
    params:
        runner=runner(next(iter(config["versions"]))),
        config=config_path,
    shell:
        "{params.runner} aggregate --config {params.config:q} "
        "--input {input:q} --output {output:q}"
