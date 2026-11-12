"""Chapter 5 exploratory preparation and analysis.

Scientific notebook settings live in notebook cells. This is rough work.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import csv
import dataclasses
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import logging
import math
import multiprocessing
import os
import pathlib
import subprocess
import time
import tomllib

import numcodecs
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
import tscompare
import tscompare.methods
import tsdate
import tsinfer
import tskit
import tszip
import yaml
import zarr

logger = logging.getLogger(__name__)


def paramsearch_load_config(path: dict) -> dict:
    """Read the analysis config and archived effective experiment/manifest."""
    config = path.copy()
    root = resolve_path(config["data_dir"], config["base_dir"])
    config["data_dir"] = root
    config["cache_dir"] = root / config["cache_subdir"]
    config["cache_dir"].mkdir(parents=True, exist_ok=True)
    config["manifest"] = json.loads((root / "manifest.json").read_text())
    config["experiment"] = yaml.safe_load((root / "config.yaml").read_text())
    return config


def input_signature(paths: list[pathlib.Path], settings: dict) -> str:
    """Invalidate plot caches on changed inputs, analysis code or dependencies."""
    stamps = [
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    ]
    versions = {
        name: importlib.metadata.version(name)
        for name in ["tsinfer", "tsdate", "tskit", "tscompare", "tszip"]
    }
    digest = hashlib.sha256()
    digest.update(pathlib.Path(__file__).read_bytes())
    code = digest.hexdigest()
    payload = json.dumps([stamps, settings, versions, code], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def cached_table(path: pathlib.Path, signature: str) -> pd.DataFrame | None:
    """Read a completed table only when its signature matches."""
    stamp = path.with_suffix(".json")
    if path.exists() and stamp.exists() and (stamp.read_text() == signature):
        return pd.read_csv(path)
    return None


def save_table(table: pd.DataFrame, path: pathlib.Path, signature: str):
    """Write the signature last, so interrupted work is never treated as complete."""
    table.to_csv(path, index=False)
    path.with_suffix(".json").write_text(signature)


def paramsearch_length_counts(config: dict) -> pd.DataFrame:
    """Stream 4 GB of region rows into lossless counts at each integer length."""
    path = config["data_dir"] / config["manifest"]["tables"]["region_lengths"]
    output = config["cache_dir"] / "paramsearch_length_counts.csv"
    signature = input_signature([path], {})
    cached = cached_table(output, signature)
    if cached is not None:
        return cached
    keys = ["dataset", "af_cutoff", "min_focal_region_size", "is_gap", "num_sites"]
    query = pl.scan_csv(path).select(keys)
    counts = query.group_by(keys).len(name="count").sort(keys)
    collected = counts.collect(engine="streaming")
    columns = {name: collected[name].to_numpy() for name in collected.columns}
    table = pd.DataFrame(columns)
    save_table(table, output, signature)
    return table


def population_summary(config: dict) -> pd.DataFrame:
    """Cache pt6 African/non-African coverage means from exported haplotypes."""
    path = config["data_dir"] / config["manifest"]["tables"]["haplotype_results"]
    output = config["cache_dir"] / "population_coverage.csv"
    signature = input_signature([path], {})
    cached = cached_table(output, signature)
    if cached is not None:
        return cached
    columns = [
        "dataset",
        "af_cutoff",
        "min_focal_region_size",
        "population",
        "focal_coverage_percentage",
    ]
    table = pd.read_csv(path, usecols=columns)
    table["population_group"] = np.where(
        table.population.eq("YRI"), "African", "Non-African"
    )
    keys = ["dataset", "af_cutoff", "min_focal_region_size", "population_group"]
    result = (
        table.groupby(keys)
        .focal_coverage_percentage.agg(["mean", "sem", "count"])
        .reset_index()
    )
    save_table(result, output, signature)
    return result


def paramsearch_weighted_quantile(table: pd.DataFrame, quantile: float) -> float:
    """Match NumPy's linearly interpolated quantile without expanding counts."""
    counts = table.groupby("num_sites")["count"].sum().sort_index()
    cumulative = counts.cumsum().to_numpy()
    rank = (int(cumulative[-1]) - 1) * quantile
    lower = np.searchsorted(cumulative, np.floor(rank), side="right")
    upper = np.searchsorted(cumulative, np.ceil(rank), side="right")
    values = counts.index.to_numpy()
    return float(
        values[lower] + (rank - np.floor(rank)) * (values[upper] - values[lower])
    )


def paramsearch_length_density(table: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Evaluate pt6's reflected KDE exactly from length multiplicities.

    Correct the weighted covariance/bandwidth to reproduce Scott's bandwidth
    for the original uncompressed interval observations, rather than treating
    unique lengths as independent observations.
    """
    values = table.num_sites.to_numpy(dtype=float)
    counts = table["count"].to_numpy(dtype=float)
    total = counts.sum()
    mean = np.average(values, weights=counts)
    variance = np.sum(counts * (values - mean) ** 2) / (total - 1)
    bandwidth = np.sqrt(variance) * total ** (-1 / 5)
    weighted_variance = np.cov(values, aweights=counts).item()
    kde = stats.gaussian_kde(
        values, weights=counts, bw_method=bandwidth / np.sqrt(weighted_variance)
    )
    return kde(grid) + kde(-grid)


@dataclasses.dataclass
class TreeJob:
    """One archived matched TS; Full HMM has no AF/minimum parameters."""

    root: pathlib.Path
    cache: pathlib.Path
    dataset: dict
    run: dict | None
    config: dict

    @property
    def raw_path(self) -> pathlib.Path:
        relative = (
            self.dataset["full_hmm"] if self.run is None else self.run["stitched"]
        )
        return self.root / relative

    @property
    def key(self) -> str:
        suffix = "full_hmm"
        if self.run is not None:
            suffix = (
                f"af_{self.run['af_cutoff']}_min_{self.run['min_focal_region_size']}"
            )
        return f"{self.dataset['name']}_{suffix}"

    @property
    def labels(self) -> dict:
        return {
            "dataset": self.dataset["name"],
            "comparison_method": "full_hmm" if self.run is None else "stitched_greedy",
            "af_cutoff": None if self.run is None else self.run["af_cutoff"],
            "min_focal_region_size": None
            if self.run is None
            else self.run["min_focal_region_size"],
        }


def tree_jobs(config: dict) -> list[TreeJob]:
    """Derive all paths from the manifest, avoiding stale HPC absolute paths."""
    jobs = []
    settings = {
        key: config[key] for key in ["mutation_rate", "dating", "helper_threads"]
    }
    for dataset in config["manifest"]["datasets"]:
        for run in [None, *dataset["runs"]]:
            jobs.append(
                TreeJob(config["data_dir"], config["cache_dir"], dataset, run, settings)
            )
    return jobs


def compare_tree(job: TreeJob) -> dict:
    """Cache raw counts and aligned ARF/TPR against truth and Full HMM."""
    truth_path = job.root / job.dataset["truth"]
    full_path = job.root / job.dataset["full_hmm"]
    signature = input_signature([job.raw_path, truth_path, full_path], {})
    output = job.cache / "tree_metrics" / f"{job.key}.csv"
    cached = cached_table(output, signature)
    if cached is not None:
        return cached.iloc[0].to_dict()
    comparison = tszip.decompress(job.raw_path)
    row = {
        **job.labels,
        "num_edges": comparison.num_edges,
        "num_trees": comparison.num_trees,
    }
    span = job.dataset["analysis_extent_bp"]
    truth = tszip.decompress(truth_path)
    mapping = truth_sample_nodes(truth)
    aligned = align_tree_sequences(comparison, truth, span, mapping)
    metric = haplotype_arf(aligned.comparison, aligned.reference)
    row.update(truth_arf=float(metric.arf), truth_tpr=float(metric.tpr))
    del truth, aligned
    if job.run is not None:
        full = tszip.decompress(full_path)
        row["edges_change_percentage"] = 100 * (
            comparison.num_edges / full.num_edges - 1
        )
        row["trees_change_percentage"] = 100 * (
            comparison.num_trees / full.num_trees - 1
        )
        aligned = align_tree_sequences(comparison, full, span)
        metric = haplotype_arf(aligned.comparison, aligned.reference)
        row.update(full_arf=float(metric.arf), full_tpr=float(metric.tpr))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_table(pd.DataFrame([row]), output, signature)
    return row


def paramsearch_date_tree(job: TreeJob) -> dict:
    """Date the saved sites only; no genotype stores or variant restoration."""
    signature = input_signature([job.raw_path], job.config)
    output = job.cache / "dated_args" / f"{job.key}.csv"
    tree_path = output.with_suffix(".trees")
    cached = cached_table(output, signature)
    if cached is not None and tree_path.exists():
        return cached.iloc[0].to_dict()
    raw = tszip.decompress(job.raw_path)
    settings = job.config["dating"]
    processed = tsinfer.post_process(raw, **settings["post_process"])
    window = window_tree_sequence(processed, job.dataset["analysis_extent_bp"])
    prepared = tsdate.preprocess_ts(window, **settings["preprocess"])
    dated = tsdate.date(
        prepared, mutation_rate=job.config["mutation_rate"], **settings["estimation"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    dated.dump(tree_path)
    row = {
        **job.labels,
        "dated_path": str(tree_path),
        "raw_sites": raw.num_sites,
        "dated_sites": dated.num_sites,
        "dated_mutations": dated.num_mutations,
    }
    save_table(pd.DataFrame([row]), output, signature)
    return row


def run_tree_jobs(config: dict, operation: str) -> pd.DataFrame:
    """Distribute independent comparisons or dates across the available CPUs."""
    jobs = tree_jobs(config)
    workers = config["workers"]
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    function = {"metrics": compare_tree, "dating": paramsearch_date_tree}[operation]
    context = multiprocessing.get_context("spawn")
    rows = []
    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=numcodecs.blosc.set_nthreads,
        initargs=(config["helper_threads"],),
    ) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        for future in cf.as_completed(futures):
            rows.append(future.result())
            logger.info(
                "%s: completed %s (%d/%d)",
                operation,
                futures[future].key,
                len(rows),
                len(jobs),
            )
    result = pd.DataFrame(rows).sort_values(
        ["dataset", "af_cutoff", "min_focal_region_size"]
    )
    result.to_csv(config["cache_dir"] / f"{operation}_summary.csv", index=False)
    return result


def shared_mutation_correlations(
    runs: pd.DataFrame, truth: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    """Use one intersection across truth and every dated run for both references.

    Reuse pt6's posterior mutation means and exact allele/position pairing.
    Recurrent sites and nonfinite ages are excluded before intersection. DAF
    comes from Full HMM and is fixed across every comparison in the sample size.
    """
    ages = []
    for run in runs.itertuples():
        ts = tskit.load(run.dated_path)
        values = mutation_ages(
            ts, posterior=True, include_frequency=run.comparison_method == "full_hmm"
        )
        ages.append(values)
    common = truth.index
    for values in ages:
        common = common.intersection(values.index, sort=False)
    common = common.sort_values()
    full_index = runs.index[runs.comparison_method.eq("full_hmm")].item()
    full = ages[full_index]
    frequencies = full.derived_frequency.loc[common]
    legacy_runs = runs.rename(columns={"af_cutoff": "ac_cutoff"}).copy()
    legacy_runs["max_mismatches"] = None
    rows = []
    for reference_method, reference in [
        ("true_arg", truth.age),
        ("full_hmm", full.age),
    ]:
        reference = reference.loc[common]
        rows.extend(
            correlation_rows(
                legacy_runs, ages, reference, frequencies, reference_method, settings
            )
        )
    return (
        pd.DataFrame(rows)
        .rename(columns={"ac_cutoff": "af_cutoff"})
        .drop(columns="max_mismatches")
    )


def paramsearch_mutation_correlations(
    config: dict, dated: pd.DataFrame
) -> pd.DataFrame:
    """Cache correlations using the same common mutation set for both references."""
    paths = [pathlib.Path(path) for path in dated.dated_path]
    paths.extend(
        (config["data_dir"] / item["truth"] for item in config["manifest"]["datasets"])
    )
    signature = input_signature(paths, config["frequency_bins"])
    output = config["cache_dir"] / "mutation_age_correlations.csv"
    cached = cached_table(output, signature)
    if cached is not None:
        return cached
    results = []
    for dataset in config["manifest"]["datasets"]:
        runs = dated.loc[dated.dataset.eq(dataset["name"])].reset_index(drop=True)
        truth = tszip.decompress(config["data_dir"] / dataset["truth"])
        truth = window_tree_sequence(truth, dataset["analysis_extent_bp"])
        ages = mutation_ages(truth, posterior=False)
        result = shared_mutation_correlations(runs, ages, config["frequency_bins"])
        results.append(result)
    table = pd.concat(results, ignore_index=True)
    save_table(table, output, signature)
    return table


def signature(paths: list[pathlib.Path], settings: dict, operation) -> str:
    """Invalidate a cache when its inputs, operation, settings or versions change."""
    stamps = [
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    ]
    versions = {
        name: importlib.metadata.version(name)
        for name in ["tsinfer", "tsdate", "tskit", "tszip"]
    }
    code = inspect.getsource(operation)
    payload = json.dumps([stamps, settings, versions, code], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_cached(path: pathlib.Path, stamp: str) -> pd.DataFrame | None:
    metadata = path.with_suffix(".json")
    if path.exists() and metadata.exists() and (metadata.read_text() == stamp):
        return pd.read_csv(path)
    return None


def save_cached(table: pd.DataFrame, path: pathlib.Path, stamp: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    path.with_suffix(".json").write_text(stamp)


def cohort_table(config: dict) -> pd.DataFrame:
    """Read sample counts from actual archived haplotype identities, not names."""
    rows = []
    for dataset in config["manifest"]["datasets"]:
        cohort = pd.DataFrame(dataset["cohort"])
        assert not cohort.duplicated(["source", "sample_id", "ploidy_index"]).any()
        individuals = cohort[["source", "sample_id"]].drop_duplicates()
        rows.append(
            {
                "dataset": dataset["name"],
                "num_samples": len(individuals),
                "num_haplotypes": len(cohort),
                "num_sites": dataset["num_sites"],
                "analysis_span_mb": dataset["analysis_span_bp"] / 1000000.0,
                "has_truth": dataset["truth"] is not None,
                "min_region_size_kb": dataset["runs"][0]["min_region_size_kb"],
                "min_focal_region_size": dataset["runs"][0]["min_focal_region_size"],
            }
        )
    return pd.DataFrame(rows).sort_values("num_samples").reset_index(drop=True)


def performance_runs(config: dict) -> pd.DataFrame:
    """Preserve one Full HMM baseline and greedy trials per sample size."""
    name = config["experiment"]["performance"]["experiment_id"]
    root = config["data_dir"] / "performance" / name
    runs = pd.read_csv(root / "runs.csv")
    quality = pd.DataFrame(json.loads((root / "quality.json").read_text()))
    assert runs.status.eq("ok").all()
    assert quality.status.eq("ok").all()
    assert quality.called_allele_errors.eq(0).all()
    assert set(quality.run_id) == set(runs.run_id)
    assert runs["mode"].eq("from_reference").all()
    assert runs.actual_workers.eq(
        config["experiment"]["performance"]["worker_counts"][0]
    ).all()
    assert not runs.duplicated(["dataset", "method", "af_cutoff", "repeat"]).any()
    cohorts = cohort_table(config)
    runs = runs.merge(
        cohorts[["dataset", "num_samples", "num_sites"]],
        on="dataset",
        validate="many_to_one",
    )
    assert runs.matched_haplotypes.eq(2 * runs.num_samples).all()
    runs["cpu_s"] = runs.user_s + runs.system_s
    runs["effective_cores"] = runs.cpu_s / runs.wall_s
    runs["wall_ms_per_haplotype"] = 1000 * runs.wall_s / runs.matched_haplotypes
    runs["trial_peak_private_gib"] = runs.trial_peak_private_bytes / 1024**3
    runs["window_peak_private_gib"] = runs.window_peak_private_bytes / 1024**3
    baseline = runs.loc[
        runs.method.eq("full_hmm"), ["dataset", "repeat", "wall_s", "cpu_s"]
    ]
    baseline = baseline.rename(columns={"wall_s": "full_wall_s", "cpu_s": "full_cpu_s"})
    runs = runs.merge(baseline, on=["dataset", "repeat"], validate="many_to_one")
    runs["speedup"] = runs.full_wall_s / runs.wall_s
    runs["cpu_speedup"] = runs.full_cpu_s / runs.cpu_s
    runs["focal_site_coverage_percentage"] = 100 * runs.focal_sites / runs.domain_sites
    runs["gap_calls_per_haplotype"] = runs.gap_calls / runs.matched_haplotypes
    runs.to_csv(config["cache_dir"] / "performance_summary.csv", index=False)
    return runs.sort_values(["num_samples", "method", "af_cutoff"])


def diagnostic_summary(config: dict) -> pd.DataFrame:
    path = config["data_dir"] / config["manifest"]["tables"]["run_summary"]
    table = pd.read_csv(path)
    cohorts = cohort_table(config)
    assert table.unmatched_sites_mean.eq(0).all()
    expected_rows = []
    for dataset in config["manifest"]["datasets"]:
        for run in dataset["runs"]:
            expected_rows.append(
                {
                    "dataset": dataset["name"],
                    "af_cutoff": run["af_cutoff"],
                    "min_region_size_kb": run["min_region_size_kb"],
                    "min_focal_region_size": run["min_focal_region_size"],
                }
            )
    expected = pd.DataFrame(expected_rows)
    check_columns = [
        "dataset",
        "af_cutoff",
        "min_region_size_kb",
        "min_focal_region_size",
    ]
    checked = table[check_columns].merge(
        expected,
        on=["dataset", "af_cutoff"],
        how="outer",
        suffixes=("", "_expected"),
        validate="one_to_one",
        indicator=True,
    )
    assert checked["_merge"].eq("both").all()
    assert checked.min_region_size_kb.eq(checked.min_region_size_kb_expected).all()
    assert checked.min_focal_region_size.eq(
        checked.min_focal_region_size_expected
    ).all()
    assert not table.duplicated(["dataset", "af_cutoff"]).any()
    table = table.merge(
        cohorts[["dataset", "num_samples"]].rename(
            columns={"num_samples": "actual_samples"}
        ),
        on="dataset",
        validate="many_to_one",
    )
    assert table.num_samples.eq(table.actual_samples).all()
    table.to_csv(config["cache_dir"] / "diagnostic_summary.csv", index=False)
    return table.sort_values(["num_samples", "af_cutoff"])


def population_coverage(config: dict) -> pd.DataFrame:
    """Aggregate physical coverage with the configured population groups."""
    path = config["data_dir"] / config["manifest"]["tables"]["haplotype_results"]
    settings = config["population_groups"]
    stamp = signature([path], settings, population_coverage)
    output = config["cache_dir"] / "population_coverage.csv"
    cached = read_cached(output, stamp)
    if cached is not None:
        return cached
    columns = ["dataset", "af_cutoff", "population", "focal_coverage_percentage"]
    table = pd.read_csv(path, usecols=columns)
    mapping = {
        population: group
        for group, members in settings.items()
        for population in members
    }
    assert set(table.population) <= set(mapping)
    table["population_group"] = table.population.map(mapping)
    result = (
        table.groupby(["dataset", "af_cutoff", "population_group"])
        .focal_coverage_percentage.agg(["mean", "sem", "count"])
        .reset_index()
    )
    sizes = cohort_table(config)[["dataset", "num_samples"]]
    result = result.merge(sizes, on="dataset", validate="many_to_one")
    save_cached(result, output, stamp)
    return result


def perf_scaling_length_counts(config: dict) -> pd.DataFrame:
    """Stream region rows into lossless site-count and physical-length counts."""
    source = config["data_dir"] / config["manifest"]["tables"]["region_lengths"]
    stamp = signature([source], {}, perf_scaling_length_counts)
    output = config["cache_dir"] / "region_length_counts.csv"
    cached = read_cached(output, stamp)
    if cached is not None:
        return cached
    keys = ["dataset", "af_cutoff", "is_gap"]
    query = pl.scan_csv(source)
    tables = []
    for column, unit, divisor in [("num_sites", "sites", 1), ("span_bp", "kb", 1000)]:
        counts = query.group_by([*keys, column]).len(name="count")
        collected = counts.collect(engine="streaming")
        table = pd.DataFrame(
            {name: collected[name].to_numpy() for name in collected.columns}
        )
        table = table.rename(columns={column: "length"})
        table["length"] /= divisor
        table["unit"] = unit
        tables.append(table)
    result = pd.concat(tables, ignore_index=True).sort_values([*keys, "unit", "length"])
    save_cached(result, output, stamp)
    return result


def perf_scaling_weighted_quantile(table: pd.DataFrame, quantile: float) -> float:
    counts = table.groupby("length")["count"].sum().sort_index()
    cumulative = counts.cumsum().to_numpy()
    rank = (int(cumulative[-1]) - 1) * quantile
    lower = np.searchsorted(cumulative, np.floor(rank), side="right")
    upper = np.searchsorted(cumulative, np.ceil(rank), side="right")
    values = counts.index.to_numpy()
    return float(
        values[lower] + (rank - np.floor(rank)) * (values[upper] - values[lower])
    )


def perf_scaling_length_density(table: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Reflected Scott KDE, weighted to equal the uncompressed observations."""
    values = table.length.to_numpy(dtype=float)
    counts = table["count"].to_numpy(dtype=float)
    total = counts.sum()
    mean = np.average(values, weights=counts)
    variance = np.sum(counts * (values - mean) ** 2) / (total - 1)
    bandwidth = np.sqrt(variance) * total ** (-1 / 5)
    weighted_variance = np.cov(values, aweights=counts).item()
    kde = stats.gaussian_kde(
        values, weights=counts, bw_method=bandwidth / np.sqrt(weighted_variance)
    )
    return kde(grid) + kde(-grid)


@dataclasses.dataclass
class DatingJob:
    root: pathlib.Path
    cache: pathlib.Path
    dataset: dict
    run: dict | None
    benchmark_tree: pathlib.Path
    settings: dict

    @property
    def raw_path(self) -> pathlib.Path:
        relative = (
            self.dataset["full_hmm"] if self.run is None else self.run["stitched"]
        )
        return self.root / relative

    @property
    def key(self) -> str:
        suffix = (
            "full_hmm"
            if self.run is None
            else f"af_{self.run['af_cutoff']}_kb_{self.run['min_region_size_kb']}"
        )
        return f"{self.dataset['name']}_{suffix}"

    @property
    def labels(self) -> dict:
        return {
            "dataset": self.dataset["name"],
            "method": "full_hmm" if self.run is None else "greedy",
            "af_cutoff": None if self.run is None else self.run["af_cutoff"],
            "min_region_size_kb": None
            if self.run is None
            else self.run["min_region_size_kb"],
            "min_focal_region_size": None
            if self.run is None
            else self.run["min_focal_region_size"],
        }


def perf_scaling_path_fingerprint(ts: tskit.TreeSequence) -> str:
    """Compare saved copying paths and mutations, excluding file provenance."""
    digest = hashlib.sha256()
    for values in [
        ts.sites_position,
        ts.edges_left,
        ts.edges_right,
        ts.edges_parent,
        ts.edges_child,
        ts.nodes_time,
        ts.nodes_flags,
        ts.mutations_site,
        ts.mutations_node,
        ts.tables.mutations.derived_state,
        ts.tables.sites.ancestral_state,
    ]:
        digest.update(memoryview(values))
    return digest.hexdigest()


def perf_scaling_date_tree(job: DatingJob) -> dict:
    """Check the timed/diagnostic outputs agree, then date only their saved sites."""
    stamp = signature(
        [job.raw_path, job.benchmark_tree], job.settings, perf_scaling_date_tree
    )
    output = job.cache / "dated_args" / f"{job.key}.csv"
    tree_path = output.with_suffix(".trees")
    cached = read_cached(output, stamp)
    if cached is not None and tree_path.exists():
        return cached.iloc[0].to_dict()
    raw = tszip.decompress(job.raw_path)
    fingerprint = perf_scaling_path_fingerprint(raw)
    timed = tszip.decompress(job.benchmark_tree)
    assert fingerprint == perf_scaling_path_fingerprint(timed), job.key
    del timed
    counts = {
        "num_edges": raw.num_edges,
        "num_trees": raw.num_trees,
        "raw_sites": raw.num_sites,
        "num_haplotypes": raw.num_samples,
        "benchmark_paths_identical": True,
    }
    settings = job.settings["dating"]
    processed = tsinfer.post_process(raw, **settings["post_process"])
    window = window_tree_sequence(processed, job.dataset["analysis_extent_bp"])
    prepared = tsdate.preprocess_ts(window, **settings["preprocess"])
    dated = tsdate.date(
        prepared, mutation_rate=job.settings["mutation_rate"], **settings["estimation"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    dated.dump(tree_path)
    row = {
        **job.labels,
        **counts,
        "dated_path": str(tree_path),
        "dated_sites": dated.num_sites,
        "dated_mutations": dated.num_mutations,
    }
    save_cached(pd.DataFrame([row]), output, stamp)
    return row


def perf_scaling_date_all(config: dict) -> pd.DataFrame:
    """Distribute independent saved tree sequences across CPU worker processes."""
    experiment = config["experiment"]["performance"]["experiment_id"]
    performance_root = config["data_dir"] / "performance" / experiment
    runs = pd.read_csv(performance_root / "runs.csv")
    jobs = []
    settings = {key: config[key] for key in ["mutation_rate", "dating"]}
    for dataset in config["manifest"]["datasets"]:
        for run in [None, *dataset["runs"]]:
            selected = runs.dataset.eq(dataset["name"])
            if run is None:
                selected &= runs.method.eq("full_hmm")
            else:
                selected &= runs.method.eq("greedy") & runs.af_cutoff.eq(
                    run["af_cutoff"]
                )
            trial = runs.loc[selected].iloc[0]
            benchmark_tree = performance_root / trial.run_id / "samples_raw.trees.tsz"
            jobs.append(
                DatingJob(
                    config["data_dir"],
                    config["cache_dir"],
                    dataset,
                    run,
                    benchmark_tree,
                    settings,
                )
            )
    workers = config["workers"]
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    rows = []
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=numcodecs.blosc.set_nthreads,
        initargs=(config["helper_threads"],),
    ) as pool:
        futures = {pool.submit(perf_scaling_date_tree, job): job for job in jobs}
        for future in cf.as_completed(futures):
            rows.append(future.result())
            logger.info("Dated %s (%d/%d)", futures[future].key, len(rows), len(jobs))
    table = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "method", "af_cutoff"])
        .reset_index(drop=True)
    )
    table = table.merge(
        cohort_table(config)[["dataset", "num_samples"]],
        on="dataset",
        validate="many_to_one",
    )
    baseline = table.loc[
        table.method.eq("full_hmm"), ["dataset", "num_edges", "num_trees"]
    ]
    baseline = baseline.rename(
        columns={"num_edges": "full_edges", "num_trees": "full_trees"}
    )
    table = table.merge(baseline, on="dataset", validate="many_to_one")
    table["edges_change_percentage"] = 100 * (table.num_edges / table.full_edges - 1)
    table["trees_change_percentage"] = 100 * (table.num_trees / table.full_trees - 1)
    table.to_csv(config["cache_dir"] / "dating_summary.csv", index=False)
    return table


def age_comparison(config: dict, dataset: dict, dated: pd.DataFrame) -> pd.DataFrame:
    """One mutation intersection across all inferred runs and optional truth."""
    rows = dated.loc[dated.dataset.eq(dataset["name"])].reset_index(drop=True)
    paths = [pathlib.Path(path) for path in rows.dated_path]
    if dataset["truth"] is not None:
        paths.append(config["data_dir"] / dataset["truth"])
    stamp = signature(paths, config["frequency_bins"], age_comparison)
    output = config["cache_dir"] / "age_pairs" / f"{dataset['name']}.parquet"
    metadata = output.with_suffix(".json")
    summary_path = output.with_suffix(".csv")
    if (
        output.exists()
        and summary_path.exists()
        and metadata.exists()
        and (metadata.read_text() == stamp)
    ):
        return pd.read_csv(summary_path)
    ages = {}
    common = None
    candidates = []
    for row in rows.itertuples():
        tree = tskit.load(row.dated_path)
        values = mutation_ages(
            tree, posterior=True, include_frequency=row.method == "full_hmm"
        )
        key = row.method if row.method == "full_hmm" else f"af_{row.af_cutoff}"
        ages[key] = values
        candidates.append(
            {
                "dataset": dataset["name"],
                "method": row.method,
                "af_cutoff": row.af_cutoff,
                "finite_single_mutation_sites": len(values),
            }
        )
        common = (
            values.index
            if common is None
            else common.intersection(values.index, sort=False)
        )
    if dataset["truth"] is not None:
        truth = tszip.decompress(config["data_dir"] / dataset["truth"])
        truth = window_tree_sequence(truth, dataset["analysis_extent_bp"])
        values = mutation_ages(truth, posterior=False)
        ages["truth"] = values
        common = common.intersection(values.index, sort=False)
    common = common.sort_values()
    pairs = pd.DataFrame({key: values.age.loc[common] for key, values in ages.items()})
    pairs["derived_frequency"] = ages["full_hmm"].derived_frequency.loc[common]
    pairs["frequency_bin"] = frequency_bins(
        pairs.derived_frequency, config["frequency_bins"]
    )
    comparisons = []
    references = ["full_hmm"] if dataset["truth"] is None else ["truth", "full_hmm"]
    for reference in references:
        methods = rows if reference == "truth" else rows.loc[rows.method.ne("full_hmm")]
        for row in methods.itertuples():
            key = row.method if row.method == "full_hmm" else f"af_{row.af_cutoff}"
            groups = [("all", pairs)]
            groups.extend(
                (
                    (label, pairs.loc[pairs.frequency_bin.eq(label)])
                    for label in config["frequency_bins"]["labels"]
                )
            )
            for label, group in groups:
                x = group[reference]
                y = group[key]
                error = y - x
                correlation = np.nan
                rank_correlation = np.nan
                if len(group) > 1 and x.nunique() > 1 and (y.nunique() > 1):
                    correlation = float(stats.pearsonr(x, y).statistic)
                    rank_correlation = float(stats.spearmanr(x, y).statistic)
                comparisons.append(
                    {
                        "dataset": dataset["name"],
                        "num_samples": row.num_samples,
                        "method": row.method,
                        "af_cutoff": row.af_cutoff,
                        "reference_method": reference,
                        "frequency_bin": label,
                        "matched_mutations": len(group),
                        "pearson_r": correlation,
                        "spearman_r": rank_correlation,
                        "mean_error_generations": error.mean(),
                        "median_absolute_error_generations": error.abs().median(),
                        "mean_estimated_age": y.mean(),
                        "mean_reference_age": x.mean(),
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs_output = pairs.reset_index()
    pl.DataFrame(
        {name: pairs_output[name].to_numpy() for name in pairs_output.columns}
    ).write_parquet(output)
    result = pd.DataFrame(comparisons)
    result.to_csv(summary_path, index=False)
    counts = pd.DataFrame(candidates)
    counts["common_mutations"] = len(common)
    counts["retained_percentage"] = (
        100 * len(common) / counts.finite_single_mutation_sites
    )
    counts.to_csv(output.with_name(f"{output.stem}_retention.csv"), index=False)
    metadata.write_text(stamp)
    return result


def perf_scaling_mutation_correlations(
    config: dict, dated: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for dataset in config["manifest"]["datasets"]:
        rows.append(age_comparison(config, dataset, dated))
        logger.info("Mutation-age comparison complete: %s", dataset["name"])
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(config["cache_dir"] / "mutation_age_correlations.csv", index=False)
    return table


@dataclasses.dataclass
class StageBreakdown:
    """Separate sequential wall phases from overlapping worker measurements."""

    recorded: pd.DataFrame
    wall: pd.DataFrame
    worker: pd.DataFrame


def stage_breakdown(config: dict, runs: pd.DataFrame) -> StageBreakdown:
    """Partition wall time and replace inclusive gap totals by disjoint parts.

    Coordinator phases occur sequentially. Child timers such as assembly and
    result wait are already inside matching/drain and must not be added again.
    Worker service times overlap across workers; thread CPU measures actual
    execution. Neither can be added to the coordinator wall-time stack.
    """
    experiment = config["experiment"]["performance"]["experiment_id"]
    root = config["data_dir"] / "performance" / experiment
    stages = pd.read_csv(root / "stages.csv")
    assert not stages.duplicated(["run_id", "stage"]).any()
    assert set(stages.run_id) <= set(runs.run_id)
    labels = [
        "run_id",
        "dataset",
        "num_samples",
        "method",
        "af_cutoff",
        "matched_haplotypes",
    ]
    recorded = stages.merge(runs[labels], on="run_id", validate="many_to_one")
    wall_rows = []
    worker_rows = []
    wall_names = [
        "input_setup",
        "reference_catalog",
        "coordinator_process_startup",
        "coordinator_packing_completion",
        "coordinator_worker_readiness",
        "coordinator_matching_drain",
        "coordinator_shutdown",
        "finalise_output",
    ]
    worker_names = [
        "reference_packing",
        "worker_setup",
        "candidate_extension_cover",
        "gap_native_find_path",
        "gap_path_conversion",
        "gap_mutation_extraction",
        "gap_final_concatenation",
    ]
    children = config["stages"]["gap_children"]
    parent = config["stages"]["gap_parent"]
    for run in runs.loc[runs.method.eq("greedy")].itertuples():
        measured = stages.loc[stages.run_id.eq(run.run_id)].set_index("stage")
        metadata = {name: getattr(run, name) for name in labels}
        wall = measured.loc[wall_names, "wall_s"]
        assert wall.notna().all()
        residual = run.wall_s - wall.sum()
        assert residual >= 0, run.run_id
        values = {**wall.to_dict(), "other": residual}
        for stage, seconds in values.items():
            wall_rows.append(
                {
                    **metadata,
                    "stage": stage,
                    "wall_s": seconds,
                    "wall_percentage": 100 * seconds / run.wall_s,
                }
            )
        for metric in ["service_s", "thread_cpu_s"]:
            parts = measured.loc[worker_names, metric]
            assert parts.notna().all()
            gap_other = (
                measured.loc[parent, metric] - measured.loc[children, metric].sum()
            )
            assert gap_other >= 0, (run.run_id, metric)
            values = {**parts.to_dict(), "gap_other": gap_other}
            for stage, seconds in values.items():
                worker_rows.append(
                    {
                        **metadata,
                        "stage": stage,
                        "metric": metric,
                        "seconds": seconds,
                        "ms_per_haplotype": 1000 * seconds / run.matched_haplotypes,
                    }
                )
    wall_table = pd.DataFrame(wall_rows)
    worker_table = pd.DataFrame(worker_rows)
    recorded.to_csv(config["cache_dir"] / "recorded_stages.csv", index=False)
    wall_table.to_csv(config["cache_dir"] / "wall_stage_breakdown.csv", index=False)
    worker_table.to_csv(config["cache_dir"] / "worker_stage_breakdown.csv", index=False)
    return StageBreakdown(recorded, wall_table, worker_table)


def recurrent_site_counts(ts: tskit.TreeSequence, positions: np.ndarray) -> dict:
    """Count sites with multiple mutation records on a specified site axis.

    This includes back mutations. The conditional mean counts all mutation
    records at recurrent sites, including the first event. Sites with no
    mutations remain in the denominator. No variants or genotypes are decoded.
    """
    mutations_per_site = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    selected = np.isin(ts.sites_position, positions, assume_unique=True)
    counts = mutations_per_site[selected]
    assert len(counts) == len(positions)
    recurrent = counts[counts > 1]
    num_recurrent = len(recurrent)
    mean_mutations = np.nan if num_recurrent == 0 else recurrent.mean()
    return {
        "num_sites": len(counts),
        "recurrent_sites": num_recurrent,
        "recurrent_site_percentage": 100 * num_recurrent / len(counts),
        "mutations_at_recurrent_sites": int(recurrent.sum()),
        "mean_mutations_per_recurrent_site": mean_mutations,
        "additional_mutations": int((recurrent - 1).sum()),
    }


@dataclasses.dataclass
class RecurrenceJob:
    root: pathlib.Path
    cache: pathlib.Path
    dataset: dict


def recurrent_dataset(job: RecurrenceJob) -> pd.DataFrame:
    """Compare raw Full HMM, stitched and truth on saved inference positions."""
    dataset = job.dataset
    cases = [{"method": "full_hmm", "af_cutoff": None, "path": dataset["full_hmm"]}]
    for run in dataset["runs"]:
        cases.append(
            {"method": "greedy", "af_cutoff": run["af_cutoff"], "path": run["stitched"]}
        )
    if dataset["truth"] is not None:
        cases.append({"method": "truth", "af_cutoff": None, "path": dataset["truth"]})
    paths = [job.root / case["path"] for case in cases]
    settings = {"cases": cases, "operation": inspect.getsource(recurrent_site_counts)}
    stamp = signature(paths, settings, recurrent_dataset)
    output = job.cache / "recurrent_sites" / f"{dataset['name']}.csv"
    cached = read_cached(output, stamp)
    if cached is not None:
        return cached
    full = tszip.decompress(paths[0])
    positions = full.sites_position
    rows = []
    for case, path in zip(cases, paths, strict=True):
        ts = full if case["method"] == "full_hmm" else tszip.decompress(path)
        if case["method"] != "truth":
            np.testing.assert_array_equal(ts.sites_position, positions)
        counts = recurrent_site_counts(ts, positions)
        rows.append(
            {
                "dataset": dataset["name"],
                "method": case["method"],
                "af_cutoff": case["af_cutoff"],
                **counts,
            }
        )
    table = pd.DataFrame(rows)
    save_cached(table, output, stamp)
    return table


def recurrent_sites(config: dict) -> pd.DataFrame:
    """Distribute independent cohort counts across worker processes and cache."""
    jobs = [
        RecurrenceJob(config["data_dir"], config["cache_dir"], dataset)
        for dataset in config["manifest"]["datasets"]
    ]
    workers = config["workers"]
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    context = multiprocessing.get_context("spawn")
    rows = []
    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=numcodecs.blosc.set_nthreads,
        initargs=(config["helper_threads"],),
    ) as pool:
        futures = {pool.submit(recurrent_dataset, job): job for job in jobs}
        for future in cf.as_completed(futures):
            rows.append(future.result())
            logger.info("Recurrent sites: %s", futures[future].dataset["name"])
    table = pd.concat(rows, ignore_index=True)
    sizes = cohort_table(config)[["dataset", "num_samples"]]
    table = table.merge(sizes, on="dataset", validate="many_to_one")
    table = table.sort_values(["num_samples", "method", "af_cutoff"])
    table.to_csv(config["cache_dir"] / "recurrent_site_summary.csv", index=False)
    return table


@dataclasses.dataclass
class RootEdgeJob:
    root: pathlib.Path
    cache: pathlib.Path
    dataset: dict
    categories: list[str]


def classify_edges(
    left, right, mask, haplotype, positions, sequence_length, categories
):
    """Assign whole observed edges to seams, gap interiors or focal coverage.

    An edge touches a seam if it has positive gap overlap and contains an
    internal focal/gap boundary, including either endpoint. Outer domain ends
    are not seams. Classification does not split edges or count them twice.
    """
    begin, end = mask["offsets"][haplotype : haplotype + 2]
    site_left = mask["left_site_index"][begin:end]
    site_right = mask["right_site_index"][begin:end]
    is_gap = mask["is_gap"][begin:end]
    boundaries = np.concatenate([positions, [sequence_length]])
    gap_left = boundaries[site_left[is_gap]].copy()
    gap_right = boundaries[site_right[is_gap]]
    gap_left[site_left[is_gap] == 0] = 0
    seam_sites = site_left[1:][is_gap[1:] != is_gap[:-1]]
    seams = positions[seam_sites]
    overlap = np.zeros(len(left), dtype=bool)
    for lo, hi in zip(gap_left, gap_right, strict=True):
        overlap |= (left < hi) & (right > lo)
    first_seam = np.searchsorted(seams, left, side="left")
    last_seam = np.searchsorted(seams, right, side="right")
    touches_seam = overlap & (last_seam > first_seam)
    labels = np.full(len(left), categories[2], dtype=object)
    labels[overlap] = categories[1]
    labels[touches_seam] = categories[0]
    return labels


def observed_root_edges(ts, reference, root, mask, intervals, categories):
    """Clip root/sample edges to the observed extent, retaining edge identity."""
    sample_nodes = ts.samples()
    expected_nodes = np.arange(reference.num_nodes, ts.num_nodes)
    np.testing.assert_array_equal(sample_nodes, expected_nodes)
    expected = list(
        zip(mask["source"], mask["sample_id"], mask["ploidy_index"], strict=True)
    )
    observed = []
    for node in sample_nodes:
        metadata = ts.node(int(node)).metadata
        observed.append(
            (metadata["source"], metadata["sample_id"], metadata["ploidy_index"])
        )
    assert observed == expected
    np.testing.assert_array_equal(ts.sites_position, reference.sites_position)
    selected = np.flatnonzero(
        (ts.edges_parent == root) & np.isin(ts.edges_child, sample_nodes)
    )
    original_left = ts.edges_left[selected]
    original_right = ts.edges_right[selected]
    span = np.zeros(len(selected), dtype=float)
    clipped_left = np.full(len(selected), np.inf)
    clipped_right = np.full(len(selected), -np.inf)
    for lo, hi in intervals:
        left = np.maximum(original_left, lo)
        right = np.minimum(original_right, hi)
        present = left < right
        span[present] += right[present] - left[present]
        clipped_left[present] = np.minimum(clipped_left[present], left[present])
        clipped_right[present] = np.maximum(clipped_right[present], right[present])
    present = span > 0
    table = pd.DataFrame(
        {
            "edge_id": selected[present],
            "haplotype_index": ts.edges_child[selected[present]] - reference.num_nodes,
            "left_bp": clipped_left[present],
            "right_bp": clipped_right[present],
            "span_bp": span[present],
        }
    )
    table["category"] = ""
    for haplotype, indices in table.groupby("haplotype_index").groups.items():
        group = table.loc[indices]
        labels = classify_edges(
            group.left_bp.to_numpy(),
            group.right_bp.to_numpy(),
            mask,
            haplotype,
            reference.sites_position,
            reference.sequence_length,
            categories,
        )
        table.loc[indices, "category"] = labels
    assert table.category.isin(categories).all()
    return table


def root_edges_analyse_dataset(job: RootEdgeJob) -> pd.DataFrame:
    """Cache whole-edge measurements for each parameter's paired focal mask."""
    paths = [job.root / job.dataset["reference"], job.root / job.dataset["full_hmm"]]
    for run in job.dataset["runs"]:
        paths.extend([job.root / run["stitched"], job.root / run["mask"]])
    settings = {
        "categories": job.categories,
        "intervals": job.dataset["analysis_intervals"],
        "helper_code": [
            inspect.getsource(classify_edges),
            inspect.getsource(observed_root_edges),
        ],
    }
    stamp = signature(paths, settings, root_edges_analyse_dataset)
    output = job.cache / "root_edges" / f"{job.dataset['name']}.csv"
    cached = read_cached(output, stamp)
    if cached is not None:
        return cached
    reference = tszip.decompress(paths[0])
    stem = reference.edge(-1)
    assert reference.edge(-2).parent != stem.parent
    root = stem.child
    full = tszip.decompress(paths[1])
    rows = []
    for run in job.dataset["runs"]:
        with np.load(job.root / run["mask"]) as archive:
            mask = dict(archive)
        for method in ["full_hmm", "greedy"]:
            matched = (
                full
                if method == "full_hmm"
                else tszip.decompress(job.root / run["stitched"])
            )
            assert matched.edge(-1).parent == stem.parent
            assert matched.edge(-1).child == root
            row = observed_root_edges(
                matched,
                reference,
                root,
                mask,
                job.dataset["analysis_intervals"],
                job.categories,
            )
            row["method"] = method
            row["af_cutoff"] = run["af_cutoff"]
            row["dataset"] = job.dataset["name"]
            rows.append(row)
            if method == "greedy":
                del matched
    table = pd.concat(rows, ignore_index=True)
    full_totals = (
        table.loc[table.method.eq("full_hmm")]
        .groupby("af_cutoff")
        .span_bp.agg(["size", "sum"])
    )
    assert full_totals["size"].nunique() == 1
    np.testing.assert_allclose(full_totals["sum"], full_totals["sum"].iloc[0])
    save_cached(table, output, stamp)
    return table


def root_edge_summary(config: dict) -> pd.DataFrame:
    """Distribute cohort analyses, retaining zero categories and whole-edge counts."""
    categories = ["seam", "gap_interior", "focal"]
    jobs = [
        RootEdgeJob(config["data_dir"], config["cache_dir"], dataset, categories)
        for dataset in config["manifest"]["datasets"]
    ]
    workers = config["workers"]
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    rows = []
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=numcodecs.blosc.set_nthreads,
        initargs=(config["helper_threads"],),
    ) as pool:
        futures = {pool.submit(root_edges_analyse_dataset, job): job for job in jobs}
        for future in cf.as_completed(futures):
            rows.append(future.result())
            logger.info("Root/sample edges: %s", futures[future].dataset["name"])
    edges = pd.concat(rows, ignore_index=True)
    keys = ["dataset", "af_cutoff", "method", "category"]
    summary = edges.groupby(keys).agg(
        edge_count=("span_bp", "size"), span_bp=("span_bp", "sum")
    )
    sizes = cohort_table(config)
    combinations = []
    for dataset in config["manifest"]["datasets"]:
        for run in dataset["runs"]:
            for method in ["full_hmm", "greedy"]:
                for category in categories:
                    combinations.append(
                        (dataset["name"], run["af_cutoff"], method, category)
                    )
    index = pd.MultiIndex.from_tuples(combinations, names=keys)
    summary = summary.reindex(index, fill_value=0).reset_index()
    summary = summary.merge(
        sizes[["dataset", "num_samples", "num_haplotypes"]],
        on="dataset",
        validate="many_to_one",
    )
    summary["edge_count_per_haplotype"] = summary.edge_count / summary.num_haplotypes
    summary["span_kb_per_haplotype"] = summary.span_bp / summary.num_haplotypes / 1000
    summary.to_csv(config["cache_dir"] / "root_edge_summary.csv", index=False)
    return summary


@dataclasses.dataclass
class PerfPt5Results:
    """Validated runs, correctness comparisons and population-labelled samples."""

    runs: pd.DataFrame
    stages: pd.DataFrame
    quality: pd.DataFrame
    fingerprints: pd.DataFrame
    comparisons: pd.DataFrame
    samples: pd.DataFrame


def perf_pt5_load_results(config_path: dict) -> PerfPt5Results:
    """Read both completed passes, preserving every timing repeat."""
    config = config_path.copy()
    root = resolve_path(config["benchmark_dir"], config_path["base_dir"])
    completion = json.loads((root / "complete.json").read_text())
    selection = config["correctness"]
    sample_settings = config["per_sample"]
    population_path = resolve_path(
        sample_settings["population_metadata"], config_path["base_dir"]
    )
    run_tables = []
    stage_tables = []
    quality_tables = []
    comparison_tables = []
    sample_tables = []
    for method, settings in config["methods"].items():
        if method == "full_hmm":
            continue
        cutoff = settings["ac_cutoff"]
        assert cutoff in completion["ac_cutoff"]
        folder = root / f"ac{cutoff}"
        manifest = json.loads((folder / "manifest.json").read_text())
        expected = manifest["configuration"]["performance"]
        assert set(expected["methods"]) == {"full_hmm", "greedy"}
        runs = pd.read_csv(folder / "runs.csv")
        runs = runs.loc[runs.dataset.eq(config["dataset"])].copy()
        expected_trials = len(expected["methods"]) * len(expected["modes"])
        expected_trials *= len(expected["worker_counts"]) * expected["repeats"]
        assert len(runs) == expected_trials
        assert runs.status.eq("ok").all() and runs.ac_cutoff.eq(cutoff).all()
        runs["method"] = runs.method.replace({"greedy": method})
        run_tables.append(runs)
        stages = pd.read_csv(folder / "stages.csv")
        stages = stages.loc[stages.run_id.isin(runs.run_id)].copy()
        stage_tables.append(stages)
        quality = pd.DataFrame(json.loads((folder / "quality.json").read_text()))
        quality = quality.loc[quality.run_id.isin(runs.run_id)].copy()
        assert len(quality) == len(runs) and quality.status.eq("ok").all()
        assert quality.reconstructed_called_allele_errors.eq(0).all()
        assert quality.mismatches.eq(0).all()
        quality_tables.append(quality)
        inputs = manifest["datasets"][config["dataset"]]["inputs"]
        chosen = runs.loc[runs["mode"].eq(selection["mode"])]
        chosen = chosen.loc[chosen["repeat"].eq(selection["repeat"])].set_index(
            "method"
        )
        assert len(chosen) == 2
        mask_path = pathlib.Path(inputs["mask"])
        trial_folder = pathlib.Path(chosen.loc[method, "output_matches"]).parent
        trial_mask_path = trial_folder / f"ac{cutoff}_focal_mask.npz"
        with np.load(mask_path, allow_pickle=False) as common_mask:
            with np.load(trial_mask_path, allow_pickle=False) as trial_mask:
                for name in common_mask.files:
                    assert np.array_equal(trial_mask[name], common_mask[name])
        table = compare_paths(
            pathlib.Path(inputs["reference"]),
            mask_path,
            pathlib.Path(chosen.loc["full_hmm", "output_matches"]),
            pathlib.Path(chosen.loc[method, "output_matches"]),
            method,
        )
        report = quality.loc[quality.run_id.eq(chosen.loc[method, "run_id"])].iloc[0]
        assert table.stitched_switches.sum() == report["switches"]
        disagreements = table.focal_disagreement_sites + table.gap_disagreement_sites
        sites = table.focal_sites + table.gap_sites
        assert sites.sum() - disagreements.sum() == report["parent_agreement_sites"]
        comparison_tables.append(table)
        resident = runs.loc[runs["mode"].eq("resident_match") & runs.method.eq(method)]
        worker_tables = []
        for run in resident.itertuples():
            path = (
                pathlib.Path(run.output_matches).parent
                / sample_settings["worker_times_filename"]
            )
            worker = pd.read_csv(path)
            measured = stages.loc[
                stages.run_id.eq(run.run_id) & stages.scope.eq("worker")
            ]
            assert np.isclose(worker.service_s.sum(), measured.service_s.sum())
            worker["method"] = method
            worker["repeat"] = run.repeat
            worker_tables.append(worker)
        samples = per_sample_data(
            pd.concat(worker_tables, ignore_index=True),
            mask_path,
            pathlib.Path(inputs["reference"]),
            population_path,
            sample_settings,
            resident,
        )
        samples["ac_cutoff"] = cutoff
        sample_tables.append(samples)
    runs = pd.concat(run_tables, ignore_index=True)
    assert not runs.run_id.duplicated().any()
    assert runs.actual_workers.nunique() == 1
    assert runs.matched_haplotypes.nunique() == 1
    jobs = runs.to_dict("records")
    workers = config["workers"]
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        fingerprints = pd.DataFrame(pool.map(perf_path_fingerprint, jobs))
    assert fingerprints.groupby("method").sha256.nunique().eq(1).all()
    assert fingerprints.haplotypes.eq(runs.matched_haplotypes.iloc[0]).all()
    return PerfPt5Results(
        runs,
        pd.concat(stage_tables, ignore_index=True),
        pd.concat(quality_tables, ignore_index=True),
        fingerprints,
        pd.concat(comparison_tables, ignore_index=True),
        pd.concat(sample_tables, ignore_index=True),
    )


@dataclasses.dataclass
class DatingCase:
    """A saved inferred ARG and the observed sample source used for dating."""

    dataset: str
    method: str
    ac_cutoff: int | None
    minimum_size: int | None
    raw_path: pathlib.Path
    native_config_path: dict
    output_dir: pathlib.Path
    span: list[float]
    mutation_rate: float
    settings: dict
    source_include: str | None
    source_exclude: str | None
    sample_store_signature: str
    max_mismatches: int | None = None

    @property
    def dated_path(self) -> pathlib.Path:
        return self.output_dir / f"{self.raw_path.stem}_dated.trees"


def store_signature(path: pathlib.Path) -> str:
    """Detect changes to observed VCZ metadata or genotype chunks for caching."""
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            stat = item.stat()
            content = [str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns]
            digest.update(json.dumps(content).encode())
    return digest.hexdigest()


def dating_load_configuration(config_path: dict) -> dict:
    """Read either the legacy mismatch grid or the minimum-size experiment."""
    config = config_path.copy()
    experiment_path = resolve_path(config["experiment_config"], config_path["base_dir"])
    experiment_base_dir = config.get("experiment_base_dir", experiment_path.parent)
    with experiment_path.open() as source:
        config["experiment"] = yaml.safe_load(source)
    config["experiment_path"] = experiment_path
    config["data_dir"] = resolve_path(
        config["experiment"]["data_dir"], experiment_base_dir
    )
    return config


def dating_cases(config_path: dict) -> list[DatingCase]:
    """Select both Full HMM outputs and the configured stitched parameter grid."""
    config = dating_load_configuration(config_path)
    experiment = config["experiment"]
    settings = config["dating"]
    experiment_datasets = {item["name"]: item for item in experiment["datasets"]}
    output_dir = resolve_path(config["output_dir"], config_path["base_dir"])
    output_dir /= settings["output_subdir"]
    cases = []
    for dataset in config["datasets"]:
        name = dataset["name"]
        native_path = config["data_dir"] / "configs" / f"{name}_ancestor_inference.toml"
        native = tsinfer.config.Config.from_toml(native_path)
        signature = store_signature(pathlib.Path(native.sources["samples"].path))
        grid = [(None, None, None)]
        minimums = experiment.get("min_focal_region_size", [None])
        grid.extend(
            itertools.product(
                experiment["ac_cutoff"], experiment["max_mismatches"], minimums
            )
        )
        for cutoff, budget, minimum in grid:
            prefix = f"{name}_inferred"
            method = "full_hmm"
            if cutoff is not None:
                prefix += f"_ac{cutoff}_k{budget}"
                if minimum is not None:
                    prefix += f"_min{minimum}"
                prefix += f"_{config['method']}_stitched"
                method = f"stitched_{config['method']}"
            cases.append(
                DatingCase(
                    name,
                    method,
                    cutoff,
                    minimum,
                    config["data_dir"] / "matches" / f"{prefix}_samples_raw.trees",
                    native_path,
                    output_dir,
                    dataset["analysis_span_bp"],
                    dataset["mutation_rate"],
                    settings,
                    experiment_datasets[name].get("include"),
                    experiment_datasets[name].get("exclude"),
                    signature,
                    max_mismatches=budget,
                )
            )
    return cases


def augmentation_config(
    native: tsinfer.config.Config, include: str | None, exclude: str | None
) -> tsinfer.config.Config:
    """Restore dataset filters, removing matching-only singleton exclusions."""
    sample_source = dataclasses.replace(
        native.sources["samples"], include=include, exclude=exclude
    )
    return dataclasses.replace(
        native,
        sources={"samples": sample_source},
        augment_sites=tsinfer.config.AugmentSitesConfig(sources=["samples"]),
    )


def date_case(case: DatingCase) -> dict:
    """Save a dated ARG, reusing it only when its inputs and settings match."""
    with case.raw_path.open("rb") as source:
        raw_hash = hashlib.file_digest(source, "sha256").hexdigest()
    signature = {
        "raw_sha256": raw_hash,
        "native_config": case.native_config_path.read_text(),
        "sample_store_signature": case.sample_store_signature,
        "span": case.span,
        "mutation_rate": case.mutation_rate,
        "augmentation_filters": {
            "include": case.source_include,
            "exclude": case.source_exclude,
        },
        "post_process": case.settings["post_process"],
        "preprocess": case.settings["preprocess"],
        "estimation": case.settings["estimation"],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["tsinfer", "tsdate", "tskit"]
        },
        "tsinfer_pipeline_sha256": hashlib.sha256(
            pathlib.Path(tsinfer.pipeline.__file__).read_bytes()
        ).hexdigest(),
        "analysis_sha256": hashlib.sha256(
            inspect.getsource(date_case).encode()
        ).hexdigest(),
    }
    manifest_path = case.dated_path.with_suffix(".json")
    if case.dated_path.exists() and manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["signature"] == signature:
            return saved["measurement"]
    case.output_dir.mkdir(parents=True, exist_ok=True)
    numcodecs.blosc.set_nthreads(case.settings["helper_threads"])
    zarr.config.set(
        {
            "async.concurrency": case.settings["helper_threads"],
            "threading.max_workers": case.settings["helper_threads"],
        }
    )
    raw = tskit.load(case.raw_path)
    native = tsinfer.config.Config.from_toml(case.native_config_path)
    native = augmentation_config(native, case.source_include, case.source_exclude)
    started = time.perf_counter()
    processed = tsinfer.post_process(raw, **case.settings["post_process"])
    processed_at = time.perf_counter()
    augmented = tsinfer.augment_sites(processed, native)
    augmented_at = time.perf_counter()
    window = window_tree_sequence(augmented, case.span)
    prepared = tsdate.preprocess_ts(window, **case.settings["preprocess"])
    prepared_at = time.perf_counter()
    dated = tsdate.date(
        prepared, mutation_rate=case.mutation_rate, **case.settings["estimation"]
    )
    dated_at = time.perf_counter()
    assert dated.num_samples == raw.num_samples
    assert dated.time_units == case.settings["estimation"]["time_units"]
    dated.dump(case.dated_path)
    measurement = {
        "dataset": case.dataset,
        "comparison_method": case.method,
        "ac_cutoff": case.ac_cutoff,
        "min_focal_region_size": case.minimum_size,
        "dated_path": str(case.dated_path),
        "raw_nodes": raw.num_nodes,
        "dated_nodes": dated.num_nodes,
        "raw_sites": raw.num_sites,
        "dated_sites": dated.num_sites,
        "dated_mutations": dated.num_mutations,
        "post_process_s": processed_at - started,
        "augment_sites_s": augmented_at - processed_at,
        "preprocess_s": prepared_at - augmented_at,
        "dating_s": dated_at - prepared_at,
    }
    manifest_path.write_text(
        json.dumps({"signature": signature, "measurement": measurement}, indent=2)
        + "\n"
    )
    return measurement


def dating_date_all(config_path: dict) -> pd.DataFrame:
    """Distribute the saved ARGs across available CPU cores."""
    config = dating_load_configuration(config_path)
    cases = dating_cases(config_path)
    workers = config["dating"]["workers"]
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(cases))
    context = multiprocessing.get_context("spawn")
    rows = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(date_case, case): case for case in cases}
        for future in cf.as_completed(futures):
            row = future.result()
            row["max_mismatches"] = futures[future].max_mismatches
            rows.append(row)
            logger.info(
                "Dated %s: %s, AC %s, minimum %s",
                row["dataset"],
                row["comparison_method"],
                row["ac_cutoff"],
                row["min_focal_region_size"],
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "comparison_method", "ac_cutoff", "min_focal_region_size"]
    )


def mutation_ages(
    ts: tskit.TreeSequence, posterior: bool, include_frequency: bool = False
) -> pd.DataFrame:
    """Return finite ages at sites containing exactly one mutation.

    Keys include position and ancestral/derived states to exclude polarity or
    allele mismatches. Recurrent and back-mutation sites have no unique pairing.
    """
    rows = []
    tree = ts.first() if include_frequency else None
    for site in ts.sites():
        if len(site.mutations) != 1:
            continue
        mutation = site.mutations[0]
        age = mutation.metadata["mn"] if posterior else mutation.time
        if np.isfinite(age):
            row = {
                "position": site.position,
                "ancestral_state": site.ancestral_state,
                "derived_state": mutation.derived_state,
                "age": age,
            }
            if tree is not None:
                tree.seek(site.position)
                row["derived_frequency"] = (
                    tree.num_samples(mutation.node) / ts.num_samples
                )
            rows.append(row)
    return pd.DataFrame(rows).set_index(
        ["position", "ancestral_state", "derived_state"]
    )


def correlate_ages(comparison: pd.Series, reference: pd.Series) -> float:
    """Calculate Pearson r on aligned mutation ages with a defined variance."""
    if not comparison.index.equals(reference.index):
        raise ValueError(
            "Mutation-age series must have identical ordered mutation keys"
        )
    if len(comparison) < 2 or comparison.nunique() < 2 or reference.nunique() < 2:
        raise ValueError("Mutation-age correlation requires at least two distinct ages")
    return float(stats.pearsonr(comparison.to_numpy(), reference.to_numpy()).statistic)


def frequency_bins(frequencies: pd.Series, settings: dict) -> pd.Series:
    """Use half-open derived-frequency bins, placing boundaries in the upper bin."""
    indices = np.searchsorted(settings["cutoffs"], frequencies.to_numpy(), side="right")
    labels = np.asarray(settings["labels"])
    return pd.Series(labels[indices], index=frequencies.index)


def correlation_rows(
    runs: pd.DataFrame,
    ages: list[pd.DataFrame],
    reference: pd.Series,
    frequencies: pd.Series,
    reference_method: str,
    settings: dict,
) -> list[dict]:
    """Use fixed mutation keys and Full HMM frequencies for every case in a bin."""
    bins = frequency_bins(frequencies.loc[reference.index], settings)
    rows = []
    groups = [("all", reference.index)]
    groups.extend(((label, bins.index[bins.eq(label)]) for label in settings["labels"]))
    for label, keys in groups:
        reference_ages = reference.loc[keys]
        for index, run in runs.iterrows():
            if reference_method == "full_hmm" and run.comparison_method == "full_hmm":
                continue
            inferred = ages[index]["age"].loc[keys]
            errors = inferred - reference_ages
            rows.append(
                {
                    "dataset": run.dataset,
                    "comparison_method": run.comparison_method,
                    "ac_cutoff": run.ac_cutoff,
                    "min_focal_region_size": run.min_focal_region_size,
                    "max_mismatches": run.max_mismatches,
                    "reference_method": reference_method,
                    "frequency_bin": label,
                    "matched_mutations": len(keys),
                    "pearson_r": correlate_ages(inferred, reference_ages),
                    "mean_estimated_age": inferred.mean(),
                    "mean_reference_age": reference_ages.mean(),
                    "mean_error_generations": errors.mean(),
                    "median_absolute_error_generations": errors.abs().median(),
                }
            )
    return rows


@dataclasses.dataclass
class MatchedMutationAges:
    """Aligned inferred ages, observed frequencies, and optional simulation truth."""

    inferred: list[pd.DataFrame]
    common: pd.MultiIndex
    frequencies: pd.Series
    baseline: pd.Series
    truth: pd.Series | None


def matched_mutation_ages(
    runs: pd.DataFrame, dataset: dict, config: dict, include_truth: bool = True
) -> MatchedMutationAges:
    """Hold mutation identity and Full HMM frequency fixed across a parameter grid."""
    ages = []
    for run in runs.itertuples():
        ts = tskit.load(run.dated_path)
        values = mutation_ages(
            ts, posterior=True, include_frequency=run.comparison_method == "full_hmm"
        )
        ages.append(values)
    common = ages[0].index
    for values in ages[1:]:
        common = common.intersection(values.index, sort=False)
    common = common.sort_values()
    baseline_index = runs.index[runs.comparison_method.eq("full_hmm")].item()
    baseline = ages[baseline_index]["age"].loc[common]
    frequencies = ages[baseline_index]["derived_frequency"]
    source = next(
        (
            item
            for item in config["experiment"]["datasets"]
            if item["name"] == dataset["name"]
        )
    )
    true_ages = None
    if include_truth and source["ts_path"] is not None:
        truth_path = resolve_path(source["ts_path"], config["experiment_path"].parent)
        truth = tskit.load(truth_path)
        truth = window_tree_sequence(truth, dataset["analysis_span_bp"])
        true_ages = mutation_ages(truth, posterior=False)["age"]
        truth_common = common.intersection(true_ages.index, sort=False).sort_values()
        true_ages = true_ages.loc[truth_common]
    return MatchedMutationAges(ages, common, frequencies, baseline, true_ages)


def stitched_mutation_age_pairs(
    dated_runs: pd.DataFrame, config_path: dict
) -> pd.DataFrame:
    """Pair stitched and Full HMM posterior ages across all frequencies.

    Each dataset uses the same common mutation keys as its Full HMM correlation
    panel. Simulation truth is neither required nor used for these comparisons.
    """
    config = dating_load_configuration(config_path)
    frames = []
    for dataset in config["datasets"]:
        name = dataset["name"]
        if name not in config["dating"]["compare_full_datasets"]:
            continue
        runs = dated_runs.loc[dated_runs.dataset.eq(name)].reset_index(drop=True)
        matched = matched_mutation_ages(runs, dataset, config, include_truth=False)
        keys = matched.common
        for index, run in runs.iterrows():
            if run.comparison_method == "full_hmm":
                continue
            pairs = pd.DataFrame(
                {
                    "full_hmm_age": matched.baseline,
                    "stitched_age": matched.inferred[index]["age"].loc[keys],
                }
            )
            for column in [
                "dataset",
                "comparison_method",
                "ac_cutoff",
                "max_mismatches",
                "min_focal_region_size",
            ]:
                pairs[column] = run[column]
            frames.append(pairs.reset_index())
    return pd.concat(frames, ignore_index=True)


def dating_mutation_correlations(
    dated_runs: pd.DataFrame, config_path: dict
) -> pd.DataFrame:
    """Compare dates overall and in fixed observed-derived-frequency bins.

    Frequencies come from the dated Full HMM sample genealogy for both panels.
    Within a panel, mutation keys and bin membership are held fixed across cases.
    """
    config = dating_load_configuration(config_path)
    settings = config["dating"]["frequency_bins"]
    output_rows = []
    for dataset in config["datasets"]:
        name = dataset["name"]
        runs = dated_runs.loc[dated_runs.dataset.eq(name)].reset_index(drop=True)
        matched = matched_mutation_ages(runs, dataset, config)
        if name in config["dating"]["compare_full_datasets"]:
            output_rows.extend(
                correlation_rows(
                    runs,
                    matched.inferred,
                    matched.baseline,
                    matched.frequencies,
                    "full_hmm",
                    settings,
                )
            )
        if matched.truth is not None:
            output_rows.extend(
                correlation_rows(
                    runs,
                    matched.inferred,
                    matched.truth,
                    matched.frequencies,
                    "true_arg",
                    settings,
                )
            )
    return pd.DataFrame(output_rows)


@dataclasses.dataclass
class SwitchCounts:
    """Partition parent changes by the provenance of their two adjacent sites."""

    focal: int
    gap: int
    seam: int

    @property
    def total(self) -> int:
        return self.focal + self.gap + self.seam


def site_intervals(mask: np.ndarray) -> np.ndarray:
    """Return maximal true runs as half-open site-index intervals."""
    padded = np.pad(mask, (1, 1), constant_values=False)
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    return transitions.reshape(-1, 2)


def focal_parent_path(
    num_sites: int, left: np.ndarray, right: np.ndarray, parents: np.ndarray
) -> np.ndarray:
    """Cover the union of focal intervals with the fewest copying segments.

    At each uncovered position, choose the available interval extending furthest
    right, breaking ties by original node ID. Keep its parent until that interval
    ends. This greedy interval cover minimizes the number of focal segments in
    each connected covered component. Sites outside the union retain parent -1;
    no HMM output or unanchored exact islands enter gap determination.
    """
    if left.shape != right.shape or left.shape != parents.shape:
        raise ValueError("Focal interval bounds and parents must have the same shape")
    invalid_bounds = (left < 0) | (left >= right) | (right > num_sites)
    if np.any(invalid_bounds) or np.any(parents < 0):
        raise ValueError("Focal intervals require valid site bounds and parent IDs")
    order = np.lexsort((parents, -right, left))
    left = left[order]
    right = right[order]
    parents = parents[order]
    result = np.full(num_sites, -1, dtype=np.int32)
    index = 0
    site = 0
    while index < len(left):
        site = max(site, int(left[index]))
        best_end = site
        best_parent = -1
        while index < len(left) and left[index] <= site:
            end = int(right[index])
            parent = int(parents[index])
            if end > best_end or (end == best_end and parent < best_parent):
                best_end = end
                best_parent = parent
            index += 1
        if best_end > site:
            result[site:best_end] = best_parent
            site = best_end
    return result


def switch_counts(parents: np.ndarray, covered: np.ndarray) -> SwitchCounts:
    """Count focal interiors, gap interiors, and seams in one complete path."""
    if parents.shape != covered.shape or np.any(parents < 0):
        raise ValueError("Switch counting requires complete, partition-aligned parents")
    changes = parents[1:] != parents[:-1]
    focal = covered[1:] & covered[:-1]
    gap = ~covered[1:] & ~covered[:-1]
    seam = covered[1:] != covered[:-1]
    counts = SwitchCounts(
        int(np.count_nonzero(changes & focal)),
        int(np.count_nonzero(changes & gap)),
        int(np.count_nonzero(changes & seam)),
    )
    assert counts.total == np.count_nonzero(changes)
    return counts


def comparison_metrics(
    full: np.ndarray, stitched: np.ndarray, covered: np.ndarray
) -> dict:
    """Summarise switches and agreement without retaining cohort-wide site arrays.

    All four agreement fractions use the entire inference-site axis as their
    denominator. Agreement compares original-reference parent IDs, not alleles.
    Extra switch fraction is the net difference divided by stitched switches;
    it is undefined (None) when the stitched path has no parent changes.
    """
    full_counts = switch_counts(full, covered)
    stitched_counts = switch_counts(stitched, covered)
    disagreement = full != stitched
    focal_disagreement = int(np.count_nonzero(covered & disagreement))
    gap_disagreement = int(np.count_nonzero(~covered & disagreement))
    focal_sites = int(np.count_nonzero(covered))
    gap_sites = len(covered) - focal_sites
    extra = stitched_counts.total - full_counts.total
    metrics = {
        "full_switches": full_counts.total,
        "stitched_switches": stitched_counts.total,
        "num_extra_switches": extra,
        "extra_switch_fraction": extra / stitched_counts.total
        if stitched_counts.total > 0
        else None,
        "focal_sites": focal_sites,
        "gap_sites": gap_sites,
        "focal_disagreement_sites": focal_disagreement,
        "gap_disagreement_sites": gap_disagreement,
        "fraction_parent_agreement": float(np.mean(~disagreement)),
        "parents_identical": bool(not np.any(disagreement)),
        "focal_agree_fraction": (focal_sites - focal_disagreement) / len(covered),
        "focal_disagree_fraction": focal_disagreement / len(covered),
        "gap_agree_fraction": (gap_sites - gap_disagreement) / len(covered),
        "gap_disagree_fraction": gap_disagreement / len(covered),
    }
    for method, counts in (("full", full_counts), ("stitched", stitched_counts)):
        metrics[f"{method}_focal_switches"] = counts.focal
        metrics[f"{method}_gap_switches"] = counts.gap
        metrics[f"{method}_seam_switches"] = counts.seam
    return metrics


@dataclasses.dataclass
class CoverageData:
    """Raw associations and the full haplotype roster, including empty candidates."""

    dataset: str
    num_sites: int
    haplotypes: pl.DataFrame
    offsets: np.ndarray
    focal_ac: np.ndarray
    left_site_index: np.ndarray
    right_site_index: np.ndarray
    max_ac: int
    mismatch_budgets: list[int]
    first_position: int
    last_position: int


@dataclasses.dataclass
class SweepTask:
    """Only numeric arrays travel through the multiprocessing task queue."""

    haplotype_index: int
    num_sites: int
    cutoffs: np.ndarray
    focal_ac: np.ndarray
    left: np.ndarray
    right: np.ndarray


@dataclasses.dataclass
class SweepResult:
    """Coverage statistics in cutoff order for one haplotype."""

    haplotype_index: int
    records: list[dict]


def focal_ac_cutoffs(max_ac: int = 200, num_bins: int = 20) -> np.ndarray:
    """Return increasing integer thresholds, starting at 2, 3, then log spaced.

    Rounding duplicates are filled from the unused integers so there are at
    least ``num_bins`` thresholds whenever the integer range permits it.
    """
    if max_ac < 2 or num_bins < 2:
        raise ValueError("max_ac and num_bins must be >= 2")
    target = min(num_bins, max_ac - 1)
    logarithmic = np.geomspace(2, max_ac, target)
    rounded = np.rint(logarithmic).astype(np.int64)
    required = np.arange(2, min(3, max_ac) + 1)
    cutoffs = np.union1d(rounded, required)
    while len(cutoffs) < target:
        gaps = np.diff(np.log(cutoffs))
        gaps[np.diff(cutoffs) < 2] = -1
        index = int(np.argmax(gaps))
        midpoint = np.sqrt(cutoffs[index] * cutoffs[index + 1])
        value = int(np.rint(midpoint))
        value = min(max(value, cutoffs[index] + 1), cutoffs[index + 1] - 1)
        cutoffs = np.union1d(cutoffs, [value])
    return cutoffs


def _truth_populations(haplotypes: pl.DataFrame, truth_path: pathlib.Path):
    """Use the same VCF identity mapping as match-eval's TS metadata reader."""
    truth = tskit.load(truth_path)
    mapping = truth.map_to_vcf_model()
    nodes_by_id = dict(
        zip(mapping.individuals_name, mapping.individuals_nodes, strict=True)
    )
    populations = []
    for row in haplotypes.iter_rows(named=True):
        node_id = nodes_by_id[row["sample_id"]][row["ploidy_index"]]
        node = truth.node(int(node_id))
        if node.population == tskit.NULL:
            raise ValueError(f"Truth node {node_id} has no population")
        label = truth.population(node.population).metadata.get("name")
        if label is None or label == "":
            raise ValueError(f"Truth population {node.population} has no name")
        populations.append(label)
    return haplotypes.with_columns(pl.Series("population", populations))


def summarise_focal_ancestor_set_sizes(
    focal_path: pathlib.Path, cutoffs: np.ndarray
) -> pl.DataFrame:
    """Average inclusive AC-filtered candidate counts over every NPZ haplotype.

    Match-eval stores distinct ancestor indices within each haplotype's ragged
    candidate set. Count their incidences across all haplotypes, then divide by
    the full haplotype roster, including empty sets. The proportion denominator
    is every ancestor in the panel, without an AC filter. Membership depends on
    carried focal alleles and AC eligibility, independently of mismatch budget.
    """
    with np.load(focal_path, allow_pickle=False) as focal:
        num_haplotypes = len(focal["sample_id"])
        num_ancestors = len(focal["ancestor_id"])
        candidate_indices = focal["ancestor_index"]
        candidate_ac = focal["derived_ac"][candidate_indices]
    if num_haplotypes == 0 or num_ancestors == 0:
        raise ValueError("Set-size proportions require haplotypes and panel ancestors")
    sorted_ac = np.sort(candidate_ac)
    total_candidates = np.searchsorted(sorted_ac, cutoffs, side="right")
    mean_candidates = total_candidates / num_haplotypes
    proportions = mean_candidates / num_ancestors
    return pl.DataFrame(
        {
            "focal_ac": cutoffs,
            "num_haplotypes": num_haplotypes,
            "num_ancestors": num_ancestors,
            "mean_num_focal_ancestors": mean_candidates,
            "mean_proportion_included": proportions,
        }
    )


def load_coverage_data(
    data_dir: pathlib.Path,
    dataset: str,
    max_ac: int,
    max_mismatches: list[int],
    truth_ts_path: pathlib.Path | None = None,
) -> CoverageData:
    """Load compact intervals and select requested budget columns without repacking.

    Validate the archive roster against the focal NPZ and site count against
    the panel. Population labels come from the statistics CSV or the original
    simulation TS via :func:`_truth_populations`. Exact AC arrays are retained;
    :func:`summarise_coverage` applies inclusive cumulative analysis cutoffs.
    Requested limits must be within those used to generate the archive.
    """
    if len(max_mismatches) == 0 or any((budget < 0 for budget in max_mismatches)):
        raise ValueError("Select at least one nonnegative mismatch budget")
    panel_path = data_dir / "ancestors" / f"{dataset}_inferred_ancestors.zarr"
    panel = zarr.open_group(panel_path, mode="r")
    positions = panel["variant_position"]
    num_sites = positions.shape[0]
    if num_sites == 0:
        raise ValueError("Coverage requires a nonempty panel site axis")
    focal_path = (
        data_dir / "focal_ancestors" / f"{dataset}_inferred_focal_ancestors.npz"
    )
    with np.load(focal_path, allow_pickle=False) as focal:
        haplotypes = pl.DataFrame(
            {"sample_id": focal["sample_id"], "ploidy_index": focal["ploidy_index"]}
        )
    archive_path = (
        data_dir
        / "haplotype_intervals"
        / f"{dataset}_inferred_focal_ancestor_intervals.npz"
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_id"]
        ploidy_indices = archive["ploidy_index"]
        if sample_ids.ndim != 1 or sample_ids.dtype.kind != "U":
            raise ValueError("Interval archive sample_id must be a Unicode roster")
        for name, expected in (
            ("sample_id", haplotypes["sample_id"].to_numpy()),
            ("ploidy_index", haplotypes["ploidy_index"].to_numpy()),
        ):
            if not np.array_equal(archive[name], expected):
                raise ValueError(
                    f"Interval archive {name} disagrees with focal NPZ roster"
                )
        for name in (
            "ploidy_index",
            "offsets",
            "focal_ac",
            "left_site_index",
            "right_site_index",
            "num_sites",
            "max_ac_cutoff",
            "max_mismatches",
        ):
            if archive[name].dtype != np.dtype("int64"):
                raise ValueError(f"Interval archive {name} must use int64")
        for name in ("num_sites", "max_ac_cutoff", "max_mismatches"):
            if archive[name].shape != ():
                raise ValueError(f"Interval archive {name} must be scalar")
        if int(archive["num_sites"]) != num_sites:
            raise ValueError("Interval archive site count disagrees with panel")
        generated_ac = int(archive["max_ac_cutoff"])
        generated_budget = int(archive["max_mismatches"])
        if generated_ac < 1 or generated_budget < 0:
            raise ValueError("Interval archive generation limits are invalid")
        if max_ac < 1 or max_ac > generated_ac:
            raise ValueError(
                "Requested AC limit exceeds interval archive generation limit"
            )
        missing = sorted(set(max_mismatches) - set(range(generated_budget + 1)))
        if len(missing) > 0:
            raise ValueError(f"Interval archive lacks mismatch budgets {missing}")
        offsets = archive["offsets"]
        counts = archive["focal_ac"]
        left = archive["left_site_index"]
        right = archive["right_site_index"]
        if counts.ndim != 1 or ploidy_indices.shape != sample_ids.shape:
            raise ValueError(
                "Interval archive association/roster dimensions are invalid"
            )
        if (
            offsets.shape != (haplotypes.height + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(counts)
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError("Interval archive has invalid ragged offsets")
        shape = (len(counts), generated_budget + 1)
        if left.shape != shape or right.shape != shape:
            raise ValueError("Interval archive bounds have invalid dimensions")
        if np.any((counts < 1) | (counts > generated_ac)):
            raise ValueError(
                "Interval archive focal AC is outside its generation limit"
            )
        if np.any((left < 0) | (left >= right) | (right > num_sites)):
            raise ValueError(
                "Interval archive bounds must satisfy 0 <= left < right <= num_sites"
            )
        left = left[:, max_mismatches]
        right = right[:, max_mismatches]
    keys = ["sample_id", "ploidy_index"]
    if haplotypes.unique(keys).height != haplotypes.height:
        raise ValueError("Focal NPZ has duplicate haplotype identities")
    stats_path = (
        data_dir / "dataframes" / f"{dataset}_inferred_focal_ancestor_stats.csv"
    )
    stats = pl.scan_csv(stats_path, schema_overrides={"sample_id": pl.String})
    if "population" in stats.collect_schema():
        labels = stats.select(*keys, "population").unique().collect()
        haplotypes = haplotypes.join(labels, on=keys, how="left", validate="1:1")
    elif truth_ts_path is not None:
        haplotypes = _truth_populations(haplotypes, truth_ts_path)
    else:
        raise ValueError("Population metadata is absent; supply the original truth TS")
    invalid_labels = haplotypes.filter(
        pl.col("population").is_null() | (pl.col("population") == "")
    )
    if invalid_labels.height > 0:
        raise ValueError("Every haplotype must have a population label")
    haplotypes = haplotypes.with_row_index("haplotype_index")
    logger.info(
        "Loaded %s: %d sites, %d haplotypes, %d associations, budgets %s",
        dataset,
        num_sites,
        haplotypes.height,
        len(counts),
        max_mismatches,
    )
    return CoverageData(
        dataset=dataset,
        num_sites=num_sites,
        haplotypes=haplotypes,
        offsets=offsets,
        focal_ac=counts,
        left_site_index=left,
        right_site_index=right,
        max_ac=max_ac,
        mismatch_budgets=list(max_mismatches),
        first_position=int(positions[0]),
        last_position=int(positions[-1]),
    )


def _sweep_haplotype(task: SweepTask) -> SweepResult:
    """Sweep sorted endpoints, updating cumulative AC eligibility once per association.

    Coverage is constant between successive endpoints. Segment lengths are
    counts of sites, so histogram weights give exact order statistics without
    allocating a per-site coverage vector. Quantiles match NumPy's default
    linear interpolation on that conceptual vector, including zero coverage.
    Time is O(C log C + B E), memory O(C + E + max coverage), for C associations,
    B cutoffs and E distinct endpoints; neither depends on BP length.
    """
    endpoints = np.concatenate(([0, task.num_sites], task.left, task.right))
    endpoints = np.unique(endpoints)
    lengths = np.diff(endpoints)
    left_events = np.searchsorted(endpoints, task.left)
    right_events = np.searchsorted(endpoints, task.right)
    delta = np.zeros(len(endpoints), dtype=np.int64)
    stop = 0
    records = []
    for cutoff in task.cutoffs:
        next_stop = int(np.searchsorted(task.focal_ac, cutoff, side="right"))
        np.add.at(delta, left_events[stop:next_stop], 1)
        np.add.at(delta, right_events[stop:next_stop], -1)
        stop = next_stop
        coverage = np.cumsum(delta)[:-1]
        histogram = np.zeros(int(coverage.max()) + 1, dtype=np.int64)
        np.add.at(histogram, coverage, lengths)
        cumulative = np.cumsum(histogram)
        ranks = np.array([0.05, 0.95]) * (task.num_sites - 1)
        lower_ranks = np.floor(ranks).astype(np.int64)
        upper_ranks = np.ceil(ranks).astype(np.int64)
        lower = np.searchsorted(cumulative, lower_ranks, side="right")
        upper = np.searchsorted(cumulative, upper_ranks, side="right")
        quantiles = lower + (upper - lower) * (ranks - lower_ranks)
        covered_sites = task.num_sites - int(histogram[0])
        total_coverage = int(np.dot(coverage, lengths))
        records.append(
            {
                "haplotype_index": task.haplotype_index,
                "focal_ac": int(cutoff),
                "evaluated_sites": task.num_sites,
                "covered_sites": covered_sites,
                "mean_coverage": total_coverage / task.num_sites,
                "min_coverage": int(coverage.min()),
                "max_coverage": int(coverage.max()),
                "q05_coverage": float(quantiles[0]),
                "q95_coverage": float(quantiles[1]),
                "fraction_covered": covered_sites / task.num_sites,
            }
        )
    return SweepResult(task.haplotype_index, records)


def _sweep_tasks(data: CoverageData, budget_column: int, cutoffs: np.ndarray):
    """Sort ragged associations by exact AC, keeping chapter 5's AC >= 2 convention."""
    for index in range(data.haplotypes.height):
        start = data.offsets[index]
        stop = data.offsets[index + 1]
        counts = data.focal_ac[start:stop]
        eligible = np.flatnonzero(counts >= 2)
        order = np.argsort(counts[eligible], kind="stable")
        selected = eligible[order]
        left = data.left_site_index[start:stop, budget_column]
        right = data.right_site_index[start:stop, budget_column]
        yield SweepTask(
            index,
            data.num_sites,
            cutoffs,
            counts[selected],
            left[selected],
            right[selected],
        )


def summarise_coverage(
    data: CoverageData,
    cutoffs: np.ndarray,
    max_mismatches: int = 0,
    workers: int | None = None,
) -> pl.DataFrame:
    """Return per-haplotype statistics for inclusive maximum focal AC thresholds.

    As in the original chapter 5 analysis, AC=1 associations do not contribute.
    They remain available in the raw archive and loaded arrays.
    ``workers=None`` uses all available CPUs, capped by the haplotype count.
    Spawned workers consume independent numeric tasks through Pool queues;
    ``workers=1`` runs the same sweep locally. See :func:`_sweep_haplotype` for
    site weights and quantile definitions. Keep both ploidy copies separate.
    """
    cutoffs = np.asarray(cutoffs)
    if cutoffs.ndim != 1 or len(cutoffs) == 0:
        raise ValueError("cutoffs must be a nonempty one-dimensional array")
    if not np.issubdtype(cutoffs.dtype, np.integer):
        raise ValueError("cutoffs must be integers")
    if np.any(cutoffs < 2) or np.any(np.diff(cutoffs) <= 0):
        raise ValueError("cutoffs must be strictly increasing and >= 2")
    if max_mismatches not in data.mismatch_budgets:
        raise ValueError("Reload intervals with the requested mismatch budget")
    if cutoffs[-1] > data.max_ac:
        raise ValueError("Summary cutoffs exceed the loaded analysis AC limit")
    if workers is None:
        workers = multiprocessing.cpu_count()
    if workers < 1:
        raise ValueError("workers must be >= 1")
    workers = min(workers, data.haplotypes.height)
    budget_column = data.mismatch_budgets.index(max_mismatches)
    tasks = _sweep_tasks(data, budget_column, cutoffs)
    if workers == 1:
        results = list(map(_sweep_haplotype, tasks))
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers) as pool:
            results = list(pool.imap_unordered(_sweep_haplotype, tasks, chunksize=1))
    records = []
    for result in results:
        records.extend(result.records)
    frame = pl.DataFrame(records)
    frame = frame.join(data.haplotypes, on="haplotype_index", validate="m:1")
    frame = frame.with_columns(
        pl.lit(data.dataset).alias("dataset"),
        pl.lit(max_mismatches).alias("max_mismatches"),
    )
    return frame.sort("haplotype_index", "focal_ac").drop("haplotype_index")


@dataclasses.dataclass
class PerfResults:
    """Saved suite metadata, timings, repeat fingerprints and haplotype metrics."""

    manifest: dict
    runs: pd.DataFrame
    stages: pd.DataFrame
    quality: pd.DataFrame
    fingerprints: pd.DataFrame
    comparisons: pd.DataFrame


def perf_path_fingerprint(run: dict) -> dict:
    """Hash scientific JSONL fields in canonical haplotype and segment order."""
    records = read_matches(pathlib.Path(run["output_matches"]))
    digest = hashlib.sha256()
    for identity, record in sorted(records.items()):
        content = [identity, record["path"], record["mutations"]]
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
        digest.update(encoded.encode())
    return {
        "run_id": run["run_id"],
        "method": run["method"],
        "haplotypes": len(records),
        "sha256": digest.hexdigest(),
    }


def compare_paths(
    reference_path: pathlib.Path,
    mask_path: pathlib.Path,
    baseline_path: pathlib.Path,
    stitched_path: pathlib.Path,
    method: str,
) -> pd.DataFrame:
    """Compare each haplotype once, with within-region agreement denominators.

    Switch categories follow :func:`comparison_metrics`. The
    full-HMM path is partitioned by the same saved mask as each stitched path.
    Percent increases use each haplotype's full-HMM switch count; a zero count
    gives an undefined percentage. Empty regions give undefined agreement.
    """
    reference = tskit.load(reference_path)
    baseline = read_matches(baseline_path)
    stitched = read_matches(stitched_path)
    with np.load(mask_path, allow_pickle=False) as archive:
        mask = {name: archive[name] for name in archive.files}
    assert int(mask["num_sites"]) == reference.num_sites
    roster = list(
        zip(mask["sample_id"].tolist(), mask["ploidy_index"].tolist(), strict=True)
    )
    assert len(roster) == len(set(roster))
    assert set(roster) == set(baseline) == set(stitched)
    rows = []
    for index, identity in enumerate(roster):
        first, last = mask["offsets"][index : index + 2]
        left = mask["left_site_index"][first:last]
        right = mask["right_site_index"][first:last]
        is_gap = mask["is_gap"][first:last]
        assert left[0] == 0 and right[-1] == reference.num_sites
        assert np.all(left < right) and np.array_equal(right[:-1], left[1:])
        covered = np.zeros(reference.num_sites, dtype=bool)
        for start, stop in zip(left[~is_gap], right[~is_gap], strict=True):
            covered[start:stop] = True
        full_parents = parents_at_sites(baseline[identity], reference)
        stitched_parents = parents_at_sites(stitched[identity], reference)
        metrics = comparison_metrics(full_parents, stitched_parents, covered)
        full_switches = metrics["full_switches"]
        metrics["switch_increase_percentage"] = (
            100 * metrics["num_extra_switches"] / full_switches
            if full_switches > 0
            else np.nan
        )
        for region in ("focal", "gap"):
            sites = metrics[f"{region}_sites"]
            disagreements = metrics[f"{region}_disagreement_sites"]
            metrics[f"{region}_agreement_percentage"] = (
                100 * (sites - disagreements) / sites if sites > 0 else np.nan
            )
        rows.append(
            {
                "method": method,
                "sample_id": identity[0],
                "ploidy_index": identity[1],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def reference_phases(runs: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    """Partition measured totals using non-overlapping process wall stages.

    Worker service times are deliberately excluded. Assembly/output is folded
    into the residual along with loading/setup, so each stack equals the
    directly measured total. Full HMM is kept as a single inclusive stage.
    """
    process = stages.loc[stages.scope.eq("process")]
    process = process.pivot(index="run_id", columns="stage", values="wall_s")
    selected = runs.loc[runs["mode"].eq("from_reference")].set_index("run_id")
    rows = []
    for run_id, run in selected.iterrows():
        if run["method"] == "full_hmm":
            rows.append(
                {
                    "run_id": run_id,
                    "method": run["method"],
                    "phase": "full_pipeline",
                    "wall_s": run["wall_s"],
                }
            )
        else:
            measured = process.loc[run_id]
            matching = measured["matching"]
            preparation = measured["focal_preparation"]
            residual = run["wall_s"] - matching - preparation
            assert residual >= 0
            values = {
                "matching": matching,
                "focal_preparation": preparation,
                "other": residual,
            }
            for phase, value in values.items():
                rows.append(
                    {
                        "run_id": run_id,
                        "method": run["method"],
                        "phase": phase,
                        "wall_s": value,
                    }
                )
    return pd.DataFrame(rows)


def per_sample_data(
    worker_times: pd.DataFrame,
    mask_path: pathlib.Path,
    reference_path: pathlib.Path,
    population_path: pathlib.Path,
    settings: dict,
    resident_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Average recorded stages over trials and both haplotypes of each sample.

    Input rows contain one summed worker-stage elapsed time per haplotype and
    trial, in seconds. Physical focal coverage uses the same clipped mask-edge
    convention as :func:`coverage_percentage`.
    """
    keys = ["method", "repeat", "sample_id", "ploidy_index"]
    assert not worker_times.duplicated(keys).any()
    assert worker_times.service_s.ge(0).all()
    positions = tskit.load(reference_path).sites_position
    coverage_rows = []
    with np.load(mask_path, allow_pickle=False) as mask:
        for index, sample_id in enumerate(mask["sample_id"]):
            first, last = mask["offsets"][index : index + 2]
            percentage = coverage_percentage(
                mask["left_site_index"][first:last],
                mask["right_site_index"][first:last],
                mask["is_gap"][first:last],
                positions,
                settings["analysis_span_bp"],
            )
            coverage_rows.append(
                {
                    "sample_id": str(sample_id),
                    "ploidy_index": int(mask["ploidy_index"][index]),
                    "focal_coverage_percentage": percentage,
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    trials = resident_runs.loc[
        resident_runs.method.ne("full_hmm"), ["method", "repeat"]
    ]
    expected = trials.merge(coverage, how="cross")
    measured = expected.merge(
        worker_times, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    assert measured["_merge"].eq("both").all()
    per_haplotype = (
        measured.groupby(["method", "sample_id", "ploidy_index"])
        .agg(
            worker_time_ms=("service_s", "mean"),
            focal_coverage_percentage=("focal_coverage_percentage", "first"),
        )
        .reset_index()
    )
    per_haplotype["worker_time_ms"] *= 1000
    samples = (
        per_haplotype.groupby(["method", "sample_id"])
        .agg(
            worker_time_ms=("worker_time_ms", "mean"),
            focal_coverage_percentage=("focal_coverage_percentage", "mean"),
            haplotypes=("ploidy_index", "nunique"),
        )
        .reset_index()
    )
    assert samples.haplotypes.eq(2).all()
    populations = pd.read_csv(population_path)
    populations = populations[["sample_id", settings["population_column"]]]
    populations = populations.rename(
        columns={settings["population_column"]: "population"}
    )
    samples = samples.merge(
        populations, on="sample_id", how="left", validate="many_to_one"
    )
    assert samples.population.notna().all()
    return samples


def perf_load_results(config_path: dict) -> PerfResults:
    """Load the saved suite, verify repeat invariance, and derive correctness."""
    config = config_path.copy()
    root = resolve_path(config["benchmark_dir"], config_path["base_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    runs = pd.read_csv(root / "runs.csv")
    runs = runs.loc[runs.dataset.eq(config["dataset"])].copy()
    stages = pd.read_csv(root / "stages.csv")
    stages = stages.loc[stages.run_id.isin(runs.run_id)].copy()
    quality = pd.DataFrame(json.loads((root / "quality.json").read_text()))
    quality = quality.loc[quality.run_id.isin(runs.run_id)].copy()
    expected = manifest["configuration"]["performance"]
    expected_trials = len(expected["methods"]) * len(expected["modes"])
    expected_trials *= len(expected["worker_counts"]) * expected["repeats"]
    assert len(runs) == expected_trials and (not runs.run_id.duplicated().any())
    assert runs.status.eq("ok").all()
    assert set(runs.method) == set(config["methods"])
    assert len(quality) == len(runs) and quality.status.eq("ok").all()
    assert quality.reconstructed_called_allele_errors.eq(0).all()
    assert quality.mismatches.eq(0).all()
    assert runs.actual_workers.nunique() == 1
    assert runs.matched_haplotypes.nunique() == 1
    jobs = runs.to_dict("records")
    workers = config["workers"]
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        fingerprints = pd.DataFrame(pool.map(perf_path_fingerprint, jobs))
    assert fingerprints.groupby("method").sha256.nunique().eq(1).all()
    assert fingerprints.haplotypes.eq(runs.matched_haplotypes.iloc[0]).all()
    selection = config["correctness"]
    chosen = runs.loc[runs["mode"].eq(selection["mode"])]
    chosen = chosen.loc[chosen["repeat"].eq(selection["repeat"])].set_index("method")
    assert len(chosen) == len(config["methods"])
    inputs = manifest["datasets"][config["dataset"]]["inputs"]
    reference_path = pathlib.Path(inputs["reference"])
    baseline_path = pathlib.Path(chosen.loc["full_hmm", "output_matches"])
    tables = []
    for method in config["methods"]:
        if method == "full_hmm":
            continue
        run = chosen.loc[method]
        mask_path = pathlib.Path(run["output_matches"]).parent / "focal_mask.npz"
        with np.load(mask_path, allow_pickle=False) as trial_mask:
            with np.load(inputs["mask"], allow_pickle=False) as common_mask:
                for name in common_mask.files:
                    assert np.array_equal(trial_mask[name], common_mask[name])
        table = compare_paths(
            reference_path,
            mask_path,
            baseline_path,
            pathlib.Path(run["output_matches"]),
            method,
        )
        report = quality.loc[quality.run_id.eq(run["run_id"])].iloc[0]
        assert table.stitched_switches.sum() == report["switches"]
        disagreements = table.focal_disagreement_sites + table.gap_disagreement_sites
        sites = table.focal_sites + table.gap_sites
        assert sites.sum() - disagreements.sum() == report["parent_agreement_sites"]
        tables.append(table)
    return PerfResults(
        manifest,
        runs,
        stages,
        quality,
        fingerprints,
        pd.concat(tables, ignore_index=True),
    )


def min_sizes_load_configuration(config_path: dict) -> dict:
    """Resolve the analysis settings and the current experiment grid."""
    config = config_path.copy()
    experiment_path = resolve_path(config["experiment_config"], config_path["base_dir"])
    experiment_base_dir = config.get("experiment_base_dir", experiment_path.parent)
    with experiment_path.open() as source:
        experiment = yaml.safe_load(source)
    assert experiment["max_mismatches"] == [0]
    assert config["method"] in experiment["focal_methods"]
    config["experiment"] = experiment
    config["experiment_path"] = experiment_path
    config["data_dir"] = resolve_path(experiment["data_dir"], experiment_base_dir)
    return config


def min_sizes_load_results(
    config_path: dict, workers: int | None = None
) -> HyperparamsResults:
    """Recompute every dataset/cutoff/minimum combination in parallel."""
    config = min_sizes_load_configuration(config_path)
    experiment = config["experiment"]
    jobs = []
    minimums = []
    for settings in config["datasets"]:
        labels = population_labels(settings, config_path)
        grid = itertools.product(
            experiment["ac_cutoff"], experiment["min_focal_region_size"]
        )
        for cutoff, minimum in grid:
            jobs.append(
                RunInputs(
                    config["data_dir"],
                    settings["name"],
                    labels,
                    settings["african_population"],
                    settings["analysis_span_bp"],
                    cutoff,
                    experiment["max_mismatches"][0],
                    config["method"],
                    experiment["focal_choice"],
                )
            )
            minimums.append(minimum)
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        results = list(pool.map(summarise_run, jobs, minimums))
    comparisons = pd.concat(
        [result.comparisons for result in results], ignore_index=True
    )
    lengths = pd.concat([result.lengths for result in results], ignore_index=True)
    return HyperparamsResults(comparisons, lengths)


def min_sizes_load_tree_sequences(config_path: dict) -> TreeSequences:
    """Load raw trees with stitched keys (dataset, cutoff, minimum size)."""
    config = min_sizes_load_configuration(config_path)
    experiment = config["experiment"]
    full = {}
    stitched = {}
    rows = []
    for settings in config["datasets"]:
        name = settings["name"]
        full[name] = tskit.load(
            config["data_dir"] / "matches" / f"{name}_inferred_samples_raw.trees"
        )
        rows.append(
            {
                "dataset": name,
                "method": "full",
                "ac_cutoff": None,
                "max_mismatches": None,
                "min_focal_region_size": None,
                "num_edges": full[name].num_edges,
                "num_trees": full[name].num_trees,
            }
        )
        grid = itertools.product(
            experiment["ac_cutoff"], experiment["min_focal_region_size"]
        )
        for cutoff, minimum in grid:
            budget = experiment["max_mismatches"][0]
            prefix = (
                f"{name}_inferred_ac{cutoff}_k{budget}_min{minimum}_{config['method']}"
            )
            path = (
                config["data_dir"] / "matches" / f"{prefix}_stitched_samples_raw.trees"
            )
            ts = tskit.load(path)
            assert np.array_equal(ts.samples(), full[name].samples())
            stitched[name, cutoff, minimum] = ts
            rows.append(
                {
                    "dataset": name,
                    "method": "stitched",
                    "ac_cutoff": cutoff,
                    "max_mismatches": budget,
                    "min_focal_region_size": minimum,
                    "num_edges": ts.num_edges,
                    "num_trees": ts.num_trees,
                }
            )
    return TreeSequences(full, stitched, pd.DataFrame(rows))


def min_sizes_arg_metrics(
    trees: TreeSequences, settings: list[dict], workers: int | None = None
) -> pd.DataFrame:
    """Compare each minimum-size Greedy ARG against its Full HMM output."""
    spans = {item["name"]: item["analysis_span_bp"] for item in settings}
    jobs = []
    minimums = []
    for (name, cutoff, minimum), comparison in trees.stitched.items():
        aligned = align_tree_sequences(comparison, trees.full[name], spans[name])
        jobs.append(
            ArgComparison(
                name,
                cutoff,
                0,
                aligned.comparison,
                aligned.reference,
                "stitched_greedy",
                "full_hmm",
            )
        )
        minimums.append(minimum)
    metrics = run_arg_comparisons(jobs, workers)
    metrics["min_focal_region_size"] = minimums
    return metrics


def min_sizes_truth_arg_metrics(
    trees: TreeSequences, config_path: dict, workers: int | None = None
) -> pd.DataFrame:
    """Compare the simulated Full HMM and every Greedy case against truth."""
    config = min_sizes_load_configuration(config_path)
    experiment_datasets = {
        item["name"]: item for item in config["experiment"]["datasets"]
    }
    jobs = []
    minimums = []
    for settings in config["datasets"]:
        name = settings["name"]
        truth_path = experiment_datasets[name]["ts_path"]
        if truth_path is None:
            continue
        truth_path = resolve_path(truth_path, config["experiment_path"].parent)
        truth = tskit.load(truth_path)
        reference_nodes = truth_sample_nodes(truth)
        aligned = align_tree_sequences(
            trees.full[name],
            truth,
            settings["analysis_span_bp"],
            reference_nodes=reference_nodes,
        )
        jobs.append(
            ArgComparison(
                name,
                None,
                None,
                aligned.comparison,
                aligned.reference,
                "full_hmm",
                "true_arg",
            )
        )
        minimums.append(None)
        for (dataset, cutoff, minimum), comparison in trees.stitched.items():
            if dataset != name:
                continue
            aligned = align_tree_sequences(
                comparison,
                truth,
                settings["analysis_span_bp"],
                reference_nodes=reference_nodes,
            )
            jobs.append(
                ArgComparison(
                    name,
                    cutoff,
                    0,
                    aligned.comparison,
                    aligned.reference,
                    "stitched_greedy",
                    "true_arg",
                )
            )
            minimums.append(minimum)
    metrics = run_arg_comparisons(jobs, workers)
    metrics["min_focal_region_size"] = minimums
    return metrics


@dataclasses.dataclass
class RunInputs:
    """Paths, population labels and parameters for one independent saved run."""

    data_dir: pathlib.Path
    dataset: str
    labels: dict
    african_population: str
    analysis_span_bp: list[float]
    ac_cutoff: int
    max_mismatches: int
    method: str
    focal_choice: str


@dataclasses.dataclass
class HyperparamsResults:
    """Per-haplotype comparisons and pooled connected-component lengths."""

    comparisons: pd.DataFrame
    lengths: pd.DataFrame


@dataclasses.dataclass
class TreeSequences:
    """All raw Full HMM and stitched matched outputs for both datasets."""

    full: dict[str, tskit.TreeSequence]
    stitched: dict[tuple[str, int, int], tskit.TreeSequence]
    statistics: pd.DataFrame


@dataclasses.dataclass
class AlignedTrees:
    """Trees with shared sample identities, node IDs and genomic coordinates."""

    comparison: tskit.TreeSequence
    reference: tskit.TreeSequence


@dataclasses.dataclass
class ArgComparison:
    """An aligned pair and method labels for one worker's ARG calculation."""

    dataset: str
    ac_cutoff: int | None
    max_mismatches: int | None
    comparison: tskit.TreeSequence
    reference: tskit.TreeSequence
    comparison_method: str
    reference_method: str


def resolve_path(value: str, base: pathlib.Path) -> pathlib.Path:
    """Resolve a configuration path relative to its owning configuration."""
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def population_labels(settings: dict, config_path: dict) -> dict:
    """Read preserved simulation labels or 1KGP pedigree superpopulations."""
    path = resolve_path(settings["population_metadata"], config_path["base_dir"])
    metadata = pd.read_csv(path)
    assert not metadata.sample_id.duplicated().any()
    labels = {}
    for row in metadata.to_dict("records"):
        for ploidy_index in range(2):
            labels[row["sample_id"], ploidy_index] = row[settings["population_column"]]
    return labels


def read_matches(path: pathlib.Path) -> dict:
    """Read unique sample paths; baseline paths are emitted right to left."""
    records = {}
    with path.open() as file:
        for line in file:
            record = json.loads(line)
            assert record["source"] == "samples"
            key = (record["sample_id"], record["ploidy_index"])
            assert key not in records
            record["path"].sort(key=lambda segment: segment["left"])
            records[key] = record
    return records


def parents_at_sites(record: dict, reference: tskit.TreeSequence) -> np.ndarray:
    """Evaluate a complete half-open path in original reference node IDs."""
    path = record["path"]
    left = np.array([segment["left"] for segment in path])
    right = np.array([segment["right"] for segment in path])
    parents = np.array([segment["parent"] for segment in path], dtype=np.int32)
    assert left[0] == 0 and right[-1] == reference.sequence_length
    assert np.all(left < right) and np.array_equal(right[:-1], left[1:])
    assert np.all(parents[1:] != parents[:-1])
    assert np.all((parents >= 0) & (parents < reference.num_nodes))
    segments = np.searchsorted(right, reference.sites_position, side="right")
    result = parents[segments]
    assert np.count_nonzero(result[1:] != result[:-1]) == len(path) - 1
    return result


def coverage_percentage(
    left: np.ndarray,
    right: np.ndarray,
    is_gap: np.ndarray,
    positions: np.ndarray,
    span: list[float],
) -> float:
    """Measure physical coverage, clipping native run edges to the analysis span.

    As in match-eval's path assembly, an internal edge is the position of its
    right-hand site. Terminal edges use the window bounds, excluding the
    unanalysed chromosome tail retained by the reference tree sequence.
    """
    edges = np.concatenate(([span[0]], positions[1:], [span[1]]))
    edges = np.clip(edges, span[0], span[1])
    widths = edges[right] - edges[left]
    return float(100 * widths[~is_gap].sum() / (span[1] - span[0]))


def summarise_run(
    inputs: RunInputs, min_focal_region_size: int | None = None
) -> HyperparamsResults:
    """Validate a saved run, optionally selecting a minimum-size focal mask.

    Derive counts, within-region agreement and lengths with the same metric
    definitions for the hyperparameter and minimum-size analyses.
    """
    name = inputs.dataset
    reference = tskit.load(
        inputs.data_dir / "ancestors" / f"{name}_inferred_ancestors.trees"
    )
    positions = reference.sites_position
    assert inputs.analysis_span_bp[0] <= positions[0]
    assert positions[-1] < inputs.analysis_span_bp[1]
    full = read_matches(
        inputs.data_dir / "matches" / f"{name}_inferred_samples_matches.jsonl"
    )
    prefix = f"{name}_inferred_ac{inputs.ac_cutoff}_k{inputs.max_mismatches}"
    if min_focal_region_size is not None:
        prefix += f"_min{min_focal_region_size}"
    stitched_path = (
        inputs.data_dir
        / "matches"
        / f"{prefix}_{inputs.method}_stitched_samples_matches.jsonl"
    )
    stitched = read_matches(stitched_path)
    mask_path = inputs.data_dir / "focal_masks" / f"{prefix}_focal_mask.npz"
    with np.load(mask_path, allow_pickle=False) as archive:
        mask = {field: archive[field] for field in archive.files}
    assert int(mask["num_sites"]) == len(positions)
    assert int(mask["ac_cutoff"]) == inputs.ac_cutoff
    assert int(mask["max_mismatches"]) == inputs.max_mismatches
    assert str(mask["focal_choice"]) == inputs.focal_choice
    if min_focal_region_size is not None:
        assert int(mask["min_focal_region_size"]) == min_focal_region_size
    roster = list(
        zip(mask["sample_id"].tolist(), mask["ploidy_index"].tolist(), strict=True)
    )
    assert len(roster) == len(set(roster))
    assert set(roster) == set(full) == set(stitched)
    rows = []
    length_tables = []
    for index, key in enumerate(roster):
        record = stitched[key]
        assert record["ac_cutoff"] == inputs.ac_cutoff
        assert record["max_mismatches"] == inputs.max_mismatches
        assert record["method"] == inputs.method
        start, stop = mask["offsets"][index : index + 2]
        left = mask["left_site_index"][start:stop]
        right = mask["right_site_index"][start:stop]
        is_gap = mask["is_gap"][start:stop]
        assert left[0] == 0 and right[-1] == len(positions)
        assert np.all(left < right) and np.array_equal(right[:-1], left[1:])
        assert np.all(is_gap[1:] != is_gap[:-1])
        covered = np.zeros(len(positions), dtype=bool)
        for first, last in zip(left[~is_gap], right[~is_gap], strict=True):
            covered[first:last] = True
        full_parents = parents_at_sites(full[key], reference)
        stitched_parents = parents_at_sites(record, reference)
        metrics = comparison_metrics(full_parents, stitched_parents, covered)
        denominator = metrics["full_switches"]
        metrics["switch_increase_percentage"] = (
            100 * metrics["num_extra_switches"] / denominator
            if denominator > 0
            else np.nan
        )
        for region in ["focal", "gap"]:
            sites = metrics[f"{region}_sites"]
            disagreements = metrics[f"{region}_disagreement_sites"]
            metrics[f"{region}_agreement_percentage"] = (
                100 * (sites - disagreements) / sites if sites > 0 else np.nan
            )
        metrics["focal_coverage_percentage"] = coverage_percentage(
            left, right, is_gap, positions, inputs.analysis_span_bp
        )
        metrics["gap_coverage_percentage"] = 100 - metrics["focal_coverage_percentage"]
        identity = {
            "dataset": name,
            "sample_id": key[0],
            "ploidy_index": key[1],
            "ac_cutoff": inputs.ac_cutoff,
            "max_mismatches": inputs.max_mismatches,
            "method": inputs.method,
        }
        if min_focal_region_size is not None:
            identity["min_focal_region_size"] = min_focal_region_size
        population = inputs.labels[key]
        assert population is not None and population != ""
        group = "African" if population == inputs.african_population else "Non-African"
        rows.append(
            {**identity, "population": population, "population_group": group, **metrics}
        )
        lengths = pd.DataFrame(
            {"num_sites": right - left, "region": np.where(is_gap, "gap", "focal")}
        )
        for field, value in identity.items():
            lengths[field] = value
        length_tables.append(lengths)
    return HyperparamsResults(
        pd.DataFrame(rows), pd.concat(length_tables, ignore_index=True)
    )


def hyperparams_load_results(
    config_path: dict, workers: int | None = None
) -> HyperparamsResults:
    """Read all configured HMM products in parallel without rerunning inference."""
    config = config_path.copy()
    experiment_path = resolve_path(config["experiment_config"], config_path["base_dir"])
    experiment_base_dir = config.get("experiment_base_dir", experiment_path.parent)
    with experiment_path.open() as file:
        experiment = yaml.safe_load(file)
    assert config["method"] in experiment["focal_methods"]
    data_dir = resolve_path(experiment["data_dir"], experiment_base_dir)
    jobs = []
    for settings in config["datasets"]:
        labels = population_labels(settings, config_path)
        grid = itertools.product(experiment["ac_cutoff"], experiment["max_mismatches"])
        for cutoff, mismatches in grid:
            jobs.append(
                RunInputs(
                    data_dir,
                    settings["name"],
                    labels,
                    settings["african_population"],
                    settings["analysis_span_bp"],
                    cutoff,
                    mismatches,
                    config["method"],
                    experiment["focal_choice"],
                )
            )
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    if workers == 1:
        results = [summarise_run(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            results = list(pool.map(summarise_run, jobs))
    comparisons = pd.concat(
        [result.comparisons for result in results], ignore_index=True
    )
    lengths = pd.concat([result.lengths for result in results], ignore_index=True)
    return HyperparamsResults(comparisons, lengths)


def hyperparams_load_tree_sequences(config_path: dict) -> TreeSequences:
    """Load saved raw TS outputs and report their unmodified edge/tree counts.

    :func:`hyperparams_arg_metrics` prepares private sample-aligned, window-restricted
    versions for Full HMM comparison. Statistics here describe the saved files,
    including the raw reference ancestry, rather than those prepared versions.
    """
    config = config_path.copy()
    experiment_path = resolve_path(config["experiment_config"], config_path["base_dir"])
    experiment_base_dir = config.get("experiment_base_dir", experiment_path.parent)
    with experiment_path.open() as file:
        experiment = yaml.safe_load(file)
    data_dir = resolve_path(experiment["data_dir"], experiment_base_dir)
    full = {}
    stitched = {}
    rows = []
    for settings in config["datasets"]:
        name = settings["name"]
        full[name] = tskit.load(
            data_dir / "matches" / f"{name}_inferred_samples_raw.trees"
        )
        rows.append(
            {
                "dataset": name,
                "method": "full",
                "ac_cutoff": None,
                "max_mismatches": None,
                "num_edges": full[name].num_edges,
                "num_trees": full[name].num_trees,
            }
        )
        grid = itertools.product(experiment["ac_cutoff"], experiment["max_mismatches"])
        for cutoff, allowance in grid:
            prefix = f"{name}_inferred_ac{cutoff}_k{allowance}_{config['method']}"
            path = data_dir / "matches" / f"{prefix}_stitched_samples_raw.trees"
            ts = tskit.load(path)
            stitched[name, cutoff, allowance] = ts
            assert np.array_equal(ts.samples(), full[name].samples())
            rows.append(
                {
                    "dataset": name,
                    "method": "stitched",
                    "ac_cutoff": cutoff,
                    "max_mismatches": allowance,
                    "num_edges": ts.num_edges,
                    "num_trees": ts.num_trees,
                }
            )
    return TreeSequences(full, stitched, pd.DataFrame(rows))


def tree_statistic_differences(statistics: pd.DataFrame) -> pd.DataFrame:
    """Subtract each dataset's raw Full HMM counts from its stitched counts."""
    metrics = ["num_edges", "num_trees"]
    full = statistics.loc[statistics.method.eq("full"), ["dataset", *metrics]]
    full = full.set_index("dataset")
    stitched = statistics.loc[statistics.method.eq("stitched")].copy()
    differences = stitched.join(
        full, on="dataset", rsuffix="_full", validate="many_to_one"
    )
    for metric in metrics:
        differences[f"{metric}_difference"] = (
            differences[metric] - differences[f"{metric}_full"]
        )
    return differences


def align_tree_sequences(
    comparison: tskit.TreeSequence,
    reference: tskit.TreeSequence,
    span: list[float],
    reference_nodes: dict | None = None,
) -> AlignedTrees:
    """Align matched sample metadata identities and retain the analysed window.

    Raw matched node metadata identifies samples by sample ID and ploidy index.
    Simplifying in comparison haplotype order maps identical biological samples
    to identical node IDs in both sequences. Unary ancestral nodes are retained
    because ARG metrics measure node spans.

    A simulation-truth reference uses the explicit identity map returned by
    :func:`truth_sample_nodes`; matched references use node metadata by default.
    """
    if reference_nodes is None:
        reference_nodes = {}
        for node_id in reference.samples():
            metadata = reference.node(int(node_id)).metadata
            identity = (metadata["sample_id"], metadata["ploidy_index"])
            reference_nodes[identity] = int(node_id)
    if len(reference_nodes) != reference.num_samples:
        raise ValueError("Reference TS sample haplotype identities must be unique")
    comparison_samples = comparison.samples()
    matched_reference_nodes = []
    identities = []
    for node_id in comparison_samples:
        metadata = comparison.node(int(node_id)).metadata
        identity = (metadata["sample_id"], metadata["ploidy_index"])
        identities.append(identity)
        matched_reference_nodes.append(reference_nodes[identity])
    if len(identities) != len(set(identities)):
        raise ValueError("Matched TS sample haplotype identities must be unique")
    if set(matched_reference_nodes) != set(reference.samples()):
        raise ValueError(
            "Comparison and reference TS must contain the same sample haplotypes"
        )
    comparison_window = window_tree_sequence(comparison, span)
    reference_window = window_tree_sequence(reference, span)
    comparison_aligned = comparison_window.simplify(
        samples=comparison_samples, keep_unary=True
    )
    reference_aligned = reference_window.simplify(
        samples=matched_reference_nodes, keep_unary=True
    )
    assert (
        comparison_aligned.sequence_length
        == reference_aligned.sequence_length
        == span[1] - span[0]
    )
    assert np.array_equal(comparison_aligned.samples(), reference_aligned.samples())
    return AlignedTrees(comparison_aligned, reference_aligned)


def window_tree_sequence(
    ts: tskit.TreeSequence, span: list[float]
) -> tskit.TreeSequence:
    """Translate the selected window to zero without trimming missing flanks.

    An output can have no ancestry at the window's terminal positions.
    Trimming to edge bounds would give compared outputs different coordinate
    systems. Keep the explicit analysis bounds, including isolated-sample spans.
    """
    window = ts.keep_intervals([span], simplify=False)
    tables = window.dump_tables()
    tables.edges.left = tables.edges.left - span[0]
    tables.edges.right = tables.edges.right - span[0]
    tables.sites.position = tables.sites.position - span[0]
    tables.sequence_length = span[1] - span[0]
    return tables.tree_sequence()


def compare_arg(inputs: ArgComparison) -> dict:
    """Compute unsupported comparison span and recovered reference span.

    The comparison TS is the focal argument. ARF is its unsupported node-span
    proportion, while TPR is the reference node-span proportion represented.
    """
    result = tscompare.haplotype_arf(inputs.comparison, inputs.reference)
    return {
        "dataset": inputs.dataset,
        "ac_cutoff": inputs.ac_cutoff,
        "max_mismatches": inputs.max_mismatches,
        "arf": float(result.arf),
        "tpr": float(result.tpr),
        "comparison_method": inputs.comparison_method,
        "reference_method": inputs.reference_method,
        "comparison_time_units": inputs.comparison.time_units,
        "reference_time_units": inputs.reference.time_units,
    }


def hyperparams_arg_metrics(
    trees: TreeSequences, settings: list[dict], workers: int | None = None
) -> pd.DataFrame:
    """Compare every stitched ARG with its dataset's Full HMM in parallel."""
    spans = {item["name"]: item["analysis_span_bp"] for item in settings}
    jobs = []
    for (name, cutoff, allowance), comparison in trees.stitched.items():
        aligned = align_tree_sequences(comparison, trees.full[name], spans[name])
        jobs.append(
            ArgComparison(
                name,
                cutoff,
                allowance,
                aligned.comparison,
                aligned.reference,
                "stitched_hmm",
                "full_hmm",
            )
        )
    return run_arg_comparisons(jobs, workers)


def run_arg_comparisons(
    jobs: list[ArgComparison], workers: int | None = None
) -> pd.DataFrame:
    """Evaluate independently aligned ARG comparisons across available CPUs."""
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    if workers == 1:
        rows = [compare_arg(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(compare_arg, jobs))
    return pd.DataFrame(rows)


def truth_sample_nodes(truth: tskit.TreeSequence) -> dict:
    """Join truth node IDs to the VCF haplotype identities used by match-eval."""
    mapping = truth.map_to_vcf_model()
    nodes_by_identity = {}
    for name, nodes in zip(
        mapping.individuals_name, mapping.individuals_nodes, strict=True
    ):
        for ploidy_index, node_id in enumerate(nodes):
            nodes_by_identity[name, ploidy_index] = int(node_id)
    return nodes_by_identity


def hyperparams_truth_arg_metrics(
    trees: TreeSequences, config_path: dict, workers: int | None = None
) -> pd.DataFrame:
    """Compare simulated Full HMM and every stitched run with the true ARG.

    The Full HMM baseline is evaluated once at the configured display cutoff;
    its mismatch allowance is undefined because it does not use focal masks.
    """
    config = config_path.copy()
    experiment_path = resolve_path(config["experiment_config"], config_path["base_dir"])
    experiment_base_dir = config.get("experiment_base_dir", experiment_path.parent)
    with experiment_path.open() as file:
        experiment = yaml.safe_load(file)
    dataset_configs = {item["name"]: item for item in experiment["datasets"]}
    jobs = []
    for settings in config["datasets"]:
        name = settings["name"]
        truth_path = dataset_configs[name]["ts_path"]
        if truth_path is None:
            continue
        truth = tskit.load(resolve_path(truth_path, experiment_base_dir))
        reference_nodes = truth_sample_nodes(truth)
        aligned = align_tree_sequences(
            trees.full[name],
            truth,
            settings["analysis_span_bp"],
            reference_nodes=reference_nodes,
        )
        display_cutoff = config["truth_comparison"]["full_hmm_cutoff"]
        jobs.append(
            ArgComparison(
                name,
                display_cutoff,
                None,
                aligned.comparison,
                aligned.reference,
                "full_hmm",
                "true_arg",
            )
        )
        for (dataset, cutoff, allowance), comparison in trees.stitched.items():
            if dataset != name:
                continue
            aligned = align_tree_sequences(
                comparison,
                truth,
                settings["analysis_span_bp"],
                reference_nodes=reference_nodes,
            )
            jobs.append(
                ArgComparison(
                    name,
                    cutoff,
                    allowance,
                    aligned.comparison,
                    aligned.reference,
                    "stitched_hmm",
                    "true_arg",
                )
            )
    return run_arg_comparisons(jobs, workers)


@dataclasses.dataclass
class GapParentJob:
    root: pathlib.Path
    cache: pathlib.Path
    dataset: dict
    choice: str
    expected_eligible: dict


def focal_seed_sites(reference: tskit.TreeSequence, choice: str) -> np.ndarray:
    """Recover original focal sites under the checked single-mutation invariant."""
    counts = np.bincount(reference.mutations_site, minlength=reference.num_sites)
    assert np.all(counts == 1), (
        "Reference focal-site recovery requires one mutation per site"
    )
    if choice == "left":
        seeds = np.full(reference.num_nodes, reference.num_sites, dtype=np.int64)
        np.minimum.at(seeds, reference.mutations_node, reference.mutations_site)
        seeds[seeds == reference.num_sites] = -1
    else:
        assert choice == "right"
        seeds = np.full(reference.num_nodes, -1, dtype=np.int64)
        np.maximum.at(seeds, reference.mutations_node, reference.mutations_site)
    return seeds


def gap_fragments(
    left, right, parents, gap_left, gap_right, positions, analysis_intervals
):
    """Aggregate edge/gap intersections by parent for one sample haplotype.

    Site counts include the last inference site. Physical spans are separately
    clipped to the archived analysis intervals, whose final endpoint is the
    last observed position. Touching edges with the same parent in one gap
    count as one copying segment.
    """
    boundaries = np.unique(np.concatenate([left, right, gap_left, gap_right]))
    first = boundaries[:-1]
    last = boundaries[1:]
    edge = np.searchsorted(left, first, side="right") - 1
    gap = np.searchsorted(gap_left, first, side="right") - 1
    valid = (edge >= 0) & (gap >= 0)
    valid &= first < right[np.maximum(edge, 0)]
    valid &= first < gap_right[np.maximum(gap, 0)]
    first, last, edge, gap = (first[valid], last[valid], edge[valid], gap[valid])
    copied = parents[edge]
    sites = np.searchsorted(positions, last) - np.searchsorted(positions, first)
    span = np.zeros(len(first), dtype=float)
    for lo, hi in analysis_intervals:
        span += np.maximum(0, np.minimum(last, hi) - np.maximum(first, lo))
    segments = np.ones(len(first), dtype=np.int64)
    segments[1:] = (
        (copied[1:] != copied[:-1]) | (gap[1:] != gap[:-1]) | (first[1:] != last[:-1])
    )
    unique, inverse = np.unique(copied, return_inverse=True)
    return pd.DataFrame(
        {
            "parent": unique,
            "sites": np.bincount(inverse, weights=sites),
            "span_bp": np.bincount(inverse, weights=span),
            "segments": np.bincount(inverse, weights=segments),
            "distinct_parents": np.ones(len(unique), dtype=int),
        }
    )


def copying_in_gaps(ts, reference, mask, positions, analysis_intervals):
    """Index sample edges once, then intersect each path with its own gap mask."""
    sample_nodes = ts.samples()
    np.testing.assert_array_equal(
        sample_nodes, np.arange(reference.num_nodes, ts.num_nodes)
    )
    np.testing.assert_array_equal(ts.sites_position, positions)
    expected = list(
        zip(mask["source"], mask["sample_id"], mask["ploidy_index"], strict=True)
    )
    observed = []
    for node_id in sample_nodes:
        metadata = ts.node(int(node_id)).metadata
        observed.append(
            (metadata["source"], metadata["sample_id"], metadata["ploidy_index"])
        )
    assert observed == expected
    selected = np.flatnonzero(ts.edges_child >= reference.num_nodes)
    assert np.all(ts.edges_parent[selected] < reference.num_nodes)
    order = np.lexsort((ts.edges_left[selected], ts.edges_child[selected]))
    selected = selected[order]
    children = ts.edges_child[selected]
    offsets = np.searchsorted(
        children, np.arange(reference.num_nodes, ts.num_nodes + 1)
    )
    left, right, parents = (
        ts.edges_left[selected],
        ts.edges_right[selected],
        ts.edges_parent[selected],
    )
    rows = []
    for haplotype in range(len(sample_nodes)):
        begin, end = mask["offsets"][haplotype : haplotype + 2]
        gaps = mask["is_gap"][begin:end]
        if not gaps.any():
            continue
        start_site = mask["left_site_index"][begin:end][gaps]
        stop_site = mask["right_site_index"][begin:end][gaps]
        gap_left = np.zeros(len(start_site), dtype=float)
        gap_left[start_site > 0] = positions[start_site[start_site > 0]]
        gap_right = np.full(len(stop_site), reference.sequence_length, dtype=float)
        gap_right[stop_site < len(positions)] = positions[
            stop_site[stop_site < len(positions)]
        ]
        first, last = offsets[haplotype : haplotype + 2]
        row = gap_fragments(
            left[first:last],
            right[first:last],
            parents[first:last],
            gap_left,
            gap_right,
            positions,
            analysis_intervals,
        )
        assert row.sites.sum() == np.sum(stop_site - start_site)
        row["haplotype_index"] = haplotype
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def carried_focal_seeds(full, reference, seeds, sample_indices, parents):
    """Decode only queried sample/seed pairs, including sample-mutation overrides.

    Reference seeds each have one origin. Descendancy from that origin gives
    the allele unless a sample-stage mutation overrides it. This handles both
    back mutations and derived alleles introduced on a terminal sample edge.
    """
    assert np.all(seeds[parents] >= 0)
    sites = seeds[parents]
    sample_nodes = full.samples()[sample_indices]
    carried = np.zeros(len(parents), dtype=bool)
    order = np.argsort(sites, kind="stable")
    tree = full.first()
    for index in order:
        site = sites[index]
        tree.seek(reference.sites_position[site])
        carried[index] = tree.is_descendant(
            int(sample_nodes[index]), int(parents[index])
        )
    terminal = full.mutations_node >= reference.num_nodes
    mutation_sites = full.mutations_site[terminal]
    mutation_samples = full.mutations_node[terminal] - reference.num_nodes
    keys = mutation_samples.astype(np.int64) * reference.num_sites + mutation_sites
    assert len(keys) == len(np.unique(keys))
    order = np.argsort(keys)
    keys = keys[order]
    full_states = tskit.unpack_strings(
        full.tables.mutations.derived_state, full.tables.mutations.derived_state_offset
    )
    full_states = np.asarray(full_states, dtype=str)[terminal][order]
    ref_states = tskit.unpack_strings(
        reference.tables.mutations.derived_state,
        reference.tables.mutations.derived_state_offset,
    )
    ref_states = np.asarray(ref_states, dtype=str)
    query_keys = sample_indices.astype(np.int64) * reference.num_sites + sites
    indices = np.searchsorted(keys, query_keys)
    present = indices < len(keys)
    present[present] &= keys[indices[present]] == query_keys[present]
    carried[present] = full_states[indices[present]] == ref_states[sites[present]]
    return carried


def gap_parents_analyse_dataset(job: GapParentJob) -> pd.DataFrame:
    """Cache per-haplotype shares for Full HMM and stitched under identical masks."""
    paths = [job.root / job.dataset["reference"], job.root / job.dataset["full_hmm"]]
    for run in job.dataset["runs"]:
        paths.extend([job.root / run["stitched"], job.root / run["mask"]])
    helpers = [focal_seed_sites, gap_fragments, copying_in_gaps, carried_focal_seeds]
    settings = {
        "focal_choice": job.choice,
        "expected_eligible": job.expected_eligible,
        "helper_code": [inspect.getsource(function) for function in helpers],
    }
    stamp = signature(paths, settings, gap_parents_analyse_dataset)
    output = job.cache / "gap_parents" / f"{job.dataset['name']}.csv"
    cached = read_cached(output, stamp)
    if cached is not None:
        return cached
    reference = tszip.decompress(paths[0])
    full = tszip.decompress(paths[1])
    seeds = focal_seed_sites(reference, job.choice)
    masks = {}
    tables = []
    for run in job.dataset["runs"]:
        cutoff = run["af_cutoff"]
        recent = reference.nodes_time <= cutoff
        assert np.all(seeds[recent] >= 0)
        assert int(recent.sum()) == job.expected_eligible[cutoff]
        masks[cutoff] = dict(np.load(job.root / run["mask"]))
        for method in ["full_hmm", "greedy"]:
            matched = (
                full
                if method == "full_hmm"
                else tszip.decompress(job.root / run["stitched"])
            )
            row = copying_in_gaps(
                matched,
                reference,
                masks[cutoff],
                reference.sites_position,
                job.dataset["analysis_intervals"],
            )
            row["method"] = method
            row["af_cutoff"] = cutoff
            tables.append(row)
            if method == "greedy":
                del matched
    copied = pd.concat(tables, ignore_index=True)
    times = reference.nodes_time[copied.parent.to_numpy()]
    recent = times <= copied.af_cutoff.to_numpy()
    keys = (
        copied.haplotype_index.to_numpy(dtype=np.int64) * reference.num_nodes
        + copied.parent.to_numpy()
    )
    unique, inverse = np.unique(keys[recent], return_inverse=True)
    sample_indices = unique // reference.num_nodes
    parents = unique % reference.num_nodes
    carried = carried_focal_seeds(full, reference, seeds, sample_indices, parents)
    categories = np.zeros(len(copied), dtype=np.int8)
    categories[recent] = np.where(carried[inverse], 2, 1)
    labels = np.array(["older", "recent_nonfocal", "focal"])
    copied["category"] = labels[categories]
    keys = ["method", "af_cutoff", "haplotype_index", "category"]
    weights = ["sites", "span_bp", "segments", "distinct_parents"]
    grouped = copied.groupby(keys)[weights].sum()
    index = pd.MultiIndex.from_product(
        [
            ["full_hmm", "greedy"],
            [run["af_cutoff"] for run in job.dataset["runs"]],
            range(full.num_samples),
            labels,
        ],
        names=keys,
    )
    grouped = grouped.reindex(index, fill_value=0).reset_index()
    for weight in weights:
        total = grouped.groupby(keys[:-1])[weight].transform("sum")
        grouped[f"{weight}_percentage"] = 100 * grouped[weight] / total.where(total > 0)
    grouped["dataset"] = job.dataset["name"]
    grouped["focal_choice"] = job.choice
    save_cached(grouped, output, stamp)
    return grouped


def gap_parent_composition(config: dict) -> pd.DataFrame:
    """Distribute independent cohort analyses and save their descriptive summaries."""
    name = config["experiment"]["performance"]["experiment_id"]
    runs = pd.read_csv(config["data_dir"] / "performance" / name / "runs.csv")
    jobs = []
    for dataset in config["manifest"]["datasets"]:
        selected = runs.loc[runs.dataset.eq(dataset["name"]) & runs.method.eq("greedy")]
        expected = dict(
            zip(
                selected.af_cutoff, selected.eligible_ancestors.astype(int), strict=True
            )
        )
        jobs.append(
            GapParentJob(
                config["data_dir"],
                config["cache_dir"],
                dataset,
                config["experiment"]["focal_choice"],
                expected,
            )
        )
    workers = config["workers"]
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    rows = []
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=numcodecs.blosc.set_nthreads,
        initargs=(config["helper_threads"],),
    ) as pool:
        futures = {pool.submit(gap_parents_analyse_dataset, job): job for job in jobs}
        for future in cf.as_completed(futures):
            rows.append(future.result())
            logger.info("Gap-parent composition: %s", futures[future].dataset["name"])
    table = pd.concat(rows, ignore_index=True)
    keys = ["dataset", "method", "af_cutoff", "category"]
    measurements = [
        "sites_percentage",
        "span_bp_percentage",
        "segments_percentage",
        "distinct_parents_percentage",
    ]
    summary = table.groupby(keys)[measurements].agg(["mean", "sem", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    sizes = cohort_table(config)[["dataset", "num_samples"]]
    summary = summary.merge(sizes, on="dataset", validate="many_to_one")
    summary.to_csv(config["cache_dir"] / "gap_parent_composition.csv", index=False)
    return summary


class BitsetCladeMap(tscompare.methods.CladeMap):
    """Use exact bitsets with the existing edge-difference iterator contract."""

    def __init__(self, ts):
        self._nil = 0
        self._nodes = collections.defaultdict(set)
        self._clades = collections.defaultdict(int)
        self.tree_sequence = ts
        self.tree = ts.first(sample_lists=True)
        for node in self.tree.nodes():
            clade = 0
            for sample in self.tree.samples(node):
                clade |= 1 << int(sample)
            self._nodes[clade].add(node)
            self._clades[node] = clade
        self._prev = self._clades.copy()
        self._diff = ts.edge_diffs()
        next(self._diff)

    def _propagate(self, edge, downdate=False):
        nodes = set()
        node = edge.parent
        clade = self._clades[edge.child]
        while node != tskit.NULL:
            last = self._clades[node]
            updated = last & ~clade if downdate else last | clade
            self._clades[node] = updated
            if last != 0:
                self._nodes[last].remove(node)
                if len(self._nodes[last]) == 0:
                    del self._nodes[last]
            self._nodes[updated].add(node)
            nodes.add(node)
            node = self.tree.parent(node)
        return nodes

    def clades(self):
        return self._nodes.keys() - {self._nil}

    def __getitem__(self, clade):
        if clade in self:
            return frozenset(self._nodes[clade])
        return frozenset()


def haplotype_arf(comparison, reference):
    """Temporarily use the exact bitset iterator in this single-threaded worker.

    Each pt7 comparison runs in an independent process. Restore the library
    class even on failure, so other callers retain tscompare's normal behavior.
    """
    original = tscompare.methods.CladeMap
    tscompare.methods.CladeMap = BitsetCladeMap
    try:
        return tscompare.haplotype_arf(comparison, reference)
    finally:
        tscompare.methods.CladeMap = original


@dataclasses.dataclass
class ExperimentTables:
    """Tables with shared dataset/method keys, saved alongside the experiment."""

    calibration: pd.DataFrame
    recurrence: pd.DataFrame
    overlap: pd.DataFrame
    haplotypes: pd.DataFrame
    sample_means: pd.DataFrame
    allele_checks: pd.DataFrame


def conditioning_load_config(path: dict) -> dict:
    """Resolve experiment paths relative to its YAML file."""
    config = path.copy()
    for key in ("pipeline_dir", "output_dir", "recombination_map"):
        config[key] = (config["base_dir"] / config[key]).resolve()
    for dataset in config["datasets"]:
        for key in ("zarr_path", "sample_list", "truth_path", "raw_zarr_path"):
            if dataset[key] is not None:
                dataset[key] = (config["base_dir"] / dataset[key]).resolve()
    return config


def calibrate_mismatch(epsilon: float, span_bp: int, num_sites: int) -> float:
    """Convert one-haplotype errors/bp to errors per included variant site.

    ``num_sites`` counts the fixed, quality-filtered regional source axis,
    before the matching cohort's singleton/polymorphism filters. Transferring
    this average to inference sites assumes the same per-site error profile.
    This is the linear expectation conversion, not a Poisson event probability.
    """
    probability = epsilon * span_bp / num_sites
    if not 0 < probability < 0.5:
        raise ValueError("The calibrated biallelic mismatch must be between 0 and 0.5")
    return probability


def _write_array(group, name, values, dimensions, config):
    chunks = []
    for size, dimension in zip(values.shape, dimensions, strict=True):
        chunk = size
        if dimension == "variants":
            chunk = min(size, config["storage"]["variant_chunk"])
        elif dimension == "samples":
            chunk = min(size, config["storage"]["sample_chunk"])
        chunks.append(chunk)
    array = group.create_array(name, data=values, chunks=tuple(chunks))
    array.attrs["_ARRAY_DIMENSIONS"] = dimensions


def _input_selection(dataset):
    sample_ids = None
    if dataset["sample_list"] is not None:
        sample_ids = pd.read_csv(dataset["sample_list"], header=None)[0].tolist()
    return {
        "source": str(dataset["zarr_path"]),
        "sample_ids": sample_ids,
        "region": dataset["region"],
        "ancestral_field": dataset["ancestral_field"],
        "exclude_masks": dataset["exclude_masks"],
    }


def prepare_dataset(config: dict, dataset: dict) -> dict:
    """Subset the source once, retaining coordinates and simulation error flags.

    Read only regional rows and selected individuals. All variant-dimensioned
    arrays in the compact store share one physical chunk size, as required by
    native vcztools readers. Source stores are opened read-only.
    """
    root = config["output_dir"] / dataset["name"]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "input_manifest.json"
    selection = _input_selection(dataset)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["input_selection"] != selection:
            raise ValueError(
                "Input selection changed; use a new output_dir for this experiment"
            )
        return manifest
    source = zarr.open_group(dataset["zarr_path"], mode="r")
    positions = source["variant_position"][:]
    left, right = dataset["region"]
    selected = (positions >= left) & (positions < right)
    for name in dataset["exclude_masks"]:
        selected &= ~source[name][:].astype(bool)
    rows = np.flatnonzero(selected)
    alleles = source["variant_allele"].oindex[rows, :]
    ancestral = source[dataset["ancestral_field"]].oindex[rows]
    known_ancestral = np.any(alleles == ancestral[:, None], axis=1)
    rows = rows[known_ancestral]
    alleles = alleles[known_ancestral]
    ancestral = ancestral[known_ancestral]
    selected_positions = positions[rows]
    assert np.all(np.diff(selected_positions) > 0)
    identifiers = source["sample_id"][:]
    if dataset["sample_list"] is None:
        columns = np.arange(len(identifiers))
    else:
        names = pd.read_csv(dataset["sample_list"], header=None)[0].astype(str)
        lookup = {name: index for index, name in enumerate(identifiers)}
        columns = np.array([lookup[name] for name in names])
    calls = source["call_genotype"].oindex[rows, columns, :]
    contigs = source["variant_contig"].oindex[rows]
    assert np.all(contigs == contigs[0])
    contig = int(contigs[0])
    output = root / "input.zarr"
    store = zarr.open_group(output, mode="w", zarr_format=3)
    store.attrs.update({"vcf_zarr_version": "0.4", "source": str(dataset["zarr_path"])})
    arrays = {
        "variant_position": (selected_positions, ["variants"]),
        "variant_allele": (alleles, ["variants", "alleles"]),
        dataset["ancestral_field"]: (ancestral, ["variants"]),
        "variant_contig": (np.zeros(len(rows), dtype=np.int8), ["variants"]),
        "contig_id": (source["contig_id"][contig : contig + 1], ["contigs"]),
        "contig_length": (source["contig_length"][contig : contig + 1], ["contigs"]),
        "sample_id": (identifiers[columns], ["samples"]),
        "call_genotype": (calls, ["variants", "samples", "ploidy"]),
        "call_genotype_phased": (
            source["call_genotype_phased"].oindex[rows, columns].astype(bool),
            ["variants", "samples"],
        ),
    }
    if dataset["truth_path"] is not None:
        for name, dimensions in (
            ("call_genotype_error_mask", ["variants", "samples", "ploidy"]),
            ("call_genotype_phase", ["variants", "samples"]),
        ):
            selection = (
                (rows, columns, slice(None))
                if len(dimensions) == 3
                else (rows, columns)
            )
            arrays[name] = (source[name].oindex[selection].astype(bool), dimensions)
    for name, (values, dimensions) in arrays.items():
        _write_array(store, name, np.asarray(values), dimensions, config)
    zarr.consolidate_metadata(output)
    probability = calibrate_mismatch(
        config["epsilon_per_haplotype_bp"], right - left, len(rows)
    )
    derived_code = np.where(ancestral == alleles[:, 0], 1, 0)
    derived_count = (calls == derived_code[:, None, None]).sum(axis=(1, 2))
    called_count = (calls >= 0).sum(axis=(1, 2))
    segregating = (derived_count > 0) & (derived_count < called_count)
    inference = segregating & (derived_count > 1)
    manifest = {
        "input_selection": selection,
        "dataset": dataset["name"],
        "label": dataset["label"],
        "source": str(dataset["zarr_path"]),
        "sample_list": str(dataset["sample_list"])
        if dataset["sample_list"] is not None
        else None,
        "left_bp": left,
        "right_bp": right,
        "span_bp": right - left,
        "num_individuals": len(columns),
        "num_haplotypes": calls.shape[1] * calls.shape[2],
        "calibration_sites": len(rows),
        "segregating_sites": int(segregating.sum()),
        "expected_inference_sites": int(inference.sum()),
        "epsilon_per_haplotype_bp": config["epsilon_per_haplotype_bp"],
        "sites_per_bp": len(rows) / (right - left),
        "mismatch_probability": probability,
    }
    np.savez(root / "source_indices.npz", rows=rows, columns=columns)
    pd.Series(identifiers[columns]).to_csv(
        root / "selected_samples.csv", index=False, header=False
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare(config: dict) -> pd.DataFrame:
    """Write one pipeline configuration per independently calibrated source."""
    records = []
    settings = config["pipeline"]
    for dataset in config["datasets"]:
        record = prepare_dataset(config, dataset)
        root = config["output_dir"] / dataset["name"]
        record["epsilon_per_haplotype_bp"] = config["epsilon_per_haplotype_bp"]
        record["mismatch_probability"] = calibrate_mismatch(
            config["epsilon_per_haplotype_bp"],
            record["span_bp"],
            record["calibration_sites"],
        )
        (root / "input_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        pipeline = dict(settings)
        del pipeline["af_cutoff"], pipeline["min_region_size_kb"]
        pipeline.update(
            {
                "config_name": dataset["name"],
                "data_dir": str(root),
                "progress_dir": str(root / "progress"),
                "temp_dir": str(root / "temporary"),
                "hmm": {
                    "recombination": None,
                    "recombination_rate": str(config["recombination_map"]),
                    "mismatch": record["mismatch_probability"],
                },
                "performance": None,
                "datasets": [
                    {
                        "name": dataset["name"],
                        "zarr_path": str(root / "input.zarr"),
                        "ts_path": None,
                        "ancestral_state": {"field": dataset["ancestral_field"]},
                        "af_cutoff": settings["af_cutoff"],
                        "min_region_size_kb": settings["min_region_size_kb"],
                    }
                ],
            }
        )
        (root / "temporary").mkdir(exist_ok=True)
        (root / "config.yaml").write_text(yaml.safe_dump(pipeline, sort_keys=False))
        records.append(
            {key: value for key, value in record.items() if key != "input_selection"}
        )
    calibration = pd.DataFrame(records)
    calibration.to_csv(config["output_dir"] / "calibration.csv", index=False)
    return calibration


def run_pipeline(config: dict) -> None:
    """Run the unmodified matching-evaluation DAG inside each experiment folder.

    The sibling project's environment supplies Snakemake and its own libraries.
    Logs, native configs, inputs, references and outputs stay in this repository.
    Snakemake schedules native threads and greedy multiprocessing workers.
    """
    pipeline_dir = config["pipeline_dir"]
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(config["output_dir"] / "uv_cache")
    environment["NUMBA_CACHE_DIR"] = str(config["output_dir"] / "numba_cache")
    environment["PYTHONPATH"] = str(pipeline_dir)
    revision = {}
    for name, path in (
        ("tsinfer_match_eval", pipeline_dir),
        ("tsinfer", pathlib.Path(tsinfer.__file__).resolve().parents[1]),
    ):
        revision[name] = {
            "commit": subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip(),
            "diff": subprocess.check_output(
                ["git", "-C", str(path), "diff"], text=True
            ),
        }
    (config["output_dir"] / "code_versions.json").write_text(
        json.dumps(revision, indent=2) + "\n"
    )
    for dataset in config["datasets"]:
        root = config["output_dir"] / dataset["name"]
        source_cache = root / "source_cache"
        source_cache.mkdir(exist_ok=True)
        command = [
            "uv",
            "run",
            "--no-sync",
            "--project",
            str(pipeline_dir),
            "snakemake",
            "--snakefile",
            str(pipeline_dir / "Snakefile"),
            "--directory",
            str(root),
            "--cores",
            "all",
            "--runtime-source-cache-path",
            str(source_cache),
        ]
        logger.info("Running %s; log: %s", dataset["name"], root / "pipeline.log")
        with (root / "pipeline.log").open("w") as log:
            subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def aligned_errors(store: zarr.Group, rows: np.ndarray) -> np.ndarray:
    """Return injected genotype errors on the observed, post-phasing call axis.

    This snapshot stores True for altered calls, before phase switching, as
    verified against its raw companion by :func:`verify_error_mask`. Swap the
    ploidy columns wherever the stored phase is 1. Phase switches themselves
    are not labelled as injected genotype errors.
    """
    errors = store["call_genotype_error_mask"].oindex[rows, :, :].astype(bool)
    phase = store["call_genotype_phase"].oindex[rows, :].astype(bool)
    return np.where(phase[:, :, None], errors[:, :, ::-1], errors)


def verify_error_mask(config: dict, dataset: dict) -> None:
    """Check this saved snapshot's mask polarity and phase axis against raw calls.

    The current generator's mask convention differs from this October snapshot.
    Use the saved data, not today's generator implementation, as the evidence.
    This read-only check runs on the compact selected region and cohort.
    """
    root = config["output_dir"] / dataset["name"]
    store = zarr.open_group(root / "input.zarr", mode="r")
    raw = zarr.open_group(dataset["raw_zarr_path"], mode="r")
    source = zarr.open_group(dataset["zarr_path"], mode="r")
    with np.load(root / "source_indices.npz") as indices:
        rows = indices["rows"]
        columns = indices["columns"]
    np.testing.assert_array_equal(
        raw["variant_position"].oindex[rows], store["variant_position"][:]
    )
    np.testing.assert_array_equal(
        raw["sample_id"].oindex[columns], store["sample_id"][:]
    )
    expected = raw["call_genotype"].oindex[rows, columns, :]
    phase = store["call_genotype_phase"][:].astype(bool)
    expected_phased = np.where(phase[:, :, None], expected[:, :, ::-1], expected)
    observed = store["call_genotype"][:]
    errors = aligned_errors(store, np.arange(len(rows)))
    np.testing.assert_array_equal(errors, observed != expected_phased)
    source_errors = source["call_genotype_error_mask"].oindex[rows, :, :].astype(bool)
    counts = source["variant_genotype_error_count"].oindex[rows]
    np.testing.assert_array_equal(source_errors.sum(axis=(1, 2)), counts)
    report = {
        "mask_true_means": "injected_genotype_error_before_phasing",
        "verified_against": str(dataset["raw_zarr_path"]),
        "selected_error_calls": int(errors.sum()),
        "phase_swapped_diplotypes": int(phase.sum()),
        "error_mask_matches_observed_minus_phased_truth": True,
    }
    (root / "error_mask_check.json").write_text(json.dumps(report, indent=2) + "\n")


def copying_events(ts: tskit.TreeSequence) -> pd.DataFrame:
    """Count sample mutations and parent changes at consecutive inference sites.

    Boundary edges in the unobserved chromosome flanks do not count. Means are
    formed per diploid individual by summing its two haplotypes first.
    """
    samples = ts.samples()
    sample_index = {int(node): index for index, node in enumerate(samples)}
    parents = np.full((len(samples), ts.num_sites), tskit.NULL, dtype=np.int32)
    for edge in ts.edges():
        if edge.child in sample_index:
            left = np.searchsorted(ts.sites_position, edge.left)
            right = np.searchsorted(ts.sites_position, edge.right)
            parents[sample_index[edge.child], left:right] = edge.parent
    mutations = np.bincount(ts.mutations_node, minlength=ts.num_nodes)
    result = []
    for index, node_id in enumerate(samples):
        metadata = ts.node(node_id).metadata
        path = parents[index]
        called = path >= 0
        adjacent = called[1:] & called[:-1]
        switches = adjacent & (path[1:] != path[:-1])
        result.append(
            {
                "sample_id": metadata["sample_id"],
                "ploidy_index": metadata["ploidy_index"],
                "switches": int(switches.sum()),
                "mutations": int(mutations[node_id]),
            }
        )
    return pd.DataFrame(result)


def called_allele_check(ts: tskit.TreeSequence, store: zarr.Group) -> dict:
    """Count output/input allele differences without adding a validation plot.

    Match individuals and haplotypes by node metadata, not table row order.
    A failed reconstruction is retained as an experimental finding rather than
    silently treating a matching mutation count as a complete representation.
    """
    rows = np.searchsorted(store["variant_position"][:], ts.sites_position)
    calls = store["call_genotype"].oindex[rows, :, :]
    alleles = store["variant_allele"].oindex[rows, :]
    lookup = {name: column for column, name in enumerate(store["sample_id"][:])}
    nodes = [ts.node(node_id) for node_id in ts.samples()]
    columns = [lookup[node.metadata["sample_id"]] for node in nodes]
    ploidies = [node.metadata["ploidy_index"] for node in nodes]
    expected = calls[:, columns, ploidies]
    evaluated = 0
    disagreements = 0
    for variant in ts.variants(isolated_as_missing=False):
        site = variant.site.id
        called = expected[site] >= 0
        expected_alleles = alleles[site, expected[site, called]]
        states = np.asarray(variant.alleles, dtype=object)
        observed_alleles = states[variant.genotypes[called]]
        disagreements += int(np.count_nonzero(observed_alleles != expected_alleles))
        evaluated += int(called.sum())
    return {"called_alleles_checked": evaluated, "allele_disagreements": disagreements}


def error_overlap(ts: tskit.TreeSequence, store: zarr.Group) -> dict:
    """Intersect recurrence with injected errors at site and sample-mutation levels.

    Internal/reference mutation records cannot be assigned to a sample call.
    Report them separately rather than calling every event at an error-bearing
    site an error. Even a direct overlap is an association, not causal proof.
    """
    positions = store["variant_position"][:]
    rows = np.searchsorted(positions, ts.sites_position)
    np.testing.assert_array_equal(positions[rows], ts.sites_position)
    errors = aligned_errors(store, rows)
    error_sites = np.any(errors, axis=(1, 2))
    counts = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    recurrent = counts > 1
    sample_ids = store["sample_id"][:]
    sample_columns = {name: index for index, name in enumerate(sample_ids)}
    recurrent_sample = 0
    recurrent_sample_error = 0
    recurrent_internal = 0
    for mutation in ts.mutations():
        if not recurrent[mutation.site]:
            continue
        node = ts.node(mutation.node)
        if not node.is_sample():
            recurrent_internal += 1
            continue
        recurrent_sample += 1
        column = sample_columns[node.metadata["sample_id"]]
        ploidy = node.metadata["ploidy_index"]
        recurrent_sample_error += int(errors[mutation.site, column, ploidy])
    return {
        "num_sites": ts.num_sites,
        "recurrent_only_sites": int(np.count_nonzero(recurrent & ~error_sites)),
        "overlap_sites": int(np.count_nonzero(recurrent & error_sites)),
        "error_only_sites": int(np.count_nonzero(~recurrent & error_sites)),
        "neither_sites": int(np.count_nonzero(~recurrent & ~error_sites)),
        "recurrent_sample_mutations": recurrent_sample,
        "recurrent_sample_mutations_at_error_calls": recurrent_sample_error,
        "recurrent_sample_mutations_at_other_calls": recurrent_sample
        - recurrent_sample_error,
        "recurrent_reference_mutations": recurrent_internal,
    }


def analyse(config: dict) -> ExperimentTables:
    """Summarise raw trees on identical inference positions, without dating."""
    calibration = []
    recurrence = []
    overlaps = []
    haplotypes = []
    allele_checks = []
    settings = config["pipeline"]
    for dataset in config["datasets"]:
        name = dataset["name"]
        root = config["output_dir"] / name
        manifest = json.loads((root / "input_manifest.json").read_text())
        del manifest["input_selection"]
        full = tskit.load(root / "matches" / f"{name}_inferred_samples_raw.trees")
        greedy_dir = root / "matches" / f"af_{settings['af_cutoff']}"
        greedy_dir /= f"min_region_kb_{settings['min_region_size_kb']}"
        greedy = tskit.load(greedy_dir / f"{name}_inferred_greedy_samples_raw.trees")
        np.testing.assert_array_equal(full.sites_position, greedy.sites_position)
        assert full.num_sites == manifest["expected_inference_sites"]
        assert full.num_samples == greedy.num_samples == manifest["num_haplotypes"]
        native_path = root / "configs" / f"{name}_ancestor_inference.toml"
        native = tsinfer.config.Config.from_toml(native_path)
        hmm = tsinfer.matching.hmm_parameters(
            native.match.sources["samples"], full.sites_position
        )
        np.testing.assert_allclose(hmm.mismatch, manifest["mismatch_probability"])
        map_path = native.match.sources["samples"].recombination_rate
        genetic_positions = tsinfer.matching.read_genetic_map(
            map_path, full.sites_position
        )
        genetic_distance = np.diff(genetic_positions)
        expected_recombination = -np.expm1(-2 * genetic_distance) / 2
        np.testing.assert_allclose(hmm.recombination[1:], expected_recombination)
        manifest.update(
            {
                "inference_sites": full.num_sites,
                "median_genetic_distance_morgans": float(np.median(genetic_distance)),
                "median_recombination_probability": float(
                    np.median(hmm.recombination[1:])
                ),
                "min_recombination_probability": float(np.min(hmm.recombination[1:])),
                "max_recombination_probability": float(np.max(hmm.recombination[1:])),
            }
        )
        calibration.append(manifest)
        store = zarr.open_group(root / "input.zarr", mode="r")
        if dataset["raw_zarr_path"] is not None:
            verify_error_mask(config, dataset)
        trees = {"full_hmm": full, "greedy": greedy}
        if dataset["truth_path"] is not None:
            trees["truth"] = tskit.load(dataset["truth_path"])
        for method, ts in trees.items():
            identity = {"dataset": name, "label": dataset["label"], "method": method}
            recurrence.append(
                {**identity, **recurrent_site_counts(ts, full.sites_position)}
            )
            if method == "truth":
                continue
            events = copying_events(ts).assign(**identity)
            haplotypes.append(events)
            allele_checks.append({**identity, **called_allele_check(ts, store)})
            if dataset["truth_path"] is not None:
                overlaps.append({**identity, **error_overlap(ts, store)})
    haplotype_table = pd.concat(haplotypes, ignore_index=True)
    keys = ["dataset", "label", "method"]
    per_sample = haplotype_table.groupby([*keys, "sample_id"], sort=False)[
        ["switches", "mutations"]
    ].sum()
    sample_means = per_sample.groupby(keys, sort=False).mean().reset_index()
    tables = ExperimentTables(
        pd.DataFrame(calibration),
        pd.DataFrame(recurrence),
        pd.DataFrame(overlaps),
        haplotype_table,
        sample_means,
        pd.DataFrame(allele_checks),
    )
    for field in dataclasses.fields(tables):
        getattr(tables, field.name).to_csv(
            config["output_dir"] / f"{field.name}.csv", index=False
        )
    return tables


@dataclasses.dataclass
class ScalingData:
    """Cohort identities and saved AF settings shared by archive figures."""

    config: dict
    cohorts: pd.DataFrame
    datasets: list[dict]
    sample_sizes: np.ndarray
    cutoffs: list[float]
    panel_cutoffs: list[float | None]


def prepare_scaling_data(config):
    """Read cohort metadata without storing presentation settings."""
    cohorts = cohort_table(config)
    by_name = {dataset["name"]: dataset for dataset in config["manifest"]["datasets"]}
    datasets = [by_name[name] for name in cohorts.dataset]
    cutoffs = sorted(
        {run["af_cutoff"] for dataset in datasets for run in dataset["runs"]}
    )
    single_setting = all((len(dataset["runs"]) == 1 for dataset in datasets))
    panel_cutoffs = [None] if single_setting and len(cutoffs) > 1 else cutoffs
    return ScalingData(
        config,
        cohorts,
        datasets,
        cohorts.num_samples.to_numpy(),
        cutoffs,
        panel_cutoffs,
    )


def make_focal_regions_buffered(intervals, min_width=5):
    regions = []
    region_left = 0
    region_right = -1
    parent = -1
    parent_left = 0
    parent_right = 0
    next_parent = -1
    paths_buffer = []
    paths = []
    if len(intervals) == 0:
        return regions
    for i, (left, right) in enumerate(intervals):
        print(f"Processing {i}")
        if left > region_right:
            if region_right > 0:
                paths_buffer.append(((parent_left, parent_right), parent))
                region_width = region_right - region_left
                if region_width >= min_width:
                    regions.append((region_left, region_right))
                    paths += paths_buffer
                    paths_buffer = []
                    next_parent = -1
            print(f"Starting region at {left}")
            region_left = left
            region_right = right
            print(f"Starting parent is {i} from {left} to {right}")
            parent = i
            parent_left = left
            parent_right = right
        else:
            if left >= parent_right:
                print(
                    f"Moved past current parent {parent} which ends at {parent_right}, now at {left}"
                )
                paths_buffer.append(((parent_left, parent_right), parent))
                parent_left = parent_right
                if left == parent_right and right > region_right:
                    print(f"New interval starts at {left} and is longest")
                    parent = i
                    parent_right = right
                    region_right = right
                else:
                    print(
                        f"Longest parent now is {next_parent} ending at {region_right - 1}"
                    )
                    parent = next_parent
                    parent_right = region_right
            elif left == parent_left and right > parent_right:
                print(f"Better starting parent is {i} ending at {right}")
                parent = i
                parent_right = right
            if right > region_right:
                region_right = right
                next_parent = i
                print(f"New next_parent is {i} ending at {right}")
    print(f"After last int, we have current ({parent_left},{parent_right}): {parent}")
    print(f"New next parent is {next_parent} ending at {region_right}")
    paths_buffer.append(((parent_left, parent_right), parent))
    if parent_right != region_right:
        paths_buffer.append(((parent_right, region_right), next_parent))
    region_width = region_right - region_left
    if region_right > 0 and region_width >= min_width:
        regions.append((region_left, region_right))
        paths += paths_buffer
    return (regions, paths)


def make_focal_regions(intervals, min_width):
    regions = []
    paths = []
    i = 0
    while i < len(intervals):
        region_left = intervals[i][0]
        cursor = region_left
        paths_buffer = []
        while True:
            parent = None
            max_right = cursor
            while i < len(intervals):
                left, right = intervals[i]
                if left > cursor:
                    break
                if right > max_right:
                    parent = i
                    max_right = right
                i += 1
            if parent is None:
                break
            paths_buffer.append(((cursor, max_right), parent))
            cursor = max_right
        region_right = cursor
        region_width = region_right - region_left
        if region_width >= min_width:
            regions.append((region_left, region_right))
            paths.extend(paths_buffer)
    return (regions, paths)


@dataclasses.dataclass
class HaplotypeDiagnostics:
    """Site-based evidence and original-node parents for one saved comparison."""

    summary: dict
    full: dict
    stitched: dict
    full_parents: np.ndarray
    stitched_parents: np.ndarray
    accepted: np.ndarray
    disagreement: np.ndarray
    categories: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.summary['sample_id']}/{self.summary['ploidy_index']}"


@dataclasses.dataclass
class DatasetDiagnostics:
    """One copied reference and the selected experiment's saved haplotypes."""

    name: str
    label: str
    reference: tskit.TreeSequence
    positions: np.ndarray
    site_edges: np.ndarray
    cases: list[HaplotypeDiagnostics]
    primary: HaplotypeDiagnostics


def zero_coverage_read_matches(path: pathlib.Path) -> dict:
    """Key JSONL records by haplotype identity, checking uniqueness."""
    records = {}
    with path.open() as file:
        for line in file:
            record = json.loads(line)
            key = (record["sample_id"], record["ploidy_index"])
            assert key not in records
            records[key] = record
    return records


def interval_mask(intervals: list, num_sites: int) -> np.ndarray:
    """Reconstruct disjoint site intervals without inferring BP coverage."""
    mask = np.zeros(num_sites, dtype=bool)
    for start, end in intervals:
        assert 0 <= start < end <= num_sites
        assert not np.any(mask[start:end])
        mask[start:end] = True
    return mask


def load_dataset(
    settings: dict,
    *,
    experiment_config,
    input_dir,
    output_dir,
    plot_terminal_site_width_bp=1,
) -> DatasetDiagnostics:
    """Validate saved summaries against paths and focal/gap site partitions."""
    name = settings["name"]
    directory = output_dir / name
    reference = tskit.load(input_dir / "ancestors" / f"{name}_inferred_ancestors.trees")
    positions = reference.sites_position
    terminal_edge = positions[-1] + plot_terminal_site_width_bp
    site_edges = np.concatenate((positions, [terminal_edge]))
    full_records = zero_coverage_read_matches(directory / "full_matches.jsonl")
    stitched_records = zero_coverage_read_matches(directory / "stitched_matches.jsonl")
    summaries = pd.read_csv(directory / "comparison.csv").to_dict("records")
    assert len(summaries) == len(full_records) == len(stitched_records)
    cases = []
    for summary in summaries:
        summary["num_extra_switches"] = (
            summary["stitched_switches"] - summary["full_switches"]
        )
        summary["extra_switch_fraction"] = (
            summary["num_extra_switches"] / summary["stitched_switches"]
            if summary["stitched_switches"] > 0
            else np.nan
        )
        key = (summary["sample_id"], summary["ploidy_index"])
        full = full_records[key]
        stitched = stitched_records[key]
        for record in (full, stitched):
            for field in (
                "dataset",
                "source",
                "sample_id",
                "ploidy_index",
                "ac_cutoff",
            ):
                assert record[field] == summary[field]
        assert summary["dataset"] == name
        assert summary["ac_cutoff"] == experiment_config["ac_cutoff"]
        full_parents = parents_at_sites(full, reference)
        stitched_parents = parents_at_sites(stitched, reference)
        accepted = interval_mask(stitched["accepted_site_intervals"], len(positions))
        gaps = interval_mask(stitched["gap_site_intervals"], len(positions))
        assert np.array_equal(gaps, ~accepted)
        disagreement = full_parents != stitched_parents
        assert summary["num_sites"] == len(positions)
        assert summary["accepted_sites"] == np.count_nonzero(accepted)
        assert summary["gap_sites"] == np.count_nonzero(gaps)
        assert summary["num_gaps"] == len(stitched["gap_site_intervals"])
        assert summary["accepted_disagreement_sites"] == np.count_nonzero(
            disagreement & accepted
        )
        assert summary["gap_disagreement_sites"] == np.count_nonzero(
            disagreement & gaps
        )
        assert np.isclose(summary["fraction_parent_agreement"], np.mean(~disagreement))
        assert summary["full_switches"] == len(full["path"]) - 1
        assert summary["stitched_switches"] == len(stitched["path"]) - 1
        assert summary["mutations_identical"] == (
            full["mutations"] == stitched["mutations"]
        )
        categories = (~accepted).astype(np.int8) * 2 + disagreement.astype(np.int8)
        cases.append(
            HaplotypeDiagnostics(
                summary,
                full,
                stitched,
                full_parents,
                stitched_parents,
                accepted,
                disagreement,
                categories,
            )
        )
    primary_settings = settings["deep_dive"]
    primary = next(
        (
            case
            for case in cases
            if case.summary["sample_id"] == primary_settings["sample_id"]
            and case.summary["ploidy_index"] == primary_settings["ploidy_index"]
        )
    )
    return DatasetDiagnostics(
        name, settings["label"], reference, positions, site_edges, cases, primary
    )


def switch_breakdown(case: HaplotypeDiagnostics, parents: np.ndarray) -> np.ndarray:
    """Partition switches into focal interiors, gap interiors, and provenance seams."""
    switches = parents[1:] != parents[:-1]
    focal_interior = case.accepted[1:] & case.accepted[:-1]
    gap_interior = ~case.accepted[1:] & ~case.accepted[:-1]
    seams = case.accepted[1:] != case.accepted[:-1]
    counts = np.array(
        [
            np.count_nonzero(switches & focal_interior),
            np.count_nonzero(switches & gap_interior),
            np.count_nonzero(switches & seams),
        ]
    )
    assert counts.sum() == np.count_nonzero(switches)
    return counts


def gap_diagnostics(dataset: DatasetDiagnostics) -> pd.DataFrame:
    """Compare independent fills with the full path, including their seam switches."""
    case = dataset.primary
    rows = []
    for index, (start, end) in enumerate(case.stitched["gap_site_intervals"]):
        row = {
            "gap": index,
            "start_site": start,
            "end_site": end,
            "num_sites": end - start,
            "start_bp": dataset.positions[start],
            "end_bp": dataset.site_edges[end],
            "agree_sites": int(np.count_nonzero(~case.disagreement[start:end])),
            "differ_sites": int(np.count_nonzero(case.disagreement[start:end])),
        }
        row["agreement_fraction"] = row["agree_sites"] / row["num_sites"]
        for method in ("full", "stitched"):
            parents = getattr(case, f"{method}_parents")
            row[f"{method}_interior_switches"] = int(
                np.count_nonzero(parents[start + 1 : end] != parents[start : end - 1])
            )
            row[f"{method}_left_seam_switch"] = (
                bool(parents[start] != parents[start - 1]) if start > 0 else None
            )
            row[f"{method}_right_seam_switch"] = (
                bool(parents[end] != parents[end - 1]) if end < len(parents) else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


@dataclasses.dataclass
class DetailWindow:
    """The target interval and its displayed context, all as half-open site bounds."""

    kind: str
    target_start: int
    target_end: int
    start: int
    end: int


def select_windows(
    dataset: DatasetDiagnostics, gap_table: pd.DataFrame, *, config
) -> list[DetailWindow]:
    """Choose disagreement and agreement examples with deterministic site-based rules."""
    case = dataset.primary
    disagree_gap = gap_table.sort_values(
        ["differ_sites", "gap"], ascending=[False, True]
    ).iloc[0]
    eligible = gap_table.loc[
        gap_table.num_sites >= config["detail"]["min_agreement_gap_sites"]
    ]
    agree_gap = eligible.sort_values(
        ["agreement_fraction", "num_sites", "gap"], ascending=[False, False, True]
    ).iloc[0]
    focal_intervals = case.stitched["accepted_site_intervals"]
    focal_counts = [
        np.count_nonzero(case.disagreement[start:end]) for start, end in focal_intervals
    ]
    focal_start, focal_end = focal_intervals[int(np.argmax(focal_counts))]
    intervals = [
        (
            "Gap disagreement",
            int(disagree_gap.start_site),
            int(disagree_gap.end_site),
            True,
        ),
        ("Gap agreement", int(agree_gap.start_site), int(agree_gap.end_site), False),
        ("Focal disagreement", focal_start, focal_end, True),
    ]
    targets = []
    for kind, start, end, look_for_disagreement in intervals:
        target_mask = case.disagreement[start:end] == look_for_disagreement
        assert np.any(target_mask), f"No {kind.lower()} example for {case.label}"
        anchor = start + int(np.flatnonzero(target_mask)[0])
        targets.append((kind, start, end, anchor))
    mixed = gap_table.loc[(gap_table.agree_sites > 0) & (gap_table.differ_sites > 0)]
    mixed_gap = mixed.sort_values(
        ["differ_sites", "gap"], ascending=[False, True]
    ).iloc[0]
    start = int(mixed_gap.start_site)
    end = int(mixed_gap.end_site)
    differs = case.disagreement[start:end]
    transitions = np.flatnonzero(differs[1:] != differs[:-1]) + 1
    anchor = start + int(transitions[0])
    targets.insert(2, ("Mixed gap fill", start, end, anchor))
    windows = []
    context = config["detail"]["context_sites"]
    max_sites = config["detail"]["max_sites"]
    for kind, start, end, anchor in targets:
        if end - start + 2 * context <= max_sites:
            left = max(0, start - context)
            right = min(len(case.accepted), end + context)
        else:
            left = max(0, anchor - context)
            right = min(len(case.accepted), left + max_sites)
        windows.append(DetailWindow(kind, start, end, left, right))
    return windows


def seam_distances(accepted: np.ndarray) -> np.ndarray:
    """Compute distances to the two adjacent sites of the nearest provenance seam."""
    seams = np.flatnonzero(accepted[1:] != accepted[:-1]) + 1
    sites = np.arange(len(accepted))
    insertion = np.searchsorted(seams, sites, side="right")
    left = np.full(len(sites), np.inf)
    right = np.full(len(sites), np.inf)
    has_left = insertion > 0
    has_right = insertion < len(seams)
    left[has_left] = sites[has_left] - seams[insertion[has_left] - 1]
    right[has_right] = seams[insertion[has_right]] - 1 - sites[has_right]
    return np.minimum(left, right)


@dataclasses.dataclass
class GapComparison:
    """Coverage counts, zero-coverage intervals, and overlap with one saved gap mask."""

    case: HaplotypeDiagnostics
    coverage: np.ndarray
    zero_gaps: np.ndarray
    categories: np.ndarray
    summary: dict


def gaps_with_overlap(intervals: np.ndarray, other_gap_mask: np.ndarray) -> int:
    """Count intersecting intervals in linear time using site-mask prefix sums."""
    prefix = np.concatenate(([0], np.cumsum(other_gap_mask)))
    overlap_sites = prefix[intervals[:, 1]] - prefix[intervals[:, 0]]
    return int(np.count_nonzero(overlap_sites > 0))


def gap_bound_rows(
    key: dict, intervals: np.ndarray, dataset: DatasetDiagnostics
) -> list[dict]:
    """Export exact site bounds; positions use the plotting convention above."""
    return [
        {
            **key,
            "gap_index": index,
            "left_site_index": int(start),
            "right_site_index": int(end),
            "num_sites": int(end - start),
            "left_position": float(dataset.positions[start]),
            "right_position": float(dataset.site_edges[end]),
        }
        for index, (start, end) in enumerate(intervals)
    ]


def compare_gap_determination(
    dataset: DatasetDiagnostics,
    *,
    config,
    coverage_input_dir,
    experiment_config,
    gap_comparison_dir,
    input_dir,
) -> list[GapComparison]:
    """Sweep new budget-zero intervals and compare every saved haplotype's gap bounds."""
    name = dataset.name
    archive_path = coverage_input_dir / "haplotype_intervals"
    archive_path /= f"{name}_inferred_focal_ancestor_intervals.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        intervals = {key: archive[key] for key in archive.files}
    num_sites = len(dataset.positions)
    assert int(intervals["num_sites"]) == num_sites
    cutoff = experiment_config["ac_cutoff"]
    assert int(intervals["max_ac_cutoff"]) >= cutoff
    budget = config["coverage"]["mismatch_budget"]
    assert budget == 0
    assert int(intervals["max_mismatches"]) >= budget
    panel = zarr.open_group(
        coverage_input_dir / "ancestors" / f"{name}_inferred_ancestors.zarr", mode="r"
    )
    assert np.array_equal(panel["variant_position"][:], dataset.positions)
    with np.load(
        input_dir / "focal_ancestors" / f"{name}_inferred_focal_ancestors.npz",
        allow_pickle=False,
    ) as focal:
        for field in ("sample_id", "ploidy_index"):
            assert np.array_equal(intervals[field], focal[field])
        minimum_ac = config["coverage"]["min_focal_ac"]
        for case in dataset.cases:
            row = np.flatnonzero(
                (focal["sample_id"] == case.summary["sample_id"])
                & (focal["ploidy_index"] == case.summary["ploidy_index"])
            )[0]
            candidates = focal["ancestor_index"][
                focal["offsets"][row] : focal["offsets"][row + 1]
            ]
            counts = focal["derived_ac"][candidates]
            assert not np.any((counts < minimum_ac) & (counts <= cutoff))
    keys = list(zip(intervals["sample_id"], intervals["ploidy_index"], strict=True))
    assert len(set(keys)) == len(keys)
    row_by_identity = {key: row for row, key in enumerate(keys)}
    results = []
    zero_gap_rows = []
    hmm_gap_rows = []
    coverage_rows = []
    for case in dataset.cases:
        row = row_by_identity[case.summary["sample_id"], case.summary["ploidy_index"]]
        start, end = intervals["offsets"][row : row + 2]
        counts = intervals["focal_ac"][start:end]
        eligible = (counts >= minimum_ac) & (counts <= cutoff)
        left = intervals["left_site_index"][start:end, budget][eligible]
        right = intervals["right_site_index"][start:end, budget][eligible]
        assert np.all((left >= 0) & (left < right) & (right <= num_sites))
        delta = np.zeros(num_sites + 1, dtype=np.int64)
        np.add.at(delta, left, 1)
        np.add.at(delta, right, -1)
        cumulative = np.cumsum(delta)
        assert cumulative[-1] == 0 and np.all(cumulative >= 0)
        coverage = cumulative[:-1]
        zero_mask = coverage == 0
        zero_gaps = site_intervals(zero_mask)
        assert np.array_equal(interval_mask(zero_gaps.tolist(), num_sites), zero_mask)
        hmm_mask = ~case.accepted
        hmm_gaps = np.asarray(case.stitched["gap_site_intervals"], dtype=int).reshape(
            -1, 2
        )
        assert np.array_equal(site_intervals(hmm_mask), hmm_gaps)
        intersection = int(np.count_nonzero(hmm_mask & zero_mask))
        hmm_only = int(np.count_nonzero(hmm_mask & ~zero_mask))
        zero_only = int(np.count_nonzero(zero_mask & ~hmm_mask))
        neither = int(np.count_nonzero(~hmm_mask & ~zero_mask))
        assert intersection + hmm_only + zero_only + neither == num_sites
        union = intersection + hmm_only + zero_only
        hmm_sites = intersection + hmm_only
        zero_sites = intersection + zero_only
        assert hmm_sites == case.summary["gap_sites"]
        zero_bounds = {tuple(bound) for bound in zero_gaps}
        key = {
            field: case.summary[field]
            for field in ("dataset", "source", "sample_id", "ploidy_index", "ac_cutoff")
        }
        key["max_mismatches"] = budget
        metrics = {
            **key,
            "num_sites": num_sites,
            "num_coverage_intervals": int(eligible.sum()),
            "hmm_num_gaps": len(hmm_gaps),
            "zero_coverage_num_gaps": len(zero_gaps),
            "hmm_gap_sites": hmm_sites,
            "zero_coverage_gap_sites": zero_sites,
            "overlap_sites": intersection,
            "hmm_only_gap_sites": hmm_only,
            "zero_only_gap_sites": zero_only,
            "neither_gap_sites": neither,
            "gap_site_iou": intersection / union if union > 0 else np.nan,
            "hmm_gap_overlap_fraction": intersection / hmm_sites
            if hmm_sites > 0
            else np.nan,
            "zero_coverage_gap_overlap_fraction": intersection / zero_sites
            if zero_sites > 0
            else np.nan,
            "hmm_gaps_with_overlap": gaps_with_overlap(hmm_gaps, zero_mask),
            "zero_coverage_gaps_with_overlap": gaps_with_overlap(zero_gaps, hmm_mask),
            "exact_gap_bound_matches": sum(
                (tuple(bound) in zero_bounds for bound in hmm_gaps)
            ),
        }
        categories = np.full(num_sites, 3, dtype=np.int8)
        categories[hmm_mask & zero_mask] = 0
        categories[hmm_mask & ~zero_mask] = 1
        categories[zero_mask & ~hmm_mask] = 2
        results.append(GapComparison(case, coverage, zero_gaps, categories, metrics))
        zero_gap_rows.extend(gap_bound_rows(key, zero_gaps, dataset))
        hmm_gap_rows.extend(gap_bound_rows(key, hmm_gaps, dataset))
        changes = np.flatnonzero(coverage[1:] != coverage[:-1]) + 1
        bounds = np.concatenate(([0], changes, [num_sites]))
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            coverage_rows.append(
                {
                    **key,
                    "left_site_index": int(lo),
                    "right_site_index": int(hi),
                    "num_sites": int(hi - lo),
                    "coverage": int(coverage[lo]),
                }
            )
    directory = gap_comparison_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(directory / "coverage_segments.csv", index=False)
    pd.DataFrame(zero_gap_rows).to_csv(
        directory / "zero_coverage_gaps.csv", index=False
    )
    pd.DataFrame(hmm_gap_rows).to_csv(directory / "hmm_extracted_gaps.csv", index=False)
    pd.DataFrame([result.summary for result in results]).to_csv(
        directory / "gap_comparison.csv", index=False
    )
    return results


@dataclasses.dataclass
class PathDiagnosticsHaplotypeDiagnostics:
    """Site-based evidence and original-node parents for one saved comparison."""

    summary: dict
    full: dict
    stitched: dict
    full_parents: np.ndarray
    stitched_parents: np.ndarray
    covered: np.ndarray
    disagreement: np.ndarray
    categories: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.summary['sample_id']}/{self.summary['ploidy_index']}"


@dataclasses.dataclass
class PathDiagnosticsDatasetDiagnostics:
    """One copied reference and the selected experiment's saved haplotypes."""

    name: str
    label: str
    reference: tskit.TreeSequence
    positions: np.ndarray
    site_edges: np.ndarray
    cases: list[PathDiagnosticsHaplotypeDiagnostics]
    primary: PathDiagnosticsHaplotypeDiagnostics


def path_diagnostics_load_dataset(
    settings: dict,
    *,
    experiment_config,
    input_dir,
    output_dir,
    plot_terminal_site_width_bp=1,
) -> PathDiagnosticsDatasetDiagnostics:
    """Validate saved summaries against paths and focal/gap site partitions."""
    name = settings["name"]
    directory = output_dir / name
    reference = tskit.load(input_dir / "ancestors" / f"{name}_inferred_ancestors.trees")
    positions = reference.sites_position
    terminal_edge = positions[-1] + plot_terminal_site_width_bp
    site_edges = np.concatenate((positions, [terminal_edge]))
    full_records = zero_coverage_read_matches(directory / "full_matches.jsonl")
    stitched_records = zero_coverage_read_matches(directory / "stitched_matches.jsonl")
    summaries = pd.read_csv(directory / "comparison.csv").to_dict("records")
    assert len(summaries) == len(full_records) == len(stitched_records)
    selected = {
        (item["sample_id"], item["ploidy_index"]) for item in settings["haplotypes"]
    }
    cases = []
    for summary in summaries:
        key = (summary["sample_id"], summary["ploidy_index"])
        if key not in selected:
            continue
        summary["num_extra_switches"] = (
            summary["stitched_switches"] - summary["full_switches"]
        )
        summary["extra_switch_fraction"] = (
            summary["num_extra_switches"] / summary["stitched_switches"]
            if summary["stitched_switches"] > 0
            else np.nan
        )
        full = full_records[key]
        stitched = stitched_records[key]
        for record in (full, stitched):
            for field in (
                "dataset",
                "source",
                "sample_id",
                "ploidy_index",
                "ac_cutoff",
            ):
                assert record[field] == summary[field]
        assert summary["dataset"] == name
        assert summary["ac_cutoff"] == experiment_config["ac_cutoff"]
        full_parents = parents_at_sites(full, reference)
        stitched_parents = parents_at_sites(stitched, reference)
        covered = interval_mask(stitched["focal_site_intervals"], len(positions))
        gaps = interval_mask(stitched["gap_site_intervals"], len(positions))
        assert np.array_equal(gaps, ~covered)
        disagreement = full_parents != stitched_parents
        assert summary["num_sites"] == len(positions)
        assert summary["focal_sites"] == np.count_nonzero(covered)
        assert summary["gap_sites"] == np.count_nonzero(gaps)
        assert summary["num_gaps"] == len(stitched["gap_site_intervals"])
        assert summary["focal_disagreement_sites"] == np.count_nonzero(
            disagreement & covered
        )
        assert summary["gap_disagreement_sites"] == np.count_nonzero(
            disagreement & gaps
        )
        assert np.isclose(summary["fraction_parent_agreement"], np.mean(~disagreement))
        assert summary["full_switches"] == len(full["path"]) - 1
        assert summary["stitched_switches"] == len(stitched["path"]) - 1
        assert summary["mutations_identical"] == (
            full["mutations"] == stitched["mutations"]
        )
        categories = (~covered).astype(np.int8) * 2 + disagreement.astype(np.int8)
        cases.append(
            PathDiagnosticsHaplotypeDiagnostics(
                summary,
                full,
                stitched,
                full_parents,
                stitched_parents,
                covered,
                disagreement,
                categories,
            )
        )
    assert len(cases) == len(selected)
    primary_settings = settings["deep_dive"]
    primary = next(
        (
            case
            for case in cases
            if case.summary["sample_id"] == primary_settings["sample_id"]
            and case.summary["ploidy_index"] == primary_settings["ploidy_index"]
        )
    )
    return PathDiagnosticsDatasetDiagnostics(
        name, settings["label"], reference, positions, site_edges, cases, primary
    )


def path_diagnostics_switch_breakdown(
    case: PathDiagnosticsHaplotypeDiagnostics, parents: np.ndarray
) -> np.ndarray:
    """Partition switches into focal interiors, gap interiors, and provenance seams."""
    switches = parents[1:] != parents[:-1]
    focal_interior = case.covered[1:] & case.covered[:-1]
    gap_interior = ~case.covered[1:] & ~case.covered[:-1]
    seams = case.covered[1:] != case.covered[:-1]
    counts = np.array(
        [
            np.count_nonzero(switches & focal_interior),
            np.count_nonzero(switches & gap_interior),
            np.count_nonzero(switches & seams),
        ]
    )
    assert counts.sum() == np.count_nonzero(switches)
    return counts


def path_diagnostics_gap_diagnostics(
    dataset: PathDiagnosticsDatasetDiagnostics,
) -> pd.DataFrame:
    """Compare independent fills with the full path, including their seam switches."""
    case = dataset.primary
    rows = []
    for index, (start, end) in enumerate(case.stitched["gap_site_intervals"]):
        row = {
            "gap": index,
            "start_site": start,
            "end_site": end,
            "num_sites": end - start,
            "start_bp": dataset.positions[start],
            "end_bp": dataset.site_edges[end],
            "agree_sites": int(np.count_nonzero(~case.disagreement[start:end])),
            "differ_sites": int(np.count_nonzero(case.disagreement[start:end])),
        }
        row["agreement_fraction"] = row["agree_sites"] / row["num_sites"]
        for method in ("full", "stitched"):
            parents = getattr(case, f"{method}_parents")
            row[f"{method}_interior_switches"] = int(
                np.count_nonzero(parents[start + 1 : end] != parents[start : end - 1])
            )
            row[f"{method}_left_seam_switch"] = (
                bool(parents[start] != parents[start - 1]) if start > 0 else None
            )
            row[f"{method}_right_seam_switch"] = (
                bool(parents[end] != parents[end - 1]) if end < len(parents) else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def path_diagnostics_select_windows(
    dataset: PathDiagnosticsDatasetDiagnostics, gap_table: pd.DataFrame, *, config
) -> list[DetailWindow]:
    """Choose agreement and disagreement examples using site-based rules."""
    case = dataset.primary
    disagree_gap = gap_table.sort_values(
        ["differ_sites", "gap"], ascending=[False, True]
    ).iloc[0]
    eligible = gap_table.loc[
        gap_table.num_sites >= config["detail"]["min_agreement_gap_sites"]
    ]
    agree_gap = eligible.sort_values(
        ["agreement_fraction", "num_sites", "gap"], ascending=[False, False, True]
    ).iloc[0]
    focal_intervals = case.stitched["focal_site_intervals"]
    focal_counts = [
        np.count_nonzero(case.disagreement[start:end]) for start, end in focal_intervals
    ]
    focal_start, focal_end = focal_intervals[int(np.argmax(focal_counts))]
    intervals = [
        (
            "Gap disagreement",
            int(disagree_gap.start_site),
            int(disagree_gap.end_site),
            True,
        ),
        ("Gap agreement", int(agree_gap.start_site), int(agree_gap.end_site), False),
        ("Focal disagreement", focal_start, focal_end, True),
    ]
    targets = []
    for kind, start, end, look_for_disagreement in intervals:
        target_mask = case.disagreement[start:end] == look_for_disagreement
        assert np.any(target_mask), f"No {kind.lower()} example for {case.label}"
        anchor = start + int(np.flatnonzero(target_mask)[0])
        targets.append((kind, start, end, anchor))
    mixed = gap_table.loc[(gap_table.agree_sites > 0) & (gap_table.differ_sites > 0)]
    mixed_gap = mixed.sort_values(
        ["differ_sites", "gap"], ascending=[False, True]
    ).iloc[0]
    start = int(mixed_gap.start_site)
    end = int(mixed_gap.end_site)
    differs = case.disagreement[start:end]
    transitions = np.flatnonzero(differs[1:] != differs[:-1]) + 1
    anchor = start + int(transitions[0])
    targets.insert(2, ("Mixed gap fill", start, end, anchor))
    windows = []
    context = config["detail"]["context_sites"]
    max_sites = config["detail"]["max_sites"]
    for kind, start, end, anchor in targets:
        if end - start + 2 * context <= max_sites:
            left = max(0, start - context)
            right = min(len(case.covered), end + context)
        else:
            left = max(0, anchor - context)
            right = min(len(case.covered), left + max_sites)
        windows.append(DetailWindow(kind, start, end, left, right))
    return windows


def path_diagnostics_seam_distances(covered: np.ndarray) -> np.ndarray:
    """Compute distances to the two adjacent sites of the nearest provenance seam."""
    seams = np.flatnonzero(covered[1:] != covered[:-1]) + 1
    sites = np.arange(len(covered))
    insertion = np.searchsorted(seams, sites, side="right")
    left = np.full(len(sites), np.inf)
    right = np.full(len(sites), np.inf)
    has_left = insertion > 0
    has_right = insertion < len(seams)
    left[has_left] = sites[has_left] - seams[insertion[has_left] - 1]
    right[has_right] = seams[insertion[has_right]] - 1 - sites[has_right]
    return np.minimum(left, right)


def ternary_coordinates(proportions: np.ndarray) -> np.ndarray:
    """Map focal/gap/seam proportions to left/right/top triangle vertices."""
    x = proportions[:, 1] + proportions[:, 2] / 2
    y = proportions[:, 2] * np.sqrt(3) / 2
    return np.column_stack((x, y))


@dataclasses.dataclass
class ParentSequence:
    """Ordered parent IDs and the site boundaries between them."""

    nodes: np.ndarray
    switch_sites: np.ndarray


@dataclasses.dataclass
class OutlierPathAnalysis:
    """Gap evidence, parent differences, proven relocations, and switch totals."""

    gaps: pd.DataFrame
    divergences: pd.DataFrame
    relocations: pd.DataFrame
    switch_counts: pd.DataFrame


def parent_sequence(parents: np.ndarray, start: int, end: int) -> ParentSequence:
    """Compress parents on [start, end), retaining absolute switch sites."""
    changes = np.flatnonzero(parents[start + 1 : end] != parents[start : end - 1])
    switch_sites = changes + start + 1
    run_starts = np.concatenate(([start], switch_sites))
    return ParentSequence(parents[run_starts], switch_sites)


def switch_location(case: PathDiagnosticsHaplotypeDiagnostics, site: int) -> str:
    """Classify the boundary immediately before a given inference site."""
    if case.covered[site] != case.covered[site - 1]:
        return "seam"
    return "focal" if case.covered[site] else "gap"


def analyse_outlier_paths(
    dataset: PathDiagnosticsDatasetDiagnostics,
    *,
    plot_outliers_transition_kinds=[
        "Identical switch",
        "Shifted parent sequence",
        "Changed parent sequence",
        "Unanchored difference",
    ],
) -> OutlierPathAnalysis:
    """Distinguish positional changes from changes in the ordered parent sequence.

    Maximal disagreement runs are compared with one agreeing site on either
    side. Equal compressed parent sequences prove preservation of the directed
    parent transitions within that run. Differences reaching the chromosome's
    inference-site boundary are reported separately because a shared flank is
    absent. Counts partition every switch; no nearest-switch pairing is used.
    """
    case = dataset.primary
    gaps = path_diagnostics_gap_diagnostics(dataset)
    full_changes = case.full_parents[1:] != case.full_parents[:-1]
    stitched_changes = case.stitched_parents[1:] != case.stitched_parents[:-1]
    matching_edges = ~case.disagreement[1:] & ~case.disagreement[:-1]
    identical = full_changes & stitched_changes & matching_edges
    kinds = plot_outliers_transition_kinds
    counts = np.zeros((len(kinds), 2), dtype=int)
    counts[0] = np.count_nonzero(identical)
    padded = np.concatenate(([False], case.disagreement, [False]))
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    runs = boundaries.reshape(-1, 2)
    difference_classes = np.zeros(len(case.covered), dtype=np.int8)
    divergence_rows = []
    relocation_rows = []
    for index, (start, end) in enumerate(runs):
        left = max(0, int(start) - 1)
        right = min(len(case.covered), int(end) + 1)
        full = parent_sequence(case.full_parents, left, right)
        stitched = parent_sequence(case.stitched_parents, left, right)
        shared_flanks = start > 0 and end < len(case.covered)
        equal_sequences = np.array_equal(full.nodes, stitched.nodes)
        if not shared_flanks:
            kind_index = 3
        elif equal_sequences:
            kind_index = 1
        else:
            kind_index = 2
        counts[kind_index, 0] += len(full.switch_sites)
        counts[kind_index, 1] += len(stitched.switch_sites)
        difference_classes[start:end] = kind_index
        divergence_rows.append(
            {
                "difference": index,
                "start_site": int(start),
                "end_site": int(end),
                "start_bp": dataset.positions[start],
                "end_bp": dataset.site_edges[end],
                "num_sites": int(end - start),
                "focal_sites": int(np.count_nonzero(case.covered[start:end])),
                "gap_sites": int(np.count_nonzero(~case.covered[start:end])),
                "kind": kinds[kind_index],
                "full_parent_sequence": " → ".join((str(node) for node in full.nodes)),
                "stitched_parent_sequence": " → ".join(
                    (str(node) for node in stitched.nodes)
                ),
                "full_switches": len(full.switch_sites),
                "stitched_switches": len(stitched.switch_sites),
                "extra_switches": len(stitched.switch_sites) - len(full.switch_sites),
            }
        )
        if shared_flanks and equal_sequences:
            for offset, (full_site, stitched_site) in enumerate(
                zip(full.switch_sites, stitched.switch_sites, strict=True)
            ):
                relocation_rows.append(
                    {
                        "difference": index,
                        "from_parent": int(full.nodes[offset]),
                        "to_parent": int(full.nodes[offset + 1]),
                        "full_site": int(full_site),
                        "stitched_site": int(stitched_site),
                        "full_bp": dataset.positions[full_site],
                        "stitched_bp": dataset.positions[stitched_site],
                        "full_location": switch_location(case, int(full_site)),
                        "stitched_location": switch_location(case, int(stitched_site)),
                        "shift_sites": int(stitched_site - full_site),
                    }
                )
    assert counts[:, 0].sum() == np.count_nonzero(full_changes)
    assert counts[:, 1].sum() == np.count_nonzero(stitched_changes)
    switch_rows = []
    for kind, values in zip(kinds, counts, strict=True):
        switch_rows.append({"kind": kind, "full": values[0], "stitched": values[1]})
    for method in ("full", "stitched"):
        interior = gaps[f"{method}_interior_switches"]
        left_seams = gaps[f"{method}_left_seam_switch"].fillna(False).astype(int)
        right_seams = gaps[f"{method}_right_seam_switch"].fillna(False).astype(int)
        gaps[f"{method}_gap_and_seam_switches"] = interior + left_seams + right_seams
        counts_by_source = path_diagnostics_switch_breakdown(
            case, getattr(case, f"{method}_parents")
        )
        assert (
            gaps[f"{method}_gap_and_seam_switches"].sum() == counts_by_source[1:].sum()
        )
    gaps["extra_interior_switches"] = (
        gaps.stitched_interior_switches - gaps.full_interior_switches
    )
    gaps["extra_gap_and_seam_switches"] = (
        gaps.stitched_gap_and_seam_switches - gaps.full_gap_and_seam_switches
    )
    gaps["extra_switches_per_site"] = gaps.extra_gap_and_seam_switches / gaps.num_sites
    for row in gaps.itertuples():
        start, end = (row.start_site, row.end_site)
        for side, site in (("left", start - 1), ("right", end)):
            parent_agrees = None
            if 0 <= site < len(case.covered):
                parent_agrees = bool(not case.disagreement[site])
            gaps.loc[row.Index, f"{side}_flank_parent_agrees"] = parent_agrees
        for kind_index, label in ((1, "shifted"), (2, "changed"), (3, "unanchored")):
            classes = difference_classes[start:end]
            gaps.loc[row.Index, f"{label}_sequence_sites"] = np.count_nonzero(
                classes == kind_index
            )
    relocations = pd.DataFrame(
        relocation_rows,
        columns=[
            "difference",
            "from_parent",
            "to_parent",
            "full_site",
            "stitched_site",
            "full_bp",
            "stitched_bp",
            "full_location",
            "stitched_location",
            "shift_sites",
        ],
    )
    return OutlierPathAnalysis(
        gaps, pd.DataFrame(divergence_rows), relocations, pd.DataFrame(switch_rows)
    )


def outlier_window(
    dataset: PathDiagnosticsDatasetDiagnostics,
    kind: str,
    start: int,
    end: int,
    anchor: int,
    *,
    config,
) -> DetailWindow:
    """Apply the shared context and size limits to an outlier target."""
    context = config["detail"]["context_sites"]
    max_sites = config["detail"]["max_sites"]
    if end - start + 2 * context <= max_sites:
        left = max(0, start - context)
        right = min(len(dataset.positions), end + context)
    else:
        left = max(0, anchor - context)
        right = min(len(dataset.positions), left + max_sites)
    return DetailWindow(kind, start, end, left, right)


def select_outlier_windows(
    dataset: PathDiagnosticsDatasetDiagnostics,
    analysis: OutlierPathAnalysis,
    *,
    config,
    plot_outliers_transition_kinds_2="Changed parent sequence",
) -> list[DetailWindow]:
    """Find short-gap excesses, poor agreement, altered itineraries, and relocations."""
    case = dataset.primary
    gaps = analysis.gaps
    windows = []
    short = gaps.loc[
        gaps.num_sites.le(config["outliers"]["short_gap_max_sites"])
        & gaps.extra_gap_and_seam_switches.gt(0)
    ]
    if not short.empty:
        chosen = short.sort_values(
            ["extra_switches_per_site", "extra_gap_and_seam_switches", "start_site"],
            ascending=[False, False, True],
        ).iloc[0]
        start, end = (int(chosen.start_site), int(chosen.end_site))
        windows.append(
            outlier_window(
                dataset,
                "Short gap with extra switches",
                start,
                end,
                start,
                config=config,
            )
        )
    poor = gaps.sort_values(
        ["agreement_fraction", "differ_sites", "start_site"],
        ascending=[True, False, True],
    ).iloc[0]
    start, end = (int(poor.start_site), int(poor.end_site))
    differences = np.flatnonzero(case.disagreement[start:end])
    anchor = start + int(differences[0])
    windows.append(
        outlier_window(
            dataset, "Lowest gap agreement", start, end, anchor, config=config
        )
    )
    focal_difference = case.covered & case.disagreement
    padded = np.concatenate(([False], focal_difference, [False]))
    boundaries = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    lengths = boundaries[:, 1] - boundaries[:, 0]
    start, end = boundaries[np.argmax(lengths)]
    windows.append(
        outlier_window(
            dataset,
            "Longest focal parent disagreement",
            int(start),
            int(end),
            int(start),
            config=config,
        )
    )
    changed = analysis.divergences.loc[
        analysis.divergences.kind.eq(plot_outliers_transition_kinds_2)
        & analysis.divergences.gap_sites.gt(0)
    ]
    if not changed.empty:
        chosen = changed.sort_values(
            ["extra_switches", "gap_sites", "start_site"],
            ascending=[False, False, True],
        ).iloc[0]
        start, end = (int(chosen.start_site), int(chosen.end_site))
        intersecting_gaps = gaps.loc[gaps.start_site.lt(end) & gaps.end_site.gt(start)]
        target_gap = intersecting_gaps.sort_values(
            ["extra_gap_and_seam_switches", "start_site"], ascending=[False, True]
        ).iloc[0]
        anchor = max(start, int(target_gap.start_site))
        windows.append(
            outlier_window(
                dataset,
                "Changed parent sequence around a gap",
                start,
                end,
                anchor,
                config=config,
            )
        )
    relocations = analysis.relocations.copy()
    if not relocations.empty:
        relocations["focal_to_gap_or_seam"] = relocations.full_location.eq(
            "focal"
        ) & relocations.stitched_location.ne("focal")
        relocations["absolute_shift"] = relocations.shift_sites.abs()
        chosen = relocations.sort_values(
            ["focal_to_gap_or_seam", "absolute_shift", "full_site"],
            ascending=[False, False, True],
        ).iloc[0]
        start = min(int(chosen.full_site), int(chosen.stitched_site))
        end = max(int(chosen.full_site), int(chosen.stitched_site)) + 1
        context = config["detail"]["context_sites"]
        left = max(0, start - context)
        right = min(len(case.covered), end + context)
        windows.append(
            DetailWindow(
                "Same parents with a relocated switch", start, end, left, right
            )
        )
    unique_windows = {}
    for window in windows:
        key = (window.start, window.end)
        if key not in unique_windows:
            unique_windows[key] = window
    return list(unique_windows.values())[
        : config["outliers"]["max_windows_per_haplotype"]
    ]


def cutoff_label(cutoff, dataset):
    """Show the AF cutoff and its rounded AC in the actual haplotype cohort."""
    num_haplotypes = len(dataset["cohort"])
    ac_cutoff = round(cutoff * num_haplotypes)
    return f"AF ≤ {cutoff:g} (AC {ac_cutoff:,})"


def all_mutation_rank_correlations(*, config, datasets, output_dir):
    runs_table = pd.read_csv(output_dir / "dating_summary.csv")
    paths = [pathlib.Path(path) for path in runs_table.dated_path]
    paths.extend((config["data_dir"] / dataset["truth"] for dataset in datasets))
    signature = input_signature(paths, {"statistic": "spearman_all_common_mutations"})
    output = output_dir / "mutation_age_spearman_correlations.csv"
    cached = cached_table(output, signature)
    if cached is not None:
        return cached
    rows = []
    for dataset in datasets:
        runs = runs_table.loc[runs_table.dataset.eq(dataset["name"])].reset_index(
            drop=True
        )
        truth = tszip.decompress(config["data_dir"] / dataset["truth"])
        truth = window_tree_sequence(truth, dataset["analysis_extent_bp"])
        true_ages = mutation_ages(truth, posterior=False)["age"]
        del truth
        ages = []
        common = true_ages.index
        for run in runs.itertuples():
            tree_sequence = tskit.load(run.dated_path)
            inferred = mutation_ages(tree_sequence, posterior=True)["age"]
            ages.append(inferred)
            common = common.intersection(inferred.index, sort=False)
            del tree_sequence
        common = common.sort_values()
        full_index = runs.index[runs.comparison_method.eq("full_hmm")].item()
        references = {"true_arg": true_ages, "full_hmm": ages[full_index]}
        for reference_method, reference in references.items():
            reference_values = reference.loc[common].to_numpy()
            for index, run in runs.iterrows():
                if (
                    reference_method == "full_hmm"
                    and run.comparison_method == "full_hmm"
                ):
                    continue
                inferred_values = ages[index].loc[common].to_numpy()
                correlation = stats.spearmanr(
                    inferred_values, reference_values
                ).statistic
                rows.append(
                    {
                        "dataset": run.dataset,
                        "comparison_method": run.comparison_method,
                        "af_cutoff": run.af_cutoff,
                        "min_focal_region_size": run.min_focal_region_size,
                        "reference_method": reference_method,
                        "frequency_bin": "all",
                        "matched_mutations": len(common),
                        "spearman_rho": correlation,
                    }
                )
        logger.info(
            "Spearman correlations: %s (%d shared mutations)",
            dataset["name"],
            len(common),
        )
    table = pd.DataFrame(rows)
    save_table(table, output, signature)
    return table


@dataclasses.dataclass
class ChunkDatasetSettings:
    """Picklable settings passed to each dataset's worker initializer."""

    name: str
    input_dir: pathlib.Path
    coverage_input_dir: pathlib.Path
    ac_cutoff: int
    min_focal_ac: int
    score_tolerance: float


@dataclasses.dataclass
class ChunkFocalIntervals:
    """AC-filtered anchored associations in canonical haplotype/ancestor order."""

    offsets: np.ndarray
    left: np.ndarray
    right: np.ndarray
    nodes: np.ndarray


@dataclasses.dataclass
class ChunkWorkerContext:
    """Original reference, observations, and reusable full-model indexes."""

    settings: ChunkDatasetSettings
    reference: tskit.TreeSequence
    positions: np.ndarray
    haplotypes: np.ndarray
    focal: dict[str, np.ndarray]
    intervals: ChunkFocalIntervals
    populations: list[str]
    rho: np.ndarray
    mu: np.ndarray
    indexes: tsinfer.matching.MatcherIndexes
    log_mismatch_penalty: float
    log_switch_penalty: float


@dataclasses.dataclass
class ChunkMatchArrays:
    """Parents and canonical copied alleles on the requested site interval."""

    parents: np.ndarray
    copied: np.ndarray


@dataclasses.dataclass
class ChunkFocalMatch:
    """Transient focal arrays and the anchored interval union."""

    arrays: ChunkMatchArrays
    mask: np.ndarray


@dataclasses.dataclass
class ChunkJobResult:
    """The only two saved matches and their scalar comparison."""

    full: dict
    stitched: dict
    comparison: dict


def chunk_load_focal(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read the candidate identities and ragged panel-column lists without pickle."""
    with np.load(path, allow_pickle=False) as archive:
        focal = {name: archive[name] for name in archive.files}
    offsets = focal["offsets"]
    columns = focal["ancestor_index"]
    num_rows = len(focal["sample_id"])
    assert focal["ploidy_index"].shape == (num_rows,)
    assert offsets.shape == (num_rows + 1,)
    assert offsets[0] == 0 and offsets[-1] == len(columns)
    assert np.all(np.diff(offsets) >= 0)
    num_ancestors = len(focal["ancestor_id"])
    assert len(np.unique(focal["ancestor_id"])) == num_ancestors
    assert focal["derived_ac"].shape == (num_ancestors,)
    assert np.all(focal["derived_ac"] > 0)
    assert np.all((columns >= 0) & (columns < num_ancestors))
    return focal


def chunk_select_rows(focal: dict, requested: list[dict] | None) -> list[int]:
    """Resolve explicit haplotypes, always preserving the stored NPZ row order."""
    keys = list(zip(focal["sample_id"], focal["ploidy_index"], strict=True))
    assert len(set(keys)) == len(keys), "Duplicate focal haplotype identities"
    if requested is None:
        return list(range(len(keys)))
    requested_keys = [(item["sample_id"], item["ploidy_index"]) for item in requested]
    if len(set(requested_keys)) != len(requested_keys):
        raise ValueError("Duplicate haplotypes in the YAML selection")
    unknown = set(requested_keys) - set(keys)
    if unknown:
        raise ValueError(f"Haplotypes absent from the focal NPZ: {sorted(unknown)}")
    requested_set = set(requested_keys)
    rows = [row for row, key in enumerate(keys) if key in requested_set]
    if len(rows) == 0:
        raise ValueError("Select at least one haplotype")
    return rows


def chunk_load_focal_intervals(
    settings: ChunkDatasetSettings,
    focal: dict,
    panel,
    positions: np.ndarray,
    column_nodes: np.ndarray,
) -> ChunkFocalIntervals:
    """Join budget-zero bounds to original nodes and apply the configured AC range.

    The upstream archive stores associations in increasing ancestor-column order,
    excluding ancestors without focal seeds and those above its generation AC
    limit. Reconstruct that ordering before applying the analysis cutoff.
    """
    interval_path = settings.coverage_input_dir / "haplotype_intervals"
    interval_path /= f"{settings.name}_inferred_focal_ancestor_intervals.npz"
    with np.load(interval_path, allow_pickle=False) as archive:
        interval_offsets = archive["offsets"]
        interval_ac = archive["focal_ac"]
        interval_left = archive["left_site_index"][:, 0]
        interval_right = archive["right_site_index"][:, 0]
        generated_ac = int(archive["max_ac_cutoff"])
        assert settings.ac_cutoff <= generated_ac
        assert int(archive["num_sites"]) == len(positions)
        for field in ("sample_id", "ploidy_index"):
            assert np.array_equal(archive[field], focal[field])
    coverage_panel = tsinfer.vcz.open_store(
        settings.coverage_input_dir
        / "ancestors"
        / f"{settings.name}_inferred_ancestors.zarr"
    )
    assert np.array_equal(coverage_panel["variant_position"][:], positions)
    assert np.array_equal(coverage_panel["sample_id"][:], panel["sample_id"][:])
    focal_positions = panel["sample_focal_positions"][:]
    has_anchor = np.any(focal_positions >= 0, axis=1)
    association_columns = []
    for row in range(len(focal["sample_id"])):
        columns = focal["ancestor_index"][
            focal["offsets"][row] : focal["offsets"][row + 1]
        ]
        eligible = (focal["derived_ac"][columns] <= generated_ac) & has_anchor[columns]
        columns = columns[eligible]
        assert np.all(np.diff(columns) > 0), (
            "Interval associations require canonical ancestor order"
        )
        assert len(columns) == interval_offsets[row + 1] - interval_offsets[row]
        association_columns.extend(columns)
    association_columns = np.asarray(association_columns, dtype=np.int64)
    assert np.array_equal(focal["derived_ac"][association_columns], interval_ac)
    interval_nodes = column_nodes[association_columns]
    eligible = (interval_ac >= settings.min_focal_ac) & (
        interval_ac <= settings.ac_cutoff
    )
    cumulative = np.concatenate(([0], np.cumsum(eligible)))
    interval_offsets = cumulative[interval_offsets]
    interval_left = interval_left[eligible]
    interval_right = interval_right[eligible]
    interval_nodes = interval_nodes[eligible]
    return ChunkFocalIntervals(
        interval_offsets, interval_left, interval_right, interval_nodes
    )


def chunk_load_context(settings: ChunkDatasetSettings) -> ChunkWorkerContext:
    """Validate local inputs and prepare the common model once per worker.

    Sample observations come from the masked VCZ, preserving missing calls. Panel
    columns join to original nodes through metadata, never by assuming node offsets.
    """
    base = settings.input_dir
    name = settings.name
    reference = tskit.load(base / "ancestors" / f"{name}_inferred_ancestors.trees")
    panel = tsinfer.vcz.open_store(
        base / "ancestors" / f"{name}_inferred_ancestors.zarr"
    )
    samples = tsinfer.vcz.open_store(base / "samples" / f"{name}_samples_masked.zarr")
    focal = chunk_load_focal(
        base / "focal_ancestors" / f"{name}_inferred_focal_ancestors.npz"
    )
    with (base / "configs" / f"{name}_ancestor_inference.toml").open("rb") as file:
        native_config = tomllib.load(file)
    assert native_config["match"]["path_compression"] is False
    intervals = reference.metadata["sequence_intervals"]
    assert len(intervals) == 1, "This experiment requires one inference interval"
    positions = reference.sites_position
    assert len(positions) > 0 and np.all(np.diff(positions) > 0)
    assert np.array_equal(positions, panel["variant_position"][:])
    assert np.array_equal(intervals, panel["sequence_intervals"][:])
    ancestral = np.asarray([site.ancestral_state for site in reference.sites()])
    panel_alleles = panel["variant_allele"][:]
    assert panel_alleles.shape == (len(positions), 2)
    assert np.array_equal(panel_alleles[:, 0], ancestral)
    assert np.all(panel_alleles[:, 0] != panel_alleles[:, 1])
    assert np.all(panel_alleles != "")
    for mutation in reference.mutations():
        assert mutation.derived_state in panel_alleles[mutation.site]
    panel_ids = np.asarray(panel["sample_id"][:].tolist(), dtype=str)
    assert np.array_equal(panel_ids, focal["ancestor_id"])
    ancestor_columns = {sample_id: column for column, sample_id in enumerate(panel_ids)}
    column_nodes = np.full(len(panel_ids), tskit.NULL, dtype=np.int32)
    assert reference.num_nodes == len(panel_ids) + 2
    assert reference.node(0).metadata == reference.node(1).metadata == {}
    assert reference.node(0).time > reference.node(1).time
    for node in reference.nodes():
        if node.id < 2:
            continue
        metadata = node.metadata
        assert metadata["source"] == "ancestors" and metadata["ploidy_index"] == 0
        column = ancestor_columns[str(metadata["sample_id"])]
        assert column_nodes[column] == tskit.NULL, "Duplicate ancestor metadata join"
        column_nodes[column] = node.id
    assert np.all(column_nodes >= 2), "An ancestor is absent from the reference"
    intervals = chunk_load_focal_intervals(
        settings, focal, panel, positions, column_nodes
    )
    populations = chunk_load_populations(settings, focal)
    source = next(
        (item for item in native_config["source"] if item["name"] == "samples")
    )
    sample_columns = tsinfer.vcz.resolve_samples_selection(
        samples, source.get("samples")
    )
    sample_ids = np.asarray(
        samples["sample_id"].oindex[sample_columns].tolist(), dtype=str
    )
    ploidy = samples["call_genotype"].shape[2]
    assert np.array_equal(focal["sample_id"], np.repeat(sample_ids, ploidy))
    assert np.array_equal(
        focal["ploidy_index"], np.tile(np.arange(ploidy), len(sample_ids))
    )
    sample_positions = samples["variant_position"][:]
    sample_rows = np.searchsorted(sample_positions, positions)
    assert np.all(sample_rows < len(sample_positions)), (
        "Reference sites absent in samples"
    )
    assert np.array_equal(sample_positions[sample_rows], positions)
    sample_alleles = samples["variant_allele"].oindex[sample_rows, :]
    assert np.all(np.count_nonzero(sample_alleles != "", axis=1) == 2)
    allowed = sample_alleles == panel_alleles[:, 0, None]
    allowed |= sample_alleles == panel_alleles[:, 1, None]
    allowed |= sample_alleles == ""
    assert np.all(allowed), "Sample and panel allele strings disagree"
    calls = samples["call_genotype"].oindex[sample_rows, sample_columns, :]
    assert np.all((calls >= -1) & (calls < sample_alleles.shape[1]))
    codes = np.where(sample_alleles == ancestral[:, None], 0, 1).astype(np.int8)
    codes[sample_alleles == ""] = -1
    lookup_calls = np.maximum(calls, 0)
    haplotypes = np.take_along_axis(codes[:, None, :], lookup_calls, axis=2)
    assert np.all(haplotypes[calls >= 0] >= 0), "Called allele is an empty string"
    haplotypes[calls < 0] = -1
    haplotypes = haplotypes.reshape(len(positions), -1)
    parameters = native_config["match"]["sources"]["samples"]
    recombination = parameters["recombination"]
    mismatch = parameters["mismatch"]
    if not 0 < recombination < 1 or not 0 < mismatch < 1:
        raise ValueError(
            "Sample recombination and mismatch must lie strictly in (0, 1)"
        )
    rho = np.full(len(positions), recombination)
    mu = np.full(len(positions), mismatch)
    num_alleles = np.full(len(positions), 2, dtype=np.uint32)
    indexes = tsinfer.matching.MatcherIndexes(
        reference, vestigial_root=False, num_alleles=num_alleles
    )
    log_mismatch_penalty = math.log(mismatch) - math.log1p(-mismatch)
    switch_baseline = -recombination + recombination / reference.num_nodes
    log_switch_penalty = math.log(recombination) - math.log(reference.num_nodes)
    log_switch_penalty -= math.log1p(switch_baseline)
    return ChunkWorkerContext(
        settings=settings,
        reference=reference,
        positions=positions,
        haplotypes=haplotypes,
        focal=focal,
        intervals=intervals,
        populations=populations,
        rho=rho,
        mu=mu,
        indexes=indexes,
        log_mismatch_penalty=log_mismatch_penalty,
        log_switch_penalty=log_switch_penalty,
    )


def chunk_initialize_worker(settings: ChunkDatasetSettings) -> None:
    """Keep a separate reusable context in each multiprocessing worker."""
    global chunk_worker_context
    chunk_worker_context = chunk_load_context(settings)


def chunk_site_parents(
    left: np.ndarray,
    right: np.ndarray,
    parents: np.ndarray,
    positions: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Convert absolute genomic path bounds to parents on [start, end)."""
    order = np.argsort(left)
    left = np.asarray(left)[order]
    right = np.asarray(right)[order]
    parents = np.asarray(parents)[order]
    assert len(left) > 0 and np.all(left < right)
    assert np.array_equal(right[:-1], left[1:]), "Path segments are not contiguous"
    starts = np.searchsorted(positions, left)
    ends = np.searchsorted(positions, right)
    assert starts[0] == start and ends[-1] == end
    result = np.full(end - start, tskit.NULL, dtype=np.int32)
    for lo, hi, parent in zip(starts, ends, parents, strict=True):
        assert start <= lo < hi <= end
        result[lo - start : hi - start] = parent
    assert np.all(result >= 0)
    return result


def chunk_run_match(
    matcher: tsinfer.matching.AncestorMatcher,
    haplotype: np.ndarray,
    start: int,
    end: int,
    buffer: np.ndarray,
    positions: np.ndarray,
) -> ChunkMatchArrays:
    """Copy requested match results before the next call overwrites native buffers."""
    left, right, parents = matcher.find_path(haplotype, start, end, buffer)
    site_path = chunk_site_parents(left, right, parents, positions, start, end)
    copied = buffer[start:end].copy()
    assert np.all((copied == 0) | (copied == 1))
    return ChunkMatchArrays(site_path, copied)


def chunk_focal_match(
    context: ChunkWorkerContext, haplotype: np.ndarray, row: int
) -> ChunkFocalMatch:
    """Fix an interval-supported focal path; gaps are exactly its uncovered sites.

    Parent selection uses :func:`nb_utils.focal_parent_path`. Copied alleles
    are read from the inferred reference, rather than assumed from the interval
    generator, so any inferred/generated difference remains visible as a mutation.
    """
    start = context.intervals.offsets[row]
    end = context.intervals.offsets[row + 1]
    parents = focal_parent_path(
        len(haplotype),
        context.intervals.left[start:end],
        context.intervals.right[start:end],
        context.intervals.nodes[start:end],
    )
    covered = parents >= 0
    copied = np.full(len(haplotype), -1, dtype=np.int8)
    nodes = np.unique(parents[covered])
    if len(nodes) > 0:
        genotypes = context.reference.genotype_matrix(
            samples=nodes, isolated_as_missing=False
        )
        sites = np.flatnonzero(covered)
        columns = np.searchsorted(nodes, parents[covered])
        copied[covered] = genotypes[sites, columns]
        assert np.all((copied[covered] == 0) | (copied[covered] == 1))
    return ChunkFocalMatch(ChunkMatchArrays(parents, copied), covered)


def chunk_load_populations(settings: ChunkDatasetSettings, focal: dict) -> list[str]:
    """Join the upstream population metadata by haplotype identity."""
    path = settings.coverage_input_dir / "dataframes"
    path /= f"{settings.name}_inferred_focal_ancestor_stats.csv"
    labels = {}
    with path.open() as file:
        for record in csv.DictReader(file):
            key = (record["sample_id"], int(record["ploidy_index"]))
            population = record["population"]
            assert population != ""
            if key in labels:
                assert labels[key] == population
            labels[key] = population
    keys = zip(focal["sample_id"], focal["ploidy_index"], strict=True)
    return [labels[str(sample), int(ploidy)] for sample, ploidy in keys]


def chunk_match_record(
    key: dict,
    result: ChunkMatchArrays,
    haplotype: np.ndarray,
    context: ChunkWorkerContext,
) -> dict:
    """Merge site-parent runs and derive mutations from the finished copied alleles.

    Site changes occur at inference-site positions; outer bounds follow the engine's
    whole-sequence convention. Reconstructing this saved path must recover every site.
    """
    assert result.parents.shape == result.copied.shape == haplotype.shape
    assert np.all(
        (result.parents >= 0) & (result.parents < context.reference.num_nodes)
    )
    assert np.all((result.copied == 0) | (result.copied == 1))
    changes = np.flatnonzero(result.parents[1:] != result.parents[:-1]) + 1
    starts = np.concatenate(([0], changes))
    bounds = np.concatenate(
        ([0.0], context.positions[changes], [context.reference.sequence_length])
    )
    path = [
        {
            "left": float(left),
            "right": float(right),
            "parent": int(result.parents[start]),
        }
        for start, left, right in zip(starts, bounds[:-1], bounds[1:], strict=True)
    ]
    reconstructed = chunk_site_parents(
        bounds[:-1],
        bounds[1:],
        result.parents[starts],
        context.positions,
        0,
        len(haplotype),
    )
    assert np.array_equal(reconstructed, result.parents)
    assert len(path) - 1 == np.count_nonzero(result.parents[1:] != result.parents[:-1])
    mismatches = (haplotype >= 0) & (haplotype != result.copied)
    mutation_sites = np.flatnonzero(mismatches)
    mutations = [
        {
            "position": float(context.positions[site]),
            "derived_state": int(haplotype[site]),
        }
        for site in mutation_sites
    ]
    assert len(mutations) == np.count_nonzero(mismatches)
    return {**key, "path": path, "mutations": mutations}


def chunk_compare_matches(
    full: dict,
    stitched: dict,
    full_arrays: ChunkMatchArrays,
    stitched_arrays: ChunkMatchArrays,
    covered: np.ndarray,
    num_focal: int,
    context: ChunkWorkerContext,
) -> dict:
    """Compare both complete paths and score them with the original full model."""
    metrics = comparison_metrics(full_arrays.parents, stitched_arrays.parents, covered)
    full_mismatches = len(full["mutations"])
    stitched_mismatches = len(stitched["mutations"])
    full_switches = len(full["path"]) - 1
    stitched_switches = len(stitched["path"]) - 1
    assert full_switches == metrics["full_switches"]
    assert stitched_switches == metrics["stitched_switches"]
    full_score = full_mismatches * context.log_mismatch_penalty
    full_score += full_switches * context.log_switch_penalty
    stitched_score = stitched_mismatches * context.log_mismatch_penalty
    stitched_score += stitched_switches * context.log_switch_penalty
    score_delta = stitched_score - full_score
    key = {
        name: full[name]
        for name in (
            "dataset",
            "source",
            "sample_id",
            "ploidy_index",
            "ac_cutoff",
            "population",
        )
    }
    comparison = {
        **key,
        "num_focal_ancestors": num_focal,
        "num_sites": len(covered),
        **metrics,
        "num_gaps": len(stitched["gap_site_intervals"]),
        "mutations_identical": full["mutations"] == stitched["mutations"],
        "full_mismatches": full_mismatches,
        "stitched_mismatches": stitched_mismatches,
        "full_score": full_score,
        "stitched_score": stitched_score,
        "score_delta": score_delta,
        "scores_equal": abs(score_delta) <= context.settings.score_tolerance,
    }
    if score_delta > 0:
        logger.warning(
            "Positive score delta %s for %s; check full-model assumptions before interpreting this result (n=%s, rho=%s, mu=%s)",
            score_delta,
            key,
            context.reference.num_nodes,
            context.rho[0],
            context.mu[0],
        )
    return comparison


def chunk_match_haplotype(row: int) -> ChunkJobResult:
    """Cover anchored focal intervals, fill zero-coverage gaps, and run the full HMM."""
    context = chunk_worker_context
    focal = context.focal
    haplotype = np.ascontiguousarray(context.haplotypes[:, row])
    focal_result = chunk_focal_match(context, haplotype, row)
    stitched_arrays = focal_result.arrays
    covered = focal_result.mask
    num_focal = context.intervals.offsets[row + 1] - context.intervals.offsets[row]
    focal_intervals = site_intervals(covered)
    gaps = site_intervals(~covered)
    coverage = np.zeros(len(haplotype), dtype=np.int8)
    for intervals in (focal_intervals, gaps):
        for start, end in intervals:
            coverage[start:end] += 1
    assert np.all(coverage == 1), "Focal coverage and gaps must partition all sites"
    buffer = np.empty(len(haplotype), dtype=np.int8)
    gap_matcher = tsinfer.matching.AncestorMatcher(
        context.indexes, context.rho, context.mu
    )
    for start, end in gaps:
        gap = chunk_run_match(
            gap_matcher, haplotype, int(start), int(end), buffer, context.positions
        )
        stitched_arrays.parents[start:end] = gap.parents
        stitched_arrays.copied[start:end] = gap.copied
    key = {
        "dataset": context.settings.name,
        "source": "samples",
        "sample_id": str(focal["sample_id"][row]),
        "ploidy_index": int(focal["ploidy_index"][row]),
        "ac_cutoff": context.settings.ac_cutoff,
        "population": context.populations[row],
    }
    stitched = chunk_match_record(key, stitched_arrays, haplotype, context)
    stitched["focal_site_intervals"] = focal_intervals.tolist()
    stitched["gap_site_intervals"] = gaps.tolist()
    full_matcher = tsinfer.matching.AncestorMatcher(
        context.indexes, context.rho, context.mu
    )
    full_arrays = chunk_run_match(
        full_matcher, haplotype, 0, len(haplotype), buffer, context.positions
    )
    full = chunk_match_record(key, full_arrays, haplotype, context)
    comparison = chunk_compare_matches(
        full, stitched, full_arrays, stitched_arrays, covered, int(num_focal), context
    )
    return ChunkJobResult(full, stitched, comparison)


def chunk_write_results(results, output_dir: pathlib.Path) -> None:
    """Stream paths and scalar cohort diagnostics in canonical NPZ order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        (output_dir / "full_matches.jsonl").open("w") as full_file,
        (output_dir / "stitched_matches.jsonl").open("w") as stitched_file,
        (output_dir / "comparison.csv").open("w", newline="") as comparison_file,
    ):
        writer = None
        for result in results:
            full_file.write(json.dumps(result.full, allow_nan=False) + "\n")
            stitched_file.write(json.dumps(result.stitched, allow_nan=False) + "\n")
            if writer is None:
                writer = csv.DictWriter(
                    comparison_file, fieldnames=list(result.comparison)
                )
                writer.writeheader()
            writer.writerow(result.comparison)
            comparison = result.comparison
            logger.info(
                "%s %s/%s: covered=%s gaps=%s agreement=%.4f extra_switch_fraction=%s",
                comparison["dataset"],
                comparison["sample_id"],
                comparison["ploidy_index"],
                comparison["focal_sites"],
                comparison["num_gaps"],
                comparison["fraction_parent_agreement"],
                comparison["extra_switch_fraction"],
            )


def chunk_run_dataset(
    settings: ChunkDatasetSettings,
    requested: list[dict] | None,
    workers: int | None,
    output_dir: pathlib.Path,
) -> None:
    """Queue whole haplotype jobs; compute the CPU default only at process launch."""
    focal_path = settings.input_dir / "focal_ancestors"
    focal_path /= f"{settings.name}_inferred_focal_ancestors.npz"
    rows = chunk_select_rows(chunk_load_focal(focal_path), requested)
    if workers is None:
        workers = multiprocessing.cpu_count()
    workers = min(workers, len(rows))
    logger.info("%s: %s haplotypes using %s workers", settings.name, len(rows), workers)
    if workers == 1:
        chunk_initialize_worker(settings)
        results = map(chunk_match_haplotype, rows)
        chunk_write_results(results, output_dir)
    else:
        processes = multiprocessing.get_context("spawn")
        with processes.Pool(
            workers, initializer=chunk_initialize_worker, initargs=(settings,)
        ) as pool:
            results = pool.imap(chunk_match_haplotype, rows, chunksize=1)
            chunk_write_results(results, output_dir)


@dataclasses.dataclass
class ComparisonDatasetContext:
    """Read-only inputs shared by all jobs for a dataset inside one worker."""

    name: str
    reference: tskit.TreeSequence
    positions: np.ndarray
    focal: dict
    intervals: dict
    masks: dict
    matches: dict
    full: dict
    old_greedy: dict
    observations: np.ndarray
    ancestor_nodes: dict
    association_nodes: np.ndarray
    rho: np.ndarray
    mu: np.ndarray
    num_alleles: np.ndarray
    populations: dict
    settings: dict


@dataclasses.dataclass
class ComparisonSitePath:
    """Original-reference copying parents and canonical alleles at sites."""

    parents: np.ndarray
    copied: np.ndarray


@dataclasses.dataclass
class ComparisonHaplotypeResult:
    """Compact cohort metrics, component diagnostics, and selected run records."""

    comparisons: list[dict]
    controls: dict
    components: list[dict]
    distances: list[dict]
    matches: dict


def comparison_read_archive(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def comparison_read_matches(path: pathlib.Path) -> dict:
    records = {}
    with path.open() as file:
        for line in file:
            record = json.loads(line)
            key = (record["sample_id"], record["ploidy_index"])
            assert key not in records
            record["path"].sort(key=lambda segment: segment["left"])
            records[key] = record
    return records


def comparison_site_parents(record: dict, reference: tskit.TreeSequence) -> np.ndarray:
    segments = record["path"]
    right = np.array([segment["right"] for segment in segments])
    parents = np.array([segment["parent"] for segment in segments], dtype=np.int32)
    indices = np.searchsorted(right, reference.sites_position, side="right")
    result = parents[indices]
    assert np.count_nonzero(result[1:] != result[:-1]) == len(segments) - 1
    return result


def comparison_read_observations(
    base: pathlib.Path, name: str, reference, focal: dict
) -> np.ndarray:
    """Read each cohort once, mapping REF/ALT to the reference ancestral state."""
    samples = zarr.open_group(
        base / "samples" / f"{name}_samples_masked.zarr", mode="r"
    )
    all_sample_ids = samples["sample_id"][:].tolist()
    sample_columns = {
        sample_id: column for column, sample_id in enumerate(all_sample_ids)
    }
    selected_ids = focal["sample_id"][::2].tolist()
    columns = [sample_columns[sample_id] for sample_id in selected_ids]
    assert np.array_equal(focal["sample_id"], np.repeat(selected_ids, 2))
    assert np.array_equal(focal["ploidy_index"], np.tile([0, 1], len(columns)))
    source_positions = samples["variant_position"][:]
    rows = np.searchsorted(source_positions, reference.sites_position)
    assert np.array_equal(source_positions[rows], reference.sites_position)
    alleles = samples["variant_allele"].oindex[rows, :]
    calls = samples["call_genotype"].oindex[rows, columns, :]
    ancestral = np.asarray([site.ancestral_state for site in reference.sites()])
    codes = np.where(alleles == ancestral[:, None], 0, 1).astype(np.int8)
    codes[alleles == ""] = -1
    observations = np.take_along_axis(codes[:, None, :], np.maximum(calls, 0), axis=2)
    observations[calls < 0] = -1
    return observations.reshape(reference.num_sites, -1)


def comparison_load_context(name: str) -> ComparisonDatasetContext:
    """Load once per worker; observations are memory-mapped from preparation."""
    settings = _comparison_settings["datasets"][name]
    base = _comparison_settings["input_dir"]
    reference = tskit.load(base / "ancestors" / f"{name}_inferred_ancestors.trees")
    focal = comparison_read_archive(
        base / "focal_ancestors" / f"{name}_inferred_focal_ancestors.npz"
    )
    intervals = comparison_read_archive(
        base / "haplotype_intervals" / f"{name}_inferred_focal_ancestor_intervals.npz"
    )
    full = comparison_read_matches(
        base / "matches" / f"{name}_inferred_samples_matches.jsonl"
    )
    masks = {}
    matches = {}
    for ac_cutoff in _comparison_settings["pipeline"]["ac_cutoff"]:
        for budget in _comparison_settings["pipeline"]["max_mismatches"]:
            prefix = f"{name}_inferred_ac{ac_cutoff}_k{budget}"
            masks[ac_cutoff, budget] = comparison_read_archive(
                base / "focal_masks" / f"{prefix}_focal_mask.npz"
            )
            matches[ac_cutoff, budget] = comparison_read_matches(
                base / "matches" / f"{prefix}_hmm_stitched_samples_matches.jsonl"
            )
    old_name = settings["saved_greedy_name"]
    old_reference = tskit.load(
        _comparison_settings["old_input_dir"]
        / "ancestors"
        / f"{old_name}_inferred_ancestors.trees"
    )
    assert reference.tables.equals(old_reference.tables, ignore_provenance=True)
    old_greedy = comparison_read_matches(
        _comparison_settings["old_output_dir"] / old_name / "stitched_matches.jsonl"
    )
    old_comparisons = pd.read_csv(
        _comparison_settings["old_output_dir"] / old_name / "comparison.csv"
    )
    populations = {
        (row.sample_id, row.ploidy_index): row.population
        for row in old_comparisons.itertuples()
    }
    observations = np.load(
        _comparison_settings["output_dir"] / f"{name}_observations.npy", mmap_mode="r"
    )
    ancestor_nodes = {}
    for node in reference.nodes():
        if node.metadata.get("source") == "ancestors":
            ancestor_nodes[node.metadata["sample_id"]] = node.id
    association_nodes = []
    for index in range(len(focal["sample_id"])):
        start, end = focal["offsets"][index : index + 2]
        columns = focal["ancestor_index"][start:end]
        columns = columns[
            focal["derived_ac"][columns] <= int(intervals["max_ac_cutoff"])
        ]
        assert (
            len(columns)
            == intervals["offsets"][index + 1] - intervals["offsets"][index]
        )
        association_nodes.extend(
            (ancestor_nodes[focal["ancestor_id"][column]] for column in columns)
        )
    with (base / "configs" / f"{name}_ancestor_inference.toml").open("rb") as file:
        native_config = tomllib.load(file)
    parameters = native_config["match"]["sources"]["samples"]
    return ComparisonDatasetContext(
        name,
        reference,
        reference.sites_position,
        focal,
        intervals,
        masks,
        matches,
        full,
        old_greedy,
        observations,
        ancestor_nodes,
        np.asarray(association_nodes, dtype=np.int32),
        np.full(reference.num_sites, parameters["recombination"]),
        np.full(reference.num_sites, parameters["mismatch"]),
        np.full(reference.num_sites, 2, dtype=np.uint32),
        populations,
        settings,
    )


def comparison_initialize_worker(settings: dict) -> None:
    global _comparison_settings, _comparison_contexts
    _comparison_settings = settings
    _comparison_contexts = {}


@dataclasses.dataclass
class ComparisonSiteRuns:
    """Covered components and gaps on the inference-site axis."""

    focal: np.ndarray
    gaps: np.ndarray


def comparison_mask_runs(mask: dict, index: int) -> ComparisonSiteRuns:
    start, end = mask["offsets"][index : index + 2]
    runs = np.column_stack(
        (mask["left_site_index"][start:end], mask["right_site_index"][start:end])
    )
    gaps = mask["is_gap"][start:end]
    return ComparisonSiteRuns(runs[~gaps], runs[gaps])


def comparison_coverage_array(runs: np.ndarray, num_sites: int) -> np.ndarray:
    covered = np.zeros(num_sites, dtype=bool)
    for left, right in runs:
        covered[left:right] = True
    return covered


def comparison_switch_metrics(
    parents: np.ndarray, mismatches: np.ndarray, covered: np.ndarray
) -> dict:
    changes = parents[1:] != parents[:-1]
    interior = {
        "focal": covered[1:] & covered[:-1],
        "gap": ~covered[1:] & ~covered[:-1],
        "seam": covered[1:] != covered[:-1],
    }
    counts = {
        f"{category}_switches": int(np.count_nonzero(changes & edges))
        for category, edges in interior.items()
    }
    counts["num_switches"] = int(np.count_nonzero(changes))
    assert (
        sum((counts[f"{category}_switches"] for category in interior))
        == counts["num_switches"]
    )
    counts["focal_mismatches"] = int(np.count_nonzero(mismatches & covered))
    counts["gap_mismatches"] = int(np.count_nonzero(mismatches & ~covered))
    counts["num_mismatches"] = int(np.count_nonzero(mismatches))
    counts["focal_sites"] = int(np.count_nonzero(covered))
    counts["gap_sites"] = len(covered) - counts["focal_sites"]
    return counts


def comparison_comparison_row(
    identity: dict,
    method: str,
    parents: np.ndarray,
    mismatches: np.ndarray,
    covered: np.ndarray,
    context: ComparisonDatasetContext,
) -> dict:
    """Score complete paths with one common, full-reference Li-Stephens model.

    Site-constant emissions and no-switch transitions cancel against the full
    path. The transition weight uses the C matcher's actual reference node count.
    """
    metrics = comparison_switch_metrics(parents, mismatches, covered)
    key = (identity["sample_id"], identity["ploidy_index"])
    full_record = context.full[key]
    full_switches = len(full_record["path"]) - 1
    extra_switches = metrics["num_switches"] - full_switches
    metrics["num_extra_switches"] = extra_switches
    metrics["switch_increase_percentage"] = (
        100 * extra_switches / full_switches if full_switches > 0 else np.nan
    )
    rho = float(context.rho[0])
    mu = float(context.mu[0])
    switch_probability = rho / context.reference.num_nodes
    stay_probability = 1 - rho + switch_probability
    switch_penalty = np.log(switch_probability / stay_probability)
    mismatch_penalty = np.log(mu / (1 - mu))
    extra_mismatches = metrics["num_mismatches"] - len(full_record["mutations"])
    metrics["full_model_log_likelihood_delta"] = (
        extra_switches * switch_penalty + extra_mismatches * mismatch_penalty
    )
    return {**identity, "implementation": method, **metrics}


def comparison_mutation_mask(record: dict, positions: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(positions), dtype=bool)
    sites = np.searchsorted(
        positions, [mutation["position"] for mutation in record["mutations"]]
    )
    mask[sites] = True
    return mask


def comparison_match_region(
    matcher, haplotype, left: int, right: int, positions: np.ndarray
) -> ComparisonSitePath:
    copied = np.empty(len(haplotype), dtype=np.int8)
    path_left, path_right, path_parent = matcher.find_path(
        haplotype, left, right, copied
    )
    parents = np.empty(right - left, dtype=np.int32)
    for start, end, node in zip(path_left, path_right, path_parent, strict=True):
        start_site = max(left, int(np.searchsorted(positions, start)))
        end_site = min(right, int(np.searchsorted(positions, end)))
        parents[start_site - left : end_site - left] = node
    return ComparisonSitePath(parents, copied[left:right].copy())


def comparison_encode_record(
    key: tuple,
    parents: np.ndarray,
    mismatch: np.ndarray,
    haplotype: np.ndarray,
    context: ComparisonDatasetContext,
) -> dict:
    changes = np.flatnonzero(parents[1:] != parents[:-1]) + 1
    starts = np.concatenate(([0], changes))
    bounds = np.concatenate(
        ([0], context.positions[changes], [context.reference.sequence_length])
    )
    path = [
        {"left": float(left), "right": float(right), "parent": int(parents[start])}
        for start, left, right in zip(starts, bounds[:-1], bounds[1:], strict=True)
    ]
    mutations = [
        {
            "position": float(context.positions[site]),
            "derived_state": int(haplotype[site]),
        }
        for site in np.flatnonzero(mismatch)
    ]
    return {
        "source": "samples",
        "sample_id": key[0],
        "ploidy_index": key[1],
        "path": path,
        "mutations": mutations,
    }


def comparison_distance_rows(
    identity: dict, paths: dict, covered: np.ndarray, config: dict
) -> list[dict]:
    """Count switches per eligible edge by distance to the nearest seam."""
    seam_sites = np.flatnonzero(covered[1:] != covered[:-1]) + 1
    sites = np.arange(len(covered))
    insertion = np.searchsorted(seam_sites, sites, side="right")
    distance = np.full(len(covered), np.inf)
    has_left = insertion > 0
    has_right = insertion < len(seam_sites)
    distance[has_left] = sites[has_left] - seam_sites[insertion[has_left] - 1]
    right_distance = seam_sites[insertion[has_right]] - 1 - sites[has_right]
    distance[has_right] = np.minimum(distance[has_right], right_distance)
    edge_distance = np.minimum(distance[:-1], distance[1:])
    edges = {"focal": covered[:-1] & covered[1:], "gap": ~covered[:-1] & ~covered[1:]}
    rows = []
    for region, region_edges in edges.items():
        for left, right in zip(
            config["distance_bin_edges"][:-1],
            config["distance_bin_edges"][1:],
            strict=True,
        ):
            eligible = region_edges & (edge_distance >= left) & (edge_distance < right)
            for method, parents in paths.items():
                changes = parents[1:] != parents[:-1]
                rows.append(
                    {
                        **identity,
                        "implementation": method,
                        "region": region,
                        "distance_left": left,
                        "distance_right": right,
                        "num_edges": int(np.count_nonzero(eligible)),
                        "num_switches": int(np.count_nonzero(changes & eligible)),
                    }
                )
    return rows


def comparison_analyse_haplotype(job: tuple[str, int]) -> ComparisonHaplotypeResult:
    name, index = job
    if name not in _comparison_contexts:
        _comparison_contexts[name] = comparison_load_context(name)
    context = _comparison_contexts[name]
    focal = context.focal
    key = (str(focal["sample_id"][index]), int(focal["ploidy_index"][index]))
    identity = {
        "dataset": name,
        "sample_id": key[0],
        "ploidy_index": key[1],
        "population": context.populations[key],
    }
    haplotype = np.ascontiguousarray(context.observations[:, index])
    full_parents = comparison_site_parents(context.full[key], context.reference)
    full_mismatches = comparison_mutation_mask(context.full[key], context.positions)
    greedy_paths = {}
    start, end = context.intervals["offsets"][index : index + 2]
    for ac_cutoff in _comparison_settings["pipeline"]["ac_cutoff"]:
        eligible = context.intervals["focal_ac"][start:end] <= ac_cutoff
        for budget in _comparison_settings["pipeline"]["max_mismatches"]:
            parents = focal_parent_path(
                len(haplotype),
                context.intervals["left_site_index"][start:end, budget][eligible],
                context.intervals["right_site_index"][start:end, budget][eligible],
                context.association_nodes[start:end][eligible],
            )
            greedy_paths[ac_cutoff, budget] = parents
    used_nodes = np.unique(
        np.concatenate([parents[parents >= 0] for parents in greedy_paths.values()])
    )
    genotypes = context.reference.genotype_matrix(
        samples=used_nodes, isolated_as_missing=False
    )
    comparisons = []
    matches = {}
    control_rows = {}
    components = []
    distances = []
    for (ac_cutoff, budget), greedy in greedy_paths.items():
        run_identity = {**identity, "ac_cutoff": ac_cutoff, "max_mismatches": budget}
        partition = comparison_mask_runs(context.masks[ac_cutoff, budget], index)
        runs = partition.focal
        covered = comparison_coverage_array(runs, len(haplotype))
        assert np.array_equal(covered, greedy >= 0)
        saved = context.matches[ac_cutoff, budget][key]
        reduced_parents = comparison_site_parents(saved, context.reference)
        reduced_mismatches = comparison_mutation_mask(saved, context.positions)
        greedy_parents = reduced_parents.copy()
        greedy_parents[covered] = greedy[covered]
        greedy_mismatches = reduced_mismatches.copy()
        sites = np.flatnonzero(covered)
        columns = np.searchsorted(used_nodes, greedy[covered])
        copied = genotypes[sites, columns]
        greedy_mismatches[covered] = (haplotype[covered] >= 0) & (
            haplotype[covered] != copied
        )
        paths = {
            "full": full_parents,
            "greedy": greedy_parents,
            "reduced_hmm": reduced_parents,
        }
        mismatch_masks = {
            "full": full_mismatches,
            "greedy": greedy_mismatches,
            "reduced_hmm": reduced_mismatches,
        }
        for method, parents in paths.items():
            row = comparison_comparison_row(
                run_identity, method, parents, mismatch_masks[method], covered, context
            )
            comparisons.append(row)
        selected = _comparison_settings["comparison"]
        if ac_cutoff != selected["ac_cutoff"] or budget != selected["max_mismatches"]:
            continue
        old = context.old_greedy[key]
        old_coverage = comparison_coverage_array(
            np.asarray(old["focal_site_intervals"]), len(haplotype)
        )
        assert np.array_equal(covered, old_coverage)
        old_parents = comparison_site_parents(old, context.reference)
        assert np.array_equal(old_parents[covered], greedy[covered])
        old_mismatches = comparison_mutation_mask(old, context.positions)
        row = comparison_comparison_row(
            run_identity, "saved_greedy", old_parents, old_mismatches, covered, context
        )
        comparisons.append(row)
        paths["saved_greedy"] = old_parents
        candidate_start, candidate_end = focal["offsets"][index : index + 2]
        columns = focal["ancestor_index"][candidate_start:candidate_end]
        eligible_columns = columns[focal["derived_ac"][columns] <= ac_cutoff]
        retained = [
            0,
            1,
            *[
                context.ancestor_nodes[focal["ancestor_id"][column]]
                for column in eligible_columns
            ],
        ]
        reduced, old_to_new = context.reference.simplify(
            retained, map_nodes=True, filter_sites=False
        )
        new_to_old = np.empty(reduced.num_nodes, dtype=np.int32)
        mapped = np.flatnonzero(old_to_new != tskit.NULL)
        new_to_old[old_to_new[mapped]] = mapped
        indexes = tsinfer.matching.MatcherIndexes(
            reduced, vestigial_root=False, num_alleles=context.num_alleles
        )
        matcher = tsinfer.matching.AncestorMatcher(indexes, context.rho, context.mu)
        whole = comparison_match_region(
            matcher, haplotype, 0, len(haplotype), context.positions
        )
        whole.parents = new_to_old[whole.parents]
        assert np.array_equal(whole.parents[covered], reduced_parents[covered])
        whole_mismatches = (haplotype >= 0) & (haplotype != whole.copied)
        assert np.array_equal(whole_mismatches[covered], reduced_mismatches[covered])
        independent = reduced_parents.copy()
        independent_mismatches = reduced_mismatches.copy()
        for component_index, (left, right) in enumerate(runs):
            local = comparison_match_region(
                matcher, haplotype, int(left), int(right), context.positions
            )
            independent[left:right] = new_to_old[local.parents]
            independent_mismatches[left:right] = (haplotype[left:right] >= 0) & (
                haplotype[left:right] != local.copied
            )
            row = {
                **run_identity,
                "component": component_index,
                "left_site": int(left),
                "right_site": int(right),
                "num_sites": int(right - left),
            }
            for method, parents in {**paths, "independent_hmm": independent}.items():
                row[f"{method}_switches"] = int(
                    np.count_nonzero(
                        parents[left + 1 : right] != parents[left : right - 1]
                    )
                )
            components.append(row)
        paths["independent_hmm"] = independent
        row = comparison_comparison_row(
            run_identity,
            "independent_hmm",
            independent,
            independent_mismatches,
            covered,
            context,
        )
        comparisons.append(row)
        independent_metrics = comparison_switch_metrics(
            independent, independent_mismatches, covered
        )
        greedy_nodes = np.unique(greedy[covered])
        original_columns = np.searchsorted(used_nodes, greedy_nodes)
        original_genotypes = genotypes[:, original_columns]
        reduced_genotypes = reduced.genotype_matrix(
            samples=old_to_new[greedy_nodes], isolated_as_missing=False
        )
        preserved = int(np.count_nonzero(original_genotypes != reduced_genotypes))
        assert preserved == 0
        control_rows = {
            **run_identity,
            "num_focal_components": len(runs),
            "num_candidates": len(eligible_columns),
            "num_reduced_nodes": reduced.num_nodes,
            "genotype_differences_after_simplify": preserved,
            "global_reduced_switches": int(
                np.count_nonzero(whole.parents[1:] != whole.parents[:-1])
            ),
            "global_reduced_focal_switches": comparison_switch_metrics(
                whole.parents, whole_mismatches, covered
            )["focal_switches"],
            "global_reduced_gap_mismatches": int(
                np.count_nonzero(whole_mismatches & ~covered)
            ),
            "independent_focal_switches": independent_metrics["focal_switches"],
            "greedy_focal_switches": comparison_switch_metrics(
                greedy_parents, greedy_mismatches, covered
            )["focal_switches"],
            "independent_focal_mismatches": independent_metrics["focal_mismatches"],
            "greedy_focal_mismatches": int(
                np.count_nonzero(greedy_mismatches & covered)
            ),
        }
        distances = comparison_distance_rows(
            run_identity, paths, covered, _comparison_settings
        )
        for method, parents in paths.items():
            if method == "full":
                matches[method] = context.full[key]
            elif method == "saved_greedy":
                matches[method] = old
            else:
                mismatch = (
                    independent_mismatches
                    if method == "independent_hmm"
                    else mismatch_masks[method]
                )
                matches[method] = comparison_encode_record(
                    key, parents, mismatch, haplotype, context
                )
        if key in {
            (item["sample_id"], item["ploidy_index"])
            for item in context.settings["haplotypes"]
        }:
            archive_path = (
                _comparison_settings["output_dir"]
                / f"{name}_{key[0].lower()}_{key[1]}_paths.npz"
            )
            np.savez_compressed(
                archive_path,
                covered=covered,
                observed=haplotype,
                reduced_whole=whole.parents,
                independent=independent,
                greedy=greedy_parents,
                saved_greedy=old_parents,
                full=full_parents,
                reduced_stitched=reduced_parents,
            )
    return ComparisonHaplotypeResult(
        comparisons, control_rows, components, distances, matches
    )


def comparison_topology_comparison(settings: dict, all_matches: dict) -> pd.DataFrame:
    """Measure marginal topology against simulated truth, over inference intervals.

    Kendall-Colijn lambda=0 measures rooted topology, not branch lengths. Sample
    ordering is explicitly joined to truth IDs, and unobserved flanks are removed.
    This single simulation is a diagnostic, not a biological accuracy claim.
    """
    rows = []
    for name, dataset in settings["datasets"].items():
        if dataset["population_source"] != "simulation":
            continue
        base = settings["input_dir"]
        reference = tskit.load(base / "ancestors" / f"{name}_inferred_ancestors.trees")
        native = tskit.load(base / "matches" / f"{name}_inferred_samples_raw.trees")
        sample_nodes = {}
        for node_id in native.samples():
            node = native.node(node_id)
            key = (node.metadata["sample_id"], node.metadata["ploidy_index"])
            sample_nodes[key] = int(node_id)
        pipeline_dataset = next(
            (item for item in settings["pipeline"]["datasets"] if item["name"] == name)
        )
        truth_path = pathlib.Path(pipeline_dataset["ts_path"]).expanduser()
        if not truth_path.is_absolute():
            truth_path = settings["pipeline_dir"] / truth_path
        truth = tskit.load(truth_path)
        truth_keys = []
        truth_nodes = []
        for individual in truth.individuals():
            for ploidy_index, node_id in enumerate(individual.nodes):
                truth_keys.append((f"tsk_{individual.id}", ploidy_index))
                truth_nodes.append(int(node_id))
        assert set(truth_keys) == set(sample_nodes)
        inferred_nodes = [sample_nodes[key] for key in truth_keys]
        intervals = reference.metadata["sequence_intervals"]
        truth = truth.simplify(truth_nodes)
        truth = truth.keep_intervals(intervals).trim()
        panel = zarr.open_group(
            base / "ancestors" / f"{name}_inferred_ancestors.zarr", mode="r"
        )
        alleles = panel["variant_allele"][:]
        for method in settings["topology_methods"]:
            tables = native.dump_tables()
            keep_edges = tables.edges.child < reference.num_nodes
            tables.edges.keep_rows(keep_edges)
            keep_mutations = tables.mutations.node < reference.num_nodes
            tables.mutations.keep_rows(keep_mutations)
            records = all_matches[name][method]
            for record in records:
                key = (record["sample_id"], record["ploidy_index"])
                child = sample_nodes[key]
                for segment in record["path"]:
                    tables.edges.add_row(
                        segment["left"], segment["right"], segment["parent"], child
                    )
                for mutation in record["mutations"]:
                    site = int(
                        np.searchsorted(reference.sites_position, mutation["position"])
                    )
                    state = str(alleles[site, mutation["derived_state"]])
                    tables.mutations.add_row(site=site, node=child, derived_state=state)
            tables.sort()
            inferred = tables.tree_sequence().simplify(inferred_nodes)
            inferred = inferred.keep_intervals(intervals).trim()
            distance = inferred.kc_distance(truth, lambda_=0)
            rows.append(
                {
                    "dataset": name,
                    "implementation": method,
                    "kc_topology_distance": distance,
                    "num_trees": inferred.num_trees,
                    "evaluated_bp": inferred.sequence_length,
                }
            )
    return pd.DataFrame(rows)
