"""Analyse the self-contained AF/minimum-region diagnostics archive for pt7.

Path summaries come from the pipeline, integer length counts stream from CSV,
and independent tree comparisons/dating jobs run in worker processes. Mutation
comparisons reuse pt6 extraction rules, with one truth/full/stitched intersection.
"""

import concurrent.futures as cf
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import pathlib

import ch5_dating
import ch5_arg_metrics
import ch5_hyperparams
import numcodecs
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
import tsdate
import tsinfer
import tskit
import tszip
import yaml

logger = logging.getLogger(__name__)


def load_config(path: pathlib.Path) -> dict:
    """Read the analysis config and archived effective experiment/manifest."""
    config = yaml.safe_load(path.read_text())
    root = ch5_hyperparams.resolve_path(config['data_dir'], path.parent)
    config['data_dir'] = root
    config['cache_dir'] = root / config['cache_subdir']
    config['cache_dir'].mkdir(parents=True, exist_ok=True)
    config['manifest'] = json.loads((root / 'manifest.json').read_text())
    config['experiment'] = yaml.safe_load((root / 'config.yaml').read_text())
    return config


def input_signature(paths: list[pathlib.Path], settings: dict) -> str:
    """Invalidate plot caches on changed inputs, analysis code or dependencies."""
    stamps = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    versions = {name: importlib.metadata.version(name)
                for name in ['tsinfer', 'tsdate', 'tskit', 'tscompare', 'tszip']}
    digest = hashlib.sha256()
    for module in [ch5_arg_metrics, ch5_dating, ch5_hyperparams]:
        digest.update(pathlib.Path(module.__file__).read_bytes())
    digest.update(pathlib.Path(__file__).read_bytes())
    code = digest.hexdigest()
    payload = json.dumps([stamps, settings, versions, code], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def cached_table(path: pathlib.Path, signature: str) -> pd.DataFrame | None:
    """Read a completed table only when its signature matches."""
    stamp = path.with_suffix('.json')
    if path.exists() and stamp.exists() and stamp.read_text() == signature:
        return pd.read_csv(path)
    return None


def save_table(table: pd.DataFrame, path: pathlib.Path, signature: str):
    """Write the signature last, so interrupted work is never treated as complete."""
    table.to_csv(path, index=False)
    path.with_suffix('.json').write_text(signature)


def length_counts(config: dict) -> pd.DataFrame:
    """Stream 4 GB of region rows into lossless counts at each integer length."""
    path = config['data_dir'] / config['manifest']['tables']['region_lengths']
    output = config['cache_dir'] / 'length_counts.csv'
    signature = input_signature([path], {})
    cached = cached_table(output, signature)
    if cached is not None:
        return cached
    keys = ['dataset', 'af_cutoff', 'min_focal_region_size', 'is_gap', 'num_sites']
    query = pl.scan_csv(path).select(keys)
    counts = query.group_by(keys).len(name='count').sort(keys)
    collected = counts.collect(engine='streaming')
    columns = {name: collected[name].to_numpy() for name in collected.columns}
    table = pd.DataFrame(columns)
    save_table(table, output, signature)
    return table


def population_summary(config: dict) -> pd.DataFrame:
    """Cache pt6 African/non-African coverage means from exported haplotypes."""
    path = config['data_dir'] / config['manifest']['tables']['haplotype_results']
    output = config['cache_dir'] / 'population_coverage.csv'
    signature = input_signature([path], {})
    cached = cached_table(output, signature)
    if cached is not None:
        return cached
    columns = ['dataset', 'af_cutoff', 'min_focal_region_size', 'population',
               'focal_coverage_percentage']
    table = pd.read_csv(path, usecols=columns)
    table['population_group'] = np.where(table.population.eq('YRI'), 'African', 'Non-African')
    keys = ['dataset', 'af_cutoff', 'min_focal_region_size', 'population_group']
    result = table.groupby(keys).focal_coverage_percentage.agg(['mean', 'sem', 'count']).reset_index()
    save_table(result, output, signature)
    return result


def weighted_quantile(table: pd.DataFrame, quantile: float) -> float:
    """Match NumPy's linearly interpolated quantile without expanding counts."""
    counts = table.groupby('num_sites')['count'].sum().sort_index()
    cumulative = counts.cumsum().to_numpy()
    rank = (int(cumulative[-1]) - 1) * quantile
    lower = np.searchsorted(cumulative, np.floor(rank), side='right')
    upper = np.searchsorted(cumulative, np.ceil(rank), side='right')
    values = counts.index.to_numpy()
    return float(values[lower] + (rank - np.floor(rank)) * (values[upper] - values[lower]))


def length_density(table: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Evaluate pt6's reflected KDE exactly from length multiplicities.

    Correct the weighted covariance/bandwidth to reproduce Scott's bandwidth
    for the original uncompressed interval observations, rather than treating
    unique lengths as independent observations.
    """
    values = table.num_sites.to_numpy(dtype=float)
    counts = table['count'].to_numpy(dtype=float)
    total = counts.sum()
    mean = np.average(values, weights=counts)
    variance = np.sum(counts * (values - mean) ** 2) / (total - 1)
    bandwidth = np.sqrt(variance) * total ** (-1 / 5)
    weighted_variance = np.cov(values, aweights=counts).item()
    kde = stats.gaussian_kde(values, weights=counts,
                             bw_method=bandwidth / np.sqrt(weighted_variance))
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
        relative = self.dataset['full_hmm'] if self.run is None else self.run['stitched']
        return self.root / relative

    @property
    def key(self) -> str:
        suffix = 'full_hmm'
        if self.run is not None:
            suffix = f"af_{self.run['af_cutoff']}_min_{self.run['min_focal_region_size']}"
        return f"{self.dataset['name']}_{suffix}"

    @property
    def labels(self) -> dict:
        return {
            'dataset': self.dataset['name'],
            'comparison_method': 'full_hmm' if self.run is None else 'stitched_greedy',
            'af_cutoff': None if self.run is None else self.run['af_cutoff'],
            'min_focal_region_size': None if self.run is None else self.run['min_focal_region_size'],
        }


def tree_jobs(config: dict) -> list[TreeJob]:
    """Derive all paths from the manifest, avoiding stale HPC absolute paths."""
    jobs = []
    settings = {key: config[key] for key in ['mutation_rate', 'dating', 'helper_threads']}
    for dataset in config['manifest']['datasets']:
        for run in [None, *dataset['runs']]:
            jobs.append(TreeJob(config['data_dir'], config['cache_dir'], dataset, run, settings))
    return jobs


def compare_tree(job: TreeJob) -> dict:
    """Cache raw counts and aligned ARF/TPR against truth and Full HMM."""
    truth_path = job.root / job.dataset['truth']
    full_path = job.root / job.dataset['full_hmm']
    signature = input_signature([job.raw_path, truth_path, full_path], {})
    output = job.cache / 'tree_metrics' / f'{job.key}.csv'
    cached = cached_table(output, signature)
    if cached is not None:
        return cached.iloc[0].to_dict()
    comparison = tszip.decompress(job.raw_path)
    row = {**job.labels, 'num_edges': comparison.num_edges, 'num_trees': comparison.num_trees}
    span = job.dataset['analysis_extent_bp']
    truth = tszip.decompress(truth_path)
    mapping = ch5_hyperparams.truth_sample_nodes(truth)
    aligned = ch5_hyperparams.align_tree_sequences(comparison, truth, span, mapping)
    metric = ch5_arg_metrics.haplotype_arf(aligned.comparison, aligned.reference)
    row.update(truth_arf=float(metric.arf), truth_tpr=float(metric.tpr))
    del truth, aligned
    if job.run is not None:
        full = tszip.decompress(full_path)
        row['edges_change_percentage'] = 100 * (comparison.num_edges / full.num_edges - 1)
        row['trees_change_percentage'] = 100 * (comparison.num_trees / full.num_trees - 1)
        aligned = ch5_hyperparams.align_tree_sequences(comparison, full, span)
        metric = ch5_arg_metrics.haplotype_arf(aligned.comparison, aligned.reference)
        row.update(full_arf=float(metric.arf), full_tpr=float(metric.tpr))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_table(pd.DataFrame([row]), output, signature)
    return row


def date_tree(job: TreeJob) -> dict:
    """Date the saved sites only; no genotype stores or variant restoration."""
    signature = input_signature([job.raw_path], job.config)
    output = job.cache / 'dated_args' / f'{job.key}.csv'
    tree_path = output.with_suffix('.trees')
    cached = cached_table(output, signature)
    if cached is not None and tree_path.exists():
        return cached.iloc[0].to_dict()
    raw = tszip.decompress(job.raw_path)
    settings = job.config['dating']
    processed = tsinfer.post_process(raw, **settings['post_process'])
    window = ch5_hyperparams.window_tree_sequence(processed, job.dataset['analysis_extent_bp'])
    prepared = tsdate.preprocess_ts(window, **settings['preprocess'])
    dated = tsdate.date(prepared, mutation_rate=job.config['mutation_rate'], **settings['estimation'])
    output.parent.mkdir(parents=True, exist_ok=True)
    dated.dump(tree_path)
    row = {**job.labels, 'dated_path': str(tree_path), 'raw_sites': raw.num_sites,
           'dated_sites': dated.num_sites, 'dated_mutations': dated.num_mutations}
    save_table(pd.DataFrame([row]), output, signature)
    return row


def run_tree_jobs(config: dict, operation: str) -> pd.DataFrame:
    """Distribute independent comparisons or dates across the available CPUs."""
    jobs = tree_jobs(config)
    workers = config['workers']
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    function = {'metrics': compare_tree, 'dating': date_tree}[operation]
    context = multiprocessing.get_context('spawn')
    rows = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                initializer=numcodecs.blosc.set_nthreads,
                                initargs=(config['helper_threads'],)) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        for future in cf.as_completed(futures):
            rows.append(future.result())
            logger.info('%s: completed %s (%d/%d)', operation, futures[future].key, len(rows), len(jobs))
    result = pd.DataFrame(rows).sort_values(['dataset', 'af_cutoff', 'min_focal_region_size'])
    result.to_csv(config['cache_dir'] / f'{operation}_summary.csv', index=False)
    return result


def shared_mutation_correlations(runs: pd.DataFrame, truth: pd.DataFrame,
                                 settings: dict) -> pd.DataFrame:
    """Use one intersection across truth and every dated run for both references.

    Reuse pt6's posterior mutation means and exact allele/position pairing.
    Recurrent sites and nonfinite ages are excluded before intersection. DAF
    comes from Full HMM and is fixed across every comparison in the sample size.
    """
    ages = []
    for run in runs.itertuples():
        ts = tskit.load(run.dated_path)
        values = ch5_dating.mutation_ages(ts, posterior=True,
                                         include_frequency=run.comparison_method == 'full_hmm')
        ages.append(values)
    common = truth.index
    for values in ages:
        common = common.intersection(values.index, sort=False)
    common = common.sort_values()
    full_index = runs.index[runs.comparison_method.eq('full_hmm')].item()
    full = ages[full_index]
    frequencies = full.derived_frequency.loc[common]
    # The existing pt6 row helper takes an AC field; adapt only at its boundary.
    legacy_runs = runs.rename(columns={'af_cutoff': 'ac_cutoff'}).copy()
    legacy_runs['max_mismatches'] = None
    rows = []
    for reference_method, reference in [('true_arg', truth.age), ('full_hmm', full.age)]:
        reference = reference.loc[common]
        rows.extend(ch5_dating.correlation_rows(
            legacy_runs, ages, reference, frequencies, reference_method, settings))
    return pd.DataFrame(rows).rename(columns={'ac_cutoff': 'af_cutoff'}).drop(columns='max_mismatches')


def mutation_correlations(config: dict, dated: pd.DataFrame) -> pd.DataFrame:
    """Cache correlations using the same common mutation set for both references."""
    paths = [pathlib.Path(path) for path in dated.dated_path]
    paths.extend(config['data_dir'] / item['truth'] for item in config['manifest']['datasets'])
    signature = input_signature(paths, config['frequency_bins'])
    output = config['cache_dir'] / 'mutation_age_correlations.csv'
    cached = cached_table(output, signature)
    if cached is not None:
        return cached
    results = []
    for dataset in config['manifest']['datasets']:
        runs = dated.loc[dated.dataset.eq(dataset['name'])].reset_index(drop=True)
        truth = tszip.decompress(config['data_dir'] / dataset['truth'])
        truth = ch5_hyperparams.window_tree_sequence(truth, dataset['analysis_extent_bp'])
        ages = ch5_dating.mutation_ages(truth, posterior=False)
        result = shared_mutation_correlations(runs, ages, config['frequency_bins'])
        results.append(result)
    table = pd.concat(results, ignore_index=True)
    save_table(table, output, signature)
    return table
