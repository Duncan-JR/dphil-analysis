"""Analyse archived HPC scaling trials and date their saved inference sites.

Performance and path diagnostics stay distinct and are linked by checked tree
fingerprints. No genotype stores, variant restoration, or ARG metrics are used.
"""

import concurrent.futures as cf
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import logging
import multiprocessing
import pathlib

import ch5_dating
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
    """Use the archived effective config and relative manifest paths."""
    config = yaml.safe_load(path.read_text())
    root = ch5_hyperparams.resolve_path(config['data_dir'], path.parent)
    config['data_dir'] = root
    config['cache_dir'] = root / config['cache_subdir']
    config['cache_dir'].mkdir(parents=True, exist_ok=True)
    config['manifest'] = json.loads((root / 'manifest.json').read_text())
    config['experiment'] = yaml.safe_load((root / 'config.yaml').read_text())
    return config


def signature(paths: list[pathlib.Path], settings: dict, operation) -> str:
    """Invalidate a cache when its inputs, operation, settings or versions change."""
    stamps = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    versions = {name: importlib.metadata.version(name)
                for name in ['tsinfer', 'tsdate', 'tskit', 'tszip']}
    code = inspect.getsource(operation)
    payload = json.dumps([stamps, settings, versions, code], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_cached(path: pathlib.Path, stamp: str) -> pd.DataFrame | None:
    metadata = path.with_suffix('.json')
    if path.exists() and metadata.exists() and metadata.read_text() == stamp:
        return pd.read_csv(path)
    return None


def save_cached(table: pd.DataFrame, path: pathlib.Path, stamp: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    path.with_suffix('.json').write_text(stamp)


def cohort_table(config: dict) -> pd.DataFrame:
    """Read sample counts from actual archived haplotype identities, not names."""
    rows = []
    for dataset in config['manifest']['datasets']:
        cohort = pd.DataFrame(dataset['cohort'])
        assert not cohort.duplicated(['source', 'sample_id', 'ploidy_index']).any()
        individuals = cohort[['source', 'sample_id']].drop_duplicates()
        rows.append({'dataset': dataset['name'], 'num_samples': len(individuals),
                     'num_haplotypes': len(cohort), 'num_sites': dataset['num_sites'],
                     'analysis_span_mb': dataset['analysis_span_bp'] / 1e6,
                     'has_truth': dataset['truth'] is not None,
                     'min_region_size_kb': dataset['runs'][0]['min_region_size_kb'],
                     'min_focal_region_size': dataset['runs'][0]['min_focal_region_size']})
    return pd.DataFrame(rows).sort_values('num_samples').reset_index(drop=True)


def performance_runs(config: dict) -> pd.DataFrame:
    """Preserve one Full HMM baseline and both greedy trials per sample size."""
    name = config['experiment']['performance']['experiment_id']
    root = config['data_dir'] / 'performance' / name
    runs = pd.read_csv(root / 'runs.csv')
    quality = pd.DataFrame(json.loads((root / 'quality.json').read_text()))
    assert runs.status.eq('ok').all()
    assert quality.status.eq('ok').all()
    assert quality.called_allele_errors.eq(0).all()
    assert set(quality.run_id) == set(runs.run_id)
    assert runs['mode'].eq('from_reference').all()
    assert runs.actual_workers.eq(config['experiment']['performance']['worker_counts'][0]).all()
    assert not runs.duplicated(['dataset', 'method', 'af_cutoff', 'repeat']).any()
    cohorts = cohort_table(config)
    runs = runs.merge(cohorts[['dataset', 'num_samples', 'num_sites']], on='dataset', validate='many_to_one')
    assert runs.matched_haplotypes.eq(2 * runs.num_samples).all()
    runs['cpu_s'] = runs.user_s + runs.system_s
    runs['effective_cores'] = runs.cpu_s / runs.wall_s
    runs['wall_ms_per_haplotype'] = 1000 * runs.wall_s / runs.matched_haplotypes
    runs['trial_peak_private_gib'] = runs.trial_peak_private_bytes / 1024 ** 3
    runs['window_peak_private_gib'] = runs.window_peak_private_bytes / 1024 ** 3
    baseline = runs.loc[runs.method.eq('full_hmm'), ['dataset', 'repeat', 'wall_s', 'cpu_s']]
    baseline = baseline.rename(columns={'wall_s': 'full_wall_s', 'cpu_s': 'full_cpu_s'})
    runs = runs.merge(baseline, on=['dataset', 'repeat'], validate='many_to_one')
    runs['speedup'] = runs.full_wall_s / runs.wall_s
    runs['cpu_speedup'] = runs.full_cpu_s / runs.cpu_s
    runs['focal_site_coverage_percentage'] = 100 * runs.focal_sites / runs.domain_sites
    runs['gap_calls_per_haplotype'] = runs.gap_calls / runs.matched_haplotypes
    runs.to_csv(config['cache_dir'] / 'performance_summary.csv', index=False)
    return runs.sort_values(['num_samples', 'method', 'af_cutoff'])


def diagnostic_summary(config: dict) -> pd.DataFrame:
    path = config['data_dir'] / config['manifest']['tables']['run_summary']
    table = pd.read_csv(path)
    cohorts = cohort_table(config)
    assert table.unmatched_sites_mean.eq(0).all()
    assert table.min_region_size_kb.eq(config['experiment']['min_region_size_kb']).all()
    assert not table.duplicated(['dataset', 'af_cutoff']).any()
    table = table.merge(cohorts[['dataset', 'num_samples']].rename(columns={'num_samples': 'actual_samples'}),
                        on='dataset', validate='many_to_one')
    assert table.num_samples.eq(table.actual_samples).all()
    table.to_csv(config['cache_dir'] / 'diagnostic_summary.csv', index=False)
    return table.sort_values(['num_samples', 'af_cutoff'])


def population_coverage(config: dict) -> pd.DataFrame:
    """Aggregate physical coverage with the configured population groups."""
    path = config['data_dir'] / config['manifest']['tables']['haplotype_results']
    settings = config['population_groups']
    stamp = signature([path], settings, population_coverage)
    output = config['cache_dir'] / 'population_coverage.csv'
    cached = read_cached(output, stamp)
    if cached is not None:
        return cached
    columns = ['dataset', 'af_cutoff', 'population', 'focal_coverage_percentage']
    table = pd.read_csv(path, usecols=columns)
    mapping = {population: group for group, members in settings.items() for population in members}
    assert set(table.population) <= set(mapping)
    table['population_group'] = table.population.map(mapping)
    result = table.groupby(['dataset', 'af_cutoff', 'population_group']).focal_coverage_percentage.agg(
        ['mean', 'sem', 'count']).reset_index()
    sizes = cohort_table(config)[['dataset', 'num_samples']]
    result = result.merge(sizes, on='dataset', validate='many_to_one')
    save_cached(result, output, stamp)
    return result


def length_counts(config: dict) -> pd.DataFrame:
    """Stream region rows into lossless site-count and physical-length counts."""
    source = config['data_dir'] / config['manifest']['tables']['region_lengths']
    stamp = signature([source], {}, length_counts)
    output = config['cache_dir'] / 'region_length_counts.csv'
    cached = read_cached(output, stamp)
    if cached is not None:
        return cached
    keys = ['dataset', 'af_cutoff', 'is_gap']
    query = pl.scan_csv(source)
    tables = []
    for column, unit, divisor in [('num_sites', 'sites', 1), ('span_bp', 'kb', 1000)]:
        counts = query.group_by([*keys, column]).len(name='count')
        collected = counts.collect(engine='streaming')
        table = pd.DataFrame({name: collected[name].to_numpy() for name in collected.columns})
        table = table.rename(columns={column: 'length'})
        table['length'] /= divisor
        table['unit'] = unit
        tables.append(table)
    result = pd.concat(tables, ignore_index=True).sort_values([*keys, 'unit', 'length'])
    save_cached(result, output, stamp)
    return result


def weighted_quantile(table: pd.DataFrame, quantile: float) -> float:
    counts = table.groupby('length')['count'].sum().sort_index()
    cumulative = counts.cumsum().to_numpy()
    rank = (int(cumulative[-1]) - 1) * quantile
    lower = np.searchsorted(cumulative, np.floor(rank), side='right')
    upper = np.searchsorted(cumulative, np.ceil(rank), side='right')
    values = counts.index.to_numpy()
    return float(values[lower] + (rank - np.floor(rank)) * (values[upper] - values[lower]))


def length_density(table: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Reflected Scott KDE, weighted to equal the uncompressed observations."""
    values = table.length.to_numpy(dtype=float)
    counts = table['count'].to_numpy(dtype=float)
    total = counts.sum()
    mean = np.average(values, weights=counts)
    variance = np.sum(counts * (values - mean) ** 2) / (total - 1)
    bandwidth = np.sqrt(variance) * total ** (-1 / 5)
    weighted_variance = np.cov(values, aweights=counts).item()
    kde = stats.gaussian_kde(values, weights=counts, bw_method=bandwidth / np.sqrt(weighted_variance))
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
        relative = self.dataset['full_hmm'] if self.run is None else self.run['stitched']
        return self.root / relative

    @property
    def key(self) -> str:
        suffix = 'full_hmm' if self.run is None else f"af_{self.run['af_cutoff']}_kb_{self.run['min_region_size_kb']}"
        return f"{self.dataset['name']}_{suffix}"

    @property
    def labels(self) -> dict:
        return {'dataset': self.dataset['name'], 'method': 'full_hmm' if self.run is None else 'greedy',
                'af_cutoff': None if self.run is None else self.run['af_cutoff'],
                'min_region_size_kb': None if self.run is None else self.run['min_region_size_kb'],
                'min_focal_region_size': None if self.run is None else self.run['min_focal_region_size']}


def path_fingerprint(ts: tskit.TreeSequence) -> str:
    """Compare saved copying paths and mutations, excluding file provenance."""
    digest = hashlib.sha256()
    for values in [ts.sites_position, ts.edges_left, ts.edges_right, ts.edges_parent, ts.edges_child,
                   ts.nodes_time, ts.nodes_flags, ts.mutations_site, ts.mutations_node,
                   ts.tables.mutations.derived_state, ts.tables.sites.ancestral_state]:
        digest.update(memoryview(values))
    return digest.hexdigest()


def date_tree(job: DatingJob) -> dict:
    """Check the timed/diagnostic outputs agree, then date only their saved sites."""
    stamp = signature([job.raw_path, job.benchmark_tree], job.settings, date_tree)
    output = job.cache / 'dated_args' / f'{job.key}.csv'
    tree_path = output.with_suffix('.trees')
    cached = read_cached(output, stamp)
    if cached is not None and tree_path.exists():
        return cached.iloc[0].to_dict()
    raw = tszip.decompress(job.raw_path)
    fingerprint = path_fingerprint(raw)
    timed = tszip.decompress(job.benchmark_tree)
    assert fingerprint == path_fingerprint(timed), job.key
    del timed
    counts = {'num_edges': raw.num_edges, 'num_trees': raw.num_trees, 'raw_sites': raw.num_sites,
              'num_haplotypes': raw.num_samples, 'benchmark_paths_identical': True}
    settings = job.settings['dating']
    processed = tsinfer.post_process(raw, **settings['post_process'])
    window = ch5_hyperparams.window_tree_sequence(processed, job.dataset['analysis_extent_bp'])
    prepared = tsdate.preprocess_ts(window, **settings['preprocess'])
    dated = tsdate.date(prepared, mutation_rate=job.settings['mutation_rate'], **settings['estimation'])
    output.parent.mkdir(parents=True, exist_ok=True)
    dated.dump(tree_path)
    row = {**job.labels, **counts, 'dated_path': str(tree_path),
           'dated_sites': dated.num_sites, 'dated_mutations': dated.num_mutations}
    save_cached(pd.DataFrame([row]), output, stamp)
    return row


def date_all(config: dict) -> pd.DataFrame:
    """Distribute independent saved tree sequences across CPU worker processes."""
    experiment = config['experiment']['performance']['experiment_id']
    performance_root = config['data_dir'] / 'performance' / experiment
    runs = pd.read_csv(performance_root / 'runs.csv')
    jobs = []
    settings = {key: config[key] for key in ['mutation_rate', 'dating']}
    for dataset in config['manifest']['datasets']:
        for run in [None, *dataset['runs']]:
            selected = runs.dataset.eq(dataset['name'])
            if run is None:
                selected &= runs.method.eq('full_hmm')
            else:
                selected &= runs.method.eq('greedy') & runs.af_cutoff.eq(run['af_cutoff'])
            trial = runs.loc[selected].iloc[0]
            benchmark_tree = performance_root / trial.run_id / 'samples_raw.trees.tsz'
            jobs.append(DatingJob(config['data_dir'], config['cache_dir'], dataset, run, benchmark_tree, settings))
    workers = config['workers']
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    rows = []
    context = multiprocessing.get_context('spawn')
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                initializer=numcodecs.blosc.set_nthreads,
                                initargs=(config['helper_threads'],)) as pool:
        futures = {pool.submit(date_tree, job): job for job in jobs}
        for future in cf.as_completed(futures):
            rows.append(future.result())
            logger.info('Dated %s (%d/%d)', futures[future].key, len(rows), len(jobs))
    table = pd.DataFrame(rows).sort_values(['dataset', 'method', 'af_cutoff']).reset_index(drop=True)
    table = table.merge(cohort_table(config)[['dataset', 'num_samples']], on='dataset', validate='many_to_one')
    baseline = table.loc[table.method.eq('full_hmm'), ['dataset', 'num_edges', 'num_trees']]
    baseline = baseline.rename(columns={'num_edges': 'full_edges', 'num_trees': 'full_trees'})
    table = table.merge(baseline, on='dataset', validate='many_to_one')
    table['edges_change_percentage'] = 100 * (table.num_edges / table.full_edges - 1)
    table['trees_change_percentage'] = 100 * (table.num_trees / table.full_trees - 1)
    table.to_csv(config['cache_dir'] / 'dating_summary.csv', index=False)
    return table


def age_comparison(config: dict, dataset: dict, dated: pd.DataFrame) -> pd.DataFrame:
    """One mutation intersection across all inferred runs and optional truth."""
    rows = dated.loc[dated.dataset.eq(dataset['name'])].reset_index(drop=True)
    paths = [pathlib.Path(path) for path in rows.dated_path]
    if dataset['truth'] is not None:
        paths.append(config['data_dir'] / dataset['truth'])
    stamp = signature(paths, config['frequency_bins'], age_comparison)
    output = config['cache_dir'] / 'age_pairs' / f"{dataset['name']}.parquet"
    metadata = output.with_suffix('.json')
    summary_path = output.with_suffix('.csv')
    if output.exists() and summary_path.exists() and metadata.exists() and metadata.read_text() == stamp:
        return pd.read_csv(summary_path)
    ages = {}
    common = None
    candidates = []
    for row in rows.itertuples():
        tree = tskit.load(row.dated_path)
        values = ch5_dating.mutation_ages(tree, posterior=True, include_frequency=row.method == 'full_hmm')
        key = row.method if row.method == 'full_hmm' else f'af_{row.af_cutoff}'
        ages[key] = values
        candidates.append({'dataset': dataset['name'], 'method': row.method, 'af_cutoff': row.af_cutoff,
                           'finite_single_mutation_sites': len(values)})
        common = values.index if common is None else common.intersection(values.index, sort=False)
    if dataset['truth'] is not None:
        truth = tszip.decompress(config['data_dir'] / dataset['truth'])
        truth = ch5_hyperparams.window_tree_sequence(truth, dataset['analysis_extent_bp'])
        values = ch5_dating.mutation_ages(truth, posterior=False)
        ages['truth'] = values
        common = common.intersection(values.index, sort=False)
    common = common.sort_values()
    pairs = pd.DataFrame({key: values.age.loc[common] for key, values in ages.items()})
    pairs['derived_frequency'] = ages['full_hmm'].derived_frequency.loc[common]
    pairs['frequency_bin'] = ch5_dating.frequency_bins(pairs.derived_frequency, config['frequency_bins'])
    comparisons = []
    references = ['full_hmm'] if dataset['truth'] is None else ['truth', 'full_hmm']
    for reference in references:
        methods = rows if reference == 'truth' else rows.loc[rows.method.ne('full_hmm')]
        for row in methods.itertuples():
            key = row.method if row.method == 'full_hmm' else f'af_{row.af_cutoff}'
            groups = [('all', pairs)]
            groups.extend((label, pairs.loc[pairs.frequency_bin.eq(label)]) for label in config['frequency_bins']['labels'])
            for label, group in groups:
                x = group[reference]
                y = group[key]
                error = y - x
                correlation = np.nan
                if len(group) > 1 and x.nunique() > 1 and y.nunique() > 1:
                    correlation = float(stats.pearsonr(x, y).statistic)
                comparisons.append({'dataset': dataset['name'], 'num_samples': row.num_samples,
                                    'method': row.method, 'af_cutoff': row.af_cutoff,
                                    'reference_method': reference, 'frequency_bin': label,
                                    'matched_mutations': len(group), 'pearson_r': correlation,
                                    'mean_error_generations': error.mean(),
                                    'median_absolute_error_generations': error.abs().median(),
                                    'mean_estimated_age': y.mean(), 'mean_reference_age': x.mean()})
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs_output = pairs.reset_index()
    pl.DataFrame({name: pairs_output[name].to_numpy() for name in pairs_output.columns}).write_parquet(output)
    result = pd.DataFrame(comparisons)
    result.to_csv(summary_path, index=False)
    counts = pd.DataFrame(candidates)
    counts['common_mutations'] = len(common)
    counts['retained_percentage'] = 100 * len(common) / counts.finite_single_mutation_sites
    counts.to_csv(output.with_name(f'{output.stem}_retention.csv'), index=False)
    metadata.write_text(stamp)
    return result


def mutation_correlations(config: dict, dated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in config['manifest']['datasets']:
        rows.append(age_comparison(config, dataset, dated))
        logger.info('Mutation-age comparison complete: %s', dataset['name'])
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(config['cache_dir'] / 'mutation_age_correlations.csv', index=False)
    return table
