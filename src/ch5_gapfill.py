"""Stage accounting and recurrent mutations in the hpc_perf3 archive.

Plotting is in :mod:`ch5_gapfill_plots`; shared dating and diagnostics are in
:mod:`ch5_perf_scaling`. All inputs are archived outputs, not genotype stores.
"""

import concurrent.futures as cf
import dataclasses
import inspect
import logging
import multiprocessing
import pathlib

import ch5_perf_scaling
import numcodecs
import numpy as np
import pandas as pd
import tskit
import tszip

logger = logging.getLogger(__name__)


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
    experiment = config['experiment']['performance']['experiment_id']
    root = config['data_dir'] / 'performance' / experiment
    stages = pd.read_csv(root / 'stages.csv')
    assert not stages.duplicated(['run_id', 'stage']).any()
    assert set(stages.run_id) <= set(runs.run_id)
    labels = ['run_id', 'dataset', 'num_samples', 'method', 'af_cutoff', 'matched_haplotypes']
    recorded = stages.merge(runs[labels], on='run_id', validate='many_to_one')
    wall_rows = []
    worker_rows = []
    wall_names = [name for name in config['plot']['wall_stages'] if name != 'other']
    worker_names = [name for name in config['plot']['worker_stages'] if name != 'gap_other']
    children = config['stages']['gap_children']
    parent = config['stages']['gap_parent']
    for run in runs.loc[runs.method.eq('greedy')].itertuples():
        measured = stages.loc[stages.run_id.eq(run.run_id)].set_index('stage')
        metadata = {name: getattr(run, name) for name in labels}
        wall = measured.loc[wall_names, 'wall_s']
        assert wall.notna().all()
        residual = run.wall_s - wall.sum()
        assert residual >= 0, run.run_id
        values = {**wall.to_dict(), 'other': residual}
        for stage, seconds in values.items():
            wall_rows.append({**metadata, 'stage': stage, 'wall_s': seconds,
                              'wall_percentage': 100 * seconds / run.wall_s})
        for metric in ['service_s', 'thread_cpu_s']:
            parts = measured.loc[worker_names, metric]
            assert parts.notna().all()
            gap_other = measured.loc[parent, metric] - measured.loc[children, metric].sum()
            assert gap_other >= 0, (run.run_id, metric)
            values = {**parts.to_dict(), 'gap_other': gap_other}
            for stage, seconds in values.items():
                worker_rows.append({**metadata, 'stage': stage, 'metric': metric,
                                    'seconds': seconds,
                                    'ms_per_haplotype': 1000 * seconds / run.matched_haplotypes})
    wall_table = pd.DataFrame(wall_rows)
    worker_table = pd.DataFrame(worker_rows)
    recorded.to_csv(config['cache_dir'] / 'recorded_stages.csv', index=False)
    wall_table.to_csv(config['cache_dir'] / 'wall_stage_breakdown.csv', index=False)
    worker_table.to_csv(config['cache_dir'] / 'worker_stage_breakdown.csv', index=False)
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
    return {'num_sites': len(counts), 'recurrent_sites': num_recurrent,
            'recurrent_site_percentage': 100 * num_recurrent / len(counts),
            'mutations_at_recurrent_sites': int(recurrent.sum()),
            'mean_mutations_per_recurrent_site': mean_mutations,
            'additional_mutations': int((recurrent - 1).sum())}


@dataclasses.dataclass
class RecurrenceJob:
    root: pathlib.Path
    cache: pathlib.Path
    dataset: dict


def recurrent_dataset(job: RecurrenceJob) -> pd.DataFrame:
    """Compare raw Full HMM, stitched and truth on saved inference positions."""
    dataset = job.dataset
    cases = [{'method': 'full_hmm', 'af_cutoff': None, 'path': dataset['full_hmm']}]
    for run in dataset['runs']:
        cases.append({'method': 'greedy', 'af_cutoff': run['af_cutoff'], 'path': run['stitched']})
    if dataset['truth'] is not None:
        cases.append({'method': 'truth', 'af_cutoff': None, 'path': dataset['truth']})
    paths = [job.root / case['path'] for case in cases]
    settings = {'cases': cases, 'operation': inspect.getsource(recurrent_site_counts)}
    stamp = ch5_perf_scaling.signature(paths, settings, recurrent_dataset)
    output = job.cache / 'recurrent_sites' / f"{dataset['name']}.csv"
    cached = ch5_perf_scaling.read_cached(output, stamp)
    if cached is not None:
        return cached
    full = tszip.decompress(paths[0])
    positions = full.sites_position
    rows = []
    for case, path in zip(cases, paths, strict=True):
        ts = full if case['method'] == 'full_hmm' else tszip.decompress(path)
        if case['method'] != 'truth':
            np.testing.assert_array_equal(ts.sites_position, positions)
        counts = recurrent_site_counts(ts, positions)
        rows.append({'dataset': dataset['name'], 'method': case['method'],
                     'af_cutoff': case['af_cutoff'], **counts})
    table = pd.DataFrame(rows)
    ch5_perf_scaling.save_cached(table, output, stamp)
    return table


def recurrent_sites(config: dict) -> pd.DataFrame:
    """Distribute independent cohort counts across worker processes and cache."""
    jobs = [RecurrenceJob(config['data_dir'], config['cache_dir'], dataset)
            for dataset in config['manifest']['datasets']]
    workers = config['workers']
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    context = multiprocessing.get_context('spawn')
    rows = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                initializer=numcodecs.blosc.set_nthreads,
                                initargs=(config['helper_threads'],)) as pool:
        futures = {pool.submit(recurrent_dataset, job): job for job in jobs}
        for future in cf.as_completed(futures):
            rows.append(future.result())
            logger.info('Recurrent sites: %s', futures[future].dataset['name'])
    table = pd.concat(rows, ignore_index=True)
    sizes = ch5_perf_scaling.cohort_table(config)[['dataset', 'num_samples']]
    table = table.merge(sizes, on='dataset', validate='many_to_one')
    table = table.sort_values(['num_samples', 'method', 'af_cutoff'])
    table.to_csv(config['cache_dir'] / 'recurrent_site_summary.csv', index=False)
    return table
