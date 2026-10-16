"""Read saved benchmark timings and compare copying paths using pt3 metrics.

:func:`load_results` validates the completed suite and checks that paths and
mutations are unchanged across modes/repeats before choosing one correctness
replicate. :func:`compare_paths` uses the same site partition and switch
definitions as :mod:`ch5_stitching`, with original reference parent IDs.
"""

import concurrent.futures as cf
import dataclasses
import hashlib
import json
import logging
import multiprocessing
import pathlib

import ch5_hyperparams
import ch5_stitching
import numpy as np
import pandas as pd
import tskit
import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Results:
    """Saved suite metadata, timings, repeat fingerprints and haplotype metrics."""

    manifest: dict
    runs: pd.DataFrame
    stages: pd.DataFrame
    quality: pd.DataFrame
    fingerprints: pd.DataFrame
    comparisons: pd.DataFrame


def path_fingerprint(run: dict) -> dict:
    """Hash scientific JSONL fields in canonical haplotype and segment order."""
    records = ch5_hyperparams.read_matches(pathlib.Path(run['output_matches']))
    digest = hashlib.sha256()
    for identity, record in sorted(records.items()):
        content = [identity, record['path'], record['mutations']]
        encoded = json.dumps(content, sort_keys=True, separators=(',', ':'))
        digest.update(encoded.encode())
    return {'run_id': run['run_id'], 'method': run['method'],
            'haplotypes': len(records), 'sha256': digest.hexdigest()}


def compare_paths(reference_path: pathlib.Path, mask_path: pathlib.Path,
                  baseline_path: pathlib.Path, stitched_path: pathlib.Path,
                  method: str) -> pd.DataFrame:
    """Compare each haplotype once, with within-region agreement denominators.

    Switch categories follow :func:`ch5_stitching.comparison_metrics`. The
    full-HMM path is partitioned by the same saved mask as each stitched path.
    Percent increases use each haplotype's full-HMM switch count; a zero count
    gives an undefined percentage. Empty regions give undefined agreement.
    """
    reference = tskit.load(reference_path)
    baseline = ch5_hyperparams.read_matches(baseline_path)
    stitched = ch5_hyperparams.read_matches(stitched_path)
    with np.load(mask_path, allow_pickle=False) as archive:
        mask = {name: archive[name] for name in archive.files}
    assert int(mask['num_sites']) == reference.num_sites
    roster = list(zip(mask['sample_id'].tolist(), mask['ploidy_index'].tolist(),
                      strict=True))
    assert len(roster) == len(set(roster))
    assert set(roster) == set(baseline) == set(stitched)
    rows = []
    for index, identity in enumerate(roster):
        first, last = mask['offsets'][index:index + 2]
        left = mask['left_site_index'][first:last]
        right = mask['right_site_index'][first:last]
        is_gap = mask['is_gap'][first:last]
        assert left[0] == 0 and right[-1] == reference.num_sites
        assert np.all(left < right) and np.array_equal(right[:-1], left[1:])
        covered = np.zeros(reference.num_sites, dtype=bool)
        for start, stop in zip(left[~is_gap], right[~is_gap], strict=True):
            covered[start:stop] = True
        full_parents = ch5_hyperparams.parents_at_sites(baseline[identity], reference)
        stitched_parents = ch5_hyperparams.parents_at_sites(stitched[identity], reference)
        metrics = ch5_stitching.comparison_metrics(full_parents, stitched_parents, covered)
        full_switches = metrics['full_switches']
        metrics['switch_increase_percentage'] = (
            100 * metrics['num_extra_switches'] / full_switches
            if full_switches > 0 else np.nan
        )
        for region in ('focal', 'gap'):
            sites = metrics[f'{region}_sites']
            disagreements = metrics[f'{region}_disagreement_sites']
            metrics[f'{region}_agreement_percentage'] = (
                100 * (sites - disagreements) / sites if sites > 0 else np.nan
            )
        rows.append({'method': method, 'sample_id': identity[0],
                     'ploidy_index': identity[1], **metrics})
    return pd.DataFrame(rows)


def reference_phases(runs: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    """Partition measured totals using non-overlapping process wall stages.

    Worker service times are deliberately excluded. Assembly/output is folded
    into the residual along with loading/setup, so each stack equals the
    directly measured total. Full HMM is kept as a single inclusive stage.
    """
    process = stages.loc[stages.scope.eq('process')]
    process = process.pivot(index='run_id', columns='stage', values='wall_s')
    selected = runs.loc[runs['mode'].eq('from_reference')].set_index('run_id')
    rows = []
    for run_id, run in selected.iterrows():
        if run['method'] == 'full_hmm':
            rows.append({'run_id': run_id, 'method': run['method'],
                         'phase': 'full_pipeline', 'wall_s': run['wall_s']})
        else:
            measured = process.loc[run_id]
            matching = measured['matching']
            preparation = measured['focal_preparation']
            residual = run['wall_s'] - matching - preparation
            assert residual >= 0
            values = {'matching': matching, 'focal_preparation': preparation,
                      'other': residual}
            for phase, value in values.items():
                rows.append({'run_id': run_id, 'method': run['method'],
                             'phase': phase, 'wall_s': value})
    return pd.DataFrame(rows)


def per_sample_data(worker_times: pd.DataFrame, mask_path: pathlib.Path,
                    reference_path: pathlib.Path, population_path: pathlib.Path,
                    settings: dict, resident_runs: pd.DataFrame) -> pd.DataFrame:
    """Average recorded stages over trials and both haplotypes of each sample.

    Input rows contain one summed worker-stage elapsed time per haplotype and
    trial, in seconds. Physical focal coverage uses the same clipped mask-edge
    convention as :func:`ch5_hyperparams.coverage_percentage`.
    """
    keys = ['method', 'repeat', 'sample_id', 'ploidy_index']
    assert not worker_times.duplicated(keys).any()
    assert worker_times.service_s.ge(0).all()
    positions = tskit.load(reference_path).sites_position
    coverage_rows = []
    with np.load(mask_path, allow_pickle=False) as mask:
        for index, sample_id in enumerate(mask['sample_id']):
            first, last = mask['offsets'][index:index + 2]
            percentage = ch5_hyperparams.coverage_percentage(
                mask['left_site_index'][first:last],
                mask['right_site_index'][first:last], mask['is_gap'][first:last],
                positions, settings['analysis_span_bp'],
            )
            coverage_rows.append({'sample_id': str(sample_id),
                                  'ploidy_index': int(mask['ploidy_index'][index]),
                                  'focal_coverage_percentage': percentage})
    coverage = pd.DataFrame(coverage_rows)
    trials = resident_runs.loc[resident_runs.method.ne('full_hmm'), ['method', 'repeat']]
    expected = trials.merge(coverage, how='cross')
    measured = expected.merge(worker_times, on=keys, how='outer',
                              validate='one_to_one', indicator=True)
    assert measured['_merge'].eq('both').all()
    per_haplotype = measured.groupby(['method', 'sample_id', 'ploidy_index']).agg(
        worker_time_ms=('service_s', 'mean'),
        focal_coverage_percentage=('focal_coverage_percentage', 'first'),
    ).reset_index()
    per_haplotype['worker_time_ms'] *= 1000
    samples = per_haplotype.groupby(['method', 'sample_id']).agg(
        worker_time_ms=('worker_time_ms', 'mean'),
        focal_coverage_percentage=('focal_coverage_percentage', 'mean'),
        haplotypes=('ploidy_index', 'nunique'),
    ).reset_index()
    assert samples.haplotypes.eq(2).all()
    populations = pd.read_csv(population_path)
    populations = populations[['sample_id', settings['population_column']]]
    populations = populations.rename(columns={settings['population_column']: 'population'})
    samples = samples.merge(populations, on='sample_id', how='left', validate='many_to_one')
    assert samples.population.notna().all()
    return samples


def load_results(config_path: pathlib.Path) -> Results:
    """Load the saved suite, verify repeat invariance, and derive correctness."""
    with config_path.open() as source:
        config = yaml.safe_load(source)
    root = ch5_hyperparams.resolve_path(config['benchmark_dir'], config_path.parent)
    manifest = json.loads((root / 'manifest.json').read_text())
    runs = pd.read_csv(root / 'runs.csv')
    runs = runs.loc[runs.dataset.eq(config['dataset'])].copy()
    stages = pd.read_csv(root / 'stages.csv')
    stages = stages.loc[stages.run_id.isin(runs.run_id)].copy()
    quality = pd.DataFrame(json.loads((root / 'quality.json').read_text()))
    quality = quality.loc[quality.run_id.isin(runs.run_id)].copy()
    expected = manifest['configuration']['performance']
    expected_trials = len(expected['methods']) * len(expected['modes'])
    expected_trials *= len(expected['worker_counts']) * expected['repeats']
    assert len(runs) == expected_trials and not runs.run_id.duplicated().any()
    assert runs.status.eq('ok').all()
    assert set(runs.method) == set(config['methods'])
    assert len(quality) == len(runs) and quality.status.eq('ok').all()
    assert quality.reconstructed_called_allele_errors.eq(0).all()
    assert quality.mismatches.eq(0).all()
    assert runs.actual_workers.nunique() == 1
    assert runs.matched_haplotypes.nunique() == 1
    jobs = runs.to_dict('records')
    workers = config['workers']
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    context = multiprocessing.get_context('spawn')
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        fingerprints = pd.DataFrame(pool.map(path_fingerprint, jobs))
    assert fingerprints.groupby('method').sha256.nunique().eq(1).all()
    assert fingerprints.haplotypes.eq(runs.matched_haplotypes.iloc[0]).all()
    selection = config['correctness']
    chosen = runs.loc[runs['mode'].eq(selection['mode'])]
    chosen = chosen.loc[chosen['repeat'].eq(selection['repeat'])].set_index('method')
    assert len(chosen) == len(config['methods'])
    inputs = manifest['datasets'][config['dataset']]['inputs']
    reference_path = pathlib.Path(inputs['reference'])
    baseline_path = pathlib.Path(chosen.loc['full_hmm', 'output_matches'])
    tables = []
    for method in config['methods']:
        if method == 'full_hmm':
            continue
        run = chosen.loc[method]
        mask_path = pathlib.Path(run['output_matches']).parent / 'focal_mask.npz'
        # All switch-composition bars must use one identical AC/budget partition.
        with np.load(mask_path, allow_pickle=False) as trial_mask:
            with np.load(inputs['mask'], allow_pickle=False) as common_mask:
                for name in common_mask.files:
                    assert np.array_equal(trial_mask[name], common_mask[name])
        table = compare_paths(reference_path, mask_path, baseline_path,
                              pathlib.Path(run['output_matches']), method)
        report = quality.loc[quality.run_id.eq(run['run_id'])].iloc[0]
        assert table.stitched_switches.sum() == report['switches']
        disagreements = table.focal_disagreement_sites + table.gap_disagreement_sites
        sites = table.focal_sites + table.gap_sites
        assert sites.sum() - disagreements.sum() == report['parent_agreement_sites']
        tables.append(table)
    return Results(manifest, runs, stages, quality, fingerprints,
                   pd.concat(tables, ignore_index=True))
