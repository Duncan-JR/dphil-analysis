"""Load Greedy AC 10/40 benchmark passes without altering the pt4 analysis.

Reuse :mod:`ch5_perf` path comparisons, timing phases and sample aggregation.
Full HMM timings are pooled across passes; Greedy cases retain distinct masks.
"""

import concurrent.futures as cf
import dataclasses
import json
import logging
import multiprocessing
import pathlib

import ch5_hyperparams
import ch5_perf
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Results:
    """Validated runs, correctness comparisons and population-labelled samples."""

    runs: pd.DataFrame
    stages: pd.DataFrame
    quality: pd.DataFrame
    fingerprints: pd.DataFrame
    comparisons: pd.DataFrame
    samples: pd.DataFrame


def load_results(config_path: pathlib.Path) -> Results:
    """Read both completed passes, preserving every timing repeat."""
    with config_path.open() as source:
        config = yaml.safe_load(source)
    root = ch5_hyperparams.resolve_path(config['benchmark_dir'], config_path.parent)
    completion = json.loads((root / 'complete.json').read_text())
    selection = config['correctness']
    sample_settings = config['per_sample']
    population_path = ch5_hyperparams.resolve_path(
        sample_settings['population_metadata'], config_path.parent)
    run_tables = []
    stage_tables = []
    quality_tables = []
    comparison_tables = []
    sample_tables = []
    for method, settings in config['methods'].items():
        if method == 'full_hmm':
            continue
        cutoff = settings['ac_cutoff']
        assert cutoff in completion['ac_cutoff']
        folder = root / f'ac{cutoff}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        expected = manifest['configuration']['performance']
        assert set(expected['methods']) == {'full_hmm', 'greedy'}
        runs = pd.read_csv(folder / 'runs.csv')
        runs = runs.loc[runs.dataset.eq(config['dataset'])].copy()
        expected_trials = len(expected['methods']) * len(expected['modes'])
        expected_trials *= len(expected['worker_counts']) * expected['repeats']
        assert len(runs) == expected_trials
        assert runs.status.eq('ok').all() and runs.ac_cutoff.eq(cutoff).all()
        runs['method'] = runs.method.replace({'greedy': method})
        run_tables.append(runs)
        stages = pd.read_csv(folder / 'stages.csv')
        stages = stages.loc[stages.run_id.isin(runs.run_id)].copy()
        stage_tables.append(stages)
        quality = pd.DataFrame(json.loads((folder / 'quality.json').read_text()))
        quality = quality.loc[quality.run_id.isin(runs.run_id)].copy()
        assert len(quality) == len(runs) and quality.status.eq('ok').all()
        assert quality.reconstructed_called_allele_errors.eq(0).all()
        assert quality.mismatches.eq(0).all()
        quality_tables.append(quality)
        inputs = manifest['datasets'][config['dataset']]['inputs']
        chosen = runs.loc[runs['mode'].eq(selection['mode'])]
        chosen = chosen.loc[chosen['repeat'].eq(selection['repeat'])].set_index('method')
        assert len(chosen) == 2
        mask_path = pathlib.Path(inputs['mask'])
        trial_folder = pathlib.Path(chosen.loc[method, 'output_matches']).parent
        trial_mask_path = trial_folder / f'ac{cutoff}_focal_mask.npz'
        with np.load(mask_path, allow_pickle=False) as common_mask:
            with np.load(trial_mask_path, allow_pickle=False) as trial_mask:
                for name in common_mask.files:
                    assert np.array_equal(trial_mask[name], common_mask[name])
        table = ch5_perf.compare_paths(
            pathlib.Path(inputs['reference']), mask_path,
            pathlib.Path(chosen.loc['full_hmm', 'output_matches']),
            pathlib.Path(chosen.loc[method, 'output_matches']), method)
        report = quality.loc[quality.run_id.eq(chosen.loc[method, 'run_id'])].iloc[0]
        assert table.stitched_switches.sum() == report['switches']
        disagreements = table.focal_disagreement_sites + table.gap_disagreement_sites
        sites = table.focal_sites + table.gap_sites
        assert sites.sum() - disagreements.sum() == report['parent_agreement_sites']
        comparison_tables.append(table)
        resident = runs.loc[runs['mode'].eq('resident_match') & runs.method.eq(method)]
        worker_tables = []
        for run in resident.itertuples():
            path = pathlib.Path(run.output_matches).parent / sample_settings['worker_times_filename']
            worker = pd.read_csv(path)
            measured = stages.loc[stages.run_id.eq(run.run_id) & stages.scope.eq('worker')]
            assert np.isclose(worker.service_s.sum(), measured.service_s.sum())
            worker['method'] = method
            worker['repeat'] = run.repeat
            worker_tables.append(worker)
        samples = ch5_perf.per_sample_data(
            pd.concat(worker_tables, ignore_index=True), mask_path,
            pathlib.Path(inputs['reference']), population_path, sample_settings, resident)
        samples['ac_cutoff'] = cutoff
        sample_tables.append(samples)
    runs = pd.concat(run_tables, ignore_index=True)
    assert not runs.run_id.duplicated().any()
    assert runs.actual_workers.nunique() == 1
    assert runs.matched_haplotypes.nunique() == 1
    jobs = runs.to_dict('records')
    workers = config['workers']
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    context = multiprocessing.get_context('spawn')
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        fingerprints = pd.DataFrame(pool.map(ch5_perf.path_fingerprint, jobs))
    assert fingerprints.groupby('method').sha256.nunique().eq(1).all()
    assert fingerprints.haplotypes.eq(runs.matched_haplotypes.iloc[0]).all()
    return Results(runs, pd.concat(stage_tables, ignore_index=True),
                   pd.concat(quality_tables, ignore_index=True), fingerprints,
                   pd.concat(comparison_tables, ignore_index=True),
                   pd.concat(sample_tables, ignore_index=True))
