"""Read minimum-focal-size experiments using the pt3 metric definitions.

Reuse :mod:`ch5_hyperparams` for path metrics, sample alignment and parallel
ARG comparisons. Every observation retains its actual AC cutoff, zero mismatch
budget and minimum size in inference sites.
"""

import concurrent.futures as cf
import itertools
import logging
import multiprocessing
import pathlib

import ch5_hyperparams
import numpy as np
import pandas as pd
import tskit
import yaml

logger = logging.getLogger(__name__)


def load_configuration(config_path: pathlib.Path) -> dict:
    """Resolve the analysis settings and the current experiment grid."""
    with config_path.open() as source:
        config = yaml.safe_load(source)
    experiment_path = ch5_hyperparams.resolve_path(config['experiment_config'], config_path.parent)
    with experiment_path.open() as source:
        experiment = yaml.safe_load(source)
    assert experiment['max_mismatches'] == [0]
    assert config['method'] in experiment['focal_methods']
    config['experiment'] = experiment
    config['experiment_path'] = experiment_path
    config['data_dir'] = ch5_hyperparams.resolve_path(experiment['data_dir'], experiment_path.parent)
    return config


def load_results(config_path: pathlib.Path, workers: int | None = None) -> ch5_hyperparams.Results:
    """Recompute every dataset/cutoff/minimum combination in parallel."""
    config = load_configuration(config_path)
    experiment = config['experiment']
    jobs = []
    minimums = []
    for settings in config['datasets']:
        labels = ch5_hyperparams.population_labels(settings, config_path)
        grid = itertools.product(experiment['ac_cutoff'], experiment['min_focal_region_size'])
        for cutoff, minimum in grid:
            jobs.append(ch5_hyperparams.RunInputs(
                config['data_dir'], settings['name'], labels, settings['african_population'],
                settings['analysis_span_bp'], cutoff, experiment['max_mismatches'][0],
                config['method'], experiment['focal_choice']))
            minimums.append(minimum)
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    context = multiprocessing.get_context('spawn')
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        results = list(pool.map(ch5_hyperparams.summarise_run, jobs, minimums))
    comparisons = pd.concat([result.comparisons for result in results], ignore_index=True)
    lengths = pd.concat([result.lengths for result in results], ignore_index=True)
    return ch5_hyperparams.Results(comparisons, lengths)


def load_tree_sequences(config_path: pathlib.Path) -> ch5_hyperparams.TreeSequences:
    """Load raw trees with stitched keys (dataset, cutoff, minimum size)."""
    config = load_configuration(config_path)
    experiment = config['experiment']
    full = {}
    stitched = {}
    rows = []
    for settings in config['datasets']:
        name = settings['name']
        full[name] = tskit.load(config['data_dir'] / 'matches' / f'{name}_inferred_samples_raw.trees')
        rows.append({'dataset': name, 'method': 'full', 'ac_cutoff': None,
                     'max_mismatches': None, 'min_focal_region_size': None,
                     'num_edges': full[name].num_edges, 'num_trees': full[name].num_trees})
        grid = itertools.product(experiment['ac_cutoff'], experiment['min_focal_region_size'])
        for cutoff, minimum in grid:
            budget = experiment['max_mismatches'][0]
            prefix = f'{name}_inferred_ac{cutoff}_k{budget}_min{minimum}_{config["method"]}'
            path = config['data_dir'] / 'matches' / f'{prefix}_stitched_samples_raw.trees'
            ts = tskit.load(path)
            assert np.array_equal(ts.samples(), full[name].samples())
            stitched[(name, cutoff, minimum)] = ts
            rows.append({'dataset': name, 'method': 'stitched', 'ac_cutoff': cutoff,
                         'max_mismatches': budget, 'min_focal_region_size': minimum,
                         'num_edges': ts.num_edges, 'num_trees': ts.num_trees})
    return ch5_hyperparams.TreeSequences(full, stitched, pd.DataFrame(rows))


def arg_metrics(trees: ch5_hyperparams.TreeSequences, settings: list[dict],
                workers: int | None = None) -> pd.DataFrame:
    """Compare each minimum-size Greedy ARG against its Full HMM output."""
    spans = {item['name']: item['analysis_span_bp'] for item in settings}
    jobs = []
    minimums = []
    for (name, cutoff, minimum), comparison in trees.stitched.items():
        aligned = ch5_hyperparams.align_tree_sequences(comparison, trees.full[name], spans[name])
        jobs.append(ch5_hyperparams.ArgComparison(
            name, cutoff, 0, aligned.comparison, aligned.reference, 'stitched_greedy', 'full_hmm'))
        minimums.append(minimum)
    metrics = ch5_hyperparams.run_arg_comparisons(jobs, workers)
    metrics['min_focal_region_size'] = minimums
    return metrics


def truth_arg_metrics(trees: ch5_hyperparams.TreeSequences, config_path: pathlib.Path,
                      workers: int | None = None) -> pd.DataFrame:
    """Compare the simulated Full HMM and every Greedy case against truth."""
    config = load_configuration(config_path)
    experiment_datasets = {item['name']: item for item in config['experiment']['datasets']}
    jobs = []
    minimums = []
    for settings in config['datasets']:
        name = settings['name']
        truth_path = experiment_datasets[name]['ts_path']
        if truth_path is None:
            continue
        truth_path = ch5_hyperparams.resolve_path(truth_path, config['experiment_path'].parent)
        truth = tskit.load(truth_path)
        reference_nodes = ch5_hyperparams.truth_sample_nodes(truth)
        aligned = ch5_hyperparams.align_tree_sequences(
            trees.full[name], truth, settings['analysis_span_bp'], reference_nodes=reference_nodes)
        jobs.append(ch5_hyperparams.ArgComparison(
            name, None, None, aligned.comparison, aligned.reference, 'full_hmm', 'true_arg'))
        minimums.append(None)
        for (dataset, cutoff, minimum), comparison in trees.stitched.items():
            if dataset != name:
                continue
            aligned = ch5_hyperparams.align_tree_sequences(
                comparison, truth, settings['analysis_span_bp'], reference_nodes=reference_nodes)
            jobs.append(ch5_hyperparams.ArgComparison(
                name, cutoff, 0, aligned.comparison, aligned.reference, 'stitched_greedy', 'true_arg'))
            minimums.append(minimum)
    metrics = ch5_hyperparams.run_arg_comparisons(jobs, workers)
    metrics['min_focal_region_size'] = minimums
    return metrics
