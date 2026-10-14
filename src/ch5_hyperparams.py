"""Summarise current match-eval paths over each saved focal-mask partition.

:func:`load_results` reads the configured parameter grid in worker processes.
Only scalar metrics and interval lengths return to the notebook; parent arrays
are evaluated one haplotype at a time using :mod:`ch5_stitching`.

:func:`load_tree_sequences` exposes saved raw output statistics, while
:func:`arg_metrics` compares sample-aligned stitched ARGs with Full HMM outputs.
:func:`truth_arg_metrics` separately compares simulated outputs with truth.
"""

import concurrent.futures as cf
import dataclasses
import itertools
import json
import logging
import multiprocessing
import pathlib

import numpy as np
import pandas as pd
import tscompare
import tskit
import yaml

import ch5_stitching

logger = logging.getLogger(__name__)


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
class Results:
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
    ac_cutoff: int
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


def population_labels(settings: dict, config_path: pathlib.Path) -> dict:
    """Read preserved simulation labels or 1KGP pedigree superpopulations."""
    path = resolve_path(settings['population_metadata'], config_path.parent)
    metadata = pd.read_csv(path)
    assert not metadata.sample_id.duplicated().any()
    labels = {}
    for row in metadata.to_dict('records'):
        for ploidy_index in range(2):
            labels[(row['sample_id'], ploidy_index)] = row[settings['population_column']]
    return labels


def read_matches(path: pathlib.Path) -> dict:
    """Read unique sample paths; baseline paths are emitted right to left."""
    records = {}
    with path.open() as file:
        for line in file:
            record = json.loads(line)
            assert record['source'] == 'samples'
            key = (record['sample_id'], record['ploidy_index'])
            assert key not in records
            record['path'].sort(key=lambda segment: segment['left'])
            records[key] = record
    return records


def parents_at_sites(record: dict, reference: tskit.TreeSequence) -> np.ndarray:
    """Evaluate a complete half-open path in original reference node IDs."""
    path = record['path']
    left = np.array([segment['left'] for segment in path])
    right = np.array([segment['right'] for segment in path])
    parents = np.array([segment['parent'] for segment in path], dtype=np.int32)
    assert left[0] == 0 and right[-1] == reference.sequence_length
    assert np.all(left < right) and np.array_equal(right[:-1], left[1:])
    assert np.all(parents[1:] != parents[:-1])
    assert np.all((parents >= 0) & (parents < reference.num_nodes))
    segments = np.searchsorted(right, reference.sites_position, side='right')
    result = parents[segments]
    assert np.count_nonzero(result[1:] != result[:-1]) == len(path) - 1
    return result


def coverage_percentage(left: np.ndarray, right: np.ndarray, is_gap: np.ndarray,
                        positions: np.ndarray, span: list[float]) -> float:
    """Measure physical coverage, clipping native run edges to the analysis span.

    As in match-eval's path assembly, an internal edge is the position of its
    right-hand site. Terminal edges use the window bounds, excluding the
    unanalysed chromosome tail retained by the reference tree sequence.
    """
    edges = np.concatenate(([span[0]], positions[1:], [span[1]]))
    edges = np.clip(edges, span[0], span[1])
    widths = edges[right] - edges[left]
    return float(100 * widths[~is_gap].sum() / (span[1] - span[0]))


def summarise_run(inputs: RunInputs) -> Results:
    """Validate a run and derive counts, within-region agreement and lengths."""
    name = inputs.dataset
    reference = tskit.load(inputs.data_dir / 'ancestors' / f'{name}_inferred_ancestors.trees')
    positions = reference.sites_position
    assert inputs.analysis_span_bp[0] <= positions[0]
    assert positions[-1] < inputs.analysis_span_bp[1]
    full = read_matches(inputs.data_dir / 'matches' / f'{name}_inferred_samples_matches.jsonl')
    prefix = f'{name}_inferred_ac{inputs.ac_cutoff}_k{inputs.max_mismatches}'
    stitched_path = inputs.data_dir / 'matches' / f'{prefix}_{inputs.method}_stitched_samples_matches.jsonl'
    stitched = read_matches(stitched_path)
    mask_path = inputs.data_dir / 'focal_masks' / f'{prefix}_focal_mask.npz'
    with np.load(mask_path, allow_pickle=False) as archive:
        mask = {field: archive[field] for field in archive.files}
    assert int(mask['num_sites']) == len(positions)
    assert int(mask['ac_cutoff']) == inputs.ac_cutoff
    assert int(mask['max_mismatches']) == inputs.max_mismatches
    assert str(mask['focal_choice']) == inputs.focal_choice
    roster = list(zip(mask['sample_id'].tolist(), mask['ploidy_index'].tolist(), strict=True))
    assert len(roster) == len(set(roster))
    assert set(roster) == set(full) == set(stitched)
    rows = []
    length_tables = []
    for index, key in enumerate(roster):
        record = stitched[key]
        assert record['ac_cutoff'] == inputs.ac_cutoff
        assert record['max_mismatches'] == inputs.max_mismatches
        assert record['method'] == inputs.method
        start, stop = mask['offsets'][index:index + 2]
        left = mask['left_site_index'][start:stop]
        right = mask['right_site_index'][start:stop]
        is_gap = mask['is_gap'][start:stop]
        assert left[0] == 0 and right[-1] == len(positions)
        assert np.all(left < right) and np.array_equal(right[:-1], left[1:])
        assert np.all(is_gap[1:] != is_gap[:-1])
        covered = np.zeros(len(positions), dtype=bool)
        for first, last in zip(left[~is_gap], right[~is_gap], strict=True):
            covered[first:last] = True
        full_parents = parents_at_sites(full[key], reference)
        stitched_parents = parents_at_sites(record, reference)
        metrics = ch5_stitching.comparison_metrics(full_parents, stitched_parents, covered)
        denominator = metrics['full_switches']
        metrics['switch_increase_percentage'] = (
            100 * metrics['num_extra_switches'] / denominator if denominator > 0 else np.nan
        )
        for region in ['focal', 'gap']:
            sites = metrics[f'{region}_sites']
            disagreements = metrics[f'{region}_disagreement_sites']
            metrics[f'{region}_agreement_percentage'] = (
                100 * (sites - disagreements) / sites if sites > 0 else np.nan
            )
        metrics['focal_coverage_percentage'] = coverage_percentage(
            left, right, is_gap, positions, inputs.analysis_span_bp
        )
        metrics['gap_coverage_percentage'] = 100 - metrics['focal_coverage_percentage']
        identity = {
            'dataset': name, 'sample_id': key[0], 'ploidy_index': key[1],
            'ac_cutoff': inputs.ac_cutoff, 'max_mismatches': inputs.max_mismatches,
            'method': inputs.method,
        }
        population = inputs.labels[key]
        assert population is not None and population != ''
        group = 'African' if population == inputs.african_population else 'Non-African'
        rows.append({**identity, 'population': population, 'population_group': group, **metrics})
        lengths = pd.DataFrame({'num_sites': right - left, 'region': np.where(is_gap, 'gap', 'focal')})
        for field, value in identity.items():
            lengths[field] = value
        length_tables.append(lengths)
    return Results(pd.DataFrame(rows), pd.concat(length_tables, ignore_index=True))


def load_results(config_path: pathlib.Path, workers: int | None = None) -> Results:
    """Read all configured HMM products in parallel without rerunning inference."""
    with config_path.open() as file:
        config = yaml.safe_load(file)
    experiment_path = resolve_path(config['experiment_config'], config_path.parent)
    with experiment_path.open() as file:
        experiment = yaml.safe_load(file)
    assert config['method'] in experiment['focal_methods']
    data_dir = resolve_path(experiment['data_dir'], experiment_path.parent)
    jobs = []
    for settings in config['datasets']:
        labels = population_labels(settings, config_path)
        grid = itertools.product(experiment['ac_cutoff'], experiment['max_mismatches'])
        for cutoff, mismatches in grid:
            jobs.append(RunInputs(data_dir, settings['name'], labels,
                                  settings['african_population'], settings['analysis_span_bp'],
                                  cutoff, mismatches, config['method'], experiment['focal_choice']))
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    if workers == 1:
        results = [summarise_run(job) for job in jobs]
    else:
        context = multiprocessing.get_context('spawn')
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            results = list(pool.map(summarise_run, jobs))
    comparisons = pd.concat([result.comparisons for result in results], ignore_index=True)
    lengths = pd.concat([result.lengths for result in results], ignore_index=True)
    return Results(comparisons, lengths)


def load_tree_sequences(config_path: pathlib.Path) -> TreeSequences:
    """Load saved raw TS outputs and report their unmodified edge/tree counts.

    :func:`arg_metrics` prepares private sample-aligned, window-restricted
    versions for Full HMM comparison. Statistics here describe the saved files,
    including the raw reference ancestry, rather than those prepared versions.
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)
    experiment_path = resolve_path(config['experiment_config'], config_path.parent)
    with experiment_path.open() as file:
        experiment = yaml.safe_load(file)
    data_dir = resolve_path(experiment['data_dir'], experiment_path.parent)
    full = {}
    stitched = {}
    rows = []
    for settings in config['datasets']:
        name = settings['name']
        full[name] = tskit.load(data_dir / 'matches' / f'{name}_inferred_samples_raw.trees')
        rows.append({'dataset': name, 'method': 'full', 'ac_cutoff': None,
                     'max_mismatches': None, 'num_edges': full[name].num_edges,
                     'num_trees': full[name].num_trees})
        grid = itertools.product(experiment['ac_cutoff'], experiment['max_mismatches'])
        for cutoff, allowance in grid:
            prefix = f'{name}_inferred_ac{cutoff}_k{allowance}_{config["method"]}'
            path = data_dir / 'matches' / f'{prefix}_stitched_samples_raw.trees'
            ts = tskit.load(path)
            stitched[(name, cutoff, allowance)] = ts
            assert np.array_equal(ts.samples(), full[name].samples())
            rows.append({'dataset': name, 'method': 'stitched', 'ac_cutoff': cutoff,
                         'max_mismatches': allowance, 'num_edges': ts.num_edges,
                         'num_trees': ts.num_trees})
    return TreeSequences(full, stitched, pd.DataFrame(rows))


def tree_statistic_differences(statistics: pd.DataFrame) -> pd.DataFrame:
    """Subtract each dataset's raw Full HMM counts from its stitched counts."""
    metrics = ['num_edges', 'num_trees']
    full = statistics.loc[statistics.method.eq('full'), ['dataset', *metrics]]
    full = full.set_index('dataset')
    stitched = statistics.loc[statistics.method.eq('stitched')].copy()
    differences = stitched.join(full, on='dataset', rsuffix='_full', validate='many_to_one')
    for metric in metrics:
        differences[f'{metric}_difference'] = differences[metric] - differences[f'{metric}_full']
    return differences


def align_tree_sequences(comparison: tskit.TreeSequence, reference: tskit.TreeSequence,
                         span: list[float],
                         reference_nodes: dict | None = None) -> AlignedTrees:
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
            identity = (metadata['sample_id'], metadata['ploidy_index'])
            reference_nodes[identity] = int(node_id)
    if len(reference_nodes) != reference.num_samples:
        raise ValueError('Reference TS sample haplotype identities must be unique')
    comparison_samples = comparison.samples()
    matched_reference_nodes = []
    identities = []
    for node_id in comparison_samples:
        metadata = comparison.node(int(node_id)).metadata
        identity = (metadata['sample_id'], metadata['ploidy_index'])
        identities.append(identity)
        matched_reference_nodes.append(reference_nodes[identity])
    if len(identities) != len(set(identities)):
        raise ValueError('Matched TS sample haplotype identities must be unique')
    if set(matched_reference_nodes) != set(reference.samples()):
        raise ValueError('Comparison and reference TS must contain the same sample haplotypes')
    comparison_window = window_tree_sequence(comparison, span)
    reference_window = window_tree_sequence(reference, span)
    comparison_aligned = comparison_window.simplify(samples=comparison_samples, keep_unary=True)
    reference_aligned = reference_window.simplify(samples=matched_reference_nodes, keep_unary=True)
    assert comparison_aligned.sequence_length == reference_aligned.sequence_length == span[1] - span[0]
    assert np.array_equal(comparison_aligned.samples(), reference_aligned.samples())
    return AlignedTrees(comparison_aligned, reference_aligned)


def window_tree_sequence(ts: tskit.TreeSequence, span: list[float]) -> tskit.TreeSequence:
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
    return {'dataset': inputs.dataset, 'ac_cutoff': inputs.ac_cutoff,
            'max_mismatches': inputs.max_mismatches,
            'arf': float(result.arf), 'tpr': float(result.tpr),
            'comparison_method': inputs.comparison_method,
            'reference_method': inputs.reference_method,
            'comparison_time_units': inputs.comparison.time_units,
            'reference_time_units': inputs.reference.time_units}


def arg_metrics(trees: TreeSequences, settings: list[dict],
                workers: int | None = None) -> pd.DataFrame:
    """Compare every stitched ARG with its dataset's Full HMM in parallel."""
    spans = {item['name']: item['analysis_span_bp'] for item in settings}
    jobs = []
    for (name, cutoff, allowance), comparison in trees.stitched.items():
        aligned = align_tree_sequences(comparison, trees.full[name], spans[name])
        jobs.append(ArgComparison(name, cutoff, allowance, aligned.comparison, aligned.reference,
                                  'stitched_hmm', 'full_hmm'))
    return run_arg_comparisons(jobs, workers)


def run_arg_comparisons(jobs: list[ArgComparison],
                        workers: int | None = None) -> pd.DataFrame:
    """Evaluate independently aligned ARG comparisons across available CPUs."""
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    if workers == 1:
        rows = [compare_arg(job) for job in jobs]
    else:
        context = multiprocessing.get_context('spawn')
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(compare_arg, jobs))
    return pd.DataFrame(rows)


def truth_sample_nodes(truth: tskit.TreeSequence) -> dict:
    """Join truth node IDs to the VCF haplotype identities used by match-eval."""
    mapping = truth.map_to_vcf_model()
    nodes_by_identity = {}
    for name, nodes in zip(mapping.individuals_name, mapping.individuals_nodes, strict=True):
        for ploidy_index, node_id in enumerate(nodes):
            nodes_by_identity[(name, ploidy_index)] = int(node_id)
    return nodes_by_identity


def truth_arg_metrics(trees: TreeSequences, config_path: pathlib.Path,
                      workers: int | None = None) -> pd.DataFrame:
    """Compare simulated Full HMM and every stitched run with the true ARG.

    The Full HMM baseline is evaluated once at the configured display cutoff;
    its mismatch allowance is undefined because it does not use focal masks.
    """
    with config_path.open() as file:
        config = yaml.safe_load(file)
    experiment_path = resolve_path(config['experiment_config'], config_path.parent)
    with experiment_path.open() as file:
        experiment = yaml.safe_load(file)
    dataset_configs = {item['name']: item for item in experiment['datasets']}
    jobs = []
    for settings in config['datasets']:
        name = settings['name']
        truth_path = dataset_configs[name]['ts_path']
        if truth_path is None:
            continue
        truth = tskit.load(resolve_path(truth_path, experiment_path.parent))
        reference_nodes = truth_sample_nodes(truth)
        aligned = align_tree_sequences(trees.full[name], truth, settings['analysis_span_bp'],
                                       reference_nodes=reference_nodes)
        display_cutoff = config['truth_comparison']['full_hmm_cutoff']
        jobs.append(ArgComparison(name, display_cutoff, None, aligned.comparison,
                                  aligned.reference, 'full_hmm', 'true_arg'))
        for (dataset, cutoff, allowance), comparison in trees.stitched.items():
            if dataset != name:
                continue
            aligned = align_tree_sequences(comparison, truth, settings['analysis_span_bp'],
                                           reference_nodes=reference_nodes)
            jobs.append(ArgComparison(name, cutoff, allowance, aligned.comparison,
                                      aligned.reference, 'stitched_hmm', 'true_arg'))
    return run_arg_comparisons(jobs, workers)
