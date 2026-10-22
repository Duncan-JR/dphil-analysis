"""Classify saved gap copying by frequency age and haplotype focal eligibility.

The archived reference has exactly one mutation at every inference site. These
identify the original focal sites; the configured left/right seed is recovered
per ancestor. Carrier status uses the saved Full HMM sample haplotypes, fixed
across methods. No genotype stores or new matching runs are required.
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
class GapParentJob:
    root: pathlib.Path
    cache: pathlib.Path
    dataset: dict
    choice: str
    expected_eligible: dict


def focal_seed_sites(reference: tskit.TreeSequence, choice: str) -> np.ndarray:
    """Recover original focal sites under the checked single-mutation invariant."""
    counts = np.bincount(reference.mutations_site, minlength=reference.num_sites)
    assert np.all(counts == 1), 'Reference focal-site recovery requires one mutation per site'
    if choice == 'left':
        seeds = np.full(reference.num_nodes, reference.num_sites, dtype=np.int64)
        np.minimum.at(seeds, reference.mutations_node, reference.mutations_site)
        seeds[seeds == reference.num_sites] = -1
    else:
        assert choice == 'right'
        seeds = np.full(reference.num_nodes, -1, dtype=np.int64)
        np.maximum.at(seeds, reference.mutations_node, reference.mutations_site)
    return seeds


def gap_fragments(left, right, parents, gap_left, gap_right, positions, analysis_intervals):
    """Aggregate edge/gap intersections by parent for one sample haplotype.

    Site counts include the last inference site. Physical spans are separately
    clipped to the archived analysis intervals, whose final endpoint is the
    last observed position. Touching edges with the same parent in one gap
    count as one copying segment.
    """
    boundaries = np.unique(np.concatenate([left, right, gap_left, gap_right]))
    first = boundaries[:-1]
    last = boundaries[1:]
    edge = np.searchsorted(left, first, side='right') - 1
    gap = np.searchsorted(gap_left, first, side='right') - 1
    valid = (edge >= 0) & (gap >= 0)
    valid &= first < right[np.maximum(edge, 0)]
    valid &= first < gap_right[np.maximum(gap, 0)]
    first, last, edge, gap = first[valid], last[valid], edge[valid], gap[valid]
    copied = parents[edge]
    sites = np.searchsorted(positions, last) - np.searchsorted(positions, first)
    span = np.zeros(len(first), dtype=float)
    for lo, hi in analysis_intervals:
        span += np.maximum(0, np.minimum(last, hi) - np.maximum(first, lo))
    segments = np.ones(len(first), dtype=np.int64)
    segments[1:] = (copied[1:] != copied[:-1]) | (gap[1:] != gap[:-1]) | (first[1:] != last[:-1])
    unique, inverse = np.unique(copied, return_inverse=True)
    return pd.DataFrame({'parent': unique,
                         'sites': np.bincount(inverse, weights=sites),
                         'span_bp': np.bincount(inverse, weights=span),
                         'segments': np.bincount(inverse, weights=segments),
                         'distinct_parents': np.ones(len(unique), dtype=int)})


def copying_in_gaps(ts, reference, mask, positions, analysis_intervals):
    """Index sample edges once, then intersect each path with its own gap mask."""
    sample_nodes = ts.samples()
    np.testing.assert_array_equal(sample_nodes, np.arange(reference.num_nodes, ts.num_nodes))
    np.testing.assert_array_equal(ts.sites_position, positions)
    expected = list(zip(mask['source'], mask['sample_id'], mask['ploidy_index'], strict=True))
    observed = []
    for node_id in sample_nodes:
        metadata = ts.node(int(node_id)).metadata
        observed.append((metadata['source'], metadata['sample_id'], metadata['ploidy_index']))
    assert observed == expected
    selected = np.flatnonzero(ts.edges_child >= reference.num_nodes)
    assert np.all(ts.edges_parent[selected] < reference.num_nodes)
    order = np.lexsort((ts.edges_left[selected], ts.edges_child[selected]))
    selected = selected[order]
    children = ts.edges_child[selected]
    offsets = np.searchsorted(children, np.arange(reference.num_nodes, ts.num_nodes + 1))
    left, right, parents = ts.edges_left[selected], ts.edges_right[selected], ts.edges_parent[selected]
    rows = []
    for haplotype in range(len(sample_nodes)):
        begin, end = mask['offsets'][haplotype:haplotype + 2]
        gaps = mask['is_gap'][begin:end]
        if not gaps.any():
            continue
        start_site = mask['left_site_index'][begin:end][gaps]
        stop_site = mask['right_site_index'][begin:end][gaps]
        gap_left = np.zeros(len(start_site), dtype=float)
        gap_left[start_site > 0] = positions[start_site[start_site > 0]]
        gap_right = np.full(len(stop_site), reference.sequence_length, dtype=float)
        gap_right[stop_site < len(positions)] = positions[stop_site[stop_site < len(positions)]]
        first, last = offsets[haplotype:haplotype + 2]
        row = gap_fragments(left[first:last], right[first:last], parents[first:last],
                            gap_left, gap_right, positions, analysis_intervals)
        assert row.sites.sum() == np.sum(stop_site - start_site)
        row['haplotype_index'] = haplotype
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
    order = np.argsort(sites, kind='stable')
    tree = full.first()
    for index in order:
        site = sites[index]
        tree.seek(reference.sites_position[site])
        carried[index] = tree.is_descendant(int(sample_nodes[index]), int(parents[index]))
    terminal = full.mutations_node >= reference.num_nodes
    mutation_sites = full.mutations_site[terminal]
    mutation_samples = full.mutations_node[terminal] - reference.num_nodes
    keys = mutation_samples.astype(np.int64) * reference.num_sites + mutation_sites
    assert len(keys) == len(np.unique(keys))
    order = np.argsort(keys)
    keys = keys[order]
    full_states = tskit.unpack_strings(full.tables.mutations.derived_state,
                                       full.tables.mutations.derived_state_offset)
    full_states = np.asarray(full_states, dtype=str)[terminal][order]
    ref_states = tskit.unpack_strings(reference.tables.mutations.derived_state,
                                      reference.tables.mutations.derived_state_offset)
    ref_states = np.asarray(ref_states, dtype=str)
    query_keys = sample_indices.astype(np.int64) * reference.num_sites + sites
    indices = np.searchsorted(keys, query_keys)
    present = indices < len(keys)
    present[present] &= keys[indices[present]] == query_keys[present]
    carried[present] = full_states[indices[present]] == ref_states[sites[present]]
    return carried


def analyse_dataset(job: GapParentJob) -> pd.DataFrame:
    """Cache per-haplotype shares for Full HMM and stitched under identical masks."""
    paths = [job.root / job.dataset['reference'], job.root / job.dataset['full_hmm']]
    for run in job.dataset['runs']:
        paths.extend([job.root / run['stitched'], job.root / run['mask']])
    helpers = [focal_seed_sites, gap_fragments, copying_in_gaps, carried_focal_seeds]
    settings = {'focal_choice': job.choice, 'expected_eligible': job.expected_eligible,
                'helper_code': [inspect.getsource(function) for function in helpers]}
    stamp = ch5_perf_scaling.signature(paths, settings, analyse_dataset)
    output = job.cache / 'gap_parents' / f"{job.dataset['name']}.csv"
    cached = ch5_perf_scaling.read_cached(output, stamp)
    if cached is not None:
        return cached
    reference = tszip.decompress(paths[0])
    full = tszip.decompress(paths[1])
    seeds = focal_seed_sites(reference, job.choice)
    masks = {}
    tables = []
    for run in job.dataset['runs']:
        cutoff = run['af_cutoff']
        recent = reference.nodes_time <= cutoff
        assert np.all(seeds[recent] >= 0)
        assert int(recent.sum()) == job.expected_eligible[cutoff]
        masks[cutoff] = dict(np.load(job.root / run['mask']))
        for method in ['full_hmm', 'greedy']:
            matched = full if method == 'full_hmm' else tszip.decompress(job.root / run['stitched'])
            row = copying_in_gaps(matched, reference, masks[cutoff], reference.sites_position,
                                   job.dataset['analysis_intervals'])
            row['method'] = method
            row['af_cutoff'] = cutoff
            tables.append(row)
            if method == 'greedy':
                del matched
    copied = pd.concat(tables, ignore_index=True)
    times = reference.nodes_time[copied.parent.to_numpy()]
    recent = times <= copied.af_cutoff.to_numpy()
    keys = copied.haplotype_index.to_numpy(dtype=np.int64) * reference.num_nodes + copied.parent.to_numpy()
    unique, inverse = np.unique(keys[recent], return_inverse=True)
    sample_indices = unique // reference.num_nodes
    parents = unique % reference.num_nodes
    carried = carried_focal_seeds(full, reference, seeds, sample_indices, parents)
    categories = np.zeros(len(copied), dtype=np.int8)
    categories[recent] = np.where(carried[inverse], 2, 1)
    labels = np.array(['older', 'recent_nonfocal', 'focal'])
    copied['category'] = labels[categories]
    keys = ['method', 'af_cutoff', 'haplotype_index', 'category']
    weights = ['sites', 'span_bp', 'segments', 'distinct_parents']
    grouped = copied.groupby(keys)[weights].sum()
    index = pd.MultiIndex.from_product([
        ['full_hmm', 'greedy'], [run['af_cutoff'] for run in job.dataset['runs']],
        range(full.num_samples), labels], names=keys)
    grouped = grouped.reindex(index, fill_value=0).reset_index()
    for weight in weights:
        total = grouped.groupby(keys[:-1])[weight].transform('sum')
        grouped[f'{weight}_percentage'] = 100 * grouped[weight] / total.where(total > 0)
    grouped['dataset'] = job.dataset['name']
    grouped['focal_choice'] = job.choice
    ch5_perf_scaling.save_cached(grouped, output, stamp)
    return grouped


def gap_parent_composition(config: dict) -> pd.DataFrame:
    """Distribute independent cohort analyses and save their descriptive summaries."""
    name = config['experiment']['performance']['experiment_id']
    runs = pd.read_csv(config['data_dir'] / 'performance' / name / 'runs.csv')
    jobs = []
    for dataset in config['manifest']['datasets']:
        selected = runs.loc[runs.dataset.eq(dataset['name']) & runs.method.eq('greedy')]
        expected = dict(zip(selected.af_cutoff, selected.eligible_ancestors.astype(int), strict=True))
        jobs.append(GapParentJob(config['data_dir'], config['cache_dir'], dataset,
                                 config['experiment']['focal_choice'], expected))
    workers = config['workers']
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(jobs))
    rows = []
    context = multiprocessing.get_context('spawn')
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                initializer=numcodecs.blosc.set_nthreads,
                                initargs=(config['helper_threads'],)) as pool:
        futures = {pool.submit(analyse_dataset, job): job for job in jobs}
        for future in cf.as_completed(futures):
            rows.append(future.result())
            logger.info('Gap-parent composition: %s', futures[future].dataset['name'])
    table = pd.concat(rows, ignore_index=True)
    keys = ['dataset', 'method', 'af_cutoff', 'category']
    measurements = ['sites_percentage', 'span_bp_percentage', 'segments_percentage', 'distinct_parents_percentage']
    summary = table.groupby(keys)[measurements].agg(['mean', 'sem', 'count'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary = summary.reset_index()
    sizes = ch5_perf_scaling.cohort_table(config)[['dataset', 'num_samples']]
    summary = summary.merge(sizes, on='dataset', validate='many_to_one')
    summary.to_csv(config['cache_dir'] / 'gap_parent_composition.csv', index=False)
    return summary
