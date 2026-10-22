"""Measure direct ultimate-ancestor copying in archived sample-matching outputs.

Edges are counted once within the common observed analysis extent. Their
location is classified using the same focal mask for both matching methods;
no dating, genotype loading or new matching is required.
"""

import concurrent.futures as cf
import dataclasses
import inspect
import logging
import multiprocessing
import pathlib

import ch5_perf_scaling
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import pandas as pd
import tszip

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RootEdgeJob:
    root: pathlib.Path
    cache: pathlib.Path
    dataset: dict
    categories: list[str]


def classify_edges(left, right, mask, haplotype, positions, sequence_length, categories):
    """Assign whole observed edges to seams, gap interiors or focal coverage.

    An edge touches a seam if it has positive gap overlap and contains an
    internal focal/gap boundary, including either endpoint. Outer domain ends
    are not seams. Classification does not split edges or count them twice.
    """
    begin, end = mask['offsets'][haplotype:haplotype + 2]
    site_left = mask['left_site_index'][begin:end]
    site_right = mask['right_site_index'][begin:end]
    is_gap = mask['is_gap'][begin:end]
    boundaries = np.concatenate([positions, [sequence_length]])
    gap_left = boundaries[site_left[is_gap]].copy()
    gap_right = boundaries[site_right[is_gap]]
    gap_left[site_left[is_gap] == 0] = 0
    seam_sites = site_left[1:][is_gap[1:] != is_gap[:-1]]
    seams = positions[seam_sites]
    overlap = np.zeros(len(left), dtype=bool)
    for lo, hi in zip(gap_left, gap_right, strict=True):
        overlap |= (left < hi) & (right > lo)
    first_seam = np.searchsorted(seams, left, side='left')
    last_seam = np.searchsorted(seams, right, side='right')
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
    expected = list(zip(mask['source'], mask['sample_id'], mask['ploidy_index'], strict=True))
    observed = []
    for node in sample_nodes:
        metadata = ts.node(int(node)).metadata
        observed.append((metadata['source'], metadata['sample_id'], metadata['ploidy_index']))
    assert observed == expected
    np.testing.assert_array_equal(ts.sites_position, reference.sites_position)
    selected = np.flatnonzero((ts.edges_parent == root) & np.isin(ts.edges_child, sample_nodes))
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
    table = pd.DataFrame({
        'edge_id': selected[present],
        'haplotype_index': ts.edges_child[selected[present]] - reference.num_nodes,
        'left_bp': clipped_left[present],
        'right_bp': clipped_right[present],
        'span_bp': span[present],
    })
    table['category'] = ''
    for haplotype, indices in table.groupby('haplotype_index').groups.items():
        group = table.loc[indices]
        labels = classify_edges(group.left_bp.to_numpy(), group.right_bp.to_numpy(), mask,
                                haplotype, reference.sites_position, reference.sequence_length,
                                categories)
        table.loc[indices, 'category'] = labels
    assert table.category.isin(categories).all()
    return table


def analyse_dataset(job: RootEdgeJob) -> pd.DataFrame:
    """Cache whole-edge measurements for each parameter's paired focal mask."""
    paths = [job.root / job.dataset['reference'], job.root / job.dataset['full_hmm']]
    for run in job.dataset['runs']:
        paths.extend([job.root / run['stitched'], job.root / run['mask']])
    settings = {'categories': job.categories, 'intervals': job.dataset['analysis_intervals'],
                'helper_code': [inspect.getsource(classify_edges), inspect.getsource(observed_root_edges)]}
    stamp = ch5_perf_scaling.signature(paths, settings, analyse_dataset)
    output = job.cache / 'root_edges' / f"{job.dataset['name']}.csv"
    cached = ch5_perf_scaling.read_cached(output, stamp)
    if cached is not None:
        return cached
    reference = tszip.decompress(paths[0])
    stem = reference.edge(-1)
    assert reference.edge(-2).parent != stem.parent
    root = stem.child
    full = tszip.decompress(paths[1])
    rows = []
    for run in job.dataset['runs']:
        with np.load(job.root / run['mask']) as archive:
            mask = dict(archive)
        for method in ['full_hmm', 'greedy']:
            matched = full if method == 'full_hmm' else tszip.decompress(job.root / run['stitched'])
            assert matched.edge(-1).parent == stem.parent
            assert matched.edge(-1).child == root
            row = observed_root_edges(matched, reference, root, mask,
                                      job.dataset['analysis_intervals'], job.categories)
            row['method'] = method
            row['af_cutoff'] = run['af_cutoff']
            row['dataset'] = job.dataset['name']
            rows.append(row)
            if method == 'greedy':
                del matched
    table = pd.concat(rows, ignore_index=True)
    full_totals = table.loc[table.method.eq('full_hmm')].groupby('af_cutoff').span_bp.agg(['size', 'sum'])
    assert full_totals['size'].nunique() == 1
    np.testing.assert_allclose(full_totals['sum'], full_totals['sum'].iloc[0])
    ch5_perf_scaling.save_cached(table, output, stamp)
    return table


def root_edge_summary(config: dict) -> pd.DataFrame:
    """Distribute cohort analyses, retaining zero categories and whole-edge counts."""
    categories = list(config['plot']['root_edge_categories'])
    jobs = [RootEdgeJob(config['data_dir'], config['cache_dir'], dataset, categories)
            for dataset in config['manifest']['datasets']]
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
            logger.info('Root/sample edges: %s', futures[future].dataset['name'])
    edges = pd.concat(rows, ignore_index=True)
    keys = ['dataset', 'af_cutoff', 'method', 'category']
    summary = edges.groupby(keys).agg(edge_count=('span_bp', 'size'), span_bp=('span_bp', 'sum'))
    sizes = ch5_perf_scaling.cohort_table(config)
    index = pd.MultiIndex.from_product([sizes.dataset, config['experiment']['af_cutoff'],
                                       ['full_hmm', 'greedy'], categories], names=keys)
    summary = summary.reindex(index, fill_value=0).reset_index()
    summary = summary.merge(sizes[['dataset', 'num_samples', 'num_haplotypes']],
                            on='dataset', validate='many_to_one')
    summary['edge_count_per_haplotype'] = summary.edge_count / summary.num_haplotypes
    summary['span_kb_per_haplotype'] = summary.span_bp / summary.num_haplotypes / 1000
    summary.to_csv(config['cache_dir'] / 'root_edge_summary.csv', index=False)
    return summary


def totals(plot, summary, metric, ylabel):
    """Stack the same location classes for Full HMM and stitched whole edges."""
    figure, axes = plot.panels(rows=len(plot.cutoffs), columns=len(plot.datasets),
                               title='Direct ultimate-root → sample copying · observed extent')
    for row, cutoff in enumerate(plot.cutoffs):
        for col, dataset in enumerate(plot.datasets):
            ax = axes[row, col]
            values = summary.loc[summary.dataset.eq(dataset['name']) & summary.af_cutoff.eq(cutoff)]
            bottom = np.zeros(2)
            for category, style in plot.style['root_edge_categories'].items():
                selected = values.loc[values.category.eq(category)].set_index('method')
                heights = selected.loc[['full_hmm', 'greedy'], metric].to_numpy()
                ax.bar([0, 1], heights, bottom=bottom, color=style['color'],
                       width=plot.style['bar_width'], label=style['label'])
                bottom += heights
            ax.set(title=f'n = {plot.sample_sizes[col]:,} · {plot.cutoff_label(cutoff, plot.sample_sizes[col])}',
                   xticks=[0, 1], xticklabels=['Full HMM', 'Stitched'], ylabel=ylabel, ylim=(0, None))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize='small')
    return figure


def additional(plot, summary):
    """Show category-wise signed differences; their sum is the net excess."""
    metrics = {'edge_count_per_haplotype': 'Additional root edges per haplotype',
               'span_kb_per_haplotype': 'Additional root-copying span (kb per haplotype)'}
    figure, axes = plot.panels(rows=len(plot.cutoffs), columns=len(metrics),
                               title='Stitched − Full HMM · signed root-copying differences')
    x = np.arange(len(plot.sample_sizes))
    for row, cutoff in enumerate(plot.cutoffs):
        source = summary.loc[summary.af_cutoff.eq(cutoff)]
        for col, (metric, ylabel) in enumerate(metrics.items()):
            ax = axes[row, col]
            positive = np.zeros(len(x))
            negative = np.zeros(len(x))
            net = np.zeros(len(x))
            for category, style in plot.style['root_edge_categories'].items():
                selected = source.loc[source.category.eq(category)]
                pivot = selected.pivot(index='num_samples', columns='method', values=metric)
                heights = (pivot.greedy - pivot.full_hmm).loc[plot.sample_sizes].to_numpy()
                bottom = np.where(heights >= 0, positive, negative)
                ax.bar(x, heights, bottom=bottom, color=style['color'],
                       width=plot.style['bar_width'], label=style['label'])
                positive += np.maximum(heights, 0)
                negative += np.minimum(heights, 0)
                net += heights
            ax.scatter(x, net, color=plot.style['full_color'], marker='D',
                       s=plot.style['marker_size'] ** 2, zorder=3, label='Net difference')
            ax.axhline(0, color=plot.style['full_color'], linewidth=1)
            ax.set(title=plot.cutoff_label(cutoff), xticks=x,
                   xticklabels=[f'{n:,}' for n in plot.sample_sizes],
                   xlabel='Samples (diploid individuals)', ylabel=ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize='small')
    return figure


def span_histograms(plot):
    """Use log-spaced bins over the complete observed span range, including tails."""
    frames = []
    for dataset in plot.datasets:
        path = plot.config['cache_dir'] / 'root_edges' / f"{dataset['name']}.csv"
        frames.append(pd.read_csv(path))
    edges = pd.concat(frames, ignore_index=True)
    minimum = edges.span_bp.min()
    maximum = edges.span_bp.max()
    bins = np.geomspace(minimum, maximum, plot.style['root_edge_histogram_bins'] + 1)
    figure, axes = plot.panels(rows=len(plot.cutoffs), columns=len(plot.datasets),
                               title='Root → sample edge spans · Full HMM and stitched locations')
    for row, cutoff in enumerate(plot.cutoffs):
        for col, dataset in enumerate(plot.datasets):
            ax = axes[row, col]
            selected = edges.loc[edges.dataset.eq(dataset['name']) & edges.af_cutoff.eq(cutoff)]
            samples = 2 * plot.sample_sizes[col]
            baseline = selected.loc[selected.method.eq('full_hmm'), 'span_bp']
            grouped = []
            colors = []
            labels = []
            for category, style in plot.style['root_edge_categories'].items():
                values = selected.loc[selected.method.eq('greedy') & selected.category.eq(category), 'span_bp']
                grouped.append(values.to_numpy())
                colors.append(style['color'])
                labels.append('Stitched: ' + style['label'])
            weights = [np.full(len(values), 1 / samples) for values in grouped]
            ax.hist(grouped, bins=bins, weights=weights, stacked=True, color=colors,
                    alpha=plot.style['root_edge_histogram_alpha'], label=labels)
            ax.hist(baseline, bins=bins, weights=np.full(len(baseline), 1 / samples),
                    histtype='step', color=plot.style['full_color'],
                    linewidth=plot.style['line_width'], label='Full HMM total')
            ax.set(xscale='log', yscale='log', xlabel='Observed edge span (bp)',
                   ylabel='Edges per haplotype per bin',
                   title=f'n = {plot.sample_sizes[col]:,} · {plot.cutoff_label(cutoff, plot.sample_sizes[col])}')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=2, fontsize='small')
    return figure
