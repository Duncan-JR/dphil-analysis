# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stitching parameter search: AF and minimum focal-region size
#
# Analyse `hpc_params1_diagnostics` without rerunning matching. The archived grid
# has **three AF cutoffs (0.01, 0.03, 0.05)**, four minimum sizes (1, 100, 300,
# 1000 inference sites), and three simulation cohorts (300, 3000, 10000 diploid
# individuals). Every plot has one column per sample size. AF-panel plots have
# one row per AF; metric-panel plots have one row per metric/reference.
#
# This adapts pt6's haplotype means, hourglass SEMs, population groups, switch
# partitions, sample alignment, ARG metrics and posterior mutation-age rules.
# Derived AF replaces AC throughout. The archived config/manifest, rather than
# the current mutable pipeline config, define this analysis. Generated tables,
# dates and PNGs are cached in `data/ch5/hpc_params1_diagnostics/for_plots`.

# %%
import dataclasses
import logging
import pathlib

import ch5_dating
import ch5_hyperparams
import ch5_paramsearch
import IPython.display as ipython_display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import tskit
import tszip

logger = logging.getLogger(__name__)

# %%
notebook_dir = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path.cwd()
repository_dir = notebook_dir
if not (repository_dir / 'experiments').is_dir():
    repository_dir = notebook_dir.parent.parent
config_path = repository_dir / 'experiments/ch5/stitching_pt7_paramsearch_config.yaml'
config = ch5_paramsearch.load_config(config_path)
manifest = config['manifest']
datasets = manifest['datasets']
experiment = config['experiment']
af_cutoffs = sorted(experiment['af_cutoff'])
minimum_sizes = sorted(experiment['min_focal_region_size'])
style = config['plot']
output_dir = config['cache_dir']
cutoff_colors = dict(zip(af_cutoffs, style['cutoff_colors'], strict=True))
minimum_colors = dict(zip(minimum_sizes, style['minimum_colors'], strict=True))
plt.rcParams.update({'figure.dpi': style['dpi'], 'font.size': style['font_size']})
logging.basicConfig(level=logging.INFO, format='%(message)s')

# %% [markdown]
# ## Pipeline changes and archived run checks
#
# Recent `tsinfer-match-eval` changes replace the old per-seed/mismatch logic
# with exact seeded extension, then a greedy minimum-run cover of touching
# interval unions. **Minimum size filters whole connected unions**, not the
# individual copying runs inside them. Ancestors qualify when cohort-derived
# seed frequency is **≤ af_cutoff**. Each complementary gap gets a fresh HMM
# likelihood calculation. Reference topology is retained between gap calls;
# this supplies no likelihood boundary condition. Seams can therefore add
# switches even when no mismatches are permitted in focal extension.
#
# The diagnostics target now reconstructs masks and exports per-haplotype
# comparisons, interval lengths, population joins and portable tree paths.
# Parallelism/prepared paths and bulk edge insertion change the execution.
# The latest revision also changes assembly and retains reference topology.
# Diagnostic-time code revisions are printed below; the manifest explicitly
# cautions that these do not establish which code produced older saved trees.
# No performance benchmark accompanies this archive, so focal coverage is only
# a proxy for HMM work avoided; this analysis cannot establish runtime speedup.

# %%
summary = pd.read_csv(config['data_dir'] / manifest['tables']['run_summary'])
populations = pd.read_csv(config['data_dir'] / manifest['tables']['sample_populations'])
grid_keys = ['dataset', 'af_cutoff', 'min_focal_region_size']
expected = pd.MultiIndex.from_product(
    [[item['name'] for item in datasets], af_cutoffs, minimum_sizes], names=grid_keys)
assert not summary.duplicated(grid_keys).any()
assert summary.set_index(grid_keys).index.sort_values().equals(expected.sort_values())
for dataset in datasets:
    rows = summary.loc[summary.dataset.eq(dataset['name'])]
    cohort = populations.loc[populations.dataset.eq(dataset['name'])]
    assert rows.num_haplotypes.eq(len(cohort)).all()
    assert rows.num_samples.eq(cohort.sample_id.nunique()).all()
    assert rows.unmatched_sites_mean.eq(0).all()
    assert rows.excluded_sites_mean.eq(0).all()
    for method in ['full', 'stitched']:
        category_columns = [f'{method}_{category}_switches_mean' for category in style['switch_categories']]
        np.testing.assert_allclose(rows[category_columns].sum(axis=1), rows[f'{method}_switches_mean'])
revisions = manifest['diagnostics_time_versions']['revisions']
ipython_display.display(pd.DataFrame([
    {'component': name, 'diagnostic_time_commit': item['commit'], 'status': item['status']}
    for name, item in revisions.items()]))
check_columns = grid_keys + ['num_haplotypes', 'num_samples', 'empty_focal_haplotypes',
                            'empty_gap_haplotypes', 'zero_full_switch_haplotypes']
ipython_display.display(summary[check_columns])

# %%
@dataclasses.dataclass
class Panels:
    """Figure and row-by-sample-size axes, following pt6."""

    figure: plt.Figure
    axes: np.ndarray


def cutoff_label(cutoff, dataset):
    """Show the AF cutoff and its rounded AC in the actual haplotype cohort."""
    num_haplotypes = len(dataset['cohort'])
    ac_cutoff = round(cutoff * num_haplotypes)
    return f'AF ≤ {cutoff:g} (AC {ac_cutoff:,})'


def panel_grid(title, row_labels, sharey='row', by_cutoff=False):
    """Always keep sample sizes in columns and the chosen quantity in rows."""
    fig, axes = plt.subplots(
        len(row_labels), len(datasets), squeeze=False, sharey=sharey, layout='constrained',
        figsize=(style['panel_width'] * len(datasets), style['panel_height'] * len(row_labels)))
    for row, label in enumerate(row_labels):
        for col, dataset in enumerate(datasets):
            cohort = populations.loc[populations.dataset.eq(dataset['name'])]
            n = cohort.sample_id.nunique()
            row_label = cutoff_label(label, dataset) if by_cutoff else label
            axes[row, col].set_title(f'{n:,} individuals\n{row_label}')
            axes[row, col].tick_params(axis='y', which='both', left=True, labelleft=True)
    fig.suptitle(title)
    return Panels(fig, axes)


def zoom_row(axes):
    """Zoom to all displayed curves/SEMs across AFs and sample sizes, with margin."""
    limits = np.array([ax.dataLim.intervaly for ax in axes])
    finite_limits = limits[np.isfinite(limits).all(axis=1)]
    if len(finite_limits) == 0:
        return
    lower = finite_limits[:, 0].min()
    upper = finite_limits[:, 1].max()
    margin = (upper - lower) * style['range_margin']
    if margin == 0:
        margin = max(abs(lower), 1) * style['range_margin']
    for ax in axes:
        ax.set_ylim(lower - margin, upper + margin)


def minimum_axis(ax, baseline=False):
    """Use equally spaced labelled sizes, keeping Full HMM distinct from min 1."""
    labels = [str(value) for value in minimum_sizes]
    if baseline:
        labels = ['Full HMM', *labels]
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
           xlabel='Minimum focal size (inference sites; equally spaced)')


def hourglass_line(ax, table, metric, color, label):
    """Reuse pt6's mean ± one haplotype SEM glyphs with categorical positions."""
    table = table.sort_values('min_focal_region_size')
    means = table[f'{metric}_mean'].to_numpy()
    sems = table[f'{metric}_sem'].to_numpy()
    x = np.arange(len(table))
    ax.plot(x, means, color=color, label=label, marker='o',
            markersize=style['marker_size'], linewidth=style['line_width'])
    for position, mean, sem in zip(x, means, sems, strict=True):
        width = style['hourglass_width']
        vertices = [(position - width, mean - sem), (position + width, mean - sem),
                    (position, mean), (position + width, mean + sem),
                    (position - width, mean + sem), (position, mean)]
        ax.add_patch(mpatches.Polygon(vertices, facecolor=color, edgecolor=color,
                                      alpha=style['hourglass_alpha']))
    minimum_axis(ax)


def metric_lines(table, metrics, title, summary_statistics=False, baseline=False):
    """Metric rows and sample-size columns; one line per AF cutoff."""
    panels = panel_grid(title, list(metrics.values()))
    for row, (metric, ylabel) in enumerate(metrics.items()):
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            selected = table.loc[table.dataset.eq(dataset['name'])]
            full = selected.loc[selected.comparison_method.eq('full_hmm')] if baseline else None
            has_curves = False
            for cutoff in af_cutoffs:
                curve = selected.loc[selected.af_cutoff.eq(cutoff)].sort_values('min_focal_region_size')
                if curve.empty:
                    continue
                has_curves = True
                color = cutoff_colors[cutoff]
                label = cutoff_label(cutoff, dataset)
                if summary_statistics:
                    hourglass_line(ax, curve, metric, color, label)
                else:
                    curve = curve.set_index('min_focal_region_size').reindex(minimum_sizes)
                    y = curve[metric].to_numpy()
                    if baseline:
                        full_value = full[metric].item() if not full.empty else np.nan
                        y = np.concatenate(([full_value], y))
                    ax.plot(np.arange(len(y)), y, color=color, label=label, marker='o',
                            markersize=style['marker_size'], linewidth=style['line_width'])
            if baseline and not has_curves and not full.empty:
                ax.plot(0, full[metric].item(), marker='o', color='gray', linestyle='none', label='Full HMM')
            minimum_axis(ax, baseline)
            ax.set_ylabel(ylabel)
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend()
        zoom_row(panels.axes[row])
    return panels.figure


def save_figure(fig, name):
    """Keep exported figures with plot caches and embed each in the notebook."""
    fig.savefig(output_dir / f'{name}.png')
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## Site-based minimum sizes in kb
#
# Bars show the genome-wide mean physical equivalent of each tested site-count
# threshold, using the same density conversion as `min_region_size_kb`:
# `minimum_sites × (last_position − first_position) / num_sites / 1000`.
# These are mean equivalents, not the distribution of local window lengths:
# local site density varies. AF does not affect this conversion.

# %%
def minimum_size_kb_plot():
    rows = []
    for dataset in datasets:
        with np.load(config['data_dir'] / dataset['sites']) as site_data:
            positions = site_data['positions']
        span_kb = (positions[-1] - positions[0]) / 1000
        mean_kb_per_site = span_kb / len(positions)
        for minimum in minimum_sizes:
            rows.append({
                'dataset': dataset['name'],
                'num_sites': len(positions),
                'site_span_kb': span_kb,
                'min_focal_region_size': minimum,
                'mean_min_region_size_kb': minimum * mean_kb_per_site,
            })
    equivalents = pd.DataFrame(rows)
    equivalents.to_csv(output_dir / 'minimum_size_kb.csv', index=False)
    panels = panel_grid('Mean physical equivalent of site-based minimum sizes', [''])
    for col, dataset in enumerate(datasets):
        ax = panels.axes[0, col]
        table = equivalents.loc[equivalents.dataset.eq(dataset['name'])]
        values = table['mean_min_region_size_kb'].to_numpy()
        bars = ax.bar(np.arange(len(minimum_sizes)), values,
                      color=[minimum_colors[minimum] for minimum in minimum_sizes],
                      width=style['bar_width'])
        ax.bar_label(bars, labels=[f'{value:.2f}' for value in values], padding=3)
        minimum_axis(ax)
        ax.set_ylabel('Mean minimum region size (kb)')
        ax.margins(y=style['range_margin'] * 2)
    return panels.figure


if __name__ == '__main__':
    save_figure(minimum_size_kb_plot(), 'minimum_size_kb')

# %% [markdown]
# ## Extra switches vs minimum focal size
#
# Percent changes are calculated per haplotype then averaged, rather than from
# the ratio of pooled switch counts. Hourglasses show mean ± one SEM. These
# describe haplotype variation within a single simulation, not independent
# simulation replicates; the paired haplotypes' dependence is not accounted for.

# %%
if __name__ == '__main__':
    fig = metric_lines(summary, {'switch_increase_percentage': 'Extra switches vs Full HMM (%)'},
                       'Extra switches vs minimum focal size', summary_statistics=True)
    save_figure(fig, 'extra_switches')

# %% [markdown]
# ## Focal region distributions
#
# Coverage uses the diagnostic physical analysis extent (first to last observed
# inference position, approximately 0.081–10 Mb), not the 64 Mb chromosome
# length retained in raw trees. African = YRI; Non-African pools CEU and CHB.
# Coverage axes zoom to the displayed population means plus a margin.
#
# Lengths count inference sites in maximal connected focal components. Only
# **minimum size 1** is shown; dashed vertical lines locate every tested size.
# The histogram has a directly editable `num_bins` argument below. Densities
# reproduce pt6's reflected Scott KDE on all intervals, including the tails,
# and are scaled to count per bin. Each AF row shares bins across sample sizes.
# The display ends at the pooled 99th percentile, extended to include all size
# thresholds; legends report omitted tail counts. Lossless integer-length counts
# are cached so changing bins does not require rereading the 4 GB source CSV.
# Count axes scale independently because the cohorts differ greatly in size.

# %%
if __name__ == '__main__':
    population_coverage = ch5_paramsearch.population_summary(config)
    lengths = ch5_paramsearch.length_counts(config)

# %%
def focal_coverage():
    panels = panel_grid('Focal region coverage', af_cutoffs, by_cutoff=True)
    for row, cutoff in enumerate(af_cutoffs):
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            selected = population_coverage.dataset.eq(dataset['name']) & population_coverage.af_cutoff.eq(cutoff)
            table = population_coverage.loc[selected]
            for group, color in style['population_colors'].items():
                curve = table.loc[table.population_group.eq(group)].sort_values('min_focal_region_size')
                ax.plot(np.arange(len(curve)), curve['mean'], color=color, label=group, marker='o')
            minimum_axis(ax)
            ax.set_ylabel('Physical focal coverage (%)')
            ax.legend()
        zoom_row(panels.axes[row])
    return panels.figure


def focal_lengths(num_bins):
    panels = panel_grid('Focal region lengths · minimum size 1', af_cutoffs, sharey=False, by_cutoff=True)
    for row, cutoff in enumerate(af_cutoffs):
        selected = lengths.af_cutoff.eq(cutoff) & ~lengths.is_gap
        selected &= lengths.min_focal_region_size.eq(style['baseline_minimum'])
        table = lengths.loc[selected]
        quantile = ch5_paramsearch.weighted_quantile(table, style['length_display_quantile'])
        upper = max(quantile, max(minimum_sizes)) + 1
        bins = np.linspace(0, upper, num_bins + 1)
        grid = np.linspace(0, upper, style['kde_grid_size'])
        bin_width = bins[1] - bins[0]
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            values = table.loc[table.dataset.eq(dataset['name'])]
            total = values['count'].sum()
            tail = values.loc[values.num_sites.gt(upper), 'count'].sum()
            color = cutoff_colors[cutoff]
            ax.hist(values.num_sites, bins=bins, weights=values['count'], color=color,
                    alpha=style['histogram_alpha'], label=f'{total:,} regions; tail {tail:,}')
            density = ch5_paramsearch.length_density(values, grid)
            ax.plot(grid, density * total * bin_width, color=color)
            for minimum, marker_color in minimum_colors.items():
                ax.axvline(minimum, color=marker_color, linestyle='--', label=f'Min {minimum}')
            ax.set(xlabel='Focal length (inference sites)', ylabel='Count per bin', xlim=(0, upper), ylim=(0, None))
            ax.legend(fontsize='small')
    return panels.figure


if __name__ == '__main__':
    save_figure(focal_coverage(), 'focal_coverage')
    save_figure(focal_lengths(num_bins=120), 'focal_lengths')

# %% [markdown]
# ## Gap distributions
#
# Density curves only, coloured by minimum region size. Matching dashed lines
# show each distribution's median. KDEs use all gap lengths with reflection at
# zero. Each row shares the pooled 99th-percentile display range and reports
# tail fractions. The bar chart below separately counts one-site gaps.

# %%
def gap_lengths():
    panels = panel_grid('Gap length distributions', af_cutoffs, by_cutoff=True)
    for row, cutoff in enumerate(af_cutoffs):
        table = lengths.loc[lengths.af_cutoff.eq(cutoff) & lengths.is_gap]
        upper = ch5_paramsearch.weighted_quantile(table, style['length_display_quantile']) + 1
        grid = np.linspace(0, upper, style['kde_grid_size'])
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            for minimum, color in minimum_colors.items():
                selected = table.dataset.eq(dataset['name']) & table.min_focal_region_size.eq(minimum)
                values = table.loc[selected]
                median = ch5_paramsearch.weighted_quantile(values, 0.5)
                density = ch5_paramsearch.length_density(values, grid)
                total = values['count'].sum()
                tail = values.loc[values.num_sites.gt(upper), 'count'].sum() / total
                ax.plot(grid, density, color=color,
                        label=f'Min {minimum}; median {median:g}; tail {tail:.1%}')
                ax.axvline(median, color=color, linestyle='--', linewidth=style['line_width'])
            ax.set(xlabel='Gap length (inference sites)', ylabel='Density per inference site', xlim=(0, upper), ylim=(0, None))
            ax.legend(fontsize='small')
        density_upper = max(ax.dataLim.ymax for ax in panels.axes[row])
        for ax in panels.axes[row]:
            ax.set_ylim(0, density_upper * (1 + style['range_margin']))
    return panels.figure


if __name__ == '__main__':
    save_figure(gap_lengths(), 'gap_lengths')

# %% [markdown]
# ### One-site gaps
#
# Bars count gap regions containing exactly one inference site, pooled across
# all haplotypes. Labels give their percentage of **all gap regions** for that
# dataset/AF/minimum combination, including the full length-distribution tails.
# Each AF has a row; sample sizes have columns; bars span all tested minima.

# %%
def one_site_gaps():
    gaps = lengths.loc[lengths.is_gap]
    totals = gaps.groupby(grid_keys)['count'].sum().rename('total_gaps')
    singles = gaps.loc[gaps.num_sites.eq(1)].groupby(grid_keys)['count'].sum()
    singles = singles.reindex(totals.index, fill_value=0).rename('one_site_gaps')
    counts = pd.concat([totals, singles], axis=1).reset_index()
    counts['one_site_percentage'] = 100 * counts.one_site_gaps / counts.total_gaps
    counts.to_csv(output_dir / 'one_site_gap_counts.csv', index=False)
    panels = panel_grid('Number of one-site gaps · labels show percentage of all gaps',
                        af_cutoffs, by_cutoff=True)
    for row, cutoff in enumerate(af_cutoffs):
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            selected = counts.dataset.eq(dataset['name']) & counts.af_cutoff.eq(cutoff)
            table = counts.loc[selected].set_index('min_focal_region_size').loc[minimum_sizes]
            bars = ax.bar(np.arange(len(minimum_sizes)), table.one_site_gaps,
                          color=[minimum_colors[minimum] for minimum in minimum_sizes],
                          width=style['bar_width'])
            ax.bar_label(bars, labels=[f'{value:.1f}%' for value in table.one_site_percentage], padding=3)
            minimum_axis(ax)
            ax.set_ylabel('Number of one-site gaps')
            ax.ticklabel_format(axis='y', style='plain')
            ax.margins(y=style['range_margin'] * 2)
    return panels.figure


if __name__ == '__main__':
    save_figure(one_site_gaps(), 'one_site_gaps')

# %% [markdown]
# ## Switch composition
#
# Raw mean switches partition into within-focal, within-gap and seam switches.
# Full HMM has one baseline bar partitioned using minimum size 1 for that AF.
# Each stitched bar uses its own mask. The Full HMM total is invariant under
# these partitions; raising the minimum changes its categories, not its path.

# %%
def switch_composition():
    panels = panel_grid('Switch composition', af_cutoffs, by_cutoff=True)
    x = np.concatenate(([0], np.arange(len(minimum_sizes)) + 2 * style['bar_width']))
    for row, cutoff in enumerate(af_cutoffs):
        for col, dataset in enumerate(datasets):
            ax = panels.axes[row, col]
            selected = summary.dataset.eq(dataset['name']) & summary.af_cutoff.eq(cutoff)
            table = summary.loc[selected].sort_values('min_focal_region_size')
            baseline = table.loc[table.min_focal_region_size.eq(style['baseline_minimum'])]
            bottom = np.zeros(len(x))
            for category, color, label in zip(style['switch_categories'], style['switch_colors'], style['switch_labels'], strict=True):
                full = baseline[f'full_{category}_switches_mean'].item()
                heights = np.concatenate(([full], table[f'stitched_{category}_switches_mean'].to_numpy()))
                ax.bar(x, heights, bottom=bottom, width=style['bar_width'], color=color, label=label)
                bottom += heights
            ax.set(xticks=x, xticklabels=['Full HMM', *[str(size) for size in minimum_sizes]],
                   xlabel='Minimum focal size (inference sites)', ylabel='Mean switches per haplotype')
    handles, labels = panels.axes[0, 0].get_legend_handles_labels()
    panels.figure.legend(handles, labels, loc='outside lower center', ncol=len(labels))
    return panels.figure


if __name__ == '__main__':
    save_figure(switch_composition(), 'switch_composition')

# %% [markdown]
# ## Parent agreement
#
# Focal and gap regions occupy separate rows, keeping sample sizes in columns.
# Each region's y range covers the observed mean ± SEM across **all AF values
# and sample sizes**, plus a margin. The denominator is the evaluated inference
# sites within that region, followed by an equal-weight haplotype mean.

# %%
if __name__ == '__main__':
    metrics = {'focal_agreement_percentage': 'Parent agreement: focal regions (%)',
               'gap_agreement_percentage': 'Parent agreement: gaps (%)'}
    save_figure(metric_lines(summary, metrics, 'Parent agreement', summary_statistics=True), 'parent_agreement')

# %% [markdown]
# ## Tree sequence comparisons
#
# Percentage change = 100 × (stitched / Full HMM − 1), using raw saved edge/tree
# counts including retained reference ancestry and chromosome extent. Edge and
# tree metrics occupy rows to preserve one column per sample size.
#
# **Load completed comparisons only. Do not calculate missing metrics.** Large
# tree comparisons were stopped at the user's request. Missing runs are left
# blank, with NaNs preventing lines from connecting across missing settings.
# The coverage table below records what was completed. Existing comparisons
# align biological sample identities and restrict both trees to the same
# diagnostic analysis window.
# For practical runtime at 20,000 haplotypes, `ch5_arg_metrics` substitutes
# exact integer bitsets for tscompare's frozenset clade iterator inside each
# worker. The tscompare span matching and ARF/TPR formulas are unchanged;
# equality is tested against its reference implementation, including unary
# ancestry, missing spans and sample sets larger than 64 haplotypes.

# %%
if __name__ == '__main__':
    completed_metrics = []
    metric_inventory = []
    for job in ch5_paramsearch.tree_jobs(config):
        path = output_dir / 'tree_metrics' / f'{job.key}.csv'
        signature = ch5_paramsearch.input_signature(
            [job.raw_path, job.root / job.dataset['truth'], job.root / job.dataset['full_hmm']], {})
        cached = ch5_paramsearch.cached_table(path, signature)
        metric_inventory.append({**job.labels, 'completed': cached is not None})
        if cached is not None:
            completed_metrics.append(cached)
    tree_metrics = pd.concat(completed_metrics, ignore_index=True)
    tree_metrics.to_csv(output_dir / 'available_tree_metrics.csv', index=False)
    metric_inventory = pd.DataFrame(metric_inventory)
    metric_inventory.to_csv(output_dir / 'tree_metric_inventory.csv', index=False)
    ipython_display.display(metric_inventory.groupby('dataset').completed.agg(['sum', 'count']))
    metrics = {'edges_change_percentage': 'Edges: change from Full HMM (%)',
               'trees_change_percentage': 'Trees: change from Full HMM (%)'}
    save_figure(metric_lines(tree_metrics, metrics, 'Raw tree sequence comparisons'), 'tree_counts')

# %% [markdown]
# ## ARG metrics vs. Full HMM
#
# `tscompare.haplotype_arf(stitched, full)` measures unsupported stitched node
# span (ARF, lower is better) and recovered Full HMM node span (TPR, higher is
# better). Sample IDs and ploidy indices determine sample order; simplification
# retains unary nodes. Full HMM agreement is distinct from biological accuracy.

# %%
if __name__ == '__main__':
    metrics = {'full_arf': 'ARF against Full HMM', 'full_tpr': 'TPR against Full HMM'}
    save_figure(metric_lines(tree_metrics, metrics, 'ARG metrics vs Full HMM'), 'arg_vs_full')

# %% [markdown]
# ## ARG metrics vs true TS
#
# Same alignment and window rules as above, with truth's VCF sample identity
# mapping. Each panel includes the Full HMM baseline as a distinct category.
# ARF measures inferred span unsupported by truth; TPR measures true span
# recovered. Baseline repetition connects each AF curve to the same Full HMM.
# As in pt6, these are raw inferred trees: their node-time units are unknown,
# whereas truth uses generations. ARF uses maximum matching span regardless
# of time. tscompare uses node times to break equal-span clade-match ties, so
# the inverse matching used by TPR also depends on that tie-breaking rule.

# %%
if __name__ == '__main__':
    metrics = {'truth_arf': 'ARF against truth', 'truth_tpr': 'TPR against truth'}
    save_figure(metric_lines(tree_metrics, metrics, 'ARG metrics vs true TS', baseline=True), 'arg_vs_truth')

# %% [markdown]
# ## Mutation age correlations
#
# Date the archived inferred trees **without restoring variants**. As in pt6,
# post-process, restrict the window, split disjoint nodes, and use tsdate's
# variational-gamma estimates with mutation rate 2.35 × 10⁻⁸ per bp/generation
# (the simulation provenance). Dates are cached per input/settings/code signature.
#
# Compare posterior **mutation means** (`metadata['mn']`) with simulated event
# times or Full HMM posterior means. For each sample size, take **one common
# intersection across truth, Full HMM and all twelve stitched runs**, using
# position plus ancestral/derived states. Exclude zero/multiple-mutation sites
# and nonfinite ages. Both reference panels use exactly this same set. Pearson r
# is calculated on ages in generations, before any transformation. Counts below
# show how many mutations survive. No heatmap plots are produced.

# %%
if __name__ == '__main__':
    dated_runs = ch5_paramsearch.run_tree_jobs(config, 'dating')
    age_correlations = ch5_paramsearch.mutation_correlations(config, dated_runs)
    counts = age_correlations.groupby(['dataset', 'reference_method', 'frequency_bin']).matched_mutations.agg(['min', 'max'])
    assert counts['min'].eq(counts['max']).all()
    ipython_display.display(counts)

# %%
def mutation_age_plots(table, groups, statistic='pearson_r'):
    """Reference/bin rows and sample-size columns with consistent AF colours."""
    references = {'true_arg': 'Inferred vs true ages', 'full_hmm': 'Stitched vs Full HMM ages'}
    row_labels = [f'{label} · {group}' for group in groups for label in references.values()]
    statistic_label = {'pearson_r': 'Pearson r', 'spearman_rho': 'Spearman ρ'}[statistic]
    panels = panel_grid(f'Mutation-age correlations · {statistic_label}', row_labels)
    for group_index, group in enumerate(groups):
        for reference_index, (reference, label) in enumerate(references.items()):
            row = 2 * group_index + reference_index
            table_group = table.loc[table.frequency_bin.eq(group) & table.reference_method.eq(reference)]
            baseline = reference == 'true_arg'
            for col, dataset in enumerate(datasets):
                ax = panels.axes[row, col]
                selected = table_group.loc[table_group.dataset.eq(dataset['name'])]
                for cutoff, color in cutoff_colors.items():
                    curve = selected.loc[selected.af_cutoff.eq(cutoff)].sort_values('min_focal_region_size')
                    y = curve[statistic].to_numpy()
                    if baseline:
                        full = selected.loc[selected.comparison_method.eq('full_hmm'), statistic].item()
                        y = np.concatenate(([full], y))
                    ax.plot(np.arange(len(y)), y, color=color, label=cutoff_label(cutoff, dataset), marker='o')
                minimum_axis(ax, baseline)
                n = selected.matched_mutations.iloc[0]
                ax.set_ylabel(f'{statistic_label} ({n:,} mutations)')
                ax.legend()
            zoom_row(panels.axes[row])
    return panels.figure


if __name__ == '__main__':
    save_figure(mutation_age_plots(age_correlations, ['all']), 'mutation_age_correlations')

# %% [markdown]
# ## Mutation age rank correlations (all mutations)
#
# Spearman ρ uses the same common mutation intersection and cached posterior
# ages as Pearson r above, with average ranks for ties. Only the all-mutation
# comparison is shown. Reference rows, sample-size columns, AF lines (with AC),
# and Full HMM baseline follow the Pearson plots exactly. No trees are redated.

# %%
def all_mutation_rank_correlations():
    runs_table = pd.read_csv(output_dir / 'dating_summary.csv')
    paths = [pathlib.Path(path) for path in runs_table.dated_path]
    paths.extend(config['data_dir'] / dataset['truth'] for dataset in datasets)
    signature = ch5_paramsearch.input_signature(paths, {'statistic': 'spearman_all_common_mutations'})
    output = output_dir / 'mutation_age_spearman_correlations.csv'
    cached = ch5_paramsearch.cached_table(output, signature)
    if cached is not None:
        return cached
    rows = []
    for dataset in datasets:
        runs = runs_table.loc[runs_table.dataset.eq(dataset['name'])].reset_index(drop=True)
        truth = tszip.decompress(config['data_dir'] / dataset['truth'])
        truth = ch5_hyperparams.window_tree_sequence(truth, dataset['analysis_extent_bp'])
        true_ages = ch5_dating.mutation_ages(truth, posterior=False)['age']
        del truth
        ages = []
        common = true_ages.index
        for run in runs.itertuples():
            tree_sequence = tskit.load(run.dated_path)
            inferred = ch5_dating.mutation_ages(tree_sequence, posterior=True)['age']
            ages.append(inferred)
            common = common.intersection(inferred.index, sort=False)
            del tree_sequence
        common = common.sort_values()
        full_index = runs.index[runs.comparison_method.eq('full_hmm')].item()
        references = {'true_arg': true_ages, 'full_hmm': ages[full_index]}
        for reference_method, reference in references.items():
            reference_values = reference.loc[common].to_numpy()
            for index, run in runs.iterrows():
                if reference_method == 'full_hmm' and run.comparison_method == 'full_hmm':
                    continue
                inferred_values = ages[index].loc[common].to_numpy()
                correlation = stats.spearmanr(inferred_values, reference_values).statistic
                rows.append({
                    'dataset': run.dataset,
                    'comparison_method': run.comparison_method,
                    'af_cutoff': run.af_cutoff,
                    'min_focal_region_size': run.min_focal_region_size,
                    'reference_method': reference_method,
                    'frequency_bin': 'all',
                    'matched_mutations': len(common),
                    'spearman_rho': correlation,
                })
        logger.info('Spearman correlations: %s (%d shared mutations)', dataset['name'], len(common))
    table = pd.DataFrame(rows)
    ch5_paramsearch.save_table(table, output, signature)
    return table


if __name__ == '__main__':
    rank_correlations = all_mutation_rank_correlations()
    comparison_keys = grid_keys + ['comparison_method', 'reference_method', 'frequency_bin']
    pearson_counts = age_correlations.loc[age_correlations.frequency_bin.eq('all')]
    matched_counts = rank_correlations.merge(
        pearson_counts[comparison_keys + ['matched_mutations']],
        on=comparison_keys, suffixes=('_spearman', '_pearson'), validate='one_to_one')
    assert len(matched_counts) == len(rank_correlations) == len(pearson_counts)
    assert matched_counts.matched_mutations_spearman.eq(matched_counts.matched_mutations_pearson).all()
    fig = mutation_age_plots(rank_correlations, ['all'], statistic='spearman_rho')
    save_figure(fig, 'mutation_age_spearman_correlations')

# %% [markdown]
# ## Mutation age correlations by DAF
#
# Fixed derived frequencies come from the dated Full HMM genealogy for the same
# common mutation intersection. Frequency bins are DAF < 0.01, 0.01 ≤ DAF < 0.5,
# and DAF ≥ 0.5; boundaries enter the upper bin. Each figure repeats the two
# reference rows across the three sample-size columns for one DAF bin. Within-bin
# correlations can change because the age range is narrower; their values should
# be interpreted within a bin. AF cutoffs select seed ancestors; these DAF bins
# group mutations for evaluation and are a separate quantity.

# %%
if __name__ == '__main__':
    for index, label in enumerate(config['frequency_bins']['labels']):
        save_figure(mutation_age_plots(age_correlations, [label]), f'mutation_age_daf_{index}')

# %% [markdown]
# ## Parameter selection for further testing
#
# Combine focal coverage (a work proxy), extra switches, truth ARG metrics and
# mutation-age correlations. More agreement with Full HMM is expected as focal
# coverage shrinks; the truth metrics are needed to assess whether that change
# improves accuracy. Each cohort is one simulation with its own site density,
# so equal site-count minima need not represent equal physical distances.
# The 300-individual simulation uses seed 2; the two larger simulations use
# seed 4. These are not controlled downsamplings or independent replicate sets.
# There are no simulation replicate error bars or runtime measurements here.
# Larger-cohort topology comparisons are incomplete and are not used to infer
# a topology trend across sample sizes. Mutation-age validation is available
# across all sample sizes; missing ARG metrics remain NaN in the saved table.

# %%
if __name__ == '__main__':
    selection_columns = grid_keys + ['switch_increase_percentage_mean', 'focal_coverage_percentage_mean',
                                    'focal_agreement_percentage_mean', 'gap_agreement_percentage_mean']
    selection = summary[selection_columns].merge(tree_metrics, on=grid_keys, how='left', validate='one_to_one')
    ages = age_correlations.loc[age_correlations.frequency_bin.eq('all') & age_correlations.reference_method.eq('true_arg')]
    selection = selection.merge(ages[grid_keys + ['pearson_r', 'matched_mutations']], on=grid_keys, validate='one_to_one')
    selection.to_csv(output_dir / 'parameter_selection.csv', index=False)
    display_columns = selection_columns[:3] + ['focal_coverage_percentage_mean',
                       'switch_increase_percentage_mean', 'truth_arf', 'truth_tpr', 'pearson_r']
    ipython_display.display(selection[display_columns].round(5))

# %% [markdown]
# ### Coverage expected for larger minimum sizes
#
# Because minimum size filters whole connected focal components, the minimum-1
# length distribution predicts **focal inference-site coverage** at larger
# minima without rerunning matching. This is a site-weighted calculation, not
# a percentage of regions or physical base-pair coverage. It estimates work
# retained, not runtime or accuracy. Candidate sizes are editable in the config.

# %%
if __name__ == '__main__':
    predicted_coverage = []
    baseline_lengths = lengths.loc[~lengths.is_gap & lengths.min_focal_region_size.eq(style['baseline_minimum'])]
    for (dataset, cutoff), table in baseline_lengths.groupby(['dataset', 'af_cutoff']):
        baseline = summary.loc[summary.dataset.eq(dataset) & summary.af_cutoff.eq(cutoff)
                               & summary.min_focal_region_size.eq(style['baseline_minimum'])].iloc[0]
        total_sites = baseline.num_haplotypes * baseline.inference_sites_mean
        for minimum in config['followup_minimum_sizes']:
            retained = table.loc[table.num_sites.ge(minimum)]
            focal_sites = (retained.num_sites * retained['count']).sum()
            predicted_coverage.append({'dataset': dataset, 'af_cutoff': cutoff,
                                       'min_focal_region_size': minimum,
                                       'predicted_focal_site_coverage_percentage': 100 * focal_sites / total_sites})
    predicted_coverage = pd.DataFrame(predicted_coverage)
    predicted_coverage.to_csv(output_dir / 'predicted_focal_site_coverage.csv', index=False)
    ipython_display.display(predicted_coverage.round(2))

# %% [markdown]
# ### Suggested next tests
#
# **Use `af_cutoff=0.05`, `min_focal_region_size=1000` as the next baseline.**
# Minimum 1000 gives the highest overall truth-age correlation among the tested
# minima at each AF in the two larger cohorts. At 300 individuals, AF 0.01 and
# 0.05 at minimum 1000 are effectively tied (r = 0.69530 vs 0.69512). Higher AF
# does add switches, but those counts are not a reliable proxy for age accuracy.
#
# | Individuals | Full HMM vs truth, r | AF 0.05 / minimum 1000 vs truth, r |
# |---:|---:|---:|
# | 300 | 0.71603 | 0.69512 |
# | 3,000 | 0.71325 | 0.58047 |
# | 10,000 | 0.71978 | 0.53002 |
#
# For the larger cohorts, prioritise **AF 0.03 and 0.05 × minima 1000, 2000,
# 3000 and 5000**. Keep 1000 as the measured baseline and test whether larger
# connected regions recover the substantial remaining age-correlation gap.
# AF 0.05 has the highest overall and within-DAF truth-age correlations at
# minimum 1000 in both larger cohorts. Retaining AF 0.03 tests the trade-off
# between seed-extension cost and HMM work; runtime was not measured here.
# The higher minima are candidates for validation, not measured age improvements.
#
# At AF 0.05, minimum 3000 is predicted to retain about **49% / 64%** focal
# inference-site coverage for 3,000 / 10,000 individuals. Minimum 10,000 retains
# only **7% / 15%**, so the 2000–5000 range is a more useful first search for
# balancing coverage and accuracy. At 300 individuals, minimum 3000 retains
# only about 15% focal sites, so keep minimum 1000 as the main small-cohort case.
#
# The deterioration at large sample sizes is also visible within the fixed DAF
# bins. At 10,000 individuals and AF 0.05/minimum 1000, truth-age r is 0.313,
# 0.380 and 0.179 across the three bins, compared with Full HMM's 0.375, 0.553
# and 0.288. This is not solely a change in the mixture of mutation frequencies.
# There is also a trade-off within the largest cohort: at AF 0.05, rare-variant
# truth-age r falls from 0.327 at minimum 1 to 0.313 at minimum 1000 even though
# overall r improves. Keep **AF 0.05 / minimum 1 as one additional control at
# 10,000 individuals** when testing larger minima, and judge the DAF panels
# alongside the overall correlation rather than optimising only the latter.
# Pearson r measures association, not age calibration: the cached correlation
# table also includes median absolute errors and mean bias. For example, median
# absolute error at 10,000 individuals is 87.3 generations for this stitched
# setting and 85.5 for Full HMM, despite the larger correlation difference.
#
# The small-cohort ARG metrics show a separate trade-off: larger minima improve
# age correlations while moving topology scores toward Full HMM. The incomplete
# larger-cohort ARG results do not establish how that topology trade-off scales.
# Treat these settings as candidates from one simulation per cohort; confirm
# age accuracy and runtime in follow-up runs and additional simulation seeds.
