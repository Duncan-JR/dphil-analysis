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
# # Stitching hyperparameters: full versus reduced HMM
#
# Read the saved pt3 experiment grid (AC 3, 5, 7, 10, 20, 40; mismatches 0/1).
# This snapshot keeps pt3 independent of later minimum-size experiments.
# Simulation is on the left and 1KGP on the right. Both diploid
# haplotypes are included; each mean gives equal weight to each haplotype.
# African means YRI in simulation and AFR in 1KGP; Non-African pools all
# remaining populations, using preserved simulation annotations and pedigree labels.

# %%
import dataclasses
import logging
import pathlib

import ch5_hyperparams
import ch5_dating
import ch5_dating_plots
import IPython.display as ipython_display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yaml

logger = logging.getLogger(__name__)

# %%
notebook_dir = (
    pathlib.Path(__file__).resolve().parent
    if '__file__' in globals()
    else pathlib.Path.cwd()
)
repository_dir = notebook_dir
if not (repository_dir / 'experiments').is_dir():
    repository_dir = notebook_dir.parent
config_path = repository_dir / 'experiments/ch5/stitching_pt3_hyperparams_config.yaml'
with config_path.open() as file:
    config = yaml.safe_load(file)
experiment_path = ch5_hyperparams.resolve_path(config['experiment_config'], config_path.parent)
with experiment_path.open() as file:
    experiment = yaml.safe_load(file)
output_dir = ch5_hyperparams.resolve_path(config['output_dir'], config_path.parent)
style = config['plot']
cutoffs = sorted(experiment['ac_cutoff'])
mismatch_allowances = sorted(experiment['max_mismatches'])
ac_colors = dict(zip(cutoffs, style['ac_colors'], strict=True))
mismatch_colors = dict(zip(mismatch_allowances, style['mismatch_colors'], strict=True))
plt.rcParams.update({'figure.dpi': style['dpi'], 'font.size': style['font_size']})

# %% [markdown]
# ## Load and validate the saved runs
#
# Each full path is compared with its stitched path in original reference
# parent IDs. Saved focal masks define maximal connected focal regions and
# complementary gaps independently of the copying paths. Counts and lengths
# are recomputed from these inputs; earlier pt3 summaries are not used.
# The independent runs are processed across available CPU cores.

# %%
if __name__ == '__main__':
    results = ch5_hyperparams.load_results(config_path, workers=config['workers'])
    comparisons = results.comparisons
    lengths = results.lengths
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output_dir / 'haplotype_results.csv', index=False)
    lengths.to_csv(output_dir / 'region_lengths.csv', index=False)
    run_keys = ['dataset', 'ac_cutoff', 'max_mismatches']
    run_summary = comparisons.groupby(run_keys).agg(
        haplotypes=('sample_id', 'size'),
        individuals=('sample_id', 'nunique'),
        undefined_switch_increases=('switch_increase_percentage', lambda values: values.isna().sum()),
        empty_focal_regions=('focal_sites', lambda values: values.eq(0).sum()),
        empty_gaps=('gap_sites', lambda values: values.eq(0).sum()),
    )
    run_summary.to_csv(output_dir / 'run_summary.csv')
    ipython_display.display(run_summary)

# %%
@dataclasses.dataclass
class Panels:
    """A figure and its dataset-by-mismatch axes."""

    figure: plt.Figure
    axes: np.ndarray


def panel_grid(title: str, by_mismatch: bool = False):
    """Keep dataset columns and, where requested, mismatch rows consistent."""
    nrows = len(mismatch_allowances) if by_mismatch else 1
    figsize = (style['panel_width'] * len(config['datasets']), style['panel_height'] * nrows)
    fig, axes = plt.subplots(nrows, len(config['datasets']), figsize=figsize,
                             squeeze=False, sharey='row', layout='constrained')
    for row in range(nrows):
        for column, settings in enumerate(config['datasets']):
            title_text = settings['label']
            if by_mismatch:
                title_text += f' · {mismatch_allowances[row]} mismatches allowed'
            axes[row, column].set_title(title_text)
    fig.suptitle(title)
    return Panels(fig, axes)


def hourglass_line(ax, table, metric: str, color: str, label: str):
    """Plot means and compact hourglass patches spanning mean ± one SEM.

    This uses the same haplotype SEM calculation as the earlier hyperparameter
    error bars. The two triangles meet at the mean; their tips mark ±SEM.
    """
    summary = table.groupby('ac_cutoff')[metric].agg(['mean', 'sem', 'count'])
    ax.plot(summary.index, summary['mean'], color=color, label=label,
            marker='o', markersize=style['marker_size'], linewidth=style['line_width'])
    for cutoff, values in summary.iterrows():
        mean = values['mean']
        sem = values['sem']
        if np.isfinite(sem):
            width = style['hourglass_width']
            vertices = [(cutoff - width, mean - sem), (cutoff + width, mean - sem),
                        (cutoff, mean), (cutoff + width, mean + sem),
                        (cutoff - width, mean + sem), (cutoff, mean)]
            patch = mpatches.Polygon(vertices, facecolor=color, edgecolor=color,
                                     alpha=style['hourglass_alpha'])
            ax.add_patch(patch)
    ax.set_xticks(cutoffs)
    ax.set_xlabel('AC cutoff')
    ax.legend()


def mismatch_lines(metric: str, title: str, ylabel: str):
    """Pool populations and use a consistent colour for each mismatch budget."""
    panels = panel_grid(title)
    fig = panels.figure
    axes = panels.axes
    for column, settings in enumerate(config['datasets']):
        ax = axes[0, column]
        dataset = comparisons.loc[comparisons.dataset.eq(settings['name'])]
        for allowance in mismatch_allowances:
            table = dataset.loc[dataset.max_mismatches.eq(allowance)]
            hourglass_line(ax, table, metric, mismatch_colors[allowance],
                           f'{allowance} mismatches allowed')
        ax.set_ylabel(ylabel)
    return fig


def coverage_lines(region: str, title: str):
    """Average physical window coverage within analogous population groups."""
    panels = panel_grid(title, by_mismatch=True)
    fig = panels.figure
    axes = panels.axes
    metric = f'{region}_coverage_percentage'
    for row, allowance in enumerate(mismatch_allowances):
        for column, settings in enumerate(config['datasets']):
            selected = comparisons.dataset.eq(settings['name'])
            selected &= comparisons.max_mismatches.eq(allowance)
            table = comparisons.loc[selected]
            ax = axes[row, column]
            for group, color in style['population_colors'].items():
                group_table = table.loc[table.population_group.eq(group)]
                means = group_table.groupby('ac_cutoff')[metric].mean()
                ax.plot(means.index, means, color=color, label=group, marker='o',
                        markersize=style['marker_size'], linewidth=style['line_width'])
            ax.set(xlabel='AC cutoff', ylabel=f'Genome covered by {region} regions (%)',
                   xticks=cutoffs, ylim=(0, 100))
            ax.legend()
    return fig


def length_distributions(region: str, title: str):
    """Overlay pooled interval counts with KDEs scaled to counts per bin.

    Each row uses common equal-width bins across datasets and AC cutoffs.
    The reflected KDE respects the positive length boundary and uses all data.
    The display limit is configured; tail counts are reported in each legend.
    A constant-length series has no KDE and is shown by a vertical line.
    """
    panels = panel_grid(title, by_mismatch=True)
    fig = panels.figure
    axes = panels.axes
    for row, allowance in enumerate(mismatch_allowances):
        selected = lengths.region.eq(region) & lengths.max_mismatches.eq(allowance)
        row_table = lengths.loc[selected]
        upper = row_table.num_sites.quantile(style['length_display_quantile'])
        bins = np.linspace(0, upper + 1, style['histogram_bins'] + 1)
        bin_width = bins[1] - bins[0]
        grid = np.linspace(0, bins[-1], style['kde_grid_size'])
        for column, settings in enumerate(config['datasets']):
            ax = axes[row, column]
            dataset = row_table.loc[row_table.dataset.eq(settings['name'])]
            for cutoff in cutoffs:
                values = dataset.loc[dataset.ac_cutoff.eq(cutoff), 'num_sites'].to_numpy()
                color = ac_colors[cutoff]
                tail_count = np.count_nonzero(values > bins[-1])
                label = f'AC {cutoff} (tail: {tail_count})'
                ax.hist(values, bins=bins, color=color, alpha=style['histogram_alpha'])
                if len(values) > 1 and np.ptp(values) > 0:
                    kde = stats.gaussian_kde(values, bw_method=style['kde_bandwidth'])
                    density = kde(grid) + kde(-grid)
                    counts = density * len(values) * bin_width
                    ax.plot(grid, counts, color=color, label=label, linewidth=style['line_width'])
                elif len(values) > 0:
                    ax.axvline(values[0], color=color, label=label)
            ax.set(xlabel=f'{region.capitalize()} region length (number of sites)',
                   ylabel='Count', xlim=(0, bins[-1]), ylim=(0, None))
            ax.legend(title='AC cutoff')
    return fig


def switch_composition():
    """Show raw mean switch counts, with one explicitly partitioned full bar."""
    panels = panel_grid('Switch composition changes', by_mismatch=True)
    fig = panels.figure
    axes = panels.axes
    baseline_cutoff = config['baseline_partition_ac_cutoff']
    assert baseline_cutoff in cutoffs
    x = hmm_bar_positions()
    stitched_x = x[1:]
    for row, allowance in enumerate(mismatch_allowances):
        for column, settings in enumerate(config['datasets']):
            selected = comparisons.dataset.eq(settings['name'])
            selected &= comparisons.max_mismatches.eq(allowance)
            table = comparisons.loc[selected]
            baseline = table.loc[table.ac_cutoff.eq(baseline_cutoff)]
            ax = axes[row, column]
            bottom = np.zeros(len(x))
            for category, label, color in zip(style['switch_categories'],
                                               style['switch_labels'], style['switch_colors'], strict=True):
                full_mean = baseline[f'full_{category}_switches'].mean()
                stitched_means = table.groupby('ac_cutoff')[f'stitched_{category}_switches'].mean()
                heights = np.concatenate(([full_mean], stitched_means.loc[cutoffs].to_numpy()))
                ax.bar(x, heights, bottom=bottom, width=style['bar_width'], color=color, label=label)
                bottom += heights
            full_label = f'Full HMM\n(AC {baseline_cutoff} partition)'
            ax.set(xticks=x, xticklabels=[full_label] + [str(cutoff) for cutoff in cutoffs],
                   ylabel='Mean switches per haplotype', ylim=(0, None))
            ax.text(stitched_x.mean(), style['group_label_y'], 'Stitched HMM (AC cutoff)',
                    transform=ax.get_xaxis_transform(), ha='center')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(labels))
    return fig


def hmm_bar_positions() -> np.ndarray:
    """Leave one configured bar width of clear space after the Full HMM bar."""
    bar_width = style['bar_width']
    clear_gap = bar_width * style['full_stitched_gap_bar_widths']
    first_stitched = bar_width + clear_gap
    stitched_x = np.arange(len(cutoffs)) + first_stitched
    return np.concatenate(([0], stitched_x))


def save_figure(fig, filename: str):
    """Export each displayed figure beside its underlying cohort tables."""
    fig.savefig(output_dir / f'{filename}.png')
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## Extra switches vs hyper parameters
#
# Extra switches are stitched minus full switches. Percent increases are
# calculated per haplotype using full switches as denominator, then averaged.
# Zero-switch full paths have undefined increases and are excluded, with counts
# reported above. Hourglasses show mean ± one haplotype SEM; these describe
# haplotype variability without accounting for within-individual dependence.

# %%
if __name__ == '__main__':
    fig = mismatch_lines('num_extra_switches', 'Extra switches vs mismatch allowance',
                         'Mean extra switches')
    save_figure(fig, 'extra_switches_vs_mismatch_allowance')

# %%
if __name__ == '__main__':
    fig = mismatch_lines('switch_increase_percentage', 'Increase in switches vs mismatch allowance',
                         'Mean percent increase in switches')
    save_figure(fig, 'switch_increase_vs_mismatch_allowance')

# %% [markdown]
# ## Focal region distributions
#
# Coverage uses base pairs within the configured analysis window: 0–10 Mb in
# simulation and 1–11 Mb in 1KGP. It uses the saved mask boundaries converted
# to physical positions, clipped to these windows, rather than the retained
# 64 Mb chromosome span. The population curves are haplotype means.
# Length distributions pool maximal focal components across all populations.
# Histograms show counts; density curves are scaled by interval count and bin
# width so that they overlay the same count axis. Rows show mismatch 0 and 1.
# The x-axis displays through the pooled 99th percentile for each mismatch row
# (configurable); each legend reports the number of intervals beyond the shown
# range. KDEs use every interval, and the exported table retains the full tails.

# %%
if __name__ == '__main__':
    fig = coverage_lines('focal', 'Focal region coverage')
    save_figure(fig, 'focal_region_coverage')

# %%
if __name__ == '__main__':
    fig = length_distributions('focal', 'Focal region length distributions')
    save_figure(fig, 'focal_region_length_distributions')

# %% [markdown]
# ## Gap distributions
#
# Gaps are maximal complements of focal coverage. Gap coverage is the physical
# complement within the same window; lengths count inference sites, including
# terminal gaps. Populations, mismatch rows, histograms and density curves follow
# the same definitions as the focal region plots.

# %%
if __name__ == '__main__':
    fig = coverage_lines('gap', 'Gap coverage')
    save_figure(fig, 'gap_coverage')

# %%
if __name__ == '__main__':
    fig = length_distributions('gap', 'Gap length distributions')
    save_figure(fig, 'gap_length_distributions')

# %% [markdown]
# ## Switch composition
#
# Bars stack raw mean counts, from bottom to top: within focal coverage, within
# gaps, and at gap seams. Switches are classified using the provenance of their
# adjacent inference sites; equal parents across a seam contribute no switch.
# Each stitched bar uses its own mask. Full HMM totals are invariant across AC
# cutoffs, but their categories depend on the mask. The single Full HMM bar uses
# **the AC 10 mask for the row's mismatch allowance**, as set in the config.

# %%
if __name__ == '__main__':
    fig = switch_composition()
    save_figure(fig, 'switch_composition_changes')

# %% [markdown]
# ## Parent agreement
#
# Agreement compares original reference parent IDs at inference sites, using
# only sites within the named region as denominator. Percentages are averaged
# across haplotypes, pooling populations; empty regions are excluded and counted
# in the run summary. Hourglasses use the same mean ± SEM helper as switch plots.

# %%
if __name__ == '__main__':
    fig = mismatch_lines('focal_agreement_percentage', 'Parent agreement in focal regions',
                         'Mean parent agreement in focal regions (%)')
    save_figure(fig, 'parent_agreement_in_focal_regions')

# %%
if __name__ == '__main__':
    fig = mismatch_lines('gap_agreement_percentage', 'Parent agreement in gaps',
                         'Mean parent agreement in gaps (%)')
    save_figure(fig, 'parent_agreement_in_gaps')

# %% [markdown]
# ## Tree sequence comparisons
#
# Load both Full HMM outputs and every stitched matched
# output in the current parameter grid. Edge and tree counts describe the
# **raw saved tree sequences**, including their reference ancestry and retained
# chromosome coordinates. Plot stitched minus Full HMM counts against AC
# cutoff: edge differences in the first row and tree differences in the second.
# Simulation is on the left and 1KGP on the right; colour identifies mismatch
# allowance. Negative values mean stitching produces fewer edges or trees.

# %%
if __name__ == '__main__':
    tree_sequences = ch5_hyperparams.load_tree_sequences(config_path)
    ts_statistics = tree_sequences.statistics
    ts_differences = ch5_hyperparams.tree_statistic_differences(ts_statistics)
    ts_statistics.to_csv(output_dir / 'ts_statistics.csv', index=False)
    ts_differences.to_csv(output_dir / 'ts_statistic_differences.csv', index=False)
    ipython_display.display(ts_differences)

# %%
def metric_row_lines(table, metrics: dict, title: str):
    """Use metric rows and dataset columns with mismatch-coloured curves."""
    figsize = (style['panel_width'] * len(config['datasets']), style['panel_height'] * len(metrics))
    fig, axes = plt.subplots(len(metrics), len(config['datasets']), figsize=figsize,
                             squeeze=False, sharex=True, sharey='row', layout='constrained')
    for row, (metric, ylabel) in enumerate(metrics.items()):
        for column, settings in enumerate(config['datasets']):
            ax = axes[row, column]
            dataset_table = table.loc[table.dataset.eq(settings['name'])]
            for allowance in mismatch_allowances:
                selected = dataset_table.max_mismatches.eq(allowance)
                observations = dataset_table.loc[selected].sort_values('ac_cutoff')
                ax.plot(observations.ac_cutoff, observations[metric],
                        color=mismatch_colors[allowance], label=f'{allowance} mismatches allowed',
                        marker='o', markersize=style['marker_size'], linewidth=style['line_width'])
            ax.set(xlabel='AC cutoff', ylabel=ylabel, xticks=cutoffs, title=settings['label'])
            ax.legend()
    fig.suptitle(title)
    return Panels(fig, axes)


def ts_statistic_differences():
    """Plot signed raw edge/tree count changes relative to Full HMM."""
    metrics = {'num_edges_difference': 'Edges: stitched HMM − Full HMM',
               'num_trees_difference': 'Trees: stitched HMM − Full HMM'}
    panels = metric_row_lines(ts_differences, metrics, 'Comparison of TS statistics')
    for ax in panels.axes.flat:
        ax.axhline(0, color='gray', linestyle='--', linewidth=style['line_width'])
    fig = panels.figure
    return fig


if __name__ == '__main__':
    fig = ts_statistic_differences()
    save_figure(fig, 'comparison_of_ts_statistics')

# %% [markdown]
# ### Comparing ARG metrics
#
# Compare each stitched output with its dataset's Full HMM output using
# [tscompare.haplotype_arf](https://tskit.dev/tscompare/docs/stable/api.html#tscompare.haplotype_arf).
# The stitched sequence is the first argument and Full HMM is the second:
# ARF is the fraction of stitched node span unsupported by Full HMM (lower is
# better); TPR is the fraction of Full HMM node span represented by stitching
# (higher is better). Rows show ARF and TPR; columns show simulation and
# 1KGP. Colour identifies mismatch allowance, with two lines in every panel.
#
# Haplotypes are aligned by `(sample_id, ploidy_index)` in matched-node metadata.
# Both sequences are restricted to the dataset's 10 Mb window,
# translated to shared coordinates and simplified in the same sample
# order, retaining unary ancestors for node-span comparisons.
# Missing terminal ancestry is retained within the explicit window.
#
# **Interpretation reminder:** ARF = 0.135 at AC 3 / zero mismatches means
# 13.5% of stitched node span is not represented in Full HMM; lower is better.
# TPR = 0.853 means 85.3% of Full HMM node span is represented in the stitched
# result; higher is better. Their denominators belong to different ARGs.

# %%
if __name__ == '__main__':
    arg_comparisons = ch5_hyperparams.arg_metrics(
        tree_sequences, config['datasets'], workers=config['workers']
    )
    arg_comparisons.to_csv(output_dir / 'arg_metrics.csv', index=False)
    ipython_display.display(arg_comparisons)

# %%
def arg_metric_lines():
    """Plot ARF and TPR against Full HMM for both datasets."""
    metrics = {'arf': 'ARF', 'tpr': 'TPR'}
    panels = metric_row_lines(arg_comparisons, metrics, 'Comparing ARG metrics')
    return panels.figure


if __name__ == '__main__':
    fig = arg_metric_lines()
    save_figure(fig, 'comparing_arg_metrics')

# %% [markdown]
# ## ARG metrics against the true simulated tree sequence
#
# Compare the simulation's Full HMM and each stitched output with the true ARG.
# Align truth's VCF haplotype identities to the matched-node metadata and use
# the same 0–10 Mb window, retaining unary ancestors. The columns show ARF and
# TPR; colour identifies mismatch allowance. The Full HMM baseline is placed
# at **1 (Full HMM)** for display, and has no actual AC cutoff or mismatch setting.
#
# Here ARF is the proportion of the matched output's node span not represented
# in truth; TPR is the proportion of true node span represented in that output.
# Lower ARF and higher TPR indicate greater agreement with the true ARG.
# Each y-axis zooms to the observed values to make small differences visible.

# %%
if __name__ == '__main__':
    truth_comparisons = ch5_hyperparams.truth_arg_metrics(
        tree_sequences, config_path, workers=config['workers']
    )
    truth_comparisons.to_csv(output_dir / 'arg_metrics_against_truth.csv', index=False)
    ipython_display.display(truth_comparisons)

# %%
def truth_arg_metric_lines():
    """Show Full HMM and stitched simulation ARF/TPR against biological truth."""
    figsize = (style['panel_width'] * 2, style['panel_height'])
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='constrained')
    full = truth_comparisons.loc[truth_comparisons.comparison_method.eq('full_hmm')]
    full_cutoff = config['truth_comparison']['full_hmm_cutoff']
    metrics = {'arf': 'ARF', 'tpr': 'TPR'}
    for ax, (metric, ylabel) in zip(axes, metrics.items(), strict=True):
        baseline = full[metric].item()
        for allowance in mismatch_allowances:
            selected = truth_comparisons.comparison_method.eq('stitched_hmm')
            selected &= truth_comparisons.max_mismatches.eq(allowance)
            observations = truth_comparisons.loc[selected].sort_values('ac_cutoff')
            x = np.concatenate(([full_cutoff], observations.ac_cutoff.to_numpy()))
            y = np.concatenate(([baseline], observations[metric].to_numpy()))
            ax.plot(x, y, color=mismatch_colors[allowance], marker='o',
                    markersize=style['marker_size'], linewidth=style['line_width'],
                    label=f'{allowance} mismatches allowed')
        ticks = [full_cutoff, *cutoffs]
        tick_labels = [f'{full_cutoff}\n(Full HMM)', *[str(cutoff) for cutoff in cutoffs]]
        ax.set(xlabel='AC cutoff', ylabel=ylabel, xticks=ticks, xticklabels=tick_labels,
               title='Simulation vs true ARG')
        ax.legend()
    fig.suptitle('ARG metrics against the true tree sequence')
    return fig


if __name__ == '__main__':
    fig = truth_arg_metric_lines()
    save_figure(fig, 'arg_metrics_against_truth')

# %%

# %% [markdown]
# ## Mutation-age correlations after dating
#
# All 26 inferred ARGs (Full HMM and twelve reduced-HMM cases per dataset) are
# post-processed with the updated local tsinfer. Singleton and other omitted
# sites are restored from the observed sample VCZ, retaining the inferred
# topology and observed alleles. The analysed window is retained, then
# `tsdate.preprocess_ts` splits disjoint nodes before `tsdate.date` uses the
# variational-gamma method. Dated ARGs and input/settings manifests are saved
# under `dated_args`; unchanged results are reused on subsequent runs.
#
# Mutation rates are 2.35 × 10⁻⁸ per bp per generation for the simulation
# (from its mutation provenance), and a configured 1.29 × 10⁻⁸ for 1KGP.
# Singletons retain their observed phase. See the
# [tsdate dating and preprocessing documentation](https://tskit.dev/tsdate/docs/stable/usage.html).
#
# **These plots compare mutation ages, not node ages.** Inferred ages are
# tsdate's posterior mutation means (`mutation.metadata['mn']`), rather than
# branch-midpoint times in the mutation table. Truth uses the simulated
# mutation event times. Mutations are paired by genomic position and identical
# ancestral/derived states. Sites with zero or multiple mutations, or nonfinite
# ages, are excluded. Each panel/dataset uses one common set across all its
# cases, so the plotted correlations are directly comparable. The CSV reports
# the number of matched mutations. Correlations are unweighted Pearson r on
# ages in generations, conditional on this common unambiguous mutation set.
#
# Left: simulation Full HMM and all reduced-HMM cases against true ages.
# Right: reduced HMM against the dated Full HMM. Colour indicates mismatch
# allowance; line style indicates dataset. AC cutoff varies on the x-axis;
# the left panel adds a separate Full HMM category.

# %%
if __name__ == '__main__':
    dated_runs = ch5_dating.date_all(config_path)
    dated_runs.to_csv(output_dir / 'dated_arg_summary.csv', index=False)
    mutation_age_results = ch5_dating.mutation_correlations(dated_runs, config_path)
    mutation_age_results.to_csv(output_dir / 'mutation_age_correlations.csv', index=False)
    ipython_display.display(mutation_age_results.loc[mutation_age_results.frequency_bin.eq('all')])


# %%
def mutation_age_correlation_plots(table):
    """Use the same paired panels for the overall and frequency-bin analyses."""
    return ch5_dating_plots.correlation_figure(table, config, experiment)


if __name__ == '__main__':
    fig = mutation_age_correlation_plots(mutation_age_results)
    save_figure(fig, 'mutation_age_correlations')

# %%

# %% [markdown]
# ## Mutation-age correlations by derived allele frequency
#
# The three rows split observed derived allele frequency (DAF) into
# **DAF < 0.01**, **0.01 ≤ DAF < 0.5**, and **DAF ≥ 0.5**. Boundaries enter
# the upper bin. Frequencies are measured among the 600 sample haplotypes in
# the dated Full HMM, and that bin assignment is held fixed for every case in
# both panels. These are derived frequencies, rather than minor frequencies.
# Each row repeats the overall truth and Full HMM comparisons on that subset;
# the same shared-mutation and posterior-mean rules apply. Within-bin Pearson r
# can be lower simply because conditioning on frequency restricts the age range.

# %%
if __name__ == '__main__':
    fig = ch5_dating_plots.correlation_figure(
        mutation_age_results, config, experiment, split_frequency=True)
    save_figure(fig, 'mutation_age_correlations_by_frequency')

# %% [markdown]
# ## Full HMM versus stitched mutation-age densities
#
# These density scatterplots compare **Full HMM posterior mutation ages on the
# x-axis** with **stitched posterior mutation ages on the y-axis**, across
# **all allele frequencies**. Simulation and 1KGP each have a separate grid.
# Columns follow the parameter on the line plots' x-axis; rows follow their
# coloured curves. Every panel within a dataset uses the same common mutation
# set as the overall Full HMM correlation panel, with identical axes, bins and
# colour scale. Ages and mutation counts both use logarithmic scales; the
# dashed line marks equal ages. Annotated Pearson r uses ages in generations
# before log transformation. No simulation truth or frequency filtering enters
# these comparisons.

# %%
if __name__ == '__main__':
    stitched_age_pairs = ch5_dating.stitched_mutation_age_pairs(dated_runs, config_path)
    for dataset in config['datasets']:
        if dataset['name'] in config['dating']['compare_full_datasets']:
            fig = ch5_dating_plots.stitched_age_density_figure(
                stitched_age_pairs, config, experiment, dataset)
            save_figure(fig, f"mutation_age_density_{dataset['name']}")
