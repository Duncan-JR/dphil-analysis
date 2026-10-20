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
# # Greedy focal stitching: AC 10 versus AC 40
#
# `greedy_only_30mbp` · 300 diploid individuals · chr20, 1–31 Mb ·
# AC ≤ 10 and AC ≤ 40 · exact focal intervals · 8 matching workers.
# Three trials per mode and cutoff; the Full HMM baseline pools all six trials.
# Greedy uses the full HMM within gaps.

# %%
import logging
import pathlib

import ch5_hyperparams
import ch5_perf
import ch5_perf_pt5
import IPython.display as ipython_display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    repository_dir = notebook_dir.parent.parent
config_path = repository_dir / 'experiments/ch5/stitching_pt5_perf_config.yaml'
with config_path.open() as source:
    config = yaml.safe_load(source)
output_dir = ch5_hyperparams.resolve_path(config['output_dir'], config_path.parent)
style = config['plot']
methods = list(config['methods'])
stitched_methods = [method for method in methods if method != 'full_hmm']
colors = {method: values['color'] for method, values in config['methods'].items()}
labels = {method: values['label'] for method, values in config['methods'].items()}
plt.rcParams.update({'figure.dpi': style['dpi'], 'font.size': style['font_size'],
                     'axes.spines.top': False, 'axes.spines.right': False})

# %%
if __name__ == '__main__':
    results = ch5_perf_pt5.load_results(config_path)
    runs = results.runs
    stages = results.stages
    comparisons = results.comparisons
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output_dir / 'haplotype_results.csv', index=False)
    results.fingerprints.to_csv(output_dir / 'path_fingerprints.csv', index=False)
    reference_phases = ch5_perf.reference_phases(runs, stages)
    reference_phases.to_csv(output_dir / 'reference_phases.csv', index=False)
    runs['wall_ms_per_haplotype'] = 1000 * runs.wall_s / runs.matched_haplotypes
    runs['effective_cores'] = (runs.user_s + runs.system_s) / runs.wall_s
    runs['system_cpu_percentage'] = 100 * runs.system_s / (runs.user_s + runs.system_s)
    timing_summary = runs.groupby(['mode', 'method']).agg(
        mean_wall_s=('wall_s', 'mean'), min_wall_s=('wall_s', 'min'),
        max_wall_s=('wall_s', 'max'),
        mean_wall_ms_per_haplotype=('wall_ms_per_haplotype', 'mean'),
        mean_effective_cores=('effective_cores', 'mean'),
        mean_system_cpu_percentage=('system_cpu_percentage', 'mean'),
        maximum_trial_rss_bytes=('trial_peak_rss_bytes', 'max'),
    )
    timing_summary.to_csv(output_dir / 'timing_summary.csv')

# %%
def panels(columns=1, rows=1):
    """Use the same panel sizing across the notebook."""
    return plt.subplots(rows, columns, squeeze=False, layout='constrained',
                        figsize=(style['panel_width'] * columns,
                                 style['panel_height'] * rows))


def save_figure(fig, name):
    """Keep displayed plots and reusable PNG/PDF exports identical."""
    fig.savefig(output_dir / f'{name}.png', dpi=style['export_dpi'], bbox_inches='tight')
    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    plt.show()


def phase_legend(ax, settings, title=None, order=None, **kwargs):
    """List stack phases top to bottom, with optional phase-specific colours."""
    legend_style = style['phase_legend']
    if order is None:
        order = list(reversed(settings))
    handles = []
    for phase in order:
        values = settings[phase]
        handle = mpatches.Patch(facecolor=values.get('legend_facecolor', legend_style['facecolor']),
                               edgecolor=legend_style['edgecolor'], hatch=values['hatch'],
                               label=values['label'])
        handles.append(handle)
    ax.legend(handles=handles, title=title, fontsize=legend_style['font_size'],
              handlelength=legend_style['handlelength'],
              handleheight=legend_style['handleheight'],
              labelspacing=legend_style['labelspacing'], **kwargs)


def trial_dots(ax, x, values, color):
    """Show every timing repeat rather than an inferred confidence interval."""
    offsets = np.linspace(min(style['repeat_offsets']), max(style['repeat_offsets']), len(values))
    ax.scatter(x + offsets, values, color=color, edgecolor='white',
               linewidth=0.5, s=style['repeat_marker_size'], zorder=5)


def distribution(ax, metric, xlabel):
    """Overlay fine count histograms and KDEs on the same count scale."""
    pooled = comparisons[metric].dropna().to_numpy()
    bins = np.linspace(pooled.min() - 0.5, pooled.max() + 0.5,
                       style['histogram_bins'] + 1)
    grid = np.linspace(bins[0], bins[-1], style['kde_grid_size'])
    bin_width = bins[1] - bins[0]
    for method in stitched_methods:
        table = comparisons.loc[comparisons.method.eq(method)]
        values = table[metric].dropna().to_numpy()
        color = colors[method]
        ax.hist(values, bins=bins, color=color, alpha=style['histogram_alpha'])
        kde = stats.gaussian_kde(values, bw_method=style['kde_bandwidth'])
        counts = kde(grid) * len(values) * bin_width
        ax.plot(grid, counts, color=color, linewidth=style['line_width'],
                label=labels[method])
        mean = values.mean()
        ax.axvline(mean, color=color, linestyle='--', linewidth=style['line_width'],
                   label=f'{labels[method]} mean: {mean:.2f}')
        excluded = table[metric].isna().sum()
        if excluded > 0:
            ax.text(0.98, 0.9 - 0.08 * stitched_methods.index(method),
                    f'{labels[method]}: {excluded} undefined',
                    ha='right', transform=ax.transAxes, color=color)
    ax.set(xlabel=xlabel, ylabel='Haplotype count', xlim=(bins[0], bins[-1]))
    ax.legend(fontsize='small')

# %% [markdown]
# ## Correctness
#
# One copy of each haplotype; paths and mutations agree across all modes and
# repeats within each case. Each Greedy case uses its own focal mask.
# The Full HMM switch-composition bar uses the AC 10 partition; its total
# switch count is independent of the partition.
# Parent agreement is averaged per haplotype within each named region.

# %%
if __name__ == '__main__':
    message = (f"**{len(results.quality)} saved checks passed:** zero allele reconstruction "
               "errors and zero mismatches.")
    ipython_display.display(ipython_display.Markdown(message))
    fig, axes = panels(columns=2, rows=2)
    distribution(axes[0, 0], 'num_extra_switches', 'Extra switches relative to Full HMM')
    axes[0, 0].set_title('Extra switches')
    distribution(axes[0, 1], 'switch_increase_percentage', 'Increase in switches (%)')
    axes[0, 1].set_title('Percent increase in switches')

    ax = axes[1, 0]
    baseline = comparisons.loc[comparisons.method.eq(config['correctness']['baseline_partition_method'])]
    for index, method in enumerate(methods):
        table = baseline if method == 'full_hmm' else comparisons.loc[comparisons.method.eq(method)]
        prefix = 'full' if method == 'full_hmm' else 'stitched'
        bottom = 0
        for category, values in style['switch_categories'].items():
            height = table[f'{prefix}_{category}_switches'].mean()
            ax.bar(index, height, bottom=bottom, width=style['bar_width'],
                   color=colors[method], alpha=values['alpha'],
                   hatch=values['hatch'], edgecolor='white')
            bottom += height
        ax.annotate(f'{bottom:.2f}', (index, bottom), xytext=(0, 4),
                    textcoords='offset points', ha='center')
    ax.set(title='Switch composition', ylabel='Mean switches per haplotype',
           xticks=np.arange(len(methods)), xticklabels=[labels[method] for method in methods])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    phase_legend(ax, style['switch_categories'], loc='upper center',
                 bbox_to_anchor=(0.5, -0.12), ncol=len(style['switch_categories']))

    ax = axes[1, 1]
    regions = ['focal', 'gap']
    positions = np.arange(len(regions))
    width = style['grouped_bar_width']
    for index, method in enumerate(stitched_methods):
        table = comparisons.loc[comparisons.method.eq(method)]
        means = [table[f'{region}_agreement_percentage'].mean() for region in regions]
        x = positions + (index - (len(stitched_methods) - 1) / 2) * width
        ax.bar(x, means, width=width, color=colors[method], label=labels[method])
        for location, mean in zip(x, means, strict=True):
            ax.annotate(f'{mean:.2f}%', (location, mean), xytext=(0, 4),
                        textcoords='offset points', ha='center', fontsize='small')
    ax.set(title='Parent agreement with Full HMM', ylabel='Mean agreement (%)',
           xticks=positions, xticklabels=['Focal regions', 'Gaps'], ylim=(0, 110))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(loc='lower right')
    save_figure(fig, 'correctness')

# %% [markdown]
# ## Performance
#
# Bars show trial means; dots show individual trials (three per Greedy case,
# six pooled across both cutoff passes for Full HMM).

# %%
if __name__ == '__main__':
    fig, axes = panels()
    fig.set_size_inches(style['reference_figure_width'], style['panel_height'])
    ax = axes[0, 0]
    selected = runs.loc[runs['mode'].eq('from_reference')]
    for index, method in enumerate(methods):
        table = reference_phases.loc[reference_phases.method.eq(method)]
        means = table.groupby('phase').wall_s.mean()
        bottom = 0
        for phase, values in style['reference_phases'].items():
            if phase not in means:
                continue
            height = means.loc[phase]
            ax.bar(index, height, bottom=bottom, width=style['bar_width'],
                   color=colors[method], alpha=values['alpha'],
                   hatch=values['hatch'], edgecolor='white')
            bottom += height
        times = selected.loc[selected.method.eq(method)].sort_values('repeat').wall_s.to_numpy()
        trial_dots(ax, index, times, colors[method])
        ax.annotate(f'{bottom:.1f} s', (index, times.max()), xytext=(0, 6),
                    textcoords='offset points', ha='center')
    ax.set(title='From ancestor reference to written outputs', ylabel='Wall time (s)',
           xticks=np.arange(len(methods)), xticklabels=[labels[method] for method in methods])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    phase_legend(ax, style['reference_phases'], order=style['reference_legend_order'],
                 loc='upper left', bbox_to_anchor=(1.02, 1))
    save_figure(fig, 'from_reference_wall_time')

# %% [markdown]
# Resident totals include worker setup and stitching. Wall time per haplotype
# measures throughput at eight workers. Worker elapsed times include waits and
# overlap across workers; Full HMM has no equivalent worker-stage breakdown.

# %%
if __name__ == '__main__':
    resident = runs.loc[runs['mode'].eq('resident_match')]
    fig, axes = panels(columns=2)
    ax = axes[0, 0]
    full_mean = resident.loc[resident.method.eq('full_hmm'), 'wall_ms_per_haplotype'].mean()
    tick_labels = []
    for index, method in enumerate(methods):
        values = resident.loc[resident.method.eq(method)].sort_values('repeat').wall_ms_per_haplotype.to_numpy()
        mean = values.mean()
        speedup = full_mean / mean
        tick_labels.append(f'{labels[method]}\n({speedup:.2f}× Full HMM speed)')
        ax.bar(index, mean, width=style['bar_width'], color=colors[method])
        trial_dots(ax, index, values, colors[method])
        ax.annotate(f'{mean:.2f} ms', (index, values.max()), xytext=(0, 6),
                    textcoords='offset points', ha='center')
    ax.set(title='Resident matching throughput', ylabel='Wall time / matched haplotypes (ms)',
           xticks=np.arange(len(methods)), xticklabels=tick_labels)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)

    ax = axes[0, 1]
    worker = stages.loc[stages.scope.eq('worker')]
    worker = worker.merge(resident[['run_id', 'method', 'matched_haplotypes']], on='run_id')
    worker['ms_per_haplotype'] = 1000 * worker.service_s / worker.matched_haplotypes
    worker['phase'] = worker.stage.where(worker.stage.isin(['focal_match', 'gap_match']), 'overhead')
    per_run = worker.groupby(['run_id', 'method', 'phase']).ms_per_haplotype.sum().reset_index()
    per_run.to_csv(output_dir / 'worker_service_times.csv', index=False)
    for index, method in enumerate(stitched_methods):
        table = per_run.loc[per_run.method.eq(method)]
        means = table.groupby('phase').ms_per_haplotype.mean()
        bottom = 0
        for phase, values in style['worker_phases'].items():
            height = means.get(phase, 0)
            ax.bar(index, height, bottom=bottom, width=style['bar_width'],
                   color=colors[method], alpha=values['alpha'],
                   hatch=values['hatch'], edgecolor='white')
            bottom += height
        matching_mean = means.loc[['focal_match', 'gap_match']].sum()
        ax.annotate(f'Matching: {matching_mean:.1f} ms', (index, bottom),
                    xytext=(0, 6), textcoords='offset points', ha='center')
    ax.set(title='Recorded worker time per haplotype', ylabel='Worker elapsed time (ms)',
           xticks=np.arange(len(stitched_methods)),
           xticklabels=[labels[method] for method in stitched_methods])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
    phase_legend(ax, style['worker_phases'], order=style['worker_legend_order'])
    save_figure(fig, 'resident_matching')

# %% [markdown]
# Memory uses sampled whole-trial peak RSS, including preparation and output.
# CPU utilisation is process CPU time / resident wall time, including system time.

# %%
if __name__ == '__main__':
    fig, axes = panels(columns=2)
    ax = axes[0, 0]
    width = style['grouped_bar_width']
    mode_count = len(style['memory_modes'])
    for mode_index, (mode, settings) in enumerate(style['memory_modes'].items()):
        selected = runs.loc[runs['mode'].eq(mode)]
        for index, method in enumerate(methods):
            values = selected.loc[selected.method.eq(method)].sort_values('repeat').trial_peak_rss_bytes.to_numpy()
            values = values / 1024 ** 3
            position = index + (mode_index - (mode_count - 1) / 2) * width
            ax.bar(position, values.mean(), width=width, color=colors[method],
                   alpha=settings['alpha'], hatch=settings['hatch'], edgecolor='white')
            trial_dots(ax, position, values, colors[method])
            ax.annotate(f'{values.mean():.1f}', (position, values.max()), xytext=(0, 6),
                        textcoords='offset points', ha='center', fontsize='small')
    ax.set(title='Peak process memory', ylabel='Whole-trial peak RSS (GiB)',
           xticks=np.arange(len(methods)), xticklabels=[labels[method] for method in methods])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    phase_legend(ax, style['memory_modes'], order=list(style['memory_modes']),
                 loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

    ax = axes[0, 1]
    workers = resident.actual_workers.iloc[0]
    for index, method in enumerate(methods):
        values = resident.loc[resident.method.eq(method)].sort_values('repeat').effective_cores.to_numpy()
        ax.bar(index, values.mean(), width=style['bar_width'], color=colors[method])
        trial_dots(ax, index, values, colors[method])
        ax.annotate(f'{values.mean():.2f}', (index, values.max()), xytext=(0, 6),
                    textcoords='offset points', ha='center')
    ax.axhline(workers, color=style['legend_color'], linestyle='--',
               linewidth=style['line_width'], label=f'{workers} matching workers')
    ax.set(title='CPU utilisation during resident matching', ylabel='Average CPU cores used',
           xticks=np.arange(len(methods)), xticklabels=[labels[method] for method in methods],
           ylim=(0, max(workers, resident.effective_cores.max()) * 1.2))
    ax.legend(loc='lower right')
    save_figure(fig, 'memory_and_cpu')

# %% [markdown]
# ## Findings
#
# Increasing the Greedy cutoff from 10 to 40 improves resident throughput and
# expands focal coverage, but adds switches and lowers parent agreement with
# Full HMM. Both Greedy cases remain slower from ancestor reference to written
# outputs: the shorter matching phase does not offset focal preparation.
# All saved quality checks pass, with identical paths and mutations across
# modes/repeats within each case; this does not imply identical copying paths
# between Greedy and Full HMM.

# %%
if __name__ == '__main__':
    summary = timing_summary.xs('resident_match').loc[methods, ['mean_wall_ms_per_haplotype']].copy()
    baseline_ms = summary.loc['full_hmm', 'mean_wall_ms_per_haplotype']
    summary['speedup_over_full_hmm'] = baseline_ms / summary.mean_wall_ms_per_haplotype
    summary['from_reference_wall_s'] = timing_summary.xs('from_reference').mean_wall_s
    correctness_means = comparisons.groupby('method').agg(
        mean_extra_switches=('num_extra_switches', 'mean'),
        mean_switch_increase_percentage=('switch_increase_percentage', 'mean'),
    )
    summary = summary.join(correctness_means)
    summary['mean_focal_coverage_percentage'] = results.samples.groupby('method').focal_coverage_percentage.mean()
    summary.to_csv(output_dir / 'analysis_summary.csv')
    ipython_display.display(summary.rename(index=labels).round(2))

# %% [markdown]
# ## Per-sample data
#
# Each point represents one diploid sample. Focal coverage is the mean physical
# coverage of its two haplotypes over the analysed 1–31 Mb interval. Recorded
# worker time sums all instrumented stages, averages across trials for each
# haplotype, then averages the two haplotypes. Colours identify populations;
# marginal histograms show sample counts using the same population colours.
# Full HMM has no recorded worker-stage timings.
#
# CHB samples have lower mean focal coverage and longer recorded worker times
# at both cutoffs. The scatter shows an association between coverage and time;
# population membership alone does not explain the variation within populations.

# %%
if __name__ == '__main__':
    sample_data = results.samples
    sample_data.to_csv(output_dir / 'per_sample_matching_performance.csv', index=False)
    sample_style = style['per_sample']
    fig = plt.figure(figsize=(sample_style['figure_width'], sample_style['figure_height']),
                     layout='constrained')
    fig.suptitle('Per-sample matching performance')
    subfigures = fig.subfigures(1, len(stitched_methods))
    population_handles = []
    for method, subfigure in zip(stitched_methods, subfigures, strict=True):
        subfigure.suptitle(labels[method])
        ratio = sample_style['marginal_ratio']
        grid = subfigure.add_gridspec(2, 2, width_ratios=[ratio, 1],
                                     height_ratios=[1, ratio])
        ax = subfigure.add_subplot(grid[1, 0])
        top = subfigure.add_subplot(grid[0, 0], sharex=ax)
        right = subfigure.add_subplot(grid[1, 1], sharey=ax)
        table = sample_data.loc[sample_data.method.eq(method)]
        assert set(table.population) <= set(sample_style['population_colors'])
        x_bins = np.linspace(0, 100, sample_style['histogram_bins'] + 1)
        y_bins = np.linspace(0, table.worker_time_ms.max(),
                             sample_style['histogram_bins'] + 1)
        for population, color in sample_style['population_colors'].items():
            group = table.loc[table.population.eq(population)]
            points = ax.scatter(group.focal_coverage_percentage, group.worker_time_ms,
                                color=color, s=sample_style['marker_size'],
                                alpha=sample_style['alpha'], label=population)
            top.hist(group.focal_coverage_percentage, bins=x_bins, color=color,
                     alpha=sample_style['alpha'])
            right.hist(group.worker_time_ms, bins=y_bins, orientation='horizontal',
                       color=color, alpha=sample_style['alpha'])
            if method == stitched_methods[0]:
                population_handles.append(points)
        ax.set(xlabel='Focal coverage of sample (%)',
               ylabel='Recorded worker time per haplotype (ms)',
               xlim=(0, 100), ylim=(0, None))
        top.set(ylabel='Samples')
        right.set(xlabel='Samples')
        top.tick_params(labelbottom=False)
        right.tick_params(labelleft=False)
    fig.legend(handles=population_handles, title='Population',
               loc='outside lower center', ncol=len(population_handles))
    save_figure(fig, 'per_sample_matching_performance')
