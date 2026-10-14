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
# # Focal stitching: correctness and performance
#
# `first_perf_benchmark` · 300 diploid individuals · chr20, 1–31 Mb ·
# AC ≤ 3 · exact focal intervals · 8 matching workers · 3 timing repeats.

# %%
import logging
import pathlib

import ch5_hyperparams
import ch5_perf
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
config_path = repository_dir / 'experiments/ch5/stitching_pt4_perf_config.yaml'
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
    results = ch5_perf.load_results(config_path)
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
    """Use opaque, dark patterns and optional top-to-bottom stack ordering."""
    legend_style = style['phase_legend']
    if order is None:
        order = list(settings)
    handles = []
    for phase in order:
        values = settings[phase]
        handle = mpatches.Patch(facecolor=legend_style['facecolor'],
                               edgecolor=legend_style['edgecolor'], hatch=values['hatch'],
                               label=values['label'])
        handles.append(handle)
    ax.legend(handles=handles, title=title, fontsize=legend_style['font_size'],
              handlelength=legend_style['handlelength'],
              handleheight=legend_style['handleheight'],
              labelspacing=legend_style['labelspacing'], **kwargs)


def trial_dots(ax, x, values, color):
    """Show every timing repeat rather than an inferred confidence interval."""
    offsets = np.asarray(style['repeat_offsets'])
    assert len(values) == len(offsets)
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
# repeats within each method. Switch categories use the same AC 3 focal mask.
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
    baseline = comparisons.loc[comparisons.method.eq(stitched_methods[0])]
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
# Bars show means across three trials; dots show individual trials.

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
# Resident totals include simplification and setup. Wall time per haplotype
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
    phase_legend(ax, style['memory_modes'])

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
