"""Share the paired overall and frequency-stratified mutation-age plots."""

import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


def correlation_figure(results, config, experiment, split_frequency=False):
    """Place truth comparisons left and Full HMM comparisons right in each row."""
    settings = config['dating']['plot']
    style = config['plot']
    x_parameter = settings['x_parameter']
    curve_parameter = settings['curve_parameter']
    x_values = sorted(experiment[x_parameter])
    curve_values = sorted(experiment[curve_parameter])
    palette = style[settings['curve_colors']]
    colors = dict(zip(curve_values, palette, strict=True))
    groups = config['dating']['frequency_bins']['labels'] if split_frequency else ['all']
    fig, axes = plt.subplots(len(groups), 2, squeeze=False, layout='constrained',
                             figsize=(style['panel_width'] * 2, style['panel_height'] * len(groups)))
    for row, group in enumerate(groups):
        table = results.loc[results.frequency_bin.eq(group)]
        truth = table.loc[table.reference_method.eq('true_arg')]
        baseline = truth.loc[truth.comparison_method.eq('full_hmm'), 'pearson_r'].item()
        ax = axes[row, 0]
        for value in curve_values:
            observations = truth.loc[truth[curve_parameter].eq(value)].sort_values(x_parameter)
            y = np.concatenate(([baseline], observations.pearson_r.to_numpy()))
            ax.plot(np.arange(len(y)), y, color=colors[value],
                    label=settings['curve_label'].format(value=value),
                    marker='o', markersize=style['marker_size'], linewidth=style['line_width'])
        prefix = f'{group} · ' if split_frequency else ''
        ax.set(title=f'{prefix}Simulation: inferred vs true ages', ylabel='Pearson r',
               xlabel=settings['x_label'], xticks=np.arange(len(x_values) + 1),
               xticklabels=['Full HMM', *[str(value) for value in x_values]])
        ax.legend()

        ax = axes[row, 1]
        reference = table.loc[table.reference_method.eq('full_hmm')]
        for dataset in config['datasets']:
            name = dataset['name']
            if name not in config['dating']['compare_full_datasets']:
                continue
            for value in curve_values:
                selected = reference.dataset.eq(name) & reference[curve_parameter].eq(value)
                observations = reference.loc[selected].sort_values(x_parameter)
                label = settings['curve_label'].format(value=value)
                ax.plot(observations[x_parameter], observations.pearson_r,
                        color=colors[value], linestyle=config['dating']['dataset_linestyles'][name],
                        label=f"{dataset['label']} · {label}", marker='o',
                        markersize=style['marker_size'], linewidth=style['line_width'])
        ax.set(title=f'{prefix}Stitched vs Full HMM ages', ylabel='Pearson r',
               xlabel=settings['x_label'], xticks=x_values)
        ax.legend()
    title = 'Mutation-age correlations by derived allele frequency' if split_frequency else 'Mutation-age correlations'
    fig.suptitle(title)
    return fig


def stitched_age_density_figure(pairs, config, experiment, dataset):
    """Arrange Full HMM versus stitched densities by the two grid parameters."""
    settings = config['dating']['age_density']
    plot_settings = config['dating']['plot']
    column_parameter = plot_settings['x_parameter']
    row_parameter = plot_settings['curve_parameter']
    column_values = sorted(experiment[column_parameter])
    row_values = sorted(experiment[row_parameter])
    selected = pairs.loc[pairs.dataset.eq(dataset['name'])]
    lower = selected[['full_hmm_age', 'stitched_age']].min().min()
    upper = selected[['full_hmm_age', 'stitched_age']].max().max()
    if settings['age_scale'] == 'log':
        edges = np.geomspace(lower, upper, settings['bins'] + 1)
    else:
        edges = np.linspace(0, upper, settings['bins'] + 1)
    cases = []
    histograms = []
    for row_value in row_values:
        for column_value in column_values:
            mask = selected[row_parameter].eq(row_value)
            mask &= selected[column_parameter].eq(column_value)
            case = selected.loc[mask]
            counts, _, _ = np.histogram2d(case.full_hmm_age, case.stitched_age, bins=edges)
            cases.append(case)
            histograms.append(counts)
    maximum = max(counts.max() for counts in histograms)
    norm = mcolors.LogNorm(vmin=1, vmax=maximum)
    fig, axes = plt.subplots(len(row_values), len(column_values), squeeze=False,
                             sharex=True, sharey=True, layout='constrained',
                             figsize=(settings['panel_size'] * len(column_values),
                                      settings['panel_size'] * len(row_values)))
    for index, ax in enumerate(axes.flat):
        row, column = divmod(index, len(column_values))
        case = cases[index]
        counts = histograms[index]
        image = ax.pcolormesh(edges, edges, np.ma.masked_equal(counts.T, 0),
                             cmap=settings['cmap'], norm=norm, rasterized=True)
        ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]],
                color=settings['identity_color'], linestyle='--', linewidth=1)
        correlation = stats.pearsonr(case.full_hmm_age, case.stitched_age).statistic
        column_label = settings['column_label'].format(value=column_values[column])
        ax.set(title=f'{column_label}\nr = {correlation:.3f}; n = {len(case):,}',
               xscale=settings['age_scale'], yscale=settings['age_scale'],
               xlim=(edges[0], edges[-1]), ylim=(edges[0], edges[-1]), aspect='equal')
        if column == 0:
            row_label = plot_settings['curve_label'].format(value=row_values[row])
            ax.set_ylabel(f'{row_label}\nStitched mutation age (generations)')
        if row == len(row_values) - 1:
            ax.set_xlabel('Full HMM mutation age (generations)')
    fig.colorbar(image, ax=axes.ravel().tolist(), label='Mutations per bin (log colour scale)',
                 shrink=settings['colorbar_shrink'])
    scale = settings['age_scale']
    fig.suptitle(f"{dataset['label']}: Full HMM vs stitched mutation ages · all frequencies\n"
                 f'Common mutations, axes and colour scale · {scale} age axes')
    return fig
