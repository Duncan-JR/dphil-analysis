"""Shared sample-size and cohort-panel figures for the HPC archives."""

import ch5_perf_scaling
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats


class ScalingPlots:
    """Keep cohort order, AF/AC labels and figure geometry consistent."""

    def __init__(self, config: dict):
        self.config = config
        self.style = config['plot']
        self.cohorts = ch5_perf_scaling.cohort_table(config)
        by_name = {dataset['name']: dataset for dataset in config['manifest']['datasets']}
        self.datasets = [by_name[name] for name in self.cohorts.dataset]
        self.sample_sizes = self.cohorts.num_samples.to_numpy()
        self.cutoffs = sorted({run['af_cutoff'] for dataset in self.datasets for run in dataset['runs']})
        self.colors = dict(zip(self.cutoffs, self.style['cutoff_colors'], strict=True))
        plt.rcParams.update({'figure.dpi': self.style['dpi'], 'font.size': self.style['font_size'],
                             'axes.spines.top': False, 'axes.spines.right': False})

    def cutoff_label(self, cutoff: float, samples: int | None = None) -> str:
        sizes = self.sample_sizes if samples is None else [samples]
        counts = [str(round(2 * int(size) * cutoff)) for size in sizes]
        return f"AF ≤ {cutoff:g} [AC {' / '.join(counts)}]"

    def panels(self, rows=1, columns=2, title=None):
        figure, axes = plt.subplots(rows, columns, squeeze=False, layout='constrained',
                                    figsize=(columns * self.style['panel_width'],
                                             rows * self.style['panel_height']))
        if title is not None:
            figure.suptitle(title)
        for ax in axes.flat:
            ax.tick_params(axis='y', labelleft=True)
        return figure, axes

    def sample_axis(self, ax):
        ax.set_xscale('log')
        ax.set_xticks(self.sample_sizes, labels=[f'{value:,}' for value in self.sample_sizes])
        ax.minorticks_off()
        ax.set_xlabel('Samples (diploid individuals)')

    def zoom_y(self, axes, lower=None, upper=None):
        axes = list(axes)
        lows, highs = zip(*(ax.get_ylim() for ax in axes), strict=True)
        minimum = min(lows)
        maximum = max(highs)
        if lower is not None:
            minimum = max(lower, minimum)
        if upper is not None:
            maximum = min(upper, maximum)
        for ax in axes:
            ax.set_ylim(minimum, maximum)

    def save(self, figure, name):
        root = self.config['cache_dir']
        figure.savefig(root / f'{name}.png', dpi=self.style['export_dpi'], bbox_inches='tight')
        figure.savefig(root / f'{name}.pdf', bbox_inches='tight')
        plt.show()
        plt.close(figure)

    def scaling(self, table, metrics, title, include_full=False, sem=False,
                reference=None, log_y=False, zoom=False):
        figure, axes = self.panels(columns=len(metrics), title=title)
        for ax, (metric, ylabel) in zip(axes.flat, metrics.items(), strict=True):
            curves = [(cutoff, self.colors[cutoff], self.cutoff_label(cutoff)) for cutoff in self.cutoffs]
            if include_full:
                curves.insert(0, (None, self.style['full_color'], 'Full HMM'))
            for cutoff, color, label in curves:
                selected = table.method.eq('full_hmm') if cutoff is None else table.af_cutoff.eq(cutoff)
                curve = table.loc[selected].sort_values('num_samples')
                yerr = curve[metric.removesuffix('_mean') + '_sem'] if sem else None
                ax.errorbar(curve.num_samples, curve[metric], yerr=yerr, color=color,
                            marker='o', markersize=self.style['marker_size'],
                            linewidth=self.style['line_width'], capsize=3, label=label)
            self.sample_axis(ax)
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale('log')
            elif zoom:
                ax.margins(y=self.style['range_margin'])
            elif not zoom:
                ax.set_ylim(bottom=0)
            if reference is not None:
                ax.axhline(reference, color='gray', linestyle='--', linewidth=1)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize='small')
        return figure

    def coverage(self, table):
        figure, axes = self.panels(columns=len(self.cutoffs), title='Physical focal coverage by population')
        for ax, cutoff in zip(axes.flat, self.cutoffs, strict=True):
            for population, color in self.style['population_colors'].items():
                curve = table.loc[table.af_cutoff.eq(cutoff) & table.population_group.eq(population)]
                curve = curve.sort_values('num_samples')
                ax.errorbar(curve.num_samples, curve['mean'], yerr=curve['sem'], color=color,
                            label=population, marker='o', capsize=3)
            self.sample_axis(ax)
            ax.set(title=self.cutoff_label(cutoff), ylabel='Physical focal coverage (%)')
            ax.margins(y=self.style['range_margin'])
            ax.legend()
        self.zoom_y(axes.flat)
        return figure

    def focal_lengths(self, lengths, unit, num_bins):
        figure, axes = self.panels(rows=len(self.cutoffs), columns=len(self.datasets),
                                   title=f'Retained focal lengths · {unit}')
        source = lengths.loc[~lengths.is_gap & lengths.unit.eq(unit)]
        for row, cutoff in enumerate(self.cutoffs):
            pooled = source.loc[source.af_cutoff.eq(cutoff)]
            upper = ch5_perf_scaling.weighted_quantile(pooled, self.style['length_display_quantile'])
            bins = np.linspace(0, upper, num_bins + 1)
            grid = np.linspace(0, upper, self.style['kde_grid_size'])
            for col, dataset in enumerate(self.datasets):
                ax = axes[row, col]
                samples = self.sample_sizes[col]
                values = pooled.loc[pooled.dataset.eq(dataset['name'])]
                total = values['count'].sum()
                tail = values.loc[values.length.gt(upper), 'count'].sum() / total
                ax.hist(values.length, bins=bins, weights=values['count'], color=self.colors[cutoff],
                        alpha=self.style['histogram_alpha'], label=f'{total:,} regions; tail {tail:.1%}')
                density = ch5_perf_scaling.length_density(values, grid)
                ax.plot(grid, density * total * (bins[1] - bins[0]), color=self.colors[cutoff])
                minimum = dataset['runs'][0]['min_focal_region_size'] if unit == 'sites' else dataset['runs'][0]['min_region_size_kb']
                label = f'{minimum:,} site minimum' if unit == 'sites' else f'{minimum:g} kb mean equivalent'
                ax.axvline(minimum, linestyle='--', color='gray', label=label)
                ax.set(title=f'n = {samples:,} · {self.cutoff_label(cutoff, samples)}',
                       xlabel=f'Focal length ({unit})', ylabel='Count per bin', xlim=(0, upper), ylim=(0, None))
                ax.legend(fontsize='small')
        return figure

    def gap_lengths(self, lengths):
        figure, axes = self.panels(columns=len(self.datasets), title='Gap length distributions')
        source = lengths.loc[lengths.is_gap & lengths.unit.eq('kb')]
        upper = ch5_perf_scaling.weighted_quantile(source, self.style['length_display_quantile'])
        grid = np.linspace(0, upper, self.style['kde_grid_size'])
        for col, dataset in enumerate(self.datasets):
            ax = axes[0, col]
            for cutoff, color in self.colors.items():
                values = source.loc[source.dataset.eq(dataset['name']) & source.af_cutoff.eq(cutoff)]
                median = ch5_perf_scaling.weighted_quantile(values, 0.5)
                density = ch5_perf_scaling.length_density(values, grid)
                tail = values.loc[values.length.gt(upper), 'count'].sum() / values['count'].sum()
                label = f'{self.cutoff_label(cutoff, self.sample_sizes[col])}\nmedian {median:.1f} kb; tail {tail:.1%}'
                ax.plot(grid, density, color=color, label=label)
                ax.axvline(median, color=color, linestyle='--', linewidth=self.style['line_width'])
            ax.set(title=f'n = {self.sample_sizes[col]:,}', xlabel='Gap length (kb)',
                   ylabel='Density per kb', xlim=(0, upper), ylim=(0, None))
            ax.legend(fontsize='small')
        self.zoom_y(axes.flat, lower=0)
        return figure

    def one_site_gaps(self, lengths):
        """Count one-inference-site gaps, annotating their share of all gaps."""
        gaps = lengths.loc[lengths.is_gap & lengths.unit.eq('sites')]
        keys = ['dataset', 'af_cutoff']
        totals = gaps.groupby(keys)['count'].sum().rename('total_gaps')
        singles = gaps.loc[gaps.length.eq(1)].groupby(keys)['count'].sum()
        singles = singles.reindex(totals.index, fill_value=0).rename('one_site_gaps')
        counts = pd.concat([totals, singles], axis=1).reset_index()
        counts['one_site_percentage'] = 100 * counts.one_site_gaps / counts.total_gaps
        counts.to_csv(self.config['cache_dir'] / 'one_site_gap_counts.csv', index=False)
        figure, axes = self.panels(columns=len(self.datasets),
                                   title='Number of one-site gaps · labels show percentage of all gaps')
        for col, dataset in enumerate(self.datasets):
            ax = axes[0, col]
            samples = self.sample_sizes[col]
            table = counts.loc[counts.dataset.eq(dataset['name'])].set_index('af_cutoff').loc[self.cutoffs]
            bars = ax.bar(np.arange(len(self.cutoffs)), table.one_site_gaps,
                          color=[self.colors[cutoff] for cutoff in self.cutoffs],
                          width=self.style['bar_width'])
            ax.bar_label(bars, labels=[f'{value:.1f}%' for value in table.one_site_percentage], padding=3)
            minimum_kb = dataset['runs'][0]['min_region_size_kb']
            minimum_sites = dataset['runs'][0]['min_focal_region_size']
            labels = [self.cutoff_label(cutoff, samples) for cutoff in self.cutoffs]
            ax.set(title=f'n = {samples:,} · minimum {minimum_kb:g} kb ({minimum_sites:,} sites)',
                   xticks=np.arange(len(self.cutoffs)), xticklabels=labels,
                   xlabel='AF cutoff', ylabel='Number of one-site gaps')
            ax.ticklabel_format(axis='y', style='plain')
            ax.margins(y=self.style['range_margin'] * 2)
        self.zoom_y(axes.flat, lower=0)
        return figure

    def switch_composition(self, summary):
        figure, axes = self.panels(rows=len(self.cutoffs), columns=len(self.datasets),
                                   title='Switch composition · each AF uses its own focal mask')
        for row, cutoff in enumerate(self.cutoffs):
            for col, dataset in enumerate(self.datasets):
                ax = axes[row, col]
                values = summary.loc[summary.dataset.eq(dataset['name']) & summary.af_cutoff.eq(cutoff)].iloc[0]
                bottom = np.zeros(2)
                for category, color, label in zip(self.style['switch_categories'], self.style['switch_colors'],
                                                   self.style['switch_labels'], strict=True):
                    height = np.array([values[f'full_{category}_switches_mean'], values[f'stitched_{category}_switches_mean']])
                    ax.bar([0, 1], height, bottom=bottom, color=color, label=label, width=self.style['bar_width'])
                    bottom += height
                ax.set(title=f'n = {self.sample_sizes[col]:,} · {self.cutoff_label(cutoff, self.sample_sizes[col])}',
                       xticks=[0, 1], xticklabels=['Full HMM', 'Stitched'], ylabel='Mean switches per haplotype')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', ncol=len(labels))
        return figure

    def gap_parents(self, table, weight):
        """Compare parent categories in identical gaps for the two matchers."""
        methods = {'full_hmm': 'Full HMM', 'greedy': 'Stitched'}
        ylabel = self.style['gap_parent_weightings'][weight]
        figure, axes = self.panels(rows=len(self.cutoffs), columns=len(methods),
                                   title=f'Parents used in gaps · {ylabel}')
        x = np.arange(len(self.sample_sizes))
        for row, cutoff in enumerate(self.cutoffs):
            for col, (method, method_label) in enumerate(methods.items()):
                ax = axes[row, col]
                bottom = np.zeros(len(x))
                for category, settings in self.style['gap_parent_categories'].items():
                    curve = table.loc[table.method.eq(method) & table.af_cutoff.eq(cutoff)
                                      & table.category.eq(category)].set_index('num_samples')
                    heights = curve.loc[self.sample_sizes, f'{weight}_percentage_mean'].to_numpy()
                    ax.bar(x, heights, bottom=bottom, color=settings['color'],
                           width=self.style['bar_width'], label=settings['label'])
                    for index, height in enumerate(heights):
                        if height >= self.style['gap_parent_label_minimum_percentage']:
                            ax.text(x[index], bottom[index] + height / 2, f'{height:.1f}%',
                                    ha='center', va='center', fontsize='small')
                    bottom += heights
                np.testing.assert_allclose(bottom, 100)
                ax.set(title=f'{method_label} · {self.cutoff_label(cutoff)}',
                       xticks=x, xticklabels=[f'{n:,}' for n in self.sample_sizes],
                       xlabel='Samples (diploid individuals)', ylabel='Mean within-haplotype share (%)',
                       ylim=(0, 100), yticks=np.arange(0, 101, 20))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize='small')
        return figure

    def extra_switch_distributions(self):
        path = self.config['data_dir'] / self.config['manifest']['tables']['haplotype_results']
        metrics = {'num_extra_switches': 'Extra switches vs Full HMM',
                   'switch_increase_percentage': 'Increase in switches (%)'}
        table = pd.read_csv(path, usecols=['dataset', 'af_cutoff', *metrics])
        figure, axes = self.panels(rows=len(metrics), columns=len(self.datasets), title='Extra-switch distributions')
        for row, (metric, label) in enumerate(metrics.items()):
            for col, dataset in enumerate(self.datasets):
                ax = axes[row, col]
                for cutoff, color in self.colors.items():
                    selected = table.dataset.eq(dataset['name']) & table.af_cutoff.eq(cutoff)
                    values = table.loc[selected, metric].dropna().to_numpy()
                    if np.ptp(values) == 0:
                        ax.axvline(values[0], color=color, label=f'{self.cutoff_label(cutoff, self.sample_sizes[col])}: constant')
                    else:
                        grid = np.linspace(values.min(), values.max(), self.style['kde_grid_size'])
                        ax.plot(grid, stats_density(values, grid), color=color,
                                label=self.cutoff_label(cutoff, self.sample_sizes[col]))
                ax.set(title=f'n = {self.sample_sizes[col]:,}', xlabel=label, ylabel='Density', ylim=(0, None))
                ax.legend(fontsize='small')
        return figure

    def age_correlations(self, table, frequency_bin='all', metrics=None):
        if metrics is None:
            metrics = {'pearson_r': 'Pearson r',
                       'median_absolute_error_generations': 'Median absolute error (generations)'}
        references = ['truth', 'full_hmm'] if self.cohorts.has_truth.all() else ['full_hmm']
        figure, axes = self.panels(rows=len(references), columns=len(metrics),
                                   title=f'Mutation ages · {frequency_bin}')
        for row, reference in enumerate(references):
            selected = table.loc[table.reference_method.eq(reference) & table.frequency_bin.eq(frequency_bin)]
            methods = [(cutoff, self.colors[cutoff], self.cutoff_label(cutoff)) for cutoff in self.cutoffs]
            if reference == 'truth':
                methods.insert(0, (None, self.style['full_color'], 'Full HMM'))
            for col, (metric, ylabel) in enumerate(metrics.items()):
                ax = axes[row, col]
                for cutoff, color, label in methods:
                    method_mask = selected.method.eq('full_hmm') if cutoff is None else selected.af_cutoff.eq(cutoff)
                    curve = selected.loc[method_mask].sort_values('num_samples')
                    ax.plot(curve.num_samples, curve[metric], marker='o', color=color, label=label)
                self.sample_axis(ax)
                reference_label = 'Truth' if reference == 'truth' else 'Full HMM'
                ax.set(title=f'Compared with {reference_label}', ylabel=ylabel)
                if metric in ['pearson_r', 'spearman_r']:
                    low, high = ax.get_ylim()
                    ax.set_ylim(max(-1, low), min(1, high))
                else:
                    ax.set_ylim(bottom=0)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize='small')
        return figure

    def age_density(self, correlations, reference):
        cases = [None, *self.cutoffs] if reference == 'truth' else self.cutoffs
        reference_label = 'True' if reference == 'truth' else 'Full HMM'
        figure, axes = self.panels(rows=len(cases), columns=len(self.datasets),
                                   title=f'Mutation-age pairs · compared with {reference_label} ages')
        tables = []
        maximum = self.style['age_minimum']
        for dataset in self.datasets:
            path = self.config['cache_dir'] / 'age_pairs' / f"{dataset['name']}.parquet"
            stored = pl.read_parquet(path)
            table = pd.DataFrame({name: stored[name].to_numpy() for name in stored.columns})
            tables.append(table)
            columns = [reference, 'full_hmm', *[f'af_{cutoff}' for cutoff in self.cutoffs]]
            quantiles = table[columns].quantile(self.style['age_maximum_quantile'])
            maximum = max(maximum, quantiles.max())
        minimum = self.style['age_minimum']
        extent = (np.log10(minimum), np.log10(maximum), np.log10(minimum), np.log10(maximum))
        collections = []
        for row, cutoff in enumerate(cases):
            key = 'full_hmm' if cutoff is None else f'af_{cutoff}'
            for col, dataset in enumerate(self.datasets):
                ax = axes[row, col]
                table = tables[col]
                selected = table[reference].ge(minimum) & table[key].ge(minimum)
                drawn = table.loc[selected]
                artist = ax.hexbin(drawn[reference], drawn[key], xscale='log', yscale='log', extent=extent,
                                   gridsize=self.style['age_hexbin_gridsize'], mincnt=1, bins='log', cmap='viridis')
                collections.append(artist)
                ax.plot([minimum, maximum], [minimum, maximum], color='red', linestyle='--', linewidth=1)
                method = 'Full HMM' if cutoff is None else self.cutoff_label(cutoff, self.sample_sizes[col])
                ax.set(title=f'n = {self.sample_sizes[col]:,} · {method}', xlim=(minimum, maximum),
                       ylim=(minimum, maximum), xlabel=f'{reference_label} mutation age (generations)',
                       ylabel='Estimated mutation age (generations)')
                scores = correlations.loc[correlations.dataset.eq(dataset['name'])
                                          & correlations.reference_method.eq(reference)
                                          & correlations.frequency_bin.eq('all')]
                scores = scores.loc[scores.method.eq('full_hmm')] if cutoff is None else scores.loc[scores.af_cutoff.eq(cutoff)]
                result = scores.iloc[0]
                ax.text(0.04, 0.96, f'r = {result.pearson_r:.3f}\nN = {result.matched_mutations:,}',
                        transform=ax.transAxes, va='top', bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'})
        upper = max(artist.get_array().max() for artist in collections)
        for artist in collections:
            artist.set_norm(mcolors.LogNorm(vmin=1, vmax=upper))
        figure.colorbar(collections[0], ax=axes, label='Mutations per hexagon', shrink=0.8)
        return figure


def stats_density(values, grid):
    """Evaluate unreflected KDE for signed extra-switch distributions."""
    counts = pd.Series(values).value_counts().sort_index()
    table = pd.DataFrame({'length': counts.index, 'count': counts.to_numpy()})
    # The switch distribution is signed: reflection would alter its meaning.
    weights = table['count'].to_numpy(dtype=float)
    observations = table.length.to_numpy(dtype=float)
    total = weights.sum()
    mean = np.average(observations, weights=weights)
    variance = np.sum(weights * (observations - mean) ** 2) / (total - 1)
    bandwidth = np.sqrt(variance) * total ** (-1 / 5)
    weighted_variance = np.cov(observations, aweights=weights).item()
    kde = stats.gaussian_kde(observations, weights=weights,
                             bw_method=bandwidth / np.sqrt(weighted_variance))
    return kde(grid)
