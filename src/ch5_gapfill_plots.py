"""Compact hpc_perf3 figures, extending :mod:`ch5_perf_scaling_plots`."""

import logging

import ch5_perf_scaling_plots
import numpy as np

logger = logging.getLogger(__name__)


class GapfillPlots(ch5_perf_scaling_plots.ScalingPlots):
    """Keep the pt9 analyses with compact layouts for three cohorts."""

    def panels(self, rows=1, columns=2, title=None):
        figure, axes = super().panels(rows=rows, columns=columns, title=title)
        figure.get_layout_engine().set(wspace=self.style['panel_wspace'])
        return figure, axes

    def bar_axis(self, ax, ylabel):
        ax.set(xticks=np.arange(len(self.sample_sizes)),
               xticklabels=[f'{n:,}' for n in self.sample_sizes],
               xlabel='Samples (diploid individuals)', ylabel=ylabel, ylim=(0, None))

    def stage_stack(self, ax, table, metric, settings):
        bottom = np.zeros(len(self.sample_sizes))
        for stage, style in settings.items():
            selected = table.loc[table.stage.eq(stage)].set_index('num_samples')
            heights = selected.loc[self.sample_sizes, metric].to_numpy()
            ax.bar(np.arange(len(bottom)), heights, bottom=bottom,
                   color=style['color'], label=style['label'], width=self.style['bar_width'])
            bottom += heights
        ax.set_ylim(0, bottom.max() * (1 + self.style['stage_label_margin']))
        return bottom

    def wall_stages(self, breakdown):
        figure, axes = self.panels(columns=1, title='Greedy pipeline · sequential wall-time stages')
        figure.set_size_inches(self.style['panel_width'] * 2, self.style['panel_height'])
        ax = axes[0, 0]
        bottom = self.stage_stack(ax, breakdown.wall, 'wall_s', self.style['wall_stages'])
        self.bar_axis(ax, 'End-to-end wall time (s)')
        for x, value in enumerate(bottom):
            ax.annotate(f'{value:.1f} s', (x, value), xytext=(0, 4),
                        textcoords='offset points', ha='center')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
        return figure

    def worker_stages(self, breakdown):
        figure, axes = self.panels(columns=2, title='Greedy worker-stage time · mean per matched haplotype')
        metrics = {'service_s': 'Worker elapsed time\nper haplotype (ms)',
                   'thread_cpu_s': 'Worker CPU time\nper haplotype (ms)'}
        for ax, (metric, label) in zip(axes.flat, metrics.items(), strict=True):
            selected = breakdown.worker.loc[breakdown.worker.metric.eq(metric)]
            bottom = self.stage_stack(ax, selected, 'ms_per_haplotype', self.style['worker_stages'])
            self.bar_axis(ax, label)
            ax.set_title('Worker elapsed time' if metric == 'service_s' else 'Worker CPU time')
            for x, value in enumerate(bottom):
                ax.annotate(f'{value:.1f} ms', (x, value), xytext=(0, 4),
                            textcoords='offset points', ha='center')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', ncol=2, fontsize='small')
        return figure

    def recurrence(self, table):
        figure, axes = self.panels(columns=2, title='Recurrent mutations at saved inference sites')
        cases = [('truth', None, 'Truth', self.style['truth_color']),
                 ('full_hmm', None, 'Full HMM', self.style['full_color'])]
        for cutoff in self.cutoffs:
            cases.append(('greedy', cutoff, 'Stitched', self.colors[cutoff]))
        metrics = {'recurrent_sites': 'Number of recurrent sites',
                   'mean_mutations_per_recurrent_site': 'Mean mutations per recurrent site'}
        width = self.style['bar_width'] / len(cases)
        x = np.arange(len(self.sample_sizes))
        for ax, (metric, ylabel) in zip(axes.flat, metrics.items(), strict=True):
            for index, (method, cutoff, label, color) in enumerate(cases):
                selected = table.method.eq(method)
                if cutoff is not None:
                    selected &= table.af_cutoff.eq(cutoff)
                values = table.loc[selected].set_index('num_samples').loc[self.sample_sizes]
                positions = x + (index - (len(cases) - 1) / 2) * width
                bars = ax.bar(positions, values[metric], width=width, color=color, label=label)
                if metric == 'recurrent_sites':
                    labels = [f'({value:.1f}%)' for value in values.recurrent_site_percentage]
                    ax.bar_label(bars, labels=labels, padding=3, fontsize='x-small', rotation=90)
            self.bar_axis(ax, ylabel)
            ax.margins(y=self.style['stage_label_margin'])
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize='small')
        return figure

    def root_totals(self, summary):
        """Group Full HMM and stitched stacks by size in two metric panels."""
        metrics = {'edge_count_per_haplotype': 'Root edges per haplotype',
                   'span_kb_per_haplotype': 'Root-copying span\n(kb per haplotype)'}
        figure, axes = self.panels(columns=2, title='Direct ultimate-root → sample copying')
        x = np.arange(len(self.sample_sizes))
        methods = ['full_hmm', 'greedy']
        width = self.style['grouped_bar_width']
        for ax, (metric, ylabel) in zip(axes.flat, metrics.items(), strict=True):
            ticks = []
            labels = []
            for index, method in enumerate(methods):
                positions = x + (index - (len(methods) - 1) / 2) * width
                bottom = np.zeros(len(x))
                for category, style in self.style['root_edge_categories'].items():
                    mask = summary.method.eq(method) & summary.category.eq(category)
                    selected = summary.loc[mask].set_index('num_samples')
                    heights = selected.loc[self.sample_sizes, metric].to_numpy()
                    ax.bar(positions, heights, bottom=bottom, color=style['color'],
                           width=width, label=style['label'] if index == 0 else None,
                           edgecolor='white', hatch='//' if index == 0 else None)
                    bottom += heights
                ticks.extend(positions)
                method_label = 'Full HMM' if method == 'full_hmm' else 'Stitched'
                labels.extend([f'{n:,}\n{method_label}' for n in self.sample_sizes])
            ax.set(xticks=ticks, xticklabels=labels, ylabel=ylabel,
                   xlabel='Samples (diploid individuals) and method', ylim=(0, None))
            ax.tick_params(axis='x', labelsize='x-small', rotation=30)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize='small')
        return figure
