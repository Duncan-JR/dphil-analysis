"""Only the recurrent-site and copying-event plots requested for pt12."""

import logging

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ConditioningPlots:
    """Draw and save the restricted pt12 figure set."""

    def __init__(self, config):
        self.config = config
        self.style = config["plot"]
        plt.rcParams.update({"font.size": self.style["font_size"]})

    def _panels(self, rows, columns, title):
        figure, axes = plt.subplots(
            rows, columns, squeeze=False,
            figsize=(columns * self.style["panel_width"], rows * self.style["panel_height"]),
            dpi=self.style["dpi"], layout="constrained",
        )
        figure.suptitle(f"{title}\n{self.style['provisional_note']}")
        return figure, axes

    def save(self, figure, name):
        directory = self.config["output_dir"] / "figures"
        directory.mkdir(exist_ok=True)
        figure.savefig(directory / f"{name}.png", dpi=self.style["export_dpi"])
        figure.savefig(directory / f"{name}.pdf")
        return figure

    def recurrence(self, table):
        """Reproduce pt11's two recurrent-site quantities for each dataset."""
        figure, axes = self._panels(len(self.config["datasets"]), 2, "Recurrent mutations at inference sites")
        methods = {"full_hmm": ("Full HMM", self.style["full_color"]),
                   "greedy": ("Greedy\nconditioned gaps", self.style["greedy_color"]),
                   "truth": ("Truth", self.style["truth_color"])}
        metrics = {"recurrent_sites": "Number of recurrent sites",
                   "mean_mutations_per_recurrent_site": "Mean mutation records per recurrent site"}
        for row, dataset in enumerate(self.config["datasets"]):
            selected = table.loc[table.dataset.eq(dataset["name"])]
            labels = [methods[value][0] for value in selected.method]
            colors = [methods[value][1] for value in selected.method]
            for ax, (metric, ylabel) in zip(axes[row], metrics.items(), strict=True):
                bars = ax.bar(labels, selected[metric], color=colors, width=self.style["bar_width"])
                if metric == "recurrent_sites":
                    annotations = [f"{count:,}\n({percentage:.1f}%)" for count, percentage in zip(
                        selected.recurrent_sites, selected.recurrent_site_percentage, strict=True)]
                else:
                    annotations = [f"{value:.2f}" if np.isfinite(value) else "undefined"
                                   for value in selected[metric]]
                ax.bar_label(bars, labels=annotations, padding=4)
                ax.margins(y=self.style["bar_headroom_fraction"])
                ax.set(title=dataset["label"], ylabel=ylabel, ylim=(0, None))
        return figure

    def error_sites(self, table):
        """Show exact set counts; circle areas are deliberately not proportional."""
        figure, axes = self._panels(1, len(table), "Simulation: recurrent sites and injected genotype errors")
        for ax, (_, record) in zip(axes.flat, table.iterrows(), strict=True):
            ax.add_patch(matplotlib.patches.Circle((0.38, 0.54), 0.26, facecolor=self.style["greedy_color"], alpha=0.3))
            ax.add_patch(matplotlib.patches.Circle((0.67, 0.54), 0.26, facecolor=self.style["error_color"], alpha=0.3))
            ax.text(0.24, 0.54, f"{record.recurrent_only_sites:,}", ha="center", va="center")
            ax.text(0.525, 0.54, f"{record.overlap_sites:,}", ha="center", va="center")
            ax.text(0.80, 0.54, f"{record.error_only_sites:,}", ha="center", va="center")
            ax.text(0.31, 0.86, "Recurrent sites", ha="center")
            ax.text(0.75, 0.86, "Sites with ≥1 error", ha="center")
            total = record.recurrent_only_sites + record.overlap_sites
            fraction = 100 * record.overlap_sites / total if total > 0 else np.nan
            ax.text(0.52, 0.19, f"{fraction:.1f}% of recurrent sites contain an injected error\n"
                    f"Neither: {record.neither_sites:,} of {record.num_sites:,} inference sites\n"
                    "Schematic areas; counts are exact", ha="center", va="top", fontsize="small")
            label = "Full HMM" if record.method == "full_hmm" else "Greedy (conditioned gaps)"
            ax.set(title=label, xlim=(0, 1.05), ylim=(0, 1), aspect="equal")
            ax.axis("off")
        return figure

    def recurrent_mutation_errors(self, table):
        """Keep unassignable reference mutations separate from sample-call overlap."""
        figure, axes = self._panels(1, 1, "Simulation: mutation records at recurrent sites")
        ax = axes[0, 0]
        labels = ["Full HMM" if method == "full_hmm" else "Greedy\nconditioned gaps" for method in table.method]
        categories = {
            "recurrent_sample_mutations_at_error_calls": ("Sample mutation at an injected-error call", self.style["error_color"]),
            "recurrent_sample_mutations_at_other_calls": ("Sample mutation without an injected genotype error", self.style["other_color"]),
            "recurrent_reference_mutations": ("Reference mutation: no individual call assignment", self.style["truth_color"]),
        }
        bottom = np.zeros(len(table))
        for column, (label, color) in categories.items():
            heights = table[column].to_numpy()
            bars = ax.bar(labels, heights, bottom=bottom, width=self.style["bar_width"], label=label, color=color)
            annotations = [f"{value:,}" if value > 0 else "" for value in heights]
            ax.bar_label(bars, labels=annotations, label_type="center", fontsize="small")
            bottom += heights
        ax.set(ylabel="Mutation records", ylim=(0, None))
        figure.legend(loc="outside lower center", fontsize="small")
        return figure

    def copying_events(self, table):
        """Stack mean switches above mean sample mutations, per diploid person."""
        figure, axes = self._panels(1, len(self.config["datasets"]), "Copying events per diploid sample")
        for ax, dataset in zip(axes.flat, self.config["datasets"], strict=True):
            selected = table.loc[table.dataset.eq(dataset["name"])]
            labels = ["Full HMM" if value == "full_hmm" else "Greedy\nconditioned gaps" for value in selected.method]
            bottom = np.zeros(len(selected))
            for metric, label, color in (("mutations", "Sample mutations", self.style["greedy_color"]),
                                         ("switches", "Copying switches", self.style["switch_color"])):
                values = selected[metric].to_numpy()
                bars = ax.bar(labels, values, bottom=bottom, color=color,
                              width=self.style["bar_width"], label=label)
                ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], label_type="center")
                bottom += values
            ax.set(title=dataset["label"], ylabel="Mean count per diploid individual", ylim=(0, None))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=2)
        return figure
