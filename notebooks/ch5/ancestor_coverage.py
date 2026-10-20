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
# # Ancestor coverage: stdpopsim and 1KGP

# %%
import pathlib
import time

import ch5_analysis
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# %%
match_eval_dir = pathlib.Path("~/work/tsinfer-match-eval/data/match_eval").expanduser()
simulation_dir = pathlib.Path(
    "~/work/tsinfer-anc-eval/data/error_eval/simulated"
).expanduser()
dataset_names = {
    "stdpopsim": "out_of_africa_n100_1mbp",
    "1KGP": "tgp_chr20_n100_1mbp",
}
simulation_ts = simulation_dir / "OutOfAfrica_4J17-chr20-L0-R1e6-n100-s1-rep0.trees"

# Uncomment both alternatives together to run the 10 Mb analysis later.
dataset_names = {"stdpopsim": "out_of_africa_n100_10mbp", "1KGP": "tgp_chr20_n100"}
simulation_ts = simulation_dir / "OutOfAfrica_4J17-chr20-L0-R1e7-n100-s1-rep0.trees"

max_ac = 200
num_ac_bins = 20
mismatch_budgets = [0, 1]
histogram_ac_cutoffs = [3, 5, 10]
coverage_histogram_bins = np.linspace(0, 1, 21)
workers = None  # All available CPUs; use 1 to avoid process startup on small runs.
cutoffs = ch5_analysis.focal_ac_cutoffs(max_ac=max_ac, num_bins=num_ac_bins)
cutoffs = np.union1d(cutoffs, histogram_ac_cutoffs)
print("Maximum focal AC thresholds:", cutoffs)

# %% [markdown]
# ## Load raw focal-ancestor intervals
#
# Generation anchors each association at its ancestor's leftmost non-padding
# focal position, before comparison. Load exact AC and ragged interval arrays;
# the analysis applies inclusive AC cutoffs without choosing focal seeds.

# %%
coverage_inputs = {}
timings = []
for label, dataset in dataset_names.items():
    started = time.perf_counter()
    truth_path = simulation_ts if label == "stdpopsim" else None
    data = ch5_analysis.load_coverage_data(
        match_eval_dir,
        dataset,
        max_ac=max_ac,
        max_mismatches=mismatch_budgets,
        truth_ts_path=truth_path,
    )
    coverage_inputs[label] = data
    timings.append(
        {
            "dataset": label,
            "stage": "load_intervals",
            "seconds": time.perf_counter() - started,
        }
    )
    print(
        f"{label}: {data.num_sites:,} inference sites, "
        f"{data.haplotypes.height} haplotypes, "
        f"positions {data.first_position:,}–{data.last_position:,}, "
        f"{len(data.focal_ac):,} associations; budgets {data.mismatch_budgets}"
    )
    print(data.haplotypes.group_by("population").len().sort("population"))

# %% [markdown]
# ## Sweep and summarise coverage
#
# Half-open site intervals `[left_site_index, right_site_index)` contribute +1
# at their left endpoint and −1 at their right endpoint. The sweep groups equal
# endpoints and incrementally adds associations with `2 ≤ focal_ac ≤ cutoff`,
# retaining the existing chapter 5 AC minimum. Constant
# coverage segments are weighted by their number of sites, including zero
# coverage outside intervals. Each independent haplotype is a multiprocessing task.
#
# The result records mean, minimum, maximum, 5th and 95th percentiles, and the
# fraction of sites with coverage > 0 for every haplotype, AC cutoff and budget.
# Percentiles (rather than quartiles) use NumPy's usual linear interpolation
# across the complete conceptual per-site coverage vector. Endpoint histograms
# avoid materialising that vector. Minimum coverage is zero whenever any site
# has no candidate, even when the covered fraction is close to one.

# %%
coverage_by_budget = {}
for budget in mismatch_budgets:
    coverage_by_budget[budget] = {}
    for label, data in coverage_inputs.items():
        started = time.perf_counter()
        coverage_by_budget[budget][label] = ch5_analysis.summarise_coverage(
            data, cutoffs, max_mismatches=budget, workers=workers
        )
        timings.append(
            {
                "dataset": label,
                "stage": f"sweep_k{budget}",
                "seconds": time.perf_counter() - started,
            }
        )
print(pl.DataFrame(timings))
coverage_by_budget[0]["stdpopsim"].head()

# %%
population_means = {}
for budget, datasets in coverage_by_budget.items():
    population_means[budget] = {}
    for label, coverage in datasets.items():
        means = coverage.group_by("population", "focal_ac").agg(
            pl.col("fraction_covered").mean(),
            pl.col("mean_coverage").mean(),
            pl.len().alias("num_haplotypes"),
        )
        population_means[budget][label] = means.sort("population", "focal_ac")

# %%
population_colors = {
    "YRI": "tab:blue",
    "AFR": "tab:blue",
    "CEU": "tab:orange",
    "EUR": "tab:orange",
    "CHB": "tab:green",
    "EAS": "tab:green",
}


def plot_population_coverage(budget):
    """Plot population means in two rows, with simulation left and 1KGP right."""
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7), sharex=True, sharey="row", layout="constrained"
    )
    metrics = ["fraction_covered", "mean_coverage"]
    ylabels = ["Fraction of sites covered", "Mean site coverage"]
    for column, label in enumerate(dataset_names):
        means = population_means[budget][label]
        populations = means["population"].unique().sort().to_list()
        for row, (metric, ylabel) in enumerate(zip(metrics, ylabels, strict=True)):
            ax = axes[row, column]
            for population in populations:
                values = means.filter(pl.col("population") == population)
                ax.plot(
                    values["focal_ac"].to_numpy(),
                    values[metric].to_numpy(),
                    marker="o",
                    markersize=3,
                    label=population,
                    color=population_colors.get(population),
                )
            ax.set_xscale("log")
            ax.set_xticks([2, 3, 5, 10, 20, 50, 100, max_ac])
            ax.set_xticklabels(["2", "3", "5", "10", "20", "50", "100", str(max_ac)])
            ax.set_xlim(cutoffs[0], cutoffs[-1])
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Maximum focal AC")
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(label)
                ax.legend(title="Population")
    axes[0, 0].set_ylim(0, 1.02)
    axes[1, 0].set_ylim(bottom=0)
    fig.suptitle(f"Focal ancestor coverage: max_mismatches = {budget}")
    plt.show()
    return fig


# %% [markdown]
# ## max_mismatches = 0
#
# Exact-match intervals: fraction covered above, mean site coverage below.
# Coverage includes every inference site in the denominator.

# %%
figure_k0 = plot_population_coverage(0)

# %% [markdown]
# ## max_mismatches = 1
#
# Allow one mismatch on either side of each selected focal site, retaining the
# same ancestor/focal choice as for zero mismatches.

# %%
figure_k1 = plot_population_coverage(1)

# %% [markdown]
# ## Sample coverage histograms: max_mismatches = 0
#
#


# %%
def plot_population_coverage_histograms(budget):
    """Compare per-haplotype coverage distributions at each requested AC cutoff."""
    fig, axes = plt.subplots(
        len(histogram_ac_cutoffs),
        2,
        figsize=(12, 10),
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    for column, label in enumerate(dataset_names):
        coverage = coverage_by_budget[budget][label]
        populations = coverage["population"].unique().sort().to_list()
        for row, cutoff in enumerate(histogram_ac_cutoffs):
            ax = axes[row, column]
            selected = coverage.filter(pl.col("focal_ac") == cutoff)
            for population in populations:
                samples = selected.filter(pl.col("population") == population)
                fractions = samples["fraction_covered"].to_numpy()
                ax.hist(
                    fractions,
                    bins=coverage_histogram_bins,
                    histtype="step",
                    linewidth=1.7,
                    label=population,
                    color=population_colors.get(population),
                )
            ax.set_title(f"{label}: maximum focal AC ≤ {cutoff}")
            ax.set_xlabel("Fraction of sites covered")
            ax.set_ylabel("Number of haplotypes")
            ax.set_xlim(0, 1)
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.25)
            ax.legend(title="Population")
    fig.suptitle(f"Sample coverage distributions: max_mismatches = {budget}")
    plt.show()
    return fig


# %%
histogram_figure_k0 = plot_population_coverage_histograms(0)

# %% [markdown]
# ## Sample coverage histograms: max_mismatches = 1
#
# The same cutoffs, bins and population colours, allowing one mismatch per
# side of the selected focal site.

# %%
histogram_figure_k1 = plot_population_coverage_histograms(1)

# %% [markdown]
# ## Mean proportion of ancestors included
#

# %%
ancestor_set_sizes = {}
for label, dataset in dataset_names.items():
    focal_path = (
        match_eval_dir / "focal_ancestors" / f"{dataset}_inferred_focal_ancestors.npz"
    )
    ancestor_set_sizes[label] = ch5_analysis.summarise_focal_ancestor_set_sizes(
        focal_path, cutoffs
    )

# %%
ancestor_set_figure, axes = plt.subplots(
    1, 2, figsize=(12, 4), sharex=True, sharey=True, layout="constrained"
)
for ax, label in zip(axes, dataset_names, strict=True):
    sizes = ancestor_set_sizes[label]
    ax.plot(
        sizes["focal_ac"].to_numpy(),
        sizes["mean_proportion_included"].to_numpy(),
        marker="o",
        markersize=3,
    )
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20, 50, 100, max_ac])
    ax.set_xticklabels(["2", "3", "5", "10", "20", "50", "100", str(max_ac)])
    ax.set_xlim(cutoffs[0], cutoffs[-1])
    ax.set_xlabel("Maximum focal AC")
    ax.set_ylabel("Mean proportion of ancestors included")
    ax.set_title(label)
    ax.grid(alpha=0.25)
axes[0].set_ylim(bottom=0)
plt.show()

# %%

# %%

# %%
