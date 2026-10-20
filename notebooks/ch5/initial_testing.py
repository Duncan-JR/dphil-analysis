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
#     display_name: dphil-analysis
#     language: python
#     name: dphil_analysis
# ---

# %% [markdown]
# # Initial testing: stdpopsim and 1KGP
#
# Compare the inferred panels for the two 100-individual, 10 Mb datasets.
# All figures place stdpopsim on the left and 1KGP on the right. Population
# labels are read directly from the pipeline's annotated statistics CSVs.

# %%
import pathlib

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axes_grid1
import numpy as np
import pandas as pd
import scipy.stats

# %%
match_eval_dir = pathlib.Path("~/work/tsinfer-match-eval/data/match_eval").expanduser()
dataset_names = {
    "stdpopsim": "out_of_africa_n100_10mbp",
    "1KGP": "tgp_chr20_n100",
}
focal_stats_by_dataset = {}
focal_ancestors_by_dataset = {}
haplotype_counts = {}
for label, dataset in dataset_names.items():
    stats_path = (
        match_eval_dir / "dataframes" / f"{dataset}_inferred_focal_ancestor_stats.csv"
    )
    stats = pd.read_csv(stats_path, dtype={"sample_id": str, "population": str})
    if "population" not in stats.columns or stats["population"].isna().any():
        raise ValueError(
            f"{stats_path} must contain pipeline-annotated population labels"
        )
    focal_stats_by_dataset[label] = stats
    focal_path = (
        match_eval_dir / "focal_ancestors" / f"{dataset}_inferred_focal_ancestors.npz"
    )
    with np.load(focal_path, allow_pickle=False) as focal_data:
        focal_ancestors_by_dataset[label] = pd.DataFrame(
            {
                "ancestor_id": focal_data["ancestor_id"],
                "derived_ac": focal_data["derived_ac"],
            }
        )
        haplotype_counts[label] = len(focal_data["sample_id"])

# %%
ac_cutoffs = sorted(focal_stats_by_dataset["stdpopsim"]["ac_cutoff"].unique())
for stats in focal_stats_by_dataset.values():
    if sorted(stats["ac_cutoff"].unique()) != ac_cutoffs:
        raise ValueError("The compared datasets must use the same allele-count cutoffs")
if len(set(haplotype_counts.values())) != 1:
    raise ValueError("Shared frequency-cutoff rows require equal haplotype counts")
num_haploid_samples = haplotype_counts["stdpopsim"]

# %% [markdown]
# ## Cumulative inferred ancestors by focal frequency
#
# Count each inferred ancestor once using its focal allele count in the NPZ.
# Dashed lines mark the coverage cutoffs divided by the number of haplotypes.
# These are panel-wide counts, before restricting to ancestors associated with
# a particular haplotype. Count divided by total haplotypes is the focal
# frequency coordinate used here, including for the cutoff labels.


# %%
def plot_cumulative_focal_ancestors(
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    num_haploid_samples: int,
    ac_cutoffs: list[int],
) -> plt.Axes:
    """Plot cumulative inferred ancestors by focal AF, marking count cutoffs.

    The dataframe has one row per ancestor and a derived_ac column from the
    focal-ancestor NPZ used for coverage evaluation. Divide by the total number
    of haplotypes to put both datasets on the same count-based frequency axis.
    This equals called-haplotype AF when there are no missing calls.
    Cutoffs use the same conversion, so the curve counts eligible ancestors
    at each dashed line, including ancestors equal to the cutoff.
    """
    focal_frequencies = dataframe["derived_ac"] / num_haploid_samples
    counts_by_frequency = focal_frequencies.value_counts().sort_index()
    cumulative_counts = counts_by_frequency.cumsum()
    x = np.concatenate(([1 / num_haploid_samples], counts_by_frequency.index, [1]))
    y = np.concatenate(([0], cumulative_counts, [len(dataframe)]))
    ax.step(x, y, where="post")
    for cutoff in ac_cutoffs:
        cutoff_frequency = cutoff / num_haploid_samples
        ax.axvline(cutoff_frequency, linestyle="--", color="0.5", linewidth=1)
        ax.text(
            cutoff_frequency,
            0.98,
            f"AC ≤ {cutoff}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            color="0.4",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlim(1 / num_haploid_samples, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Focal allele frequency")
    ax.set_ylabel("Cumulative count of inferred ancestors")
    return ax


# %%
fig, axes = plt.subplots(
    1, 2, figsize=(14, 5), sharex=True, sharey=True, layout="constrained"
)
for ax, label in zip(axes, dataset_names, strict=True):
    plot_cumulative_focal_ancestors(
        ax, focal_ancestors_by_dataset[label], haplotype_counts[label], ac_cutoffs
    )
    ax.set_title(label)
plt.show()

# %% [markdown]
# ## Focal coverage across haploid samples
#
# Each row in a statistics CSV is one haplotype at one cumulative allele-count
# cutoff. Matching uses the full panel at every cutoff; the coverage fraction
# counts inference sites copied from eligible focal ancestors. Box plots show
# these fractions at the same frequency cutoffs in both datasets.

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(14, 5), sharex=True, sharey=True, layout="constrained"
)
frequency_labels = [f"{cutoff / num_haploid_samples:g}" for cutoff in ac_cutoffs]
for ax, label in zip(axes, dataset_names, strict=True):
    stats = focal_stats_by_dataset[label]
    coverage_groups = []
    for cutoff in ac_cutoffs:
        cutoff_rows = stats.loc[stats["ac_cutoff"] == cutoff]
        coverage_groups.append(cutoff_rows["fraction_covered_sites"].to_numpy())
    ax.boxplot(coverage_groups, tick_labels=frequency_labels)
    ax.set_xlabel("Cumulative focal frequency cutoff")
    ax.set_ylabel("Fraction of covered inference sites per haplotype")
    ax.set_ylim(0, 1)
    ax.set_title(label)
plt.show()

# %% [markdown]
# ## Match score and focal coverage at each frequency cutoff
#
# `path_log_likelihood` is a normalized log score: values closer to zero
# indicate fewer mismatch and switch penalties. Each haplotype's score stays
# fixed across cutoffs; only its focal coverage changes.
#
# Each row compares one frequency cutoff, with stdpopsim on the left and 1KGP
# on the right. Population codes shared by both datasets use the same colour
# for points, fitted lines, and stacked marginal histograms. Coloured lines
# and legend `r` values show within-population fits and Pearson correlations;
# the dashed grey line and annotation show the pooled fit and correlation.
# All panels share score/coverage limits and histogram bin edges. Populations
# with constant scores have an undefined correlation (r = n/a) and no fitted line.


# %%
def plot_focal_match_stats(
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    cutoff: int,
    score_bins: np.ndarray,
    coverage_bins: np.ndarray,
    population_colors: dict[str, str],
    title: str,
) -> plt.Axes:
    """Plot one cutoff's score/coverage association with marginal histograms.

    The dataframe contains one row per haplotype and cutoff. Use common bin
    edges across calls to compare distributions at different cutoffs. The
    histograms are attached to ax and share its corresponding coordinate axis.
    Population colours identify points, within-population fits and stacked
    histograms; the dashed grey line shows the pooled fit.
    """
    cutoff_rows = dataframe.loc[dataframe["ac_cutoff"] == cutoff]
    scores = cutoff_rows["path_log_likelihood"].to_numpy()
    coverage = cutoff_rows["fraction_covered_sites"].to_numpy()
    pooled_correlation_label = "n/a"
    if len(scores) > 1 and np.ptp(scores) > 0:
        regression = scipy.stats.linregress(scores, coverage)
        if np.isfinite(regression.rvalue):
            pooled_correlation_label = f"{regression.rvalue:.3f}"
        fitted_scores = np.array([scores.min(), scores.max()])
        fitted_coverage = regression.intercept + regression.slope * fitted_scores
        ax.plot(
            fitted_scores, fitted_coverage, color="0.5", linestyle="--", linewidth=1.5
        )
    score_groups = []
    coverage_groups = []
    histogram_colors = []
    for population, color in population_colors.items():
        population_rows = cutoff_rows.loc[cutoff_rows["population"] == population]
        if population_rows.empty:
            continue
        population_scores = population_rows["path_log_likelihood"].to_numpy()
        population_coverage = population_rows["fraction_covered_sites"].to_numpy()
        population_fit = None
        correlation_label = "n/a"
        if len(population_scores) > 1 and np.ptp(population_scores) > 0:
            population_fit = scipy.stats.linregress(
                population_scores, population_coverage
            )
            if np.isfinite(population_fit.rvalue):
                correlation_label = f"{population_fit.rvalue:.3f}"
        ax.scatter(
            population_scores,
            population_coverage,
            s=18,
            alpha=0.6,
            color=color,
            label=(f"{population}: r = {correlation_label}, n = {len(population_rows)}"),
        )
        if population_fit is not None:
            fitted_scores = np.array([population_scores.min(), population_scores.max()])
            fitted_coverage = (
                population_fit.intercept + population_fit.slope * fitted_scores
            )
            ax.plot(fitted_scores, fitted_coverage, color=color, linewidth=2)
        score_groups.append(population_scores)
        coverage_groups.append(population_coverage)
        histogram_colors.append(color)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(
        0.04,
        0.20,
        f"Pooled r = {pooled_correlation_label}\nn = {len(cutoff_rows)} haplotypes",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.set_xlabel("Path log likelihood (higher is better)")
    ax.set_ylabel("Fraction of covered inference sites")
    ax.set_xlim(score_bins[0], score_bins[-1])
    ax.set_ylim(coverage_bins[0], coverage_bins[-1])

    divider = axes_grid1.make_axes_locatable(ax)
    score_ax = divider.append_axes("top", size="25%", pad=0.12, sharex=ax)
    coverage_ax = divider.append_axes("right", size="25%", pad=0.12, sharey=ax)
    score_ax.hist(
        score_groups, bins=score_bins, stacked=True, color=histogram_colors, alpha=0.7
    )
    coverage_ax.hist(
        coverage_groups,
        bins=coverage_bins,
        orientation="horizontal",
        stacked=True,
        color=histogram_colors,
        alpha=0.7,
    )
    score_ax.tick_params(axis="x", labelbottom=False)
    coverage_ax.tick_params(axis="y", labelleft=False)
    score_ax.set_ylabel("Haplotypes")
    coverage_ax.set_xlabel("Haplotypes")
    score_ax.set_title(title)
    return ax


# %%
combined_stats = pd.concat(focal_stats_by_dataset.values(), ignore_index=True)
population_names = sorted(combined_stats["population"].unique())
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
population_colors = {
    population: color_cycle[index % len(color_cycle)]
    for index, population in enumerate(population_names)
}
score_range = combined_stats["path_log_likelihood"].agg(["min", "max"])
score_padding = (score_range["max"] - score_range["min"]) * 0.05
score_bins = np.linspace(
    score_range["min"] - score_padding,
    score_range["max"] + score_padding,
    21,
)
coverage_bins = np.linspace(0, 1, 21)
fig, axes = plt.subplots(
    len(ac_cutoffs),
    2,
    figsize=(15, 6 * len(ac_cutoffs)),
    squeeze=False,
    sharex=True,
    sharey=True,
)
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.03, top=0.98, wspace=0.4, hspace=0.5)
for row, cutoff in enumerate(ac_cutoffs):
    frequency = cutoff / num_haploid_samples
    for column, label in enumerate(dataset_names):
        plot_focal_match_stats(
            axes[row, column],
            focal_stats_by_dataset[label],
            cutoff,
            score_bins,
            coverage_bins,
            population_colors,
            title=f"{label}: focal frequency ≤ {frequency:g} (AC ≤ {cutoff})",
        )
        axes[row, column].tick_params(labelbottom=True, labelleft=True)
plt.show()

# %%
