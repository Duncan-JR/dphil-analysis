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

# %%
import dataclasses
import logging
import pathlib

import IPython.display as ipython_display
import numpy as np
import pandas as pd

from ch5 import nb_plot, nb_utils

repository_dir = pathlib.Path.cwd()
logger = logging.getLogger(__name__)

# %%
repository_dir = pathlib.Path.cwd()
diagnostic_config_path = {
    "datasets": [
        {
            "name": "out_of_africa_n100_10mbp",
            "label": "Simulation, 100 individuals, 10 Mb",
            "deep_dive": {"sample_id": "tsk_18", "ploidy_index": 0},
            "haplotypes": [
                {"sample_id": "tsk_18", "ploidy_index": 0},
                {"sample_id": "tsk_40", "ploidy_index": 1},
                {"sample_id": "tsk_96", "ploidy_index": 1},
            ],
            "populations": ["YRI", "CEU", "CHB"],
        },
        {
            "name": "tgp_chr20_n100",
            "label": "1KGP chromosome 20, 100 individuals, 10 Mb",
            "deep_dive": {"sample_id": "NA18582", "ploidy_index": 0},
            "haplotypes": [
                {"sample_id": "NA07019", "ploidy_index": 1},
                {"sample_id": "NA18582", "ploidy_index": 0},
                {"sample_id": "NA19150", "ploidy_index": 0},
            ],
            "populations": ["AFR", "EUR", "EAS"],
        },
    ],
    "detail": {"context_sites": 25, "max_sites": 240, "min_agreement_gap_sites": 5},
    "outliers": {
        "output_dir": repository_dir / "data/ch5/stitching_outliers",
        "criteria": [
            {
                "metric": "num_extra_switches",
                "label": "Most extra switches",
                "ascending": False,
            },
            {
                "metric": "focal_agreement_percentage",
                "label": "Lowest focal agreement",
                "ascending": True,
            },
            {
                "metric": "gap_agreement_percentage",
                "label": "Lowest gap agreement",
                "ascending": True,
            },
        ],
        "short_gap_max_sites": 10,
        "short_gap_site_axis": True,
        "gaps_to_display": 8,
        "divergences_to_display": 8,
        "max_windows_per_haplotype": 5,
    },
    "distance_bin_edges": [0, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000, 100000],
    "cohort": {
        "num_haplotypes": 200,
        "num_individuals": 100,
        "output_dir": repository_dir / "data/ch5/stitching_cohort",
    },
}
diagnostic_config_path["base_dir"] = repository_dir
config = diagnostic_config_path
experiment_config = {
    "input_dir": repository_dir / "data/ch5/match_eval",
    "coverage_input_dir": repository_dir.parent / "tsinfer-match-eval/data/match_eval",
    "output_dir": repository_dir / "data/ch5/chunk_matching_zero_coverage",
    "ac_cutoff": 10,
    "min_focal_ac": 2,
    "workers": None,
    "score_tolerance": 1e-08,
    "datasets": [
        {"name": "out_of_africa_n100_10mbp", "haplotypes": None},
        {"name": "tgp_chr20_n100", "haplotypes": None},
    ],
}
input_dir = pathlib.Path(experiment_config["input_dir"]).expanduser()
output_dir = pathlib.Path(experiment_config["output_dir"]).expanduser()
if not input_dir.is_absolute():
    input_dir = repository_dir / input_dir
if not output_dir.is_absolute():
    output_dir = repository_dir / output_dir
stitching_pt2_result = nb_plot.make_path_colormaps()
category_cmap = stitching_pt2_result.category_cmap
category_norm = stitching_pt2_result.category_norm
source_cmap = stitching_pt2_result.source_cmap
source_norm = stitching_pt2_result.source_norm

# %% [markdown]
# # Extra switches vs hyperparameters
#
# The saved cohort contains only AC cutoff 10, and its outputs do not record
# whether zero or one mismatches per side were allowed. The requested mismatch
# panels are therefore marked unavailable. The final row shows the aggregate
# for the current saved configuration without assigning it to either setting.

# %%
hyperparameter_output_dir = pathlib.Path(config["cohort"]["output_dir"]).expanduser()
if not hyperparameter_output_dir.is_absolute():
    hyperparameter_output_dir = repository_dir / hyperparameter_output_dir
hyperparameter_results = pd.read_csv(
    hyperparameter_output_dir / "haplotype_results.csv"
)
switch_counts = hyperparameter_results.pivot(
    index=["dataset", "sample_id", "ploidy_index", "population", "ac_cutoff"],
    columns="method",
    values="num_switches",
).reset_index()
switch_counts["num_extra_switches"] = switch_counts.stitched - switch_counts.full
full_switch_counts = switch_counts.full.replace(0, np.nan)
switch_counts["switch_increase_percentage"] = (
    100 * switch_counts.num_extra_switches / full_switch_counts
)
hyperparameter_metrics = [
    ("num_extra_switches", "Mean extra switches"),
    ("switch_increase_percentage", "Mean percentage increase"),
]
requested_mismatch_settings = [0, 1]
nb_plot.plot_extra_switches_vs_hyperparameters(
    hyperparameter_metrics=hyperparameter_metrics,
    requested_mismatch_settings=requested_mismatch_settings,
    switch_counts=switch_counts,
    figsize=[12, 8],
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# The saved cohort has one AC cutoff. These means and standard errors describe
# the current configuration across all populations; the mismatch allowance is
# not recorded in its output metadata.

# %%
nb_plot.plot_extra_switches_vs_hyperparameters_2(
    hyperparameter_metrics=hyperparameter_metrics,
    switch_counts=switch_counts,
    figsize=[12, 5],
    marker="o",
    fontsize="small",
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %%
coverage_handles = nb_plot.make_path_legend()

# %% [markdown]
# # Population-level results
#
# Compare all saved haplotypes, grouped by population, in the simulation and
# 1KGP datasets. Each observation is one haplotype. Settings and plot styles
# live in the scientific settings cell below.

# %%
population_output_dir = pathlib.Path(config["cohort"]["output_dir"]).expanduser()
if not population_output_dir.is_absolute():
    population_output_dir = repository_dir / population_output_dir
population_output_dir.mkdir(parents=True, exist_ok=True)
population_tables = []
for settings in config["datasets"]:
    table = pd.read_csv(output_dir / settings["name"] / "comparison.csv")
    assert not table.duplicated(["sample_id", "ploidy_index"]).any()
    assert len(table) == config["cohort"]["num_haplotypes"]
    assert table.dataset.eq(settings["name"]).all()
    assert table.population.notna().all()
    assert set(table.population) == set(settings["populations"])
    population_tables.append(table)
population_comparisons = pd.concat(population_tables, ignore_index=True)
population_comparisons["num_extra_switches"] = (
    population_comparisons.stitched_switches - population_comparisons.full_switches
)
full_switch_denominator = population_comparisons.full_switches.replace(0, np.nan)
population_comparisons["switch_increase_percentage"] = (
    100 * population_comparisons.num_extra_switches / full_switch_denominator
)
for region in ("focal", "gap"):
    region_sites = population_comparisons[f"{region}_sites"].replace(0, np.nan)
    disagreeing_sites = population_comparisons[f"{region}_disagreement_sites"]
    population_comparisons[f"{region}_agreement_percentage"] = 100 * (
        1 - disagreeing_sites / region_sites
    )

# %% [markdown]
# ## Extra switches by population
#
# Extra switches are `stitched_switches - full_switches`. Histograms show the
# distribution across haplotypes; each point beneath a histogram represents one
# haplotype. Solid curves show kernel densities and dashed lines show means.

# %%
fig = nb_plot.plot_population_histograms(
    "num_extra_switches",
    "Extra switches",
    "Extra switches by population",
    integer_bins=True,
    config=config,
    plot_full_color="#222222",
    plot_population_results_count_bin_width=1,
    plot_population_results_density_tick_count=5,
    plot_population_results_histogram_alpha=0.3,
    plot_population_results_histogram_bins=30,
    plot_population_results_histogram_figsize=[14, 10],
    plot_population_results_kde_bandwidth="scott",
    plot_population_results_kde_grid_size=512,
    plot_population_results_line_width=2,
    plot_population_results_point_alpha=0.6,
    plot_population_results_point_size=12,
    plot_population_results_point_strip_depth=0.15,
    plot_population_results_population_colors={
        "YRI": "tab:blue",
        "AFR": "tab:blue",
        "CEU": "tab:orange",
        "EUR": "tab:orange",
        "CHB": "tab:gray",
        "EAS": "tab:gray",
    },
    plot_population_results_random_seed=42,
    plot_population_results_range_padding=0.05,
    population_comparisons=population_comparisons,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
nb_plot.plot_extra_switches_by_population(
    fig=fig, population_output_dir=population_output_dir
)

# %% [markdown]
# ## Percentage increase in switches by population
#
# Percentage increase is
# `100 * (stitched_switches - full_switches) / full_switches`.
# It is undefined when the full HMM has no switches, including when both paths
# have zero switches. Each panel reports the number of undefined observations
# excluded from its histogram, points, kernel density, and mean.

# %%
fig = nb_plot.plot_population_histograms(
    "switch_increase_percentage",
    "Increase in switches over full HMM (%)",
    "Percentage increase in switches by population",
    config=config,
    plot_full_color="#222222",
    plot_population_results_count_bin_width=1,
    plot_population_results_density_tick_count=5,
    plot_population_results_histogram_alpha=0.3,
    plot_population_results_histogram_bins=30,
    plot_population_results_histogram_figsize=[14, 10],
    plot_population_results_kde_bandwidth="scott",
    plot_population_results_kde_grid_size=512,
    plot_population_results_line_width=2,
    plot_population_results_point_alpha=0.6,
    plot_population_results_point_size=12,
    plot_population_results_point_strip_depth=0.15,
    plot_population_results_population_colors={
        "YRI": "tab:blue",
        "AFR": "tab:blue",
        "CEU": "tab:orange",
        "EUR": "tab:orange",
        "CHB": "tab:gray",
        "EAS": "tab:gray",
    },
    plot_population_results_random_seed=42,
    plot_population_results_range_padding=0.05,
    population_comparisons=population_comparisons,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
nb_plot.plot_percentage_increase_in_switches_by_population(
    fig=fig, population_output_dir=population_output_dir
)

# %% [markdown]
# ## Gap properties
#
# Each observation is one saved haplotype. The histograms pool populations within
# each dataset; solid curves show kernel densities and dashed lines show means.
# Points are coloured consistently by population: blue for YRI/AFR, orange for
# CEU/EUR, and gray for CHB/EAS. Mean sites per gap is `gap_sites / num_gaps` and
# is undefined for haplotypes with no gaps. Exclusion counts appear in each panel.

# %%
gap_denominator = population_comparisons.num_gaps.replace(0, np.nan)
population_comparisons["mean_sites_per_gap"] = (
    population_comparisons.gap_sites / gap_denominator
)
gap_population_handles = nb_plot.plot_gap_properties(
    config=config,
    population_comparisons=population_comparisons,
    population_output_dir=population_output_dir,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %%
fig = nb_plot.plot_gap_property_histograms(
    "mean_sites_per_gap",
    "Mean sites per gap",
    "Mean sites per gap per haplotype",
    config=config,
    gap_population_handles=gap_population_handles,
    plot_full_color="#222222",
    plot_population_results_count_bin_width=1,
    plot_population_results_density_tick_count=5,
    plot_population_results_gap_histogram_color="tab:gray",
    plot_population_results_gap_histogram_figsize=[14, 5],
    plot_population_results_histogram_alpha=0.3,
    plot_population_results_histogram_bins=30,
    plot_population_results_kde_bandwidth="scott",
    plot_population_results_kde_grid_size=512,
    plot_population_results_line_width=2,
    plot_population_results_point_alpha=0.6,
    plot_population_results_point_size=12,
    plot_population_results_point_strip_depth=0.15,
    plot_population_results_population_colors={
        "YRI": "tab:blue",
        "AFR": "tab:blue",
        "CEU": "tab:orange",
        "EUR": "tab:orange",
        "CHB": "tab:gray",
        "EAS": "tab:gray",
    },
    plot_population_results_random_seed=42,
    plot_population_results_range_padding=0.05,
    population_comparisons=population_comparisons,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
nb_plot.plot_gap_properties_2(fig=fig, population_output_dir=population_output_dir)

# %% [markdown]
# Compare each haplotype's mean gap length with its number of eligible focal
# ancestors (top row) and number of gaps (bottom row). Haplotypes without gaps
# are excluded.

# %%
nb_plot.plot_gap_properties_3(
    config=config,
    gap_population_handles=gap_population_handles,
    population_comparisons=population_comparisons,
    population_output_dir=population_output_dir,
    figsize=[14, 9],
    s=12,
    alpha=0.6,
    linewidths=0,
    ylabel="Mean sites per gap",
    alpha_2=0.2,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# Relate the number of extra switches to the number of gaps, focal ancestors,
# and mean sites per gap. Each row is one dataset; points are haplotypes coloured
# by population. Haplotype-level gaps and extra switches are integer counts.

# %%
gap_predictors = [
    ("num_gaps", "Number of gaps"),
    ("num_focal_ancestors", "Number of focal ancestors"),
    ("mean_sites_per_gap", "Mean sites per gap"),
]
nb_plot.plot_gap_properties_4(
    config=config,
    gap_population_handles=gap_population_handles,
    gap_predictors=gap_predictors,
    population_comparisons=population_comparisons,
    population_output_dir=population_output_dir,
    figsize=[16, 8],
    s=12,
    alpha=0.6,
    linewidths=0,
    alpha_2=0.2,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# ## Decomposing switches by type
#
# Switches are partitioned into those within focal coverage, within gaps, and
# at focal/gap seams, using the same region boundaries for both methods.
# Each haplotype's counts are divided by its own method's total switches before
# averaging within populations. The stacked bars therefore describe mean
# haplotype compositions, and each bar sums to 100%. Haplotypes with zero
# switches have undefined compositions and are excluded, as noted below.

# %%
population_switch_categories = ["focal", "gap", "seam"]
population_switch_tables = []
for method in ("full", "stitched"):
    method_table = population_comparisons[
        ["dataset", "sample_id", "ploidy_index", "population"]
    ].copy()
    method_table["method"] = method
    totals = population_comparisons[f"{method}_switches"]
    counts = population_comparisons[
        [f"{method}_{category}_switches" for category in population_switch_categories]
    ]
    assert counts.sum(axis=1).eq(totals).all()
    denominator = totals.replace(0, np.nan)
    for category in population_switch_categories:
        category_counts = population_comparisons[f"{method}_{category}_switches"]
        method_table[category] = category_counts / denominator
    defined = totals.gt(0)
    assert np.allclose(
        method_table.loc[defined, population_switch_categories].sum(axis=1), 1
    )
    population_switch_tables.append(method_table)
population_switch_proportions = pd.concat(population_switch_tables, ignore_index=True)
population_switch_means = population_switch_proportions.groupby(
    ["dataset", "population", "method"]
)[population_switch_categories].mean()
population_switch_excluded = (
    population_switch_proportions.assign(
        excluded=population_switch_proportions.focal.isna()
    )
    .groupby(["dataset", "population", "method"])
    .excluded.sum()
)
excluded_compositions = population_switch_excluded.rename(
    "Undefined compositions excluded"
)
ipython_display.display(excluded_compositions.to_frame())
nb_plot.plot_decomposing_switches_by_type(
    config=config,
    population_output_dir=population_output_dir,
    population_switch_categories=population_switch_categories,
    population_switch_means=population_switch_means,
    figsize=[14, 6],
    height=0.32,
    xlim=(0, 100),
    xlabel="Mean proportion of switches (%)",
    alpha=0.2,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# The ternary plots show the same compositions for individual haplotypes.
# A corner represents 100% of its switch type; the opposite edge represents 0%.
# Colours identify populations, consistently across datasets: blue for YRI/AFR,
# orange for CEU/EUR, and gray for CHB/EAS. Circles show full HMM haplotypes and
# triangles show stitched HMM haplotypes. Large outlined markers show population
# means, labelled by population. Undefined compositions are excluded as in the bars.

# %%
nb_plot.plot_decomposing_switches_by_type_2(
    config=config,
    population_output_dir=population_output_dir,
    population_switch_categories=population_switch_categories,
    population_switch_means=population_switch_means,
    population_switch_proportions=population_switch_proportions,
    figsize=[14, 7],
    color="#222222",
    color_2="lightgray",
    linewidth=0.7,
    fontsize="small",
    s=14,
    alpha=0.2,
    linewidths=0,
    s_2=90,
    xlim=(-0.15, 1.15),
    linestyle="none",
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# ## Parent agreement proportions
#
# Each strip shows the mean proportions of all inference sites assigned to
# focal/agree, focal/disagree, gaps/agree, and gaps/disagree. The four fractions
# share the same denominator and sum to 100%. Focal and gap segments are kept
# together, so their boundary also shows mean focal coverage.

# %%
population_agreement_columns = [
    "focal_agree_fraction",
    "focal_disagree_fraction",
    "gap_agree_fraction",
    "gap_disagree_fraction",
]
assert np.allclose(population_comparisons[population_agreement_columns].sum(axis=1), 1)
population_agreement_means = population_comparisons.groupby(["dataset", "population"])[
    population_agreement_columns
].mean()
nb_plot.plot_parent_agreement_proportions(
    config=config,
    population_agreement_columns=population_agreement_columns,
    population_agreement_means=population_agreement_means,
    population_output_dir=population_output_dir,
    figsize=[14, 6],
    xlim=(0, 100),
    xlabel="Mean proportion of all inference sites (%)",
    alpha=0.2,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# ## Parent agreement in focal regions
#
# Agreement within focal regions is
# `100 * (focal_sites - focal_disagreement_sites) / focal_sites`.
# Each haplotype contributes its own percentage; agreement and disagreement
# within its focal regions sum to 100%. Haplotypes without focal sites have
# undefined agreement percentages and are excluded, with counts in each panel.

# %%
fig = nb_plot.plot_population_histograms(
    "focal_agreement_percentage",
    "Parent agreement within focal regions (%)",
    "Parent agreement in focal regions by population",
    bounded_percentage=True,
    config=config,
    plot_full_color="#222222",
    plot_population_results_count_bin_width=1,
    plot_population_results_density_tick_count=5,
    plot_population_results_histogram_alpha=0.3,
    plot_population_results_histogram_bins=30,
    plot_population_results_histogram_figsize=[14, 10],
    plot_population_results_kde_bandwidth="scott",
    plot_population_results_kde_grid_size=512,
    plot_population_results_line_width=2,
    plot_population_results_point_alpha=0.6,
    plot_population_results_point_size=12,
    plot_population_results_point_strip_depth=0.15,
    plot_population_results_population_colors={
        "YRI": "tab:blue",
        "AFR": "tab:blue",
        "CEU": "tab:orange",
        "EUR": "tab:orange",
        "CHB": "tab:gray",
        "EAS": "tab:gray",
    },
    plot_population_results_random_seed=42,
    plot_population_results_range_padding=0.05,
    population_comparisons=population_comparisons,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
nb_plot.plot_parent_agreement_in_focal_regions(
    fig=fig, population_output_dir=population_output_dir
)

# %% [markdown]
# ## Parent agreement in gaps
#
# Agreement within gaps is
# `100 * (gap_sites - gap_disagreement_sites) / gap_sites`.
# Each haplotype contributes its own percentage; agreement and disagreement
# within its gaps sum to 100%. Haplotypes without gap sites have undefined
# agreement percentages and are excluded, with counts in each panel.

# %%
fig = nb_plot.plot_population_histograms(
    "gap_agreement_percentage",
    "Parent agreement within gaps (%)",
    "Parent agreement in gaps by population",
    bounded_percentage=True,
    config=config,
    plot_full_color="#222222",
    plot_population_results_count_bin_width=1,
    plot_population_results_density_tick_count=5,
    plot_population_results_histogram_alpha=0.3,
    plot_population_results_histogram_bins=30,
    plot_population_results_histogram_figsize=[14, 10],
    plot_population_results_kde_bandwidth="scott",
    plot_population_results_kde_grid_size=512,
    plot_population_results_line_width=2,
    plot_population_results_point_alpha=0.6,
    plot_population_results_point_size=12,
    plot_population_results_point_strip_depth=0.15,
    plot_population_results_population_colors={
        "YRI": "tab:blue",
        "AFR": "tab:blue",
        "CEU": "tab:orange",
        "EUR": "tab:orange",
        "CHB": "tab:gray",
        "EAS": "tab:gray",
    },
    plot_population_results_random_seed=42,
    plot_population_results_range_padding=0.05,
    population_comparisons=population_comparisons,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
nb_plot.plot_parent_agreement_in_gaps(
    fig=fig, population_output_dir=population_output_dir
)


# %% [markdown]
# # Analysis of outliers
#
# Select one haplotype per dataset for each criterion: most extra switches,
# lowest parent agreement within focal regions, and lowest parent agreement
# within gaps. Undefined agreement percentages are excluded from selection.
# Ties are resolved by sample ID, then ploidy index. The table identifies the
# populations and the denominators before inspecting any copying paths.

# %%
outlier_rows = []
for settings in config["datasets"]:
    table = population_comparisons.loc[
        population_comparisons.dataset.eq(settings["name"])
    ]
    for criterion in config["outliers"]["criteria"]:
        metric = criterion["metric"]
        eligible = table.dropna(subset=[metric])
        ordered = eligible.sort_values(
            [metric, "sample_id", "ploidy_index"],
            ascending=[criterion["ascending"], True, True],
        )
        selected = ordered.iloc[0].to_dict()
        selected["criterion"] = criterion["label"]
        selected["tied_haplotypes"] = int(eligible[metric].eq(selected[metric]).sum())
        selected["haplotype"] = f"{selected['sample_id']}/{selected['ploidy_index']}"
        outlier_rows.append(selected)
outlier_selection = pd.DataFrame(outlier_rows)
ipython_display.display(
    outlier_selection[
        [
            "dataset",
            "criterion",
            "haplotype",
            "population",
            "num_extra_switches",
            "focal_agreement_percentage",
            "focal_sites",
            "gap_agreement_percentage",
            "gap_sites",
            "num_gaps",
            "tied_haplotypes",
        ]
    ].round(2)
)
outlier_output_dir = pathlib.Path(config["outliers"]["output_dir"]).expanduser()
if not outlier_output_dir.is_absolute():
    outlier_output_dir = repository_dir / outlier_output_dir
outlier_output_dir.mkdir(parents=True, exist_ok=True)
outlier_selection.to_csv(outlier_output_dir / "selected_haplotypes.csv", index=False)
outlier_datasets = []
for settings in config["datasets"]:
    selected = outlier_selection.loc[outlier_selection.dataset.eq(settings["name"])]
    identities = selected[["sample_id", "ploidy_index"]].drop_duplicates()
    haplotypes = identities.to_dict("records")
    outlier_settings = settings | {"haplotypes": haplotypes, "deep_dive": haplotypes[0]}
    dataset = nb_utils.path_diagnostics_load_dataset(
        outlier_settings,
        experiment_config=experiment_config,
        input_dir=input_dir,
        output_dir=output_dir,
        plot_terminal_site_width_bp=1,
    )
    cases_by_identity = {
        (case.summary["sample_id"], case.summary["ploidy_index"]): case
        for case in dataset.cases
    }
    dataset.cases = [
        cases_by_identity[identity["sample_id"], identity["ploidy_index"]]
        for identity in haplotypes
    ]
    outlier_datasets.append(dataset)
outlier_case_datasets = [
    dataclasses.replace(dataset, primary=case)
    for dataset in outlier_datasets
    for case in dataset.cases
]

# %% [markdown]
# ## Summary plots
#
# Compare each outlier's four site categories and full/stitched switch counts,
# then locate its focal regions, gaps, and parent disagreement along the genome.
# Switch-type colours match the population plots. Whole-span parent IDs identify
# copying parents; their vertical spacing has no biological distance meaning.

# %%
outlier_summary = nb_plot.plot_case_summary(
    outlier_datasets,
    "Outlier haplotypes: site proportions and switch counts",
    plot_category_colors=["#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
    plot_category_labels=[
        "Focal / agree",
        "Focal / differ",
        "Gap fill / agree",
        "Gap fill / differ",
    ],
    plot_cohort_figsize=[16, 8],
    plot_switch_colors=["#009E73", "#E69F00", "#7B61A8"],
    plot_switch_labels=["Within focal coverage", "Within gaps", "At focal/gap seams"],
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
nb_plot.save_notebook_figure(
    outlier_summary.figure, outlier_output_dir / "outlier_site_and_switch_summary.png"
)
nb_plot.plot_summary_plots(
    category_cmap=category_cmap,
    category_norm=category_norm,
    coverage_handles=coverage_handles,
    outlier_datasets=outlier_datasets,
    outlier_output_dir=outlier_output_dir,
    outlier_summary=outlier_summary,
    source_cmap=source_cmap,
    source_norm=source_norm,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %%
nb_plot.plot_summary_plots_2(
    category_cmap=category_cmap,
    category_norm=category_norm,
    coverage_handles=coverage_handles,
    outlier_case_datasets=outlier_case_datasets,
    outlier_output_dir=outlier_output_dir,
    source_cmap=source_cmap,
    source_norm=source_norm,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# ## Deep dive
#
# A switch has a position and a directed parent transition, such as A → B.
# Losing a focal-interior switch and gaining a seam switch can therefore mean
# that the same transition moved, or that the ordered parent sequence changed.
# We distinguish these using maximal runs of parent disagreement, including one
# agreeing site on each side when both flanks exist:
#
# - **Identical switch:** same position and same parents on both sides.
# - **Shifted parent sequence:** both flanking parents agree and the complete
#   ordered parent sequences are identical, while switch positions differ.
# - **Changed parent sequence:** flanking parents agree but the ordered sequences
#   differ. Some individual transitions may still be shared within these regions.
# - **Unanchored difference:** a disagreement reaches an inference-site boundary,
#   so there is no pair of agreeing flanks for this comparison.
#
# The switch-count partition includes every full and stitched switch. Proven
# relocations preserve parent IDs and direction; they are not inferred by pairing
# whichever switch happens to be closest. Counts associated with a gap include
# its interior and its existing left/right seams. They are signed net differences,
# and omit switches in focal interiors. This distinction matters when a switch
# moves between the two types of region.

# %%

# %%
outlier_analyses = {}
outlier_windows = {}
outlier_change_rows = []
outlier_relocation_rows = []
for dataset in outlier_case_datasets:
    case = dataset.primary
    key = (dataset.name, case.label)
    analysis = nb_utils.analyse_outlier_paths(
        dataset,
        plot_outliers_transition_kinds=[
            "Identical switch",
            "Shifted parent sequence",
            "Changed parent sequence",
            "Unanchored difference",
        ],
    )
    outlier_analyses[key] = analysis
    outlier_windows[key] = nb_utils.select_outlier_windows(
        dataset,
        analysis,
        config=config,
        plot_outliers_transition_kinds_2="Changed parent sequence",
    )
    prefix = f"{dataset.name}_{case.summary['sample_id'].lower()}_{case.summary['ploidy_index']}"
    for name, table in (
        ("gaps", analysis.gaps),
        ("parent_differences", analysis.divergences),
        ("relocated_switches", analysis.relocations),
    ):
        table.to_csv(outlier_output_dir / f"{prefix}_{name}.csv", index=False)
    for row in analysis.switch_counts.itertuples():
        outlier_change_rows.append(
            {
                "dataset": dataset.name,
                "haplotype": case.label,
                "population": case.summary["population"],
                "kind": row.kind,
                "full_switches": row.full,
                "stitched_switches": row.stitched,
                "extra_switches": row.stitched - row.full,
            }
        )
    for row in analysis.relocations.itertuples():
        outlier_relocation_rows.append(
            {
                "dataset": dataset.name,
                "haplotype": case.label,
                "full_location": row.full_location,
                "stitched_location": row.stitched_location,
                "absolute_shift_sites": abs(row.shift_sites),
            }
        )
outlier_parent_changes = pd.DataFrame(outlier_change_rows)
outlier_parent_changes.to_csv(
    outlier_output_dir / "outlier_parent_transition_summary.csv", index=False
)
ipython_display.display(outlier_parent_changes)
outlier_relocations = pd.DataFrame(outlier_relocation_rows)
relocation_summary = outlier_relocations.groupby(
    ["dataset", "haplotype", "full_location", "stitched_location"]
).absolute_shift_sites.agg(["count", "median", "max"])
print("Proven parent-preserving relocations (haplotypes without any are absent):")
ipython_display.display(relocation_summary)
relocation_summary.to_csv(outlier_output_dir / "outlier_relocation_summary.csv")

# %% [markdown]
# The bars below distinguish relocation of a preserved parent sequence from
# switches occurring in regions with different parent sequences. Relocation
# preserves the number of transitions within its disagreement run. A positive
# net excess in a changed-sequence region cannot be explained by moving the
# same complete sequence of parent transitions alone.
# A gap with complete parent agreement can still gain seam switches when its
# neighbouring focal parents differ from the full path.

# %%
nb_plot.plot_deep_dive(
    outlier_analyses=outlier_analyses,
    outlier_datasets=outlier_datasets,
    outlier_output_dir=outlier_output_dir,
    figsize=[16, 8],
    height=0.32,
    xlabel="Parent switches",
    alpha=0.2,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# Inspect every outlier's gap table and genomic switch distribution below.
# Switch-position rugs compare positions only; the parent-sequence tables and
# categorical zooms establish whether the parents also change. The cumulative
# curve accounts for switch gains and losses across the entire inference span.
# Gap plots label gaps as g0, g1, …, matching their table indices.

# %%
nb_plot.plot_deep_dive_2(
    category_cmap=category_cmap,
    category_norm=category_norm,
    config=config,
    coverage_handles=coverage_handles,
    outlier_analyses=outlier_analyses,
    outlier_case_datasets=outlier_case_datasets,
    outlier_output_dir=outlier_output_dir,
    source_cmap=source_cmap,
    source_norm=source_norm,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# Choose detailed windows reproducibly: a short gap with high net extra switches
# per site, the gap with lowest parent agreement, the longest uninterrupted
# focal disagreement, a changed parent sequence around a gap, and a proven
# parent-preserving relocation when available. Duplicate displayed windows are
# shown once. Tables report the entire target and displayed coordinates.
# Short-gap zooms use inference-site indices to keep even one-site gaps visible;
# their genomic coordinates are reported in the tables. Other zooms use kb.
#
# Most long targets are clipped to the configured site limit around a relevant
# anchor. Relocation windows retain both switch positions, even when farther
# apart, so the before/after parents and the intervening disagreement are visible.
# Shared categorical rows identify the actual parents; black dashed is full HMM
# and solid purple is stitched. Dotted lines mark target boundaries.

# %%
outlier_window_rows = []
nb_plot.plot_deep_dive_3(
    category_cmap=category_cmap,
    category_norm=category_norm,
    config=config,
    coverage_handles=coverage_handles,
    outlier_case_datasets=outlier_case_datasets,
    outlier_output_dir=outlier_output_dir,
    outlier_window_rows=outlier_window_rows,
    outlier_windows=outlier_windows,
    source_cmap=source_cmap,
    source_norm=source_norm,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
pd.DataFrame(outlier_window_rows).to_csv(
    outlier_output_dir / "outlier_detail_windows.csv", index=False
)

# %% [markdown]
# Finally, compare disagreement with distance from the nearest seam separately
# for every outlier and source region. Distance is in inference-site steps;
# the sites immediately beside a seam have distance zero. These are descriptive
# profiles of selected outliers, with denominators exported alongside the plot.

# %%
outlier_distance_rows = nb_plot.plot_deep_dive_4(
    config=config,
    outlier_datasets=outlier_datasets,
    outlier_output_dir=outlier_output_dir,
    outlier_selection=outlier_selection,
    figsize=[16, 9],
    marker="o",
    ylim=(0, 1),
    alpha=0.2,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
pd.DataFrame(outlier_distance_rows).to_csv(
    outlier_output_dir / "outlier_seam_distance_counts.csv", index=False
)


# %% [markdown]
# # Diagnosing stitched focal and gap matches
#
# Compare full HMM and stitched copying paths for a 10 Mb simulation and a
# 10 Mb region of chromosome 20 from 1KGP. Focal coverage is the union of
# zero-mismatch intervals anchored at each eligible ancestor's leftmost focal
# site, using 2 ≤ focal AC ≤ 10. Gaps are maximal uncovered site runs.
# Within coverage, a covering ancestor is kept until its interval ends; the
# next available interval extending furthest right is chosen, with original
# node ID breaking ties. Independent full-reference HMM matches fill the gaps.
# All parents are original-reference node IDs.
#
# Questions:
# 1. Which focal pieces and gap fills agree with the full HMM parent path?
# 2. Where do the additional switches occur: inside focal pieces, inside gaps,
#    or at the seams between them?
# 3. Can we inspect both successful and unsuccessful gap fills, and focal pieces
#    that already differ from the full path?
#
# Settings and deep-dive identities live in
# the scientific settings cell below. The notebook reads existing
# outputs and copied references; it does not rerun matching or change its inputs.
# Edit this `.py` file and use Jupytext to synchronize the paired notebook.

# %%
datasets = [
    nb_utils.path_diagnostics_load_dataset(
        settings,
        experiment_config=experiment_config,
        input_dir=input_dir,
        output_dir=output_dir,
        plot_terminal_site_width_bp=1,
    )
    for settings in config["datasets"]
]
summary = pd.DataFrame([case.summary for dataset in datasets for case in dataset.cases])
summary["focal_agreement_fraction"] = (
    1 - summary.focal_disagreement_sites / summary.focal_sites
)
summary["gap_agreement_fraction"] = (
    1 - summary.gap_disagreement_sites / summary.gap_sites
)
ipython_display.display(
    summary[
        [
            "dataset",
            "sample_id",
            "ploidy_index",
            "num_focal_ancestors",
            "focal_sites",
            "gap_sites",
            "num_gaps",
            "fraction_parent_agreement",
            "focal_agreement_fraction",
            "gap_agreement_fraction",
            "full_switches",
            "stitched_switches",
            "num_extra_switches",
            "extra_switch_fraction",
            "full_mismatches",
            "stitched_mismatches",
        ]
    ].round(4)
)

# %% [markdown]
# ## Coverage, agreement, and switches across the saved haplotypes
#
# Bar lengths below are **fractions of inference sites**, not physical coverage.
# Focal/gap provenance and parent agreement are independent dimensions. An exact
# focal allele match may still copy a different parent than the full HMM.
#
# Switches are changes between adjacent site parents. A seam is the edge where
# provenance changes between focal coverage and a gap fill; it contributes
# a switch only if the parent changes there. The same edge categories are used for
# the full HMM path so their totals can be compared directly.


# %%
case_summary = nb_plot.plot_case_summary(
    datasets,
    f"Saved haplotypes at focal AC ≤ {experiment_config['ac_cutoff']}",
    plot_category_colors=["#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
    plot_category_labels=[
        "Focal / agree",
        "Focal / differ",
        "Gap fill / agree",
        "Gap fill / differ",
    ],
    plot_cohort_figsize=[16, 8],
    plot_switch_colors=["#009E73", "#E69F00", "#7B61A8"],
    plot_switch_labels=["Within focal coverage", "Within gaps", "At focal/gap seams"],
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
nb_plot.show_figure()
switch_table = case_summary.switches
ipython_display.display(switch_table)

# %% [markdown]
# ## Extra switch fraction
#
# `num_extra_switches = stitched_switches - full_switches`, and
# `extra_switch_fraction = num_extra_switches / stitched_switches`.
# This reports the net extra switches as a proportion of all stitched switches,
# using the stitched count as the denominator. It is undefined when the stitched
# path has no switches, and can be negative if stitching reduces the count.
# The count alone does not identify which individual switches are unnecessary.

# %%
nb_plot.plot_extra_switch_fraction_2(
    datasets=datasets,
    figsize=[16, 4],
    color="#6A3D9A",
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# ## Where covered focal pieces and gap fills agree with the full path
#
# Each haplotype has two tracks: provenance (covered focal or gap fill), then
# the four-way agreement classification. Genomic widths are a display convention:
# a site's classification extends to the next inference-site position, with one
# configured plotting width at the final site. This is **not** evidence of exact
# focal matching between sites or beyond ancestor support. All counts and rates
# remain site-based. The chromosome's unobserved flanks are excluded from plots.


# %%
fig = nb_plot.plot_case_tracks(
    datasets,
    "Each haplotype: provenance above, parent agreement below",
    category_cmap=category_cmap,
    category_norm=category_norm,
    coverage_handles=coverage_handles,
    plot_genome_unit_bp=1000000,
    plot_genome_unit_label="Mb",
    plot_tracks_figsize=[16, 7],
    source_cmap=source_cmap,
    source_norm=source_norm,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
nb_plot.show_figure()

# %% [markdown]
# ## Whole-span paths for the selected simulated and real haplotypes
#
# The full HMM and stitched paths use the same original-node ID scale. Node IDs
# identify parents; their vertical spacing has no biological distance meaning.
# The source and agreement tracks show which sites have focal coverage and which were
# rematched. A gap's parent can agree with the full HMM for only part of that gap.


# %%
nb_plot.plot_whole_span_paths_for_the_selected_simulated_and_real(
    category_cmap=category_cmap,
    category_norm=category_norm,
    coverage_handles=coverage_handles,
    datasets=datasets,
    source_cmap=source_cmap,
    source_norm=source_norm,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# ## Gap diagnostics and reproducible zoom selection
#
# Each gap is summarized with its number of sites, agreement fraction, and the
# switch counts strictly inside the gap. Left/right seam switches are reported
# separately. Chromosome-edge gaps have no seam on their outer edge.
#
# For each selected haplotype the notebook chooses four windows from the saved
# site arrays, rather than hand-picking genomic coordinates:
# - **Gap disagreement:** the gap with the most disagreeing sites; ties use NPZ
#   interval order. The zoom is anchored on its first disagreeing site.
# - **Gap agreement:** among gaps of at least the configured minimum size, choose
#   the highest agreement fraction, breaking ties by longer gap then earlier
#   interval. The zoom is anchored on its first agreeing site.
# - **Mixed gap fill:** among gaps with both agreement and disagreement, choose
#   the one with the most disagreeing sites, then the earliest gap. Anchor the
#   zoom on its first within-gap transition between agreement and disagreement.
# - **Focal disagreement:** the covered focal interval with the most disagreeing
#   sites. The zoom is anchored on its first disagreement.
#
# Short target intervals include their full extent and context on both sides.
# Long intervals are clipped to the configured maximum site count around the
# selected anchor, so a zoom may show only part of the target. Dashed lines mark
# target boundaries when they are inside the displayed window. The table reports
# both the complete target and the displayed half-open site window.


# %%
gap_tables = {}
detail_windows = {}
for dataset in datasets:
    gaps = nb_utils.path_diagnostics_gap_diagnostics(dataset)
    gap_tables[dataset.name] = gaps
    detail_windows[dataset.name] = nb_utils.path_diagnostics_select_windows(
        dataset, gaps, config=config
    )
    print(f"{dataset.label}, {dataset.primary.label}: {len(gaps)} gaps")
    ipython_display.display(
        gaps.sort_values("differ_sites", ascending=False).head(10).round(4)
    )
    window_rows = []
    for window in detail_windows[dataset.name]:
        target_disagreement = dataset.primary.disagreement[
            window.target_start : window.target_end
        ]
        window_rows.append(
            {
                **dataclasses.asdict(window),
                "target_agree_sites": int(np.count_nonzero(~target_disagreement)),
                "target_differ_sites": int(np.count_nonzero(target_disagreement)),
                "display_start_bp": dataset.positions[window.start],
                "display_end_bp": dataset.site_edges[window.end],
            }
        )
    ipython_display.display(pd.DataFrame(window_rows))

# %% [markdown]
# ## Detailed parent paths: covered focal pieces, gap fills, agreement, and difference
#
# These zooms put both paths on **categorical rows labeled by original ancestor
# identity and node ID**. Rows are shared within each panel; matching parent IDs
# occupy exactly the same row. Solid purple is stitched; dashed black is full HMM.
# Markers locate inference sites. Background shading shows provenance, and the
# lower agreement strip distinguishes all four source/agreement combinations.
# The bottom rug marks full-path switches above and stitched-path switches below.
#
# Focal intervals are exact in generated ancestor calls. Parent agreement tests
# copying identity; the saved mutation records separately test copied alleles
# in the inferred reference.


# %%
nb_plot.plot_detailed_parent_paths_covered_focal_pieces_gap_fills(
    category_cmap=category_cmap,
    category_norm=category_norm,
    coverage_handles=coverage_handles,
    datasets=datasets,
    detail_windows=detail_windows,
    source_cmap=source_cmap,
    source_norm=source_norm,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %% [markdown]
# ## Does disagreement concentrate near focal/gap seams?
#
# Pool the saved haplotypes **within each dataset**, separating covered focal
# sites from gap-fill sites. Distance is the number of inference-site steps to
# the nearest provenance seam, with the immediately adjacent site on either side
# assigned distance zero. Fractions use the actual site counts in each distance
# bin; the accompanying table gives the denominators.
#
# These are selected diagnostic haplotypes, not a representative population
# sample. Proximity is descriptive and does not isolate a causal boundary effect.
# In particular, distant disagreement within covered focal pieces means that
# conditioning gap endpoints alone cannot recover this particular full parent path.


# %%
distance_rows = []
bins = np.array(config["distance_bin_edges"])
nb_plot.plot_does_disagreement_concentrate_near_focal_gap_seams_2(
    bins=bins,
    datasets=datasets,
    distance_rows=distance_rows,
    figsize=[16, 4],
    marker="o",
    ylim=(0, 1),
    xlabel="Distance to nearest seam (inference-site steps)",
    alpha=0.2,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
distance_table = pd.DataFrame(distance_rows)
ipython_display.display(distance_table.loc[distance_table.num_sites > 0].round(4))
