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
# # Initial error-rate estimation
#
# This notebook records the production-style call used for the chromosome 17
# error-estimation dataset. The estimator cell is intentionally not executed
# as part of repository checks.

# %% [markdown]
# ## Validation against old version

# %%
import dataclasses
import pathlib

import IPython.display as ipython_display
import numpy as np
import pandas as pd
import zarr

from ch4 import nb_plot, nb_utils

repository_dir = pathlib.Path.cwd()


# %%
#zarr_dir = Path(
#).expanduser()
#zarr_prefix = "OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0"
#ts_path = Path(
#).expanduser()
# zarr_dir = Path(
#    "~/work/tsinfer-anc-eval/data/anc_eval/zarr_vcfs/"
# ).expanduser()
# zarr_prefix = "OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0"
# ts_path = Path(
#    "~/work/tsinfer-anc-eval/data/anc_eval/simulated/"
#    "OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0.trees"
# ).expanduser()
zarr_dir = pathlib.Path(
    "~/work/tsinfer-anc-eval/data/error_eval/zarr_vcfs/"
).expanduser()
zarr_prefix = "OutOfAfrica_4J17-chr17-L0-R22.7e6-n3202-s1-rep0"
ts_path = pathlib.Path(
    "~/work/tsinfer-anc-eval/data/error_eval/simulated/OutOfAfrica_4J17-chr17-L0-R22.7e6-n3202-s1-rep0.trees"
).expanduser()
recombination = (
    repository_dir / "data/general/HapMapII_GRCh38/genetic_map_Hg38_chr17.txt"
).expanduser()
window_sizes = np.geomspace(100, 1000000, 100)
config = nb_utils.make_estimation_config(
    window_sizes=window_sizes,
    num_doubletons=10000,
    random_seed=42,
    exclude_high_mismatch_proportion=0.25,
    L_mismatch_trim=window_sizes[-1],
)


# %%
error_multipliers = [0, 1, 2, 5, 10]
true_diversity = nb_utils.get_ts_diversity(ts_path)
zero_error_zarr_path = zarr_dir / f"{zarr_prefix}-geno-0.0-phase0.0-mispol0.0.zarr"
true_doubletons_config = dataclasses.replace(
    config, num_doubletons=7500, exclude_high_mismatch_proportion=0
)
true_doubletons = nb_utils.sample_doubletons(
    zero_error_zarr_path, true_doubletons_config
)
true_doubletons_path = repository_dir / "data/ch4/validation/true_doubletons.csv"
true_doubletons_path.parent.mkdir(parents=True, exist_ok=True)
nb_utils.write_doubletons_csv(true_doubletons, true_doubletons_path)
ascertained_estimates = {}
true_doubleton_estimates = {}
rows = []
for multiplier in error_multipliers:
    zarr_path = (
        zarr_dir / f"{zarr_prefix}-geno-{multiplier:.1f}-phase0.0-mispol0.0.zarr"
    )
    ascertained_estimate = nb_utils.estimate_error_rate(
        zarr_path, recombination=recombination, config=config
    )
    true_doubleton_estimate = nb_utils.estimate_error_rate(
        zarr_path,
        recombination=recombination,
        config=true_doubletons_config,
        fixed_doubletons_path=true_doubletons_path,
    )
    ascertained_estimates[multiplier] = ascertained_estimate
    true_doubleton_estimates[multiplier] = true_doubleton_estimate
    rows.append(
        {
            "error_multiplier": multiplier,
            "diversity": ascertained_estimate.diversity.global_pi_per_bp,
            "epsilon": ascertained_estimate.fit.epsilon,
            "mu": ascertained_estimate.fit.mu,
            "pi": ascertained_estimate.fit.pi,
            "sigma_sq": ascertained_estimate.fit.sigma_sq,
            "true_doubleton_epsilon": true_doubleton_estimate.fit.epsilon,
            "true_doubleton_mu": true_doubleton_estimate.fit.mu,
            "true_doubleton_pi": true_doubleton_estimate.fit.pi,
            "true_doubleton_sigma_sq": true_doubleton_estimate.fit.sigma_sq,
            "true_error": nb_utils.get_true_error_rate(zarr_path),
            "true_diversity": true_diversity,
        }
    )
fit_df = pd.DataFrame(rows)
fit_df


# %%

# %%
fit_df.to_csv("data/ch4/validation/dphil-analysis-results.csv", index=None)

# %%
nb_plot.plot_error_estimates(fit_df, "Estimated (ascertained doubletons)")


# %%
## Doubleton ascertainment testing


# %%
cutoff_proportions = [0.1, 0.25, 0.5, 0.75]
L_mismatch_trim_values = [config.window_sizes[0], config.window_sizes[-1]]
classification_records = []
true_doubleton_masks = {}
for multiplier, estimate in ascertained_estimates.items():
    true_doubleton_mask = nb_utils.get_true_doubleton_mask(
        estimate.doubletons, zero_error_zarr_path
    )
    true_doubleton_masks[multiplier] = true_doubleton_mask
    records = nb_utils.evaluate_mismatch_cutoffs(
        estimate.mismatches,
        true_doubleton_mask,
        cutoff_proportions=cutoff_proportions,
        L_mismatch_trim_values=L_mismatch_trim_values,
    )
    for record in records:
        record["error_multiplier"] = multiplier
    classification_records.extend(records)
classification_df = pd.DataFrame(classification_records)

# %%
classification_df

# %% [markdown]
# ### Mismatch rate as a doubleton classifier

# %%
fig = nb_plot.plot_mismatch_rate_as_a_doubleton_classifier(
    L_mismatch_trim_values=L_mismatch_trim_values,
    ascertained_estimates=ascertained_estimates,
    config=config,
    error_multipliers=error_multipliers,
    true_doubleton_masks=true_doubleton_masks,
    figsize=(16, 10),
    linestyle="--",
    color="0.7",
    fontsize=8,
)

# %%
prevalence_df = (
    classification_df[["error_multiplier", "prevalence"]]
    .drop_duplicates()
    .sort_values("error_multiplier")
)
fig = nb_plot.plot_mismatch_rate_as_a_doubleton_classifier_2(
    prevalence_df=prevalence_df, figsize=(6, 4.5), marker="o"
)

# %%
metric_specs = [
    ("precision", "Precision (PPV)"),
    ("tpr", "Recall (TPR / sensitivity)"),
    ("fpr", "False positive rate"),
    ("tnr", "Specificity (TNR)"),
    ("npv", "Negative predictive value"),
    ("accuracy", "Classification accuracy"),
]
fig = nb_plot.plot_mismatch_rate_as_a_doubleton_classifier_3(
    L_mismatch_trim_values=L_mismatch_trim_values,
    classification_df=classification_df,
    cutoff_proportions=cutoff_proportions,
    metric_specs=metric_specs,
    figsize=(12, 11),
    marker="o",
    fontsize=8,
)

# %% [markdown]
# ### Cumulative counts by doubleton type

# %%

# %%

# %%
profile_distances = np.geomspace(100, config.window_sizes[-1], 60)
cumulative_profiles = {}
for multiplier in [1, 5, 10]:
    profiles = nb_utils.cumulative_mismatch_profiles(
        ascertained_estimates[multiplier],
        zero_error_zarr_path,
        distances=profile_distances,
    )
    cumulative_profiles[multiplier] = profiles
    profile_fig = nb_plot.plot_cumulative_mismatch_profiles(
        profiles, f"{multiplier}x simulated genotype error: one-sided mismatch profiles"
    )
    ipython_display.display(profile_fig)
    nb_plot.close_figure(profile_fig)

# %% [markdown]
# ### Per-doubleton mismatch summaries

# %%
for multiplier in [1, 5, 10]:
    summary_fig = nb_plot.plot_doubleton_mismatch_summaries(
        cumulative_profiles[multiplier],
        f"{multiplier}x simulated genotype error: mismatch summaries",
    )
    ipython_display.display(summary_fig)
    nb_plot.close_figure(summary_fig)

# %% [markdown]
# ## Fitting models to doubleton data

# %%
for multiplier in [0, 2]:
    cumulative_profiles[multiplier] = nb_utils.cumulative_mismatch_profiles(
        ascertained_estimates[multiplier],
        zero_error_zarr_path,
        distances=profile_distances,
    )
classifier_tiers = ["tier_0", "tier_1", "tier_1b", "tier_2", "tier_3"]
classifier_labels = {
    "tier_0": "Fixed 25% mismatch trim",
    "tier_1": "First-mismatch threshold",
    "tier_1b": "Clean-count threshold",
    "tier_2": "Two-feature logistic",
    "tier_3": "Logistic with interaction",
}
training_multipliers = [1, 5, 10]
classifier_rows = []
classifier_curves = {}
held_out_models = {}
for held_out_multiplier in training_multipliers:
    training_profiles = [
        cumulative_profiles[multiplier]
        for multiplier in training_multipliers
        if multiplier != held_out_multiplier
    ]
    test_profiles = cumulative_profiles[held_out_multiplier]
    estimate = ascertained_estimates[held_out_multiplier]
    true_epsilon = fit_df.loc[
        fit_df["error_multiplier"] == held_out_multiplier, "true_error"
    ].iloc[0]
    for tier in classifier_tiers:
        if tier == "tier_0":
            model = nb_utils.make_doubleton_classifier(tier=tier, threshold=None)
        else:
            model = nb_utils.fit_doubleton_classifier(tier, training_profiles)
        held_out_models[held_out_multiplier, tier] = model
        scores, retained = nb_utils.classify_doubletons(
            model, test_profiles, estimate.mismatches
        )
        metrics = nb_utils.classifier_metrics(
            scores, test_profiles.is_true_doubleton, retained
        )
        fitted = nb_utils.fit_error_rate_with_selected_doubletons(estimate, retained)
        metrics.update(
            {
                "tier": tier,
                "test_multiplier": held_out_multiplier,
                "fitted_epsilon": fitted.epsilon,
                "epsilon_relative_error": abs(fitted.epsilon - true_epsilon)
                / true_epsilon,
                "fitted_mu": fitted.mu,
                "fitted_sigma_sq": fitted.sigma_sq,
            }
        )
        classifier_rows.append(metrics)
        classifier_curves[held_out_multiplier, tier] = nb_utils.classifier_curves(
            scores, test_profiles.is_true_doubleton
        )
classifier_cv_df = pd.DataFrame(classifier_rows)
classifier_cv_summary = classifier_cv_df.groupby("tier", sort=False)[
    [
        "auroc",
        "average_precision",
        "tpr",
        "fpr",
        "precision",
        "fraction_retained",
        "epsilon_relative_error",
    ]
].mean()
ipython_display.display(classifier_cv_df)
ipython_display.display(classifier_cv_summary)

# %%
fig = nb_plot.plot_fitting_models_to_doubleton_data(
    classifier_curves=classifier_curves,
    classifier_labels=classifier_labels,
    classifier_tiers=classifier_tiers,
    cumulative_profiles=cumulative_profiles,
    training_multipliers=training_multipliers,
    figsize=(16, 9),
    color="0.7",
    linestyle="--",
    fontsize=8,
)
fig

# %%
fig = nb_plot.plot_fitting_models_to_doubleton_data_2(
    classifier_cv_df=classifier_cv_df,
    classifier_labels=classifier_labels,
    classifier_tiers=classifier_tiers,
    training_multipliers=training_multipliers,
    figsize=(11, 8),
    marker="o",
    fontsize=8,
)
fig

# %% [markdown]
# ### Interpolation at 2x and fitted decision surfaces

# %%
pooled_training_profiles = [
    cumulative_profiles[multiplier] for multiplier in training_multipliers
]
pooled_models = {}
interpolation_rows = []
interpolation_curves = {}
interpolation_profiles = cumulative_profiles[2]
interpolation_estimate = ascertained_estimates[2]
interpolation_true_error = fit_df.loc[
    fit_df["error_multiplier"] == 2, "true_error"
].iloc[0]
for tier in classifier_tiers:
    if tier == "tier_0":
        model = nb_utils.make_doubleton_classifier(tier=tier, threshold=None)
    else:
        model = nb_utils.fit_doubleton_classifier(tier, pooled_training_profiles)
    pooled_models[tier] = model
    scores, retained = nb_utils.classify_doubletons(
        model, interpolation_profiles, interpolation_estimate.mismatches
    )
    metrics = nb_utils.classifier_metrics(
        scores, interpolation_profiles.is_true_doubleton, retained
    )
    fitted = nb_utils.fit_error_rate_with_selected_doubletons(
        interpolation_estimate, retained
    )
    metrics.update(
        {
            "tier": tier,
            "fitted_epsilon": fitted.epsilon,
            "epsilon_relative_error": abs(fitted.epsilon - interpolation_true_error)
            / interpolation_true_error,
        }
    )
    interpolation_rows.append(metrics)
    interpolation_curves[tier] = nb_utils.classifier_curves(
        scores, interpolation_profiles.is_true_doubleton
    )
interpolation_df = pd.DataFrame(interpolation_rows)
ipython_display.display(interpolation_df)
fig = nb_plot.plot_interpolation_at_2x_and_fitted_decision_surfaces(
    classifier_labels=classifier_labels,
    classifier_tiers=classifier_tiers,
    interpolation_curves=interpolation_curves,
    interpolation_profiles=interpolation_profiles,
    figsize=(11, 4.5),
    color="0.7",
    linestyle="--",
    fontsize=8,
)
fig

# %%
interpolation_features = nb_utils.doubleton_features(interpolation_profiles)
horizontal = np.linspace(
    interpolation_features[:, 0].min(), interpolation_features[:, 0].max(), 120
)
vertical = np.linspace(
    interpolation_features[:, 1].min(), interpolation_features[:, 1].max(), 120
)
grid_x, grid_y = np.meshgrid(horizontal, vertical)
grid_features = np.column_stack((grid_x.ravel(), grid_y.ravel()))
fig = nb_plot.plot_interpolation_at_2x_and_fitted_decision_surfaces_2(
    classifier_labels=classifier_labels,
    grid_features=grid_features,
    grid_x=grid_x,
    grid_y=grid_y,
    interpolation_features=interpolation_features,
    interpolation_profiles=interpolation_profiles,
    pooled_models=pooled_models,
    figsize=(12, 5),
    cmap="RdBu",
    alpha=0.6,
    colors="black",
    linewidths=2,
    s=3,
    alpha_2=0.08,
    fontsize=8,
    label="Fitted probability of a true doubleton",
)
fig

# %% [markdown]
# ### Selected classifier and downstream error model
#
# Tier 3 is selected: it improves the downstream epsilon fit in all three
# held-out error-level comparisons. Its 2x interpolation result is effectively
# tied with Tier 2 (relative epsilon error 0.21% versus 0.22%), but the
# consistent held-out gain supports the interaction.
# The learned threshold targets 95% recall in pooled training data, but recall
# varies substantially across held-out error levels. At 10x, the pooled model
# retains only 78.7% of true doubletons and still underestimates epsilon by
# about 33%. These error levels share one underlying simulation, so this
# comparison is not independent validation.

# %%
selected_tier = "tier_3"
selected_model = pooled_models[selected_tier]
selected_fit_rows = []
selected_classification_rows = []
for multiplier in error_multipliers:
    estimate = ascertained_estimates[multiplier]
    profiles = cumulative_profiles[multiplier]
    scores, retained = nb_utils.classify_doubletons(
        selected_model, profiles, estimate.mismatches
    )
    fitted = nb_utils.fit_error_rate_with_selected_doubletons(estimate, retained)
    selected_fit_rows.append(
        {
            "error_multiplier": multiplier,
            "epsilon": fitted.epsilon,
            "mu": fitted.mu,
            "sigma_sq": fitted.sigma_sq,
        }
    )
    classification_row = {
        "error_multiplier": multiplier,
        "retained_doubletons": int(retained.sum()),
        "total_doubletons": len(retained),
        "fraction_retained": retained.mean(),
    }
    truth = profiles.is_true_doubleton
    if truth.any() and (~truth).any():
        metrics = nb_utils.classifier_metrics(scores, truth, retained)
        classification_row.update(
            {
                "tpr": metrics["tpr"],
                "fpr": metrics["fpr"],
                "precision": metrics["precision"],
            }
        )
    selected_classification_rows.append(classification_row)
selected_fit_df = fit_df.drop(columns=["epsilon", "mu", "sigma_sq"]).merge(
    pd.DataFrame(selected_fit_rows), on="error_multiplier", validate="one_to_one"
)
selected_classification_df = pd.DataFrame(selected_classification_rows)
ipython_display.display(selected_classification_df)
ipython_display.display(
    selected_fit_df[
        [
            "error_multiplier",
            "true_error",
            "epsilon",
            "true_doubleton_epsilon",
            "mu",
            "sigma_sq",
        ]
    ]
)

# %%
nb_plot.plot_error_estimates(
    selected_fit_df,
    "Estimated (Tier 3 retained doubletons)",
    baseline_frame=fit_df,
    title="Error-model fits after Tier 3 doubleton selection",
)

# %%
diagnostic_profiles = cumulative_profiles[10]
diagnostic_features = nb_utils.doubleton_features(diagnostic_profiles)
diagnostic_scores, diagnostic_retained = nb_utils.classify_doubletons(
    selected_model, diagnostic_profiles, ascertained_estimates[10].mismatches
)
diagnostic_truth = diagnostic_profiles.is_true_doubleton
outcomes = [
    (diagnostic_retained & diagnostic_truth, "TP: retained true", "tab:blue"),
    (diagnostic_retained & ~diagnostic_truth, "FP: retained false", "tab:orange"),
    (~diagnostic_retained & diagnostic_truth, "FN: excluded true", "tab:red"),
    (~diagnostic_retained & ~diagnostic_truth, "TN: excluded false", "tab:green"),
]
fig = nb_plot.plot_selected_classifier_and_downstream_error_model(
    diagnostic_features=diagnostic_features,
    outcomes=outcomes,
    selected_model=selected_model,
    figsize=(8, 6),
    colors="black",
    s=4,
    alpha=0.16,
    fontsize=8,
)
fig

# %% [markdown]
# ## Real data
#
# The requested TGP density mask does not itself remove multiallelic rows that
# share a position. Combine it with the store's duplicate-position mask so the
# estimator receives one biallelic row per genomic position. The local chr17
# map copy extends the source map's terminal zero-rate interval to the end of
# the Zarr; the map README records this assumption.

# %%
tgp_zarr_path = repository_dir / "data/general/zarr_vcfs/tgp/chr17/data.zarr"
tgp_recombination = (
    repository_dir
    / "data/general/HapMapII_GRCh38/genetic_map_Hg38_chr17_for_tgp_zarr.txt"
)
tgp_variant_masks = [
    "variant_all_subset_chr17p_region_filterNton23_site_density_threshold_sites_per_kbp_5_window_size_100000_mask",
    "variant_duplicate_position_mask",
]
tgp_estimate = nb_utils.estimate_error_rate(
    tgp_zarr_path,
    recombination=tgp_recombination,
    config=config,
    variant_mask_name=tgp_variant_masks,
)
tgp_estimate.fit.epsilon

# %% [markdown]
# The Johnsson et al. (2021) sex-averaged map uses Sscrofa11.1 coordinates.
# The local derived copy extends its terminal zero-rate interval to the final
# coordinate in this Zarr; `data/ch4/pig_recombination_map/README.txt` records the
# source and the exact adjustment.

# %%
pig_window_sizes = np.unique(np.r_[np.geomspace(1000, 1000000, 50), 100000.0])
config2 = nb_utils.make_estimation_config(
    window_sizes=pig_window_sizes,
    num_doubletons=10000,
    random_seed=42,
    exclude_high_mismatch_proportion=0.25,
    L_mismatch_trim=pig_window_sizes[-1],
)
pig_zarr_path = repository_dir / "data/general/zarr_vcfs/pigs/chr18/data.zarr"
pig_recombination = (
    repository_dir
    / "data/ch4/pig_recombination_map/Johnsson_2021_sex_averaged_chr18_for_data_zarr.txt"
)
pig_variant_mask = "variant_allSample_subset_chr18_region_filterDensity_site_density_threshold_sites_per_kbp_10_window_size_1000_mask"
pig_estimate = nb_utils.estimate_error_rate(
    pig_zarr_path,
    recombination=pig_recombination,
    config=config2,
    variant_mask_name=pig_variant_mask,
)
pig_estimate.fit.epsilon

# %% [markdown]
# ### Pig chr18 site density and retained quality flags
#
# The current density-threshold mask is applied on its own. Show how much of
# chr18 it retains, and which other Zarr quality flags remain among those sites.

# %%
pig_group = zarr.open_group(pig_zarr_path, mode="r")
pig_all_positions = np.asarray(pig_group["variant_position"][:], dtype=float)
pig_excluded = np.asarray(pig_group[pig_variant_mask][:], dtype=bool)
pig_positions = pig_all_positions[~pig_excluded]
pig_bin_width = 100000
pig_bins = np.arange(
    0,
    np.ceil(pig_all_positions[-1] / pig_bin_width) * pig_bin_width + pig_bin_width,
    pig_bin_width,
)
pig_bin_midpoints_mb = (pig_bins[:-1] + pig_bins[1:]) / 2000000.0
all_sites_per_bin, _ = np.histogram(pig_all_positions, bins=pig_bins)
retained_sites_per_bin, _ = np.histogram(pig_positions, bins=pig_bins)
pig_quality_masks = {
    "Non-SNP": "variant_not_snps_mask",
    "Low-quality ancestral allele": "variant_low_quality_ancestral_allele_mask",
    "Bad ancestral allele": "variant_bad_ancestral_mask",
    "Dataset filterDensity mask": "variant_allSample_subset_chr18_region_filterDensity_mask",
}
pig_quality_rows = []
pig_flag_rates = {}
for label, mask_name in pig_quality_masks.items():
    flagged = np.asarray(pig_group[mask_name][:], dtype=bool)[~pig_excluded]
    flagged_counts, _ = np.histogram(pig_positions[flagged], bins=pig_bins)
    pig_flag_rates[label] = np.divide(
        flagged_counts,
        retained_sites_per_bin,
        out=np.zeros_like(flagged_counts, dtype=float),
        where=retained_sites_per_bin > 0,
    )
    pig_quality_rows.append(
        {
            "flag": label,
            "retained_sites_flagged": np.count_nonzero(flagged),
            "fraction_of_retained_sites": flagged.mean(),
        }
    )
pig_quality_df = pd.DataFrame(pig_quality_rows)
ipython_display.display(pig_quality_df)
fig = nb_plot.plot_pig_chr18_site_density_and_retained_quality_flags(
    all_sites_per_bin=all_sites_per_bin,
    pig_bin_midpoints_mb=pig_bin_midpoints_mb,
    pig_excluded=pig_excluded,
    pig_flag_rates=pig_flag_rates,
    pig_positions=pig_positions,
    retained_sites_per_bin=retained_sites_per_bin,
    figsize=(13, 7),
    label="All Zarr sites",
    fontsize=8,
)
fig

# %%
pig_mask_comparisons = {"Current density-threshold mask": pig_estimate}
pig_additional_masks = {
    "Plus dataset density-filter flag": [
        pig_variant_mask,
        "variant_allSample_subset_chr18_region_filterDensity_mask",
    ],
    "Plus non-SNP and ancestral flags": [
        pig_variant_mask,
        "variant_not_snps_mask",
        "variant_bad_ancestral_mask",
        "variant_low_quality_ancestral_allele_mask",
    ],
}
for label, masks in pig_additional_masks.items():
    pig_mask_comparisons[label] = nb_utils.estimate_error_rate(
        pig_zarr_path,
        recombination=pig_recombination,
        config=config2,
        variant_mask_name=masks,
    )
pig_mask_rows = []
for label, estimate in pig_mask_comparisons.items():
    pig_mask_rows.append(
        {
            "site_selection": label,
            "num_sites": estimate.diversity.num_sites,
            "eligible_doubletons": estimate.doubletons.num_eligible,
            "epsilon": estimate.fit.epsilon,
            "mu": estimate.fit.mu,
            "sigma_sq": estimate.fit.sigma_sq,
        }
    )
pig_mask_comparison_df = pd.DataFrame(pig_mask_rows)
ipython_display.display(pig_mask_comparison_df)

# %% [markdown]
# ### Pig fit sensitivity to doubleton trimming
#
# Refit the same sampled doubletons and mismatch windows, ranking exclusions
# by the observed count at 100 kb. This isolates trimming from data sampling.

# %%
pig_trim_proportions = [0, 0.1, 0.25, 0.5, 0.75]
pig_trim_L = pig_window_sizes[-1]
pig_trim_rows = []
for proportion in pig_trim_proportions:
    trim_config = dataclasses.replace(
        config2, exclude_high_mismatch_proportion=proportion, L_mismatch_trim=pig_trim_L
    )
    fit = nb_utils.fit_error_model(
        pig_estimate.mismatches, pi=pig_estimate.fit.pi, config=trim_config
    )
    pig_trim_rows.append(
        {
            "excluded_proportion": proportion,
            "num_doubletons": fit.num_doubletons,
            "epsilon": fit.epsilon,
            "mu": fit.mu,
            "sigma_sq": fit.sigma_sq,
            "objective": fit.objective,
        }
    )
pig_trim_df = pd.DataFrame(pig_trim_rows)
ipython_display.display(pig_trim_df)
fig = nb_plot.plot_pig_fit_sensitivity_to_doubleton_trimming(
    pig_trim_df=pig_trim_df,
    pig_trim_proportions=pig_trim_proportions,
    figsize=(14, 4),
    marker="o",
)


# %%
fig = nb_plot.plot_pig_fit_sensitivity_to_doubleton_trimming_2(
    pig_estimate=pig_estimate,
    pig_window_sizes=pig_window_sizes,
    figsize=(11, 4),
    marker=".",
    label="Observed",
    label_2="Fitted",
    color="0.5",
    linestyle="--",
)
fig

# %% [markdown]
# ### Pig doubleton mismatches and local site density

# %%
pig_trim_index = np.flatnonzero(pig_window_sizes == pig_trim_L)[0]
pig_trim_counts = pig_estimate.mismatches.counts[pig_trim_index]
pig_focal_positions = pig_estimate.doubletons.positions
pig_local_left = np.searchsorted(
    pig_positions, pig_focal_positions - pig_trim_L, side="left"
)
pig_local_right = np.searchsorted(
    pig_positions, pig_focal_positions + pig_trim_L, side="right"
)
pig_local_sites_per_kb = (pig_local_right - pig_local_left) / (2 * pig_trim_L / 1000)
pig_raw_focal_indices = np.flatnonzero(~pig_excluded)[
    pig_estimate.doubletons.site_indices
]
pig_focal_quality_rows = []
for label, mask_name in pig_quality_masks.items():
    flagged = np.asarray(pig_group[mask_name][:], dtype=bool)[pig_raw_focal_indices]
    pig_focal_quality_rows.append(
        {
            "flag": label,
            "flagged_doubletons": np.count_nonzero(flagged),
            "median_mismatches_flagged": np.median(pig_trim_counts[flagged]),
            "median_mismatches_unflagged": np.median(pig_trim_counts[~flagged]),
        }
    )
pig_focal_quality_df = pd.DataFrame(pig_focal_quality_rows)
ipython_display.display(pig_focal_quality_df)
fig = nb_plot.plot_pig_doubleton_mismatches_and_local_site_density(
    pig_local_sites_per_kb=pig_local_sites_per_kb,
    pig_trim_counts=pig_trim_counts,
    figsize=(12, 4),
    gridsize=45,
    bins="log",
    label="Number of doubletons",
)
fig

# %% [markdown]
# The current mask retains nearly all sites across 0.06–55.89 Mb. Several
# quality flags remain among the retained rows, and flagged focal doubletons
# tend to have higher mismatch counts at 100 kb. Local site density also varies
# markedly, but the scatter shows a broad range of mismatch counts at any given
# density. The aggregate fitted curve follows the observed mean closely while
# the fitted parameters change by orders of magnitude under trimming; the
# residual objective also rises sharply with the most aggressive exclusions.
# Removing the additional flagged site sets reduces the fitted variance only
# modestly.
# Those mask comparisons also resample eligible doubletons and recalculate
# diversity, so they are sensitivity checks rather than isolated flag effects.
# These checks identify heterogeneity and fit sensitivity; they do not by
# themselves distinguish technical error from population structure or a model
# mismatch.

# %% [markdown]
# ## Using doubleton classifiers on real data
#
# Doubletons are sampled from each dataset's masked site list. Verify that the
# selected-to-raw Zarr mapping used for one-sided features is the same mapping
# used for the original mismatch summary. Then reuse each estimate's sampled
# doubletons, global diversity and recombination rates for every classifier.

# %%
real_datasets = {
    "TGP chr17": (tgp_estimate, tgp_zarr_path, tgp_variant_masks),
    "Pig chr18": (pig_estimate, pig_zarr_path, [pig_variant_mask]),
}
real_profiles = {}
real_selections = {}
real_rows = []
for dataset, (estimate, zarr_path, mask_names) in real_datasets.items():
    group = zarr.open_group(zarr_path, mode="r")
    excluded_variants = np.zeros(group["variant_position"].shape[0], dtype=bool)
    for mask_name in mask_names:
        excluded_variants |= np.asarray(group[mask_name][:], dtype=bool)
    expected_raw_indices = np.flatnonzero(~excluded_variants)
    assert np.array_equal(estimate.included_variant_indices, expected_raw_indices)
    raw_positions = np.asarray(group["variant_position"][:])
    assert np.array_equal(
        estimate.included_positions, raw_positions[expected_raw_indices]
    )
    assert np.array_equal(
        estimate.doubletons.positions,
        estimate.included_positions[estimate.doubletons.site_indices],
    )
    max_L = estimate.config.window_sizes[-1]
    profiles = nb_utils.cumulative_mismatch_profiles(
        estimate, distances=np.array([max_L])
    )
    real_profiles[dataset] = profiles
    assert profiles.is_true_doubleton is None
    assert profiles.max_clean_count.shape == estimate.doubletons.positions.shape
    assert (
        profiles.max_first_mismatch_distance.shape
        == estimate.doubletons.positions.shape
    )
    assert np.array_equal(
        profiles.max_left_count + profiles.max_right_count,
        estimate.mismatches.counts[-1],
    )
    real_selections[dataset] = {}
    for tier in classifier_tiers:
        model = pooled_models[tier]
        scores, retained = nb_utils.classify_doubletons(
            model,
            profiles,
            estimate.mismatches,
            baseline_excluded=estimate.config.exclude_high_mismatch_proportion,
        )
        assert len(retained) == len(estimate.doubletons.positions)
        real_selections[dataset][tier] = retained
        if tier == "tier_0":
            fitted = estimate.fit
            assert fitted.num_doubletons == retained.sum()
        else:
            fitted = nb_utils.fit_error_rate_with_selected_doubletons(
                estimate, retained
            )
        real_rows.append(
            {
                "dataset": dataset,
                "tier": tier,
                "sampled_doubletons": len(retained),
                "retained_doubletons": int(retained.sum()),
                "excluded_doubletons": int((~retained).sum()),
                "epsilon": fitted.epsilon,
                "diversity": estimate.diversity.global_pi_per_bp,
                "mu": fitted.mu,
                "sigma_sq": fitted.sigma_sq,
                "fit_success": fitted.success,
                "first_mismatch_censored": int(
                    profiles.max_first_mismatch_censored.sum()
                ),
            }
        )
real_classifier_df = pd.DataFrame(real_rows)
ipython_display.display(real_classifier_df)

# %% [markdown]
# ### Exclusions by predictor
#
# Tier 1 uses mismatch-free distance; Tier 1b uses clean-side count. The
# overlap partitions describe their individual rules, while Tiers 2 and 3
# combine the predictors and have no unique per-predictor attribution.

# %%
fig = nb_plot.plot_exclusions_by_predictor(
    classifier_labels=classifier_labels,
    classifier_tiers=classifier_tiers,
    real_classifier_df=real_classifier_df,
    real_datasets=real_datasets,
    real_selections=real_selections,
    figsize=(14, 8),
    fontsize=8,
)
fig

# %%
feature_sets = {
    "Training simulations (1x, 5x, 10x)": np.concatenate(
        [nb_utils.doubleton_features(profiles) for profiles in pooled_training_profiles]
    )
}
for dataset, profiles in real_profiles.items():
    feature_sets[dataset] = nb_utils.doubleton_features(profiles)
fig = nb_plot.plot_exclusions_by_predictor_2(
    feature_sets=feature_sets, figsize=(12, 4.5), fontsize=8
)

# %% [markdown]
# ### Error-model estimates by tier
#
# Diversity uses all variants retained by the dataset's site masks, so it is
# the same for every doubleton classifier. Tier 1b remains in the results
# table above as an independent predictor check; these figures show the four
# numbered tiers requested for comparison.
# The real-data predictor distributions differ from those in the training
# simulations. In particular, the pig clean-count rule excludes many more
# doubletons than the distance-only rule. The large changes in pig epsilon and
# sigma-squared across tiers are fit sensitivity, not validation of the
# simulation-trained classifier on that dataset. With the current masks, Tier
# 3 retains 8,369 TGP and 4,454 pig doubletons; the corresponding epsilon fits
# are approximately 1.08e-5 and 1.71e-5 errors per haplotype-bp. TGP has
# 1,008 first-mismatch distances censored at 1 Mb, compared with 10 in pig.
# These shifts may reflect genotype quality, population structure, or model
# transfer; they do not identify the real error rate without external truth.

# %%
for dataset in real_datasets:
    ipython_display.display(
        nb_plot.plot_real_classifier_estimates(real_classifier_df, dataset)
    )

# %% [markdown]
# ### Real Tier 3 epsilon against simulation estimates
#
# The horizontal lines show the fitted values on the real datasets. The
# simulated multiplier is a reference axis, not an inferred real-data error
# multiplier.

# %%
fig = nb_plot.plot_real_tier_3_epsilon_against_simulation_estimates(
    fit_df=fit_df,
    real_classifier_df=real_classifier_df,
    selected_fit_df=selected_fit_df,
    figsize=(13, 5),
    marker="o",
    markersize=3,
    linestyle="--",
    fontsize=8,
)
fig

# %%

# %%

# %%
