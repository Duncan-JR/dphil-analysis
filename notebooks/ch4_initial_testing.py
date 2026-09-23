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
from pathlib import Path

import error_estimation
import error_validation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
zarr_dir = Path(
    "~/work/tsinfer-anc-eval/data/error_eval/zarr_vcfs/"
).expanduser()
zarr_prefix = "OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0"
ts_path = Path(
    "~/work/tsinfer-anc-eval/data/error_eval/simulated/"
    "OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0.trees"
).expanduser()
#zarr_dir = Path(
#    "~/work/tsinfer-anc-eval/data/anc_eval/zarr_vcfs/"
#).expanduser()
#zarr_prefix = "OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0"
#ts_path = Path(
#    "~/work/tsinfer-anc-eval/data/anc_eval/simulated/"
#    "OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0.trees"
#).expanduser()

recombination = Path(
    "~/work/tsinfer-paper/data/HapMapII_GRCh38/"
    "genetic_map_Hg38_chr17.txt"
).expanduser()

window_sizes = np.geomspace(100, 1_000_000, 100)
config = error_estimation.EstimationConfig(
    window_sizes=window_sizes,
    num_doubletons=10_000,
    random_seed=42,
    exclude_high_mismatch_proportion=0.25,
    L_mismatch_trim=window_sizes[-1],
)


# %%
error_multipliers = [0, 1, 2, 5, 10]
true_diversity = error_validation.get_ts_diversity(ts_path)
zero_error_zarr_path = zarr_dir / (
    f"{zarr_prefix}-geno-0.0-phase0.0-mispol0.0.zarr"
)
true_doubletons_config = dataclasses.replace(
    config,
    num_doubletons=7_500,
    exclude_high_mismatch_proportion=0,
)
true_doubletons = error_validation.sample_doubletons(
    zero_error_zarr_path,
    true_doubletons_config,
)
true_doubletons_path = Path("../data/tmp/true_doubletons.csv")
true_doubletons_path.parent.mkdir(parents=True, exist_ok=True)
error_validation.write_doubletons_csv(true_doubletons, true_doubletons_path)

ascertained_estimates = {}
true_doubleton_estimates = {}
rows = []

for multiplier in error_multipliers:
    zarr_path = zarr_dir / (
        f"{zarr_prefix}-geno-{multiplier:.1f}-phase0.0-mispol0.0.zarr"
    )
    ascertained_estimate = error_estimation.estimate_error_rate(
        zarr_path,
        recombination=recombination,
        config=config,
    )
    true_doubleton_estimate = error_estimation.estimate_error_rate(
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
            "true_error": error_validation.get_true_error_rate(zarr_path),
            "true_diversity": true_diversity,
        }
    )

fit_df = pd.DataFrame(rows)
fit_df


# %%

# %%
fit_df.to_csv("../data/tmp/dphil-analysis-results.csv", index=None)

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

axes[0, 0].plot(
    fit_df["error_multiplier"],
    fit_df["true_error"],
    marker="o",
    label="True",
)
axes[0, 0].plot(
    fit_df["error_multiplier"],
    fit_df["epsilon"],
    marker="o",
    label="Estimated (ascertained doubletons)",
)
axes[0, 0].plot(
    fit_df["error_multiplier"],
    fit_df["true_doubleton_epsilon"],
    marker="o",
    label="Estimated (true doubletons)",
)
axes[0, 0].set_ylabel("Errors per haplotype-bp")
axes[0, 0].set_title("Genotype error rate")
axes[0, 0].legend()

axes[0, 1].plot(
    fit_df["error_multiplier"],
    fit_df["true_diversity"],
    marker="o",
    label="True",
)
axes[0, 1].plot(
    fit_df["error_multiplier"],
    fit_df["diversity"],
    marker="o",
    label="Estimated (ascertained doubletons)",
)
axes[0, 1].plot(
    fit_df["error_multiplier"],
    fit_df["true_doubleton_pi"],
    marker="o",
    label="Estimated (true doubletons)",
)
axes[0, 1].set_ylabel("Diversity per bp")
axes[0, 1].set_title("Diversity")
axes[0, 1].legend()

axes[1, 0].plot(
    fit_df["error_multiplier"],
    fit_df["mu"],
    marker="o",
    label="Estimated (ascertained doubletons)",
)
axes[1, 0].plot(
    fit_df["error_multiplier"],
    fit_df["true_doubleton_mu"],
    marker="o",
    label="Estimated (true doubletons)",
)
axes[1, 0].set_ylabel(r"Estimated $\mu$")
axes[1, 0].set_title("Gamma mean")
axes[1, 0].legend()

axes[1, 1].plot(
    fit_df["error_multiplier"],
    fit_df["sigma_sq"],
    marker="o",
    label="Estimated (ascertained doubletons)",
)
axes[1, 1].plot(
    fit_df["error_multiplier"],
    fit_df["true_doubleton_sigma_sq"],
    marker="o",
    label="Estimated (true doubletons)",
)
axes[1, 1].set_ylabel(r"Estimated $\sigma^2$")
axes[1, 1].set_title("Gamma variance")
axes[1, 1].legend()

for ax in axes.flat:
    ax.set_xlabel("Genotype error-rate multiplier")


# %% [markdown]
# ## Doubleton ascertainment testing

# %%
cutoff_proportions = [0.10, 0.25, 0.50,0.75]
L_mismatch_trim_values = [config.window_sizes[0], config.window_sizes[-1]]
classification_records = []
true_doubleton_masks = {}
for multiplier, estimate in ascertained_estimates.items():
    true_doubleton_mask = error_validation.get_true_doubleton_mask(
        estimate.doubletons,
        zero_error_zarr_path,
    )
    true_doubleton_masks[multiplier] = true_doubleton_mask
    records = error_validation.evaluate_mismatch_cutoffs(
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
fig, axes = plt.subplots(
    3,
    len(error_multipliers),
    figsize=(16, 10),
    sharex="row",
    sharey="row",
    constrained_layout=True,
)

for row, L_mismatch_trim in enumerate(L_mismatch_trim_values):
    window_index = list(config.window_sizes).index(L_mismatch_trim)
    for col, multiplier in enumerate(error_multipliers):
        ax = axes[row, col]
        estimate = ascertained_estimates[multiplier]
        true_mask = true_doubleton_masks[multiplier]
        mismatch_rate = (
            estimate.mismatches.counts[window_index]
            / (2 * L_mismatch_trim)
        )
        for label, select in [
            ("True doubletons", true_mask),
            ("False doubletons", ~true_mask),
        ]:
            values = sorted(mismatch_rate[select])
            if len(values) > 0:
                cumulative = (pd.RangeIndex(len(values)) + 1) / len(values)
                ax.plot(values, cumulative, label=label)
        ax.set_title(f"{multiplier:g}x error, L={L_mismatch_trim:g}")
        ax.set_xlabel("Mismatches per bp")

for col, multiplier in enumerate(error_multipliers):
    ax = axes[2, col]
    estimate = ascertained_estimates[multiplier]
    true_mask = true_doubleton_masks[multiplier]
    window_index = list(config.window_sizes).index(L_mismatch_trim_values[-1])
    mismatch_rate = (
        estimate.mismatches.counts[window_index]
        / (2 * L_mismatch_trim_values[-1])
    )
    roc = error_validation.mismatch_roc(mismatch_rate, true_mask)
    if roc is not None:
        fpr, tpr, auroc = roc
        ax.plot(fpr, tpr)
        ax.annotate(
            f"AUROC = {auroc:.3f}",
            (0.96, 0.04),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
        )
    else:
        ax.annotate("ROC undefined: one class absent", (0.05, 0.05),
                    xycoords="axes fraction")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.7")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{multiplier:g}x error, ROC at L={L_mismatch_trim_values[-1]:g}")
    ax.set_xlabel("False positive rate")

for ax in axes[:2, 0]:
    ax.set_ylabel("Empirical cumulative proportion")
axes[2, 0].set_ylabel("True positive rate")
axes[0, 0].legend(fontsize=8)

# %%
prevalence_df = (
    classification_df[
        ["error_multiplier", "prevalence"]
    ]
    .drop_duplicates()
    .sort_values("error_multiplier")
)

fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
ax.plot(
    prevalence_df["error_multiplier"],
    prevalence_df["prevalence"],
    marker="o",
)
ax.set_xlabel("Genotype error-rate multiplier")
ax.set_ylabel("True-doubleton prevalence")
ax.set_ylim(0, 1)
ax.set_title("True-doubleton prevalence")

# %%
metric_specs = [
    ("precision", "Precision (PPV)"),
    ("tpr", "Recall (TPR / sensitivity)"),
    ("fpr", "False positive rate"),
    ("tnr", "Specificity (TNR)"),
    ("npv", "Negative predictive value"),
    ("accuracy", "Classification accuracy"),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
for ax, (metric, title) in zip(axes.flat, metric_specs):
    for L_mismatch_trim in L_mismatch_trim_values:
        for cutoff_proportion in cutoff_proportions:
            select = (
                (classification_df["L_mismatch_trim"] == L_mismatch_trim)
                & (
                    classification_df["cutoff_proportion"]
                    == cutoff_proportion
                )
            )
            plot_df = classification_df[select].sort_values(
                "error_multiplier"
            )
            ax.plot(
                plot_df["error_multiplier"],
                plot_df[metric],
                marker="o",
                label=(
                    f"L={L_mismatch_trim:g}, "
                    f"cutoff={cutoff_proportion:g}"
                ),
            )
    ax.set_xlabel("Genotype error-rate multiplier")
    ax.set_ylabel(title)
    ax.set_ylim(0, 1)
    ax.set_title(title)

axes[0, 0].legend(fontsize=8)

# %% [markdown]
# ### Cumulative counts by doubleton type

# %%
def plot_cumulative_mismatch_profiles(profiles, title):
    """Compare fixed clean/dirty sides and mismatch-free distances by truth."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    max_L = profiles.distances[-1]
    minimum_distance = profiles.distances[0]
    powers = np.arange(
        np.ceil(np.log10(minimum_distance)),
        np.floor(np.log10(max_L)) + 1,
    )
    ticks = 10.0 ** powers
    classes = [
        (profiles.is_true_doubleton, "True doubletons"),
        (~profiles.is_true_doubleton, "False doubletons"),
    ]
    for col, (select, label) in enumerate(classes):
        count = np.count_nonzero(select)
        axes[0, col].set_title(f"{label} (n = {count})")
        for row, (side_counts, side_label) in enumerate([
            (profiles.clean_count, "Clean side"),
            (profiles.dirty_count, "Dirty side"),
        ]):
            ax = axes[row, col]
            if count > 0:
                selected_counts = side_counts[:, select]
                mean = selected_counts.mean(axis=1)
                if count == 1:
                    sem = np.zeros_like(mean)
                else:
                    sem = selected_counts.std(axis=1, ddof=1) / np.sqrt(count)
                lower = np.maximum(0, mean - 1.96 * sem)
                upper = mean + 1.96 * sem
                ax.plot(profiles.distances, mean)
                ax.fill_between(profiles.distances, lower, upper, alpha=0.25)
            ax.set_xscale("log")
            ax.set_xlim(minimum_distance, max_L)
            ax.set_xticks(ticks)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(f"{side_label}: mean cumulative mismatches")
            ax.set_xlabel("Physical distance from focal doubleton (bp)")

        ax = axes[2, col]
        values = profiles.max_first_mismatch_distance[select]
        censored = profiles.max_first_mismatch_censored[select]
        if count > 0:
            observed = np.sort(values[~censored])
            num_below = np.searchsorted(observed, minimum_distance, side="left")
            visible = observed[num_below:]
            ecdf = np.arange(num_below + 1, len(observed) + 1) / count
            ax.step(
                np.r_[minimum_distance, visible],
                np.r_[num_below / count, ecdf],
                where="post",
            )
            ax.annotate(
                f"No mismatch by max L: {censored.mean():.1%}",
                (0.04, 0.96), xycoords="axes fraction", va="top",
            )
        ax.set_xscale("log")
        ax.set_xlim(minimum_distance, max_L)
        ax.set_xticks(ticks)
        ax.set_ylim(0, 1)
        ax.set_ylabel("First-mismatch distance ECDF")
        ax.set_xlabel("Physical distance from focal doubleton (bp)")

    for row in (0, 1):
        upper = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
        for ax in axes[row]:
            ax.set_ylim(0, upper)
    fig.suptitle(title)
    return fig


# %%
profile_distances = np.geomspace(100, config.window_sizes[-1], 60)
cumulative_profiles = {}
for multiplier in [1, 5, 10]:
    profiles = error_validation.cumulative_mismatch_profiles(
        ascertained_estimates[multiplier],
        zero_error_zarr_path,
        distances=profile_distances,
    )
    cumulative_profiles[multiplier] = profiles
    profile_fig = plot_cumulative_mismatch_profiles(
        profiles, f"{multiplier}x simulated genotype error: one-sided mismatch profiles"
    )
    display(profile_fig)
    plt.close(profile_fig)

# %% [markdown]
# ## Real data
#
# The requested TGP density mask does not itself remove multiallelic rows that
# share a position. Combine it with the store's duplicate-position mask so the
# estimator receives one biallelic row per genomic position. The local chr17
# map copy extends the source map's terminal zero-rate interval to the end of
# the Zarr; the map README records this assumption.

# %%
tgp_zarr_path = Path("../data/zarr_vcfs/tgp/chr17/data.zarr")
tgp_recombination = Path(
    "../data/HapMapII_GRCh38/genetic_map_Hg38_chr17_for_tgp_zarr.txt"
)
tgp_variant_masks = [
    (
        "variant_all_subset_chr17p_region_filterNton23_"
        "site_density_threshold_sites_per_kbp_5_window_size_100000_mask"
    ),
    "variant_duplicate_position_mask",
]

tgp_estimate = error_estimation.estimate_error_rate(
    tgp_zarr_path,
    recombination=tgp_recombination,
    config=config,
    variant_mask_name=tgp_variant_masks,
)
tgp_estimate.fit

# %% [markdown]
# The Johnsson et al. (2021) sex-averaged map uses Sscrofa11.1 coordinates.
# The local derived copy extends its terminal zero-rate interval to the final
# coordinate in this Zarr; `data/pig_recombination_map/README.txt` records the
# source and the exact adjustment.

# %%
window_sizes = np.geomspace(1000, 1_000_000, 50)
window_sizes

# %%
import numpy as np

window_sizes = np.geomspace(1000, 1_000_000, 50)
config2 = error_estimation.EstimationConfig(
    window_sizes=window_sizes,
    num_doubletons=10_000,
    random_seed=42,
    exclude_high_mismatch_proportion=0.25,
    L_mismatch_trim=window_sizes[-1],
)

pig_zarr_path = Path("../data/zarr_vcfs/pigs/chr18/data.zarr")
pig_recombination = Path(
    "../data/pig_recombination_map/"
    "Johnsson_2021_sex_averaged_chr18_for_data_zarr.txt"
)
pig_variant_mask = (
    "variant_allSample_subset_chr18_region_filterDensity_"
    "site_density_threshold_sites_per_kbp_10_window_size_1000_mask"
)

pig_estimate = error_estimation.estimate_error_rate(
    pig_zarr_path,
    recombination=pig_recombination,
    config=config2,
    variant_mask_name=pig_variant_mask,
)
pig_estimate.fit

# %%
# The result of epsilon fitting depends on the window sizes: add some plots below showing the density of sites along the genome in the zarr: how wide is the region?

# %%

# %%
# Plot Proportion of excluded doubletons 0, 0.1,0.25,0.5,0.75 on x and three panes for epsilon, mu and sigma^2 on y with L_mismatch_trem = 1e5,

# %%
import pandas as pd

fit_df = pd.read_csv("../data/tmp/dphil-analysis-results.csv")

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
ax.plot(
    fit_df["error_multiplier"],
    fit_df["true_error"],
    marker="o",
    label="True simulation error",
)
ax.plot(
    fit_df["error_multiplier"],
    fit_df["epsilon"],
    marker="o",
    label="Estimated (ascertained doubletons)",
)
ax.plot(
    fit_df["error_multiplier"],
    fit_df["true_doubleton_epsilon"],
    marker="o",
    label="Estimated (true doubletons)",
)
ax.axhline(
    tgp_estimate.fit.epsilon,
    color="tab:purple",
    linestyle="--",
    label="TGP chr17 estimate",
)
ax.axhline(
    pig_estimate.fit.epsilon,
    color="tab:brown",
    linestyle="--",
    label="Pig chr18 estimate",
)
ax.set_xlabel("Genotype error-rate multiplier")
ax.set_ylabel("Errors per haplotype-bp")
ax.set_title("Simulated and real-data error estimates")
ax.legend()

# %%
