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
import pandas as pd


# %%
#zarr_dir = Path(
#    "~/work/tsinfer-anc-eval/data/error_eval/zarr_vcfs/"
#).expanduser()
#zarr_prefix = "OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0"
#ts_path = Path(
#    "~/work/tsinfer-anc-eval/data/error_eval/simulated/"
#    "OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0.trees"
#).expanduser()
zarr_dir = Path(
    "~/work/tsinfer-anc-eval/data/anc_eval/zarr_vcfs/"
).expanduser()
zarr_prefix = "OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0"
ts_path = Path(
    "~/work/tsinfer-anc-eval/data/anc_eval/simulated/"
    "OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0.trees"
).expanduser()

recombination = Path(
    "~/work/tsinfer-paper/data/HapMapII_GRCh38/"
    "genetic_map_Hg38_chr20.txt"
).expanduser()

config = error_estimation.EstimationConfig(
    window_sizes=[1e3, 5e3, 7e3, 1e4, 5e4, 7e4, 1e5, 5e5, 7e5, 1e6],
    num_doubletons=10_000,
    random_seed=42,
    exclude_high_mismatch_proportion=0.25,
    L_mismatch_trim=1e3,
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

fig

# %% [markdown]
# ## Doubleton ascertainment testing

# %%
cutoff_proportions = [0.10, 0.25, 0.50,0.75]
L_mismatch_trim_values = [config.window_sizes[0], config.window_sizes[-1]]
classification_records = []
true_doubleton_masks = {}
# Define true positive as true doubleton and false positive as false doubleton. 
# Add TPR and FPR to dataframe, relabeling everything to use standard TP/FP nomenclature.
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

# %% [markdown]
# ### Mismatch count as classifier

# %%
fig, axes = plt.subplots(
    len(L_mismatch_trim_values),
    len(error_multipliers),
    figsize=(16, 7),
    sharex="row",
    sharey=True,
    constrained_layout=True,
)

# Add legend on rightmost plot of each row to ensure that False doubletons is included in legend
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

#add row for ROC for each error rate, with annotation at bottom right for AUROC
for ax in axes[:, 0]:
    ax.set_ylabel("Empirical cumulative proportion")
axes[0, 0].legend(fontsize=8)

# %%
true_proportion_df = (
    classification_df[
        ["error_multiplier", "true_proportion"]
    ]
    .drop_duplicates()
    .sort_values("error_multiplier")
)

fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
ax.plot(
    true_proportion_df["error_multiplier"],
    true_proportion_df["true_proportion"],
    marker="o",
)
ax.set_xlabel("Genotype error-rate multiplier")
ax.set_ylabel("Proportion of observed doubletons that are true")
ax.set_ylim(0, 1)
ax.set_title("Proportion of doubletons that are real")

# %%
metric_specs = [
    ("retained_true_proportion", "True proportion among retained"),
    ("true_retention_rate", "True-doubleton retention rate"),
    ("false_removal_rate", "False-doubleton removal rate"), 
    ("accuracy", "Classification accuracy"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
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

# %%

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

config2 = error_estimation.EstimationConfig(
    window_sizes=[1e3, 5e3, 7e3, 1e4, 5e4, 7e4, 1e5, 5e5, 7e5],
    num_doubletons=10_000,
    random_seed=42,
    exclude_high_mismatch_proportion=0.5,
    L_mismatch_trim=7e5,
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

# %% [markdown]
#

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
