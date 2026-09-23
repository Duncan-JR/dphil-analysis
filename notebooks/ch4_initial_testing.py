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
import scipy.stats
import zarr


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
    """Compare mean cumulative counts on fixed clean and dirty sides."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
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
    colors = ["tab:blue", "tab:orange"]
    for class_index, (select, label) in enumerate(classes):
        count = np.count_nonzero(select)
        legend_label = f"{label} (n = {count})"
        color = colors[class_index]
        for panel, (side_counts, side_label) in enumerate([
            (profiles.clean_count, "Clean side"),
            (profiles.dirty_count, "Dirty side"),
        ]):
            ax = axes[panel]
            if count > 0:
                selected_counts = side_counts[:, select]
                mean = selected_counts.mean(axis=1)
                if count == 1:
                    sem = np.zeros_like(mean)
                else:
                    sem = selected_counts.std(axis=1, ddof=1) / np.sqrt(count)
                lower = np.maximum(0, mean - 1.96 * sem)
                upper = mean + 1.96 * sem
                ax.plot(profiles.distances, mean, color=color, label=legend_label)
                ax.fill_between(
                    profiles.distances, lower, upper, color=color, alpha=0.2
                )
            ax.set_title(side_label)
            ax.set_ylabel("Mean cumulative mismatches")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(minimum_distance, max_L)
        ax.set_xticks(ticks)
        ax.set_xlabel("Physical distance from focal doubleton (bp)")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
    fig.suptitle(f"{title} (shading: 95% CI of mean)")
    return fig


# %%
def plot_doubleton_mismatch_summaries(profiles, title):
    """Compare scalar mismatch summaries with densities and ECDFs by class."""
    summaries = [
        (
            "Longer mismatch-free side",
            profiles.max_first_mismatch_distance,
            "First-mismatch distance (bp)",
            profiles.max_first_mismatch_censored,
        ),
        ("Max clean count", profiles.max_clean_count, "Mismatches at max L", None),
        ("Max dirty count", profiles.max_dirty_count, "Mismatches at max L", None),
    ]
    classes = [
        (profiles.is_true_doubleton, "True doubletons", "tab:blue"),
        (~profiles.is_true_doubleton, "False doubletons", "tab:orange"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)
    for col, (heading, all_values, xlabel, all_censored) in enumerate(summaries):
        histogram_ax = axes[0, col]
        ecdf_ax = axes[1, col]
        maximum = max(float(np.max(all_values)), 1.0)
        if col == 0:
            minimum = max(float(np.min(all_values)), 1.0)
            maximum = float(profiles.distances[-1])
            bins = np.geomspace(minimum, maximum, 41)
            grid = np.geomspace(minimum, maximum, 250)
        else:
            minimum = 0.0
            bins = np.linspace(minimum, maximum, 41)
            grid = np.linspace(minimum, maximum, 250)

        for class_index, (select, label, color) in enumerate(classes):
            values = all_values[select]
            count = len(values)
            if count == 0:
                continue
            legend_label = f"{label} (n = {count})"
            if all_censored is None:
                censored = np.zeros(count, dtype=bool)
            else:
                censored = all_censored[select]
            observed = values[~censored]
            if len(observed) > 0:
                histogram_ax.hist(
                    observed, bins=bins, density=True, alpha=0.25,
                    color=color, label=legend_label,
                )
                if np.ptp(observed) > 0:
                    if col == 0:
                        log_values = np.log(observed)
                        log_density = scipy.stats.gaussian_kde(log_values)
                        density = log_density(np.log(grid)) / grid
                    else:
                        density_estimate = scipy.stats.gaussian_kde(observed)
                        density = density_estimate(grid)
                    histogram_ax.plot(grid, density, color=color)

            mean = values.mean()
            histogram_ax.axvline(mean, color=color, linestyle="--")
            ecdf_ax.axvline(mean, color=color, linestyle="--")
            ordered = np.sort(observed)
            num_below = np.searchsorted(ordered, minimum, side="left")
            visible = ordered[num_below:]
            cumulative = np.arange(num_below + 1, len(ordered) + 1) / count
            ecdf_ax.step(
                np.r_[minimum, visible],
                np.r_[num_below / count, cumulative],
                where="post", color=color, label=legend_label,
            )
            if np.any(censored):
                ecdf_ax.text(
                    0.04, 0.96 - 0.08 * class_index,
                    f"{label}: {censored.mean():.1%} censored at max L",
                    transform=ecdf_ax.transAxes, va="top", color=color,
                )

        histogram_ax.set_title(heading)
        histogram_ax.set_ylabel("Density")
        ecdf_ax.set_ylabel("Empirical cumulative proportion")
        ecdf_ax.set_ylim(0, 1)
        for ax in (histogram_ax, ecdf_ax):
            ax.set_xlim(minimum, maximum)
            ax.set_xlabel(xlabel)
            ax.legend(fontsize=8)
            if col == 0:
                ax.set_xscale("log")
    first_mismatch_log10 = np.log10(profiles.max_first_mismatch_distance)
    scatter_specs = [
        (
            first_mismatch_log10,
            profiles.max_clean_count,
            "log10 first-mismatch distance (bp)",
            "Max clean count",
        ),
        (
            first_mismatch_log10,
            profiles.max_dirty_count,
            "log10 first-mismatch distance (bp)",
            "Max dirty count",
        ),
        (
            profiles.max_clean_count,
            profiles.max_dirty_count,
            "Max clean count",
            "Max dirty count",
        ),
    ]
    for col, (x_values, y_values, x_label, y_label) in enumerate(scatter_specs):
        ax = axes[2, col]
        for class_index, (select, label, color) in enumerate(classes):
            x = x_values[select]
            y = y_values[select]
            count = len(x)
            legend_label = f"{label} (n = {count})"
            ax.scatter(x, y, color=color, alpha=0.15, s=10, label=legend_label)
            if count > 1 and np.ptp(x) > 0 and np.ptp(y) > 0:
                regression = scipy.stats.linregress(x, y)
                regression_x = np.linspace(x.min(), x.max(), 100)
                regression_y = regression.intercept + regression.slope * regression_x
                ax.plot(regression_x, regression_y, color=color, linestyle="--")
                ax.text(
                    0.03, 0.97 - 0.09 * class_index,
                    f"{label}: Pearson r = {regression.rvalue:.2f}",
                    transform=ax.transAxes, va="top", color=color,
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(fontsize=8)

    mean_note = "dashed lines: class means above, class regressions below"
    if np.any(profiles.max_first_mismatch_censored):
        mean_note += "; censored distances capped at max L"
    fig.suptitle(f"{title} ({mean_note})")
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
# ### Per-doubleton mismatch summaries

# %%
for multiplier in [1, 5, 10]:
    summary_fig = plot_doubleton_mismatch_summaries(
        cumulative_profiles[multiplier],
        f"{multiplier}x simulated genotype error: mismatch summaries",
    )
    display(summary_fig)
    plt.close(summary_fig)

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
pig_window_sizes = np.unique(
    np.r_[np.geomspace(1_000, 1_000_000, 50), 100_000.0]
)
config2 = error_estimation.EstimationConfig(
    window_sizes=pig_window_sizes,
    num_doubletons=10_000,
    random_seed=42,
    exclude_high_mismatch_proportion=0.25,
    L_mismatch_trim=pig_window_sizes[-1],
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
# ### Pig chr18 site density and retained quality flags
#
# The current density-threshold mask is applied on its own. Show how much of
# chr18 it retains, and which other Zarr quality flags remain among those sites.

# %%
pig_group = zarr.open_group(pig_zarr_path, mode="r")
pig_all_positions = np.asarray(pig_group["variant_position"][:], dtype=float)
pig_excluded = np.asarray(pig_group[pig_variant_mask][:], dtype=bool)
pig_positions = pig_all_positions[~pig_excluded]
pig_bin_width = 100_000
pig_bins = np.arange(
    0, np.ceil(pig_all_positions[-1] / pig_bin_width) * pig_bin_width + pig_bin_width,
    pig_bin_width,
)
pig_bin_midpoints_mb = (pig_bins[:-1] + pig_bins[1:]) / 2e6
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
    pig_quality_rows.append({
        "flag": label,
        "retained_sites_flagged": np.count_nonzero(flagged),
        "fraction_of_retained_sites": flagged.mean(),
    })
pig_quality_df = pd.DataFrame(pig_quality_rows)
display(pig_quality_df)

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
axes[0].plot(pig_bin_midpoints_mb, all_sites_per_bin / 100, label="All Zarr sites")
axes[0].plot(
    pig_bin_midpoints_mb, retained_sites_per_bin / 100,
    label=(
        f"Retained by current mask ({len(pig_positions):,} sites; "
        f"{np.count_nonzero(pig_excluded):,} removed)"
    ),
)
axes[0].set_ylabel("Sites per kb in 100-kb bins")
axes[0].set_title(
    f"Pig chr18: retained span {pig_positions[0] / 1e6:.2f}–"
    f"{pig_positions[-1] / 1e6:.2f} Mb"
)
axes[0].legend()
for label, rates in pig_flag_rates.items():
    axes[1].plot(pig_bin_midpoints_mb, rates, label=label)
axes[1].set_xlabel("Pig chr18 position (Mb)")
axes[1].set_ylabel("Fraction of retained sites flagged")
axes[1].set_ylim(0, 1)
axes[1].legend(ncol=2, fontsize=8)
fig

# %%
pig_mask_comparisons = {
    "Current density-threshold mask": pig_estimate,
}
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
    pig_mask_comparisons[label] = error_estimation.estimate_error_rate(
        pig_zarr_path,
        recombination=pig_recombination,
        config=config2,
        variant_mask_name=masks,
    )
pig_mask_rows = []
for label, estimate in pig_mask_comparisons.items():
    pig_mask_rows.append({
        "site_selection": label,
        "num_sites": estimate.diversity.num_sites,
        "eligible_doubletons": estimate.doubletons.num_eligible,
        "epsilon": estimate.fit.epsilon,
        "mu": estimate.fit.mu,
        "sigma_sq": estimate.fit.sigma_sq,
    })
pig_mask_comparison_df = pd.DataFrame(pig_mask_rows)
display(pig_mask_comparison_df)

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
        config2,
        exclude_high_mismatch_proportion=proportion,
        L_mismatch_trim=pig_trim_L,
    )
    fit = error_estimation.fit_error_model(
        pig_estimate.mismatches, pi=pig_estimate.fit.pi, config=trim_config
    )
    pig_trim_rows.append({
        "excluded_proportion": proportion,
        "num_doubletons": fit.num_doubletons,
        "epsilon": fit.epsilon,
        "mu": fit.mu,
        "sigma_sq": fit.sigma_sq,
        "objective": fit.objective,
    })
pig_trim_df = pd.DataFrame(pig_trim_rows)
display(pig_trim_df)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for ax, (column, ylabel) in zip(axes, [
    ("epsilon", "Errors per haplotype-bp"),
    ("mu", r"Fitted $\mu$"),
    ("sigma_sq", r"Fitted $\sigma^2$"),
]):
    ax.plot(pig_trim_df["excluded_proportion"], pig_trim_df[column], marker="o")
    ax.set_xlabel("Proportion of doubletons excluded")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_xticks(pig_trim_proportions)
fig.suptitle("Pig chr18 fit sensitivity (trim at 100 kb)")


# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes[0].plot(
    pig_window_sizes, pig_estimate.fit.observed_means,
    marker=".", label="Observed",
)
axes[0].plot(
    pig_window_sizes, pig_estimate.fit.fitted_means,
    label="Fitted",
)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("One-sided window length L (bp)")
axes[0].set_ylabel("Mean full-window mismatch count")
axes[0].legend()
relative_residual = (
    pig_estimate.fit.observed_means - pig_estimate.fit.fitted_means
) / pig_estimate.fit.observed_means
axes[1].plot(pig_window_sizes, relative_residual, marker=".")
axes[1].axhline(0, color="0.5", linestyle="--")
axes[1].set_xscale("log")
axes[1].set_xlabel("One-sided window length L (bp)")
axes[1].set_ylabel("(Observed − fitted) / observed")
fig.suptitle("Pig chr18: aggregate fit across window sizes")
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
pig_local_sites_per_kb = (pig_local_right - pig_local_left) / (2 * pig_trim_L / 1_000)
pig_raw_focal_indices = np.flatnonzero(~pig_excluded)[pig_estimate.doubletons.site_indices]
pig_focal_quality_rows = []
for label, mask_name in pig_quality_masks.items():
    flagged = np.asarray(pig_group[mask_name][:], dtype=bool)[pig_raw_focal_indices]
    pig_focal_quality_rows.append({
        "flag": label,
        "flagged_doubletons": np.count_nonzero(flagged),
        "median_mismatches_flagged": np.median(pig_trim_counts[flagged]),
        "median_mismatches_unflagged": np.median(pig_trim_counts[~flagged]),
    })
pig_focal_quality_df = pd.DataFrame(pig_focal_quality_rows)
display(pig_focal_quality_df)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
ordered_counts = np.sort(pig_trim_counts)
axes[0].plot(ordered_counts, np.arange(1, len(ordered_counts) + 1) / len(ordered_counts))
axes[0].set_xlabel("Mismatch count within ±100 kb")
axes[0].set_ylabel("Empirical cumulative proportion")
axes[0].set_title("Large variation among sampled doubletons")
density_plot = axes[1].hexbin(
    pig_local_sites_per_kb, pig_trim_counts, gridsize=45, bins="log", mincnt=1
)
fig.colorbar(density_plot, ax=axes[1], label="Number of doubletons")
axes[1].set_xlabel("Local retained sites per kb (±100 kb)")
axes[1].set_ylabel("Mismatch count within ±100 kb")
axes[1].set_title("Mismatch count versus local site density")
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

# %%
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
