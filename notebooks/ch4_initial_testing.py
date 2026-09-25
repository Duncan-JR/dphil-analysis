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
import importlib
import error_estimation
import error_validation
importlib.reload(error_validation)
importlib.reload(error_estimation)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import zarr


# %%
zarr_dir = Path(
    "~/work/tsinfer-anc-eval/data/error_eval/zarr_vcfs/"
).expanduser()
zarr_prefix = "OutOfAfrica_4J17-chr17-L0-R22.7e6-n3202-s1-rep0"
ts_path = Path(
    "~/work/tsinfer-anc-eval/data/error_eval/simulated/"
    "OutOfAfrica_4J17-chr17-L0-R22.7e6-n3202-s1-rep0.trees"
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
def plot_error_estimates(frame, estimate_label, *, baseline_frame=None, title=None):
    """Plot the four fitted quantities against simulation truth and references."""
    specs = [
        ("epsilon", "true_error", "true_doubleton_epsilon", "Genotype error rate", "Errors per haplotype-bp"),
        ("diversity", "true_diversity", "true_doubleton_pi", "Diversity", "Diversity per bp"),
        ("mu", None, "true_doubleton_mu", "Gamma mean", r"Estimated $\mu$"),
        ("sigma_sq", None, "true_doubleton_sigma_sq", "Gamma variance", r"Estimated $\sigma^2$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (column, truth_column, fixed_column, heading, ylabel) in zip(axes.flat, specs):
        if truth_column is not None:
            ax.plot(frame["error_multiplier"], frame[truth_column], marker="o", label="True")
        ax.plot(frame["error_multiplier"], frame[column], marker="o", label=estimate_label)
        if baseline_frame is not None:
            ax.plot(
                baseline_frame["error_multiplier"], baseline_frame[column],
                marker="o", label="Original 25% mismatch trim",
            )
        ax.plot(
            frame["error_multiplier"], frame[fixed_column],
            marker="o", label="Estimated (fixed true doubletons)",
        )
        ax.set_xlabel("Genotype error-rate multiplier")
        ax.set_ylabel(ylabel)
        ax.set_title(heading)
        ax.legend(fontsize=8)
    if title is not None:
        fig.suptitle(title)
    return fig


plot_error_estimates(fit_df, "Estimated (ascertained doubletons)")


# %%
## Doubleton ascertainment testing

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
# ## Fitting models to doubleton data

# %%
import importlib
importlib.reload(error_estimation)
importlib.reload(error_validation)

for multiplier in [0, 2]:
    cumulative_profiles[multiplier] = error_validation.cumulative_mismatch_profiles(
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
            model = error_validation.DoubletonClassifier(tier=tier, threshold=None)
        else:
            model = error_validation.fit_doubleton_classifier(tier, training_profiles)
        held_out_models[(held_out_multiplier, tier)] = model
        scores, retained = error_validation.classify_doubletons(
            model, test_profiles, estimate.mismatches
        )
        metrics = error_validation.classifier_metrics(
            scores, test_profiles.is_true_doubleton, retained
        )
        fitted = error_validation.fit_error_rate_with_selected_doubletons(
            estimate, retained
        )
        metrics.update({
            "tier": tier,
            "test_multiplier": held_out_multiplier,
            "fitted_epsilon": fitted.epsilon,
            "epsilon_relative_error": abs(fitted.epsilon - true_epsilon) / true_epsilon,
            "fitted_mu": fitted.mu,
            "fitted_sigma_sq": fitted.sigma_sq,
        })
        classifier_rows.append(metrics)
        classifier_curves[(held_out_multiplier, tier)] = (
            error_validation.classifier_curves(
                scores, test_profiles.is_true_doubleton
            )
        )

classifier_cv_df = pd.DataFrame(classifier_rows)
classifier_cv_summary = classifier_cv_df.groupby("tier", sort=False)[
    ["auroc", "average_precision", "tpr", "fpr", "precision", "fraction_retained", "epsilon_relative_error"]
].mean()
display(classifier_cv_df)
display(classifier_cv_summary)

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
for col, multiplier in enumerate(training_multipliers):
    for tier in classifier_tiers:
        curves = classifier_curves[(multiplier, tier)]
        label = classifier_labels[tier]
        axes[0, col].plot(curves["fpr"], curves["tpr"], label=label)
        axes[1, col].plot(curves["tpr"], curves["precision"], label=label)
    axes[0, col].plot([0, 1], [0, 1], color="0.7", linestyle="--")
    prevalence = cumulative_profiles[multiplier].is_true_doubleton.mean()
    axes[1, col].axhline(prevalence, color="0.7", linestyle="--")
    axes[0, col].set_title(f"Held-out {multiplier}x: ROC")
    axes[1, col].set_title(f"Held-out {multiplier}x: precision-recall")
    axes[0, col].set_xlabel("False positive rate")
    axes[0, col].set_ylabel("True positive rate")
    axes[1, col].set_xlabel("Recall (true positive rate)")
    axes[1, col].set_ylabel("Precision")
    for ax in axes[:, col]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Classifiers trained on the other two error multipliers")
fig

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for tier in classifier_tiers:
    tier_rows = classifier_cv_df[classifier_cv_df["tier"] == tier]
    axes[0, 0].plot(
        tier_rows["test_multiplier"], tier_rows["fpr"],
        marker="o", label=classifier_labels[tier],
    )
    axes[0, 1].plot(
        tier_rows["test_multiplier"], tier_rows["tpr"],
        marker="o", label=classifier_labels[tier],
    )
    axes[1, 0].plot(
        tier_rows["test_multiplier"], tier_rows["fraction_retained"],
        marker="o", label=classifier_labels[tier],
    )
    axes[1, 1].plot(
        tier_rows["test_multiplier"], tier_rows["epsilon_relative_error"],
        marker="o", label=classifier_labels[tier],
    )
axes[0, 0].set_ylabel("False positive rate")
axes[0, 1].set_ylabel("True positive rate")
axes[1, 0].set_ylabel("Fraction of doubletons retained")
axes[1, 1].set_ylabel("Absolute relative error in fitted epsilon")
for ax in axes.flat:
    ax.set_xlabel("Held-out error multiplier")
    ax.set_xticks(training_multipliers)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Held-out classification and downstream error-rate fit")
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
        model = error_validation.DoubletonClassifier(tier=tier, threshold=None)
    else:
        model = error_validation.fit_doubleton_classifier(tier, pooled_training_profiles)
    pooled_models[tier] = model
    scores, retained = error_validation.classify_doubletons(
        model, interpolation_profiles, interpolation_estimate.mismatches
    )
    metrics = error_validation.classifier_metrics(
        scores, interpolation_profiles.is_true_doubleton, retained
    )
    fitted = error_validation.fit_error_rate_with_selected_doubletons(
        interpolation_estimate, retained
    )
    metrics.update({
        "tier": tier,
        "fitted_epsilon": fitted.epsilon,
        "epsilon_relative_error": (
            abs(fitted.epsilon - interpolation_true_error) / interpolation_true_error
        ),
    })
    interpolation_rows.append(metrics)
    interpolation_curves[tier] = error_validation.classifier_curves(
        scores, interpolation_profiles.is_true_doubleton
    )
interpolation_df = pd.DataFrame(interpolation_rows)
display(interpolation_df)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for tier in classifier_tiers:
    curves = interpolation_curves[tier]
    axes[0].plot(curves["fpr"], curves["tpr"], label=classifier_labels[tier])
    axes[1].plot(curves["tpr"], curves["precision"], label=classifier_labels[tier])
axes[0].plot([0, 1], [0, 1], color="0.7", linestyle="--")
axes[1].axhline(interpolation_profiles.is_true_doubleton.mean(), color="0.7", linestyle="--")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[1].set_xlabel("Recall (true positive rate)")
axes[1].set_ylabel("Precision")
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
axes[0].legend(fontsize=8)
fig.suptitle("2x interpolation test; training used 1x, 5x and 10x")
fig

# %%
interpolation_features = error_validation.doubleton_features(interpolation_profiles)
horizontal = np.linspace(interpolation_features[:, 0].min(), interpolation_features[:, 0].max(), 120)
vertical = np.linspace(interpolation_features[:, 1].min(), interpolation_features[:, 1].max(), 120)
grid_x, grid_y = np.meshgrid(horizontal, vertical)
grid_features = np.column_stack((grid_x.ravel(), grid_y.ravel()))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, tier in zip(axes, ["tier_2", "tier_3"]):
    model = pooled_models[tier]
    grid_scores = error_validation.doubleton_scores(model, grid_features)
    grid_probability = scipy.special.expit(grid_scores).reshape(grid_x.shape)
    filled = ax.contourf(
        grid_x, grid_y, grid_probability,
        levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.6,
    )
    ax.contour(
        grid_x, grid_y, grid_scores.reshape(grid_x.shape),
        levels=[model.threshold], colors="black", linewidths=2,
    )
    for select, label, color in [
        (interpolation_profiles.is_true_doubleton, "True doubletons", "tab:blue"),
        (~interpolation_profiles.is_true_doubleton, "False doubletons", "tab:orange"),
    ]:
        ax.scatter(
            interpolation_features[select, 0], interpolation_features[select, 1],
            s=3, alpha=0.08, color=color, label=label,
        )
    ax.set_title(classifier_labels[tier])
    ax.set_xlabel("log10 first-mismatch distance (bp)")
    ax.set_ylabel("Max clean count at max L")
axes[0].legend(fontsize=8)
fig.colorbar(filled, ax=axes, label="Fitted probability of a true doubleton")
fig.suptitle("2x doubletons and pooled logistic decision boundaries")
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
    scores, retained = error_validation.classify_doubletons(
        selected_model, profiles, estimate.mismatches
    )
    fitted = error_validation.fit_error_rate_with_selected_doubletons(
        estimate, retained
    )
    selected_fit_rows.append({
        "error_multiplier": multiplier,
        "epsilon": fitted.epsilon,
        "mu": fitted.mu,
        "sigma_sq": fitted.sigma_sq,
    })
    classification_row = {
        "error_multiplier": multiplier,
        "retained_doubletons": int(retained.sum()),
        "total_doubletons": len(retained),
        "fraction_retained": retained.mean(),
    }
    truth = profiles.is_true_doubleton
    if truth.any() and (~truth).any():
        metrics = error_validation.classifier_metrics(scores, truth, retained)
        classification_row.update({
            "tpr": metrics["tpr"],
            "fpr": metrics["fpr"],
            "precision": metrics["precision"],
        })
    selected_classification_rows.append(classification_row)

selected_fit_df = fit_df.drop(columns=["epsilon", "mu", "sigma_sq"]).merge(
    pd.DataFrame(selected_fit_rows), on="error_multiplier", validate="one_to_one"
)
selected_classification_df = pd.DataFrame(selected_classification_rows)
display(selected_classification_df)
display(selected_fit_df[[
    "error_multiplier", "true_error", "epsilon", "true_doubleton_epsilon",
    "mu", "sigma_sq",
]])

# %%
plot_error_estimates(
    selected_fit_df,
    "Estimated (Tier 3 retained doubletons)",
    baseline_frame=fit_df,
    title="Error-model fits after Tier 3 doubleton selection",
)

# %%
diagnostic_profiles = cumulative_profiles[10]
diagnostic_features = error_validation.doubleton_features(diagnostic_profiles)
diagnostic_scores, diagnostic_retained = error_validation.classify_doubletons(
    selected_model, diagnostic_profiles, ascertained_estimates[10].mismatches
)
diagnostic_truth = diagnostic_profiles.is_true_doubleton
outcomes = [
    (diagnostic_retained & diagnostic_truth, "TP: retained true", "tab:blue"),
    (diagnostic_retained & ~diagnostic_truth, "FP: retained false", "tab:orange"),
    (~diagnostic_retained & diagnostic_truth, "FN: excluded true", "tab:red"),
    (~diagnostic_retained & ~diagnostic_truth, "TN: excluded false", "tab:green"),
]
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
diagnostic_x = np.linspace(diagnostic_features[:, 0].min(), diagnostic_features[:, 0].max(), 120)
diagnostic_y = np.linspace(diagnostic_features[:, 1].min(), diagnostic_features[:, 1].max(), 120)
diagnostic_grid_x, diagnostic_grid_y = np.meshgrid(diagnostic_x, diagnostic_y)
diagnostic_grid_features = np.column_stack((
    diagnostic_grid_x.ravel(), diagnostic_grid_y.ravel()
))
diagnostic_grid_scores = error_validation.doubleton_scores(
    selected_model, diagnostic_grid_features
).reshape(diagnostic_grid_x.shape)
ax.contour(
    diagnostic_grid_x, diagnostic_grid_y, diagnostic_grid_scores,
    levels=[selected_model.threshold], colors="black", linestyles="--",
)
for select, label, color in outcomes:
    ax.scatter(
        diagnostic_features[select, 0], diagnostic_features[select, 1],
        s=4, alpha=0.16, color=color, label=f"{label} (n = {select.sum()})",
    )
ax.set_xlabel("log10 first-mismatch distance (bp)")
ax.set_ylabel("Max clean count at max L")
ax.set_title("10x diagnostic with pooled Tier 3 classifier")
ax.legend(markerscale=3, fontsize=8)
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
tgp_estimate.fit.epsilon

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
    profiles = error_validation.cumulative_mismatch_profiles(
        estimate, distances=np.array([max_L])
    )
    real_profiles[dataset] = profiles
    assert profiles.is_true_doubleton is None
    assert profiles.max_clean_count.shape == estimate.doubletons.positions.shape
    assert profiles.max_first_mismatch_distance.shape == estimate.doubletons.positions.shape
    assert np.array_equal(
        profiles.max_left_count + profiles.max_right_count,
        estimate.mismatches.counts[-1],
    )

    real_selections[dataset] = {}
    for tier in classifier_tiers:
        model = pooled_models[tier]
        scores, retained = error_validation.classify_doubletons(
            model, profiles, estimate.mismatches,
            baseline_excluded=estimate.config.exclude_high_mismatch_proportion,
        )
        assert len(retained) == len(estimate.doubletons.positions)
        real_selections[dataset][tier] = retained
        if tier == "tier_0":
            fitted = estimate.fit
            assert fitted.num_doubletons == retained.sum()
        else:
            fitted = error_validation.fit_error_rate_with_selected_doubletons(
                estimate, retained
            )
        real_rows.append({
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
        })

real_classifier_df = pd.DataFrame(real_rows)
display(real_classifier_df)

# %% [markdown]
# ### Exclusions by predictor
#
# Tier 1 uses mismatch-free distance; Tier 1b uses clean-side count. The
# overlap partitions describe their individual rules, while Tiers 2 and 3
# combine the predictors and have no unique per-predictor attribution.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
for col, dataset in enumerate(real_datasets):
    rows = real_classifier_df[real_classifier_df["dataset"] == dataset]
    excluded_counts = [
        rows.loc[rows["tier"] == tier, "excluded_doubletons"].iloc[0]
        for tier in classifier_tiers
    ]
    labels = [classifier_labels[tier] for tier in classifier_tiers]
    axes[0, col].bar(labels, excluded_counts)
    axes[0, col].tick_params(axis="x", labelrotation=35)
    axes[0, col].set_ylabel("Excluded doubletons")
    axes[0, col].set_title(
        f"{dataset}: exclusions by classifier (n = {rows['sampled_doubletons'].iloc[0]:,})"
    )
    for index, count in enumerate(excluded_counts):
        axes[0, col].text(index, count, f"{count:,}", ha="center", va="bottom")

    distance_excluded = ~real_selections[dataset]["tier_1"]
    clean_excluded = ~real_selections[dataset]["tier_1b"]
    overlap = [
        ("Distance only", int((distance_excluded & ~clean_excluded).sum())),
        ("Clean count only", int((~distance_excluded & clean_excluded).sum())),
        ("Both", int((distance_excluded & clean_excluded).sum())),
        ("Neither", int((~distance_excluded & ~clean_excluded).sum())),
    ]
    left = 0
    for label, count in overlap:
        axes[1, col].barh("Sampled doubletons", count, left=left, label=f"{label}: {count:,}")
        left += count
    axes[1, col].set_xlim(0, left)
    axes[1, col].set_xlabel("Number of doubletons")
    axes[1, col].set_title(f"{dataset}: overlap of one-predictor exclusions")
    axes[1, col].legend(fontsize=8)
fig.suptitle("Real-data doubleton exclusions from simulation-trained classifiers")
fig

# %%
feature_sets = {
    "Training simulations (1x, 5x, 10x)": np.concatenate([
        error_validation.doubleton_features(profiles)
        for profiles in pooled_training_profiles
    ]),
}
for dataset, profiles in real_profiles.items():
    feature_sets[dataset] = error_validation.doubleton_features(profiles)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for label, features in feature_sets.items():
    distance = np.sort(features[:, 0])
    clean = np.sort(np.log1p(features[:, 1]))
    ecdf = np.arange(1, len(features) + 1) / len(features)
    axes[0].plot(distance, ecdf, label=label)
    axes[1].plot(clean, ecdf, label=label)
axes[0].set_xlabel("log10 longer first-mismatch distance (bp)")
axes[1].set_xlabel("log1p max clean-side mismatch count")
for ax in axes:
    ax.set_ylabel("Fraction of sampled doubletons")
    ax.legend(fontsize=8)
fig.suptitle("Classifier predictor distributions in training and real data")


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
def plot_real_classifier_estimates(results, dataset):
    """Compare the four numbered doubleton-selection tiers for one dataset."""
    tiers = ["tier_0", "tier_1", "tier_2", "tier_3"]
    selected = results[results["dataset"] == dataset].set_index("tier").loc[tiers]
    labels = [f"Tier {tier[-1]}\n(n={n:,})" for tier, n in zip(
        tiers, selected["retained_doubletons"]
    )]
    quantities = [
        ("epsilon", "Genotype error rate", "Errors per haplotype-bp"),
        ("diversity", "Diversity", "Diversity per bp"),
        ("mu", "Gamma mean", r"Estimated $\mu$"),
        ("sigma_sq", "Gamma variance", r"Estimated $\sigma^2$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (column, heading, ylabel) in zip(axes.flat, quantities):
        ax.plot(labels, selected[column], marker="o")
        ax.set_title(heading)
        ax.set_ylabel(ylabel)
        if column == "sigma_sq":
            ax.set_yscale("log")
    fig.suptitle(f"{dataset}: error-model estimates by doubleton-selection tier")


for dataset in real_datasets:
    display(plot_real_classifier_estimates(real_classifier_df, dataset))

# %% [markdown]
# ### Real Tier 3 epsilon against simulation estimates
#
# The horizontal lines show the fitted values on the real datasets. The
# simulated multiplier is a reference axis, not an inferred real-data error
# multiplier.

# %%
fig, (ax, zoom_ax) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
simulation_curves = [
    (fit_df, "true_error", "True simulation error"),
    (fit_df, "epsilon", "Estimated (ascertained doubletons)"),
    (fit_df, "true_doubleton_epsilon", "Estimated (fixed true doubletons)"),
    (selected_fit_df, "epsilon", "Estimated (Tier 2 doubletons)"),
]
for frame, column, label in simulation_curves:
    ax.plot(frame["error_multiplier"], frame[column], marker="o", label=label)
    zoom_ax.plot(frame["error_multiplier"], frame[column], marker="o", markersize=3)
real_epsilon_values = {}
for dataset, color in [("TGP chr17", "tab:purple"), ("Pig chr18", "tab:brown")]:
    epsilon = real_classifier_df.loc[
        (real_classifier_df["dataset"] == dataset)
        & (real_classifier_df["tier"] == "tier_2"),
        "epsilon",
    ].iloc[0]
    real_epsilon_values[dataset] = epsilon
    ax.axhline(epsilon, color=color, linestyle="--", label=f"{dataset} Tier 2")
    zoom_ax.axhline(epsilon, color=color, linestyle="--")
zoom_ax.set_xlim(-0.1, 2.5)
zoom_ax.set_ylim(0, 2.5 * max(real_epsilon_values.values()))
ax.set_title("Full range")
zoom_ax.set_title("Low-error range")
for panel in (ax, zoom_ax):
    panel.set_xlabel("Genotype error-rate multiplier in simulation")
    panel.set_ylabel("Errors per haplotype-bp")
ax.legend(fontsize=8)
fig.suptitle("Simulated and real-data error estimates")
fig

# %%

# %%

# %%
