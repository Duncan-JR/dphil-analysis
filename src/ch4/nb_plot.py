"""Chapter 4 notebook figures."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from . import nb_utils


def plot_error_estimates(frame, estimate_label, *, baseline_frame=None, title=None):
    """Plot the four fitted quantities against simulation truth and references."""
    specs = [
        (
            "epsilon",
            "true_error",
            "true_doubleton_epsilon",
            "Genotype error rate",
            "Errors per haplotype-bp",
        ),
        (
            "diversity",
            "true_diversity",
            "true_doubleton_pi",
            "Diversity",
            "Diversity per bp",
        ),
        ("mu", None, "true_doubleton_mu", "Gamma mean", "Estimated $\\mu$"),
        (
            "sigma_sq",
            None,
            "true_doubleton_sigma_sq",
            "Gamma variance",
            "Estimated $\\sigma^2$",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (column, truth_column, fixed_column, heading, ylabel) in zip(
        axes.flat, specs
    ):
        if truth_column is not None:
            ax.plot(
                frame["error_multiplier"], frame[truth_column], marker="o", label="True"
            )
        ax.plot(
            frame["error_multiplier"], frame[column], marker="o", label=estimate_label
        )
        if baseline_frame is not None:
            ax.plot(
                baseline_frame["error_multiplier"],
                baseline_frame[column],
                marker="o",
                label="Original 25% mismatch trim",
            )
        ax.plot(
            frame["error_multiplier"],
            frame[fixed_column],
            marker="o",
            label="Estimated (fixed true doubletons)",
        )
        ax.set_xlabel("Genotype error-rate multiplier")
        ax.set_ylabel(ylabel)
        ax.set_title(heading)
        ax.legend(fontsize=8)
    if title is not None:
        fig.suptitle(title)
    return fig


def plot_cumulative_mismatch_profiles(profiles, title):
    """Compare mean cumulative counts on fixed clean and dirty sides."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    max_L = profiles.distances[-1]
    minimum_distance = profiles.distances[0]
    powers = np.arange(
        np.ceil(np.log10(minimum_distance)), np.floor(np.log10(max_L)) + 1
    )
    ticks = 10.0**powers
    classes = [
        (profiles.is_true_doubleton, "True doubletons"),
        (~profiles.is_true_doubleton, "False doubletons"),
    ]
    colors = ["tab:blue", "tab:orange"]
    for class_index, (select, label) in enumerate(classes):
        count = np.count_nonzero(select)
        legend_label = f"{label} (n = {count})"
        color = colors[class_index]
        for panel, (side_counts, side_label) in enumerate(
            [(profiles.clean_count, "Clean side"), (profiles.dirty_count, "Dirty side")]
        ):
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
                    observed,
                    bins=bins,
                    density=True,
                    alpha=0.25,
                    color=color,
                    label=legend_label,
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
                where="post",
                color=color,
                label=legend_label,
            )
            if np.any(censored):
                ecdf_ax.text(
                    0.04,
                    0.96 - 0.08 * class_index,
                    f"{label}: {censored.mean():.1%} censored at max L",
                    transform=ecdf_ax.transAxes,
                    va="top",
                    color=color,
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
            if count > 1 and np.ptp(x) > 0 and (np.ptp(y) > 0):
                regression = scipy.stats.linregress(x, y)
                regression_x = np.linspace(x.min(), x.max(), 100)
                regression_y = regression.intercept + regression.slope * regression_x
                ax.plot(regression_x, regression_y, color=color, linestyle="--")
                ax.text(
                    0.03,
                    0.97 - 0.09 * class_index,
                    f"{label}: Pearson r = {regression.rvalue:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    color=color,
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(fontsize=8)
    mean_note = "dashed lines: class means above, class regressions below"
    if np.any(profiles.max_first_mismatch_censored):
        mean_note += "; censored distances capped at max L"
    fig.suptitle(f"{title} ({mean_note})")
    return fig


def plot_real_classifier_estimates(results, dataset):
    """Compare the four numbered doubleton-selection tiers for one dataset."""
    tiers = ["tier_0", "tier_1", "tier_2", "tier_3"]
    selected = results[results["dataset"] == dataset].set_index("tier").loc[tiers]
    labels = [
        f"Tier {tier[-1]}\n(n={n:,})"
        for tier, n in zip(tiers, selected["retained_doubletons"])
    ]
    quantities = [
        ("epsilon", "Genotype error rate", "Errors per haplotype-bp"),
        ("diversity", "Diversity", "Diversity per bp"),
        ("mu", "Gamma mean", "Estimated $\\mu$"),
        ("sigma_sq", "Gamma variance", "Estimated $\\sigma^2$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (column, heading, ylabel) in zip(axes.flat, quantities):
        ax.plot(labels, selected[column], marker="o")
        ax.set_title(heading)
        ax.set_ylabel(ylabel)
        if column == "sigma_sq":
            ax.set_yscale("log")
    fig.suptitle(f"{dataset}: error-model estimates by doubleton-selection tier")


def plot_mismatch_rate_as_a_doubleton_classifier(
    *,
    L_mismatch_trim_values,
    ascertained_estimates,
    config,
    error_multipliers,
    true_doubleton_masks,
    figsize=(16, 10),
    linestyle="--",
    color="0.7",
    fontsize=8,
):
    """Mismatch rate as a doubleton classifier. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        3,
        len(error_multipliers),
        figsize=figsize,
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
            mismatch_rate = estimate.mismatches.counts[window_index] / (
                2 * L_mismatch_trim
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
        mismatch_rate = estimate.mismatches.counts[window_index] / (
            2 * L_mismatch_trim_values[-1]
        )
        roc = nb_utils.mismatch_roc(mismatch_rate, true_mask)
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
            ax.annotate(
                "ROC undefined: one class absent",
                (0.05, 0.05),
                xycoords="axes fraction",
            )
        ax.plot([0, 1], [0, 1], linestyle=linestyle, color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{multiplier:g}x error, ROC at L={L_mismatch_trim_values[-1]:g}")
        ax.set_xlabel("False positive rate")
    for ax in axes[:2, 0]:
        ax.set_ylabel("Empirical cumulative proportion")
    axes[2, 0].set_ylabel("True positive rate")
    axes[0, 0].legend(fontsize=fontsize)
    return fig


def plot_mismatch_rate_as_a_doubleton_classifier_2(
    *, prevalence_df, figsize=(6, 4.5), marker="o"
):
    """Mismatch rate as a doubleton classifier. Presentation choices are supplied by the notebook."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(
        prevalence_df["error_multiplier"], prevalence_df["prevalence"], marker=marker
    )
    ax.set_xlabel("Genotype error-rate multiplier")
    ax.set_ylabel("True-doubleton prevalence")
    ax.set_ylim(0, 1)
    ax.set_title("True-doubleton prevalence")
    return fig


def plot_mismatch_rate_as_a_doubleton_classifier_3(
    *,
    L_mismatch_trim_values,
    classification_df,
    cutoff_proportions,
    metric_specs,
    figsize=(12, 11),
    marker="o",
    fontsize=8,
):
    """Mismatch rate as a doubleton classifier. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        for L_mismatch_trim in L_mismatch_trim_values:
            for cutoff_proportion in cutoff_proportions:
                select = (classification_df["L_mismatch_trim"] == L_mismatch_trim) & (
                    classification_df["cutoff_proportion"] == cutoff_proportion
                )
                plot_df = classification_df[select].sort_values("error_multiplier")
                ax.plot(
                    plot_df["error_multiplier"],
                    plot_df[metric],
                    marker=marker,
                    label=f"L={L_mismatch_trim:g}, cutoff={cutoff_proportion:g}",
                )
        ax.set_xlabel("Genotype error-rate multiplier")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.set_title(title)
    axes[0, 0].legend(fontsize=fontsize)
    return fig


def plot_fitting_models_to_doubleton_data(
    *,
    classifier_curves,
    classifier_labels,
    classifier_tiers,
    cumulative_profiles,
    training_multipliers,
    figsize=(16, 9),
    color="0.7",
    linestyle="--",
    fontsize=8,
):
    """Fitting models to doubleton data. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    for col, multiplier in enumerate(training_multipliers):
        for tier in classifier_tiers:
            curves = classifier_curves[multiplier, tier]
            label = classifier_labels[tier]
            axes[0, col].plot(curves["fpr"], curves["tpr"], label=label)
            axes[1, col].plot(curves["tpr"], curves["precision"], label=label)
        axes[0, col].plot([0, 1], [0, 1], color=color, linestyle=linestyle)
        prevalence = cumulative_profiles[multiplier].is_true_doubleton.mean()
        axes[1, col].axhline(prevalence, color=color, linestyle=linestyle)
        axes[0, col].set_title(f"Held-out {multiplier}x: ROC")
        axes[1, col].set_title(f"Held-out {multiplier}x: precision-recall")
        axes[0, col].set_xlabel("False positive rate")
        axes[0, col].set_ylabel("True positive rate")
        axes[1, col].set_xlabel("Recall (true positive rate)")
        axes[1, col].set_ylabel("Precision")
        for ax in axes[:, col]:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=fontsize)
    fig.suptitle("Classifiers trained on the other two error multipliers")
    return fig


def plot_fitting_models_to_doubleton_data_2(
    *,
    classifier_cv_df,
    classifier_labels,
    classifier_tiers,
    training_multipliers,
    figsize=(11, 8),
    marker="o",
    fontsize=8,
):
    """Fitting models to doubleton data. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    for tier in classifier_tiers:
        tier_rows = classifier_cv_df[classifier_cv_df["tier"] == tier]
        axes[0, 0].plot(
            tier_rows["test_multiplier"],
            tier_rows["fpr"],
            marker=marker,
            label=classifier_labels[tier],
        )
        axes[0, 1].plot(
            tier_rows["test_multiplier"],
            tier_rows["tpr"],
            marker=marker,
            label=classifier_labels[tier],
        )
        axes[1, 0].plot(
            tier_rows["test_multiplier"],
            tier_rows["fraction_retained"],
            marker=marker,
            label=classifier_labels[tier],
        )
        axes[1, 1].plot(
            tier_rows["test_multiplier"],
            tier_rows["epsilon_relative_error"],
            marker=marker,
            label=classifier_labels[tier],
        )
    axes[0, 0].set_ylabel("False positive rate")
    axes[0, 1].set_ylabel("True positive rate")
    axes[1, 0].set_ylabel("Fraction of doubletons retained")
    axes[1, 1].set_ylabel("Absolute relative error in fitted epsilon")
    for ax in axes.flat:
        ax.set_xlabel("Held-out error multiplier")
        ax.set_xticks(training_multipliers)
    axes[0, 0].legend(fontsize=fontsize)
    fig.suptitle("Held-out classification and downstream error-rate fit")
    return fig


def plot_interpolation_at_2x_and_fitted_decision_surfaces(
    *,
    classifier_labels,
    classifier_tiers,
    interpolation_curves,
    interpolation_profiles,
    figsize=(11, 4.5),
    color="0.7",
    linestyle="--",
    fontsize=8,
):
    """Interpolation at 2x and fitted decision surfaces. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for tier in classifier_tiers:
        curves = interpolation_curves[tier]
        axes[0].plot(curves["fpr"], curves["tpr"], label=classifier_labels[tier])
        axes[1].plot(curves["tpr"], curves["precision"], label=classifier_labels[tier])
    axes[0].plot([0, 1], [0, 1], color=color, linestyle=linestyle)
    axes[1].axhline(
        interpolation_profiles.is_true_doubleton.mean(),
        color=color,
        linestyle=linestyle,
    )
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_xlabel("Recall (true positive rate)")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=fontsize)
    fig.suptitle("2x interpolation test; training used 1x, 5x and 10x")
    return fig


def plot_interpolation_at_2x_and_fitted_decision_surfaces_2(
    *,
    classifier_labels,
    grid_features,
    grid_x,
    grid_y,
    interpolation_features,
    interpolation_profiles,
    pooled_models,
    figsize=(12, 5),
    cmap="RdBu",
    alpha=0.6,
    colors="black",
    linewidths=2,
    s=3,
    alpha_2=0.08,
    fontsize=8,
    label="Fitted probability of a true doubleton",
):
    """Interpolation at 2x and fitted decision surfaces. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for ax, tier in zip(axes, ["tier_2", "tier_3"]):
        model = pooled_models[tier]
        grid_scores = nb_utils.doubleton_scores(model, grid_features)
        grid_probability = scipy.special.expit(grid_scores).reshape(grid_x.shape)
        filled = ax.contourf(
            grid_x,
            grid_y,
            grid_probability,
            levels=np.linspace(0, 1, 21),
            cmap=cmap,
            alpha=alpha,
        )
        ax.contour(
            grid_x,
            grid_y,
            grid_scores.reshape(grid_x.shape),
            levels=[model.threshold],
            colors=colors,
            linewidths=linewidths,
        )
        for select, label, color in [
            (interpolation_profiles.is_true_doubleton, "True doubletons", "tab:blue"),
            (
                ~interpolation_profiles.is_true_doubleton,
                "False doubletons",
                "tab:orange",
            ),
        ]:
            ax.scatter(
                interpolation_features[select, 0],
                interpolation_features[select, 1],
                s=s,
                alpha=alpha_2,
                color=color,
                label=label,
            )
        ax.set_title(classifier_labels[tier])
        ax.set_xlabel("log10 first-mismatch distance (bp)")
        ax.set_ylabel("Max clean count at max L")
    axes[0].legend(fontsize=fontsize)
    fig.colorbar(filled, ax=axes, label=label)
    fig.suptitle("2x doubletons and pooled logistic decision boundaries")
    return fig


def plot_selected_classifier_and_downstream_error_model(
    *,
    diagnostic_features,
    outcomes,
    selected_model,
    figsize=(8, 6),
    colors="black",
    s=4,
    alpha=0.16,
    fontsize=8,
):
    """Selected classifier and downstream error model. Presentation choices are supplied by the notebook."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    diagnostic_x = np.linspace(
        diagnostic_features[:, 0].min(), diagnostic_features[:, 0].max(), 120
    )
    diagnostic_y = np.linspace(
        diagnostic_features[:, 1].min(), diagnostic_features[:, 1].max(), 120
    )
    diagnostic_grid_x, diagnostic_grid_y = np.meshgrid(diagnostic_x, diagnostic_y)
    diagnostic_grid_features = np.column_stack(
        (diagnostic_grid_x.ravel(), diagnostic_grid_y.ravel())
    )
    diagnostic_grid_scores = nb_utils.doubleton_scores(
        selected_model, diagnostic_grid_features
    ).reshape(diagnostic_grid_x.shape)
    ax.contour(
        diagnostic_grid_x,
        diagnostic_grid_y,
        diagnostic_grid_scores,
        levels=[selected_model.threshold],
        colors=colors,
        linestyles="--",
    )
    for select, label, color in outcomes:
        ax.scatter(
            diagnostic_features[select, 0],
            diagnostic_features[select, 1],
            s=s,
            alpha=alpha,
            color=color,
            label=f"{label} (n = {select.sum()})",
        )
    ax.set_xlabel("log10 first-mismatch distance (bp)")
    ax.set_ylabel("Max clean count at max L")
    ax.set_title("10x diagnostic with pooled Tier 3 classifier")
    ax.legend(markerscale=3, fontsize=fontsize)
    return fig


def plot_pig_chr18_site_density_and_retained_quality_flags(
    *,
    all_sites_per_bin,
    pig_bin_midpoints_mb,
    pig_excluded,
    pig_flag_rates,
    pig_positions,
    retained_sites_per_bin,
    figsize=(13, 7),
    label="All Zarr sites",
    fontsize=8,
):
    """Pig chr18 site density and retained quality flags. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True, constrained_layout=True
    )
    axes[0].plot(pig_bin_midpoints_mb, all_sites_per_bin / 100, label=label)
    axes[0].plot(
        pig_bin_midpoints_mb,
        retained_sites_per_bin / 100,
        label=f"Retained by current mask ({len(pig_positions):,} sites; {np.count_nonzero(pig_excluded):,} removed)",
    )
    axes[0].set_ylabel("Sites per kb in 100-kb bins")
    axes[0].set_title(
        f"Pig chr18: retained span {pig_positions[0] / 1000000.0:.2f}–{pig_positions[-1] / 1000000.0:.2f} Mb"
    )
    axes[0].legend()
    for label, rates in pig_flag_rates.items():
        axes[1].plot(pig_bin_midpoints_mb, rates, label=label)
    axes[1].set_xlabel("Pig chr18 position (Mb)")
    axes[1].set_ylabel("Fraction of retained sites flagged")
    axes[1].set_ylim(0, 1)
    axes[1].legend(ncol=2, fontsize=fontsize)
    return fig


def plot_pig_fit_sensitivity_to_doubleton_trimming(
    *, pig_trim_df, pig_trim_proportions, figsize=(14, 4), marker="o"
):
    """Pig fit sensitivity to doubleton trimming. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    for ax, (column, ylabel) in zip(
        axes,
        [
            ("epsilon", "Errors per haplotype-bp"),
            ("mu", "Fitted $\\mu$"),
            ("sigma_sq", "Fitted $\\sigma^2$"),
        ],
    ):
        ax.plot(pig_trim_df["excluded_proportion"], pig_trim_df[column], marker=marker)
        ax.set_xlabel("Proportion of doubletons excluded")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_xticks(pig_trim_proportions)
    fig.suptitle("Pig chr18 fit sensitivity (trim at 100 kb)")
    return fig


def plot_pig_fit_sensitivity_to_doubleton_trimming_2(
    *,
    pig_estimate,
    pig_window_sizes,
    figsize=(11, 4),
    marker=".",
    label="Observed",
    label_2="Fitted",
    color="0.5",
    linestyle="--",
):
    """Pig fit sensitivity to doubleton trimming. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    axes[0].plot(
        pig_window_sizes, pig_estimate.fit.observed_means, marker=marker, label=label
    )
    axes[0].plot(pig_window_sizes, pig_estimate.fit.fitted_means, label=label_2)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("One-sided window length L (bp)")
    axes[0].set_ylabel("Mean full-window mismatch count")
    axes[0].legend()
    relative_residual = (
        pig_estimate.fit.observed_means - pig_estimate.fit.fitted_means
    ) / pig_estimate.fit.observed_means
    axes[1].plot(pig_window_sizes, relative_residual, marker=marker)
    axes[1].axhline(0, color=color, linestyle=linestyle)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("One-sided window length L (bp)")
    axes[1].set_ylabel("(Observed − fitted) / observed")
    fig.suptitle("Pig chr18: aggregate fit across window sizes")
    return fig


def plot_pig_doubleton_mismatches_and_local_site_density(
    *,
    pig_local_sites_per_kb,
    pig_trim_counts,
    figsize=(12, 4),
    gridsize=45,
    bins="log",
    label="Number of doubletons",
):
    """Pig doubleton mismatches and local site density. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ordered_counts = np.sort(pig_trim_counts)
    axes[0].plot(
        ordered_counts, np.arange(1, len(ordered_counts) + 1) / len(ordered_counts)
    )
    axes[0].set_xlabel("Mismatch count within ±100 kb")
    axes[0].set_ylabel("Empirical cumulative proportion")
    axes[0].set_title("Large variation among sampled doubletons")
    density_plot = axes[1].hexbin(
        pig_local_sites_per_kb, pig_trim_counts, gridsize=gridsize, bins=bins, mincnt=1
    )
    fig.colorbar(density_plot, ax=axes[1], label=label)
    axes[1].set_xlabel("Local retained sites per kb (±100 kb)")
    axes[1].set_ylabel("Mismatch count within ±100 kb")
    axes[1].set_title("Mismatch count versus local site density")
    return fig


def plot_exclusions_by_predictor(
    *,
    classifier_labels,
    classifier_tiers,
    real_classifier_df,
    real_datasets,
    real_selections,
    figsize=(14, 8),
    fontsize=8,
):
    """Exclusions by predictor. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
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
            axes[1, col].barh(
                "Sampled doubletons", count, left=left, label=f"{label}: {count:,}"
            )
            left += count
        axes[1, col].set_xlim(0, left)
        axes[1, col].set_xlabel("Number of doubletons")
        axes[1, col].set_title(f"{dataset}: overlap of one-predictor exclusions")
        axes[1, col].legend(fontsize=fontsize)
    fig.suptitle("Real-data doubleton exclusions from simulation-trained classifiers")
    return fig


def plot_exclusions_by_predictor_2(*, feature_sets, figsize=(12, 4.5), fontsize=8):
    """Exclusions by predictor. Presentation choices are supplied by the notebook."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
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
        ax.legend(fontsize=fontsize)
    fig.suptitle("Classifier predictor distributions in training and real data")
    return fig


def plot_real_tier_3_epsilon_against_simulation_estimates(
    *,
    fit_df,
    real_classifier_df,
    selected_fit_df,
    figsize=(13, 5),
    marker="o",
    markersize=3,
    linestyle="--",
    fontsize=8,
):
    """Real Tier 3 epsilon against simulation estimates. Presentation choices are supplied by the notebook."""
    fig, (ax, zoom_ax) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    simulation_curves = [
        (fit_df, "true_error", "True error in simulation"),
        (
            fit_df,
            "true_doubleton_epsilon",
            "Estimated error in simulation (fixed true doubletons)",
        ),
        (
            selected_fit_df,
            "epsilon",
            "Estimated error in simulation (ascertained doubletons)",
        ),
    ]
    for frame, column, label in simulation_curves:
        ax.plot(frame["error_multiplier"], frame[column], marker=marker, label=label)
        zoom_ax.plot(
            frame["error_multiplier"],
            frame[column],
            marker=marker,
            markersize=markersize,
        )
    real_epsilon_values = {}
    for dataset, color in [("TGP chr17", "tab:purple"), ("Pig chr18", "tab:brown")]:
        epsilon = real_classifier_df.loc[
            (real_classifier_df["dataset"] == dataset)
            & (real_classifier_df["tier"] == "tier_2"),
            "epsilon",
        ].iloc[0]
        real_epsilon_values[dataset] = epsilon
        ax.axhline(epsilon, color=color, linestyle=linestyle, label=f"{dataset}")
        zoom_ax.axhline(epsilon, color=color, linestyle=linestyle)
    zoom_ax.set_xlim(-0.1, 2.5)
    zoom_ax.set_ylim(0, 2.5 * max(real_epsilon_values.values()))
    ax.set_title("Full range")
    zoom_ax.set_title("Low-error range")
    for panel in (ax, zoom_ax):
        panel.set_xlabel("Genotype error-rate multiplier in simulation")
        panel.set_ylabel("Errors per haplotype-bp")
    ax.legend(fontsize=fontsize)
    fig.suptitle("Comparison: Simulated vs. real-data error estimates")
    return fig


def configure_notebook(settings):
    """Apply explicit notebook display defaults."""
    plt.rcParams.update(settings)


def close_figure(figure):
    """Release an already-displayed notebook figure."""
    plt.close(figure)


def save_notebook_figure(
    figure, path, *, dpi=150, bbox_inches="tight", facecolor="white"
):
    """Save a local exploratory figure at its explicit notebook destination."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)


def resize_figure(figure, *, figsize):
    """Set a displayed figure's physical dimensions explicitly."""
    figure.set_size_inches(*figsize)
