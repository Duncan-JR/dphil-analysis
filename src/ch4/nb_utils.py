"""Chapter 4 notebook preparation; methods live in the retained libraries."""

from __future__ import annotations

import logging

import numpy as np

from . import error_estimation, error_validation

logger = logging.getLogger(__name__)


def fit_error_model(summary, *, pi, config):
    """Delegate to :func:`error_estimation.fit_error_model`."""
    return error_estimation.fit_error_model(summary=summary, pi=pi, config=config)


def estimate_error_rate(
    zarr_path,
    *,
    recombination,
    config,
    pi=None,
    fixed_doubletons_path=None,
    variant_mask_name=None,
):
    """Delegate to :func:`error_estimation.estimate_error_rate`."""
    return error_estimation.estimate_error_rate(
        zarr_path=zarr_path,
        recombination=recombination,
        config=config,
        pi=pi,
        fixed_doubletons_path=fixed_doubletons_path,
        variant_mask_name=variant_mask_name,
    )


def doubleton_features(profiles):
    """Delegate to :func:`error_validation.doubleton_features`."""
    return error_validation.doubleton_features(profiles=profiles)


def fit_doubleton_classifier(tier, training_profiles, *, target_recall=0.95):
    """Delegate to :func:`error_validation.fit_doubleton_classifier`."""
    return error_validation.fit_doubleton_classifier(
        tier=tier, training_profiles=training_profiles, target_recall=target_recall
    )


def doubleton_scores(classifier, features):
    """Delegate to :func:`error_validation.doubleton_scores`."""
    return error_validation.doubleton_scores(classifier=classifier, features=features)


def classify_doubletons(classifier, profiles, summary, *, baseline_excluded=0.25):
    """Delegate to :func:`error_validation.classify_doubletons`."""
    return error_validation.classify_doubletons(
        classifier=classifier,
        profiles=profiles,
        summary=summary,
        baseline_excluded=baseline_excluded,
    )


def classifier_curves(scores, truth):
    """Delegate to :func:`error_validation.classifier_curves`."""
    return error_validation.classifier_curves(scores=scores, truth=truth)


def classifier_metrics(scores, truth, retained):
    """Delegate to :func:`error_validation.classifier_metrics`."""
    return error_validation.classifier_metrics(
        scores=scores, truth=truth, retained=retained
    )


def fit_error_rate_with_selected_doubletons(estimate, retained):
    """Delegate to :func:`error_validation.fit_error_rate_with_selected_doubletons`."""
    return error_validation.fit_error_rate_with_selected_doubletons(
        estimate=estimate, retained=retained
    )


def get_true_error_rate(zarr_path):
    """Delegate to :func:`error_validation.get_true_error_rate`."""
    return error_validation.get_true_error_rate(zarr_path=zarr_path)


def get_ts_diversity(ts_path):
    """Delegate to :func:`error_validation.get_ts_diversity`."""
    return error_validation.get_ts_diversity(ts_path=ts_path)


def sample_doubletons(zarr_path, config):
    """Delegate to :func:`error_validation.sample_doubletons`."""
    return error_validation.sample_doubletons(zarr_path=zarr_path, config=config)


def write_doubletons_csv(doubletons, path):
    """Delegate to :func:`error_validation.write_doubletons_csv`."""
    return error_validation.write_doubletons_csv(doubletons=doubletons, path=path)


def get_true_doubleton_mask(doubletons, truth_zarr_path):
    """Delegate to :func:`error_validation.get_true_doubleton_mask`."""
    return error_validation.get_true_doubleton_mask(
        doubletons=doubletons, truth_zarr_path=truth_zarr_path
    )


def cumulative_mismatch_profiles(estimate, truth_zarr_path=None, *, distances=None):
    """Delegate to :func:`error_validation.cumulative_mismatch_profiles`."""
    return error_validation.cumulative_mismatch_profiles(
        estimate=estimate, truth_zarr_path=truth_zarr_path, distances=distances
    )


def evaluate_mismatch_cutoffs(
    summary, true_doubleton_mask, *, cutoff_proportions, L_mismatch_trim_values
):
    """Delegate to :func:`error_validation.evaluate_mismatch_cutoffs`."""
    return error_validation.evaluate_mismatch_cutoffs(
        summary=summary,
        true_doubleton_mask=true_doubleton_mask,
        cutoff_proportions=cutoff_proportions,
        L_mismatch_trim_values=L_mismatch_trim_values,
    )


def mismatch_roc(mismatch_rate, is_true_doubleton):
    """Delegate to :func:`error_validation.mismatch_roc`."""
    return error_validation.mismatch_roc(
        mismatch_rate=mismatch_rate, is_true_doubleton=is_true_doubleton
    )


def make_estimation_config(
    *,
    window_sizes,
    num_doubletons=10000,
    random_seed=None,
    exclude_high_mismatch_proportion=0.25,
    L_mismatch_trim=None,
    num_workers=None,
    queue_depth=2,
    worker_poll_seconds=0.2,
    optimizer_bounds=((-30, 30), (-30, 30), (-50, float(np.log(0.5)))),
    start_alpha=(0.25, 0.5, 0.75),
    start_log_beta=(-12, -9, -6, -3, 0),
    start_log_epsilon=(-50, -30, -20, -12),
    objective_penalty=1e100,
    residual_guard=1e50,
):
    """Initialise notebook settings for the estimator."""
    return error_estimation.EstimationConfig(
        window_sizes=window_sizes,
        num_doubletons=num_doubletons,
        random_seed=random_seed,
        exclude_high_mismatch_proportion=exclude_high_mismatch_proportion,
        L_mismatch_trim=L_mismatch_trim,
        num_workers=num_workers,
        queue_depth=queue_depth,
        worker_poll_seconds=worker_poll_seconds,
        optimizer_bounds=optimizer_bounds,
        start_alpha=start_alpha,
        start_log_beta=start_log_beta,
        start_log_epsilon=start_log_epsilon,
        objective_penalty=objective_penalty,
        residual_guard=residual_guard,
    )


def make_doubleton_classifier(*, tier, threshold):
    """Construct the retained classifier for a notebook control."""
    return error_validation.DoubletonClassifier(tier=tier, threshold=threshold)
