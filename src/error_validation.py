"""Validation quantities for simulated genotype-error analyses."""

import csv
import dataclasses
import pathlib
import warnings

import error_estimation
import numpy as np
import scipy.optimize
import scipy.special
import tskit
import zarr


@dataclasses.dataclass
class CumulativeMismatchProfiles:
    """One-sided mismatch counts for selected doubletons on a physical grid.

    Counts have shape (distance, doubleton). A side's clean/dirty identity is
    fixed by its count at the largest distance, with the left side clean on ties.
    """

    distances: np.ndarray
    left_count: np.ndarray
    right_count: np.ndarray
    is_true_doubleton: np.ndarray | None
    left_first_mismatch_distance: np.ndarray
    right_first_mismatch_distance: np.ndarray
    left_first_mismatch_censored: np.ndarray
    right_first_mismatch_censored: np.ndarray

    @property
    def max_left_count(self):
        return self.left_count[-1]

    @property
    def max_right_count(self):
        return self.right_count[-1]

    @property
    def left_is_clean(self):
        return self.max_left_count <= self.max_right_count

    @property
    def clean_count(self):
        return np.where(self.left_is_clean[None, :], self.left_count, self.right_count)

    @property
    def dirty_count(self):
        return np.where(self.left_is_clean[None, :], self.right_count, self.left_count)

    @property
    def max_clean_count(self):
        return np.minimum(self.max_left_count, self.max_right_count)

    @property
    def max_dirty_count(self):
        return np.maximum(self.max_left_count, self.max_right_count)

    @property
    def max_first_mismatch_distance(self):
        return np.maximum(
            self.left_first_mismatch_distance,
            self.right_first_mismatch_distance,
        )

    @property
    def max_first_mismatch_censored(self):
        maximum = self.max_first_mismatch_distance
        return (
            (self.left_first_mismatch_distance == maximum)
            & self.left_first_mismatch_censored
        ) | (
            (self.right_first_mismatch_distance == maximum)
            & self.right_first_mismatch_censored
        )


@dataclasses.dataclass
class DoubletonClassifier:
    """Simulation-trained score and operating threshold for true doubletons."""

    tier: str
    threshold: float | None
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    coefficients: np.ndarray | None = None


def doubleton_features(profiles):
    """Return log first-mismatch distance and max clean-side mismatch count."""
    if np.any(profiles.max_first_mismatch_censored):
        warnings.warn("Some first-mismatch distances are censored at max_L")
    log_distance = np.log10(profiles.max_first_mismatch_distance)
    return np.column_stack((log_distance, profiles.max_clean_count))


def _design_matrix(features, mean, scale, interaction):
    standardized = (features - mean) / scale
    columns = [np.ones(len(features)), standardized[:, 0], standardized[:, 1]]
    if interaction:
        columns.append(standardized[:, 0] * standardized[:, 1])
    return np.column_stack(columns)


def fit_doubleton_classifier(tier, training_profiles, *, target_recall=0.95):
    """Fit a tier and calibrate its threshold from true training doubletons."""
    if tier not in ("tier_1", "tier_1b", "tier_2", "tier_3"):
        raise ValueError(f"Unknown learned classifier tier: {tier}")
    features = np.concatenate([doubleton_features(p) for p in training_profiles])
    truth = np.concatenate([p.is_true_doubleton for p in training_profiles])
    if tier in ("tier_1", "tier_1b"):
        classifier = DoubletonClassifier(tier=tier, threshold=None)
    else:
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        design = _design_matrix(features, mean, scale, tier == "tier_3")
        labels = truth.astype(float)

        def objective(coefficients):
            logits = design @ coefficients
            loss = np.logaddexp(0, logits).sum() - labels @ logits
            gradient = design.T @ (scipy.special.expit(logits) - labels)
            return loss, gradient

        result = scipy.optimize.minimize(
            objective,
            np.zeros(design.shape[1]),
            jac=True,
            method="L-BFGS-B",
        )
        if not result.success:
            raise RuntimeError(f"{tier} logistic fit failed: {result.message}")
        classifier = DoubletonClassifier(
            tier=tier,
            threshold=None,
            feature_mean=mean,
            feature_scale=scale,
            coefficients=result.x,
        )
    scores = doubleton_scores(classifier, features)
    positive_scores = np.sort(scores[truth])
    retained_positive = int(np.ceil(target_recall * len(positive_scores)))
    threshold_index = len(positive_scores) - retained_positive
    classifier.threshold = float(positive_scores[threshold_index])
    return classifier


def doubleton_scores(classifier, features):
    """Return scores increasing with evidence for a true doubleton."""
    if classifier.tier == "tier_1":
        return features[:, 0]
    if classifier.tier == "tier_1b":
        return -features[:, 1]
    design = _design_matrix(
        features,
        classifier.feature_mean,
        classifier.feature_scale,
        classifier.tier == "tier_3",
    )
    return design @ classifier.coefficients


def classify_doubletons(classifier, profiles, summary, *, baseline_excluded=0.25):
    """Score and retain doubletons for one estimate, including the Tier 0 rule."""
    if classifier.tier == "tier_0":
        counts = summary.counts[-1]
        scores = -counts.astype(float)
        num_excluded = int(baseline_excluded * len(scores))
        order = np.argsort(counts, kind="stable")
        retained = np.zeros(len(scores), dtype=bool)
        retained[order[:len(scores) - num_excluded]] = True
        return scores, retained
    features = doubleton_features(profiles)
    scores = doubleton_scores(classifier, features)
    return scores, scores >= classifier.threshold


def classifier_curves(scores, truth):
    """ROC and precision-recall curves over every distinct score threshold."""
    truth = np.asarray(truth, dtype=bool)
    positives = np.count_nonzero(truth)
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Both truth classes are required for ROC and PR curves")
    order = np.argsort(scores, kind="stable")[::-1]
    ordered_scores = scores[order]
    ordered_truth = truth[order]
    group_end = np.r_[np.diff(ordered_scores) != 0, True]
    tp = np.cumsum(ordered_truth)[group_end]
    fp = np.cumsum(~ordered_truth)[group_end]
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    precision = np.r_[1.0, tp / (tp + fp)]
    average_precision = float(np.sum(np.diff(tpr) * precision[1:]))
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "auroc": float(np.trapezoid(tpr, fpr)),
        "average_precision": average_precision,
    }


def classifier_metrics(scores, truth, retained):
    """Summarize ranking and retained/excluded classification performance."""
    curves = classifier_curves(scores, truth)
    tp = np.count_nonzero(retained & truth)
    fp = np.count_nonzero(retained & ~truth)
    positives = np.count_nonzero(truth)
    negatives = len(truth) - positives
    return {
        "auroc": curves["auroc"],
        "average_precision": curves["average_precision"],
        "tpr": tp / positives,
        "fpr": fp / negatives,
        "precision": tp / (tp + fp),
        "fraction_retained": np.mean(retained),
    }


def fit_error_rate_with_selected_doubletons(estimate, retained):
    """Refit the existing aggregate error model on a validation-selected subset."""
    summary = estimate.mismatches
    selected_summary = error_estimation.MismatchSummary(
        window_sizes=summary.window_sizes,
        counts=summary.counts[:, retained],
        rates=summary.rates[:, retained],
    )
    config = dataclasses.replace(
        estimate.config,
        exclude_high_mismatch_proportion=0,
    )
    return error_estimation.fit_error_model(
        selected_summary,
        pi=estimate.fit.pi,
        config=config,
    )


def get_true_error_rate(zarr_path):
    """Return realised genotype errors per haplotype-bp from a simulation Zarr."""
    path = pathlib.Path(zarr_path).expanduser()
    group = zarr.open_group(path, mode="r")
    genotype = group["call_genotype"]
    positions = np.asarray(group["variant_position"][:], dtype=float)
    num_haplotypes = genotype.shape[1] * genotype.shape[2]
    sequence_span = positions[-1] - positions[0]
    num_errors = np.asarray(group["variant_genotype_error_count"][:]).sum()
    return float(num_errors / (num_haplotypes * sequence_span))


def get_ts_diversity(ts_path):
    """Return tree-sequence diversity per bp over its first-to-last site span."""
    path = pathlib.Path(ts_path).expanduser()
    ts = tskit.load(path)
    positions = np.asarray(ts.sites_position, dtype=float)
    sequence_span = positions[-1] - positions[0]
    diversity = ts.diversity(mode="site", span_normalise=False)
    return float(diversity / sequence_span)


def sample_doubletons(zarr_path, config):
    """Sample doubletons without running mismatch calculation or model fitting."""
    path = pathlib.Path(zarr_path).expanduser()
    group = zarr.open_group(path, mode="r")
    positions = np.asarray(group["variant_position"][:], dtype=float)
    G = np.asarray(group["call_genotype"][:])
    ac = G.sum(axis=(1, 2))
    return error_estimation._sample_doubletons(G, ac, positions, config)


def write_doubletons_csv(doubletons, path):
    """Write the complete :class:`error_estimation.Doubletons` state to CSV."""
    path = pathlib.Path(path).expanduser()
    fieldnames = [
        "site_index",
        "position",
        "sample_0",
        "sample_1",
        "ploidy_0",
        "ploidy_1",
        "num_eligible",
    ]
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(doubletons.site_indices)):
            writer.writerow(
                {
                    "site_index": doubletons.site_indices[index],
                    "position": doubletons.positions[index],
                    "sample_0": doubletons.sample_indices[index, 0],
                    "sample_1": doubletons.sample_indices[index, 1],
                    "ploidy_0": doubletons.ploidy_indices[index, 0],
                    "ploidy_1": doubletons.ploidy_indices[index, 1],
                    "num_eligible": doubletons.num_eligible,
                }
            )


def get_true_doubleton_mask(doubletons, truth_zarr_path):
    """Identify observed doubletons whose site and carriers match truth data."""
    path = pathlib.Path(truth_zarr_path).expanduser()
    group = zarr.open_group(path, mode="r")
    focal_G = np.asarray(
        group["call_genotype"].get_orthogonal_selection(
            (doubletons.site_indices, slice(None), slice(None))
        )
    )
    focal_ac = focal_G.sum(axis=(1, 2))
    row = np.arange(len(doubletons.site_indices))[:, None]
    carrier_alleles = focal_G[
        row,
        doubletons.sample_indices,
        doubletons.ploidy_indices,
    ]
    return (focal_ac == 2) & np.all(carrier_alleles == 1, axis=1)


def cumulative_mismatch_profiles(
    estimate,
    truth_zarr_path=None,
    *,
    distances=None,
):
    """Reread bounded genotype chunks for one-sided validation profiles.

    ``distances`` is independent of the fitting windows and must end at max_L.
    The truth store is optional for datasets without known doubleton labels.
    """
    max_L = float(estimate.config.window_sizes[-1])
    if distances is None:
        distances = np.geomspace(min(100.0, max_L), max_L, 60)
    distances = np.asarray(distances, dtype=float)
    if (
        distances.ndim != 1
        or len(distances) == 0
        or np.any(~np.isfinite(distances))
        or np.any(distances <= 0)
        or np.any(np.diff(distances) <= 0)
        or distances[-1] != max_L
    ):
        raise ValueError("distances must increase from positive values to max_L")

    doubletons = estimate.doubletons
    positions = estimate.included_positions
    raw_site_indices = estimate.included_variant_indices
    focal = doubletons.site_indices
    if not np.array_equal(positions[focal], doubletons.positions):
        raise ValueError("Selected variants do not match the estimate")
    focal_positions = doubletons.positions
    left = np.searchsorted(
        positions, focal_positions[None, :] - distances[:, None], side="left"
    )
    right = np.searchsorted(
        positions, focal_positions[None, :] + distances[:, None], side="right"
    )
    genotype = zarr.open_group(estimate.inference_path, mode="r")["call_genotype"]
    carriers = doubletons.sample_indices * genotype.shape[2] + doubletons.ploidy_indices
    num_doubletons = len(focal)
    shape = (len(distances), num_doubletons)
    left_count = np.zeros(shape, dtype=np.int64)
    right_count = np.zeros(shape, dtype=np.int64)
    left_first = np.full(num_doubletons, np.inf)
    right_first = np.full(num_doubletons, np.inf)

    chunk_size = genotype.chunks[0]
    num_chunks = (len(positions) + chunk_size - 1) // chunk_size
    for index in range(num_chunks):
        start = index * chunk_size
        stop = min(start + chunk_size, len(positions))
        first = np.searchsorted(right[-1], start, side="right")
        last = np.searchsorted(left[-1], stop, side="left")
        if first == last:
            continue
        selected_sites = raw_site_indices[start:stop]
        selection = (selected_sites, slice(None), slice(None))
        block = np.asarray(genotype.get_orthogonal_selection(selection))
        block = block.reshape(stop - start, -1)
        active = carriers[first:last]
        mismatch = block[:, active[:, 0]] != block[:, active[:, 1]]
        local_focal = focal[first:last] - start
        inside = (local_focal >= 0) & (local_focal < len(block))
        columns = np.arange(last - first)
        mismatch[local_focal[inside], columns[inside]] = False

        cumulative = np.zeros((len(block) + 1, last - first), dtype=np.int64)
        np.cumsum(mismatch, axis=0, out=cumulative[1:])
        local_left = np.clip(left[:, first:last] - start, 0, len(block))
        local_right = np.clip(right[:, first:last] - start, 0, len(block))
        local_focal_boundary = np.clip(local_focal, 0, len(block))
        left_count[:, first:last] += (
            cumulative[local_focal_boundary, columns]
            - cumulative[local_left, columns]
        )
        right_count[:, first:last] += (
            cumulative[local_right, columns]
            - cumulative[local_focal_boundary, columns]
        )

        mismatch_rows, mismatch_columns = np.nonzero(mismatch)
        doubleton_columns = first + mismatch_columns
        displacement = (
            positions[start + mismatch_rows] - focal_positions[doubleton_columns]
        )
        within = np.abs(displacement) <= max_L
        on_left = within & (displacement < 0)
        on_right = within & (displacement > 0)
        np.minimum.at(
            left_first, doubleton_columns[on_left], -displacement[on_left]
        )
        np.minimum.at(
            right_first, doubleton_columns[on_right], displacement[on_right]
        )

    left_censored = np.isinf(left_first)
    right_censored = np.isinf(right_first)
    left_first[left_censored] = max_L
    right_first[right_censored] = max_L
    is_true = None
    if truth_zarr_path is not None:
        is_true = get_true_doubleton_mask(doubletons, truth_zarr_path)
    return CumulativeMismatchProfiles(
        distances=distances,
        left_count=left_count,
        right_count=right_count,
        is_true_doubleton=is_true,
        left_first_mismatch_distance=left_first,
        right_first_mismatch_distance=right_first,
        left_first_mismatch_censored=left_censored,
        right_first_mismatch_censored=right_censored,
    )


def evaluate_mismatch_cutoffs(
    summary,
    true_doubleton_mask,
    *,
    cutoff_proportions,
    L_mismatch_trim_values,
):
    """Classify retained true doubletons as positive at each cutoff."""
    records = []
    num_doubletons = len(true_doubleton_mask)
    num_true = np.count_nonzero(true_doubleton_mask)
    num_false = num_doubletons - num_true
    for L_mismatch_trim in L_mismatch_trim_values:
        row = np.flatnonzero(summary.window_sizes == L_mismatch_trim)[0]
        order = np.argsort(summary.counts[row], kind="stable")
        for cutoff_proportion in cutoff_proportions:
            num_excluded = int(cutoff_proportion * num_doubletons)
            num_retained = num_doubletons - num_excluded
            retained = order[:num_retained]
            excluded = order[num_retained:]
            tp = np.count_nonzero(true_doubleton_mask[retained])
            fp = num_retained - tp
            tn = np.count_nonzero(~true_doubleton_mask[excluded])
            fn = num_excluded - tn
            records.append(
                {
                    "L_mismatch_trim": L_mismatch_trim,
                    "cutoff_proportion": cutoff_proportion,
                    "num_doubletons": num_doubletons,
                    "num_actual_positive": num_true,
                    "num_actual_negative": num_false,
                    "num_retained": num_retained,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "prevalence": num_true / num_doubletons,
                    "tpr": tp / num_true if num_true > 0 else np.nan,
                    "fpr": fp / num_false if num_false > 0 else np.nan,
                    "tnr": tn / num_false if num_false > 0 else np.nan,
                    "precision": tp / num_retained if num_retained > 0 else np.nan,
                    "npv": tn / num_excluded if num_excluded > 0 else np.nan,
                    "accuracy": (tp + tn) / num_doubletons,
                }
            )
    return records


def mismatch_roc(mismatch_rate, is_true_doubleton):
    """ROC at every distinct score; lower mismatch predicts a true doubleton."""
    score = -np.asarray(mismatch_rate)
    truth = np.asarray(is_true_doubleton, dtype=bool)
    positives = np.count_nonzero(truth)
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="stable")[::-1]
    ordered_score = score[order]
    ordered_truth = truth[order]
    last_in_group = np.r_[np.diff(ordered_score) != 0, True]
    tp = np.cumsum(ordered_truth)[last_in_group]
    fp = np.cumsum(~ordered_truth)[last_in_group]
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    auroc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auroc
