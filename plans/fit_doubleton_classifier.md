# Fit a doubleton classifier on simulated validation data

## Scope and source

Implement the final 2026-09-23 14:24:32 message in
`/Users/duncan/exo/documents/chat_threads/2026-09-23-refactor-error-estimation.md`.
Use the existing chr17 simulation estimates and zero-error Zarr truth labels.
Classifier fitting belongs in `src/error_validation.py`; plots and analysis
belong in `notebooks/ch4_initial_testing.py`. Leave the production estimator,
real-data cells, and an independent simulation replicate untouched. Exclude the
suggested shallow decision tree and larger model families.

## Models and selection

Use each sampled doubleton's `max_first_mismatch_distance` and
`max_clean_count`, with `log10` distance as the first feature. Warn if any
first-mismatch distance is censored at `max_L`; do not use censoring as a
predictor. Fit these tiers:

1. Tier 0: the existing highest-25%-total-mismatch exclusion at fixed
   `L_mismatch_trim`, with no classifier training.
2. Tier 1: a learned threshold on log first-mismatch distance.
3. Tier 1b: a learned threshold on max clean count.
4. Tier 2: unpenalized logistic regression on both features.
5. Tier 3: the same logistic regression plus their interaction.

For learned tiers, standardize predictors using training data only. Choose an
operating threshold from training labels that retains at least 95% of true
doubletons. Evaluate true doubletons as the positive class. Keep raw
classification scores so ROC, precision-recall, and AUROC/AP use all distinct
thresholds, not just the operating point.

## Validation and figures

Train on two of the 1x, 5x, and 10x sets and hold out the third in each of
three rotations. Pool 1x, 5x, and 10x for the final fitted classifier, and use
2x as an interpolation test. Treat these as shared-ancestry simulation checks,
not independent-replicate evidence. Report per-tier AUROC, average precision,
precision, TPR, FPR, and fraction retained; show ROC and precision-recall
curves, operating-point comparisons, and a two-feature decision surface for
the logistic candidates. Inspect a TP/FP/FN/TN feature-space diagnostic at
10x, where transfer of the operating threshold is weakest. Compare Tier 1b
even if the headline comparison emphasizes the four tiers in the source message.

Select the simplest tier whose held-out false-positive rate at target recall
is close to the best tier, then check the downstream estimator result before
finalizing. Apply the selected pooled classifier to the ascertained doubletons
at 0x, 1x, 2x, 5x, and 10x. Refit the existing aggregate error model using
only retained doubletons and plot its epsilon, pi, mu, and sigma-squared
results with the existing four-panel figure refactored into a reusable
function. Show the existing ascertained and fixed-true-doubleton fits for
comparison.

## Checks and interpretation

Confirm score direction, class labels, selected counts, and that training
thresholds and feature scaling are never derived from held-out labels. Inspect
all classifier and downstream figures. State that filtering may select a
nonrepresentative subset of genuine doubletons, and that the error levels
share one underlying simulation; reserve a scientific generalization claim
for a future independent replicate.

## Result of this analysis

Select Tier 3. Across the three held-out multipliers, its mean FPR is 0.379
versus 0.383 for Tier 2, and its mean AUROC is 0.826 versus 0.819. Its
downstream epsilon relative error is lower in all three held-out comparisons
(mean 0.340 versus 0.418). On the 2x interpolation set, the epsilon relative
errors are effectively tied: 0.00208 for Tier 3 and 0.00223 for Tier 2. The
pooled Tier 3 classifier retains 75.9% of the 2x sampled doubletons and gives
an epsilon estimate close to the simulation truth.

The result is not uniformly good. At 10x, the pooled threshold retains only
78.7% of true doubletons, and its downstream epsilon estimate is about 33%
below the realised error rate. Thus the trained operating point does not
transfer as a 95%-recall guarantee across error levels. The figure and tables
in the notebook keep this limitation visible.
