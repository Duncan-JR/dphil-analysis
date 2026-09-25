# Apply fitted doubleton classifiers to real data

## Aim and scope

Prototype the simulation-trained classifiers on the existing TGP chr17 and pig
chr18 estimates. Reuse their current variant masks, sampled doubletons,
recombination maps, and aggregate error-rate fitting equations. Make only the
small estimator change needed to recover classifier predictors from the same
masked variant set. Do not add unit tests, a generic classifier framework, or
real-data training labels.

## Recommended low-friction route

Use a **two-stage analysis** rather than putting a classifier into
`EstimationConfig` yet:

1. Run `estimate_error_rate` once for each dataset, as the notebook already
   does. Its Tier 0 fit remains the current fixed 25% mismatch trim.
2. Calculate the two one-sided predictors once per dataset for those exact
   sampled doubletons. Apply each already-fitted `DoubletonClassifier` object
   from `pooled_models` to the same predictor arrays. Treat its returned
   retained mask as the selection decision; do not recreate tier-specific
   threshold rules inside `error_estimation.py`.
3. For each learned tier, slice the existing `MismatchSummary.counts` and
   `.rates` by that mask and call the existing aggregate `fit_error_model`
   with further mismatch trimming disabled. Reuse the same dataset-wide
   diversity/`pi` for every tier. The existing validation helper
   `fit_error_rate_with_selected_doubletons` already implements this refit.

This keeps one expensive genotype/mismatch pass per dataset and makes all tier
comparisons use identical sampled doubletons. A `config.doubleton_classifier`
field would currently force an awkward callback into the estimator or repeat
the full estimate for each model. Add that single-call convenience only after
the prototype demonstrates a need for it. The notebook can simply use
`pooled_models["tier_1"]`, `["tier_2"]`, and `["tier_3"]`; Tier 1b remains a
separate clean-count predictor diagnostic. Keep `fixed_doubletons_path`
separate: it specifies *which* doubletons enter an estimate, whereas a
classifier decides which of those sampled doubletons enter the fit.

## Carry the variant masks through predictor extraction

`_open_store` already produces both masked positions and their raw Zarr row
indices. `Doubletons.site_indices` indexes those masked positions, while the
current `cumulative_mismatch_profiles` reopens the unmasked store and would
therefore use the wrong focal rows and neighboring variants on real data.

Make the minimal change in `error_estimation.py`: retain copies of the
selected positions and raw row indices on `ErrorRateEstimate` (for example,
`included_positions` and `included_variant_indices`). These are small
coordinate arrays, not additional genotype or one-sided-count state. The
existing masked estimator calculation remains unchanged.

Adapt the bounded one-sided reader in `error_validation.py` to use those
stored arrays rather than calling `_open_store` without masks. Compute
physical windows from included positions, map each included row to its raw
Zarr row when reading genotype chunks, and keep the focal indices in the
masked coordinate system throughout. At `max_L`, derive the clean-side count
and longer first-mismatch distance with the same definitions used for
simulation training. Allow real-data predictor extraction without a zero-error
truth Zarr; do not invent truth labels or change the fitted model parameters.
Warn about first-mismatch distances censored at `max_L`, as the current
feature code does.

Before interpreting results, inspect in the notebook that the stored raw
indices equal the complement of each configured mask combination, focal
positions match the sampled doubletons, and the predictor and selection arrays
have one entry per sampled doubleton in the same order. Confirm that all tiers
retain the same global diversity and differ only in the fitted subset. These
are direct analysis checks, not a new unit-test suite.

## Notebook section: `## Using doubleton classifiers on real data`

Use the existing `tgp_estimate` with both its density and duplicate-position
masks, and `pig_estimate` with its current density mask. Do not silently switch
the pig dataset to an additional quality mask; the earlier mask-sensitivity
analysis remains a separate comparison. Generate masked predictors once for
each estimate, classify with the pooled simulation models, then refit epsilon,
mu, and sigma-squared from each retained subset.

For **each of humans and pigs**, show:

- A bar plot of excluded-doubleton counts for Tier 0, distance-only Tier 1,
  clean-count-only Tier 1b, two-feature Tier 2, and interaction Tier 3.
  Include the common sampled-doubleton denominator and a compact breakdown
  of the overlap between the two one-predictor exclusion sets. This makes
  “excluded by predictor” literal without trying to assign a unique cause to
  a logistic decision that combines predictors.
- A four-panel plot across the **four numbered tiers 0, 1, 2, and 3** for
  epsilon, diversity, mu, and sigma-squared, with clearly marked model/tier
  labels. Report retained counts alongside the estimates. Tier 1b appears in
  the exclusion diagnostic and a small results table, while the requested
  four-tier estimate figure stays uncluttered. Diversity should coincide
  across tiers because it is computed from all included variants before
  doubleton filtering; label this explicitly. Use a readable scale for the
  pig variance if its magnitude otherwise hides the other values.

At the bottom, extend the existing simulation-versus-real epsilon plot. Keep
the simulation true-error and estimated-epsilon curves (including the
simulation Tier 3 curve), then overlay **horizontal dashed lines** at the pig
and TGP **Tier 3** epsilon estimates, with species/chromosome labels. These
lines are visual references, not evidence that a real dataset has a particular
simulation error multiplier.

## Interpretation and stopping point

Compare retained fractions, predictor distributions, and fit changes before
interpreting the absolute epsilon values. The models were trained on one
simulation family and may respond to pig data quality, population structure,
or masking in ways unrelated to genotype error. Record censoring and any
unexpected exclusion pattern in notebook prose. Stop after producing and
inspecting these plots; do not alter the error-rate model or expand the API
unless the prototype exposes a concrete need.
