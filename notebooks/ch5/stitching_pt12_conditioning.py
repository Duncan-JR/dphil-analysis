# -*- coding: utf-8 -*-
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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stitching pt12: calibrated mismatch and conditioned gaps
#
# Compare **Full HMM** and **greedy copying with conditioned gap HMMs** in
# two 500 kb windows, chr20 **[1,000,000, 1,500,000)** in GRCh38 coordinates.
# 1KGP uses the existing `tsinfer-match-eval/data/sample_lists/tgp_chr20_n100.csv`
# (100 diploid individuals). The error-containing Out-of-Africa dataset uses
# all 300 individuals, retaining its genotype-error and phase masks. No new
# data are simulated. Raw outputs share an inferred ancestor reference within
# each dataset, with no simplification, dating or variant augmentation.
#
# The only figures are the two recurrent-site quantities from pt11, recurrent
# site/mutation overlap with injected genotype errors, and mean sample mutations
# with mean copying switches stacked above them.
#
# **Pipeline limitation found in this pilot:** the greedy shared-reference HMM
# path assumes every reference mutation is derived. The calibrated ancestor
# runs contain back mutations, which this representation does not encode.
# Greedy leaves **117 / 516,800** called alleles incorrect in 1KGP and
# **119 / 1,728,600** in simulation; Full HMM has zero disagreements in both.
# The plots show the actual current pipeline outputs, but the greedy mutation
# counts and comparisons are **provisional** until that representation is fixed.

# %%
import logging
import pathlib

import ch5_conditioning
import ch5_conditioning_plots
import IPython.display as ipython_display

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

notebook_dir = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
repository_dir = notebook_dir if (notebook_dir / "experiments").is_dir() else notebook_dir.parent.parent
config_path = repository_dir / "experiments/ch5/stitching_pt12_conditioning_config.yaml"
config = ch5_conditioning.load_config(config_path)
plot = ch5_conditioning_plots.ConditioningPlots(config)

# %% [markdown]
# ## From errors per haplotype-bp to a mismatch probability
#
# Assume **ε = 10⁻⁵ errors per haplotype-bp** for both datasets. Chapter 4's
# [initial_testing](../ch4/initial_testing.py) reports a Tier 3 1KGP chr17
# estimate of about 1.08 × 10⁻⁵; here 10⁻⁵ is an assumed transfer to chr20
# and the simulation, not a newly fitted or independently validated rate.
#
# The precise earlier guidance is in the Codex discussion of **24 September
# 2026**, [`rollout-2026-09-24T08-46-39`](/Users/duncan/.codex/sessions/2026/09/24/rollout-2026-09-24T08-46-39-01a0d261-7b35-7c22-bd1f-d586bacd118e.jsonl).
# The [archived 29 September discussion](/Users/duncan/exo/archive/journals/this_week_2026-09-27_to_2026-10-03.org)
# agrees that emissions can be constant while map-based transition costs vary.
# The archived discussion separates unit conversion from empirical calibration.
#
# With **S quality-filtered source variant sites** in an analysed span **L bp**,
# the average density is λ = S/L and
#
# $$p_{\mathrm{mismatch}} = \frac{\epsilon}{\lambda}
#     = \epsilon\frac{L}{S}.$$
#
# Equating εL expected errors per haplotype to Sp gives this linear conversion.
# There is **no factor of two**: ε is already per haplotype. Here L is the
# requested 500,000 bp window, not the retained chromosome length. S counts
# regional source sites after fixed quality/ancestry filters, **before** the
# selected cohort's singleton and monomorphic-site exclusions. Replacing S
# with the smaller HMM site count would inflate p just because the cohort or
# inference filters changed. All retained sites have one row per position and
# known biallelic ancestry; 1KGP also uses its SNP/biallelic/bad-ancestry/duplicate
# masks. We assume this average per-site error probability transfers to the
# inference subset. The original chapter 4 chr17 masks differ, so this is a
# transparent working assumption rather than a re-estimate on the same sites.

# %%
if __name__ == "__main__":
    calibration = ch5_conditioning.prepare(config)
    columns = ["label", "span_bp", "num_individuals", "calibration_sites",
               "segregating_sites", "expected_inference_sites", "sites_per_bp",
               "epsilon_per_haplotype_bp", "mismatch_probability"]
    ipython_display.display(calibration[columns])

# %% [markdown]
# ## Recombination map and conditioned matching
#
# All three matching stages (ancestors, Full HMM samples, greedy gap HMMs) use
# the same chr20 HapMapII GRCh38 map and the dataset's calculated p. The pipeline
# integrates the three-column cM/Mb map into a cumulative map. At adjacent
# **actual inference positions**, tsinfer computes genetic distance dᵢ in
# Morgans and uses
#
# $$r_i = \frac{1-e^{-2d_i}}{2}, \qquad P(\mathrm{match})=1-p,
#       \quad P(\mathrm{mismatch})=p.$$
#
# Thus local recombination costs automatically reflect the map and site spacing;
# p is passed directly as a constant emission probability. No mismatch-ratio
# inversion or further multiplication/division by recombination is needed.
# These checks confirm the parameter units and implementation, not an empirical
# optimum for p. Absolute genomic coordinates are retained in both inputs.
#
# Greedy uses **AF ≤ 0.1** and a **25 kb mean-density equivalent** minimum
# connected focal region, the small-cohort setting from pt11. Exact copying
# remains fixed. Residual gaps condition on available left and right copied
# parents with native `initial_state` / `final_state`; their flanking
# observations supply the seam transition costs. Supported one-site gaps use
# the existing direct-copy rule and add a sample mutation when necessary.
# This does not globally optimise the greedy cover. Both methods see the same
# reference and HMM parameters. The existing pipeline runs with all available
# cores; only ordinary matching targets are run, without its diagnostic suite.

# %%
if __name__ == "__main__":
    ch5_conditioning.run_pipeline(config)
    tables = ch5_conditioning.analyse(config)
    columns = ["label", "inference_sites", "mismatch_probability",
               "median_genetic_distance_morgans", "median_recombination_probability",
               "min_recombination_probability", "max_recombination_probability"]
    ipython_display.display(tables.calibration[columns])
    ipython_display.display(tables.allele_checks)

# %% [markdown]
# The allele check is a table, with no additional validation figure. Greedy's
# [`share_reference`](../../../tsinfer-match-eval/lib/haplotypes.py) exports mutation
# site/node arrays, and native `MatcherIndexes.from_prepared` assumes those
# mutations all have allele code 1. The native Full HMM instead passes actual
# encoded mutation states, including code 0 for back mutations. This distinction
# matters when the calibrated ancestor references contain back mutations. A direct native index
# control is recorded alongside the data; fixing the shared representation is
# a separate pipeline change, not an adjustment to the requested p conversion.

# %% [markdown]
# ## Recurrent sites (pt11 definitions)
#
# A recurrent site has **at least two mutation records**, including back
# mutations. Percentages use **all saved inference sites** as the denominator;
# the conditional mean includes the first mutation record. An empty recurrent
# set has an undefined mean. Raw trees include reference mutations as in pt11.
# Simulation truth is evaluated at exactly the same inference positions with
# the same 300-individual cohort. There is no truth panel for real 1KGP.

# %%
if __name__ == "__main__":
    ipython_display.display(tables.recurrence)
    plot.save(plot.recurrence(tables.recurrence), "recurrent_sites")

# %% [markdown]
# ## Which recurrent sites and mutations overlap known errors?
#
# In this saved snapshot, `call_genotype_error_mask` has **True = injected
# error**. We verify its polarity against the companion error-free raw Zarr
# and `variant_genotype_error_count`; the current generator's opposite mask
# convention does not describe these saved data. It was recorded **before**
# phasing; where `call_genotype_phase` is true,
# swap the mask's two haplotype columns to align it with observed calls. These
# flags identify **injected genotype errors**, not phase-switch errors or
# erroneous ancestor states. Real 1KGP has no corresponding truth mask.
#
# First compare sets of sites: recurrent sites versus inference sites with at
# least one injected genotype error among the selected individuals. Circle
# areas are schematic. A shared site does not prove every mutation there is
# an error. The mutation plot therefore counts direct sample mutation records
# at flagged calls separately from sample records without an injected genotype error and
# internal/reference records, which have no unique individual call assignment.
# Even a direct overlap is not causal proof that the event was introduced by
# that error; biological recurrence and errors can coexist.

# %%
if __name__ == "__main__":
    ipython_display.display(tables.overlap)
    plot.save(plot.error_sites(tables.overlap), "recurrent_error_site_overlap")

# %%
if __name__ == "__main__":
    plot.save(plot.recurrent_mutation_errors(tables.overlap), "recurrent_mutation_error_overlap")

# %% [markdown]
# ## Mean sample mutations with copying switches stacked above
#
# Count mutation records **on sample nodes** across all inference sites and
# copying-parent changes between consecutive inference sites. Unobserved
# chromosome flanks do not contribute switches. Sum both haplotypes within
# each diploid individual and then average individuals equally. Reference
# mutations are excluded from this per-sample statistic. The stack combines
# two event counts for comparison; its total is not a likelihood or error score.

# %%
if __name__ == "__main__":
    ipython_display.display(tables.sample_means)
    plot.save(plot.copying_events(tables.sample_means), "sample_mutations_switches")

# %% [markdown]
# ## Reading this pilot
#
# In these provisional greedy outputs, greedy has fewer recurrent sites in both datasets, but a higher conditional
# mean of mutation records at those recurrent sites, and more copying switches.
# In simulation, 47.8% of Full HMM recurrent sites and 52.1% of greedy recurrent
# sites overlap at least one injected genotype error. Among direct sample
# mutation records at recurrent sites, the corresponding error-call overlaps
# are 43.9% and 50.8%. Most recurrent mutation records are on the reference,
# so the raw recurrence counts should not be read as counts of sample errors.
# The two methods retain the same reference; its contribution to *recurrent*
# records differs because their recurrent site sets differ.
# These are observations at the chosen p, not evidence that p is optimal.
# The allele disagreements above prevent treating the greedy comparison as a
# validated result of correctly reproducing all sample calls.
#
# %% [markdown]
# All native configurations, pipeline logs, sample identities, code revisions,
# raw trees, CSV tables and PNG/PDF versions of these figures are saved under
# `data/ch5/stitching_pt12_conditioning/`. The paired `.py` is the editable
# notebook source; rerunning reuses completed pipeline products.
