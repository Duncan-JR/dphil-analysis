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
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stitching pt12: fixed alleles, conditioned gaps and low ε
#
# This replaces the provisional 500 kb pilot with the **completed, allele-corrected
# conditioning experiment** from 14 November 2026. Simulations contain 50, 500
# and 5,000 diploid individuals across chr20p (28.1 Mb), with 1% injected genotype
# error and no phase error. 1KGP contains 3,202 diploid individuals. The actual
# inference-site extents are shown below; unobserved chromosome flanks are excluded.
# All three methods share each dataset's inferred ancestor reference. Back mutations
# are represented correctly, and every saved called-allele check passed.
#
# Scientific diagnostics use `conditioning_test_rfloor_1e20_diagnostics.tar.gz` and
# `conditioning_test_control_diagnostics.tar.gz`. The control disables gap conditioning
# and retains the same epsilon, map floor, focal settings, inputs and ancestor references.
# **Only the performance section** uses the later
# `conditioning_greedy_optimized_20261114` trials. No superseded timings are plotted.
# The optimisation checks establish identical greedy edge and mutation arrays
# across all eight dataset/mode comparisons. Raw outputs and source snapshots
# remain intact. The replaced notebook and its saved outputs are backed up under
# `data/ch5/provenance/pt12_before_20261116/`.

# %%
import logging
import pathlib

import pandas as pd

import IPython.display as ipython_display

from ch5 import nb_plot, nb_utils

repository_dir = pathlib.Path.cwd()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# %%
config_path = {
    "base_dir": repository_dir,
    "data_dir": repository_dir / "data/ch5/stitching_pt12_conditioning/corrected_run",
    "cache_subdir": "analysis_20261116",
    "workers": None,
    "helper_threads": 1,
    "mutation_rate": 2.35e-8,
    "dating": {
        "post_process": {"split_ultimate": True, "erase_flanks": True},
        "preprocess": {"delete_intervals": [], "split_disjoint": True},
        "estimation": {
            "method": "variational_gamma",
            "time_units": "generations",
            "set_metadata": True,
            "progress": False,
            "singletons_phased": True,
        },
    },
    "frequency_bins": {
        "cutoffs": [0.01, 0.5],
        "labels": ["DAF < 0.01", "0.01 ≤ DAF < 0.5", "DAF ≥ 0.5"],
    },
}
performance_dir = (
    repository_dir.parent
    / "tsinfer-match-eval/data/performance/conditioning_greedy_optimized_20261114"
)
config = nb_utils.paramsearch_load_config(config_path)
plot_data = nb_utils.prepare_scaling_data(config)
cohort_labels = {
    "out_of_africa_n50_28_1mbp_error": "50 OOA",
    "out_of_africa_n500_28_1mbp_error": "500 OOA",
    "out_of_africa_n5000_28_1mbp_error": "5,000 OOA",
    "tgp_chr20p_n3202": "3,202 1KGP",
}
control_path = dict(config_path)
control_path["data_dir"] = (
    repository_dir / "data/ch5/stitching_pt12_conditioning/unconditioned_run"
)
control = nb_utils.paramsearch_load_config(control_path)
comparison = dict(config)
comparison["cache_dir"] = config["data_dir"].parent / "comparison_20261116"
comparison["cache_dir"].mkdir(exist_ok=True)
plot_data = nb_utils.prepare_scaling_data(comparison)
method_colors = {"full_hmm": "#555555", "greedy": "#0072B2", "unconditioned": "#D55E00"}
method_labels = {
    "full_hmm": "Full HMM",
    "greedy": "Conditioned greedy",
    "unconditioned": "Unconditioned greedy",
}


# %% [markdown]
# ## Scientific settings and allele checks
#
# The archived sample settings use **ε = 10⁻⁹ errors per haplotype-bp**.
# With S quality-filtered source sites across L = 28,100,000 bp, the constant
# sample mismatch probability is **p = εL/S**, before cohort singleton and
# monomorphic inference exclusions. It is not ε itself and is not fitted here.
# The resulting p values range from about 4.26 × 10⁻⁸ to 1.93 × 10⁻⁷.
# For the simulated 1% error data, this is deliberately a low configured rate,
# not a calibrated estimate of the injected error process.
#
# Samples use the chr20 HapMap map with **rᵢ = (1 − exp(−2dᵢ))/2** and a
# **recombination floor of 10⁻²⁰** on transitions after the first site. This
# changes 4,536 / 11,936 / 30,409 transitions in the three simulations and none
# in 1KGP; positive map transitions are unchanged. The floor is a modelling
# change: it permits switching where the original map was exactly zero.
# A captured 258-site simulated gap had different pinned endpoints and zero
# recombination throughout, so its two-sided constraint was impossible without
# such a change. Likelihood flooring and both pruning thresholds are **zero**.
# Ancestors retain their stock native probabilities, without the sample map or
# sample ε. These runs therefore do not apply the low ε to ancestor inference.
#
# Residual greedy gaps condition on available left/right focal parents; exact
# focal copying remains fixed. Supported one-site gaps retain the direct-copy
# rule. AF cutoffs are 0.1 / 0.05 / 0.05 / 0.05 and minimum focal lengths are
# 25 / 50 / 100 / 150 kb (mean-density site-count equivalents).
# Disabling conditioning removes endpoint pins and the extra flanking sites
# from residual HMM windows; the fixed focal cover and supported one-site
# direct-copy rule remain the same. Full HMM is the unconstrained whole-span sample comparison. The matched unconditioned-greedy control isolates gap conditioning at fixed ε.
# There is no ε sweep, so these results do not isolate the effect of changing ε.

# %%
if __name__ == "__main__":
    cohorts = nb_utils.cohort_table(config)
    ipython_display.display(cohorts)
    settings = nb_utils.conditioning_review_settings(config)
    ipython_display.display(settings)
    allele_checks = nb_utils.conditioning_review_quality(config)
    ipython_display.display(allele_checks)
    control_settings = nb_utils.conditioning_review_settings(control)
    ipython_display.display(control_settings)
    input_checks = nb_utils.conditioning_review_control_inputs(config, control)
    ipython_display.display(input_checks)


# %% [markdown]
# ## Paired path weights, mismatches and additional switches
#
# At fixed haploid cohort size n, staying has weight **1 − rᵢ + rᵢ/n** and
# switching to a particular parent has weight **rᵢ/n**. A biallelic match has
# emission 1 − p and a mismatch p; missing observations have neutral emission.
# Each saved path is scored using log switch odds and log mismatch odds,
# relative to common all-stay/all-match factors. The common initial factor and
# missing-call factors cancel between methods for the same haplotype.
# These are **Viterbi path weights**, not forward marginal likelihoods or
# posterior probabilities of genealogies. Compare paired differences within
# each cohort, rather than raw scores across datasets.
#
# Terminal sample mutations measure mismatches to the copied reference,
# including back mutations. All raw copying paths share the inference-site
# axis; no new matching or genotype simulation is launched. Additional parent
# changes are partitioned into focal interiors, gap interiors and focal/gap
# seams using the same saved focal mask for both methods. The fixed greedy
# cover need not maximise whole-span weight, even when each HMM gap is
# optimised conditional on its endpoints. Low p makes residual mismatches
# costly, but extra switches also incur map- and n-dependent costs.

# %%
if __name__ == "__main__":
    paired_paths = nb_utils.conditioning_review_paths(config)
    paired_paths["conditioning"] = "greedy"
    control_paths = nb_utils.conditioning_review_paths(control)
    control_paths["conditioning"] = "unconditioned"
    paired_paths = pd.concat([paired_paths, control_paths], ignore_index=True)
    summaries = []
    for method, frame in paired_paths.groupby("conditioning", sort=False):
        summary = nb_utils.conditioning_review_summary(frame)
        summary["conditioning"] = method
        summaries.append(summary)
    path_summary = pd.concat(summaries, ignore_index=True)
    paired_paths.to_csv(
        comparison["cache_dir"] / "paired_path_diagnostics.csv", index=False
    )
    path_summary.to_csv(comparison["cache_dir"] / "path_summary.csv", index=False)
    ipython_display.display(path_summary.round(4))
    nb_plot.save_cohort_figure(
        plot_data,
        nb_plot.plot_conditioning_review_paths(
            paired_paths,
            path_summary,
            cohort_labels=cohort_labels,
            figsize=(15, 10),
            dpi=120,
            bins=60,
            full_color="#555555",
            method_colors={
                key: color for key, color in method_colors.items() if key != "full_hmm"
            },
            method_labels=method_labels,
        ),
        "conditioned_path_costs",
        export_dpi=180,
    )

# %% [markdown]
# ## Recurrent mutations on the common inference-site axis
#
# A recurrent site has at least two mutation records, including reversions.
# The denominator is every inference site, and the conditional mean includes
# the first mutation. Counts use **raw** reference-plus-sample trees before
# simplification or dating. Truth uses the identical simulated inference-site
# positions. The raw methods share their reference mutations, so differences
# reflect sample matching; recurrent sites themselves need not be genotype
# errors. Real 1KGP has no mutation truth panel.

# %%
if __name__ == "__main__":
    recurrence = nb_utils.recurrent_sites(config)
    control_recurrence = nb_utils.recurrent_sites(control)
    control_recurrence = control_recurrence.loc[
        control_recurrence.method.eq("greedy")
    ].copy()
    control_recurrence["method"] = "unconditioned"
    recurrence = pd.concat([recurrence, control_recurrence], ignore_index=True)
    recurrence.to_csv(
        comparison["cache_dir"] / "recurrent_site_summary.csv", index=False
    )
    ipython_display.display(recurrence)
    nb_plot.save_cohort_figure(
        plot_data,
        nb_plot.plot_recurrence(
            plot_data,
            recurrence,
            cohort_labels=cohort_labels,
            method_colors=method_colors,
            method_labels=method_labels,
            panel_width=5.5,
            panel_height=4.0,
            stage_label_margin=0.3,
        ),
        "conditioned_recurrent_sites",
        export_dpi=180,
    )

# %% [markdown]
# ## Inferred allele ages against Full HMM and simulation truth
#
# Date all three saved inferred sample genealogies with tsdate variational gamma,
# mutation rate 2.35 × 10⁻⁸ per bp per generation, after the same post-processing
# and preprocessing used by pt13. Matching is not rerun. Dates are posterior
# mutation means in generations; true mutation times come from the simulations.
# Reference ancestor times are not themselves these inferred allele ages.
#
# Each cohort uses **one common position/ancestral-allele/derived-allele
# intersection across all three methods and truth where available**. Only finite,
# single-mutation sites have a unique pairing. Recurrent sites are therefore
# excluded from age comparisons and analysed separately above. Retention counts
# below show this selection. DAF bins use Full HMM frequencies for all methods.
# Pearson r, Spearman ρ, signed mean age error and median absolute age error
# are reported; correlation alone does not establish age accuracy.
# 1KGP compares both greedy methods with Full HMM only. Hexbin display limits clip the
# 99.9th percentile for readability; statistics retain all matched mutations.

# %%
if __name__ == "__main__":
    dated = nb_utils.perf_scaling_date_all(config)
    control_dated = nb_utils.conditioning_review_date_control(control)
    dated = pd.concat([dated, control_dated], ignore_index=True)
    age_correlations = nb_utils.perf_scaling_mutation_correlations(comparison, dated)
    retention = nb_utils.conditioning_review_age_retention(comparison)
    ipython_display.display(retention)
    ipython_display.display(age_correlations.round(4))
    nb_plot.save_cohort_figure(
        plot_data,
        nb_plot.plot_age_correlations(
            plot_data,
            age_correlations,
            frequency_bin="all",
            metrics={
                "pearson_r": "Pearson r",
                "spearman_r": "Spearman ρ",
                "median_absolute_error_generations": "Median absolute age error (generations)",
            },
            cohort_labels=cohort_labels,
            method_colors=method_colors,
            method_labels=method_labels,
            panel_width=4.2,
            panel_height=3.5,
            sample_tick_rotation=20,
        ),
        "conditioned_age_correlations",
        export_dpi=180,
    )

# %%
if __name__ == "__main__":
    for reference in ["truth", "full_hmm"]:
        nb_plot.save_cohort_figure(
            plot_data,
            nb_plot.plot_age_density(
                plot_data,
                age_correlations,
                reference,
                cohort_labels={
                    name: label + "\n" for name, label in cohort_labels.items()
                },
                method_labels=method_labels,
                method_colors=method_colors,
                panel_width=5.2,
                panel_height=4.2,
                age_hexbin_gridsize=65,
                age_maximum_quantile=0.999,
                age_minimum=1,
            ),
            f"conditioned_age_pairs_{reference}",
            export_dpi=180,
        )

# %% [markdown]
# ## Performance: final optimisation trials only
#
# These are the final 14 November trials after batched regional calls, native
# numeric path assembly, regional tree seeking and overlapped input preparation.
# There is one trial per dataset/method/mode with 96 matching workers; no
# replicate uncertainty is implied. **From reference** includes preparation,
# sample matching and finished output from an existing reference. **Resident
# match** times matching/assembly with resident prepared inputs. Neither includes
# upstream ancestor inference. These timers are kept separate.
#
# The latest suite includes 1KGP. All 16 trials pass called-allele checks and
# eight greedy outputs have exactly the corrected baseline edge/mutation arrays.
# The archived optimisation logs report 688 TSLS checks and 55 pipeline checks
# passing. Native sources were dirty; `source_snapshot/` and
# `final_validation.json` beside the final trials preserve their identities.
# This notebook reads those results without rebuilding dependencies or rerunning
# their test suites. USS is sampled process-tree private memory; CPU/wall is
# achieved average cores. The small cohorts can remain slower end to end even
# when resident matching is faster. Cohorts have different settings and site
# densities; connecting lines are visual guides, not controlled scaling curves.

# %%
if __name__ == "__main__":
    performance = nb_utils.conditioning_review_performance(
        performance_dir, config["cache_dir"]
    )
    columns = [
        "dataset",
        "mode",
        "method",
        "wall_s",
        "wall_ms_per_haplotype",
        "speedup",
        "effective_cores",
        "window_peak_private_gib",
    ]
    ipython_display.display(performance[columns].round(4))
    nb_plot.save_cohort_figure(
        plot_data,
        nb_plot.plot_conditioning_review_performance(
            performance,
            cohort_labels=cohort_labels,
            figsize=(12, 7),
            dpi=120,
            full_color="#555555",
            greedy_color="#0072B2",
        ),
        "latest_optimised_performance",
        export_dpi=180,
    )

# %% [markdown]
# ## Interpretation of the matched control
#
# Conditioning improves or preserves whole-span saved-path weight for every one of
# the 17,504 paired haplotypes (numerical tolerance 1e-6 in log weight). Mean
# conditioned minus unconditioned differences at the same epsilon are:
#
# | Cohort | Log path weight | Terminal mutations | Parent switches |
# | --- | ---: | ---: | ---: |
# | out_of_africa_n50_28_1mbp_error | +471.22 | +32.16 | -48.78 |
# | out_of_africa_n500_28_1mbp_error | +650.94 | +28.47 | -44.06 |
# | out_of_africa_n5000_28_1mbp_error | +583.69 | +19.25 | -30.03 |
# | tgp_chr20p_n3202 | +112.88 | +5.60 | -7.64 |
#
# Full HMM remains the common whole-span baseline. Age statistics use one common
# nonrecurrent mutation intersection across all three inferred methods and truth
# where available. Read retention alongside age errors: recurrent and unpaired
# mutations are excluded. Full HMM agreement in 1KGP is not a truth assessment.
# Performance retains the original conditioned suite. There is no epsilon sweep.

# %% [markdown]
# ## Deep dive: why the zero-recombination conditioned match was impossible
#
# This is the **captured n50 failure**, not a toy example: `tsk_0`, haplotype 0,
# job 0 in the 14 November replay log. The request covers inference sites
# `[90955, 91213)` (258 sites including both pinned flanks). Its 256-site residual
# gap lies between the left anchor at **26,656,210 bp** (parent **50854**) and the
# right anchor at **26,758,466 bp** (parent **50856**). All **257 transitions**
# between the included sites have recombination probability exactly zero.
#
# Evidence below is preserved from `tmp/conditioning_numerics_20261114/`, including
# the captured query, request, replay log and numerical controls. The Full HMM
# comparator is the **original zero-map Full HMM tree** from the completed n50
# trial; it passed its recorded called-allele check. It is not substituted by a
# new full-sequence run. We also check the saved floored Full HMM and conditioned
# greedy paths, and replay only this small window with native TSLS. No cohort
# matching, ancestor inference, performance trials or dating runs here.
#
# The simulation stores literal binary allele labels **0** and **1**. The figures
# show these actual saved alleles, not invented nucleotide letters or parent IDs.
# Every saved sample haplotype is checked against the complete captured query.

# %%
if __name__ == "__main__":
    failure_dir = (
        repository_dir
        / "data/ch5/stitching_pt12_conditioning/failure_deep_dive_20261116"
    )
    failure_data = nb_utils.conditioning_failure_analysis(
        config,
        evidence_dir=failure_dir / "evidence",
        output_dir=failure_dir,
        recombination_floor=1e-20,
        context_sites=12,
        diagnostic_mismatch=0.01,
    )
    ipython_display.display(pd.Series(failure_data.facts, name="value").to_frame())
    ipython_display.display(failure_data.replay_summary)
    archived_floor_controls = pd.read_json(
        failure_dir / "evidence/recombination_floor_controls.json"
    )
    ipython_display.display(archived_floor_controls)
    failure_config = dict(config)
    failure_config["cache_dir"] = failure_dir
    failure_plot_data = nb_utils.prepare_scaling_data(failure_config)

# %% [markdown]
# ### Where the zeros originate, and what the two saved paths do
#
# The original three-column HapMap source assigns rate **0 cM/Mb** at
# **26,328,619 bp**, with the next map entry at **30,187,708 bp**. Integrating that
# interval rate produces the archived cumulative-map plateau at approximately
# **50.9869712 cM**. The failed window lies strictly inside this 3.86 Mb plateau.
# There are no intervening map markers. This is what the supplied map assigns;
# it does not establish that biological recombination is literally impossible
# throughout the interval.
#
# The helper verifies the normalized map against the original source, checks that
# all 257 genetic distances are **exactly zero before** applying Haldane's
# conversion, and confirms that the stable `-expm1(-2d)/2` expression is also zero.
# This failure is therefore not caused by cancellation in `1 - exp(-2d)`, tiny
# likelihood underflow, or pruning. The archived likelihood/pruning sweeps failed
# too; the new replay keeps all three approximation thresholds at zero.
#
# The original zero-map Full HMM stays on parent **58721** in this window. The
# saved conditioned run with floor `1e-20` follows **50854 → 58721 → 50856**.
# The latter can join the two fixed focal parents because the floor assigns a
# positive probability to both otherwise forbidden changes. The replay is checked
# against that saved path, and the zero-map and floored Full HMM paths are checked
# to agree inside this window.
#
# The final table above is the **archived local floor sensitivity diagnostic**,
# not a new cohort experiment. At floors `1e-300` and `1e-100` the constrained
# window has one switch and eight mismatches; at `1e-20` it has two switches and
# zero mismatches. A positive floor repairs reachability, but its value also
# changes the optimum. It is therefore a scientific parameter, not a harmless
# way of avoiding a floating-point exception.

# %%
if __name__ == "__main__":
    nb_plot.save_cohort_figure(
        failure_plot_data,
        nb_plot.plot_conditioning_failure_overview(
            failure_data,
            figsize=(17, 11),
            dpi=120,
            map_limits_mb=(26.1, 30.4),
            analysis_end_bp=28100000,
            full_color="#222222",
            floor_color="#0072B2",
            impossible_color="#B2182B",
            focal_color="#B8DED4",
            gap_color="#FFE3A3",
        ),
        "zero_recombination_failure_overview",
        export_dpi=180,
    )

# %% [markdown]
# ### Why adding mismatches cannot rescue both endpoint constraints
#
# A mismatch is an **emission**: the sample can differ from the allele of the
# parent it continues to copy. It does not change which parent is copied.
# For this HMM, the transition weights are
#
# \[
# T_i(u,v)=\begin{cases}1-r_i+r_i/n,&u=v,\\r_i/n,&u\ne v.\end{cases}
# \]
#
# Here `r_i = 0`, so staying has weight 1 and changing parent has weight 0.
# The original mismatch probability is positive (about `1.93e-7`), so any finite
# number of mismatches has a positive emission weight. But a path from pinned
# parent **50854** to different pinned parent **50856** must contain at least one
# parent change. Its transition product is zero regardless of how many mismatches
# we allow: **positive emission weight × zero transition weight = zero**.
#
# The one-sided replays show the distinction directly. Keeping only the left pin
# allows TSLS to stay on 50854 and add **eight mismatches**; keeping only the right pin
# allows it to stay on 50856 and add **ten mismatches**. Neither constant path satisfies
# both pins. Even raising the diagnostic mismatch probability to `0.01` leaves
# the two-pin zero-map match impossible. The table reports `MatchImpossible`
# rather than inventing a returned path. The dotted red line in the allele panels
# is the successful **left-pin-only diagnostic**, whose final parent violates the
# original right pin; it is not a solution of the failed request.
#
# In native TSLS, `tsls_hmm_seed_initial_state` gives the initial parent finite
# cost and the other states infinite cost. A zero-rate transition cannot reach
# the required final parent, and the constrained traceback rejects its infinite
# cost (`tsls/lib/hmm.c`, initial-state seeding and final-state traceback). The
# error is the mathematically correct response to these constraints.
#
# The pins constrain **reference node identity**, not merely the allele at the
# anchor. At both actual anchor sites the sample and all three displayed parents
# have allele **0**. Matching that allele with parent 58721 is entirely possible
# and is what Full HMM does; it still does not satisfy a hard pin to parent 50854
# or 50856. Mutations can change the emitted allele, not the identity of the
# copying state.
#
# There is also a particularly clear unconstrained solution here: parent **58721**
# matches **all 258 observed alleles with zero mismatches and zero switches**.
# That is the original Full HMM path in this window. The failure does not indicate
# that TSLS has run out of matching haplotypes. All six diagnostic outcomes were
# also reproduced in the pipeline's installed TSLS runtime, independently of the
# notebook analysis runtime.

# %% [markdown]
# ### Actual haplotypes, all 258 sites
#
# These panels use the same categorical-parent path convention as Part 1, with
# **every actual 0/1 allele printed below it**. Equal-width columns are inference
# sites, not equal physical distances; both site indices and physical coordinates
# are labelled. Open circles and blue-tinted columns identify the two pinned
# flanking sites. Red frames identify reference/copied alleles differing from the
# captured sample. The complete site-by-site table is exported as
# `actual_haplotypes.csv`, including each method's parent and copied allele.
#
# The Full HMM and floored greedy copied-allele rows can agree even while their
# parent IDs differ. Conversely, accepting mismatches on a fixed parent does not
# make it possible to jump to the other endpoint parent.

# %%
if __name__ == "__main__":
    sites_per_panel = 43
    sites_per_figure = 86
    for number, start_site in enumerate(
        range(
            failure_data.request["start"], failure_data.request["end"], sites_per_figure
        ),
        start=1,
    ):
        end_site = min(start_site + sites_per_figure, failure_data.request["end"])
        nb_plot.save_cohort_figure(
            failure_plot_data,
            nb_plot.plot_conditioning_failure_alleles(
                failure_data,
                start=start_site,
                end=end_site,
                columns=sites_per_panel,
                figsize=(17, 11),
                dpi=120,
                full_color="#222222",
                floor_color="#0072B2",
                impossible_color="#B2182B",
                allele_colors=("#F6F6F6", "#F2CE72", "#D9D9D9"),
                mismatch_edge_color="#D73027",
                allele_fontsize=9,
                coordinate_fontsize=7,
            ),
            f"zero_recombination_actual_haplotypes_{number}",
            export_dpi=180,
        )

# %% [markdown]
# ### What would avoid specifying a recombination floor?
#
# If the supplied zero rates are retained, the greedy constraints must admit a
# path under that model. In this example a valid alternative is the Full HMM's
# single parent 58721 through the window, accepting whatever mismatches are
# necessary. That requires reconsidering the adjoining fixed focal choices,
# not merely changing the number of mutations inside the gap.
#
# A principled no-floor implementation would detect connected runs of zero-rate
# transitions and choose compatible copying/focal states across the whole block
# (or rematch an expanded region while permitting the conflicting focal choices
# to change). Checking only each gap independently is insufficient: fixed focal
# runs can themselves require forbidden switches inside the same zero-map block.
# This needs an explicit matching-policy change and is not implemented by this
# investigation.
#
# Simply dropping one endpoint when a gap fails would make the regional call
# succeed, but stitching it back to a different fixed parent can reintroduce a
# zero-probability switch at the seam. It is therefore not a valid silent repair
# of the unchanged whole-span model. Alternatively, supplying a map with justified
# positive rates changes the model; the `1e-20` floor is one such change, not a
# numerically neutral stabilizer. No floor is needed for an unconstrained match
# of this captured window. The incompatibility is **different hard parent pins
# inside an exactly zero-transition block**, not an inability to tolerate poor
# allelic matches.
