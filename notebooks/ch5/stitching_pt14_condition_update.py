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
# # Stitching pt14 · Revised gap conditioning
#
# The revised run and its matched control cover chr20p (28.1 Mb) in simulations
# of 500 and 5,000 diploid individuals with 1% injected genotype error and
# zero phase error, and in 3,202 diploid 1KGP individuals. All scientific
# comparisons use the saved **from-reference** trees and diagnostic tables.
# Data come from the sibling tsinfer-match-eval project. Tables and figures
# from this notebook are written under data/ch5/stitching_pt14_condition_update.

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
run_name = "conditioning_test_revised"
control_name = f"{run_name}_control"
diagnostics_dir = repository_dir.parent / "tsinfer-match-eval/data"
performance_dir = diagnostics_dir / "performance" / run_name
output_dir = repository_dir / "data/ch5/stitching_pt14_condition_update"
config_path = {
    "base_dir": repository_dir,
    "data_dir": diagnostics_dir / run_name / "diagnostics" / run_name,
    "cache_dir": output_dir / "conditioned",
    "performance_dir": performance_dir,
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
config = nb_utils.paramsearch_load_config(config_path)
control = nb_utils.paramsearch_load_config(
    dict(
        config_path,
        data_dir=diagnostics_dir / control_name / "diagnostics" / control_name,
        cache_dir=output_dir / "unconditioned",
    )
)
comparison = dict(config, cache_dir=output_dir)
plot_data = nb_utils.prepare_scaling_data(comparison)
cohort_labels = {
    "out_of_africa_n500_28_1mbp_error": "500 OOA",
    "tgp_chr20p_n3202": "3,202 1KGP",
    "out_of_africa_n5000_28_1mbp_error": "5,000 OOA",
}
method_colors = {"full_hmm": "#555555", "greedy": "#0072B2", "unconditioned": "#D55E00"}
method_labels = {
    "full_hmm": "Full HMM",
    "greedy": "Conditioned greedy",
    "unconditioned": "Unconditioned greedy",
}


# %% [markdown]
# ## Scientific settings and input checks
#
# The revised run sets $\epsilon=10^{-5}$ errors per haplotype-bp. With $S$
# quality-filtered source sites across $L=28{,}100{,}000$ bp, sample matching
# uses the constant biallelic mismatch probability $p=\epsilon L/S$. Ancestor
# matching uses its saved parameters. The map preserves zero recombination
# rates. Greedy removes entire connected focal regions touching a zero-rate
# transition and assigns those sites to HMM gaps. The control uses the same
# focal policy and removes the endpoint constraints on residual gap HMMs.
# AF cutoffs and minimum lengths are listed in the cohort table. Minimum
# lengths use a mean-density equivalent number of inference sites.
#
# The two arms share sample identities, the ancestor reference, the map,
# sample matching parameters and Full HMM paths. The benchmark allele table
# covers the timed Full HMM and conditioned outputs.
#
# %%
cohorts = nb_utils.cohort_table(config)
ipython_display.display(cohorts)
ipython_display.display(nb_utils.conditioning_review_settings(config))
ipython_display.display(nb_utils.conditioning_review_control_inputs(config, control))

# %% [markdown]
# ## What the path-likelihood plots measure
#
# A saved copying path $z=(z_1,\dots,z_m)$ selects one reference parent per
# inference site. Its **path weight** $W(z;x\mid z_1)$ is the product of
# transition and emission weights for that particular parent sequence and
# observed haplotype $x$, conditional on the first parent. Here $n$ is the
# configured haploid cohort size. With genetic distance $d_i$ in Morgans
# between sites $i-1$ and $i$, the boundary recombination probability is
# $r_i=(1-e^{-2d_i})/2$. Put $s_i=1-r_i+r_i/n$. Then
#
# $$
# W(z;x\mid z_1)=\prod_{i=2}^{m}T_i(z_{i-1},z_i)\prod_{i=1}^{m}E_i(x_i,z_i),
# \qquad
# T_i(a,b)=\begin{cases}s_i & a=b,\\ r_i/n & a\ne b.\end{cases}
# $$
#
# The emission factor $E_i$ is $1-p$ for a called allele that agrees with its
# parent, $p$ for a mismatch, and 1 for a missing call. Terminal sample mutation
# records, including reversions, count mismatches. With $C$ called sites,
# $M(z)$ mismatches and switch boundaries $J(z)=\{i:z_{i-1}\ne z_i\}$,
#
# $$
# \log W(z;x\mid z_1)=
# \underbrace{\sum_{i=2}^{m}\log s_i+C\log(1-p)}_{\text{all-stay, all-match baseline}}
# +\underbrace{\sum_{i\in J(z)}\log\frac{r_i/n}{s_i}}_{\text{transition log odds}}
# +\underbrace{M(z)\log\frac{p}{1-p}}_{\text{emission log odds}}.
# $$
#
# The total panel averages natural log weights over haplotypes. Higher values
# give more weight to the saved path. The HMM marginal likelihood of the
# observed haplotype would sum weights over all paths with their starting
# probabilities. Both greedy paths are scored across the complete span,
# including focal/gap seams, using the same weights as Full HMM. The score
# therefore also includes seam transitions for the unconditioned control.
#
# In the cost panel, Full HMM minus greedy is calculated within each haplotype
# for each log-odds term, then averaged. The baseline cancels. Positive values
# are extra cost for greedy. The transition and emission differences add to
# $\log W_{\mathrm{full}}-\log W_{\mathrm{greedy}}$. The same number of extra
# switches can have different costs depending on where they fall. A switch at
# $r_i=0$ gives zero weight and infinite cost. The table counts such paths.
# A cohort mean of $-\infty$ or an infinite transition cost has no point on the
# finite plotting scale. Event counts still include all haplotypes.
#
# The switch-increase percentage averages haplotype percentages where Full HMM
# has at least one switch. Focal coverage uses the shared masks and pt13
# population groups. Coverage error bars show one haplotype SEM.

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
    path_plot_summary = nb_utils.conditioning_path_plot_summary(
        dict(config, cache_dir=comparison["cache_dir"]), paired_paths
    )
    path_plot_summary.to_csv(
        comparison["cache_dir"] / "path_plot_summary.csv", index=False
    )
    ipython_display.display(
        path_plot_summary[["dataset", "method", "zero_weight_paths", "log_path_weight"]]
    )
    nb_plot.save_cohort_figure(
        plot_data,
        nb_plot.plot_conditioning_review_paths(
            plot_data,
            path_plot_summary,
            cohort_labels=cohort_labels,
            figsize=(13, 8),
            dpi=120,
            method_colors=method_colors,
            method_labels=method_labels,
            sample_tick_rotation=15,
            show_total_switches=True,
        ),
        "conditioned_path_costs",
        export_dpi=180,
    )
    nb_plot.save_cohort_figure(
        plot_data,
        nb_plot.plot_scaling(
            plot_data,
            path_plot_summary,
            metrics={"switch_increase_percentage": "Extra switches vs Full HMM (%)"},
            title="Extra switches versus Full HMM",
            cohort_labels=cohort_labels,
            method_colors=method_colors,
            method_labels=method_labels,
            panel_width=6,
            panel_height=4,
        ),
        "extra_switches_scaling",
        export_dpi=180,
    )
    coverage_config = dict(
        config,
        cache_dir=comparison["cache_dir"],
        population_groups={
            "African": ["YRI", "AFR"],
            "Non-African": ["CEU", "CHB", "EUR", "EAS", "SAS", "AMR"],
        },
    )
    population_coverage = nb_utils.population_coverage(coverage_config)
    nb_plot.save_cohort_figure(
        plot_data,
        nb_plot.plot_coverage(
            plot_data,
            population_coverage,
            cohort_labels=cohort_labels,
            population_colors={"African": "#009E73", "Non-African": "#CC79A7"},
            panel_width=6,
            panel_height=4,
        ),
        "focal_population_coverage",
        export_dpi=180,
    )

# %% [markdown]
# In all three cohorts, conditioning improves the mean saved-path log weight
# relative to unconditioned greedy. It cuts seam switches by about 75, 70
# and 18 per haplotype for 500 OOA, 5,000 OOA and 1KGP, respectively. Switches
# within gaps increase, but total switches fall. Both greedy methods have
# fewer terminal sample mutations than Full HMM, so their emission terms
# offset part of their higher transition cost. The transition cost dominates
# in all three cohorts. Every saved path has finite weight.

# %% [markdown]
# ## Tree counts and ARG agreement
#
# Tree counts come from the raw matched tree sequences on their saved
# coordinates, including reference ancestry, as in pt6. For each greedy
# method, the percentage change is $100(N_{\mathrm{method}}/N_{\mathrm{Full\ HMM}}-1)$.
# A negative value means fewer trees than Full HMM.

# %%
tree_counts = nb_utils.conditioning_tree_counts(config, control)
tree_counts.to_csv(output_dir / "tree_counts.csv", index=False)
ipython_display.display(tree_counts.round(3))
nb_plot.save_cohort_figure(
    plot_data,
    nb_plot.plot_scaling(
        plot_data,
        tree_counts,
        metrics={"trees_change_percentage": "Change in tree count (%)"},
        title="Tree count change relative to Full HMM",
        reference=0,
        zoom=True,
        cohort_labels=cohort_labels,
        method_colors=method_colors,
        method_labels=method_labels,
        panel_width=6,
        panel_height=4,
    ),
    "tree_count_change",
    export_dpi=180,
)

# %% [markdown]
# ### ARF and TPR
#
# Following [pt6](stitching_pt6_min_sizes.ipynb), pass the focal output as the
# first argument to tscompare's haplotype ARF and the benchmark as the second.
# ARF is the fraction of the focal output's node span unsupported by its
# best matches in the benchmark. TPR is the fraction of benchmark node span
# represented by the focal output. Lower ARF and higher TPR indicate greater
# agreement with the chosen benchmark.
#
# Align biological haplotypes by sample ID and ploidy index, translating the
# cohort's saved observed interval to common coordinates and simplifying
# with unary ancestors retained. Simulation truth uses its VCF sample
# identities. Both greedy methods are scored against Full HMM in all three
# cohorts. All three methods are scored against the true ARG for simulations.

# %%
arg_metrics = nb_utils.conditioning_arg_metrics(comparison, control)
arg_metrics.to_csv(output_dir / "arg_metrics.csv", index=False)
ipython_display.display(arg_metrics.round(5))

# %%
for reference_method, label in [("full_hmm", "Full HMM"), ("true_arg", "simulation truth")]:
    scores = arg_metrics.loc[arg_metrics.reference_method.eq(reference_method)]
    nb_plot.save_cohort_figure(
        plot_data,
        nb_plot.plot_scaling(
            plot_data,
            scores,
            metrics={"arf": "ARF", "tpr": "TPR"},
            title=f"ARG agreement with {label}",
            include_full=reference_method == "true_arg",
            datasets=scores.dataset.unique(),
            zoom=True,
            cohort_labels=cohort_labels,
            method_colors=method_colors,
            method_labels=method_labels,
            sample_tick_alignments={},
            sample_tick_rotation=15,
            panel_width=5.5,
            panel_height=4,
        ),
        f"arg_metrics_vs_{reference_method}",
        export_dpi=180,
    )


# %% [markdown]
# ## Direct ultimate-root to sample edges
#
# Count raw edges from the ultimate ancestor (the child of the vestigial root)
# to sample haplotypes over each cohort's common observed extent. Each edge
# counts once. Its span is clipped to the analysis interval. Counts and summed
# spans are divided by the number of haplotypes. The log-scale bars show totals.
#
# Using the shared focal masks, an edge is classified as a focal/gap seam if it
# overlaps a gap and touches an internal boundary. The other categories are gap
# interior and focal only. The signed panels subtract unconditioned from
# conditioned greedy; diamonds show the net difference across categories.
# The histograms use common log-spaced bins and show edges per haplotype per bin.

# %%
if __name__ == "__main__":
    diagnostic_dir = comparison["cache_dir"] / "root_edges_and_recurrence"
    diagnostic_dir.mkdir(exist_ok=True)
    root_config = dict(config, cache_dir=diagnostic_dir / "conditioned")
    root_control = dict(control, cache_dir=diagnostic_dir / "unconditioned")
    diagnostic_config = dict(comparison, cache_dir=diagnostic_dir)
    diagnostic_plots = nb_utils.prepare_scaling_data(diagnostic_config)
    root_plot_data = nb_utils.prepare_scaling_data(root_config)

    root_edges = nb_utils.root_edge_summary(root_config)
    control_edges = nb_utils.root_edge_summary(root_control)
    control_edges = control_edges.loc[control_edges.method.eq("greedy")].copy()
    control_edges["method"] = "unconditioned"
    root_edges = pd.concat([root_edges, control_edges], ignore_index=True)
    root_edges.to_csv(diagnostic_dir / "root_edge_summary.csv", index=False)
    root_edge_categories = {
        "seam": {"label": "Touches a focal/gap seam", "color": "#7B61A8"},
        "gap_interior": {"label": "Gap interior", "color": "#E69F00"},
        "focal": {"label": "Focal only", "color": "#009E73"},
    }
    nb_plot.save_cohort_figure(
        diagnostic_plots,
        nb_plot.plot_root_totals(
            root_plot_data,
            root_edges,
            cohort_labels=cohort_labels,
            method_colors=method_colors,
            method_labels=method_labels,
            stacked=False,
            grouped_bar_width=0.24,
            log_y=True,
            panel_width=5.5,
            panel_height=4.0,
        ),
        "root_edges_counts_spans",
        export_dpi=180,
    )
    nb_plot.save_cohort_figure(
        diagnostic_plots,
        nb_plot.plot_root_edge_differences(
            root_plot_data,
            root_edges,
            comparison_methods=("greedy", "unconditioned"),
            cohort_labels=cohort_labels,
            method_colors=method_colors,
            method_labels=method_labels,
            root_edge_categories=root_edge_categories,
            panel_width=5.5,
            panel_height=4.0,
        ),
        "root_edges_conditioning_difference",
        export_dpi=180,
    )
    nb_plot.save_cohort_figure(
        diagnostic_plots,
        nb_plot.plot_root_edge_spans(
            root_plot_data,
            cohort_labels=cohort_labels,
            method_colors=method_colors,
            method_labels=method_labels,
            root_edge_categories=root_edge_categories,
            additional_method_cache_dirs={"unconditioned": root_control["cache_dir"]},
            root_edge_histogram_bins=55,
            panel_width=4.4,
            panel_height=4.0,
        ),
        "root_edges_span_histograms",
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
# ## Precision and recall of inferred recurrent sites
#
# Following [Chapter 4's epsilon analysis](../ch4/epsilon_testing.ipynb), we
# evaluate the same inference sites for all three methods. At each simulated
# site, we first pool the two sources of truth by taking their **union**. A
# site is truth-positive if it has at least two mutation records in the
# simulated tree sequence or at least one injected genotype error, using the
# source store's per-site error count. A site with both counts once.
#
# A prediction is positive if the raw inferred tree sequence has at least
# two mutation records at that site, including back mutations. We match by
# position regardless of which samples carry the mutations or errors. An
# inferred site with multiple affected samples still contributes one site.
# Precision is
# the fraction of predicted positives in the pooled truth set. Recall is the
# fraction of the pooled truth set flagged by inferred recurrence. These are
# site-level scores for the two simulations. 1KGP has no truth labels.

# %%
if __name__ == "__main__":
    recurrence_scores = nb_utils.conditioning_recurrence_scores(config, control)
    recurrence_scores.to_csv(
        diagnostic_dir / "recurrent_site_precision_recall.csv", index=False
    )
    nb_plot.save_cohort_figure(
        diagnostic_plots,
        nb_plot.plot_scaling(
            diagnostic_plots,
            recurrence_scores,
            metrics={"precision": "Precision", "recall": "Recall"},
            title="Inferred recurrence against true recurrence or injected error",
            include_full=True,
            datasets=recurrence_scores.dataset.unique(),
            y_limits=(0, 1),
            cohort_labels=cohort_labels,
            method_colors=method_colors,
            method_labels=method_labels,
            sample_tick_alignments={},
            panel_width=5.5,
            panel_height=4.0,
        ),
        "recurrent_site_precision_recall",
        export_dpi=180,
    )

# %% [markdown]
# ## Performance and bottlenecks
#
# One recorded trial per cohort, timing scope and method at 186 workers. The
# timed suite contains Full HMM and conditioned greedy. The unconditioned
# run produced diagnostic trees without a comparable timed trial.
# From-reference includes input preparation, matching and finalisation from
# an existing ancestor reference. Resident-match starts with prepared inputs
# and times matching/drain. Ancestor inference and subsequent quality checks
# sit outside these wall times. Peak USS is sampled process-tree private
# memory within the timed window. CPU seconds divided by wall seconds gives
# average busy cores within that window.
#
# Packing and drain are sequential from-reference stages. Result wait,
# first-result latency and output assembly are diagnostics nested within
# drain. Worker CPU is summed across threads and reported separately in
# core-seconds. The timer named gap_native_find_path covers whole-span native
# HMM search for Full HMM and native gap searches for greedy.

# %%
benchmark = nb_utils.conditioning_benchmark(performance_dir, output_dir)
runs = benchmark.runs
assert runs.actual_workers.eq(186).all()
assert not runs.duplicated(["dataset", "mode", "method"]).any()
ipython_display.display(benchmark.quality)
ipython_display.display(
    runs[[
        "dataset", "mode", "method", "wall_s", "speedup", "haplotypes_per_s",
        "effective_cores", "window_peak_private_gib",
    ]].round(3)
)
nb_plot.save_cohort_figure(
    plot_data,
    nb_plot.plot_conditioning_review_performance(
        runs, cohort_labels=cohort_labels, figsize=(12, 7), dpi=120,
        full_color=method_colors["full_hmm"], greedy_color=method_colors["greedy"],
    ),
    "performance",
    export_dpi=180,
)

# %%
bottlenecks = nb_utils.conditioning_bottlenecks(benchmark)
bottlenecks.to_csv(output_dir / "bottlenecks.csv", index=False)
ipython_display.display(bottlenecks.drop(columns="run_id").round(3))
nb_plot.save_cohort_figure(
    plot_data,
    nb_plot.plot_conditioning_wall_breakdown(
        bottlenecks,
        cohort_labels=cohort_labels,
        stage_colors={
            "Reference packing": "#E69F00", "Matching / drain": "#0072B2",
            "Other": "#B8B8B8",
        },
        figsize=(12, 4), dpi=120,
    ),
    "performance_wall_breakdown",
    export_dpi=180,
)

# %% [markdown]
# **500 OOA and 1KGP.** Greedy spends 81 s (65%) and 530 s (88%) of
# from-reference time waiting for reference packing to finish, respectively.
# Once inputs are resident, greedy takes 19 s against 22 s for 500 OOA and
# 28 s against 96 s for 1KGP. From reference, greedy takes 3.7 and 4.7 times
# as long as Full HMM. Packing accounts for most of the reversal.
#
# **5,000 OOA.** Both methods spend about 6,140 s in from-reference drain.
# Native search consumes about 1.12 million thread CPU seconds for each
# method, or roughly 182 cores averaged across drain. Greedy takes about
# 735 s more for packing and has effectively the same resident time
# (6,109 s versus 6,113 s). Within this trial, the native HMM search is
# the dominant cost even after preparation is excluded.
#
# Packing completion measures time on the critical path. Worker service time
# and result wait can include synchronisation or overlap with other work, so
# they cannot identify storage bandwidth or lock contention on their own.
# Per-haplotype service distributions below expose variation hidden by the
# mean. The Spearman columns associate gap sites and gap calls with worker
# CPU within each cohort and timer scope. They do not measure a causal effect.

# %%
haplotype_timing_summary = nb_utils.conditioning_haplotype_timings(benchmark.timings)
haplotype_timing_summary.to_csv(output_dir / "haplotype_timing_summary.csv", index=False)
ipython_display.display(haplotype_timing_summary.round(3))
nb_plot.save_cohort_figure(
    plot_data,
    nb_plot.plot_conditioning_haplotype_cpu(
        benchmark.timings, cohort_labels=cohort_labels, color=method_colors["greedy"],
        figsize=(12, 4), dpi=120, point_size=8, alpha=0.2,
    ),
    "resident_haplotype_cpu",
    export_dpi=180,
)

# %% [markdown]
# ### Why the timing changed since perf5
#
# The table uses the archived from-reference trials. Packing and drain are
# wall seconds. Native search is summed thread CPU in core-seconds. Each
# arrow goes from perf5 (96 workers) to the revised trial (186 workers).
#
# | Cohort | Greedy total wall, s | Greedy packing wall, s | Greedy drain wall, s | Greedy native search, core-s | Full HMM total wall, s |
# | --- | ---: | ---: | ---: | ---: | ---: |
# | 500 OOA | 52 → 124 | 20 → 81 | 12 → 26 | 171 → 2,998 | 23 → 33 |
# | 3,202 1KGP | 115 → 601 | 22 → 530 | 61 → 45 | 3,709 → 4,665 | 195 → 128 |
# | 5,000 OOA | 284 → 6,918 | 31 → 735 | 205 → 6,139 | 14,408 → 1,118,491 | 599 → 6,157 |
#
# **Packing explains the 1KGP reversal.** The revised packing workers
# decode required Zarr tiles and copy ancestor spans into the shared
# in-memory reference. Greedy waits for every packing job before matching.
# In 1KGP, packing adds 509 s while drain drops 16 s. The 530 s at the
# packing barrier accompany only 14 summed worker CPU seconds in the packing
# timer. The recorded wall/CPU split points to waiting on the preparation
# path. These trials cannot separate storage, decoding, scheduling and
# synchronisation, or isolate the effect of switching to resident storage.
# The 500 OOA run also spends most of its extra wall time in packing.
# The 14 November TSLS 1KGP trial took 29.2 s with resident inputs, close to
# 28.3 s in pt14. That comparison also puts the main increase outside matching.
#
# **Native search explains the large simulation.** Greedy's gap-site count
# grows by 12% (1.14 to 1.27 billion), while native CPU grows 78×. Full HMM
# also takes about 10× longer end to end, so this slowdown extends to
# whole-span matching. Perf5 used the earlier tsinfer matcher, constant
# $r=0.01$, and $p=10^{-20}$. The revised trials use TSLS, the genetic map
# including true zeros, calibrated $p$ (about $4.26\times10^{-4}$ for
# 5,000 OOA), fixed haploid sample size and conditioned gap searches.
# Focal AF cutoffs, reference construction and worker counts also differ.
#
# The revised focal filter sends whole components touching a zero-rate
# transition to the HMM. At zero recombination, a state can only inherit its
# own previous score. With no switch from the previous best state to cap the
# cost of staying, distinct likelihoods can persist. With likelihood
# flooring and pruning disabled, TSLS can retain many distinct finite
# likelihoods and update them at every site. Short trees can also prevent
# the default indexed sparse path from activating, leaving full likelihood
# list scans. Two archived 1,024-site probes on the larger 50,000-individual
# simulation averaged 4,215 likelihood values and 0.137 CPU s in a zero-rate
# window versus 394 values and 0.014 CPU s in a positive-rate window.
# These windows differ in topology as well as rate. They illustrate
# workload sensitivity and give no isolated estimate of the zero-rate cost.
#
# An archived same-input **500 OOA Full HMM** control used the revised
# no-floor parameters at 96 workers. The earlier archived TSLS implementation
# took 20.14 s resident versus 20.12 s for the revised implementation, with
# identical copying paths and mutations. This control covers the recent
# TSLS revisions and the 500 OOA workload. A comparable 5,000 OOA replay
# holding inputs and probabilities fixed would be needed to quantify the
# native implementation's contribution to the perf5-to-pt14 gap.
#
# Sources on BMRC are runs.csv, stages.csv and manifest.json under the
# sibling tsinfer-match-eval/data/performance/{perf5,conditioning_test_revised}
# directories, the saved native matching TOMLs, and the archived
# full_hmm_baseline_20261117_36rVlb/comparison.json and
# further_perf_20261118/native.json in the user scratch directory.

# %% [markdown]
# **Per-haplotype workload.** In resident 1KGP, gap sites closely track worker
# CPU (Spearman ρ = 0.98), with a median of 0.75 s and a 95th-percentile
# elapsed service time of 1.71 s. In 5,000 OOA, per-haplotype elapsed
# times are much longer (118 s at the median and 175 s at the 95th
# percentile), and gap-site counts explain little of the rank variation
# in CPU time (ρ = −0.04). Gap-call counts show a similar contrast
# (ρ = 0.88 in 1KGP and −0.04 in 5,000 OOA). Counting gap sites or calls
# alone thus gives a poor workload proxy for that simulation.

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

