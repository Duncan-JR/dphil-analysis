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
# # hpc_perf1: performance and diagnostics across sample sizes
#
# Analyse the archived effective config, timed trials and diagnostics for
# **Simulated cohorts**. AF cutoffs are 0.01 and 0.1; the requested minimum is
# 150 kb, converted separately to a genome-wide mean inference-site count for
# each cohort. The figures label AF with the corresponding nominal AC rounded
# from 2n × AF. Multi-cohort legends list ACs in increasing sample-size order.
#
# Each setting has **one trial**, with 32 matching workers. Lines connect
# observations; they do not estimate replicate uncertainty. Full HMM supplies
# one shared baseline per cohort. All comparisons begin at prepared ancestor
# references and include loading, setup, matching and output. Resident timings
# and stage breakdowns were not collected. No ARG metrics are calculated.
#
# Outputs, dates, matched mutation tables and figures are cached beside the
# archive contents under `for_plots/`. This notebook uses only archived data.

# %%
import logging
import pathlib

import ch5_perf_scaling
import ch5_perf_scaling_plots
import ch5_root_edges
import ch5_gap_parents
import IPython.display as ipython_display
import pandas as pd

logger = logging.getLogger(__name__)

# %%
notebook_dir = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path.cwd()
repository_dir = notebook_dir
if not (repository_dir / 'experiments').is_dir():
    repository_dir = notebook_dir.parent.parent
config_path = repository_dir / 'experiments/ch5/stitching_pt8_hpc_perf1_config.yaml'
config = ch5_perf_scaling.load_config(config_path)
plot = ch5_perf_scaling_plots.ScalingPlots(config)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# %% [markdown]
# ## Archived run checks
#
# Sample counts come from saved cohort identities. Saved quality checks verify
# called alleles and sample/reference metadata; they do not establish genealogy
# accuracy. The resolved site-count minimum is included below. Diagnostic
# coverage uses the observed inference-site extent, excluding chromosome flanks.

# %%
if __name__ == '__main__':
    cohorts = ch5_perf_scaling.cohort_table(config)
    performance = ch5_perf_scaling.performance_runs(config)
    summary = ch5_perf_scaling.diagnostic_summary(config)
    ipython_display.display(cohorts)
    columns = ['num_samples', 'method', 'af_cutoff', 'wall_s', 'wall_ms_per_haplotype',
               'speedup', 'cpu_s', 'effective_cores', 'trial_peak_private_gib']
    ipython_display.display(performance[columns].round(3))

# %% [markdown]
# ## Performance against sample size
#
# Wall time is the measured end-to-end window. Throughput divides that wall
# time by the number of matched haplotypes at fixed worker count. Private memory
# is sampled **sum of process-tree USS**, not RSS; shared mappings are not
# multiplied by the number of workers. The whole-trial peak includes setup and
# output. Average cores = (user + system CPU seconds) / measured wall seconds.
# Speedup = Full HMM wall time / method wall time within the same cohort.
# The sample-size axis is logarithmic; every measured size is labelled.

# %%
if __name__ == '__main__':
    metrics = {'wall_s': 'End-to-end wall time (s)',
               'wall_ms_per_haplotype': 'Wall time per haplotype (ms)'}
    plot.save(plot.scaling(performance, metrics, 'Performance vs sample size',
                           include_full=True, log_y=True), 'performance_wall_throughput')

# %%
if __name__ == '__main__':
    metrics = {'trial_peak_private_gib': 'Whole-trial peak private memory (GiB)',
               'effective_cores': 'Average CPU cores used'}
    plot.save(plot.scaling(performance, metrics, 'Memory and CPU vs sample size',
                           include_full=True), 'performance_memory_cpu')

# %%
if __name__ == '__main__':
    metrics = {'speedup': 'Wall-time speedup over Full HMM', 'cpu_s': 'User + system CPU time (s)'}
    plot.save(plot.scaling(performance, metrics, 'Speedup and total CPU cost',
                           include_full=True), 'performance_speedup_cpu_cost')

# %% [markdown]
# ## Recorded workload
#
# Focal site coverage uses the timed runner's summed focal/domain site counts,
# rather than physical spans. Gap calls count separate HMM invocations. These
# are workload measures, not timing decompositions.

# %%
if __name__ == '__main__':
    metrics = {'focal_site_coverage_percentage': 'Retained focal inference sites (%)',
               'gap_calls_per_haplotype': 'Gap HMM calls per haplotype'}
    plot.save(plot.scaling(performance, metrics, 'Workload vs sample size'), 'workload_scaling')

# %% [markdown]
# ## Extra switches and parent agreement
#
# Each diagnostic mean weights haplotypes equally. Error bars are one
# haplotype SEM within a cohort; paired haplotypes are dependent and these are
# not simulation or benchmark replicate errors. Percentage switch increases
# are calculated per haplotype before averaging. Parent agreement denominators
# are evaluated inference sites in focal regions or gaps. Its y axes cover the
# displayed means and SEMs rather than starting at zero.

# %%
if __name__ == '__main__':
    metrics = {'num_extra_switches_mean': 'Extra switches per haplotype',
               'switch_increase_percentage_mean': 'Extra switches vs Full HMM (%)'}
    plot.save(plot.scaling(summary, metrics, 'Extra switches vs sample size', sem=True), 'extra_switches_scaling')
    plot.save(plot.extra_switch_distributions(), 'extra_switch_distributions')

# %%
if __name__ == '__main__':
    metrics = {'focal_agreement_percentage_mean': 'Parent agreement in focal regions (%)',
               'gap_agreement_percentage_mean': 'Parent agreement in gaps (%)'}
    plot.save(plot.scaling(summary, metrics, 'Parent agreement vs sample size', sem=True, zoom=True),
              'parent_agreement_scaling')

# %% [markdown]
# ## Physical focal coverage and region lengths
#
# Population groups follow the analysis config. Curves show physical coverage
# over the archived analysis extent, with one haplotype SEM. Histograms show
# **retained** connected focal regions at 150 kb; unfiltered regions were not
# archived. `num_bins` is directly editable below. Dashed site thresholds are
# the actual filters. The 150-kb marker in the physical plot is a mean-density
# equivalent, not a strict physical cutoff for each region.
#
# Reflected Scott KDEs use every interval, including tails outside the displayed
# pooled 99th percentile. Legends report the omitted tail fraction. Site and
# physical lengths are cached as lossless frequency tables. Gap plots use
# density curves and dashed median markers, with no histogram or coverage plot.

# %%
if __name__ == '__main__':
    population_coverage = ch5_perf_scaling.population_coverage(config)
    lengths = ch5_perf_scaling.length_counts(config)
    plot.save(plot.coverage(population_coverage), 'focal_population_coverage')

# %%
if __name__ == '__main__':
    num_bins = 120
    plot.save(plot.focal_lengths(lengths, unit='sites', num_bins=num_bins), 'focal_lengths_sites')
    plot.save(plot.focal_lengths(lengths, unit='kb', num_bins=num_bins), 'focal_lengths_kb')
    plot.save(plot.gap_lengths(lengths), 'gap_lengths_kb')

# %% [markdown]
# ### One-site gaps
#
# Bars count gap regions containing exactly one inference site, pooled across
# all haplotypes. Labels show their percentage of **all gap regions** for that
# dataset and AF cutoff, including the full length-distribution tails. Each
# sample size has a column; each AF cutoff has a bar at the fixed kb minimum.

# %%
if __name__ == '__main__':
    plot.save(plot.one_site_gaps(lengths), 'one_site_gaps')

# %% [markdown]
# ## Switch composition
#
# Switches partition into within-focal, within-gap and seam categories. For
# each AF and cohort, Full HMM is partitioned with that same focal mask; its
# total does not change with the mask. All columns display their y-axis ticks.

# %%
if __name__ == '__main__':
    plot.save(plot.switch_composition(summary), 'switch_composition')

# %% [markdown]
# ## Which ancestors supply the gap copying paths?
#
# This diagnoses Gil's 15 October proposal to restrict gap HMM matching to an
# older reference. It examines the existing paths, without rerunning matching.
# For each AF cutoff, Full HMM and stitched paths use the **same stitched gap
# mask**. All categories are specific to the sample haplotype:
#
# - **Older:** reference node time is strictly greater than the AF cutoff.
# - **Recent, not focal:** node time is at or below the cutoff, but the sample
#   does not carry that ancestor's selected focal seed.
# - **Focal candidate:** node time is at or below the cutoff and the sample
#   carries its selected seed, even if no retained greedy region used it.
#   The seed can lie outside the particular gap being measured.
#
# These are tsinfer's frequency-based ancestor times, not dates in generations.
# The two artificial old roots belong to the older category. Every archived
# reference has one mutation per inference site; these identify the original
# focal sites. Recover the configured leftmost seed per ancestor and check that
# the recent-node counts match the benchmark's eligible-ancestor counts.
#
# Carrier status is reconstructed from the saved **Full HMM** haplotypes and
# held fixed for both methods, including terminal sample-mutation overrides.
# These are the saved alleles; original missing-call flags are not archived.
# No genotype stores are opened. Blue would be retained by an older-only HMM;
# orange and green show existing matches that such a restriction would exclude.
# This does not predict which replacement parents the restricted HMM would use.
#
# The first plot partitions gap base pairs. The second counts distinct parents
# once per haplotype across all its gaps. Percentages are calculated within each
# haplotype, then averaged; haplotypes with an empty denominator are omitted.
# Site-weighted and copying-segment-weighted percentages are also cached.

# %%
if __name__ == '__main__':
    gap_parents = ch5_gap_parents.gap_parent_composition(config)
    columns = ['num_samples', 'method', 'af_cutoff', 'category',
               'span_bp_percentage_mean', 'distinct_parents_percentage_mean']
    ipython_display.display(gap_parents[columns].round(2))
    for weight in config['plot']['gap_parent_weightings']:
        plot.save(plot.gap_parents(gap_parents, weight), f'gap_parents_{weight}')

# %% [markdown]
# ## Direct ultimate-root to sample edges
#
# These are raw copying edges from the ultimate ancestor (the child of the
# vestigial root) directly to sample haplotypes, before root splitting or dating.
# Count each edge once if it overlaps the common archived analysis extent;
# its span is the number of observed base pairs, excluding chromosome flanks.
# Counts and summed spans are divided by the cohort's haplotype count.
#
# Each AF uses its own focal mask for **both** Full HMM and stitched paths.
# An edge is **at a seam** if it overlaps a gap and contains an internal
# focal/gap boundary, including either edge endpoint. An edge in a gap with
# no such boundary is **gap interior**. Remaining edges are **focal only**.
# Chromosome/domain ends are not seams. Edges spanning multiple regions are
# counted once with seam classification taking precedence.
#
# The difference panels subtract Full HMM from stitched within each category;
# these are signed aggregate differences, not a pairing of individual edges.
# Positive and negative contributions are stacked separately, with a diamond
# marking the net difference. Moving an edge between categories can therefore
# create opposite contributions even without changing the total edge count.
#
# Histograms use common logarithmic bins over the **complete** observed span
# range. Stitched bars are stacked by location; Full HMM is the outline. Both
# axes are logarithmic, and counts are edges per haplotype per bin.

# %%
if __name__ == '__main__':
    root_edges = ch5_root_edges.root_edge_summary(config)
    columns = ['num_samples', 'af_cutoff', 'method', 'category', 'edge_count',
               'span_bp', 'edge_count_per_haplotype', 'span_kb_per_haplotype']
    ipython_display.display(root_edges[columns].round(3))
    plot.save(ch5_root_edges.totals(plot, root_edges, 'edge_count_per_haplotype',
                                  'Root edges per haplotype'), 'root_edges_counts')
    plot.save(ch5_root_edges.totals(plot, root_edges, 'span_kb_per_haplotype',
                                  'Root-copying span (kb per haplotype)'), 'root_edges_spans')
    plot.save(ch5_root_edges.additional(plot, root_edges), 'root_edges_additional')
    plot.save(ch5_root_edges.span_histograms(plot), 'root_edges_span_histograms')

# %% [markdown]
# ## Saved tree counts and mutation dating
#
# The following step checks that diagnostic and timed outputs have identical
# copying paths and mutations, then records raw edge/tree counts. It computes
# no ARF, TPR or other ARG metrics. Percentage changes use the raw Full HMM
# counts, including retained reference ancestry.
#
# Dating follows pt6/pt7: post-process saved trees, restrict to the common
# archived extent, preprocess disjoint nodes and run variational-gamma tsdate.
# **No variants are restored and no genotype stores are opened.** Mutation
# rates are configured as 2.35 × 10⁻⁸ for the simulations and 1.29 × 10⁻⁸ for
# 1kgp, per bp per generation. Independent trees run in worker processes and
# their dated outputs are cached.

# %%
if __name__ == '__main__':
    dated = ch5_perf_scaling.date_all(config)
    ipython_display.display(dated[['dataset', 'method', 'af_cutoff', 'raw_sites', 'dated_sites',
                                  'num_edges', 'num_trees', 'benchmark_paths_identical']])
    metrics = {'edges_change_percentage': 'Change in raw edges vs Full HMM (%)',
               'trees_change_percentage': 'Change in raw trees vs Full HMM (%)'}
    plot.save(plot.scaling(dated, metrics, 'Raw tree counts', zoom=True), 'raw_tree_counts')

# %% [markdown]
# ## Mutation ages: association and calibration
#
# Posterior **mutation means** (`mn`) are compared with simulated event times
# where truth exists, and with Full HMM posterior means. 1kgp has no known true
# ages; Full HMM is a comparison method, not truth.
#
# For each cohort, use one common variant intersection across Full HMM, both
# stitched runs and, for simulations, truth. Match position and ancestral/derived
# states; exclude recurrent/back-mutation sites and nonfinite ages. All methods,
# reference comparisons and frequency bins use that same cohort intersection.
# The cached tables include its retention relative to each candidate set.
#
# Pearson r uses untransformed ages in generations. Median absolute error
# supplements association with a calibration measure. DAF is fixed from Full
# HMM for every method; it is distinct from the AF cutoff used to select focal
# ancestors. Bin boundaries enter the upper bin. Empty or constant-age bins
# have undefined correlation and remain blank in the plots.

# %%
if __name__ == '__main__':
    ages = ch5_perf_scaling.mutation_correlations(config, dated)
    checks = ages.groupby(['dataset', 'frequency_bin']).matched_mutations.agg(['min', 'max'])
    assert checks['min'].eq(checks['max']).all()
    ipython_display.display(checks['min'].rename('matched_mutations').unstack('frequency_bin'))
    retention_paths = sorted((config['cache_dir'] / 'age_pairs').glob('*_retention.csv'))
    retention = pd.concat([pd.read_csv(path) for path in retention_paths], ignore_index=True)
    ipython_display.display(retention.round(2))
    ipython_display.display(ages.loc[ages.frequency_bin.eq('all')].round(4))
    plot.save(plot.age_correlations(ages), 'mutation_age_correlations')

# %%
if __name__ == '__main__':
    for index, label in enumerate(config['frequency_bins']['labels']):
        plot.save(plot.age_correlations(ages, label), f'mutation_age_daf_{index}')

# %% [markdown]
# ## Paired mutation ages
#
# Hexagons show counts with one shared logarithmic colour scale. The dashed
# line is equality. Display bounds omit ages below one generation and use a
# shared upper quantile; correlations and errors above use all matched ages,
# including display tails. The annotated N is the complete common intersection.

# %%
if __name__ == '__main__':
    if cohorts.has_truth.all():
        plot.save(plot.age_density(ages, 'truth'), 'mutation_ages_vs_truth')
    plot.save(plot.age_density(ages, 'full_hmm'), 'mutation_ages_vs_full_hmm')
