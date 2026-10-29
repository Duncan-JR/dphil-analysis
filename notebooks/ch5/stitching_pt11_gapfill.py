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
# # hpc_perf3: gap filling, performance and diagnostics
#
# Archived Out-of-Africa simulations on chr20p (28.1 Mb), with 1% genotype
# error: **50, 500 and 5,000 diploid samples**. Each cohort compares Full HMM
# with greedy stitching at AF ≤ 0.1. The minimum retained focal length is the
# mean-density equivalent of **25, 50 and 100 kb**, respectively. These are
# different cohorts, simulation seeds and length filters, so sample-size trends
# do not isolate the effect of increasing sample size.
#
# Each setting has **one trial**, with 94 matching workers. Lines connect
# observations; they do not estimate replicate uncertainty. Timing starts from
# prepared ancestor references and includes input loading, setup, matching and
# output; ancestor inference and allele validation are outside this window.
#
# All inputs come from `data/ch5/hpc_perf3_diagnostics/` (extract the adjacent
# `.tar.gz` there if needed). Tables, dated trees and PNG/PDF figures are cached
# under its `for_plots/` directory. No genotype stores or ARG metrics are used.

# %%
import logging
import pathlib

import ch5_gapfill
import ch5_gapfill_plots
import ch5_perf_scaling
import ch5_root_edges
import IPython.display as ipython_display
import pandas as pd

logger = logging.getLogger(__name__)

# %%
notebook_dir = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path.cwd()
repository_dir = notebook_dir
if not (repository_dir / 'experiments').is_dir():
    repository_dir = notebook_dir.parent.parent
config_path = repository_dir / 'experiments/ch5/stitching_pt11_gapfill_config.yaml'
config = ch5_perf_scaling.load_config(config_path)
plot = ch5_gapfill_plots.GapfillPlots(config)
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
               'speedup', 'effective_cores', 'trial_peak_private_gib']
    ipython_display.display(performance[columns].round(3))

# %% [markdown]
# ## Performance against sample size
#
# Wall time is the measured end-to-end window. Throughput divides that wall
# time by the number of matched haplotypes at fixed worker count. Private memory
# is sampled **sum of process-tree USS**, not RSS; shared mappings are not
# multiplied by the number of workers. Whole-trial memory also covers work
# outside the wall timer, including validation; timed-window memory covers the
# same window as wall time. Average cores = (user + system CPU seconds) / wall seconds.
# Speedup = Full HMM wall time / method wall time within the same cohort.
# The dotted parallelism line marks the 94 matching workers. This is achieved
# CPU concurrency, not speedup relative to a single-worker run. Setup and I/O
# inside the measured window are included, as in the chapter 3 CPU/wall metric.
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
               'window_peak_private_gib': 'Timed-window peak private memory (GiB)'}
    plot.save(plot.scaling(performance, metrics, 'Private memory vs sample size',
                           include_full=True), 'performance_memory')

# %%
if __name__ == '__main__':
    metrics = {'speedup': 'Wall-time speedup over Full HMM',
               'effective_cores': 'Average active CPU cores'}
    figure = plot.scaling(performance, metrics, 'Speedup and achieved parallelism', include_full=True)
    workers = config['experiment']['performance']['worker_counts'][0]
    figure.axes[0].axhline(1, color='gray', linestyle=':', linewidth=1)
    figure.axes[1].axhline(workers, color='gray', linestyle=':', linewidth=1)
    figure.axes[1].set_ylim(top=workers * (1 + config['plot']['range_margin']))
    figure.axes[1].set_title(f'{workers} matching workers (dotted line)')
    plot.save(figure, 'performance_speedup_parallelism')

# %% [markdown]
# ## Where greedy matching spends its time
#
# The first stack partitions end-to-end wall time into sequential coordinator
# and process phases. Matching/drain includes sample loading, result waiting
# and assembly: its nested timers are not stacked again. Worker packing and
# setup can overlap with coordinator phases; only the coordinator's elapsed
# windows appear in this plot. The residual makes each stack equal the measured
# trial total. In particular, startup and shutdown dominate the smallest run.

# %%
if __name__ == '__main__':
    stages = ch5_gapfill.stage_breakdown(config, performance)
    plot.save(plot.wall_stages(stages), 'greedy_wall_stages')
    latency_stages = ['sample_loading_remapping', 'first_sample_slab_latency',
                      'first_result_latency', 'coordinator_matching_drain']
    latency = stages.recorded.loc[stages.recorded.stage.isin(latency_stages)]
    ipython_display.display(latency[['num_samples', 'stage', 'wall_s', 'service_s']].round(3))

# %% [markdown]
# The next plot separates worker elapsed time from worker thread CPU time,
# dividing each summed stage by the number of matched haplotypes. Gap HMM
# `find_path`, path conversion, mutation extraction and concatenation replace
# the inclusive gap timer; their remainder is shown as other gap work. These
# worker totals overlap in wall time and are **not wall-time decompositions**.
# Reference packing and worker setup are amortised over matched haplotypes.
# Loader timings are shown separately in the table above.
#
# **Per-sample timings are unavailable:** the archive retains aggregate stage
# totals and call counts, without timings keyed by sample or haplotype. The
# per-haplotype values here are workload averages, not individual measurements;
# a coverage-versus-time scatter plot cannot be recovered from these totals.

# %%
if __name__ == '__main__':
    plot.save(plot.worker_stages(stages), 'greedy_worker_stages')

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
# **retained** connected focal regions at the cohort-specific minimum;
# unfiltered regions were not archived. `num_bins` is directly editable below.
# Dashed site thresholds are the actual filters. The physical-length markers
# are mean-density equivalents, not strict physical cutoffs for each region.
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
    num_bins = config['plot']['focal_histogram_bins']
    plot.save(plot.focal_lengths(lengths, unit='sites', num_bins=num_bins), 'focal_lengths_sites')
    plot.save(plot.focal_lengths(lengths, unit='kb', num_bins=num_bins), 'focal_lengths_kb')
    plot.save(plot.gap_lengths(lengths), 'gap_lengths_kb')

# %% [markdown]
# ### Recurrent sites
#
# A recurrent site contains **two or more mutation records**, including back
# mutations. Compare saved raw Full HMM and stitched trees before simplification
# or dating, with truth restricted to the same archived inference-site positions.
# Raw inferred trees retain reference ancestry, as in the tree-count plots below.
# Counts use `np.bincount(ts.mutations_site)` rather than decoding genotypes.
#
# The left panel counts recurrent sites; brackets above each bar give their
# percentage of **all saved inference sites** in that cohort. The right panel
# gives the mean number of mutation records **conditional on a site being
# recurrent**, including the first event. Truth has no genotype errors; the
# inferred trees were reconstructed from the error-containing calls.
# Full HMM has no recurrent sites here, so its conditional mean is undefined
# and has no bar in the right panel.

# %%
if __name__ == '__main__':
    recurrence = ch5_gapfill.recurrent_sites(config)
    ipython_display.display(recurrence.round(3))
    plot.save(plot.recurrence(recurrence), 'recurrent_sites')

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
    plot.save(plot.root_totals(root_edges), 'root_edges_counts_spans')
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
# rate is 2.35 × 10⁻⁸ per bp per generation. Independent trees run in worker
# processes and their dated outputs are cached.

# %%
if __name__ == '__main__':
    dated = ch5_perf_scaling.date_all(config)
    ipython_display.display(dated[['dataset', 'method', 'af_cutoff', 'raw_sites', 'dated_sites',
                                  'num_edges', 'num_trees', 'benchmark_paths_identical']])
    metrics = {'edges_change_percentage': 'Change in raw edges vs Full HMM (%)',
               'trees_change_percentage': 'Change in raw trees vs Full HMM (%)'}
    plot.save(plot.scaling(dated, metrics, 'Raw tree counts', zoom=True), 'raw_tree_counts')

# %% [markdown]
# ## Mutation ages: linear and rank correlation
#
# Posterior **mutation means** (`mn`) are compared with simulated event times
# and with Full HMM posterior means. Full HMM is a comparison method, not truth.
#
# For each cohort, use one common variant intersection across Full HMM, the
# stitched run and truth. Match position and ancestral/derived
# states; exclude recurrent/back-mutation sites and nonfinite ages. All methods,
# reference comparisons and frequency bins use that same cohort intersection.
# The cached tables include its retention relative to each candidate set.
#
# Pearson r uses untransformed ages in generations. Spearman rank correlation
# compares age ordering using average ranks for tied ages. DAF is fixed from Full
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
    age_metrics = {'pearson_r': 'Pearson r', 'spearman_r': 'Spearman rank correlation'}
    plot.save(plot.age_correlations(ages, metrics=age_metrics), 'mutation_age_correlations')

# %%
if __name__ == '__main__':
    for index, label in enumerate(config['frequency_bins']['labels']):
        plot.save(plot.age_correlations(ages, label, metrics=age_metrics), f'mutation_age_daf_{index}')

# %% [markdown]
# ## Paired mutation ages
#
# Hexagons show counts with one shared logarithmic colour scale. The dashed
# line is equality. Display bounds omit ages below one generation and use a
# shared upper quantile; correlations above use all matched ages,
# including display tails. The annotated N is the complete common intersection.

# %%
if __name__ == '__main__':
    if cohorts.has_truth.all():
        plot.save(plot.age_density(ages, 'truth'), 'mutation_ages_vs_truth')
    plot.save(plot.age_density(ages, 'full_hmm'), 'mutation_ages_vs_full_hmm')
