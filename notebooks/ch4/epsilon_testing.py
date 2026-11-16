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
#     display_name: Epsilon testing (TSLS sample n)
#     language: python
#     name: epsilon_testing
# ---

# %% [markdown]
# # Epsilon sanity check with inferred ancestors
#
# BMRC / cloudcomp113, 13 November 2026. Both panels are inferred from the noisy Zarr genotypes; the simulated ARG supplies evaluation labels only. No true ancestors enter matching.
#
# The initial sweep is **1e-4, 3e-5, 1e-5, 1e-6 errors per haplotype-bp**. Following the recombination/mismatch review, the biallelic HMM emission is `p = epsilon × simulated_span_bp / source_site_count`, using source sites before the ancestor inference filters. Ancestor matching uses stock parameters (constant recombination 0.01, mismatch 1e-20, no map). The fixed ancestor ARG is reused for every epsilon. Only sample matching uses the calibrated emission and the simulation's HapMapII_GRCh38 map. The new transition model uses fixed **n = 100 / 1,000 haplotypes** throughout. This is provisional calibration for chapter 5 performance work, not a held-out estimate of an optimal biological model.

# %%
get_ipython().run_line_magic("matplotlib", "inline")
import json
import logging
import pathlib
import subprocess
import sys

import IPython.display as ipython_display
import matplotlib.pyplot as plt
import pandas as pd
import tsinfer
import tsls

from ch4 import nb_plot, nb_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
repository_dir = pathlib.Path("/well/kelleher/users/uuc395/dphil-analysis")
input_dir = pathlib.Path("/well/kelleher/users/uuc395/tsinfer-anc-eval/data/anc_perf")
output_dir = repository_dir / "data/ch4/epsilon_testing/run_20261113_exact_samples"
figure_dir = repository_dir / "figures/ch4/epsilon_testing/run_20261113_exact_samples"
figure_dir.mkdir(parents=True, exist_ok=True)

datasets = (
    nb_utils.EpsilonDataset(
        name="n50", zarr_path=next((input_dir / "zarr_vcfs").glob("*-n50-*.zarr")),
        truth_path=next((input_dir / "simulated").glob("*-n50-*.trees")),
        left=0, right=28_100_000,
    ),
    nb_utils.EpsilonDataset(
        name="n500", zarr_path=next((input_dir / "zarr_vcfs").glob("*-n500-*.zarr")),
        truth_path=next((input_dir / "simulated").glob("*-n500-*.trees")),
        left=0, right=28_100_000,
    ),
)
config = nb_utils.EpsilonSweepConfig(
    datasets=datasets,
    epsilons=(1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 1e-8, 1e-9, 1e-10),
    genetic_map=pathlib.Path("/users/kelleher/uuc395/.cache/stdpopsim/genetic_maps/HomSap/HapMapII_GRCh38/genetic_map_Hg38_chr20.txt"),
    output_dir=output_dir, num_workers=4, threads_per_match=8, cache_size_mb=65536, ancestor_exclude="INFO/observed_derived_ac == 1", sample_likelihood_threshold=0.0,
)


# %% [markdown]
# ## Runtime and provenance
#
# An isolated source build uses TSLS `e4cf233` (sample-size scaling) and tsinfer `e2f8de7`, with a small local patch passing `MatchSourceConfig.sample_size` through `HmmParameters` into every TSLS HMM. The parent checkouts and their environments are preserved. The patch and exact versions are retained with the run. This kernel uses the dedicated build; ordinary project `uv run` retains its earlier dependency pins.

# %%
runtime = {
    "host": subprocess.check_output(["hostname"], text=True).strip(),
    "python": sys.executable,
    "tsinfer_path": tsinfer.__file__, "tsls_path": tsls.__file__,
    "tsinfer_version": tsinfer.__version__, "tsls_version": tsls.__version__,
    "tsinfer_revision": subprocess.check_output(["git", "-C", str(pathlib.Path(tsinfer.__file__).parents[1]), "rev-parse", "HEAD"], text=True).strip(),
    "tsls_revision": subprocess.check_output(["git", "-C", str(pathlib.Path(tsls.__file__).parents[1]), "rev-parse", "HEAD"], text=True).strip(),
    "epsilon_units": "errors per haplotype-bp", "transition": "ancestors: stock, no map; samples: Haldane(map distance); fixed haploid sample_size",
    "workers": config.num_workers, "threads_per_match": config.threads_per_match, "cache_size_mb": config.cache_size_mb,
}
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
patch = subprocess.check_output(["git", "-C", str(pathlib.Path(tsinfer.__file__).parents[1]), "diff"], text=True)
(output_dir / "tsinfer_sample_size.patch").write_text(patch)
ipython_display.display(pd.Series(runtime))

# %% [markdown]
# ## Inputs, labels and ancestor generation
#
# The source headers have a full-chromosome length, while the simulation was restricted to `[0, 28,100,000)` bp. Calibration uses this explicit simulated span. Stored genotype-error masks are checked against the per-site error counts; Zarr and truth sites must align exactly by physical position. The cached three-column simulation map is integrated into a four-column map for tsinfer and checked at every source site.
#
# Path compression is disabled, as in the matching pipeline. Ancestors exclude observed derived singletons using the same cohort-count convention as the matching pipeline. Counts are recomputed from noisy calls; the stored pre-error singleton annotation is not used. Without this exclusion, singleton ancestors let samples copy their own observations and all four initial epsilon values gave zero recurrent calls on the 50-individual panel. The unfiltered attempt is retained separately under `run_20261113_stock_ancestors`. Source-site density for epsilon conversion still includes singleton sites.
#
# **Primary evaluation unit: an inference site.** A truth-positive site has at least one false `call_genotype_error_mask` entry (these stores use true for unchanged alleles), or at least two mutation records in the simulated ARG. A predicted-positive site has at least two mutation records in the inferred ARG, including back mutations. TP/FP/FN/TN are the intersection/differences of these sets. Error and biological recurrence can overlap; component recall is shown separately. This detects problematic sites, not the exact erroneous allele or recurrent event.
#
# The secondary covered-source view also includes sites added by parsimony after matching. Four source sites lie outside the smaller panel's inferred flanks and are reported as omitted; no inference site is omitted. Those sites did not have an HMM decision, so the primary tuning decision uses inference sites. Ordinary true terminal mutations are negative unless the site is recurrent or error-affected.

# %%
dataset_summary = nb_utils.prepare_epsilon_sweep(config)
ipython_display.display(dataset_summary.T)

# %% [markdown]
# ## Matching sweep
#
# Ancestors are generated and stock-matched once per dataset from its noisy observations. Every epsilon matches the samples against that same fixed ancestor ARG. Epsilon and the map apply only to sample matching. Four worker processes each use eight matching threads (32 matching threads maximum; 64 GiB cache budget per run). Per-run trees, site predictions, configurations, logs and metrics are retained. Rerunning this cell reuses completed run metrics; use a new output directory if changing scientific inputs or the implementation.

# %%
ancestor_references = nb_utils.prepare_epsilon_references(config)
ipython_display.display(ancestor_references)

# %%
metrics = nb_utils.run_epsilon_sweep(config)
primary = metrics[metrics.scope == "inference_sites"].sort_values(["individuals", "epsilon"])
columns = ["dataset", "epsilon", "mismatch_probability", "tp", "fp", "fn", "tn", "precision", "recall", "false_positive_rate", "f1", "mcc", "match_seconds", "source_sites_outside_flanks"]
ipython_display.display(primary[columns])

# %% [markdown]
# ## Classifier operating points
#
# Sample matching disables likelihood flooring (`likelihood_threshold=0`) to retain the exact Viterbi costs at zero-map intervals; the default floor caused reproducible traceback failures on the 50-individual data. The map is unchanged and TSLS has no diagnostic patches. Ancestor matching keeps all stock defaults.
#
# The grid was extended downward because the preliminary 500-individual F1/MCC optima were at 1e-6, the initial lower edge. Each point is a separately inferred ARG. Epsilon is a model parameter, not a threshold applied to one fixed ranking; the connected ROC/PR points need not form a monotone curve. No extrapolated endpoints or AUC are used. F1 and MCC are more informative than accuracy when positives are uncommon. All tuning is on one simulation seed per sample size; differences cannot be attributed to sample size alone.

# %%
fig = nb_plot.plot_epsilon_classifier(metrics, scope="inference_sites", figsize=(13, 9), dpi=130, annotate=True, annotate_epsilons=(1e-4, 1e-6, 1e-8, 1e-9, 1e-10), colors=("tab:blue", "tab:orange"))
fig.savefig(figure_dir / "classifier_inference_sites.png", bbox_inches="tight")
plt.show()

# %%
fig = nb_plot.plot_epsilon_truth_components(metrics, scope="inference_sites", figsize=(13, 4), dpi=130)
fig.savefig(figure_dir / "truth_components.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Recurrence summaries (as in stitching pt12)
#
# A recurrent site contains at least two mutation records. The conditional mean counts **all** records at those sites, including the first event. These include mutations on inferred internal ancestors and on sample nodes; they are not a count of sample errors. Dotted lines show simulated biological recurrence, which excludes injected errors.

# %%
fig = nb_plot.plot_epsilon_recurrence(metrics, scope="inference_sites", figsize=(12, 8), dpi=130)
fig.savefig(figure_dir / "recurrence_inference_sites.png", bbox_inches="tight")
plt.show()
ipython_display.display(primary[["dataset", "epsilon", "recurrent_sites", "recurrent_site_percentage", "mean_mutations_per_recurrent_site", "additional_mutations", "recurrent_records_on_samples", "recurrent_records_on_internal_nodes"]])

# %% [markdown]
# ## Secondary view: covered source sites
#
# This view includes post-matching augmentation over covered flanks and should not be used alone to judge the emission parameter. The omitted-site counts are included in the metrics table.

# %%
fig = nb_plot.plot_epsilon_classifier(metrics, scope="covered_source_sites", figsize=(13, 9), dpi=130, annotate=True, annotate_epsilons=(1e-4, 1e-6, 1e-8, 1e-9, 1e-10), colors=("tab:blue", "tab:orange"))
fig.savefig(figure_dir / "classifier_covered_source_sites.png", bbox_inches="tight")
plt.show()
fig = nb_plot.plot_epsilon_recurrence(metrics, scope="covered_source_sites", figsize=(12, 8), dpi=130)
fig.savefig(figure_dir / "recurrence_covered_source_sites.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Provisional tuning result
#
# The table compares the best tested F1, MCC and balanced-accuracy operating points. A maximum at a grid edge is a reason to expand the grid. The practical chapter 5 choice should balance precision and recall, rather than match the aggregate number of mutations to the number of injected errors. This sanity check deliberately does not use true ancestor panels, clean/noisy paired controls, multiple seeds, or a held-out evaluation set.

# %%
recommendations = nb_utils.epsilon_recommendations(metrics)
recommendations.to_csv(output_dir / "recommendations.csv", index=False)
ipython_display.display(recommendations[["dataset", "criterion", "epsilon", "mismatch_probability", "precision", "recall", "false_positive_rate", "f1", "mcc"]])

# %%
macro_scores = nb_utils.epsilon_macro_scores(metrics)
macro_scores.to_csv(output_dir / "macro_scores.csv", index=False)
ipython_display.display(macro_scores.sort_values("f1", ascending=False))

# %% [markdown]
# ## Practical choice for the chapter 5 sanity check
#
# **Use epsilon = 1e-9 as the initial shared tuning value.** It has the best tested F1 on both datasets and the highest equal-weight mean F1/MCC across the two. The calibrated sample emissions are **1.93e-7** (50 individuals) and **8.84e-8** (500 individuals).
#
# | Individuals | Precision | Recall | F1 | MCC |
# | --- | ---: | ---: | ---: | ---: |
# | 50 | 0.744 | 0.663 | 0.701 | 0.641 |
# | 500 | 0.799 | 0.810 | 0.804 | 0.652 |
#
# The optimum is broad: epsilon = 1e-6 gives more recall (0.790 / 0.874) at lower precision (0.615 / 0.743), with mean F1 only about 0.005 lower. The 500-individual MCC maximum is at the lower grid edge, 1e-10; do not interpret 1e-9 as a precisely established universal optimum.
#
# This is a copying-model tuning value with inferred, noisy-data ancestors, **not a re-estimate of the physical observation-error rate**. The realised injected rates are 9.48e-6 and 1.08e-5 per haplotype-bp. True-ancestor controls, independent seeds and larger cohorts remain future work. For these performance checks, keep the stock ancestor stage fixed, apply the map and calibrated emission only to samples, and keep sample likelihood flooring disabled.
#
