# Chapter 5 exploratory analyses

Run notebooks and paired sources from the repository root. Shared preparation,
workers and exploratory algorithms live in `ch5.nb_utils`; every figure and
axis change lives in `ch5.nb_plot`. Settings are initialised in cells and
presentation choices are explicit call arguments. Saved outputs retain their
previous execution; a selected perf5 loading/performance/stage check was run
separately into a temporary destination and visually inspected.

The notes below preserve the original notebook-specific methods and caveats.
All former analysis YAML settings are now in notebook cells. Upstream archive
`config.yaml` files remain read-only run provenance beside their data.

# Zero-coverage focal stitching

`scripts/chunk_matching.py` reads the copied reference, panel, sample calls,
and focal candidates from its explicit input directories. It reads anchored focal intervals
and population labels from the configured `coverage_input_dir`. Paths inside
the upstream TOML files are not followed.

Focal coverage is the union of the budget-zero intervals with
`min_focal_ac <= focal_ac <= ac_cutoff`. The intervals extend from each
ancestor's leftmost focal seed until a mismatch, missing call, or support bound.
Gaps are maximal runs outside that union, regardless of any HMM path.

Within each covered component, select an available interval extending furthest
right and keep that parent until the interval ends. Break equal-endpoint ties
by original reference node ID. This minimizes focal copying segments; it does
not optimize gap endpoint continuity. Copied alleles come from the inferred
reference, so differences from generated ancestor calls remain visible as
mutations. Each gap is independently matched against the full reference, and a
separate whole-span HMM provides the comparison path.

Run it from the repository root using the explicit CLI described in
[`scripts/README.md`](../../scripts/README.md). This starts matching work;
opening a result notebook only reads completed products.

Omitting `--haplotypes` evaluates the complete focal roster. Omitting `--workers` uses all available CPU cores, with queued whole-haplotype jobs
and reusable matcher indexes in each worker. Results stream to separate full
and stitched JSONL files and `comparison.csv`, including population labels,
switches within focal coverage, within gaps and at seams, total extra switch
fraction, and four site-weighted parent-agreement fractions. Agreement fractions
use all inference sites; extra switch fraction uses stitched switches.

Open the committed output-bearing notebook with
`uv run jupyter lab notebooks/ch5/stitching_pt2.ipynb`.

The notebook reads the existing results, displays selected-haplotype diagnostics,
and builds an 800-row table covering both methods for the two 200-haplotype
cohorts. Its final section exports tables and population plots under the
configured cohort output directory. Error bars show one sample standard
deviation across haplotypes. Both haplotypes of each diploid individual are
included, with three population groups per dataset.

## Fixed-pipeline hyperparameter analysis

`notebooks/ch5/stitching_pt3_hyperparams.py` (paired with `.ipynb`) reads the
retained upstream grid in
`data/ch5/match_eval/stitching_pt3_experiment_config.yaml`, with scientific
settings initialised in its cells. The snapshot is read against the explicit
original relative base (`data/ch5/`), preserving its bytes. It compares all configured AC cutoffs
and mismatch allowances using saved full and stitched HMM paths and focal
masks. It does not run inference or use previous pt3 result tables.

The notebook includes extra switches, per-haplotype percent switch increases,
physical focal/gap coverage, pooled region lengths, raw mean switch
compositions, and within-region parent agreement. Simulation is on the left
and 1KGP on the right; African/Non-African labels use preserved simulation annotations and
pedigree superpopulations. Hourglasses show mean ± one haplotype SEM.

Coverage is clipped to the configured 10 Mb windows, excluding chromosome
tails retained by the tree sequences. The single Full HMM composition bar
uses the configured baseline AC cutoff's partition for each mismatch row.
Length plots show the pooled 99th percentile range and report tail counts;
their KDEs and exported length tables use all intervals.

The tree sequence section loads both Full HMM and all stitched raw matched
outputs. It plots stitched minus Full HMM edge counts in the first row and tree
counts in the second, against AC cutoff. Simulation is on the left and 1KGP on
the right; each panel has two mismatch-coloured lines. Switch composition
retains a one-bar-width gap between the Full HMM and stitched bars.

For both datasets, `tscompare.haplotype_arf(stitched, full)` computes ARF, TPR
for every stitched parameter setting, treating Full HMM as the
reference. Biological haplotypes are joined using sample ID and ploidy index
in both matched outputs' node metadata, aligned to identical sample node IDs,
restricted to the dataset's 10 Mb window, and simplified with unary ancestors
retained. ARF measures stitched node span unsupported by Full HMM; TPR measures
Full HMM node span recovered by stitching. Metric rows are ARF and TPR,
with simulation/1KGP columns and mismatch-coloured lines. A short interpretation
note gives the AC 3 / zero-mismatch simulation values in those terms.

The final section separately compares the simulated Full HMM and all eight
stitched outputs against the true simulated ARG. The Full HMM baseline is
displayed at `1 (Full HMM)` and calculated once. ARF/TPR are separate columns,
with mismatch-coloured lines and y-axes scaled to the observed differences.
Truth sample IDs use the VCF identity mapping and the same 10 Mb window; unary
ancestors and missing terminal ancestry are retained. Population annotations
remain in `sim_n300_10mbp_error_populations.csv`.

Run and export all twelve figures and the underlying CSV tables with:

```sh
uv run python notebooks/ch5/stitching_pt3_hyperparams.py
```

Outputs go to `data/ch5/stitching_pt3_hyperparams/`. Independent parameter runs
use worker processes across available CPU cores; set `workers=1` in the
notebook settings for sequential loading. The notebook's run summary records
empty regions and undefined percent increases excluded from the means.
`ts_statistics.csv` records raw output counts, `ts_statistic_differences.csv`
records signed stitched-minus-full counts, and `arg_metrics.csv` records all
16 stitched-to-full comparisons, including reference method and time units.
`arg_metrics_against_truth.csv` records the nine simulated outputs compared
with truth. Metric tables and plots contain ARF and TPR only.

## Archived AF/minimum-region parameter search (pt7)

`notebooks/ch5/stitching_pt7_paramsearch.py` (paired with `.ipynb`) analyses
`data/ch5/hpc_params1_diagnostics` using its archived config and manifest.
Analysis parameters are initialised in the notebook. Sample sizes
occupy columns; AF cutoffs or metrics occupy rows. Focal length histograms use
minimum size 1, with `num_bins` editable in the notebook. Gap plots show only
densities and median markers. The final sections compare tree statistics,
ARF/TPR against Full HMM and truth, and mutation-age correlations overall and
by DAF. No heatmaps or genotype-store restoration are used.

All generated summaries, figures and dated trees are cached under the input
directory's `for_plots/`. Integer region-length counts preserve the complete
distributions without repeatedly loading the 4 GB CSV. Tree metric plots load
completed cached comparisons only; missing values/panels stay blank, and the
notebook never starts further ARG metric calculations. Dating runs in worker
processes (`workers=None` uses available CPUs), with per-run input/settings/code
signatures. Mutation-age comparisons use one
position/allele intersection across truth, Full HMM and every stitched run
within each sample size, for both reference comparisons and all DAF bins.
For large cohorts, `ch5.nb_utils.BitsetCladeMap` uses exact integer clade bitsets in each
single-threaded worker; tscompare's metric formulas and matching code are
unchanged. The original clade-bitset calculation is retained; rough chapter 5 code has no automated test suite.

```sh
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/ch5/stitching_pt7_paramsearch.ipynb
```

## HPC performance scaling (pt8 and pt9)

`stitching_pt8_hpc_perf1.py` and `stitching_pt9_hpc_perf2.py`, paired with
notebooks of the same names, analyse the two extracted diagnostic archives
under `data/ch5/`. Their scientific settings and explicit plot arguments are in their cells.
Performance plots put actual diploid sample counts on a logarithmic x axis,
with one shared Full HMM baseline per cohort. They show recorded end-to-end
wall time, throughput, CPU time, average cores and sampled process-tree private
memory; stage/resident measurements are absent from these experiments.

Path diagnostics include extra switches, population coverage, focal/gap lengths,
switch partitions, parent agreement and raw edge/tree counts. Dating checks
that the diagnostic and timed trees have matching copying paths and mutations.
Mutation ages use one common position/allele intersection per cohort, including
truth for simulations; 1kgp comparisons use Full HMM only. No ARG metrics or
genotype-store restoration are run. Results, matched age pairs, figures and
dated trees are cached in each extracted archive's `for_plots/` directory.
The notebooks describe methods and show results; interpretation stays in chat.

Their gap-parent section compares Full HMM and stitched copying under the same
gap masks. Parents are older than the AF cutoff, recent without a carried focal
seed, or recent with a carried focal seed. Seeds are recovered from the checked
single-mutation reference sites; saved Full HMM haplotypes provide a common
carrier definition, including terminal-mutation overrides. Stacked percentages
show mean per-haplotype gap-bp and distinct-parent shares. Per-haplotype caches
also retain inference-site and copying-segment weights. This is an analysis of
existing paths, not a run of an age-restricted HMM.

## Gap filling and instrumented HPC performance (pt11)

`notebooks/ch5/stitching_pt11_gapfill.py`, paired with `.ipynb`, analyses
`data/ch5/hpc_perf3_diagnostics` using scientific settings initialised in its cells.
Extract `data/ch5/hpc_perf3_diagnostics.tar.gz` into that directory first.
The three simulated cohorts contain 50, 500 and 5,000 diploid samples, with
AF cutoff 0.1 and respective focal-length minima of 25, 50 and 100 kb.
Each method/cohort has one trial at 94 matching workers.

Performance figures show wall time, throughput, private memory, speedup and
achieved parallelism (process CPU time / measured wall time). Additional
figures partition sequential pipeline wall time and worker elapsed/CPU time.
Nested gap-stage totals are replaced by their children plus a residual;
overlapping worker service times are never added to the wall-time stack.
Only aggregate timings are archived, so per-haplotype timing values are means
and per-sample timing scatter plots are unavailable.

The remaining pt9 diagnostics are retained, with compact root-copying panels.
Recurrent-site counts replace the one-site-gap plot and use vectorised
mutation-site counts on a common inference-site axis, including truth.
Parentheses above bars show the percentage of all inference sites; the right
panel shows mean mutations conditional on recurrence. Pearson and Spearman
mutation-age correlations use the same cohort mutation intersection. The
gap-parent ancestor-category section is omitted. Tables, dated trees and
PNG/PDF figures are cached under the extracted archive's `for_plots/`.

```sh
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/ch5/stitching_pt11_gapfill.ipynb
```

## Simulations and 1KGP supervisor update (pt13)

`notebooks/ch5/stitching_pt13_jerome_update.py`, paired with `.ipynb`, uses
cell-initialised settings and the `perf5_diagnostics.tar.gz`
archive. It includes simulations with 50, 500, 5,000 and 50,000 diploid
individuals (up to 100,000 haplotypes), plus 3,202 diploid 1KGP individuals.
Each method/cohort has one trial at 96 matching workers.

Lines connect all cohorts with one colour per method, labelled Full HMM and
Greedy HMM, and a common point shape. These heterogeneous cohorts are visual
comparisons rather than a controlled scaling experiment. AF/AC settings appear
in dataset-panel titles. Tables, dated trees and PNG/PDF figures are cached
under `data/ch5/perf5_diagnostics/for_plots/`.

```sh
mkdir -p data/ch5/perf5_diagnostics
tar -xzf data/ch5/perf5_diagnostics.tar.gz -C data/ch5/perf5_diagnostics
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/ch5/stitching_pt13_jerome_update.ipynb
```
