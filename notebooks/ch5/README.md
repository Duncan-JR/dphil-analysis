# Chapter 5 exploratory analyses

Run notebooks and paired sources from the repository root. Shared preparation,
workers and exploratory algorithms live in `ch5.nb_utils`; every figure and
axis change lives in `ch5.nb_plot`. Settings are initialised in cells and
presentation choices are explicit call arguments. Historical saved outputs
retain their previous execution, except pt12, which was explicitly replaced
and rerun on 16 November 2026 as described below. A selected perf5
loading/performance/stage check was run separately into a temporary destination
and visually inspected.

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

## Corrected conditioned-gap and low-epsilon analysis (pt12, 16 November 2026)

`stitching_pt12_conditioning.py` and `.ipynb` replace the old 500 kb pilot with
completed allele-corrected runs for OOA n50/n500/n5000 and 1KGP n3202. The
pipeline fixed back-mutation allele handling in commit `6be4459`; the previous
notebook had never been replaced. The later optimisation report explicitly
recorded that it modified no notebook. The unsynchronised changes were checked:
the previous pair's cell sources agreed, and its notebook-only change was
metadata. Both original files, including visible outputs, are backed up in
`data/ch5/provenance/pt12_before_20261116/` on BMRC.

Scientific inputs are extracted from
`/well/kelleher/users/uuc395/tsinfer-match-eval/data/conditioning_test_rfloor_1e20/diagnostics/conditioning_test_rfloor_1e20_diagnostics.tar.gz`
into `data/ch5/stitching_pt12_conditioning/corrected_run/`. The archive includes
its native configurations, recombination maps, raw matched/reference trees,
simulated truth, cohort identities, masks and quality records. All 16 original
called-allele checks pass. Scientific settings are read from this immutable run
provenance and checked against the notebook's stated experiment; the main comparison reads saved trees. The later Deep dive section explicitly
replays the captured 258-site failure, without rerunning cohorts or ancestors.

Samples use epsilon 1e-9 per haplotype-bp, calibrated over 28.1 Mb of source
sites to constant per-site mismatch probabilities. The map-based sample
recombination floor is 1e-20; likelihood flooring and both pruning thresholds
are zero. Ancestor matching retains stock native probabilities without the
sample map/epsilon. AF cutoffs are 0.1/0.05/0.05/0.05 and mean-density minimum
focal sizes are 25/50/100/150 kb. Simulations contain 1% genotype error and no
phase error. The exact-zero map controls and the floor's scientific consequence
are explained in the notebook. The completed unconditioned control now isolates gap conditioning at fixed epsilon;
there is no epsilon sweep.

Paired haplotype diagnostics score saved paths with cohort-scaled switch odds
and mismatch odds, reporting transition/emission contributions, terminal sample
mutations and switches in focal interiors, gaps and seams. Path scores are
Viterbi weights, not forward or genealogy probabilities. Recurrent sites use
raw trees on the same inference-site axis, with truth for simulations only.
Allele ages use tsdate posterior mutation means after the existing pt13
post-processing/preprocessing. Comparisons hold one position/allele intersection
fixed across methods and optional truth, exclude recurrent sites, report
retention and use Full HMM frequencies for DAF bins. Pearson/Spearman, signed
bias and absolute error accompany paired density figures.

Only the performance section reads the final optimisation suite from
`../tsinfer-match-eval/data/performance/conditioning_greedy_optimized_20261114/`.
It includes 1KGP, uses one trial at 96 workers for each dataset/method/mode,
and displays no superseded timings. From-reference and resident-match timer
scopes are separate. The final validation records 16 successful called-allele
checks and eight exact greedy path/mutation comparisons against the corrected
baseline. Native/pipeline sources were dirty; `source_snapshot/`,
`final_validation.json` and archived logs preserve their identities. Existing
logs report 688 TSLS checks and 55 pipeline checks passing; these suites were
reviewed, not rerun or added to dphil-analysis.

The replacement is executed on cloudcomp113 through the existing Python 3.12
analysis environment under `tmp/epsilon_testing_20261113/env`, after `loadpy`.
An in-process IPython batch captures tables/figures into the notebook without
starting a Jupyter server or kernel. This saved-tree runtime is independent of
the final benchmark's native environment; package versions are recorded in the
batch validation record. Run logs and the executor are under
`/well/kelleher/users/uuc395/tmp/pt12_review_20261116/`.
Tables, dated trees, mutation pairs and PNG/PDF figures are under
`data/ch5/stitching_pt12_conditioning/corrected_run/analysis_20261116/`.
All eight notebook code cells executed successfully, with six saved figures
and six PNG/PDF export pairs across 17,504 paired haplotypes. Paired n50 path
score differences reproduce the independent archived 100-haplotype control;
reference-continuity checks pass for all four cohorts and both methods. All
six plots were visually inspected; overlapping age-panel titles were adjusted
and those two figures refreshed from cached age pairs. Final numerical/source
validation is recorded in `tmp/pt12_review_20261116/validation.json`.
No chapter 5 tests are introduced. Older notebooks and unrelated chapter 4
working changes are preserved.


The matched unconditioned control (16 November 2026) is extracted from
`../tsinfer-match-eval/data/conditioning_test_control/diagnostics/conditioning_test_control_diagnostics.tar.gz`
into `data/ch5/stitching_pt12_conditioning/unconditioned_run/`. Every scientific
figure includes unconditioned greedy; performance retains its original output.
Runtime checks require identical native settings/maps, cohort identities,
reference paths and Full HMM paths across both archives. Control greedy trees
are dated directly without benchmark inputs. Age comparisons use a newly
computed common nonrecurrent mutation intersection across Full HMM, conditioned
greedy, unconditioned greedy and optional truth; the earlier two-method caches
are preserved. Combined tables and scientific figures are in
`data/ch5/stitching_pt12_conditioning/comparison_20261116/`. Pre-update notebook
outputs and helpers are backed up under
`/well/kelleher/users/uuc395/tmp/pt12_control_20261116/`. Execution and validation
logs are kept there too. Shared plot defaults retain older notebooks' behavior.


Part 12 control validation: seven code cells reran successfully from saved trees;
five scientific PNG/PDF pairs were regenerated, and the sixth performance figure
and its saved cell outputs were preserved exactly. All five updated scientific
figures were visually inspected. All 17,504 matched haplotypes
have conditioned path weight at least the unconditioned weight within 1e-6 log
weight. Both arms share exact Full HMM/reference fingerprints; all age methods
use equal common-intersection counts within each cohort. Pair sources agree; notebook
cell IDs and attachments are preserved. No matching, benchmarks or tests ran.


Part 12 compares direct ultimate-root edge counts and observed spans per
haplotype across the saved Full HMM and both greedy runs. It uses the archived
focal masks and observed intervals, including signed conditioned-minus-
unconditioned differences and common-bin span histograms. The precision and
recall plots evaluate inferred recurrence at the shared simulation inference
sites. The truth set is the union of sites with simulated recurrence and sites
with at least one injected error, joined by position using the original
simulation Zarr stores. Sites with both labels count once. 1KGP has no truth
scores. The new exports are in
`data/ch5/stitching_pt12_conditioning/comparison_20261116/root_edges_and_recurrence/`.


The pt12 path panel shows mean sample mismatch counts, extra switches, and
total saved-path log weights by method. The total includes all-stay transition
weights and all-match emission weights over called inference sites, and is
conditional on the initial parent. Called-site counts come from the native
sample Zarr stores identified in the archived configurations. The extra-switch
percentage averages per-haplotype ratios with positive Full HMM denominators.
Physical focal coverage uses the shared mask and pt13 population groups.
Root-edge totals use grouped log-scale bars. The updated path and coverage
exports are in the comparison directory above.


Part 12 now ends with a **Deep dive** into the captured n50 `tsk_0`/haplotype-0
failure at sites `[90955, 91213)`. Evidence from
`tmp/conditioning_numerics_20261114/`, the original zero-map Full HMM tree and its
trial/quality records, and the original source map are preserved with hashes in
`data/ch5/stitching_pt12_conditioning/failure_deep_dive_20261116/evidence/`.
It replays only the small captured window with the existing conditioning-capable
TSLS analysis runtime. Controls remove each endpoint, retain no endpoints, add
the configured floor, or increase mismatch probability. The map/path/impossibility
diagram and three allele-resolved figures cover all 258 sites, using the actual
binary simulation alleles. Tables include every parent and copied allele.
The source map assigns exactly zero rate over 26,328,619–30,187,708 bp; the
failure is an incompatible pair of hard parent pins inside that plateau, not
floating-point cancellation. The section explains why a no-floor policy needs
to reconsider incompatible focal choices across zero-transition blocks.
Pre-addition notebook outputs/helpers and execution records are under
`/well/kelleher/users/uuc395/tmp/pt12_failure_20261116/`. Earlier scientific and
performance outputs are preserved. All three new code cells executed successfully;
all four new PNG/PDF figure pairs were visually inspected. The original 15 cells,
including their outputs, IDs and attachments, and notebook metadata are preserved
exactly. All 258 exported allele rows agree with the captured sample and the saved
paths; the floored window replay exactly matches the saved conditioned path.
Both the analysis and installed pipeline TSLS runtimes agree on all six controls.
Increasing mismatch probability to 0.01 still cannot connect the two hard pins
with zero-rate transitions. The archived floor sensitivity table is also shown.
Jupytext source equality and source syntax checks pass; no tests or large jobs
ran. No pipeline/native implementation is changed.

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

## Preserved notebooks and migration validation (12 November 2026)

The committed `.ipynb` files remain the readable historical record. All saved
outputs, attachments, execution counts and cell types match the pre-migration
hashes in `plans/refactor_notebook_inventory.json`. Source cells and paired
files use the migrated interfaces. The checks below ran separately, writing
new validation figures/tables only under `/private/tmp/dphil_refactor_validation/`;
they did not replace saved notebook outputs. Code-cell numbers below are
zero-based among code cells. Detailed evidence is in
`plans/refactor_notebook_reruns.json`. These are selected-cell checks, not a
claim that every expensive historical computation was rerun.

| Notebook (paired `.py`/`.ipynb`) | Purpose | Original/retained inputs | Separate check |
| --- | --- | --- | --- |
| [ancestor_coverage](ancestor_coverage.ipynb) | Exact interval sweeps and population ancestor-coverage distributions | Same saved interval NPZs and simulated truth; no new matching | Code cells 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 passed; historical visible outputs preserved |
| [initial_testing](initial_testing.ipynb) | Early focal-frequency/coverage/likelihood diagnostics | Existing `../tsinfer-match-eval/data/match_eval` focal statistics CSVs and ancestor NPZs | Code cells 0, 1, 2, 3, 4, 5, 6, 7, 8 passed; historical visible outputs preserved |
| [stitching__scratch](stitching__scratch.ipynb) | Historical greedy interval-cover algorithms on tiny explicit examples | Cell-defined intervals only | Code cells 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 passed; historical visible outputs preserved |
| [stitching_pt1](stitching_pt1.ipynb) | Retained-piece/gap matching on the original three-haplotype records | `data/ch5/chunk_matching_experiment/`, copied `match_eval/` panel and saved coverage input | Code cells 0, 1, 2, 3, 4, 5, 6, 7 passed; historical visible outputs preserved |
| [stitching_pt10_age_performance_review](stitching_pt10_age_performance_review.ipynb) | Read-only display of the retained 22 October review and evidence tables | `data/ch5/mutation_age_performance_review_2026_10_22/` HTML/CSV evidence | Code cells 0, 1, 2, 3, 4 passed; historical visible outputs preserved |
| [stitching_pt11_gapfill](stitching_pt11_gapfill.ipynb) | Instrumented perf3 stage timing and gap-filling diagnostics | `data/ch5/hpc_perf3_diagnostics/` | Code cells 0, 1, 2, 3, 4, 5, 6, 7 passed; historical visible outputs preserved |
| [stitching_pt12_conditioning](stitching_pt12_conditioning.ipynb) | Provisional conditioned-gap pilot and error/recurrence overlap | Retained CSVs in `data/ch5/stitching_pt12_conditioning/`; raw upstream matching was not started | Code cells 0, 3, 4, 5, 6 passed; historical visible outputs preserved |
| [stitching_pt13_jerome_update](stitching_pt13_jerome_update.ipynb) | perf5 simulations/1KGP diagnostics and supervisor figures | `data/ch5/perf5_diagnostics/` manifest, quality, performance and stage tables | Code cells 0, 1, 2, 3, 4, 5, 6, 7 passed; historical visible outputs preserved |
| [stitching_pt2](stitching_pt2.ipynb) | Zero-coverage stitching, population summaries and outlier paths | `data/ch5/stitching_cohort/`, `chunk_matching_zero_coverage/` and copied `match_eval/` panel | Code cells 0, 1, 2, 3, 4, 5 passed; historical visible outputs preserved |
| [stitching_pt3_hyperparams](stitching_pt3_hyperparams.ipynb) | Saved AC/mismatch grid and aligned ARG diagnostics | Retained upstream grid and `data/ch5/stitching_pt3_hyperparams/{haplotype_results,region_lengths}.csv` | Code cells 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12 passed; historical visible outputs preserved |
| [stitching_pt4_perf](stitching_pt4_perf.ipynb) | First benchmark: correctness, elapsed time, stages and memory | `../tsinfer-match-eval/data/performance/first_perf_benchmark/` | Code cells 0, 1, 2, 3, 4, 5, 6, 7 passed; historical visible outputs preserved |
| [stitching_pt5](stitching_pt5.ipynb) | Greedy AC10/40 benchmark with shared Full HMM baseline | `../tsinfer-match-eval/data/performance/greedy_only_30mbp/` | Code cells 0, 1, 2, 3, 4, 5, 6, 7 passed; historical visible outputs preserved |
| [stitching_pt6_min_sizes](stitching_pt6_min_sizes.ipynb) | AC10/20 × minimum focal-size grid and genealogy/age diagnostics | `data/ch5/stitching_pt6_min_sizes/{haplotype_results,region_lengths,run_summary}.csv`; original raw products in sibling `stitching_eval/` | Code cells 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12 passed; historical visible outputs preserved |
| [stitching_pt7_paramsearch](stitching_pt7_paramsearch.ipynb) | Archived AF/minimum-size grid at three simulation sizes | `data/ch5/hpc_params1_diagnostics/` manifest, run summary and saved physical-minimum table | Code cells 0, 1, 2, 4, 5 passed; historical visible outputs preserved |
| [stitching_pt8_hpc_perf1](stitching_pt8_hpc_perf1.ipynb) | Simulation HPC performance and path/age diagnostics | `data/ch5/hpc_perf1_diagnostics/` | Code cells 0, 1, 2, 3, 4, 5 passed; historical visible outputs preserved |
| [stitching_pt9_hpc_perf2](stitching_pt9_hpc_perf2.ipynb) | 1KGP HPC performance and path/age diagnostics | `data/ch5/hpc_perf2_diagnostics/` | Code cells 0, 1, 2, 3, 4, 5 passed; historical visible outputs preserved |
| [tie_breaking](tie_breaking.ipynb) | Saved controlled youngest/last-tie experiments, without launching reproduction | `data/ch5/tie_breaking/` tables, diffs, toy cases and provenance | Code cells 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 passed; historical visible outputs preserved |

All four performance archives passed cohort/quality checks and their selected
performance plots; perf3/perf5 stage figures also ran. pt3/pt6 plot checks use
the retained per-haplotype/length tables. Dating, full ARG metric recomputation
and upstream inference/matching were not started merely to validate a move.
The old drivers' native `MatcherIndexes`/`AncestorMatcher` API is absent from
the current pinned tsinfer: reproducing those matching jobs requires their
original upstream revision/environment, while the saved-result analyses run.

pt1 explicitly selects `chunk_matching_experiment`, whose records contain
`accepted_site_intervals`; pt2 selects the separate zero-coverage archive with
`focal_site_intervals`. The earlier shared mutable YAML had blurred those
identities. pt6's historical grid (AC 10/20; minima 1/25/50/100/200; zero
mismatches) is initialised in its cells from the preserved run-summary and raw
filenames, because the mutable sibling config now describes a different run.
The original full pt6 pipeline YAML is unavailable; no newly invented run
configuration is presented as original provenance.

The original pt12 called-allele limitation and provisional interpretation
are preserved with its output-bearing pair in
`data/ch5/provenance/pt12_before_20261116/`. The active pt12 pair now contains
the corrected conditioning analysis described above. `tie_breaking` retains
its archived reproduction source as evidence beside the inputs; ordinary execution reads saved outputs
and leaves its explicit expensive-reproduction flag false.

The optional native tie-breaking command now uses the flat
`scripts/compare_tie_breaking.py` entry point, with preparation/workers and
scientific summaries in `ch5.nb_utils`. Its original source snapshots remain
immutable provenance, rather than active imported helpers. The explicit
`run_experiments=False` cell preserves ordinary saved-result execution.
See `scripts/README.md` for the historical native-API/compiler requirement.

Notebook plot DPI and spine visibility are explicit call keywords; no global
style-setting cell or notebook plotting config object is needed. The shared
benchmark panel constructor returns named figure/axes fields, matching the
grid-panel interface. All affected consumers were updated together.
`ancestor_coverage` reads shared truth through
`data/general/simulated_error_truth`, also used by chapter 4.

Older ignored autosave checkpoints are retained unchanged, with their visible
outputs and original source snapshots, in the committed
`data/ch5/provenance/notebook_checkpoints.zip` recovery archive. They are
historical records; use the directly tracked current notebook pairs above.

On 13 November 2026, all chapter notebook pairs were synchronised with Jupytext
at the user's request. Saved outputs, attachments and execution counts are
unchanged. Jupytext's own comparison and contents-manager load checks pass;
no notebook was executed during this synchronisation.

## Revised gap conditioning (pt14)

`stitching_pt14_condition_update.py` and its paired notebook use the saved
`conditioning_test_revised` and `conditioning_test_revised_control` diagnostics
from sibling `tsinfer-match-eval/data/`. Run from the analysis repository root
on BMRC. The notebook covers 500 and 5,000 diploid simulated individuals and
3,202 diploid 1KGP individuals across chr20p. It reads the recorded sample
parameters and checks the shared Full HMM paths and ancestor references. It
retains pt12's main path, root-edge, recurrence and allele-age analyses. The
path-weight formulas score the complete parent sequence and explicitly handle
switches at map boundaries with zero recombination probability.

The instrumented benchmark in `data/performance/conditioning_test_revised/`
of the matching repository contains one 186-worker trial per cohort, timing
scope and method for Full HMM and conditioned greedy. Pt14 compares packing
and matching wall time, native HMM thread CPU, private memory and per-haplotype
workload. The unconditioned control supplies scientific diagnostics. Tables,
dated trees and PNG/PDF figures go under
`data/ch5/stitching_pt14_condition_update/` in this analysis repository.
Running the notebook reads completed matching products and dates the saved
trees.

The saved notebook was executed on `cloudcomp113` with the existing analysis
environment in `../tmp/epsilon_testing_20261113/env`. All 12 code cells ran,
the paired sources agree and all 14 figures were inspected. The nine dated
trees yield common mutation sets of 85,960 (500 OOA), 92,167 (5,000 OOA)
and 306,996 (1KGP). The batch logs are in `../tmp/pt14_xyCrg6/`.

The path panel shows mean total copying switches per haplotype for all three
methods. The performance discussion compares saved perf5 and pt14 stage
timings and uses archived same-input controls to qualify the TSLS comparison.

The tree-count plot uses raw tree sequences and reports percentage change
relative to Full HMM for both greedy arms. ARF and TPR follow pt6's comparison
direction and metadata-based sample alignment, using each pt14 cohort's
observed extent and retaining unary ancestors. Both greedy arms are compared
with Full HMM across all cohorts, and all three methods are compared with
simulation truth for the two simulated cohorts. The metrics use tscompare's
node-span calculations with the chapter's exact bitset clade iterator.
The source tables are `tree_counts.csv` and `arg_metrics.csv` under the pt14
output directory.
The three new cells ran on `cloudcomp113`. All nine counts matched the
earlier raw-tree counts, all 12 ARF/TPR comparisons completed, and the three
new figures were inspected. The batch log is in `../tmp/pt14_arg_r19xnK/`.
