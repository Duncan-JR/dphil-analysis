# Zero-coverage focal stitching

`chunk_matching.py` reads the copied reference, panel, sample calls, and focal
candidates in `chunk_matching_config.yaml`. It reads anchored focal intervals
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

Run from the repository root using the existing native tsinfer environment:

```sh
PYTHONPATH=src uv run --no-project \
  --python ~/work/tsinfer-match-eval/.venv/bin/python \
  python experiments/ch5/chunk_matching.py \
  --config experiments/ch5/chunk_matching_config.yaml
```

A null `haplotypes` selection evaluates the complete focal roster. A null
`workers` value uses all available CPU cores, with queued whole-haplotype jobs
and reusable matcher indexes in each worker. Results stream to separate full
and stitched JSONL files and `comparison.csv`, including population labels,
switches within focal coverage, within gaps and at seams, total extra switch
fraction, and four site-weighted parent-agreement fractions. Agreement fractions
use all inference sites; extra switch fraction uses stitched switches.

Open `notebooks/ch5/stitching_pt2.ipynb`, or generate it from the tracked source:

```sh
uv run jupytext --to notebook notebooks/ch5/stitching_pt2.py
```

The notebook reads the existing results, displays selected-haplotype diagnostics,
and builds an 800-row table covering both methods for the two 200-haplotype
cohorts. Its final section exports tables and population plots under the
configured cohort output directory. Error bars show one sample standard
deviation across haplotypes. Both haplotypes of each diploid individual are
included, with three population groups per dataset.

## Fixed-pipeline hyperparameter analysis

`notebooks/ch5/stitching_pt3_hyperparams.py` (paired with `.ipynb`) reads the
current `tsinfer-match-eval/config.yaml` through
`stitching_pt3_hyperparams_config.yaml`. It compares all configured AC cutoffs
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
uv run jupytext --sync notebooks/ch5/stitching_pt3_hyperparams.py
```

Outputs go to `data/ch5_stitching_pt3_hyperparams/`. Independent parameter runs
use worker processes across available CPU cores; set `workers: 1` in the
analysis config for sequential loading. The notebook's run summary records
empty regions and undefined percent increases excluded from the means.
`ts_statistics.csv` records raw output counts, `ts_statistic_differences.csv`
records signed stitched-minus-full counts, and `arg_metrics.csv` records all
16 stitched-to-full comparisons, including reference method and time units.
`arg_metrics_against_truth.csv` records the nine simulated outputs compared
with truth. Metric tables and plots contain ARF and TPR only.
