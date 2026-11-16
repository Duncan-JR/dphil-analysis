# Chapter 4 error estimation

`initial_testing` records validation, doubleton classification, fitted error
models and empirical 1KGP/pig analyses. Run from the repository root. Notebook
calls go through `ch4.nb_utils` and `ch4.nb_plot`; estimator and validation
methods remain in their own chapter libraries. Settings are initialised in
cells. Saved outputs are preserved; full upstream estimation was not rerun
as a migration check. Original simulation stores are external inputs.

The estimator assumes complete, phased, fixed-ploidy, biallelic 0/1 genotypes
on one sequence, with sorted, unique variant positions. It uses observed
doubletons with complete windows and global diversity from the same store.
Recombination is supplied explicitly, for example as a HapMap file with
position and rate in columns 1 and 2 (zero-indexed).

Example script caller:

```python
from ch4 import error_estimation


if __name__ == "__main__":
    config = error_estimation.EstimationConfig(
        window_sizes=[1_000, 5_000, 10_000, 50_000, 100_000, 250_000],
        num_doubletons=10_000,
        random_seed=42,
        exclude_high_mismatch_proportion=0.25,
        L_mismatch_trim=1_000,
    )
    result = error_estimation.estimate_error_rate(
        "data.zarr",
        recombination="genetic_map.txt",
        config=config,
    )
    print(result.fit.epsilon)
    print(result.fit.success, result.fit.message)
```

The main guard supports multiprocessing; set `num_workers=1` for serial use.
`epsilon` means additive errors per haplotype-bp, not a per-genotype probability.
Before fitting, the estimator ranks doubletons by their mismatch counts at
`L_mismatch_trim` and excludes the configured highest-count proportion from
every window. `L_mismatch_trim` defaults to the smallest configured window. The
full mismatch arrays remain available in the result. Pass
`fixed_doubletons_path="doubletons.csv"` to use a fixed set created with
`ch4.error_validation.write_doubletons_csv()` instead of sampling from the input
Zarr.

Pass `variant_mask_name="variant_mask"` to exclude sites where that boolean
Zarr array is true. A sequence of mask names excludes the union of their true
values. Doubleton ascertainment, diversity, region span, mismatch counts and
recombination summaries all use the resulting included sites.


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
| [initial_testing](initial_testing.ipynb) | Estimator validation, doubleton classification and empirical error fits | `data/ch4/validation/dphil-analysis-results.csv`; configured simulation/1KGP/pig stores and recombination maps | Code cells 0, 1 passed; retained fitted-result figure rendered; historical visible outputs preserved |

The complete doubleton sampling, classifier cross-validation and fitting runs
were not restarted for the reorganisation. The original inputs are explicitly
selected in cells. The retained numerical result table was loaded and plotted;
that does not claim recomputation of the fitted values. New final-library
correctness checks use synthetic data, not personal stores.

The shared error-evaluation stores and simulated truth are exposed through
`data/general/zarr_vcfs/sim` and `data/general/simulated_error_truth`. Their
external symlink targets are recorded in `plans/refactor_data_symlinks.json`.
Older ignored autosave checkpoints, including their original outputs and
unsynchronised source snapshots, are retained unchanged in the committed
`data/ch4/provenance/notebook_checkpoints.zip` recovery archive. Current work
uses the directly tracked paired notebook above.

On 13 November 2026, all chapter notebook pairs were synchronised with Jupytext
at the user's request. Saved outputs, attachments and execution counts are
unchanged. Jupytext's own comparison and contents-manager load checks pass;
no notebook was executed during this synchronisation.

## Epsilon sanity check on BMRC (13 November 2026)

`epsilon_testing.ipynb` and its percent-format pair compare inferred-ancestor
matching for the 50- and 500-individual `tsinfer-anc-eval/data/anc_perf` stores.
Run on `cloudcomp113`; no true ancestor panel is used. `ch4.nb_utils` owns
preparation, matching workers and classifier summaries; `ch4.nb_plot` owns the
figures. The initial epsilon grid is `1e-4, 3e-5, 1e-5, 1e-6`.

Epsilon is additive errors per haplotype-bp. The constant biallelic emission is
`epsilon * 28_100_000 / source_site_count`, using the simulated interval rather
than the full chromosome length in the Zarr header, and counting source sites
before ancestor filtering. Ancestors are matched once with stock recombination
`0.01` and mismatch `1e-20`, with no map or epsilon. Every sample run uses that
same fixed ancestor ARG; only sample matching uses the calibrated emission and
the simulation's HapMapII_GRCh38 map. The new TSLS model uses a fixed haploid
cohort size (100 or 1,000) throughout. The cached
three-column map is integrated to the cumulative-map format required by tsinfer;
the two maps are compared at every source position.

In these stores, `call_genotype_error_mask=True` means **unchanged**, so false
entries identify injected errors. Preparation checks the inverted mask against
`variant_genotype_error_count` and the actual differences from the simulated
genotypes. The simulated tree sequence contributes biological recurrence labels
(two or more mutation records, including back mutations). A truth-positive site
has an error or biological recurrence; a predicted-positive site has two or more
mutation records in the inferred ARG. The primary classifier uses inference
sites. The covered-source view additionally includes sites augmented by parsimony. Four
source sites in the 50-individual store lie outside the inferred flanks; these
are reported as omitted. Every inference site is included.
This is site detection, not event or allele localisation. Error-only and
biology-only recall are shown separately.

The notebook records ROC operating points, precision–recall, F1, MCC,
balanced accuracy, confusion counts, recurrence percentages, conditional mean
mutation counts (including the first event), mutation counts on samples versus
internal nodes, and runtime. Epsilon changes the inferred ARG, so these are not
thresholds on one fixed ranking and no interpolated AUC is reported. Optima use
one seed per cohort and are provisional tuning results, not held-out estimates.

Remote run artifacts are under
`data/ch4/epsilon_testing/run_20261113_exact_samples/`; figures are under
`figures/ch4/epsilon_testing/run_20261113_exact_samples/`. Each epsilon has its own
sample-matched ARG, predictions, matching log and metrics. Shared stock-matched
ancestor ARGs and their parameters/logs are retained separately in each dataset
folder. Four processes use eight matching threads each, with a 64 GiB cache
budget per process. Completed metrics are reused
on rerun; choose a new output directory after changing scientific inputs or
implementation. The output-bearing notebook remains the readable Git record.

The dedicated runtime is
`/well/kelleher/users/uuc395/tmp/epsilon_testing_20261113/env/`.
It builds TSLS `e4cf233` and tsinfer `e2f8de7` in private worktrees, with the
recorded tsinfer patch passing a per-source `sample_size` into every TSLS HMM.
The parent's checkouts and environments are unchanged. The `epsilon_testing`
kernel in this environment is required; the repository's normal dependency pins
still select the earlier TSLS. Exact revisions, paths and patch are saved in
`runtime.json` and `tsinfer_sample_size.patch` with the run. Temporary setup,
execution and server scripts/logs are retained under the same remote `tmp/`
folder. `loadpy` must precede all compute shells; Python tooling is invoked via
`uv run --no-project --python .../env/bin/python`.

JupyterLab runs in cloudcomp113 tmux session `epsilon_jupyter_8886_20261113`,
port 8886, rooted at `/well/kelleher/users/uuc395`. With the usual BMRC forwarding,
the notebook is at
`http://localhost:8886/lab/tree/dphil-analysis/notebooks/ch4/epsilon_testing.ipynb`.
Keep normal server authentication enabled.

Ancestor generation excludes observed derived singletons, following
`tsinfer-match-eval/lib/utils.py`. Counts are recomputed from noisy genotypes in
an analysis-owned Zarr v3 view (`ancestor_input.zarr`), without modifying the
source stores or using their pre-error singleton masks. Path compression is
disabled, matching that pipeline. The source-site density used to calibrate
epsilon still includes singleton sites. The primary HMM universe excludes them;
the secondary all-source view restores them through augmentation.

The earlier unfiltered stock-ancestor attempt is retained under
`run_20261113_stock_ancestors`: its 50-individual sample runs gave zero recurrent
calls for every initial epsilon because singleton ancestors allowed copying
sample observations. It is excluded from tuning. An earlier mapped-ancestor
attempt (`run_20261113_sample_n`) likewise supplies no valid comparison results.
Diagnostics are retained under the remote task `tmp/` directory. TSLS was
restored and rebuilt without diagnostic changes before the corrected runs.

The final sample grid extends down to `1e-10` because the initial larger-panel
optimum lay at the lower edge (`1e-6`). Sample matching uses
`likelihood_threshold=0`, disabling likelihood flooring while preserving exact
zero recombination probabilities. The default floor caused reproducible
traceback errors on the smaller panel; no TSLS C patch or recombination floor
is used. Ancestor matching keeps its stock numerical defaults. Per-run JSON
records the sample HMM setting and thread count.

The final run reuses its Zarr views and stock-matched reference ARGs by symlink
from `run_20261113_nonsingletons`; retain that preparation directory alongside
`run_20261113_exact_samples`. Preliminary larger-panel results with the default
floor remain in the preparation directory and are excluded from the final
comparison. All final sample ARGs are rebuilt without flooring. Exact runtime
dependencies, genotype-reconstruction checks and notebook-output checks are
retained with the final run.

The completed ten-value sweep (20 sample ARGs) selects `epsilon=1e-9` by F1
for both cohorts, corresponding to sample emissions `1.93e-7` and `8.84e-8`.
Precision/recall are 0.744/0.663 and 0.799/0.810 on inference sites. The
notebook compares equal-weight mean scores and notes the broad optimum and
higher-recall `1e-6` alternative. This tuning result is not a new physical
error-rate estimate; it uses one seed per cohort and inferred noisy ancestors.

Validation completed on cloudcomp113: all 20 inferred ARGs reproduce the noisy
sample genotypes exactly at covered source positions; all cohort sizes, sample
HMM parameters and confusion-count identities were checked. All 11 notebook
code cells executed without errors, five plot outputs are embedded, and the
Jupytext pair agrees. The final classifier, recurrence and truth-component
figures were visually inspected. `git diff --check` passes. The latest TSLS
model/recombination checks passed (220 cases); no rough-work tests were added.
