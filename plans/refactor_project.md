# Refactor the DPhil analysis workspace

## Scope

This is the revised plan, not an instruction to implement the refactor now.
The current change updates only this document and the root `AGENTS.md`.

Keep a simple personal research workspace with two distinct responsibilities:
rough notebook work, and reproducible, polished thesis outputs. Consolidate
exploratory code aggressively, delete the old test suite rather than archiving
it, and remove `experiments/` entirely. The chapter 3 performance pipeline has
already moved out of this repository; relocating it or recreating its runtime
is no longer part of this plan. Its exported results remain inputs to analyses.

Preserve existing user edits, scientific analyses, notebook narratives and saved
outputs. Remove superseded Python files and configuration files after their
required content and callers have migrated. Do not preserve obsolete structure
through compatibility modules, archived tests or another experiments directory.
This is an organisational refactor, not a reason to change scientific methods,
filters or conclusions. Notebook interfaces and plotting calls may change to
share an updated plotting function instead of preserving every old API.

## Target structure

Use chapter packages directly under `src/`, without an extra package hierarchy.
Keep notebook names and Jupytext pairs. Only create files with an actual use;
in particular, do not create empty chapter 4/5 thesis implementations.

```text
AGENTS.md
README.md
Snakefile                       # maintained thesis workflow
thesis_config.yaml              # thesis inputs, parameters and destinations
pyproject.toml
uv.lock
src/
  ch3/
    __init__.py
    nb_utils.py                 # all reusable exploratory preparation/analysis
    nb_plot.py                  # all notebook plotting
    thesis_utils.py             # final thesis preparation, analysis and statistics
    thesis_plot.py              # final thesis plotting
    steps.py                    # only if rule orchestration needs it
  ch4/
    __init__.py
    nb_utils.py
    nb_plot.py
    error_estimation.py         # retain the substantive method library
    error_validation.py         # retain the substantive validation library
    # thesis_utils.py, thesis_plot.py and optional steps.py when needed
  ch5/
    __init__.py
    nb_utils.py
    nb_plot.py
    # thesis_utils.py, thesis_plot.py and optional steps.py when finalised
notebooks/
  ch3/
    README.md
    figure_gen.py
    figure_gen.ipynb             # saved plots and other outputs retained
  ch4/
    README.md
    initial_testing.py
    initial_testing.ipynb
  ch5/
    README.md
    ...                         # all existing notebooks and any moved from experiments
scripts/
  README.md                     # purpose, associated chapter, inputs and commands
  extract_zarr_validation_data.py
  extract_phase_data.py
  chunk_matching.py             # existing standalone drivers, where still needed
  stitching_method_comparison.py
figures/
  ch3/                          # notebook exports within this analysis repository
  ch4/
  ch5/
data/
  general/                      # only inputs actually shared across chapters
  ch3/
  ch4/
  ch5/
tests/
  ch3/                          # a few correctness cases for final thesis functions
  ch4/                          # minimal correctness cases for retained method code
plans/
```

The thesis workflow's configured destination is the thesis repository, with
explicit generated outputs under `figures/chN/` and `statistics/chN.tex` there.
For example, with `thesis_root: ../dphil-thesis`, chapter 3 produces
`../dphil-thesis/figures/ch3/<figure>` and
`../dphil-thesis/statistics/ch3.tex`. A temporary thesis root can be used for
validation. Analysis notebook exports remain in this repository's `figures/`.

There is no `experiments/`, `archive/tests/`, notebook `_support/`, replacement
collection of exploratory modules, or local copy of the external performance
pipeline in the target layout.

## Notebook code: exactly two shared exploratory modules per chapter

Every chapter has `nb_utils.py` and `nb_plot.py`. These are rough work, even
when many notebooks use them, and must never have automated tests.

- Put all notebook figure construction, styles, axes changes, annotations,
  plot composition and figure-saving helpers in `nb_plot.py`. Notebook cells
  call these functions and display the results; they do not build another set
  of plots inline.
- Put all other reusable notebook functions in `nb_utils.py`: loading, data
  preparation, numerical summaries, caching, exploratory algorithms, config
  classes and importable multiprocessing workers.
- Notebook cells initialise settings, select inputs, orchestrate calls and show
  results. Reusable definitions belong in the two modules, rather than cells
  or new analysis-specific Python files.
- Notebook-facing chapter imports are `from chN import nb_utils, nb_plot`.
  The chapter 4 notebook uses this interface too; `nb_utils` delegates to the
  retained estimator/validation libraries without duplicating their algorithms.
  Direct use of ordinary third-party libraries for simple cell orchestration
  does not require wrapping their APIs.
- Reuse and adapt existing plotting functions. Prefer explicit arguments that
  select data, metrics or presentation over near-identical functions named
  after individual notebooks. Update older notebook calls together when a
  shared plotting API changes; do not keep compatibility shims.
- Keep the two modules readable with coherent functions, dataclasses,
  docstrings and sections. Their size alone is not a reason to split rough
  work back into many modules. `nb_plot` can use `nb_utils`; keep preparation
  independent of plotting.

Audit definitions and consumers before moving code, especially the chapter 5
chains between stitching, hyperparameters, dating, performance, gap filling,
conditioning, parent categories and ARG metrics. Consolidate at function level
and resolve name collisions explicitly. Retain the existing calculations and
scientific caveats while removing redundant implementations and dead code.

| Current source | Destination |
| --- | --- |
| `src/ch3_plot.py`, `src/ch3_ancestor_extension.py` | Notebook plotting to `ch3/nb_plot.py`, preparation/analysis to `ch3/nb_utils.py`; extract the final thesis subset separately as described below |
| `src/ch3_anc_perf.py`, `src/ch3_anc_perf_analysis.py` | Locally used result analysis and plotting to the appropriate chapter 3 notebook/thesis files; remove unused code, without bringing the external benchmark back |
| `src/error_estimation.py`, `src/error_validation.py` | `src/ch4/error_estimation.py` and `src/ch4/error_validation.py`; exploratory helpers and plots still belong in the chapter's `nb_` files |
| All `src/ch5_*.py` | Consolidate reusable non-plotting code into `src/ch5/nb_utils.py` and plotting into `src/ch5/nb_plot.py`; no separate maintained chapter 5 library or tests |
| Notebook-local reusable helpers and inline plotting | Move to the corresponding chapter's `nb_` files and replace cells with explicit calls |

Configure Hatch to install the `ch3`, `ch4` and `ch5` packages from `src/`.
Update all consumers and remove former top-level imports; no `sys.path` hacks
or aliases. Keep `__init__.py` files minimal. Document one repository-root path
and working-directory convention that works in Jupyter and when running paired
sources with `uv run python notebooks/chN/<name>.py`.

## Configuration and chapter documentation

Notebook configuration lives in notebook cells. Define a small dataclass in
`nb_utils.py` when useful and initialise it in the notebook with its actual
paths, datasets and scientific settings. Do not create a config framework.

Plotting parameters must be explicit keyword arguments at the notebook call
site: figure size, DPI, colours, labels, bins, limits, scales, quantiles and
other adjustable presentation choices. Do not hide them in YAML, a plotting
config object or a notebook-specific style preset. Keep scientific filtering
choices distinguishable from display limits. For example:

```python
from ch5 import nb_utils, nb_plot

config = nb_utils.AnalysisConfig(input_dir=data_dir, workers=None)
summary = nb_utils.load_summary(config)
figure = nb_plot.plot_performance(
    summary,
    metric="wall_time",
    figsize=(8, 4),
    dpi=150,
    colors={"full_hmm": "#555555", "greedy": "#0072B2"},
    xlabel="Diploid individuals",
    xscale="log",
)
display(figure)
```

This example describes the interface, not an API to introduce verbatim.
`thesis_config.yaml` is for the maintained thesis workflow; notebooks do not
read it or need to stay in sync with it. Standalone scripts use CLI arguments
and configuration classes where useful. Upstream run configs may remain as
provenance/data inputs, not as a new notebook-parameter layer.

Move the notebook-specific explanations in `experiments/ch5/README.md`,
including the current pt13 supervisor-update notes, into
`notebooks/ch5/README.md`. Preserve scientific qualifications, input identities,
methods, cache meanings and execution notes, while updating obsolete paths,
config references and claims about tests. Each chapter README should list its
notebooks, purpose, inputs, outputs, current/historical use and any missing
inputs or unverified execution. Preserve the root README's estimator guidance
in the chapter 4 README. Keep root documentation focused on setup, layout,
notebook use, scripts and thesis commands.

## Remove experiments and establish standalone scripts

Remove the entire `experiments/` directory after relocating its useful content:

1. Move any notebooks and paired sources to `notebooks/chN/`, preserving outputs.
2. Move genuinely standalone drivers to the flat `scripts/` directory. No
   chapter subdirectories or `ch3_`/`ch5_` filename prefixes. Reusable rough
   functions used by notebooks/drivers belong in the chapter's `nb_` files;
   scripts retain the actual command-line entry point and orchestration.
3. Move `src/ch3_extract_data.py` to
   `scripts/extract_zarr_validation_data.py` and
   `src/ch3_extract_phase_data.py` to `scripts/extract_phase_data.py`.
   Preserve their ability to be copied and run standalone on the data-holding
   machine, with their existing NumPy/pandas/Zarr requirements. Do not force
   these self-contained extraction scripts to import the notebook package.
4. Fold active notebook YAML settings into notebook initialisation and explicit
   plot arguments, then delete the superseded YAML. Transfer any settings
   still needed by standalone drivers to their CLI/configuration classes;
   do not move the old YAML collection into a renamed config directory.
5. Move data, cached results and figures out of `experiments/` into their owning
   chapter's data/figure directories without overwrites. Preserve upstream run
   configurations needed to identify results alongside those data.
6. Move documentation to chapter notebook READMEs or `scripts/README.md`, then
   delete the emptied directory and any obsolete entry points.

Update README commands, module docstrings, help text and notebook references
for every script rename. Describe the associated chapter in those strings,
rather than encoding it in filenames. Distinguish loading existing results
from starting expensive inference or dating jobs.

## Preserve notebooks and their visible outputs

Retain all existing notebooks, including chapter 3's broad `figure_gen` notebook
for trying plots. Keep its scientific scope, narrative and saved outputs while
migrating calls to `ch3.nb_utils`/`ch3.nb_plot`. Remove its thesis-directory
creation, copy/export helper, export calls and `#include` export mechanism.
It may still save exploratory figures locally. It is not the thesis build.

Before editing Jupytext pairs, inspect both files for unsynchronised changes
and preserve a copy of the output-bearing `.ipynb`. Update source/code cells
deliberately without clearing markdown, outputs, attachments or execution
counts. Do not bulk-sync using timestamps as the source of truth.

When the original inputs and required environment are available, run affected
cells and inspect the refactored figures/tables. Do not launch long upstream
experiments merely to check reorganisation. When inputs are unavailable, still
migrate the imports, calls and paths to the new organisation; preserve the
existing outputs exactly and record that they came from the earlier execution.
Missing historical inputs do not block the source migration, and preserved
outputs must not be reported as a successful rerun.

Replace the blanket `.ipynb` ignore with rules that allow preserving the
output-bearing chapter notebooks. Retain a readable, backed-up output-bearing
copy and document its durable location if an individual artifact is too large
for Git. A source-only `.py` file does not preserve visible results. Check saved
output/attachment hashes for notebooks not executed during migration. Do not
add notebook tests or mandatory HTML-report infrastructure.

## Reproducible thesis outputs with Snakemake

Use a root `Snakefile` and `thesis_config.yaml`. The notebook is neither an
execution dependency nor the maintained source of thesis output. Implement
chapter 3 first by working backwards from its nearly final plots, current
notebook analysis and the figures/numbers used in
`../dphil-thesis/tex/ch3_anc.tex`.

### Chapter functions and rules

- `src/ch3/thesis_utils.py` owns final preparation, analysis and all derived
  numerical claims. `src/ch3/thesis_plot.py` owns polished plotting functions.
  Keep responsibilities clear and functions documented; use the same pattern
  for other chapters when they have final outputs.
- Extract only the final subset from existing work. The maintained thesis
  modules must not import `nb_utils`, `nb_plot`, notebook files or exploratory
  scripts. A small initial separation of rough and final implementations is
  acceptable: subsequent notebook experiments must not change thesis results.
- Use a `ch3_plots` rule with an explicit list of final figure files as outputs,
  and a `ch3_statistics` rule producing `statistics/ch3.tex` beneath the
  configured thesis root. Add corresponding plot/statistics rules for chapters
  4 and 5 when their outputs are defined; no successful empty placeholder rules.
  A root target collects the implemented chapter outputs.
- Rules may call thesis functions directly when concise. If orchestration is
  elaborate, put it in `src/chN/steps.py` and have the rule call the relevant
  step with explicit inputs, outputs and parameters. Keep scientific logic in
  `thesis_utils`/`thesis_plot`, not a large Snakefile or collection of drivers.
- Use `thesis_config.yaml` for input identities/paths, output root, scientific
  settings, plot parameters, numerical precision and formatting. Explicit
  notebook plot arguments are a separate policy for rough work.

### Explicit data dependencies

Inventory the exact inputs for every maintained plot and numerical claim.
Chapter 3 currently uses aggregate simulation CSVs (including the different
demographies), exported Zarr validation summaries, phase-run exports, empirical
1KGP ancestor-span tables, the empirical DAF Zarr input or a sufficient retained
export, and the external benchmark's performance CSVs. Toy illustrations also
have explicit defining parameters. Distinguish generated plots from any
manually supplied illustrations; preserve source assets and existing TeX figure
names/formats.

Declare all consumed data in the appropriate rule's `input`, including any
per-dataset files selected from configuration. A directory path or unchecked
cache discovered inside a function is not a substitute for declaring the files
actually required. For Zarr stores, represent the complete store dependency
explicitly, using a declared store directory and its metadata/chunk dependencies
where needed to detect updates. Do not rely on hidden sibling-repository
lookups, notebook state, undeclared globs or an old `for_plots` cache.

If preparation writes derived tables, model it as a producing rule with its
own full inputs/outputs and make downstream rules depend on those outputs.
Track the relevant thesis source files as dependencies too, so changes to an
imported analysis or plotting function invalidate its generated outputs.
Routine thesis generation should use retained, identified analysis inputs;
it must not silently launch the external benchmark or upstream inference.
Keep plot and statistics calculations consistent by using `thesis_utils`
and, where useful, the same declared prepared tables.

### Chapter statistics

Audit the chapter's specific quantitative claims, not just numbers printed
by the notebook. Start with its existing numerical check tables: focal/ancestor
counts, span curves and frequency patterns, rare overshoot tails, demographic
comparisons, whole-overlap error rates, genotype-error summaries, empirical
site extents, and stage/total performance ratios. Add any quoted quantities
not covered there. Keep dataset/version selections, units, denominators,
aggregation order, rounding and haploid/diploid distinctions explicit.

Generate named LaTeX macros or another small, directly usable TeX interface in
`statistics/ch3.tex`, with names tied to the claims and comments identifying
source quantities and units. Calculate values from declared data, rather than
copying rendered notebook values or hard-coding thesis claims. Supply a clear
mapping from each quantitative claim to its generated value; fixed experimental
settings can come from the declared thesis configuration. Preserve existing
thesis prose when integrating generated values through mechanical TeX
references. No prose rewriting is part of the refactor.

Apply the same statistics rule and `statistics/chN.tex` convention as other
chapters mature. Missing required thesis inputs must fail the affected target
with a useful error; do not emit blank figures, placeholder statistics or claim
reproduction is complete. This stricter requirement applies to the thesis
workflow, while historical notebook migration can finish without rerunning.

### Thesis validation

Record the root lockfile, Python version and actual editable dependency
revisions; a moving sibling checkout is not a reproducible dependency. Keep
required input identities, producing commands/revisions and backup locations
in concise workflow documentation or an input manifest where needed.

Add Snakemake to the appropriate project dependencies during implementation.
Validate a dry run, then build chapter 3 into a temporary configured thesis
root. Compare the expected file list, formats and numerical results, and
visually inspect figures against the current thesis outputs. Check that
changing an input or relevant parameter rebuilds its outputs and that an
unchanged completed workflow has nothing to do. Use the workflow for thesis
exports; remove the old notebook-based selection/copy proposal entirely.

## Data moves

Retain the chapter ownership cleanup without adding a metadata framework:

- Keep chapter inputs and results under `data/chN/<dataset_or_analysis>/`.
  Move `data/ch5_stitching_*` under `data/ch5/` and migrate all data currently
  beneath `experiments/`. Use `data/general/` only for proven cross-chapter
  inputs such as genuinely shared genotype stores or maps.
- Inventory consumers before moving ambiguous loose files or caches. Preserve
  scientific run identifiers and third-party filenames; normalise container
  names rather than rewriting provenance.
- Move large stores as units. Check collisions before moving and compare file
  counts/sizes; use checksums for required thesis inputs and record external
  symlink targets. Do not merge by overwrite or deduplicate by similar names.
- Update live notebook, script, workflow and documentation paths together.
  Preserve historical run configs/logs as provenance rather than rewriting
  them. Keep a simple old-to-new path record while executing the migration.

Deleting tests and superseded code/configuration does not authorise deleting
scientific data, saved results or existing user work.

## Strict test policy

Delete the existing test suite from the repository during implementation,
including stale external-pipeline tests, notebook/plot tests and chapter 5
metric/algorithm tests. Do not archive, move outside discovery, disable, skip
or retain compatibility code for them. Git history is sufficient for old tests;
record any current uncommitted work before applying the planned deletions.

Then establish only the deliberately small suite justified by finalised code:

| Code | Automated tests |
| --- | --- |
| Every chapter's `nb_utils.py`, `nb_plot.py`, notebooks and rough script helpers | None, even when reused or numerically involved |
| Chapter 3 final `thesis_utils.py`/`thesis_plot.py` functions | A few small correctness cases for meaningful calculations used in final outputs |
| Chapter 4 retained estimator/validation libraries | Minimal known-answer/invariant checks only for finalised methods; no development or plotting suite |
| Chapter 5 | None; do not recreate or retain `tests/ch5/` |

Do not migrate the old suite wholesale under new names. A small number of
existing numerical assertions may inform the new final-code checks, but their
selection must be deliberate. No plot-label/artist-count tests, image snapshots,
notebook execution tests, coverage targets or framework-heavy fixtures. Keep
simple class-based pytest tests with fixtures and tiny synthetic inputs; no
personal datasets, HPC allocations or sibling checkout requirements. Configure
root pytest discovery to the small `tests/` suite. Visual inspection and source/
output-preservation checks are validation activities, not reasons to add tests
for rough work.

## Required root AGENTS.md content

Keep the existing Git usage, Code Style and Tool use sections, including the
explicit exception for notebook plot arguments instead of configuration files.
Git guidance must require plain, capitalised, descriptive commit titles without
category prefixes, and the exact phase titles below during implementation.
Retain the prohibition on `Co-authored-By` lines and the `uv run` requirement.

The following sections are the required final policy, not optional suggestions
or a link that future agents must discover inside an old plan. Put them directly
in the root `AGENTS.md`. During migration, add a short note that the layout is
the target and record which phases are complete. Remove that temporary note in
the final phase; the ownership and maintenance rules remain permanently.
Chapter READMEs document analyses and commands, not exceptions to root policy.

```markdown
## Repository structure

- This is a personal research and thesis workspace. These are the permanent
  ownership rules for the implemented structure and all future changes. Read
  this guidance and the relevant chapter README before editing. Superseded
  plans and historical layouts do not override these rules.
- Keep chapter packages directly in `src/ch3/`, `src/ch4/` and `src/ch5/`, with
  minimal `__init__.py` files. Import modules explicitly without compatibility
  aliases or `sys.path` edits.
- Each chapter has exactly two shared exploratory modules: `nb_utils.py` for
  reusable loading, preparation, analysis, configuration classes, caching and
  multiprocessing workers; `nb_plot.py` for all notebook plotting, styling,
  composition, axes changes and figure-saving helpers. These are rough work.
- Notebook-facing chapter code must use the corresponding `nb_utils` and
  `nb_plot` modules. Cells initialise settings, orchestrate calls and display
  results. Move reusable definitions into those modules; do not add separate
  exploratory source files or notebook-local `_support` modules.
- Reuse and adapt plotting functions across a chapter's notebooks. Update older
  calls when a shared plot changes instead of adding near-identical functions
  or retaining old APIs solely for historical notebooks.
- Retain the chapter 4 estimator and validation libraries under `src/ch4/`.
  Its notebook helpers delegate to those libraries without duplicating methods.
- Keep notebooks and Jupytext pairs in `notebooks/chN/`. Put notebook-specific
  methods, input/output descriptions, run notes and caveats in that folder's
  `README.md`. Read it before changing the chapter's notebooks.
- There is no `experiments/` directory. Put notebook work, data, documentation
  and standalone drivers in their owners above. Do not recreate an experiments
  collection under another name.
- Put standalone entry points in a flat `scripts/` directory, with descriptive
  names and no chapter prefixes or chapter subdirectories. Use
  `extract_zarr_validation_data.py` and `extract_phase_data.py` for the existing
  extractors. State the associated chapter, inputs and invocation in script
  docstrings/help and `scripts/README.md`. Keep these extractors self-contained
  for copying to the data-holding machine.
- Keep chapter data in `data/chN/`, truly shared inputs in `data/general/`, and
  local notebook figure exports in `figures/chN/`. Do not overwrite earlier
  scientific runs or discard data/results as incidental cleanup.
- The chapter 3 performance pipeline already lives outside this repository.
  Consume its exported results; do not recreate its workflow or environment here.

## Notebook configuration and saved outputs

- Initialise notebook data paths and scientific parameters explicitly in cells,
  using a small configuration class from `nb_utils` where useful. Do not create
  standalone configuration files simply to parameterise exploratory notebooks.
- Put adjustable plot choices directly in the notebook's plot-call keyword
  arguments: sizes, DPI, colours, labels, bins, axis limits/scales, quantiles and
  other presentation settings. Do not hide them in YAML, large config objects
  or notebook-specific style presets. Keep scientific filters distinct from
  display limits.
- Preserve notebook narratives, scientific caveats, outputs and attachments.
  Inspect both Jupytext files for unsynchronised changes before editing; never
  clear outputs or bulk-sync just to tidy the repository.
- If original inputs are available, run relevant cells and inspect the resulting
  plots/tables. Do not launch expensive upstream jobs merely to verify a move.
- If inputs are unavailable, still update imports, calls and paths to the new
  organisation while preserving the saved outputs. Record that the source was
  migrated without rerunning and that visible outputs are from the earlier run.
- Keep output-bearing chapter notebooks in Git where practical; otherwise
  record their durable backup location. A paired source file alone does not
  preserve the visible research record.

## Maintained thesis workflow

- Use the root `Snakefile` and `thesis_config.yaml` for thesis generation. Provide
  chapter plot targets and a statistics target producing `statistics/chN.tex`
  under the configured thesis root. Declare final figure files individually.
- Put polished final plotting in `src/chN/thesis_plot.py` and final preparation,
  analysis and statistics in `src/chN/thesis_utils.py`. Use `src/chN/steps.py`
  only when orchestration is too elaborate for a concise rule call.
- Maintained thesis modules must not import notebook files, `nb_` modules or
  exploratory scripts. Extract the final subset rather than promoting all rough
  work. Add later chapters' thesis files/rules when outputs are actually defined.
- Declare every required data dependency in rule inputs and every generated
  artifact in rule outputs. Model derived-table preparation as rules; do not
  depend on hidden caches, notebook state or undeclared sibling-repository data.
  Track the relevant thesis source files so helper edits trigger regeneration.
- Compute every quoted derived chapter quantity from declared inputs and emit
  named TeX values with clear units, aggregation, formatting and claim mappings.
  Experimental settings may come from thesis configuration. Preserve thesis
  prose when integrating generated values through mechanical TeX references.
- Start from the nearly final chapter 3 figures and numerical analyses. Preserve
  current thesis figure basenames/formats and record required data identities,
  producing commands, dependency revisions and backup locations.
- Keep chapter 3's figure notebook for exploration, but remove its writes/copies
  to `dphil-thesis`. Notebooks may evolve independently of the maintained
  workflow and do not have to stay in sync with it.
- Validate relevant thesis changes with a dry run, generation into a temporary
  destination, numerical checks and figure inspection. Missing required thesis
  data must fail the target clearly; do not emit blanks or placeholder numbers
  or claim complete reproduction when inputs are missing.

## Testing

- Organise tests in classes, not flat functions. Use pytest fixtures for setup.
- The old analysis test suite has been deleted. Do not restore or archive it,
  move it outside collection, skip it or keep compatibility code for it. Git
  history is the reference for retired tests.
- Tests are only for finalised code: a few meaningful correctness cases for
  functions used by the final thesis plots/statistics and minimal chapter 4
  estimator/validation checks. Do not migrate the old suite wholesale.
- Never test any chapter's `nb_utils.py`, `nb_plot.py`, notebooks or other rough
  work, regardless of reuse. Keep no chapter 5 tests, including metric or
  algorithm tests. Do not create a chapter 5 suite.
- Use tiny synthetic known-answer cases or scientific invariants. Do not test
  plot labels, artist counts, styling, exact image/PDF bytes or notebook execution.
  Do not add coverage targets, generic test frameworks or personal-data/HPC
  requirements. Configure discovery to the small maintained `tests/` suite.
- Checking source references, preserving outputs and visually inspecting plots
  are the appropriate validation for rough work; they do not justify adding tests.
- Report what changed, what was checked and what could not be executed.

## Maintaining the structure

- Treat the ownership rules above as the continuing repository contract, not
  just migration advice. Before adding code, choose its existing owner:
  notebook plotting in `nb_plot`, other rough helpers in `nb_utils`, final
  thesis work in `thesis_plot`/`thesis_utils`, retained chapter 4 methods in
  their libraries, and standalone entry points in `scripts/`.
- Do not add a source module because rough code has become long, a helper has
  several consumers, or an old implementation used separate files. Do not
  recreate experiments, exploratory YAML, notebook support modules, compatibility
  wrappers or forbidden tests. A different structure requires an explicit user
  request; convenience is not an exception to these rules.
- Before changing a shared notebook helper, find every affected caller across
  all chapter notebooks, paired sources and scripts. Update them together,
  preserving saved outputs where inputs are unavailable. Do not leave an older
  notebook on a removed API or ask the user to identify its consumers.
- Update thesis functions, rule dependencies, configuration, output declarations
  and generated-statistic mappings together when their contract changes.
  Notebook experiments remain independent of the maintained thesis build.
- Keep chapter and script READMEs current in the same change. Search active
  source, notebooks, scripts, packaging, workflow and documentation for obsolete
  imports, paths and commands before finishing. Historical run logs/configs and
  superseded plans may record old names; they are not implementation guidance.
- For a requested refactor, complete all specified phases and their acceptance
  checks without stopping for routine decisions or phase-by-phase approval.
  This does not authorise starting the refactor during a planning-only task.
- Record phase completion, commit hashes, validation and genuine blockers in
  the plan's execution record. Continue independent work when data are missing.
  Do not describe a partial migration, placeholder rule or unresolved required
  thesis build as complete, or leave source/call-site cleanup for the user.
- Before handover, check the final file layout, every affected consumer, saved
  outputs, allowed test collection and relevant thesis targets against the plan.
  Perform these checks as task validation; do not add an architecture framework
  or tests for rough work to enforce them.
```

## Implementation phases and commit titles

This remains a planning-only change: do not execute these phases or make their
commits now. Once implementation is explicitly requested, carry the complete
plan through all phases without asking the user to choose routine destinations,
find callers or approve each phase. Use judgement within the stated structure;
only a genuinely missing scientific decision or unavailable required input
justifies a blocker. Continue all independent work and record the exact gap.

Use one local commit per phase with the exact title shown below. These follow
the repository's existing style, for example `Reorganise notebooks into
subfolders` and `Remove ch3_anc_perf pipeline from repo`; do not add category or
phase-number prefixes. Each commit must include its affected consumers and
documentation, not leave broken imports or missing call-site updates for the
next commit. Preserve pre-existing user work and do not bundle unrelated edits.
Notebook preservation applies from the first notebook edit, not just phase 5.

### Phase 1: establish the rules and inventory

Commit: **`Document the repository structure and refactor plan`**

- Install the required root guidance above, with the temporary migration-status
  note. Inventory all source definitions/consumers, scripts, tests, experiment
  contents, data destinations and chapter 3 thesis figures/quantitative claims.
- Inspect Git changes and both sides of every Jupytext pair. Preserve ignored
  output-bearing notebooks and scientific artifacts before any destructive move;
  record output/attachment hashes and protect uncommitted work.
- Keep the inventory and progress in this plan's execution record or simple
  tables linked from it. Do not introduce a metadata service or framework.

Complete when every current analysis, source file and experiment artifact has a
destination or a specific recorded scientific/data question. Existing user edits
and saved notebook results are recoverable. Uncertainty about a historical input
does not leave its code destination undecided.

### Phase 2: delete the old suite

Commit: **`Remove old analysis tests`**

- Delete all existing test files and obsolete test-only fixtures/configuration,
  including chapter 5 and external-pipeline tests. Record the removed files.
- Do not archive, skip or relocate them, and do not repair scientific code to
  satisfy them. New final-code tests come in phase 7 after their owners exist.

Complete when no old test suite or alternate collection remains in the working
tree. This includes untracked old tests identified in phase 1, after protecting
their current user changes. An intentionally absent suite at this intermediate
phase is not a claim of successful test execution.

### Phase 3: consolidate chapter code and update every notebook

Commit: **`Reorganise chapter code and notebook helpers`**

- Create/install the direct chapter packages. Consolidate every exploratory
  function into the appropriate `nb_utils` or `nb_plot`, including functions
  currently defined in notebook cells. Retain the chapter 4 method libraries.
- Update all chapter notebook pairs and other current callers atomically.
  Adapt older calls to shared plotting APIs, preserve scientific behaviour and
  outputs, and delete superseded modules rather than adding aliases.
- Put notebook scientific settings in initialised objects/cells and plot
  parameters in explicit call arguments. Remove notebook writes to the thesis.
  Update still-existing experiment drivers as consumers so they remain valid
  until moved in phase 4.

Complete when each chapter has exactly its two rough modules, every notebook
uses those chapter interfaces, and no consumer imports a removed module.
Packaging/import checks pass without executing analysis bodies. No reusable
notebook definitions or inline figure-building code remain outside the `nb_`
files; saved outputs for nonexecuted notebooks match their recorded hashes.

### Phase 4: move scripts and remove experiments

Commit: **`Move standalone scripts and remove experiments`**

- Move and rename standalone entry points, including both extractors, into the
  flat `scripts/` directory; update CLI help, docstrings, invocations and imports.
- Move any experiment notebooks, documentation, scientific data and useful
  results to their declared owners. Apply the inventory-backed chapter data
  moves and update all live references without rewriting historical run logs.
- Delete superseded notebook YAML after its settings have moved to notebook
  cells. Move standalone driver settings to CLI/config classes and delete the
  complete `experiments/` tree. Do not create a replacement config collection.
- Update chapter/script READMEs in this commit for moved content, including
  notebook-specific explanations currently in the experiments README.

Complete when `experiments/` is absent, all useful artifacts are accounted for,
and no active import, config load or command points there. Scripts have no
chapter prefixes/subdirectories, both extractors remain independently runnable,
and moved data counts/sizes and required checksums match the inventory.

### Phase 5: secure the visible notebook record

Commit: **`Preserve notebook outputs and document chapter analyses`**

- Allow output-bearing chapter notebooks through `.gitignore` and preserve the
  paired notebooks/results in Git or documented durable storage where too large.
- Finish all chapter README entries, including missing inputs, output locations,
  scientific caveats and whether each migrated notebook was actually rerun.
- Run affected cells when the original data and environment are available and
  inspect their figures/tables. Otherwise verify migrated source references and
  exact saved-output preservation. Record evidence per notebook, not merely
  that one representative notebook succeeded.

Complete when every existing notebook, including any moved from experiments,
has an updated pair and a readable preserved output record. Missing source data
has not left old imports, old plotting calls or an unfinished source migration.

### Phase 6: implement the chapter 3 thesis workflow

Commit: **`Add the CH3 thesis plot and statistics workflow`**

- Extract the final chapter 3 subset into polished thesis modules; add the root
  Snakefile/config and optional steps with every input, preparation dependency,
  final figure output and generated statistic declared.
- Generate `statistics/ch3.tex` for all quoted derived quantities and document
  claim mappings, exact input identities, units, precision and dependency
  revisions. Preserve figure names/formats and thesis prose.
- Validate a dry run and generation into a temporary thesis root, compare
  numerical results and inspect figures. Check rebuilds after relevant input,
  parameter and helper-code changes, and no work after an unchanged build.
- Document later chapters' use of the same conventions without adding speculative
  outputs or successful empty rules. Keep the notebook independent of this build.

Complete when chapter 3 has real plot/statistics targets and no dependence on
notebook state or rough modules. Required missing data is recorded as a thesis
reproduction blocker, not replaced with blank results; complete all implementable
rules/functions and independent structural changes even if a build is blocked.

### Phase 7: add only the minimal final-code checks

Commit: **`Add minimal tests for finalised chapter code`**

- Add the few selected correctness cases for final chapter 3 thesis calculations
  and minimal final chapter 4 method checks. Use small synthetic known answers,
  class-based tests and fixtures. Do not recreate the former suite.
- Configure discovery to these tests only. Inspect collection for rough-code,
  notebook or chapter 5 coverage, then run the entire small maintained suite.

Complete when the maintained checks pass without personal data/HPC requirements,
all tests have final-code owners, and no repository-owned chapter 5, `nb_` or
presentation tests remain under any name or location.

### Phase 8: finish the reorganisation and permanent guidance

Commit: **`Complete the repository reorganisation`**

- Audit all files and consumers against the inventory and final structure,
  including ignored/untracked artifacts. Remove any remaining superseded
  modules, helpers, imports, YAML, aliases or stale active commands.
- Finish root/chapter/script documentation with exact working-directory, setup,
  notebook, extraction and Snakemake commands. Keep analysis-specific guidance
  in chapter folders and ensure root documentation points to it.
- Install the final root policy verbatim, remove migration-only target wording,
  and leave the permanent maintenance instructions in place. Record each phase's
  commit hash, evidence and any genuine data blockers below.

Complete only after the final acceptance audit below; this commit is substantive
cleanup and handover, not a label used to conceal unfinished earlier phases.

## Final acceptance audit and ongoing ownership

The implementing agent performs and records every check below. Do not leave
verification or caller cleanup as instructions for the user or a later agent.

- Every original source/experiment artifact has a recorded migration outcome.
  All chapter notebooks call their corresponding `nb_` modules; pairs and visible
  outputs are preserved. Available-data reruns and nonexecuted migrations are
  distinguished per notebook.
- The source layout matches this plan. Chapter 4 method libraries remain.
  No `experiments/`, `_support`, extra rough-work modules, compatibility wrappers,
  old imports, notebook plotting YAML or notebook thesis exports remain.
- All standalone entry points are in flat `scripts/` with the specified extractor
  names, updated help/docstrings and documented commands. Data moves preserve
  scientific artifacts and leave no unresolved live path references.
- Old tests are deleted rather than archived or disabled. Only the minimal
  final-code suite exists; collection and its results are recorded. There are
  no chapter 5 or `nb_` tests under alternative names/locations.
- Chapter 3 thesis plots and all quoted derived numbers have declared targets,
  inputs and source dependencies. Generation, numerical/visual checks and
  incremental rebuild behaviour are evidenced, or the precise missing input
  and blocked target are recorded. A partial required thesis build is not a
  completed reproduction claim.
- Root `AGENTS.md` contains the permanent policy in this plan's required-content
  section, rather than relying on the plan to enforce it. README commands match
  the implemented layout. Historical plans/logs can retain old names but do not
  override the current root guidance.

Structural reorganisation is complete only when every structural check passes;
missing notebook inputs do not excuse unmigrated code. Full thesis reproduction
additionally requires the real chapter 3 build to pass. If that is blocked,
report these two outcomes separately and name the missing input and affected
target. Do not silently defer code, documentation or consumer cleanup.

## Execution record

Leave this table planned during the current documentation-only task. During
implementation, update it in each phase commit with actual evidence and the
preceding phase's commit hash. In phase 8, use `this commit` for its own hash
cell and report the exact final hash in the handover; a commit cannot include
its own hash. Keep any remaining data/blocker details with the relevant phase
and chapter README. Do not add an ongoing tracking framework or leave the final
table as an unverified checklist.

| Phase | Exact commit title | Status | Commit | Evidence or precise blocker |
| --- | --- | --- | --- | --- |
| 1 | Document the repository structure and refactor plan | Complete | a1334c7 | [Inventory](refactor_inventory.md), hashes and external recovery archive; read both sides of all 19 pairs and thesis claims. |
| 2 | Remove old analysis tests | Complete | 8b0caa3 | Removed every old test and checkpoint/cache, including protected untracked CH5 test; original list in inventory. No suite is claimed at this intermediate stage. |
| 3 | Reorganise chapter code and notebook helpers | Complete | 77a8681 | Direct Hatch packages installed; 297 source definitions and all notebook definitions assigned to owners; all 19 pairs migrated with exact visible-record hashes. Import, reference, compilation and call-signature checks pass. Shared grid/benchmark geometries consolidated. CH3 toy cells and perf5 first eight code cells ran separately; figures visually inspected. |
| 4 | Move standalone scripts and remove experiments | Complete | b04a5f6 | Four flat scripts; exporter/driver help passes; reusable workers in CH5 nb_utils. Moved 22 data units with equal counts/bytes and verified 13,895 experiment scientific files by SHA-256. All four recorded symlinks resolve. experiments is absent; active references and docs updated. |
| 5 | Preserve notebook outputs and document chapter analyses | Complete | 2ec585f | All 19 pairs/visible records preserved and allowed by ignore rules; every notebook has inputs, methods and explicit selected-cell evidence. 19 selected checks passed in isolated temporary destinations, using multiprocessing queues; representative figures inspected. No upstream inference/dating jobs launched. |
| 6 | Add the CH3 thesis plot and statistics workflow | Complete | this commit | Independent final subset; six producing preparation jobs, 23 individually declared figures and named TeX/statistics outputs. Temporary and configured-destination builds passed; nine retained numerical tables match, all figures inspected, incremental and missing-input checks passed. Mechanical thesis integration committed as 691af19 in dphil-thesis. See thesis_inputs.json, thesis_claims.md, thesis_validation.json and refactor_validation.md. |
| 7 | Add minimal tests for finalised chapter code | Planned | — | — |
| 8 | Complete the repository reorganisation | Planned | — | — |
