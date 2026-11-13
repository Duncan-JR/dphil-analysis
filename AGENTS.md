# AGENTS.md

## Git usage

- Don't include Co-authored-By lines in git commits.
- Use plain, capitalised, descriptive commit titles in the repository's existing
  style, such as `Reorganise notebooks into subfolders`. Do not use `feat:`,
  `fix:`, `chore:` or other category prefixes. During the refactor, use the
  phase titles specified in `plans/refactor_project.md`.

## Code Style

- Do not be overly defensive - defend only against circumstances that can
  occur within the current codebase.
- Keep scientific settings and paths in configuration objects or explicit CLI
  arguments. Initialise notebook configuration in notebook cells. Pass notebook
  plotting parameters as explicit keyword arguments in plot calls, not through
  YAML or a plotting config object. Maintained thesis settings belong in
  `thesis_config.yaml`.
- Do not make production code more complex for the sake of minimising
  changes to the test suite. Simplicity and clarity of the production code
  is imperative.
- Do not combine multiple complex operations in a single statement. Prefer
  to keep a single operation per statement, and use intermediate variables
  as a form of documentation. For example:
  ```python
  # Bad — multiple operations in one expression
  result = sorted(k for k, v in mapping.items() if v in set(x.name for x in sources))

  # Good — intermediate variable makes intent clear
  source_names = {x.name for x in sources}
  result = sorted(k for k, v in mapping.items() if v in source_names)
  ```
- Prefer dataclasses over tuples when returning multiple values.
- Use explicit `None` comparisons: `if x is not None` not `if x`.
- Import all modules at the top of the file, not inside functions or methods.
- Prefer importing a module and using module.function instead of
  using ``from module import function``. This applies to intra-package
  imports too: use ``from . import config`` then ``config.X``, not
  ``from .config import X``. Exceptions: ``from typing import ...`` is
  acceptable; ``from .X import Y`` is acceptable in ``__init__.py`` for
  defining the public API. Use ``import concurrent.futures as cf``.
- Use idiomatic pathlib.Path operations instead of os.path operations.
- When a parameter has a computed default derived from another parameter,
  compute it once at the point of use (the leaf function), not at every
  layer in the call chain. Pass `None` through intermediate layers.
- Zarr v3 is used (dependency: `zarr>=3`). Do not use Zarr v2 APIs.
- Use PEP 604 union syntax: `int | None`, not `Optional[int]`.
- One `logger = logging.getLogger(__name__)` per module at top level.
- Use underscores with lower case for variable names in CSVs and for files and
  folders except .agents, where you can use the - separation convention.
- Prioritise semantic correctness, well-defined invariants, and clear error
  handling over micro-optimizations or clever tricks.
- Maintain inter-linked docstrings and use comments where they add context and
  explanation, but avoid redundant or obvious comments that do not enhance
  understanding.
- Structure code to be modular, with coherent responsibilities, good separation
  of concerns, and an architecture that can be extended without large-scale
  rewrites.
- Apply DRY carefully: factor out shared logic when it improves clarity and
  maintenance, but avoid over-abstraction or generic frameworks that obscure
  intent.
- Use discernment and good judgement to write code that is correct,
  maintainable, and efficient, while avoiding unnecessary complexity or
  over-engineering.
- Consider algorithmic time and memory complexity when choosing data structures
  and approaches, especially for code on critical paths or large problem sizes.
- Aim for readable, explicit code with simple control flow and minimal “magic”;
  prefer straightforward implementations to compact but opaque idioms.
- Keep implementations concise but not cryptic; remove dead or unused code and
  avoid unnecessary indirection.
- For tests, prefer simple, direct assertions that fail fast; test code should
  not be defensive, over-general, or attempt to mask or recover from failures.
- For APIs and function interfaces prefer defined contracts with clear error
  states over flexible but loosely defined behavior that attempts to infer
  intent or handle edge cases implicitly.
- Where there is likely to be a long running process and it can be parallelised
  use multiprocessing with queues to distribute work across all available CPU
  cores.
- Don't bend over backwards to make anything backwards compatible with previous
  implementations.

## Tool use

- Use `uv run` for all Python tooling (never bare `python -m`)

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
