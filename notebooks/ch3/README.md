# Chapter 3 exploratory figures

Run Jupyter or paired sources from the repository root. `figure_gen` retains
its broad narrative, error-profile analyses, numerical check tables, toy
illustrations and historical outputs. Scientific settings are initialised in
cells; plot choices are keyword arguments. Shared code lives in `ch3.nb_utils`
and `ch3.nb_plot`. Local exports go to `figures/ch3/`; this notebook does not
write to the thesis repository.

Inputs: the identified simulation aggregate and exported validation/phase CSVs
under `data/ch3/`, empirical ancestor-span tables, the empirical DAF store,
other demographies' aggregates, and exported external benchmark performance.
Later buffering experiments use separate alternative-branch inputs and caches.
Do not overwrite these earlier runs. The complete input/rerun audit is recorded
in the refactor execution record; saved outputs remain from their prior runs.

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
| [figure_gen](figure_gen.ipynb) | Broad CH3 ancestor accuracy, errors, DAF, toy and performance figures | `data/ch3/` aggregate, validation export and current focal tables; toy parameters; other inputs listed above | Code cells 0, 5, 6, 11 passed; historical visible outputs preserved |

The first five toy cells also ran during phase 3, including genotype and replay
assertions and PNG/PDF rendering. Full maintained-figure/numerical validation
is recorded with the independent thesis workflow; exploratory buffering and
alternative-branch analyses retain their previous outputs.
