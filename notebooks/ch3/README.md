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
