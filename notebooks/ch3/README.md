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

## Maintained thesis workflow

`src/ch3/thesis_utils.py` and `thesis_plot.py` contain only the subset used by
the final chapter. They do not import the exploratory modules. `steps.py`
distributes independent figure jobs through multiprocessing queues, using the
CPU allocation supplied by Snakemake. The workflow consumes retained inputs,
produces explicit Parquet tables, then produces 23 individually named figures
and `statistics/ch3.tex` beneath the configured thesis root. The manually drawn
`anc_eval_pipe.png` is a declared retained asset, copied unchanged.

From the analysis root:

```sh
uv sync --locked
uv run snakemake --cores all --dry-run
uv run snakemake ch3_plots ch3_statistics --cores all
# Isolated validation destination; scientific inputs remain the same:
uv run snakemake --cores all \
  --config thesis_root=/private/tmp/ch3_thesis work_root=/private/tmp/ch3_prepared
```

`rsvg-convert` from librsvg is required for the PDF illustration. For a
restricted shell, the validation used `UV_CACHE_DIR=/private/tmp/dphil_uv_cache`,
`XDG_CACHE_HOME=/private/tmp/dphil_tool_cache` and
`MPLCONFIGDIR=/private/tmp/dphil_mpl`; these are cache locations, not scientific
settings. Ordinary installations can use their normal user caches.

The [input manifest](../../plans/thesis_inputs.json) gives SHA-256/size
identities for every retained CSV/manual asset and every consumed empirical
Zarr metadata/chunk file, producer commands, observed producer revisions and
backup locations. Historical producer invocations/revisions were not retained
with every aggregate; current upstream configurations have moved to other
experiments and must not be mistaken for the original 20 Mb run. This does
not prevent reproducing the final analyses from the identified retained
inputs. Raw stores and expensive upstream inference were not regenerated.
The external performance pipeline remains outside this repository.

Retain `data/ch3/` inputs as research data, including `demographies/`,
`performance/perf_combined.csv` and `thesis_assets/anc_eval_pipe.png`.
`data/ch3/thesis_prepared/` is rebuildable only while those inputs and the
identified empirical arrays remain available. The DAF preparation declares
the store directory and every metadata/chunk in exactly the three arrays it
reads. Changes to those files, settings, or maintained helper sources invalidate
the appropriate outputs. Missing required inputs fail the target; there are
no blank figures or fallback statistics.

The final calculations preserve focal-site multiplicity. Both sides enter
overshoot distributions; left rows count each whole overlap once. Pooled
mismatch divides summed errors by summed compared inference sites, rather
than averaging overlap rates. Genotype errors count diploid genotypes;
performance sample sizes count haplotypes. Stage speedups average paired
old/new ratios. Total speedups sum stages within each run before pairing.
Quoted error-regime comparisons exclude the exploratory lambda=100 stress
panel, and exclude mispolarised focal sites where the text specifies this.
Histogram differences are relative bin-count changes on shared physical bins.
Severe-bin claims use midpoints below the declared severe-undershoot threshold.

`statistics/ch3_claims.json` maps every named TeX value to its inputs,
calculation, units and unrounded result. `ch3_checks.json` retains focal counts,
span curves, rare tails, demographic quantiles, pooled overlap errors, genotype
errors/transitions, overshoot summaries, histogram changes, severe-focal counts
and paired performance summaries. Neither output reads historical check tables.
The committed [claim review](../../plans/thesis_claims.md) maps macros to thesis
passages and records rounding discrepancies resolved by generation. Literature
estimates and claims referring to chapter 2 remain attributed prose, not purported
derivations from these inputs. Fixed experimental settings can be emitted from
`thesis_config.yaml`; notebook parameters remain independent.

Python 3.12.13 and the root `uv.lock` were used. tsinfer is pinned to
`e2f8de7c950179215ece27e1d31c7b096308b4c8`, tsls to
`c560b405761ae09625e266d39c2356ccc4b54b69`. Their source Git bundles are retained
at `data/general/dependencies/{tsinfer,tsls}.bundle` and tracked in Git.
To restore the unpublished tsls source when the sibling checkout is absent:

```sh
git clone data/general/dependencies/tsls.bundle ../tsls
git -C ../tsls checkout c560b405761ae09625e266d39c2356ccc4b54b69
uv sync --locked
```

On a different directory layout, point only the tsls `file://` source URL in
`pyproject.toml` to the restored clone and run `uv lock`, retaining the same
commit. The absolute URL reflects this personal workspace; the committed
bundle makes the source recoverable without assuming an unpublished commit
exists on GitHub. Neither dependency is an editable moving sibling source.
