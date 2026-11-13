# DPhil analysis

Install and run from this repository root:

```sh
uv sync
uv run jupyter lab notebooks/ch3/figure_gen.ipynb
uv run python notebooks/ch5/stitching_pt13_jerome_update.py
```

Use the repository root as the Jupyter kernel working directory too. Check `%pwd`
in Jupyter; if the kernel started in `notebooks/chN/`, run `%cd ../..`
before the first code cell. Notebook cells use `pathlib.Path.cwd()` for this
root and initialise scientific settings
explicitly. Presentation choices are plot-call keyword arguments. The committed
`.ipynb` files preserve visible historical outputs; inspect both sides of a
Jupytext pair before editing and reconcile code cells without clearing outputs.

- `src/ch3/`, `src/ch4/`, `src/ch5/`: installed chapter packages. Notebook
  preparation lives in `nb_utils`, plotting in `nb_plot`. Chapter 4 retains
  its estimator and validation libraries.
- `notebooks/chN/`: paired sources, saved notebooks and chapter-specific
  methods/input/output notes. See [chapter 3](notebooks/ch3/README.md),
  [chapter 4](notebooks/ch4/README.md), [chapter 5](notebooks/ch5/README.md).
- `scripts/`: flat standalone entry points. See [commands](scripts/README.md)
  for portable Zarr/phase exporters and the historical matching drivers.
- `data/chN/`, `data/general/`: chapter-owned results and genuinely shared
  stores/maps/metadata and simulated truth. Run configurations beside archived data are provenance.
- `figures/chN/`: local exploratory exports, independent of maintained thesis
  output. Do not overwrite an earlier scientific run as incidental cleanup.

Chapter 3 consumes results from the external performance pipeline; this
workspace does not contain or run that pipeline. The refactor inventory,
checksum-verified data moves and validation evidence are in
[the execution record](plans/refactor_project.md#execution-record).

The maintained thesis build uses `Snakefile` and `thesis_config.yaml`:

```sh
uv run snakemake --cores all --dry-run
uv run snakemake --cores all
uv run snakemake ch3_plots ch3_statistics --cores all \
  --config thesis_root=/private/tmp/ch3_thesis work_root=/private/tmp/ch3_prepared \
  statistics_dir=/private/tmp/ch3_statistics
uv run pytest
```

Figures go to `../dphil-thesis/figures/ch3/` by default. Statistics are generated
in this repository's `statistics/`: `ch3.tex`, `ch3_claims.json`, `ch3_checks.json`.
The config owns all 23 figure filenames; input paths are explicit in the rules.
Plot defaults reproduce the finished figures. Only shared renderers' differing
arguments appear in config, and the extension example is fixed in code.
Preparation reads retained inputs; generation never starts upstream inference
or benchmarks. Install `rsvg-convert` (librsvg) for the tree PDF. See
[chapter 3 workflow and provenance](notebooks/ch3/README.md#maintained-thesis-workflow)
for data identities, claim mappings and dependency restoration. Later chapters
acquire thesis rules only when their final outputs are defined.

The lockfile pins tsinfer and a retained tsls source distribution built from
commit `c560b405761ae09625e266d39c2356ccc4b54b69`. Both dependency Git histories
are backed up under `data/general/dependencies/`. The relative source archive
and explicit same-version UV override remove the unpublished dependency's
machine-specific sibling URL. No sibling checkout is needed for installation.

The maintained test suite contains six synthetic correctness cases for final
chapter 3 calculations and retained chapter 4 methods. It needs no research
data or HPC access. Exploratory helpers, notebooks and chapter 5 have no tests.

To reconcile notebook pairs after editing either representation, preserve the
saved outputs with Jupytext's sync command:

```sh
uv run jupytext --sync notebooks/ch5/stitching__scratch.ipynb
```
