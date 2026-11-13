# DPhil analysis

Install and run from this repository root:

```sh
uv sync
uv run jupyter lab notebooks/ch3/figure_gen.ipynb
uv run python notebooks/ch5/stitching_pt13_jerome_update.py
```

Use the repository root as the Jupyter kernel working directory too. Notebook
cells use `pathlib.Path.cwd()` for this root and initialise scientific settings
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
  stores/maps/metadata. Run configurations beside archived data are provenance.
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
  --config thesis_root=/private/tmp/ch3_thesis work_root=/private/tmp/ch3_prepared
uv run pytest
```

The default destination is `../dphil-thesis`. All 23 chapter 3 figure files and
`statistics/ch3.tex`, `ch3_claims.json`, `ch3_checks.json` are declared outputs.
Preparation reads retained inputs; generation never starts upstream inference
or benchmarks. Install `rsvg-convert` (librsvg) for the tree PDF. See
[chapter 3 workflow and provenance](notebooks/ch3/README.md#maintained-thesis-workflow)
for data identities, claim mappings and dependency restoration. Later chapters
acquire thesis rules only when their final outputs are defined.

The lockfile pins tsinfer and tsls source commits. The unpublished tsls commit
is also retained in a committed Git bundle; a new installation needs that
local Git object before `uv sync --locked`. This workspace's source URL names
`/Users/duncan/work/tsls`; restoration and relocation are documented in the
chapter 3 README. The bundle, rather than a moving checkout, identifies the code.

The maintained test suite contains six synthetic correctness cases for final
chapter 3 calculations and retained chapter 4 methods. It needs no research
data or HPC access. Exploratory helpers, notebooks and chapter 5 have no tests.
