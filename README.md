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
