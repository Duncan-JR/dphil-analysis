# DPhil analysis

Install the package and notebook environment from this repository root:

```sh
uv sync
uv run python notebooks/ch4_error_estimation/implementation_testing.py
```

The error-estimation validation notebook uses local, read-only genotype stores
and explicit HapMap files described in `plans/refactor_error_estimation.md`.
It creates temporary 200-sample, 4 Mb fixtures; results are software smoke tests,
not cohort error-rate estimates. Edit the paired percent-format Python source.

Execute the notebook from the repository root:

```sh
export JUPYTER_DATA_DIR="$PWD/.venv/share/jupyter"
export JUPYTER_RUNTIME_DIR="$PWD/.venv/jupyter_runtime"
export IPYTHONDIR="$PWD/.venv/ipython"
uv run python -m ipykernel install --prefix .venv --name dphil_analysis --display-name "dphil-analysis"
uv run jupytext --to ipynb notebooks/ch4_error_estimation/implementation_testing.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dphil_analysis --ExecutePreprocessor.timeout=-1 notebooks/ch4_error_estimation/implementation_testing.ipynb
uv run jupytext --sync notebooks/ch4_error_estimation/implementation_testing.ipynb
```
