# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mutation-age accuracy and matching performance
#
# Detailed report and supporting diagnostics, 22 October 2026.
# The report includes the root-copying mechanism, rescaling controls, true-topology
# controls, HPC cache audit, occupancy analysis and local native matching timings.
# Expensive experiments are saved with their source scripts and configuration.

# %%
import pathlib

import IPython.display as display
import pandas as pd

# %%
notebook_dir = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path.cwd()
repository_dir = notebook_dir if (notebook_dir / 'data').is_dir() else notebook_dir.parent.parent
artifact_dir = repository_dir / 'data/ch5/mutation_age_performance_review_2026_10_22'
display.HTML((artifact_dir / 'mutation_age_performance_report.html').read_text())

# %% [markdown]
# The following cells expose the evidence tables for inspection. No original
# experiment caches are changed and no ARG metrics are calculated.

# %%
age_metrics = pd.read_csv(artifact_dir / 'age_metrics.csv')
display.display(age_metrics)

# %%
for name in ['root_copying', 'preprocessing', 'root_error', 'large_control_metrics', 'control_metrics', 'cache_audit']:
    display.display(display.Markdown(f'**{name}**'))
    display.display(pd.read_csv(artifact_dir / f'{name}.csv'))

# %%
for path in sorted(artifact_dir.glob('micro_*_summary.csv')):
    display.display(display.Markdown(f'**{path.stem}**'))
    display.display(pd.read_csv(path))
