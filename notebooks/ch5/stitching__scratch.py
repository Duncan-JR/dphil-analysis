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
# # Greedy stitching scratch
#
# Each example is a list of `(left, right)` tuples sorted by `left`.
# Bounds are site indexes; `right` is exclusive. Each row represents a different
# candidate ancestor, including rows with identical bounds.
#
# All examples use the same site domain `[0, num_sites)`. The minimum focal
# region size applies to a connected union of intervals, including touching
# intervals. It does not filter individual ancestors.
#
# The plots show only the input intervals. Use the empty cells to explore your
# own copying paths, gaps and treatment of short connected regions.

# %%
import pathlib

from ch5 import nb_plot, nb_utils

repository_dir = pathlib.Path.cwd()
config = {
    "num_sites": 40,
    "min_focal_region": 10,
    "figure_width": 10,
    "row_height": 0.45,
    "minimum_figure_height": 2.4,
    "tick_step": 5,
}

# %% [markdown]
# ## 1. Overlapping chain across the whole domain
#
# Start with a connected example whose coverage reaches both domain boundaries.

# %%
intervals_chain = [(0, 8), (5, 15), (12, 22), (20, 30), (28, 40)]
nb_plot.plot_intervals(
    intervals_chain,
    "Overlapping chain",
    config=config,
    plot_figure_width=10,
    plot_minimum_figure_height=2.4,
    plot_row_height=0.45,
    plot_tick_step=5,
)

# %%
# Save the last region if it satisfies criteria
# Update parent
# Better starting parent
# Update region bounds
nb_utils.make_focal_regions_buffered(intervals_chain)

# %%
# attempt 2
# Outer loop (1) only advances when a focal region finishes
# Middle loop (2): moves cursor forward through the region
# Inner loop (3): for fixed cursor, find parent with longest reach
# Exit loop 3 - either no parent or found the best
# Exit Loop 2 because we have hit a gap
# Move cursor forward to where the next path starts
# Still inside outer loop: decide whether to retain this region.
# Outer loop repeats as long as there are intervals untouched

# %%

# %% [markdown]
# ## 2. Containment and ties
#
# Consider equal left endpoints, equal right endpoints, contained intervals and
# duplicate bounds. The row index can serve as the ancestor identity.

# %%

# %%
intervals_nested = [(0, 12), (0, 18), (3, 8), (5, 18), (12, 24), (12, 24), (18, 30)]
nb_plot.plot_intervals(
    intervals_nested,
    "Containment, ties and duplicate bounds",
    config=config,
    plot_figure_width=10,
    plot_minimum_figure_height=2.4,
    plot_row_height=0.45,
    plot_tick_step=5,
)

# %%
nb_utils.make_focal_regions(intervals_nested, min_width=1)

# %% [markdown]
# ## 3. Touching endpoints and a one-site gap
#
# Pay attention to the exclusive right endpoints when deciding where there is
# continuous coverage and where a gap begins.

# %%
intervals_touching = [(0, 6), (6, 13), (13, 20), (21, 29), (29, 40)]
nb_plot.plot_intervals(
    intervals_touching,
    "Touching endpoints and a one-site gap",
    config=config,
    plot_figure_width=10,
    plot_minimum_figure_height=2.4,
    plot_row_height=0.45,
    plot_tick_step=5,
)

# %%
nb_utils.make_focal_regions(intervals_touching, min_width=1)

# %% [markdown]
# ## 4. Multiple components and boundary gaps
#
# Include the beginning and end of the domain when recording gaps. Apply the
# minimum focal-region size to each connected component.

# %%
intervals_gaps = [(3, 9), (7, 12), (17, 23), (22, 27), (32, 35)]
nb_plot.plot_intervals(
    intervals_gaps,
    "Separated components with leading and trailing gaps",
    config=config,
    plot_figure_width=10,
    plot_minimum_figure_height=2.4,
    plot_row_height=0.45,
    plot_tick_step=5,
)

# %%
nb_utils.make_focal_regions(intervals_gaps, min_width=10)

# %% [markdown]
# ## 5. Total length and remaining reach
#
# Revisit the choice available after reaching site 8. You can first explore the
# covering choices without minimum-size filtering, then apply the configured
# minimum to the connected region.

# %%
intervals_reach = [(0, 8), (6, 12), (8, 13)]
nb_plot.plot_intervals(
    intervals_reach,
    "Total interval length and remaining reach",
    config=config,
    plot_figure_width=10,
    plot_minimum_figure_height=2.4,
    plot_row_height=0.45,
    plot_tick_step=5,
)

# %%
nb_utils.make_focal_regions(intervals_reach, min_width=1)

# %% [markdown]
# ## 6. Connected regions around the minimum
#
# These components span one site below, exactly at, and one site above the
# configured minimum of ten sites. Every individual interval is shorter than
# that minimum. Check the equality case and how discarded regions affect gaps.

# %%
intervals_minimum = [(0, 4), (3, 9), (12, 18), (17, 22), (25, 31), (30, 36)]
nb_plot.plot_intervals(
    intervals_minimum,
    "Connected-region lengths around the minimum",
    config=config,
    plot_figure_width=10,
    plot_minimum_figure_height=2.4,
    plot_row_height=0.45,
    plot_tick_step=5,
)

# %%
nb_utils.make_focal_regions(intervals_minimum, min_width=11)

# %% [markdown]
# ## 7. No focal candidates
#
# Consider what the output should contain over the full domain.

# %%
intervals_empty = []
nb_plot.plot_intervals(
    intervals_empty,
    "Empty candidate list",
    config=config,
    plot_figure_width=10,
    plot_minimum_figure_height=2.4,
    plot_row_height=0.45,
    plot_tick_step=5,
)

# %%
nb_utils.make_focal_regions(intervals_empty, min_width=11)

# %%
