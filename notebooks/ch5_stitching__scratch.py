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
import matplotlib.pyplot as plt

config = {
    "num_sites": 40,
    "min_focal_region": 10,
    "figure_width": 10,
    "row_height": 0.45,
    "minimum_figure_height": 2.4,
    "tick_step": 5,
}


def plot_intervals(intervals, title):
    """Show input ancestors with closed left and open right endpoints."""
    height = max(
        config["minimum_figure_height"],
        config["row_height"] * len(intervals) + 1.2,
    )
    fig, ax = plt.subplots(figsize=(config["figure_width"], height))
    for row, (left, right) in enumerate(intervals):
        ax.hlines(row, left, right, color="tab:blue", linewidth=3)
        ax.plot(left, row, "o", color="tab:blue", markersize=7)
        ax.plot(
            right, row, "o", color="tab:blue", markerfacecolor="white",
            markersize=7, markeredgewidth=1.5,
        )
    rows = range(len(intervals))
    labels = [f"{row}: [{left}, {right})" for row, (left, right) in enumerate(intervals)]
    ax.set_yticks(rows, labels)
    ax.set_ylim(max(len(intervals) - 0.5, 0.5), -0.5)
    ax.set_xlim(0, config["num_sites"])
    ticks = range(0, config["num_sites"] + 1, config["tick_step"])
    ax.set_xticks(ticks)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlabel("Site boundary (filled left endpoint; open right endpoint)")
    ax.set_ylabel("Ancestor row")
    ax.set_title(title)
    if len(intervals) == 0:
        ax.text(0.5, 0.5, "No candidate intervals", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 1. Overlapping chain across the whole domain
#
# Start with a connected example whose coverage reaches both domain boundaries.

# %%
intervals_chain = [(0, 8), (5, 15), (12, 22), (20, 30), (28, 40)]
plot_intervals(intervals_chain, "Overlapping chain")

# %%
import numpy as np 

def make_focal_regions(intervals, min_width=5):
    regions = []
    region_left = 0
    region_right = -1
    parent = -1
    parent_left = 0
    parent_right = 0
    next_parent = -1
    paths_buffer = []
    paths = []
    if len(intervals) == 0:
        return regions
    for i, (left, right) in enumerate(intervals):
        print(f'Processing {i}')
        if left > region_right:
            if region_right > 0:
                paths_buffer.append(((parent_left, parent_right),parent))
                # Save the last region if it satisfies criteria
                region_width = region_right - region_left
                if region_width >= min_width:
                    regions.append((region_left, region_right))
                    paths += paths_buffer
                    paths_buffer = []
                    next_parent = -1
            print(f'Starting region at {left}')
            region_left = left
            region_right = right
            # Update parent
            print(f'Starting parent is {i} from {left} to {right}')
            parent = i
            parent_left = left
            parent_right = right
        else:
            if left >= parent_right:
                print(f'Moved past current parent {parent} which ends at {parent_right}, now at {left}')
                paths_buffer.append(((parent_left, parent_right),parent))
                
                parent_left = parent_right
                if left == parent_right and right > region_right:
                    print(f'New interval starts at {left} and is longest')
                    parent = i
                    parent_right = right
                    region_right = right
                else:
                    print(f'Longest parent now is {next_parent} ending at {region_right-1}')
                    parent = next_parent
                    parent_right = region_right
            elif left == parent_left and right > parent_right:
                # Better starting parent
                print(f'Better starting parent is {i} ending at {right}')
                parent = i
                parent_right = right
            # Update region bounds
            if right > region_right:
                region_right = right
                next_parent = i
                print(f'New next_parent is {i} ending at {right}')
    print(f'After last int, we have current ({parent_left},{parent_right}): {parent}')
    print(f'New next parent is {next_parent} ending at {region_right}')
    paths_buffer.append(((parent_left, parent_right),parent))
    if parent_right != region_right:
        paths_buffer.append(((parent_right, region_right), next_parent))
    region_width = region_right - region_left
    if region_right > 0 and region_width >= min_width:
        regions.append((region_left, region_right))
        paths += paths_buffer
    return regions, paths

make_focal_regions(intervals_chain)
    

# %%
# attempt 2
def make_focal_regions(intervals, min_width):
    regions = []
    paths = []
    i = 0

    # Outer loop (1) only advances when a focal region finishes
    while i < len(intervals):
        region_left = intervals[i][0]
        cursor = region_left
        paths_buffer = []

        # Middle loop (2): moves cursor forward through the region
        while True:
            parent = None
            max_right = cursor

            # Inner loop (3): for fixed cursor, find parent with longest reach
            while i < len(intervals):
                left, right = intervals[i]

                if left > cursor:
                    break  # Exit loop 3 - either no parent or found the best

                if right > max_right:
                    parent = i
                    max_right = right

                i += 1

            if parent is None:
                break  # Exit Loop 2 because we have hit a gap

            paths_buffer.append(((cursor, max_right), parent))
            # Move cursor forward to where the next path starts
            cursor = max_right

        # Still inside outer loop: decide whether to retain this region.
        region_right = cursor
        region_width = region_right - region_left

        if region_width >= min_width:
            regions.append((region_left, region_right))
            paths.extend(paths_buffer)
        # Outer loop repeats as long as there are intervals untouched

    return regions, paths


# %%

# %% [markdown]
# ## 2. Containment and ties
#
# Consider equal left endpoints, equal right endpoints, contained intervals and
# duplicate bounds. The row index can serve as the ancestor identity.

# %%

# %%
intervals_nested = [
    (0, 12), (0, 18), (3, 8), (5, 18), (12, 24), (12, 24), (18, 30),
]
plot_intervals(intervals_nested, "Containment, ties and duplicate bounds")

# %%

make_focal_regions(intervals_nested, min_width=1)

# %% [markdown]
# ## 3. Touching endpoints and a one-site gap
#
# Pay attention to the exclusive right endpoints when deciding where there is
# continuous coverage and where a gap begins.

# %%
intervals_touching = [(0, 6), (6, 13), (13, 20), (21, 29), (29, 40)]
plot_intervals(intervals_touching, "Touching endpoints and a one-site gap")

# %%

make_focal_regions(intervals_touching, min_width=1)

# %% [markdown]
# ## 4. Multiple components and boundary gaps
#
# Include the beginning and end of the domain when recording gaps. Apply the
# minimum focal-region size to each connected component.

# %%
intervals_gaps = [(3, 9), (7, 12), (17, 23), (22, 27), (32, 35)]
plot_intervals(intervals_gaps, "Separated components with leading and trailing gaps")

# %%
make_focal_regions(intervals_gaps, min_width=10)

# %% [markdown]
# ## 5. Total length and remaining reach
#
# Revisit the choice available after reaching site 8. You can first explore the
# covering choices without minimum-size filtering, then apply the configured
# minimum to the connected region.

# %%
intervals_reach = [(0, 8), (6, 12), (8, 13)]
plot_intervals(intervals_reach, "Total interval length and remaining reach")

# %%
make_focal_regions(intervals_reach, min_width=1)

# %% [markdown]
# ## 6. Connected regions around the minimum
#
# These components span one site below, exactly at, and one site above the
# configured minimum of ten sites. Every individual interval is shorter than
# that minimum. Check the equality case and how discarded regions affect gaps.

# %%
intervals_minimum = [(0, 4), (3, 9), (12, 18), (17, 22), (25, 31), (30, 36)]
plot_intervals(intervals_minimum, "Connected-region lengths around the minimum")

# %%
make_focal_regions(intervals_minimum, min_width=11)

# %% [markdown]
# ## 7. No focal candidates
#
# Consider what the output should contain over the full domain.

# %%
intervals_empty = []
plot_intervals(intervals_empty, "Empty candidate list")

# %%
make_focal_regions(intervals_empty, min_width=11)

# %%
