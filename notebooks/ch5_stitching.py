# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dphil-analysis
#     language: python
#     name: dphil_analysis
# ---

# %% [markdown]
# # Diagnosing stitched focal and gap matches
#
# Compare the saved full HMM and stitched paths for a simulated dataset and
# chromosome 20 from 1KGP. The stitched path combines **retained exact focal
# pieces** with **independent full-reference gap fills**. In this notebook,
# "gap fill" refers to the replaced regions, while "stitched" means the complete
# combined path. All parents are original-reference node IDs.
#
# Questions:
# 1. Which focal pieces and gap fills agree with the full HMM parent path?
# 2. Where do the additional switches occur: inside focal pieces, inside gaps,
#    or at the seams between them?
# 3. Can we inspect both successful and unsuccessful gap fills, and focal pieces
#    that already differ from the full path?
#
# Settings and deep-dive identities live in
# `experiments/ch5/stitching_diagnostics_config.yaml`. The notebook reads existing
# outputs and copied references; it does not rerun matching or change its inputs.
# Edit this `.py` file and use Jupytext to synchronize the paired notebook.

# %%
import dataclasses
import json
import logging
import pathlib

import IPython.display as ipython_display
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tskit
import yaml
import zarr

logger = logging.getLogger(__name__)

# %%
notebook_dir = (
    pathlib.Path(__file__).resolve().parent
    if "__file__" in globals()
    else pathlib.Path.cwd()
)
diagnostic_config_path = (
    notebook_dir.parent / "experiments/ch5/stitching_diagnostics_config.yaml"
)
with diagnostic_config_path.open() as file:
    config = yaml.safe_load(file)
experiment_config_path = diagnostic_config_path.parent / config["experiment_config"]
with experiment_config_path.open() as file:
    experiment_config = yaml.safe_load(file)
input_dir = pathlib.Path(experiment_config["input_dir"]).expanduser()
output_dir = pathlib.Path(experiment_config["output_dir"]).expanduser()
if not input_dir.is_absolute():
    input_dir = experiment_config_path.parent / input_dir
if not output_dir.is_absolute():
    output_dir = experiment_config_path.parent / output_dir
coverage_input_dir = pathlib.Path(config["coverage_input_dir"]).expanduser()
gap_comparison_dir = pathlib.Path(config["gap_comparison_dir"]).expanduser()
if not coverage_input_dir.is_absolute():
    coverage_input_dir = diagnostic_config_path.parent / coverage_input_dir
if not gap_comparison_dir.is_absolute():
    gap_comparison_dir = diagnostic_config_path.parent / gap_comparison_dir
style = config["plot"]
plt.rcParams.update({"figure.dpi": style["dpi"], "font.size": style["font_size"]})
category_cmap = mcolors.ListedColormap(style["category_colors"])
category_norm = mcolors.BoundaryNorm(np.arange(5) - 0.5, category_cmap.N)
source_cmap = mcolors.ListedColormap(style["source_colors"])
source_norm = mcolors.BoundaryNorm(np.arange(3) - 0.5, source_cmap.N)


# %%
@dataclasses.dataclass
class HaplotypeDiagnostics:
    """Site-based evidence and original-node parents for one saved comparison."""

    summary: dict
    full: dict
    stitched: dict
    full_parents: np.ndarray
    stitched_parents: np.ndarray
    accepted: np.ndarray
    disagreement: np.ndarray
    categories: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.summary['sample_id']}/{self.summary['ploidy_index']}"


@dataclasses.dataclass
class DatasetDiagnostics:
    """One copied reference and the selected experiment's saved haplotypes."""

    name: str
    label: str
    reference: tskit.TreeSequence
    positions: np.ndarray
    site_edges: np.ndarray
    cases: list[HaplotypeDiagnostics]
    primary: HaplotypeDiagnostics


def read_matches(path: pathlib.Path) -> dict:
    """Key JSONL records by haplotype identity, checking uniqueness."""
    records = {}
    with path.open() as file:
        for line in file:
            record = json.loads(line)
            key = (record["sample_id"], record["ploidy_index"])
            assert key not in records
            records[key] = record
    return records


def parents_at_sites(record: dict, reference: tskit.TreeSequence) -> np.ndarray:
    """Evaluate a saved, half-open path only at reference inference sites."""
    path = record["path"]
    left = np.array([segment["left"] for segment in path])
    right = np.array([segment["right"] for segment in path])
    parents = np.array([segment["parent"] for segment in path], dtype=np.int32)
    assert left[0] == 0 and right[-1] == reference.sequence_length
    assert np.all(left < right) and np.array_equal(right[:-1], left[1:])
    assert np.all(parents[1:] != parents[:-1])
    assert np.all((parents >= 0) & (parents < reference.num_nodes))
    segments = np.searchsorted(right, reference.sites_position, side="right")
    result = parents[segments]
    assert np.count_nonzero(result[1:] != result[:-1]) == len(path) - 1
    return result


def interval_mask(intervals: list, num_sites: int) -> np.ndarray:
    """Reconstruct disjoint site intervals without inferring BP coverage."""
    mask = np.zeros(num_sites, dtype=bool)
    for start, end in intervals:
        assert 0 <= start < end <= num_sites
        assert not np.any(mask[start:end])
        mask[start:end] = True
    return mask


def load_dataset(settings: dict) -> DatasetDiagnostics:
    """Validate saved summaries against paths and focal/gap site partitions."""
    name = settings["name"]
    directory = output_dir / name
    reference = tskit.load(input_dir / "ancestors" / f"{name}_inferred_ancestors.trees")
    positions = reference.sites_position
    terminal_edge = positions[-1] + style["terminal_site_width_bp"]
    site_edges = np.concatenate((positions, [terminal_edge]))
    full_records = read_matches(directory / "full_matches.jsonl")
    stitched_records = read_matches(directory / "stitched_matches.jsonl")
    summaries = pd.read_csv(directory / "comparison.csv").to_dict("records")
    assert len(summaries) == len(full_records) == len(stitched_records)
    cases = []
    for summary in summaries:
        summary["num_extra_switches"] = (
            summary["stitched_switches"] - summary["full_switches"]
        )
        summary["extra_switch_fraction"] = (
            summary["num_extra_switches"] / summary["stitched_switches"]
            if summary["stitched_switches"] > 0
            else np.nan
        )
        key = (summary["sample_id"], summary["ploidy_index"])
        full = full_records[key]
        stitched = stitched_records[key]
        for record in (full, stitched):
            for field in ("dataset", "source", "sample_id", "ploidy_index", "ac_cutoff"):
                assert record[field] == summary[field]
        assert summary["dataset"] == name
        assert summary["ac_cutoff"] == experiment_config["ac_cutoff"]
        full_parents = parents_at_sites(full, reference)
        stitched_parents = parents_at_sites(stitched, reference)
        accepted = interval_mask(stitched["accepted_site_intervals"], len(positions))
        gaps = interval_mask(stitched["gap_site_intervals"], len(positions))
        assert np.array_equal(gaps, ~accepted)
        disagreement = full_parents != stitched_parents
        assert summary["num_sites"] == len(positions)
        assert summary["accepted_sites"] == np.count_nonzero(accepted)
        assert summary["gap_sites"] == np.count_nonzero(gaps)
        assert summary["num_gaps"] == len(stitched["gap_site_intervals"])
        assert summary["accepted_disagreement_sites"] == np.count_nonzero(
            disagreement & accepted
        )
        assert summary["gap_disagreement_sites"] == np.count_nonzero(disagreement & gaps)
        assert np.isclose(summary["fraction_parent_agreement"], np.mean(~disagreement))
        assert summary["full_switches"] == len(full["path"]) - 1
        assert summary["stitched_switches"] == len(stitched["path"]) - 1
        assert summary["mutations_identical"] == (
            full["mutations"] == stitched["mutations"]
        )
        categories = (~accepted).astype(np.int8) * 2 + disagreement.astype(np.int8)
        cases.append(
            HaplotypeDiagnostics(
                summary,
                full,
                stitched,
                full_parents,
                stitched_parents,
                accepted,
                disagreement,
                categories,
            )
        )
    primary_settings = settings["deep_dive"]
    primary = next(
        case
        for case in cases
        if case.summary["sample_id"] == primary_settings["sample_id"]
        and case.summary["ploidy_index"] == primary_settings["ploidy_index"]
    )
    return DatasetDiagnostics(
        name, settings["label"], reference, positions, site_edges, cases, primary
    )


datasets = [load_dataset(settings) for settings in config["datasets"]]
summary = pd.DataFrame([case.summary for dataset in datasets for case in dataset.cases])
summary["focal_agreement_fraction"] = (
    1 - summary.accepted_disagreement_sites / summary.accepted_sites
)
summary["gap_agreement_fraction"] = (
    1 - summary.gap_disagreement_sites / summary.gap_sites
)
ipython_display.display(
    summary[
        [
            "dataset",
            "sample_id",
            "ploidy_index",
            "num_focal_ancestors",
            "accepted_sites",
            "gap_sites",
            "num_gaps",
            "fraction_parent_agreement",
            "focal_agreement_fraction",
            "gap_agreement_fraction",
            "full_switches",
            "stitched_switches",
            "num_extra_switches",
            "extra_switch_fraction",
            "full_mismatches",
            "stitched_mismatches",
        ]
    ].round(4)
)

# %% [markdown]
# ## Coverage, agreement, and switches across the saved haplotypes
#
# Bar lengths below are **fractions of inference sites**, not physical coverage.
# Focal/gap provenance and parent agreement are independent dimensions. An exact
# focal allele match may still copy a different parent than the full HMM.
#
# Switches are changes between adjacent site parents. A seam is the edge where
# provenance changes between retained focal coverage and a gap fill; it contributes
# a switch only if the parent changes there. The same edge categories are used for
# the full HMM path so their totals can be compared directly.


# %%
def switch_breakdown(case: HaplotypeDiagnostics, parents: np.ndarray) -> np.ndarray:
    """Partition switches into focal interiors, gap interiors, and provenance seams."""
    switches = parents[1:] != parents[:-1]
    focal_interior = case.accepted[1:] & case.accepted[:-1]
    gap_interior = ~case.accepted[1:] & ~case.accepted[:-1]
    seams = case.accepted[1:] != case.accepted[:-1]
    counts = np.array(
        [
            np.count_nonzero(switches & focal_interior),
            np.count_nonzero(switches & gap_interior),
            np.count_nonzero(switches & seams),
        ]
    )
    assert counts.sum() == np.count_nonzero(switches)
    return counts


fig, axes = plt.subplots(
    2,
    len(datasets),
    figsize=style["cohort_figsize"],
    layout="constrained",
    squeeze=False,
)
switch_rows = []
for column, dataset in enumerate(datasets):
    y = np.arange(len(dataset.cases))
    accumulated = np.zeros(len(y))
    for category, (label, color) in enumerate(
        zip(style["category_labels"], style["category_colors"], strict=True)
    ):
        fractions = np.array(
            [np.mean(case.categories == category) for case in dataset.cases]
        )
        axes[0, column].barh(y, fractions, left=accumulated, color=color, label=label)
        accumulated += fractions
    axes[0, column].set(
        yticks=y,
        yticklabels=[case.label for case in dataset.cases],
        xlim=(0, 1),
        xlabel="Fraction of inference sites",
        title=dataset.label,
    )
    axes[0, column].invert_yaxis()
    axes[0, column].legend(
        fontsize="small", loc="lower left", bbox_to_anchor=(0, 1.10), ncol=2
    )
    for offset, method in zip((-0.2, 0.2), ("full", "stitched"), strict=True):
        parents = [getattr(case, f"{method}_parents") for case in dataset.cases]
        counts = np.array(
            [
                switch_breakdown(case, path)
                for case, path in zip(dataset.cases, parents, strict=True)
            ]
        )
        accumulated = np.zeros(len(y))
        for category, color in enumerate(style["switch_colors"]):
            axes[1, column].barh(
                y + offset,
                counts[:, category],
                left=accumulated,
                height=0.35,
                color=color,
                alpha=1 if method == "stitched" else 0.5,
            )
            accumulated += counts[:, category]
        for case, values in zip(dataset.cases, counts, strict=True):
            switch_rows.append(
                {
                    "dataset": dataset.label,
                    "haplotype": case.label,
                    "method": method,
                    **dict(zip(style["switch_labels"], values, strict=True)),
                }
            )
    axes[1, column].set(
        yticks=y,
        yticklabels=[case.label for case in dataset.cases],
        xlabel="Parent switches (upper: full HMM; lower: stitched)",
    )
    axes[1, column].invert_yaxis()
    handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(
            style["switch_colors"], style["switch_labels"], strict=True
        )
    ]
    axes[1, column].legend(handles=handles, fontsize="small")
fig.suptitle(f"Saved haplotypes at focal AC ≤ {experiment_config['ac_cutoff']}")
plt.show()
switch_table = pd.DataFrame(switch_rows)
ipython_display.display(switch_table)

# %% [markdown]
# ## Extra switch fraction
#
# `num_extra_switches = stitched_switches - full_switches`, and
# `extra_switch_fraction = num_extra_switches / stitched_switches`.
# This reports the net extra switches as a proportion of all stitched switches,
# using the stitched count as the denominator. It is undefined when the stitched
# path has no switches, and can be negative if stitching reduces the count.
# The count alone does not identify which individual switches are unnecessary.

# %%
fig, axes = plt.subplots(
    1,
    len(datasets),
    figsize=style["extra_switch_figsize"],
    layout="constrained",
    squeeze=False,
    sharey=True,
)
for ax, dataset in zip(axes[0], datasets, strict=True):
    fractions = [case.summary["extra_switch_fraction"] for case in dataset.cases]
    bars = ax.bar(np.arange(len(fractions)), fractions, color=style["stitched_color"])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in fractions], padding=3)
    ax.set(
        xticks=np.arange(len(fractions)),
        xticklabels=[case.label for case in dataset.cases],
        ylim=(0, 1),
        title=dataset.label,
    )
axes[0, 0].set_ylabel("Net extra switches / stitched switches")
plt.show()

# %% [markdown]
# ## Where retained focal pieces and gap fills agree with the full path
#
# Each haplotype has two tracks: provenance (retained focal or gap fill), then
# the four-way agreement classification. Genomic widths are a display convention:
# a site's classification extends to the next inference-site position, with one
# configured plotting width at the final site. This is **not** evidence of exact
# focal matching between sites or beyond ancestor support. All counts and rates
# remain site-based. The chromosome's unobserved flanks are excluded from plots.


# %%
def draw_strip(
    ax: plt.Axes, edges: np.ndarray, values: np.ndarray, source: bool = False
) -> None:
    """Draw site categories using their conventional genomic display bins."""
    cmap = source_cmap if source else category_cmap
    norm = source_norm if source else category_norm
    ax.pcolormesh(
        edges,
        [0, 1],
        values[None, :],
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    ax.set_yticks([])


fig, axes = plt.subplots(
    1,
    len(datasets),
    figsize=style["tracks_figsize"],
    layout="constrained",
    squeeze=False,
)
for ax, dataset in zip(axes[0], datasets, strict=True):
    edges = dataset.site_edges / style["genome_unit_bp"]
    for index, case in enumerate(dataset.cases):
        y = index * 2
        ax.pcolormesh(
            edges,
            [y, y + 0.7],
            (~case.accepted)[None, :],
            cmap=source_cmap,
            norm=source_norm,
            shading="flat",
            rasterized=True,
        )
        ax.pcolormesh(
            edges,
            [y + 0.8, y + 1.5],
            case.categories[None, :],
            cmap=category_cmap,
            norm=category_norm,
            shading="flat",
            rasterized=True,
        )
    ax.set(
        yticks=np.arange(len(dataset.cases)) * 2 + 0.75,
        yticklabels=[case.label for case in dataset.cases],
        xlabel=f"Genomic position ({style['genome_unit_label']})",
        title=dataset.label,
    )
    ax.set_xlim(edges[0], edges[-1])
    ax.invert_yaxis()
coverage_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(
        style["source_colors"] + style["category_colors"],
        style["source_labels"] + style["category_labels"],
        strict=True,
    )
]
fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
fig.suptitle("Each haplotype: provenance above, parent agreement below")
plt.show()

# %% [markdown]
# ## Whole-span paths for the selected simulated and real haplotypes
#
# The full HMM and stitched paths use the same original-node ID scale. Node IDs
# identify parents; their vertical spacing has no biological distance meaning.
# The source and agreement tracks show which parts were retained and which were
# rematched. A gap's parent can agree with the full HMM for only part of that gap.


# %%
def plot_overview(dataset: DatasetDiagnostics) -> None:
    """Align provenance, both parent paths, and agreement over the inference span."""
    case = dataset.primary
    edges = dataset.site_edges / style["genome_unit_bp"]
    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=style["overview_figsize"],
        gridspec_kw={"height_ratios": [0.5, 2, 2, 0.5]},
        layout="constrained",
    )
    draw_strip(axes[0], edges, (~case.accepted).astype(int), source=True)
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    for ax, parents, label, color in zip(
        axes[1:3],
        (case.full_parents, case.stitched_parents),
        ("Full HMM", "Stitched"),
        (style["full_color"], style["stitched_color"]),
        strict=True,
    ):
        ax.stairs(parents, edges, baseline=None, color=color, linewidth=0.8)
        ax.set_ylabel(f"{label}\nOriginal node ID")
        ax.grid(alpha=0.2)
    parent_min = min(case.full_parents.min(), case.stitched_parents.min())
    parent_max = max(case.full_parents.max(), case.stitched_parents.max())
    for ax in axes[1:3]:
        ax.set_ylim(parent_min - 1, parent_max + 1)
    draw_strip(axes[3], edges, case.categories)
    axes[3].set_ylabel("Agreement", rotation=0, ha="right")
    axes[3].set_xlabel(f"Genomic position ({style['genome_unit_label']})")
    axes[3].set_xlim(edges[0], edges[-1])
    fig.suptitle(
        f"{dataset.label}: {case.label} | "
        f"agreement {case.summary['fraction_parent_agreement']:.1%} | "
        f"switches {case.summary['full_switches']} → "
        f"{case.summary['stitched_switches']} | "
        f"extra switch prop. {case.summary['extra_switch_fraction']:.1%}"
    )
    fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
    plt.show()


for dataset in datasets:
    plot_overview(dataset)

# %% [markdown]
# ## Gap diagnostics and reproducible zoom selection
#
# Each gap is summarized with its number of sites, agreement fraction, and the
# switch counts strictly inside the gap. Left/right seam switches are reported
# separately. Chromosome-edge gaps have no seam on their outer edge.
#
# For each selected haplotype the notebook chooses four windows from the saved
# site arrays, rather than hand-picking genomic coordinates:
# - **Gap disagreement:** the gap with the most disagreeing sites; ties use NPZ
#   interval order. The zoom is anchored on its first disagreeing site.
# - **Gap agreement:** among gaps of at least the configured minimum size, choose
#   the highest agreement fraction, breaking ties by longer gap then earlier
#   interval. The zoom is anchored on its first agreeing site.
# - **Mixed gap fill:** among gaps with both agreement and disagreement, choose
#   the one with the most disagreeing sites, then the earliest gap. Anchor the
#   zoom on its first within-gap transition between agreement and disagreement.
# - **Focal disagreement:** the retained focal interval with the most disagreeing
#   sites. The zoom is anchored on its first disagreement.
#
# Short target intervals include their full extent and context on both sides.
# Long intervals are clipped to the configured maximum site count around the
# selected anchor, so a zoom may show only part of the target. Dashed lines mark
# target boundaries when they are inside the displayed window. The table reports
# both the complete target and the displayed half-open site window.


# %%
def gap_diagnostics(dataset: DatasetDiagnostics) -> pd.DataFrame:
    """Compare independent fills with the full path, including their seam switches."""
    case = dataset.primary
    rows = []
    for index, (start, end) in enumerate(case.stitched["gap_site_intervals"]):
        row = {
            "gap": index,
            "start_site": start,
            "end_site": end,
            "num_sites": end - start,
            "start_bp": dataset.positions[start],
            "end_bp": dataset.site_edges[end],
            "agree_sites": int(np.count_nonzero(~case.disagreement[start:end])),
            "differ_sites": int(np.count_nonzero(case.disagreement[start:end])),
        }
        row["agreement_fraction"] = row["agree_sites"] / row["num_sites"]
        for method in ("full", "stitched"):
            parents = getattr(case, f"{method}_parents")
            row[f"{method}_interior_switches"] = int(
                np.count_nonzero(parents[start + 1 : end] != parents[start : end - 1])
            )
            row[f"{method}_left_seam_switch"] = (
                bool(parents[start] != parents[start - 1]) if start > 0 else None
            )
            row[f"{method}_right_seam_switch"] = (
                bool(parents[end] != parents[end - 1]) if end < len(parents) else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


@dataclasses.dataclass
class DetailWindow:
    """The target interval and its displayed context, all as half-open site bounds."""

    kind: str
    target_start: int
    target_end: int
    start: int
    end: int


def select_windows(
    dataset: DatasetDiagnostics, gap_table: pd.DataFrame
) -> list[DetailWindow]:
    """Choose disagreement and agreement examples with deterministic site-based rules."""
    case = dataset.primary
    disagree_gap = gap_table.sort_values(
        ["differ_sites", "gap"], ascending=[False, True]
    ).iloc[0]
    eligible = gap_table.loc[
        gap_table.num_sites >= config["detail"]["min_agreement_gap_sites"]
    ]
    agree_gap = eligible.sort_values(
        ["agreement_fraction", "num_sites", "gap"], ascending=[False, False, True]
    ).iloc[0]
    focal_intervals = case.stitched["accepted_site_intervals"]
    focal_counts = [
        np.count_nonzero(case.disagreement[start:end]) for start, end in focal_intervals
    ]
    focal_start, focal_end = focal_intervals[int(np.argmax(focal_counts))]
    intervals = [
        (
            "Gap disagreement",
            int(disagree_gap.start_site),
            int(disagree_gap.end_site),
            True,
        ),
        ("Gap agreement", int(agree_gap.start_site), int(agree_gap.end_site), False),
        ("Focal disagreement", focal_start, focal_end, True),
    ]
    targets = []
    for kind, start, end, look_for_disagreement in intervals:
        target_mask = case.disagreement[start:end] == look_for_disagreement
        assert np.any(target_mask), f"No {kind.lower()} example for {case.label}"
        anchor = start + int(np.flatnonzero(target_mask)[0])
        targets.append((kind, start, end, anchor))
    mixed = gap_table.loc[(gap_table.agree_sites > 0) & (gap_table.differ_sites > 0)]
    mixed_gap = mixed.sort_values(["differ_sites", "gap"], ascending=[False, True]).iloc[
        0
    ]
    start = int(mixed_gap.start_site)
    end = int(mixed_gap.end_site)
    differs = case.disagreement[start:end]
    transitions = np.flatnonzero(differs[1:] != differs[:-1]) + 1
    anchor = start + int(transitions[0])
    targets.insert(2, ("Mixed gap fill", start, end, anchor))
    windows = []
    context = config["detail"]["context_sites"]
    max_sites = config["detail"]["max_sites"]
    for kind, start, end, anchor in targets:
        if end - start + 2 * context <= max_sites:
            left = max(0, start - context)
            right = min(len(case.accepted), end + context)
        else:
            left = max(0, anchor - context)
            right = min(len(case.accepted), left + max_sites)
        windows.append(DetailWindow(kind, start, end, left, right))
    return windows


gap_tables = {}
detail_windows = {}
for dataset in datasets:
    gaps = gap_diagnostics(dataset)
    gap_tables[dataset.name] = gaps
    detail_windows[dataset.name] = select_windows(dataset, gaps)
    print(f"{dataset.label}, {dataset.primary.label}: {len(gaps)} gaps")
    ipython_display.display(
        gaps.sort_values("differ_sites", ascending=False).head(10).round(4)
    )
    window_rows = []
    for window in detail_windows[dataset.name]:
        target_disagreement = dataset.primary.disagreement[
            window.target_start : window.target_end
        ]
        window_rows.append(
            {
                **dataclasses.asdict(window),
                "target_agree_sites": int(np.count_nonzero(~target_disagreement)),
                "target_differ_sites": int(np.count_nonzero(target_disagreement)),
                "display_start_bp": dataset.positions[window.start],
                "display_end_bp": dataset.site_edges[window.end],
            }
        )
    ipython_display.display(pd.DataFrame(window_rows))

# %% [markdown]
# ## Detailed parent paths: retained focal pieces, gap fills, agreement, and difference
#
# These zooms put both paths on **categorical rows labeled by original ancestor
# identity and node ID**. Rows are shared within each panel; matching parent IDs
# occupy exactly the same row. Solid purple is stitched; dashed black is full HMM.
# Markers locate inference sites. Background shading shows provenance, and the
# lower agreement strip distinguishes all four source/agreement combinations.
# The bottom rug marks full-path switches above and stitched-path switches below.
#
# A retained focal piece is exact in observed alleles by construction. A different
# parent there is a copying-path difference, not evidence of an allele mismatch.


# %%
def plot_detail(dataset: DatasetDiagnostics, window: DetailWindow) -> None:
    """Plot both paths on shared categorical parent rows and annotate the target."""
    case = dataset.primary
    start, end = window.start, window.end
    positions = dataset.positions[start:end] / style["detail_unit_bp"]
    edges = dataset.site_edges[start : end + 1] / style["detail_unit_bp"]
    full = case.full_parents[start:end]
    stitched = case.stitched_parents[start:end]
    nodes = np.union1d(full, stitched)
    full_rows = np.searchsorted(nodes, full)
    stitched_rows = np.searchsorted(nodes, stitched)
    figsize = list(style["detail_figsize"])
    figsize[1] = max(figsize[1], len(nodes) * style["parent_row_height_in"] + 2)
    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [0.4, 4, 0.4, 0.6]},
        layout="constrained",
    )
    draw_strip(axes[0], edges, (~case.accepted[start:end]).astype(int), source=True)
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    source = (~case.accepted[start:end]).astype(int)
    changes = np.flatnonzero(source[1:] != source[:-1]) + 1
    run_bounds = np.concatenate(([0], changes, [len(source)]))
    for lo, hi in zip(run_bounds[:-1], run_bounds[1:], strict=True):
        axes[1].axvspan(
            edges[lo],
            edges[hi],
            color=style["source_colors"][source[lo]],
            alpha=0.5,
            linewidth=0,
        )
    axes[1].stairs(
        full_rows,
        edges,
        baseline=None,
        color=style["full_color"],
        linestyle="--",
        linewidth=1.5,
        label="Full HMM",
    )
    axes[1].stairs(
        stitched_rows,
        edges,
        baseline=None,
        color=style["stitched_color"],
        linewidth=1.5,
        label="Stitched",
    )
    axes[1].scatter(
        positions, full_rows, color=style["full_color"], s=12, marker="o", zorder=3
    )
    axes[1].scatter(
        positions,
        stitched_rows,
        color=style["stitched_color"],
        s=16,
        marker="x",
        zorder=4,
    )
    labels = []
    for node_id in nodes:
        metadata = dataset.reference.node(int(node_id)).metadata
        identity = metadata.get("sample_id", "synthetic root")
        labels.append(f"{identity} (node {node_id})")
    axes[1].set(
        yticks=np.arange(len(nodes)),
        yticklabels=labels,
        ylim=(-0.7, len(nodes) - 0.3),
        ylabel="Copying parent",
    )
    axes[1].tick_params(axis="y", labelsize="small")
    axes[1].grid(axis="y", alpha=0.2)
    path_handles = axes[1].get_legend_handles_labels()[0]
    draw_strip(axes[2], edges, case.categories[start:end])
    axes[2].set_ylabel("Agreement", rotation=0, ha="right")
    for parents, level, color in zip(
        (case.full_parents, case.stitched_parents),
        (1, 0),
        (style["full_color"], style["stitched_color"]),
        strict=True,
    ):
        switches = np.flatnonzero(parents[1:] != parents[:-1]) + 1
        switches = switches[(switches >= start) & (switches < end)]
        x = dataset.positions[switches] / style["detail_unit_bp"]
        axes[3].scatter(x, np.full(len(x), level), marker="|", color=color, s=100)
    axes[3].set(
        yticks=[0, 1],
        yticklabels=["Stitched", "Full"],
        ylim=(-0.7, 1.7),
        xlabel=f"Genomic position ({style['detail_unit_label']}); "
        "ticks mark parent switches",
    )
    for boundary in (window.target_start, window.target_end):
        if start <= boundary <= end:
            x = dataset.site_edges[boundary] / style["detail_unit_bp"]
            for ax in axes:
                ax.axvline(x, color="0.3", linestyle=":", linewidth=1)
    axes[-1].set_xlim(edges[0], edges[-1])
    fraction = np.mean(~case.disagreement[start:end])
    target_fraction = np.mean(
        ~case.disagreement[window.target_start : window.target_end]
    )
    fig.suptitle(
        f"{dataset.label}: {case.label} — {window.kind}\n"
        f"Target sites [{window.target_start}, {window.target_end}), "
        f"agreement {target_fraction:.1%}; displayed [{start}, {end}), "
        f"agreement {fraction:.1%}"
    )
    fig.legend(
        handles=coverage_handles + path_handles, loc="outside lower center", ncol=4
    )
    plt.show()


for dataset in datasets:
    for window in detail_windows[dataset.name]:
        plot_detail(dataset, window)

# %% [markdown]
# ## Does disagreement concentrate near focal/gap seams?
#
# Pool the saved haplotypes **within each dataset**, separating retained focal
# sites from gap-fill sites. Distance is the number of inference-site steps to
# the nearest provenance seam, with the immediately adjacent site on either side
# assigned distance zero. Fractions use the actual site counts in each distance
# bin; the accompanying table gives the denominators.
#
# These are selected diagnostic haplotypes, not a representative population
# sample. Proximity is descriptive and does not isolate a causal boundary effect.
# In particular, distant disagreement within retained focal pieces means that
# conditioning gap endpoints alone cannot recover this particular full parent path.


# %%
def seam_distances(accepted: np.ndarray) -> np.ndarray:
    """Compute distances to the two adjacent sites of the nearest provenance seam."""
    seams = np.flatnonzero(accepted[1:] != accepted[:-1]) + 1
    sites = np.arange(len(accepted))
    insertion = np.searchsorted(seams, sites, side="right")
    left = np.full(len(sites), np.inf)
    right = np.full(len(sites), np.inf)
    has_left = insertion > 0
    has_right = insertion < len(seams)
    left[has_left] = sites[has_left] - seams[insertion[has_left] - 1]
    right[has_right] = seams[insertion[has_right]] - 1 - sites[has_right]
    return np.minimum(left, right)


distance_rows = []
bins = np.array(config["distance_bin_edges"])
fig, axes = plt.subplots(
    1,
    len(datasets),
    figsize=style["distance_figsize"],
    layout="constrained",
    squeeze=False,
    sharey=True,
)
for ax, dataset in zip(axes[0], datasets, strict=True):
    for focal, label, color in zip(
        (True, False),
        style["source_labels"],
        (style["category_colors"][1], style["category_colors"][3]),
        strict=True,
    ):
        totals = np.zeros(len(bins) - 1, dtype=int)
        differences = np.zeros(len(bins) - 1, dtype=int)
        for case in dataset.cases:
            distance = seam_distances(case.accepted)
            selected = case.accepted == focal
            totals += np.histogram(distance[selected], bins=bins)[0]
            differences += np.histogram(
                distance[selected & case.disagreement], bins=bins
            )[0]
        rate = np.divide(
            differences, totals, out=np.full(len(totals), np.nan), where=totals > 0
        )
        labels = [f"{lo}–{hi - 1}" for lo, hi in zip(bins[:-1], bins[1:], strict=True)]
        ax.plot(np.arange(len(rate)), rate, marker="o", label=label, color=color)
        for lo, hi, count, difference, fraction in zip(
            bins[:-1], bins[1:], totals, differences, rate, strict=True
        ):
            distance_rows.append(
                {
                    "dataset": dataset.label,
                    "source": label,
                    "distance_start": lo,
                    "distance_end": hi,
                    "num_sites": count,
                    "disagree_sites": difference,
                    "disagreement_fraction": fraction,
                }
            )
    ax.set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        ylim=(0, 1),
        xlabel="Distance to nearest seam (inference-site steps)",
        title=dataset.label,
    )
    ax.tick_params(axis="x", labelrotation=60)
    ax.grid(alpha=0.2)
    ax.legend()
axes[0, 0].set_ylabel("Fraction of sites with different parents")
plt.show()
distance_table = pd.DataFrame(distance_rows)
ipython_display.display(distance_table.loc[distance_table.num_sites > 0].round(4))

# %% [markdown]
# ## Gap determination
#
# The latest match-eval commit (`cbb7b57`) writes compact focal-ancestor **interval
# NPZs**, replacing the old chunk CSV generation. The corresponding analysis
# commit (`294486b`) sweeps those intervals for coverage. We use these new archives
# and export coverage segments and both sets of gap bounds as CSVs below.
# Older `*_focal_ancestor_chunks.csv` files are not inputs to this comparison.
#
# The two definitions differ:
# - **HMM-extracted gaps** are maximal runs not accepted from the reduced HMM
#   path. Acceptance is checked independently at each site: the observed call
#   must equal the copied allele, the parent must be an eligible focal candidate,
#   and the site must lie inside that ancestor's support. A mismatch is rejected
#   at that site, but the same parent can be accepted again beyond the mismatch.
#   An accepted piece need not contain that ancestor's focal seed.
# - **Zero-coverage gaps** are maximal runs outside the union of zero-mismatch
#   intervals for all eligible ancestors. Each interval extends from the
#   ancestor's **leftmost focal seed** and stops at the first mismatch or missing
#   call in each direction, or at support/inference-interval bounds. It does not
#   restart on a later exact island. This terminates the anchored extension;
#   it does not globally remove the ancestor from all haplotypes.
#
# Both methods use the same saved haplotypes, reference-site axis, and inclusive
# upper AC cutoff. The coverage analysis also excludes AC=1, as its current
# `summarise_coverage` implementation does. We check that no saved focal candidate
# falls below that minimum here, so the eligibility sets agree for these runs.
# Coverage uses generated ancestor calls; the HMM uses the inferred reference.
# Their inferred/imputed alleles may also differ at missing ancestor calls.
#
# Overlap is measured in **inference sites**. Report the intersection, each
# method's exclusive sites, the fraction of each gap mask overlapped, and gap
# intersection/union (IoU). Counts of maximal intervals are separate: fragmented
# HMM gaps can overlap a much smaller number of long zero-coverage gaps. We also
# count gaps with any overlap and exactly equal half-open bounds. No HMM path is
# changed in this diagnostic comparison.


# %%
@dataclasses.dataclass
class GapComparison:
    """Coverage counts, zero-coverage intervals, and overlap with one saved gap mask."""

    case: HaplotypeDiagnostics
    coverage: np.ndarray
    zero_gaps: np.ndarray
    categories: np.ndarray
    summary: dict


def mask_intervals(mask: np.ndarray) -> np.ndarray:
    """Find maximal true runs including leading, trailing, empty, and complete masks."""
    padded = np.pad(mask, (1, 1), constant_values=False)
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    return transitions.reshape(-1, 2)


def gaps_with_overlap(intervals: np.ndarray, other_gap_mask: np.ndarray) -> int:
    """Count intersecting intervals in linear time using site-mask prefix sums."""
    prefix = np.concatenate(([0], np.cumsum(other_gap_mask)))
    overlap_sites = prefix[intervals[:, 1]] - prefix[intervals[:, 0]]
    return int(np.count_nonzero(overlap_sites > 0))


def gap_bound_rows(
    key: dict,
    intervals: np.ndarray,
    dataset: DatasetDiagnostics,
) -> list[dict]:
    """Export exact site bounds; positions use the plotting convention above."""
    return [
        {
            **key,
            "gap_index": index,
            "left_site_index": int(start),
            "right_site_index": int(end),
            "num_sites": int(end - start),
            "left_position": float(dataset.positions[start]),
            "right_position": float(dataset.site_edges[end]),
        }
        for index, (start, end) in enumerate(intervals)
    ]


def compare_gap_determination(dataset: DatasetDiagnostics) -> list[GapComparison]:
    """Sweep new budget-zero intervals and compare every saved haplotype's gap bounds."""
    name = dataset.name
    archive_path = coverage_input_dir / "haplotype_intervals"
    archive_path /= f"{name}_inferred_focal_ancestor_intervals.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        intervals = {key: archive[key] for key in archive.files}
    num_sites = len(dataset.positions)
    assert int(intervals["num_sites"]) == num_sites
    cutoff = experiment_config["ac_cutoff"]
    assert int(intervals["max_ac_cutoff"]) >= cutoff
    budget = config["coverage"]["mismatch_budget"]
    assert budget == 0
    assert int(intervals["max_mismatches"]) >= budget
    panel = zarr.open_group(
        coverage_input_dir / "ancestors" / f"{name}_inferred_ancestors.zarr", mode="r"
    )
    assert np.array_equal(panel["variant_position"][:], dataset.positions)
    with np.load(
        input_dir / "focal_ancestors" / f"{name}_inferred_focal_ancestors.npz",
        allow_pickle=False,
    ) as focal:
        for field in ("sample_id", "ploidy_index"):
            assert np.array_equal(intervals[field], focal[field])
        minimum_ac = config["coverage"]["min_focal_ac"]
        for case in dataset.cases:
            row = np.flatnonzero(
                (focal["sample_id"] == case.summary["sample_id"])
                & (focal["ploidy_index"] == case.summary["ploidy_index"])
            )[0]
            candidates = focal["ancestor_index"][
                focal["offsets"][row] : focal["offsets"][row + 1]
            ]
            counts = focal["derived_ac"][candidates]
            assert not np.any((counts < minimum_ac) & (counts <= cutoff))
    keys = list(zip(intervals["sample_id"], intervals["ploidy_index"], strict=True))
    assert len(set(keys)) == len(keys)
    row_by_identity = {key: row for row, key in enumerate(keys)}
    results = []
    zero_gap_rows = []
    hmm_gap_rows = []
    coverage_rows = []
    for case in dataset.cases:
        row = row_by_identity[(case.summary["sample_id"], case.summary["ploidy_index"])]
        start, end = intervals["offsets"][row : row + 2]
        counts = intervals["focal_ac"][start:end]
        eligible = (counts >= minimum_ac) & (counts <= cutoff)
        left = intervals["left_site_index"][start:end, budget][eligible]
        right = intervals["right_site_index"][start:end, budget][eligible]
        assert np.all((left >= 0) & (left < right) & (right <= num_sites))
        delta = np.zeros(num_sites + 1, dtype=np.int64)
        np.add.at(delta, left, 1)
        np.add.at(delta, right, -1)
        cumulative = np.cumsum(delta)
        assert cumulative[-1] == 0 and np.all(cumulative >= 0)
        coverage = cumulative[:-1]
        zero_mask = coverage == 0
        zero_gaps = mask_intervals(zero_mask)
        assert np.array_equal(interval_mask(zero_gaps.tolist(), num_sites), zero_mask)
        hmm_mask = ~case.accepted
        hmm_gaps = np.asarray(case.stitched["gap_site_intervals"], dtype=int).reshape(
            -1, 2
        )
        assert np.array_equal(mask_intervals(hmm_mask), hmm_gaps)
        intersection = int(np.count_nonzero(hmm_mask & zero_mask))
        hmm_only = int(np.count_nonzero(hmm_mask & ~zero_mask))
        zero_only = int(np.count_nonzero(zero_mask & ~hmm_mask))
        neither = int(np.count_nonzero(~hmm_mask & ~zero_mask))
        assert intersection + hmm_only + zero_only + neither == num_sites
        union = intersection + hmm_only + zero_only
        hmm_sites = intersection + hmm_only
        zero_sites = intersection + zero_only
        assert hmm_sites == case.summary["gap_sites"]
        zero_bounds = {tuple(bound) for bound in zero_gaps}
        key = {
            field: case.summary[field]
            for field in (
                "dataset",
                "source",
                "sample_id",
                "ploidy_index",
                "ac_cutoff",
            )
        }
        key["max_mismatches"] = budget
        metrics = {
            **key,
            "num_sites": num_sites,
            "num_coverage_intervals": int(eligible.sum()),
            "hmm_num_gaps": len(hmm_gaps),
            "zero_coverage_num_gaps": len(zero_gaps),
            "hmm_gap_sites": hmm_sites,
            "zero_coverage_gap_sites": zero_sites,
            "overlap_sites": intersection,
            "hmm_only_gap_sites": hmm_only,
            "zero_only_gap_sites": zero_only,
            "neither_gap_sites": neither,
            "gap_site_iou": intersection / union if union > 0 else np.nan,
            "hmm_gap_overlap_fraction": intersection / hmm_sites
            if hmm_sites > 0
            else np.nan,
            "zero_coverage_gap_overlap_fraction": intersection / zero_sites
            if zero_sites > 0
            else np.nan,
            "hmm_gaps_with_overlap": gaps_with_overlap(hmm_gaps, zero_mask),
            "zero_coverage_gaps_with_overlap": gaps_with_overlap(zero_gaps, hmm_mask),
            "exact_gap_bound_matches": sum(
                tuple(bound) in zero_bounds for bound in hmm_gaps
            ),
        }
        categories = np.full(num_sites, 3, dtype=np.int8)
        categories[hmm_mask & zero_mask] = 0
        categories[hmm_mask & ~zero_mask] = 1
        categories[zero_mask & ~hmm_mask] = 2
        results.append(GapComparison(case, coverage, zero_gaps, categories, metrics))
        zero_gap_rows.extend(gap_bound_rows(key, zero_gaps, dataset))
        hmm_gap_rows.extend(gap_bound_rows(key, hmm_gaps, dataset))
        changes = np.flatnonzero(coverage[1:] != coverage[:-1]) + 1
        bounds = np.concatenate(([0], changes, [num_sites]))
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            coverage_rows.append(
                {
                    **key,
                    "left_site_index": int(lo),
                    "right_site_index": int(hi),
                    "num_sites": int(hi - lo),
                    "coverage": int(coverage[lo]),
                }
            )
    directory = gap_comparison_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(directory / "coverage_segments.csv", index=False)
    pd.DataFrame(zero_gap_rows).to_csv(directory / "zero_coverage_gaps.csv", index=False)
    pd.DataFrame(hmm_gap_rows).to_csv(directory / "hmm_extracted_gaps.csv", index=False)
    pd.DataFrame([result.summary for result in results]).to_csv(
        directory / "gap_comparison.csv", index=False
    )
    return results


gap_comparisons = {
    dataset.name: compare_gap_determination(dataset) for dataset in datasets
}
gap_summary = pd.DataFrame(
    [result.summary for results in gap_comparisons.values() for result in results]
)
ipython_display.display(gap_summary.round(4))

# %%
fig, axes = plt.subplots(
    1,
    len(datasets),
    figsize=style["gap_counts_figsize"],
    layout="constrained",
    squeeze=False,
)
for ax, dataset in zip(axes[0], datasets, strict=True):
    results = gap_comparisons[dataset.name]
    x = np.arange(len(results))
    for offset, field, label, color in zip(
        (-0.2, 0.2),
        ("hmm_num_gaps", "zero_coverage_num_gaps"),
        style["gap_method_labels"],
        style["gap_method_colors"],
        strict=True,
    ):
        counts = [result.summary[field] for result in results]
        bars = ax.bar(x + offset, counts, width=0.35, color=color, label=label)
        ax.bar_label(bars, padding=3)
    ax.set(
        xticks=x,
        xticklabels=[result.case.label for result in results],
        title=dataset.label,
        ylabel="Number of maximal gap intervals",
    )
    ax.margins(y=0.15)
    ax.legend()
fig.suptitle(
    f"Gap counts for the same haplotypes, AC ≤ {experiment_config['ac_cutoff']}, "
    "zero mismatches"
)
plt.show()

# %%
fig, ax = plt.subplots(figsize=style["gap_counts_figsize"], layout="constrained")
x = np.arange(len(datasets))
for offset, field, label, color in zip(
    (-0.2, 0.2),
    ("hmm_num_gaps", "zero_coverage_num_gaps"),
    style["gap_method_labels"],
    style["gap_method_colors"],
    strict=True,
):
    counts = [
        sum(result.summary[field] for result in gap_comparisons[dataset.name])
        for dataset in datasets
    ]
    bars = ax.bar(x + offset, counts, width=0.35, label=label, color=color)
    ax.bar_label(bars, padding=3)
ax.set(
    xticks=x,
    xticklabels=[
        f"{dataset.label}\n{len(dataset.cases)} saved haplotypes" for dataset in datasets
    ],
    ylabel="Total number of gap intervals across saved haplotypes",
    title="Total gaps: HMM extraction versus zero-mismatch coverage",
)
ax.margins(y=0.15)
ax.legend()
plt.show()

# %%
fig, axes = plt.subplots(
    1,
    len(datasets),
    figsize=style["gap_overlap_figsize"],
    layout="constrained",
    squeeze=False,
    sharex=True,
)
for ax, dataset in zip(axes[0], datasets, strict=True):
    results = gap_comparisons[dataset.name]
    y = np.arange(len(results))
    left = np.zeros(len(y))
    for field, label, color in zip(
        (
            "overlap_sites",
            "hmm_only_gap_sites",
            "zero_only_gap_sites",
            "neither_gap_sites",
        ),
        style["gap_overlap_labels"],
        style["gap_overlap_colors"],
        strict=True,
    ):
        fractions = [
            result.summary[field] / result.summary["num_sites"] for result in results
        ]
        ax.barh(y, fractions, left=left, label=label, color=color)
        left += fractions
    ax.set(
        yticks=y,
        yticklabels=[result.case.label for result in results],
        xlim=(0, 1),
        xlabel="Fraction of inference sites",
        title=dataset.label,
    )
    ax.invert_yaxis()
handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(
        style["gap_overlap_colors"],
        style["gap_overlap_labels"],
        strict=True,
    )
]
fig.legend(handles=handles, loc="outside lower center", ncol=4)
plt.show()

# %%
overlap_cmap = mcolors.ListedColormap(style["gap_overlap_colors"])
overlap_norm = mcolors.BoundaryNorm(np.arange(5) - 0.5, overlap_cmap.N)
for dataset in datasets:
    result = next(
        item for item in gap_comparisons[dataset.name] if item.case is dataset.primary
    )
    edges = dataset.site_edges / style["genome_unit_bp"]
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=style["gap_bounds_figsize"],
        layout="constrained",
    )
    for ax, mask, label, color in zip(
        axes[:2],
        (~result.case.accepted, result.coverage == 0),
        style["gap_method_labels"],
        style["gap_method_colors"],
        strict=True,
    ):
        cmap = mcolors.ListedColormap([style["gap_overlap_colors"][-1], color])
        ax.pcolormesh(
            edges, [0, 1], mask[None, :], cmap=cmap, vmin=0, vmax=1, rasterized=True
        )
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
    axes[2].pcolormesh(
        edges,
        [0, 1],
        result.categories[None, :],
        cmap=overlap_cmap,
        norm=overlap_norm,
        rasterized=True,
    )
    axes[2].set(
        yticks=[],
        xlim=(edges[0], edges[-1]),
        xlabel=f"Genomic position ({style['genome_unit_label']})",
    )
    axes[2].set_ylabel("Overlap", rotation=0, ha="right", va="center")
    fig.suptitle(
        f"{dataset.label}: {result.case.label} | "
        f"gap-site IoU {result.summary['gap_site_iou']:.1%} | "
        f"{result.summary['exact_gap_bound_matches']} exactly matching gap bounds"
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    fig.canvas.draw()
    plt.show()

# %% [markdown]
# HMM-only gap sites have at least one zero-mismatch focal interval available,
# but the reduced HMM's selected parent was not accepted there. Zero-coverage-only
# gap sites were accepted by the original stitching calculation despite lying
# outside every focal-anchored zero-mismatch interval. These are direct evidence
# that the two acceptance definitions differ, rather than an endpoint plotting
# convention. The comparison keeps their original definitions intact.

# %%
for dataset in datasets:
    results = gap_comparisons[dataset.name]
    hmm_count = sum(result.summary["hmm_num_gaps"] for result in results)
    zero_count = sum(result.summary["zero_coverage_num_gaps"] for result in results)
    intersection = sum(result.summary["overlap_sites"] for result in results)
    hmm_only = sum(result.summary["hmm_only_gap_sites"] for result in results)
    zero_only = sum(result.summary["zero_only_gap_sites"] for result in results)
    iou = intersection / (intersection + hmm_only + zero_only)
    print(
        f"{dataset.label}: {hmm_count} HMM gaps versus {zero_count} zero-coverage gaps; "
        f"site-weighted gap IoU {iou:.1%}; "
        f"{zero_only:,} accepted sites outside all anchored exact intervals."
    )
print("CSV directory (relative to diagnostics YAML):", config["gap_comparison_dir"])

# %% [markdown]
# ## Interpretation and next diagnostic questions
#
# The saved comparison tables report identical mutation records for these runs:
# both paths have zero called mismatches. Their score differences therefore come
# entirely from extra switches in the stitched paths. Report their net count as
# a proportion of stitched switches rather than the raw score difference.
# Parent disagreement remains
# meaningful even when the observed alleles are equally well explained.
#
# Use the switch decomposition and gap zooms to identify seams worth a boundary-
# conditioned follow-up. Use the focal-disagreement zooms to inspect fixed pieces
# that already select a different ancestor than the full HMM. A boundary-conditioned
# gap matcher cannot change those retained site parents, so recovering the exact
# full parent path would also require reconsidering which focal pieces are fixed.
# These diagnostics compare the two saved paths; they do not establish which
# copying genealogy is biologically correct or exclude equivalent optimal paths.

# %%
assert summary.mutations_identical.all()
assert (summary.full_mismatches == 0).all()
assert (summary.stitched_mismatches == 0).all()
assert (summary.num_extra_switches > 0).all()
for dataset in datasets:
    case = dataset.primary
    full_counts = switch_breakdown(case, case.full_parents)
    stitched_counts = switch_breakdown(case, case.stitched_parents)
    extra = stitched_counts - full_counts
    print(
        f"{dataset.label}, {case.label}: "
        f"parent agreement {case.summary['fraction_parent_agreement']:.1%}; "
        f"extra switches {int(extra.sum())} = "
        f"focal {extra[0]} + gaps {extra[1]} + seams {extra[2]}; "
        f"extra switch prop. {case.summary['extra_switch_fraction']:.1%}"
    )
