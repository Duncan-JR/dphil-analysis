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

# %%
import dataclasses
import json
import logging
import pathlib

import IPython.display as ipython_display
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import tskit
import yaml

logger = logging.getLogger(__name__)

# %%
notebook_dir = (
    pathlib.Path(__file__).resolve().parent
    if "__file__" in globals()
    else pathlib.Path.cwd()
)
diagnostic_config_path = (
    notebook_dir.parent.parent / "experiments/ch5/stitching_diagnostics_config.yaml"
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
style = config["plot"]
plt.rcParams.update({"figure.dpi": style["dpi"], "font.size": style["font_size"]})
category_cmap = mcolors.ListedColormap(style["category_colors"])
category_norm = mcolors.BoundaryNorm(np.arange(5) - 0.5, category_cmap.N)
source_cmap = mcolors.ListedColormap(style["source_colors"])
source_norm = mcolors.BoundaryNorm(np.arange(3) - 0.5, source_cmap.N)

# %% [markdown]
# # Extra switches vs hyperparameters
#
# The saved cohort contains only AC cutoff 10, and its outputs do not record
# whether zero or one mismatches per side were allowed. The requested mismatch
# panels are therefore marked unavailable. The final row shows the aggregate
# for the current saved configuration without assigning it to either setting.

# %%
hyperparameter_output_dir = pathlib.Path(config["cohort"]["output_dir"]).expanduser()
if not hyperparameter_output_dir.is_absolute():
    hyperparameter_output_dir = diagnostic_config_path.parent / hyperparameter_output_dir
hyperparameter_results = pd.read_csv(
    hyperparameter_output_dir / "haplotype_results.csv"
)
switch_counts = hyperparameter_results.pivot(
    index=["dataset", "sample_id", "ploidy_index", "population", "ac_cutoff"],
    columns="method",
    values="num_switches",
).reset_index()
switch_counts["num_extra_switches"] = switch_counts.stitched - switch_counts.full
full_switch_counts = switch_counts.full.replace(0, np.nan)
switch_counts["switch_increase_percentage"] = (
    100 * switch_counts.num_extra_switches / full_switch_counts
)
hyperparameter_metrics = [
    ("num_extra_switches", "Mean extra switches"),
    ("switch_increase_percentage", "Mean percentage increase"),
]
requested_mismatch_settings = [0, 1]
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=style["population_results"]["hyperparameter_figsize"],
    sharex=True,
)
available_cutoffs = sorted(switch_counts.ac_cutoff.unique())
for column, mismatch_count in enumerate(requested_mismatch_settings):
    for row, (metric, ylabel) in enumerate(hyperparameter_metrics):
        ax = axes[row, column]
        ax.text(
            0.5,
            0.5,
            "No results with a recorded\nmismatch allowance",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"{mismatch_count} mismatches allowed per side")
        ax.set_ylabel(ylabel)
        ax.set_xticks(available_cutoffs)
for ax in axes[-1]:
    ax.set_xlabel("AC cutoff")
fig.suptitle("Extra switches by mismatch allowance")
fig.tight_layout()
plt.show()

# %% [markdown]
# The saved cohort has one AC cutoff. These means and standard errors describe
# the current configuration across all populations; the mismatch allowance is
# not recorded in its output metadata.

# %%
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=style["population_results"]["current_config_figsize"],
)
for ax, (metric, ylabel) in zip(axes, hyperparameter_metrics, strict=True):
    summary = switch_counts.groupby("ac_cutoff")[metric].agg(["mean", "sem"])
    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["sem"],
        marker="o",
        capsize=style["error_bar_capsize"],
    )
    ax.set_xlabel("AC cutoff")
    ax.set_ylabel(ylabel)
    ax.set_xticks(summary.index)
    if metric == "switch_increase_percentage":
        undefined_count = switch_counts[metric].isna().sum()
        ax.text(
            0.98,
            0.96,
            f"{undefined_count} undefined increases excluded",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
fig.suptitle("Current saved configuration (mean ± SEM)")
fig.tight_layout()
plt.show()


# %%
@dataclasses.dataclass
class HaplotypeDiagnostics:
    """Site-based evidence and original-node parents for one saved comparison."""

    summary: dict
    full: dict
    stitched: dict
    full_parents: np.ndarray
    stitched_parents: np.ndarray
    covered: np.ndarray
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
    selected = {
        (item["sample_id"], item["ploidy_index"]) for item in settings["haplotypes"]
    }
    cases = []
    for summary in summaries:
        key = (summary["sample_id"], summary["ploidy_index"])
        if key not in selected:
            continue
        summary["num_extra_switches"] = (
            summary["stitched_switches"] - summary["full_switches"]
        )
        summary["extra_switch_fraction"] = (
            summary["num_extra_switches"] / summary["stitched_switches"]
            if summary["stitched_switches"] > 0
            else np.nan
        )
        full = full_records[key]
        stitched = stitched_records[key]
        for record in (full, stitched):
            for field in (
                "dataset",
                "source",
                "sample_id",
                "ploidy_index",
                "ac_cutoff",
            ):
                assert record[field] == summary[field]
        assert summary["dataset"] == name
        assert summary["ac_cutoff"] == experiment_config["ac_cutoff"]
        full_parents = parents_at_sites(full, reference)
        stitched_parents = parents_at_sites(stitched, reference)
        covered = interval_mask(stitched["focal_site_intervals"], len(positions))
        gaps = interval_mask(stitched["gap_site_intervals"], len(positions))
        assert np.array_equal(gaps, ~covered)
        disagreement = full_parents != stitched_parents
        assert summary["num_sites"] == len(positions)
        assert summary["focal_sites"] == np.count_nonzero(covered)
        assert summary["gap_sites"] == np.count_nonzero(gaps)
        assert summary["num_gaps"] == len(stitched["gap_site_intervals"])
        assert summary["focal_disagreement_sites"] == np.count_nonzero(
            disagreement & covered
        )
        assert summary["gap_disagreement_sites"] == np.count_nonzero(
            disagreement & gaps
        )
        assert np.isclose(summary["fraction_parent_agreement"], np.mean(~disagreement))
        assert summary["full_switches"] == len(full["path"]) - 1
        assert summary["stitched_switches"] == len(stitched["path"]) - 1
        assert summary["mutations_identical"] == (
            full["mutations"] == stitched["mutations"]
        )
        categories = (~covered).astype(np.int8) * 2 + disagreement.astype(np.int8)
        cases.append(
            HaplotypeDiagnostics(
                summary,
                full,
                stitched,
                full_parents,
                stitched_parents,
                covered,
                disagreement,
                categories,
            )
        )
    assert len(cases) == len(selected)
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


def switch_breakdown(case: HaplotypeDiagnostics, parents: np.ndarray) -> np.ndarray:
    """Partition switches into focal interiors, gap interiors, and provenance seams."""
    switches = parents[1:] != parents[:-1]
    focal_interior = case.covered[1:] & case.covered[:-1]
    gap_interior = ~case.covered[1:] & ~case.covered[:-1]
    seams = case.covered[1:] != case.covered[:-1]
    counts = np.array(
        [
            np.count_nonzero(switches & focal_interior),
            np.count_nonzero(switches & gap_interior),
            np.count_nonzero(switches & seams),
        ]
    )
    assert counts.sum() == np.count_nonzero(switches)
    return counts


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


coverage_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(
        style["source_colors"] + style["category_colors"],
        style["source_labels"] + style["category_labels"],
        strict=True,
    )
]


def plot_overview(dataset: DatasetDiagnostics) -> plt.Figure:
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
    draw_strip(axes[0], edges, (~case.covered).astype(int), source=True)
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
        f"extra switches {case.summary['num_extra_switches']}"
    )
    fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
    return fig


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
    """Choose agreement and disagreement examples using site-based rules."""
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
    focal_intervals = case.stitched["focal_site_intervals"]
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
    mixed_gap = mixed.sort_values(
        ["differ_sites", "gap"], ascending=[False, True]
    ).iloc[0]
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
            right = min(len(case.covered), end + context)
        else:
            left = max(0, anchor - context)
            right = min(len(case.covered), left + max_sites)
        windows.append(DetailWindow(kind, start, end, left, right))
    return windows


def plot_detail(
    dataset: DatasetDiagnostics, window: DetailWindow, *, site_axis: bool = False
) -> plt.Figure:
    """Plot both paths on shared categorical parent rows and annotate the target."""
    case = dataset.primary
    start, end = window.start, window.end
    positions = dataset.positions[start:end] / style["detail_unit_bp"]
    edges = dataset.site_edges[start : end + 1] / style["detail_unit_bp"]
    if site_axis:
        positions = np.arange(start, end)
        edges = np.arange(start, end + 1)
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
    draw_strip(axes[0], edges, (~case.covered[start:end]).astype(int), source=True)
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    source = (~case.covered[start:end]).astype(int)
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
        x = switches if site_axis else dataset.positions[switches] / style["detail_unit_bp"]
        axes[3].scatter(x, np.full(len(x), level), marker="|", color=color, s=100)
    position_label = f"Genomic position ({style['detail_unit_label']})"
    if site_axis:
        position_label = "Inference-site index"
    axes[3].set(
        yticks=[0, 1],
        yticklabels=["Stitched", "Full"],
        ylim=(-0.7, 1.7),
        xlabel=f"{position_label}; ticks mark parent switches",
    )
    for boundary in (window.target_start, window.target_end):
        if start <= boundary <= end:
            x = boundary if site_axis else dataset.site_edges[boundary] / style["detail_unit_bp"]
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
    return fig


def seam_distances(covered: np.ndarray) -> np.ndarray:
    """Compute distances to the two adjacent sites of the nearest provenance seam."""
    seams = np.flatnonzero(covered[1:] != covered[:-1]) + 1
    sites = np.arange(len(covered))
    insertion = np.searchsorted(seams, sites, side="right")
    left = np.full(len(sites), np.inf)
    right = np.full(len(sites), np.inf)
    has_left = insertion > 0
    has_right = insertion < len(seams)
    left[has_left] = sites[has_left] - seams[insertion[has_left] - 1]
    right[has_right] = seams[insertion[has_right]] - 1 - sites[has_right]
    return np.minimum(left, right)


@dataclasses.dataclass
class CaseSummaryFigure:
    """Summary figure and its switch-count table."""

    figure: plt.Figure
    switches: pd.DataFrame


def plot_case_summary(datasets: list[DatasetDiagnostics], title: str) -> CaseSummaryFigure:
    """Compare site compositions and switch-type counts for selected haplotypes."""
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
    fig.suptitle(title)
    switch_table = pd.DataFrame(switch_rows)
    return CaseSummaryFigure(fig, switch_table)


def plot_case_tracks(datasets: list[DatasetDiagnostics], title: str) -> plt.Figure:
    """Align provenance and parent agreement for all selected haplotypes."""
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
                (~case.covered)[None, :],
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
    fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
    fig.suptitle(title)
    return fig


# %% [markdown]
# # Population-level results
#
# Compare all saved haplotypes, grouped by population, in the simulation and
# 1KGP datasets. Each observation is one haplotype. Settings and plot styles
# live in `experiments/ch5/stitching_diagnostics_config.yaml`.

# %%
population_style = style["population_results"]
population_output_dir = pathlib.Path(config["cohort"]["output_dir"]).expanduser()
if not population_output_dir.is_absolute():
    population_output_dir = diagnostic_config_path.parent / population_output_dir
population_output_dir.mkdir(parents=True, exist_ok=True)
population_tables = []
for settings in config["datasets"]:
    table = pd.read_csv(output_dir / settings["name"] / "comparison.csv")
    assert not table.duplicated(["sample_id", "ploidy_index"]).any()
    assert len(table) == config["cohort"]["num_haplotypes"]
    assert table.dataset.eq(settings["name"]).all()
    assert table.population.notna().all()
    assert set(table.population) == set(settings["populations"])
    population_tables.append(table)
population_comparisons = pd.concat(population_tables, ignore_index=True)
population_comparisons["num_extra_switches"] = (
    population_comparisons.stitched_switches - population_comparisons.full_switches
)
full_switch_denominator = population_comparisons.full_switches.replace(0, np.nan)
population_comparisons["switch_increase_percentage"] = (
    100 * population_comparisons.num_extra_switches / full_switch_denominator
)
for region in ("focal", "gap"):
    region_sites = population_comparisons[f"{region}_sites"].replace(0, np.nan)
    disagreeing_sites = population_comparisons[f"{region}_disagreement_sites"]
    population_comparisons[f"{region}_agreement_percentage"] = (
        100 * (1 - disagreeing_sites / region_sites)
    )

def plot_population_histograms(
    metric: str,
    xlabel: str,
    title: str,
    *,
    integer_bins: bool = False,
    bounded_percentage: bool = False,
) -> plt.Figure:
    """Show distributions, every defined haplotype, KDEs, and population means.

    Bins and axes are shared by populations within each dataset. KDEs are
    reflected at zero and, for agreement percentages, at 100 to keep density
    inside the metric's support. Zero denominators remain missing observations.
    """
    fig, axes = plt.subplots(
        len(config["datasets"][0]["populations"]),
        len(config["datasets"]),
        figsize=population_style["histogram_figsize"],
        layout="constrained",
        sharex="col",
        sharey="col",
        squeeze=False,
    )
    rng = np.random.default_rng(population_style["random_seed"])
    for column, settings in enumerate(config["datasets"]):
        dataset_table = population_comparisons.loc[
            population_comparisons.dataset.eq(settings["name"])
        ]
        dataset_values = dataset_table[metric].dropna().to_numpy()
        lower = min(0, dataset_values.min())
        upper = dataset_values.max()
        if bounded_percentage:
            lower, upper = 0, 100
        if integer_bins:
            bin_width = population_style["count_bin_width"]
            bins = np.arange(lower - bin_width / 2, upper + bin_width, bin_width)
        else:
            if not bounded_percentage:
                upper += max(upper - lower, 1) * population_style["range_padding"]
            bins = np.linspace(lower, upper, population_style["histogram_bins"] + 1)
        grid = np.linspace(lower, upper, population_style["kde_grid_size"])
        density_max = 0
        for row, population in enumerate(settings["populations"]):
            ax = axes[row, column]
            color = population_style["population_colors"][population]
            observations = dataset_table.loc[dataset_table.population.eq(population), metric]
            values = observations.dropna().to_numpy()
            excluded = observations.isna().sum()
            densities, _ = np.histogram(values, bins=bins, density=True)
            density_max = max(density_max, densities.max())
            ax.hist(
                values,
                bins=bins,
                density=True,
                color=color,
                alpha=population_style["histogram_alpha"],
                edgecolor="white",
            )
            if len(values) > 1 and np.ptp(values) > 0:
                kde = stats.gaussian_kde(values, bw_method=population_style["kde_bandwidth"])
                density = kde(grid)
                if lower == 0:
                    density += kde(-grid)
                if bounded_percentage:
                    density += kde(200 - grid)
                ax.plot(
                    grid,
                    density,
                    color=color,
                    linewidth=population_style["line_width"],
                )
                density_max = max(density_max, density.max())
            mean = values.mean()
            ax.axvline(
                mean,
                color=color,
                linestyle="--",
                linewidth=population_style["line_width"],
            )
            annotation_x = 0.02 if bounded_percentage else 0.98
            annotation_alignment = "left" if bounded_percentage else "right"
            ax.text(
                annotation_x,
                0.95,
                f"{population}: n = {len(values)}; excluded = {excluded}\nMean = {mean:.2f}",
                transform=ax.transAxes,
                ha=annotation_alignment,
                va="top",
            )
            ax.set_ylabel("Density")
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title(settings["label"])
            if row == len(settings["populations"]) - 1:
                ax.set_xlabel(xlabel)
        strip_depth = density_max * population_style["point_strip_depth"]
        for row, population in enumerate(settings["populations"]):
            ax = axes[row, column]
            color = population_style["population_colors"][population]
            values = dataset_table.loc[dataset_table.population.eq(population), metric]
            values = values.dropna().to_numpy()
            point_y = -rng.uniform(strip_depth / 4, strip_depth * 3 / 4, len(values))
            ax.scatter(
                values,
                point_y,
                s=population_style["point_size"],
                alpha=population_style["point_alpha"],
                color=color,
                linewidths=0,
            )
            ax.axhline(0, color=style["full_color"], linewidth=0.5)
            density_upper = density_max * 1.25
            tick_locator = mticker.MaxNLocator(
                nbins=population_style["density_tick_count"]
            )
            density_ticks = tick_locator.tick_values(0, density_upper)
            density_ticks = density_ticks[
                (density_ticks >= 0) & (density_ticks <= density_upper)
            ]
            ax.set_yticks(density_ticks)
            ax.set(xlim=(bins[0], bins[-1]), ylim=(-strip_depth, density_upper))
    fig.suptitle(f"{title}\nSolid: kernel density; dashed: mean; points: haplotypes")
    return fig


# %% [markdown]
# ## Extra switches by population
#
# Extra switches are `stitched_switches - full_switches`. Histograms show the
# distribution across haplotypes; each point beneath a histogram represents one
# haplotype. Solid curves show kernel densities and dashed lines show means.

# %%
fig = plot_population_histograms(
    "num_extra_switches", "Extra switches", "Extra switches by population", integer_bins=True
)
fig.savefig(population_output_dir / "population_extra_switch_histograms.png")
plt.show()

# %% [markdown]
# ## Percentage increase in switches by population
#
# Percentage increase is
# `100 * (stitched_switches - full_switches) / full_switches`.
# It is undefined when the full HMM has no switches, including when both paths
# have zero switches. Each panel reports the number of undefined observations
# excluded from its histogram, points, kernel density, and mean.

# %%
fig = plot_population_histograms(
    "switch_increase_percentage",
    "Increase in switches over full HMM (%)",
    "Percentage increase in switches by population",
)
fig.savefig(population_output_dir / "population_switch_increase_histograms.png")
plt.show()

# %% [markdown]
# ## Gap properties
#
# Each observation is one saved haplotype. The histograms pool populations within
# each dataset; solid curves show kernel densities and dashed lines show means.
# Points are coloured consistently by population: blue for YRI/AFR, orange for
# CEU/EUR, and gray for CHB/EAS. Mean sites per gap is `gap_sites / num_gaps` and
# is undefined for haplotypes with no gaps. Exclusion counts appear in each panel.

# %%
gap_denominator = population_comparisons.num_gaps.replace(0, np.nan)
population_comparisons["mean_sites_per_gap"] = (
    population_comparisons.gap_sites / gap_denominator
)
gap_population_handles = [
    mpatches.Patch(
        color=population_style["population_colors"][population], label=label
    )
    for population, label in (
        ("YRI", "YRI / AFR"),
        ("CEU", "CEU / EUR"),
        ("CHB", "CHB / EAS"),
    )
]


def plot_gap_property_histograms(
    metric: str, xlabel: str, title: str, *, integer_bins: bool = False
) -> plt.Figure:
    """Pool haplotype gap properties by dataset, retaining population point colours."""
    fig, axes = plt.subplots(
        1, len(config["datasets"]),
        figsize=population_style["gap_histogram_figsize"],
        layout="constrained", squeeze=False,
    )
    rng = np.random.default_rng(population_style["random_seed"])
    for ax, settings in zip(axes[0], config["datasets"], strict=True):
        dataset_table = population_comparisons.loc[
            population_comparisons.dataset.eq(settings["name"])
        ]
        observations = dataset_table[metric]
        values = observations.dropna().to_numpy()
        excluded = observations.isna().sum()
        upper = values.max()
        if integer_bins:
            bin_width = population_style["count_bin_width"]
            bins = np.arange(-bin_width / 2, upper + bin_width, bin_width)
        else:
            upper += upper * population_style["range_padding"]
            bins = np.linspace(0, upper, population_style["histogram_bins"] + 1)
        grid = np.linspace(0, upper, population_style["kde_grid_size"])
        densities, _ = np.histogram(values, bins=bins, density=True)
        density_max = densities.max()
        ax.hist(
            values, bins=bins, density=True,
            color=population_style["gap_histogram_color"],
            alpha=population_style["histogram_alpha"], edgecolor="white",
        )
        if len(values) > 1 and np.ptp(values) > 0:
            kde = stats.gaussian_kde(
                values, bw_method=population_style["kde_bandwidth"]
            )
            density = kde(grid) + kde(-grid)
            ax.plot(
                grid, density, color=style["full_color"],
                linewidth=population_style["line_width"],
            )
            density_max = max(density_max, density.max())
        mean = values.mean()
        ax.axvline(
            mean, color=style["full_color"], linestyle="--",
            linewidth=population_style["line_width"],
        )
        ax.text(
            0.98, 0.95,
            f"n = {len(values)}; excluded = {excluded}\nMean = {mean:.2f}",
            transform=ax.transAxes, ha="right", va="top",
        )
        strip_depth = density_max * population_style["point_strip_depth"]
        for population in settings["populations"]:
            population_values = dataset_table.loc[
                dataset_table.population.eq(population), metric
            ]
            population_values = population_values.dropna().to_numpy()
            point_y = -rng.uniform(
                strip_depth / 4, strip_depth * 3 / 4, len(population_values)
            )
            ax.scatter(
                population_values, point_y,
                color=population_style["population_colors"][population],
                s=population_style["point_size"],
                alpha=population_style["point_alpha"], linewidths=0,
            )
        density_upper = density_max * 1.25
        tick_locator = mticker.MaxNLocator(
            nbins=population_style["density_tick_count"]
        )
        density_ticks = tick_locator.tick_values(0, density_upper)
        density_ticks = density_ticks[
            (density_ticks >= 0) & (density_ticks <= density_upper)
        ]
        ax.set_yticks(density_ticks)
        ax.set(
            title=settings["label"], xlabel=xlabel, ylabel="Density",
            xlim=(bins[0], bins[-1]), ylim=(-strip_depth, density_upper),
        )
        ax.axhline(0, color=style["full_color"], linewidth=0.5)
        ax.grid(axis="y", alpha=0.2)
    fig.legend(handles=gap_population_handles, loc="outside lower center", ncol=3)
    fig.suptitle(f"{title}\nSolid: kernel density; dashed: mean; points: haplotypes")
    return fig


fig = plot_gap_property_histograms(
    "num_gaps", "Number of gaps", "Number of gaps per haplotype", integer_bins=True
)
fig.savefig(population_output_dir / "population_number_of_gaps_histograms.png")
plt.show()

# %%
fig = plot_gap_property_histograms(
    "mean_sites_per_gap", "Mean sites per gap", "Mean sites per gap per haplotype"
)
fig.savefig(population_output_dir / "population_mean_sites_per_gap_histograms.png")
plt.show()

# %% [markdown]
# Compare each haplotype's mean gap length with its number of eligible focal
# ancestors (top row) and number of gaps (bottom row). Haplotypes without gaps
# are excluded.

# %%
fig, axes = plt.subplots(
    2, len(config["datasets"]), figsize=population_style["gap_scatter_figsize"],
    layout="constrained", squeeze=False, sharey="col",
)
for column, settings in enumerate(config["datasets"]):
    dataset_table = population_comparisons.loc[
        population_comparisons.dataset.eq(settings["name"])
    ]
    defined_table = dataset_table.dropna(subset=["mean_sites_per_gap"])
    excluded = len(dataset_table) - len(defined_table)
    for row, (metric, xlabel) in enumerate(
        (("num_focal_ancestors", "Number of focal ancestors"),
         ("num_gaps", "Number of gaps"))
    ):
        ax = axes[row, column]
        for population in settings["populations"]:
            observations = defined_table.loc[
                defined_table.population.eq(population)
            ]
            ax.scatter(
                observations[metric], observations.mean_sites_per_gap,
                color=population_style["population_colors"][population],
                s=population_style["point_size"],
                alpha=population_style["point_alpha"], linewidths=0,
            )
        ax.set(xlabel=xlabel, ylabel="Mean sites per gap")
        ax.text(
            0.98, 0.95, f"n = {len(defined_table)}; excluded = {excluded}",
            transform=ax.transAxes, ha="right", va="top",
        )
        if defined_table[metric].nunique() == 1:
            ax.set_xticks(defined_table[metric].unique())
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=0.2)
        if row == 0:
            ax.set_title(settings["label"])
fig.legend(handles=gap_population_handles, loc="outside lower center", ncol=3)
fig.suptitle("Gap length and focal ancestor availability")
fig.savefig(population_output_dir / "population_gap_properties_scatter.png")
plt.show()

# %% [markdown]
# Relate the number of extra switches to the number of gaps, focal ancestors,
# and mean sites per gap. Each row is one dataset; points are haplotypes coloured
# by population. Haplotype-level gaps and extra switches are integer counts.

# %%
gap_predictors = [
    ("num_gaps", "Number of gaps"),
    ("num_focal_ancestors", "Number of focal ancestors"),
    ("mean_sites_per_gap", "Mean sites per gap"),
]
fig, axes = plt.subplots(
    len(config["datasets"]), len(gap_predictors),
    figsize=population_style["extra_switch_scatter_figsize"],
    layout="constrained", squeeze=False, sharey="row",
)
for row, settings in enumerate(config["datasets"]):
    row_label = population_style["scatter_dataset_row_labels"][row]
    dataset_table = population_comparisons.loc[
        population_comparisons.dataset.eq(settings["name"])
    ]
    defined_table = dataset_table.dropna(subset=["mean_sites_per_gap"])
    excluded = len(dataset_table) - len(defined_table)
    for column, (metric, xlabel) in enumerate(gap_predictors):
        ax = axes[row, column]
        for population in settings["populations"]:
            observations = defined_table.loc[
                defined_table.population.eq(population)
            ]
            ax.scatter(
                observations[metric], observations.num_extra_switches,
                color=population_style["population_colors"][population],
                s=population_style["point_size"],
                alpha=population_style["point_alpha"], linewidths=0,
            )
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        if column == 0:
            ax.set_ylabel("Extra switches")
        if column == 0:
            ax.set_title(f"{row_label} · {xlabel}")
        elif row == 0:
            ax.set_title(xlabel)
        ax.text(
            0.98, 0.95, f"n = {len(defined_table)}; excluded = {excluded}",
            transform=ax.transAxes, ha="right", va="top",
        )
        ax.grid(alpha=0.2)
fig.legend(handles=gap_population_handles, loc="outside lower center", ncol=3)
fig.suptitle("Extra switches and gap properties by population")
fig.savefig(population_output_dir / "population_extra_switch_gap_scatter.png")
plt.show()

# %% [markdown]
# ## Decomposing switches by type
#
# Switches are partitioned into those within focal coverage, within gaps, and
# at focal/gap seams, using the same region boundaries for both methods.
# Each haplotype's counts are divided by its own method's total switches before
# averaging within populations. The stacked bars therefore describe mean
# haplotype compositions, and each bar sums to 100%. Haplotypes with zero
# switches have undefined compositions and are excluded, as noted below.

# %%
population_switch_categories = ["focal", "gap", "seam"]
population_switch_tables = []
for method in ("full", "stitched"):
    method_table = population_comparisons[
        ["dataset", "sample_id", "ploidy_index", "population"]
    ].copy()
    method_table["method"] = method
    totals = population_comparisons[f"{method}_switches"]
    counts = population_comparisons[
        [f"{method}_{category}_switches" for category in population_switch_categories]
    ]
    assert counts.sum(axis=1).eq(totals).all()
    denominator = totals.replace(0, np.nan)
    for category in population_switch_categories:
        category_counts = population_comparisons[f"{method}_{category}_switches"]
        method_table[category] = category_counts / denominator
    defined = totals.gt(0)
    assert np.allclose(method_table.loc[defined, population_switch_categories].sum(axis=1), 1)
    population_switch_tables.append(method_table)
population_switch_proportions = pd.concat(population_switch_tables, ignore_index=True)
population_switch_means = population_switch_proportions.groupby(
    ["dataset", "population", "method"]
)[population_switch_categories].mean()
population_switch_excluded = population_switch_proportions.assign(
    excluded=population_switch_proportions.focal.isna()
).groupby(["dataset", "population", "method"]).excluded.sum()
excluded_compositions = population_switch_excluded.rename(
    "Undefined compositions excluded"
)
ipython_display.display(excluded_compositions.to_frame())

fig, axes = plt.subplots(
    1, len(config["datasets"]), figsize=population_style["composition_figsize"],
    layout="constrained", squeeze=False, sharex=True,
)
for ax, settings in zip(axes[0], config["datasets"], strict=True):
    populations = settings["populations"]
    y = np.arange(len(populations))
    for method_index, method in enumerate(("full", "stitched")):
        offset = (2 * method_index - 1) * population_style["bar_offset"]
        centers = y + offset
        keys = [(settings["name"], population, method) for population in populations]
        means = population_switch_means.loc[keys]
        left = np.zeros(len(populations))
        for category, color in zip(
            population_switch_categories, style["switch_colors"], strict=True
        ):
            widths = means[category].to_numpy() * 100
            ax.barh(centers, widths, left=left, height=population_style["bar_height"], color=color)
            left += widths
    method_ticks = np.column_stack(
        (y - population_style["bar_offset"], y + population_style["bar_offset"])
    ).ravel()
    method_labels = [
        f"{population} · {method}"
        for population in populations
        for method in ("Full", "Stitched")
    ]
    ax.set(
        yticks=method_ticks, yticklabels=method_labels, xlim=(0, 100),
        xlabel="Mean proportion of switches (%)", title=settings["label"],
        xticks=population_style["percentage_ticks"],
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
population_switch_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(style["switch_colors"], population_style["switch_labels"], strict=True)
]
fig.legend(handles=population_switch_handles, loc="outside lower center", ncol=3)
fig.suptitle("Switch composition by population: mean of haplotype proportions")
fig.savefig(population_output_dir / "population_switch_compositions.png")
plt.show()

# %% [markdown]
# The ternary plots show the same compositions for individual haplotypes.
# A corner represents 100% of its switch type; the opposite edge represents 0%.
# Colours identify populations, consistently across datasets: blue for YRI/AFR,
# orange for CEU/EUR, and gray for CHB/EAS. Circles show full HMM haplotypes and
# triangles show stitched HMM haplotypes. Large outlined markers show population
# means, labelled by population. Undefined compositions are excluded as in the bars.

# %%
def ternary_coordinates(proportions: np.ndarray) -> np.ndarray:
    """Map focal/gap/seam proportions to left/right/top triangle vertices."""
    x = proportions[:, 1] + proportions[:, 2] / 2
    y = proportions[:, 2] * np.sqrt(3) / 2
    return np.column_stack((x, y))


fig, axes = plt.subplots(
    1, len(config["datasets"]), figsize=population_style["ternary_figsize"],
    layout="constrained", squeeze=False,
)
triangle_height = np.sqrt(3) / 2
for ax, settings in zip(axes[0], config["datasets"], strict=True):
    ax.plot([0, 1, 0.5, 0], [0, 0, triangle_height, 0], color=style["full_color"])
    for fraction in population_style["ternary_grid_fractions"]:
        for category in range(len(population_switch_categories)):
            endpoints = np.zeros((2, len(population_switch_categories)))
            endpoints[:, category] = fraction
            other_categories = [
                index
                for index in range(len(population_switch_categories))
                if index != category
            ]
            endpoints[0, other_categories[0]] = 1 - fraction
            endpoints[1, other_categories[1]] = 1 - fraction
            coordinates = ternary_coordinates(endpoints)
            ax.plot(
                coordinates[:, 0], coordinates[:, 1],
                color="lightgray", linewidth=0.7, zorder=0,
            )
        ax.text(
            fraction, -0.025, f"{1 - fraction:.0%}",
            ha="center", va="top", fontsize="small",
        )
        ax.text(
            0.5 * fraction - 0.02, triangle_height * fraction,
            f"{fraction:.0%}", ha="right", fontsize="small",
        )
        ax.text(
            1 - 0.5 * fraction + 0.02, triangle_height * fraction,
            f"{1 - fraction:.0%}", ha="left", fontsize="small",
        )
    for method in ("full", "stitched"):
        for population, label_offset in zip(
            settings["populations"],
            population_style["ternary_mean_label_offsets"],
            strict=True,
        ):
            color = population_style["population_colors"][population]
            selected = population_switch_proportions.dataset.eq(settings["name"])
            selected &= population_switch_proportions.method.eq(method)
            selected &= population_switch_proportions.population.eq(population)
            proportions = population_switch_proportions.loc[
                selected, population_switch_categories
            ]
            proportions = proportions.dropna().to_numpy()
            coordinates = ternary_coordinates(proportions)
            ax.scatter(
                coordinates[:, 0], coordinates[:, 1], color=color,
                marker=population_style["ternary_method_markers"][method],
                s=population_style["ternary_point_size"],
                alpha=population_style["ternary_point_alpha"], linewidths=0,
            )
            mean = population_switch_means.loc[(settings["name"], population, method)].to_numpy()
            coordinate = ternary_coordinates(mean[np.newaxis, :])[0]
            ax.scatter(
                coordinate[0], coordinate[1], color=color,
                marker=population_style["ternary_method_markers"][method],
                s=population_style["ternary_mean_size"], edgecolors="white", zorder=3,
            )
            ax.annotate(
                population, coordinate, xytext=label_offset,
                textcoords="offset points", color=color, fontsize="small",
                ha="center", va="center",
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7},
            )
    ax.text(0, -0.07, "Within focal\ncoverage", ha="center", va="top")
    ax.text(1, -0.07, "Within gaps", ha="center", va="top")
    ax.text(0.5, triangle_height + 0.03, "At focal/gap seams", ha="center", va="bottom")
    ax.set(
        title=settings["label"], aspect="equal", xlim=(-0.15, 1.15),
        ylim=(-0.16, triangle_height + 0.1),
    )
    ax.set_axis_off()
population_method_handles = [
    mlines.Line2D(
        [], [], color=style["full_color"], linestyle="none",
        marker=population_style["ternary_method_markers"][method], label=label,
    )
    for method, label in (
        ("full", "Full HMM"),
        ("stitched", "Stitched HMM"),
    )
]
population_color_handles = [
    mpatches.Patch(
        color=population_style["population_colors"][population], label=label
    )
    for population, label in (
        ("YRI", "YRI / AFR"),
        ("CEU", "CEU / EUR"),
        ("CHB", "CHB / EAS"),
    )
]
fig.legend(
    handles=population_color_handles + population_method_handles,
    loc="outside lower center", ncol=5,
)
fig.suptitle("Switch compositions: haplotypes and population means (large markers)")
fig.savefig(population_output_dir / "population_switch_ternary.png")
plt.show()

# %% [markdown]
# ## Parent agreement proportions
#
# Each strip shows the mean proportions of all inference sites assigned to
# focal/agree, focal/disagree, gaps/agree, and gaps/disagree. The four fractions
# share the same denominator and sum to 100%. Focal and gap segments are kept
# together, so their boundary also shows mean focal coverage.

# %%
population_agreement_columns = [
    "focal_agree_fraction",
    "focal_disagree_fraction",
    "gap_agree_fraction",
    "gap_disagree_fraction",
]
assert np.allclose(population_comparisons[population_agreement_columns].sum(axis=1), 1)
population_agreement_means = population_comparisons.groupby(
    ["dataset", "population"]
)[population_agreement_columns].mean()
fig, axes = plt.subplots(
    1, len(config["datasets"]), figsize=population_style["composition_figsize"],
    layout="constrained", squeeze=False, sharex=True,
)
for ax, settings in zip(axes[0], config["datasets"], strict=True):
    populations = settings["populations"]
    y = np.arange(len(populations))
    keys = [(settings["name"], population) for population in populations]
    means = population_agreement_means.loc[keys]
    left = np.zeros(len(populations))
    for category, color in zip(
        population_agreement_columns, style["category_colors"], strict=True
    ):
        widths = means[category].to_numpy() * 100
        ax.barh(y, widths, left=left, height=population_style["bar_height"] * 2, color=color)
        left += widths
    ax.set(
        yticks=y, yticklabels=populations, xlim=(0, 100),
        xlabel="Mean proportion of all inference sites (%)",
        title=settings["label"], xticks=population_style["percentage_ticks"],
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
population_agreement_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(
        style["category_colors"], population_style["agreement_labels"], strict=True
    )
]
fig.legend(handles=population_agreement_handles, loc="outside lower center", ncol=4)
fig.suptitle("Parent agreement proportions by population")
fig.savefig(population_output_dir / "population_parent_agreement_strips.png")
plt.show()

# %% [markdown]
# ## Parent agreement in focal regions
#
# Agreement within focal regions is
# `100 * (focal_sites - focal_disagreement_sites) / focal_sites`.
# Each haplotype contributes its own percentage; agreement and disagreement
# within its focal regions sum to 100%. Haplotypes without focal sites have
# undefined agreement percentages and are excluded, with counts in each panel.

# %%
fig = plot_population_histograms(
    "focal_agreement_percentage", "Parent agreement within focal regions (%)",
    "Parent agreement in focal regions by population", bounded_percentage=True,
)
fig.savefig(population_output_dir / "population_focal_agreement_histograms.png")
plt.show()

# %% [markdown]
# ## Parent agreement in gaps
#
# Agreement within gaps is
# `100 * (gap_sites - gap_disagreement_sites) / gap_sites`.
# Each haplotype contributes its own percentage; agreement and disagreement
# within its gaps sum to 100%. Haplotypes without gap sites have undefined
# agreement percentages and are excluded, with counts in each panel.

# %%
fig = plot_population_histograms(
    "gap_agreement_percentage", "Parent agreement within gaps (%)",
    "Parent agreement in gaps by population", bounded_percentage=True,
)
fig.savefig(population_output_dir / "population_gap_agreement_histograms.png")
plt.show()


# %% [markdown]
# # Analysis of outliers
#
# Select one haplotype per dataset for each criterion: most extra switches,
# lowest parent agreement within focal regions, and lowest parent agreement
# within gaps. Undefined agreement percentages are excluded from selection.
# Ties are resolved by sample ID, then ploidy index. The table identifies the
# populations and the denominators before inspecting any copying paths.

# %%
outlier_rows = []
for settings in config["datasets"]:
    table = population_comparisons.loc[
        population_comparisons.dataset.eq(settings["name"])
    ]
    for criterion in config["outliers"]["criteria"]:
        metric = criterion["metric"]
        eligible = table.dropna(subset=[metric])
        ordered = eligible.sort_values(
            [metric, "sample_id", "ploidy_index"],
            ascending=[criterion["ascending"], True, True],
        )
        selected = ordered.iloc[0].to_dict()
        selected["criterion"] = criterion["label"]
        selected["tied_haplotypes"] = int(eligible[metric].eq(selected[metric]).sum())
        selected["haplotype"] = f"{selected['sample_id']}/{selected['ploidy_index']}"
        outlier_rows.append(selected)
outlier_selection = pd.DataFrame(outlier_rows)
ipython_display.display(
    outlier_selection[
        [
            "dataset", "criterion", "haplotype", "population", "num_extra_switches",
            "focal_agreement_percentage", "focal_sites", "gap_agreement_percentage",
            "gap_sites", "num_gaps", "tied_haplotypes",
        ]
    ].round(2)
)
outlier_output_dir = pathlib.Path(config["outliers"]["output_dir"]).expanduser()
if not outlier_output_dir.is_absolute():
    outlier_output_dir = diagnostic_config_path.parent / outlier_output_dir
outlier_output_dir.mkdir(parents=True, exist_ok=True)
outlier_selection.to_csv(outlier_output_dir / "selected_haplotypes.csv", index=False)
outlier_datasets = []
for settings in config["datasets"]:
    selected = outlier_selection.loc[outlier_selection.dataset.eq(settings["name"])]
    identities = selected[["sample_id", "ploidy_index"]].drop_duplicates()
    haplotypes = identities.to_dict("records")
    outlier_settings = settings | {"haplotypes": haplotypes, "deep_dive": haplotypes[0]}
    dataset = load_dataset(outlier_settings)
    cases_by_identity = {
        (case.summary["sample_id"], case.summary["ploidy_index"]): case
        for case in dataset.cases
    }
    dataset.cases = [
        cases_by_identity[(identity["sample_id"], identity["ploidy_index"])]
        for identity in haplotypes
    ]
    outlier_datasets.append(dataset)
outlier_case_datasets = [
    dataclasses.replace(dataset, primary=case)
    for dataset in outlier_datasets
    for case in dataset.cases
]

# %% [markdown]
# ## Summary plots
#
# Compare each outlier's four site categories and full/stitched switch counts,
# then locate its focal regions, gaps, and parent disagreement along the genome.
# Switch-type colours match the population plots. Whole-span parent IDs identify
# copying parents; their vertical spacing has no biological distance meaning.

# %%
outlier_summary = plot_case_summary(
    outlier_datasets, "Outlier haplotypes: site proportions and switch counts"
)
outlier_summary.figure.savefig(outlier_output_dir / "outlier_site_and_switch_summary.png")
plt.show()
ipython_display.display(outlier_summary.switches)
outlier_summary.switches.to_csv(outlier_output_dir / "outlier_switch_counts.csv", index=False)
fig = plot_case_tracks(outlier_datasets, "Outliers: provenance above, parent agreement below")
fig.savefig(outlier_output_dir / "outlier_provenance_and_agreement.png")
plt.show()

# %%
for dataset in outlier_case_datasets:
    case = dataset.primary
    prefix = f"{dataset.name}_{case.summary['sample_id'].lower()}_{case.summary['ploidy_index']}"
    fig = plot_overview(dataset)
    fig.savefig(outlier_output_dir / f"{prefix}_whole_span_paths.png")
    plt.show()

# %% [markdown]
# ## Deep dive
#
# A switch has a position and a directed parent transition, such as A → B.
# Losing a focal-interior switch and gaining a seam switch can therefore mean
# that the same transition moved, or that the ordered parent sequence changed.
# We distinguish these using maximal runs of parent disagreement, including one
# agreeing site on each side when both flanks exist:
#
# - **Identical switch:** same position and same parents on both sides.
# - **Shifted parent sequence:** both flanking parents agree and the complete
#   ordered parent sequences are identical, while switch positions differ.
# - **Changed parent sequence:** flanking parents agree but the ordered sequences
#   differ. Some individual transitions may still be shared within these regions.
# - **Unanchored difference:** a disagreement reaches an inference-site boundary,
#   so there is no pair of agreeing flanks for this comparison.
#
# The switch-count partition includes every full and stitched switch. Proven
# relocations preserve parent IDs and direction; they are not inferred by pairing
# whichever switch happens to be closest. Counts associated with a gap include
# its interior and its existing left/right seams. They are signed net differences,
# and omit switches in focal interiors. This distinction matters when a switch
# moves between the two types of region.

# %%
@dataclasses.dataclass
class ParentSequence:
    """Ordered parent IDs and the site boundaries between them."""

    nodes: np.ndarray
    switch_sites: np.ndarray


@dataclasses.dataclass
class OutlierPathAnalysis:
    """Gap evidence, parent differences, proven relocations, and switch totals."""

    gaps: pd.DataFrame
    divergences: pd.DataFrame
    relocations: pd.DataFrame
    switch_counts: pd.DataFrame


def parent_sequence(parents: np.ndarray, start: int, end: int) -> ParentSequence:
    """Compress parents on [start, end), retaining absolute switch sites."""
    changes = np.flatnonzero(parents[start + 1 : end] != parents[start : end - 1])
    switch_sites = changes + start + 1
    run_starts = np.concatenate(([start], switch_sites))
    return ParentSequence(parents[run_starts], switch_sites)


def switch_location(case: HaplotypeDiagnostics, site: int) -> str:
    """Classify the boundary immediately before a given inference site."""
    if case.covered[site] != case.covered[site - 1]:
        return "seam"
    return "focal" if case.covered[site] else "gap"


def analyse_outlier_paths(dataset: DatasetDiagnostics) -> OutlierPathAnalysis:
    """Distinguish positional changes from changes in the ordered parent sequence.

    Maximal disagreement runs are compared with one agreeing site on either
    side. Equal compressed parent sequences prove preservation of the directed
    parent transitions within that run. Differences reaching the chromosome's
    inference-site boundary are reported separately because a shared flank is
    absent. Counts partition every switch; no nearest-switch pairing is used.
    """
    case = dataset.primary
    gaps = gap_diagnostics(dataset)
    full_changes = case.full_parents[1:] != case.full_parents[:-1]
    stitched_changes = case.stitched_parents[1:] != case.stitched_parents[:-1]
    matching_edges = ~case.disagreement[1:] & ~case.disagreement[:-1]
    identical = full_changes & stitched_changes & matching_edges
    kinds = config["outliers"]["plot"]["transition_kinds"]
    counts = np.zeros((len(kinds), 2), dtype=int)
    counts[0] = np.count_nonzero(identical)
    padded = np.concatenate(([False], case.disagreement, [False]))
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    runs = boundaries.reshape(-1, 2)
    difference_classes = np.zeros(len(case.covered), dtype=np.int8)
    divergence_rows = []
    relocation_rows = []
    for index, (start, end) in enumerate(runs):
        left = max(0, int(start) - 1)
        right = min(len(case.covered), int(end) + 1)
        full = parent_sequence(case.full_parents, left, right)
        stitched = parent_sequence(case.stitched_parents, left, right)
        shared_flanks = start > 0 and end < len(case.covered)
        equal_sequences = np.array_equal(full.nodes, stitched.nodes)
        if not shared_flanks:
            kind_index = 3
        elif equal_sequences:
            kind_index = 1
        else:
            kind_index = 2
        counts[kind_index, 0] += len(full.switch_sites)
        counts[kind_index, 1] += len(stitched.switch_sites)
        difference_classes[start:end] = kind_index
        divergence_rows.append(
            {
                "difference": index,
                "start_site": int(start),
                "end_site": int(end),
                "start_bp": dataset.positions[start],
                "end_bp": dataset.site_edges[end],
                "num_sites": int(end - start),
                "focal_sites": int(np.count_nonzero(case.covered[start:end])),
                "gap_sites": int(np.count_nonzero(~case.covered[start:end])),
                "kind": kinds[kind_index],
                "full_parent_sequence": " → ".join(str(node) for node in full.nodes),
                "stitched_parent_sequence": " → ".join(str(node) for node in stitched.nodes),
                "full_switches": len(full.switch_sites),
                "stitched_switches": len(stitched.switch_sites),
                "extra_switches": len(stitched.switch_sites) - len(full.switch_sites),
            }
        )
        if shared_flanks and equal_sequences:
            for offset, (full_site, stitched_site) in enumerate(
                zip(full.switch_sites, stitched.switch_sites, strict=True)
            ):
                relocation_rows.append(
                    {
                        "difference": index,
                        "from_parent": int(full.nodes[offset]),
                        "to_parent": int(full.nodes[offset + 1]),
                        "full_site": int(full_site),
                        "stitched_site": int(stitched_site),
                        "full_bp": dataset.positions[full_site],
                        "stitched_bp": dataset.positions[stitched_site],
                        "full_location": switch_location(case, int(full_site)),
                        "stitched_location": switch_location(case, int(stitched_site)),
                        "shift_sites": int(stitched_site - full_site),
                    }
                )
    assert counts[:, 0].sum() == np.count_nonzero(full_changes)
    assert counts[:, 1].sum() == np.count_nonzero(stitched_changes)
    switch_rows = []
    for kind, values in zip(kinds, counts, strict=True):
        switch_rows.append(
            {"kind": kind, "full": values[0], "stitched": values[1]}
        )
    for method in ("full", "stitched"):
        interior = gaps[f"{method}_interior_switches"]
        left_seams = gaps[f"{method}_left_seam_switch"].fillna(False).astype(int)
        right_seams = gaps[f"{method}_right_seam_switch"].fillna(False).astype(int)
        gaps[f"{method}_gap_and_seam_switches"] = interior + left_seams + right_seams
        counts_by_source = switch_breakdown(case, getattr(case, f"{method}_parents"))
        assert gaps[f"{method}_gap_and_seam_switches"].sum() == counts_by_source[1:].sum()
    gaps["extra_interior_switches"] = (
        gaps.stitched_interior_switches - gaps.full_interior_switches
    )
    gaps["extra_gap_and_seam_switches"] = (
        gaps.stitched_gap_and_seam_switches - gaps.full_gap_and_seam_switches
    )
    gaps["extra_switches_per_site"] = gaps.extra_gap_and_seam_switches / gaps.num_sites
    for row in gaps.itertuples():
        start, end = row.start_site, row.end_site
        for side, site in (("left", start - 1), ("right", end)):
            parent_agrees = None
            if 0 <= site < len(case.covered):
                parent_agrees = bool(not case.disagreement[site])
            gaps.loc[row.Index, f"{side}_flank_parent_agrees"] = parent_agrees
        for kind_index, label in ((1, "shifted"), (2, "changed"), (3, "unanchored")):
            classes = difference_classes[start:end]
            gaps.loc[row.Index, f"{label}_sequence_sites"] = np.count_nonzero(classes == kind_index)
    relocations = pd.DataFrame(
        relocation_rows,
        columns=[
            "difference", "from_parent", "to_parent", "full_site", "stitched_site",
            "full_bp", "stitched_bp", "full_location", "stitched_location", "shift_sites",
        ],
    )
    return OutlierPathAnalysis(
        gaps, pd.DataFrame(divergence_rows), relocations, pd.DataFrame(switch_rows)
    )


def outlier_window(
    dataset: DatasetDiagnostics, kind: str, start: int, end: int, anchor: int
) -> DetailWindow:
    """Apply the shared context and size limits to an outlier target."""
    context = config["detail"]["context_sites"]
    max_sites = config["detail"]["max_sites"]
    if end - start + 2 * context <= max_sites:
        left = max(0, start - context)
        right = min(len(dataset.positions), end + context)
    else:
        left = max(0, anchor - context)
        right = min(len(dataset.positions), left + max_sites)
    return DetailWindow(kind, start, end, left, right)


def select_outlier_windows(
    dataset: DatasetDiagnostics, analysis: OutlierPathAnalysis
) -> list[DetailWindow]:
    """Find short-gap excesses, poor agreement, altered itineraries, and relocations."""
    case = dataset.primary
    gaps = analysis.gaps
    windows = []
    short = gaps.loc[
        gaps.num_sites.le(config["outliers"]["short_gap_max_sites"])
        & gaps.extra_gap_and_seam_switches.gt(0)
    ]
    if not short.empty:
        chosen = short.sort_values(
            ["extra_switches_per_site", "extra_gap_and_seam_switches", "start_site"],
            ascending=[False, False, True],
        ).iloc[0]
        start, end = int(chosen.start_site), int(chosen.end_site)
        windows.append(outlier_window(dataset, "Short gap with extra switches", start, end, start))
    poor = gaps.sort_values(
        ["agreement_fraction", "differ_sites", "start_site"],
        ascending=[True, False, True],
    ).iloc[0]
    start, end = int(poor.start_site), int(poor.end_site)
    differences = np.flatnonzero(case.disagreement[start:end])
    anchor = start + int(differences[0])
    windows.append(outlier_window(dataset, "Lowest gap agreement", start, end, anchor))
    focal_difference = case.covered & case.disagreement
    padded = np.concatenate(([False], focal_difference, [False]))
    boundaries = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    lengths = boundaries[:, 1] - boundaries[:, 0]
    start, end = boundaries[np.argmax(lengths)]
    windows.append(
        outlier_window(
            dataset, "Longest focal parent disagreement", int(start), int(end), int(start)
        )
    )
    changed = analysis.divergences.loc[
        analysis.divergences.kind.eq(config["outliers"]["plot"]["transition_kinds"][2])
        & analysis.divergences.gap_sites.gt(0)
    ]
    if not changed.empty:
        chosen = changed.sort_values(
            ["extra_switches", "gap_sites", "start_site"], ascending=[False, False, True]
        ).iloc[0]
        start, end = int(chosen.start_site), int(chosen.end_site)
        intersecting_gaps = gaps.loc[gaps.start_site.lt(end) & gaps.end_site.gt(start)]
        target_gap = intersecting_gaps.sort_values(
            ["extra_gap_and_seam_switches", "start_site"], ascending=[False, True]
        ).iloc[0]
        anchor = max(start, int(target_gap.start_site))
        windows.append(
            outlier_window(dataset, "Changed parent sequence around a gap", start, end, anchor)
        )
    relocations = analysis.relocations.copy()
    if not relocations.empty:
        relocations["focal_to_gap_or_seam"] = (
            relocations.full_location.eq("focal") & relocations.stitched_location.ne("focal")
        )
        relocations["absolute_shift"] = relocations.shift_sites.abs()
        chosen = relocations.sort_values(
            ["focal_to_gap_or_seam", "absolute_shift", "full_site"],
            ascending=[False, False, True],
        ).iloc[0]
        start = min(int(chosen.full_site), int(chosen.stitched_site))
        end = max(int(chosen.full_site), int(chosen.stitched_site)) + 1
        context = config["detail"]["context_sites"]
        left = max(0, start - context)
        right = min(len(case.covered), end + context)
        windows.append(DetailWindow("Same parents with a relocated switch", start, end, left, right))
    unique_windows = {}
    for window in windows:
        key = (window.start, window.end)
        if key not in unique_windows:
            unique_windows[key] = window
    return list(unique_windows.values())[:config["outliers"]["max_windows_per_haplotype"]]


def plot_outlier_switch_genome(
    dataset: DatasetDiagnostics, analysis: OutlierPathAnalysis
) -> plt.Figure:
    """Locate gaps, parent differences, switch gains/losses, and cumulative excess."""
    case = dataset.primary
    edges = dataset.site_edges / style["genome_unit_bp"]
    fig, axes = plt.subplots(
        5, 1, sharex=True, figsize=config["outliers"]["plot"]["genome_figsize"],
        gridspec_kw={"height_ratios": [0.3, 0.3, 1, 1, 1]}, layout="constrained",
    )
    draw_strip(axes[0], edges, (~case.covered).astype(int), source=True)
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    draw_strip(axes[1], edges, case.categories)
    axes[1].set_ylabel("Agreement", rotation=0, ha="right")
    full_changes = case.full_parents[1:] != case.full_parents[:-1]
    stitched_changes = case.stitched_parents[1:] != case.stitched_parents[:-1]
    delta = stitched_changes.astype(int) - full_changes.astype(int)
    x = dataset.positions[1:] / style["genome_unit_bp"]
    for selected, level, color, label in (
        (delta > 0, 1, style["stitched_color"], "Stitched switch only"),
        (delta < 0, -1, style["full_color"], "Full switch only"),
    ):
        selected_count = np.count_nonzero(selected)
        axes[2].scatter(
            x[selected], np.full(selected_count, level),
            marker="|", s=100, color=color, label=label,
        )
    axes[2].set(yticks=[-1, 1], yticklabels=["Full only", "Stitched only"], ylim=(-1.5, 1.5))
    axes[2].set_ylabel("Switch\npositions")
    axes[3].step(x, np.cumsum(delta), where="post", color=style["stitched_color"])
    axes[3].set_ylabel("Cumulative\nextra switches")
    gap_midpoints = (analysis.gaps.start_bp + analysis.gaps.end_bp) / 2
    axes[4].scatter(
        gap_midpoints / style["genome_unit_bp"], analysis.gaps.extra_gap_and_seam_switches,
        color=population_style["population_colors"][case.summary["population"]],
        s=population_style["point_size"],
    )
    axes[4].set_ylabel("Gap + seams\nnet extra switches")
    for ax in axes[2:]:
        ax.axhline(0, color="0.5", linewidth=0.5)
        ax.grid(alpha=0.2)
    axes[-1].set(xlim=(edges[0], edges[-1]), xlabel=f"Genomic position ({style['genome_unit_label']})")
    fig.suptitle(
        f"{dataset.label}: {case.label} ({case.summary['population']}) — where switches change"
    )
    fig.legend(handles=coverage_handles, loc="outside lower center", ncol=3)
    return fig


def plot_outlier_gap_properties(
    dataset: DatasetDiagnostics, analysis: OutlierPathAnalysis
) -> plt.Figure:
    """Show gap length against its local switch excess and parent agreement."""
    case = dataset.primary
    gaps = analysis.gaps
    fig, axes = plt.subplots(
        1, 2, figsize=config["outliers"]["plot"]["gap_figsize"], layout="constrained",
    )
    color = population_style["population_colors"][case.summary["population"]]
    for ax, metric, ylabel in (
        (axes[0], "extra_gap_and_seam_switches", "Net extra switches in gap + adjacent seams"),
        (axes[1], "agreement_fraction", "Parent agreement within gap"),
    ):
        ax.scatter(gaps.num_sites, gaps[metric], color=color, alpha=population_style["point_alpha"])
        ax.set(xscale="log", xlabel="Sites in gap (log scale)", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0].axhline(0, color="0.5", linewidth=0.5)
    axes[1].set_ylim(-0.05, 1.05)
    ranked = gaps.sort_values("extra_switches_per_site", ascending=False)
    labelled = ranked.head(config["outliers"]["gaps_to_display"])
    for (num_sites, extra), group in labelled.groupby(
        ["num_sites", "extra_gap_and_seam_switches"]
    ):
        label = ", ".join(f"g{gap}" for gap in group.gap)
        axes[0].annotate(
            label, (num_sites, extra), xytext=(4, 4),
            textcoords="offset points", fontsize="small",
        )
    fig.suptitle(f"{dataset.label}: {case.label} ({case.summary['population']}) — individual gaps")
    return fig


# %%
outlier_analyses = {}
outlier_windows = {}
outlier_change_rows = []
outlier_relocation_rows = []
for dataset in outlier_case_datasets:
    case = dataset.primary
    key = (dataset.name, case.label)
    analysis = analyse_outlier_paths(dataset)
    outlier_analyses[key] = analysis
    outlier_windows[key] = select_outlier_windows(dataset, analysis)
    prefix = f"{dataset.name}_{case.summary['sample_id'].lower()}_{case.summary['ploidy_index']}"
    for name, table in (
        ("gaps", analysis.gaps),
        ("parent_differences", analysis.divergences),
        ("relocated_switches", analysis.relocations),
    ):
        table.to_csv(outlier_output_dir / f"{prefix}_{name}.csv", index=False)
    for row in analysis.switch_counts.itertuples():
        outlier_change_rows.append(
            {
                "dataset": dataset.name, "haplotype": case.label,
                "population": case.summary["population"], "kind": row.kind,
                "full_switches": row.full, "stitched_switches": row.stitched,
                "extra_switches": row.stitched - row.full,
            }
        )
    for row in analysis.relocations.itertuples():
        outlier_relocation_rows.append(
            {
                "dataset": dataset.name, "haplotype": case.label,
                "full_location": row.full_location,
                "stitched_location": row.stitched_location,
                "absolute_shift_sites": abs(row.shift_sites),
            }
        )
outlier_parent_changes = pd.DataFrame(outlier_change_rows)
outlier_parent_changes.to_csv(outlier_output_dir / "outlier_parent_transition_summary.csv", index=False)
ipython_display.display(outlier_parent_changes)
outlier_relocations = pd.DataFrame(outlier_relocation_rows)
relocation_summary = outlier_relocations.groupby(
    ["dataset", "haplotype", "full_location", "stitched_location"]
).absolute_shift_sites.agg(["count", "median", "max"])
print("Proven parent-preserving relocations (haplotypes without any are absent):")
ipython_display.display(relocation_summary)
relocation_summary.to_csv(outlier_output_dir / "outlier_relocation_summary.csv")

# %% [markdown]
# The bars below distinguish relocation of a preserved parent sequence from
# switches occurring in regions with different parent sequences. Relocation
# preserves the number of transitions within its disagreement run. A positive
# net excess in a changed-sequence region cannot be explained by moving the
# same complete sequence of parent transitions alone.
# A gap with complete parent agreement can still gain seam switches when its
# neighbouring focal parents differ from the full path.

# %%
fig, axes = plt.subplots(
    1, len(outlier_datasets), figsize=config["outliers"]["plot"]["transition_figsize"],
    layout="constrained", squeeze=False,
)
transition_style = config["outliers"]["plot"]
for ax, dataset in zip(axes[0], outlier_datasets, strict=True):
    y = np.arange(len(dataset.cases))
    for method_index, method in enumerate(("full", "stitched")):
        centers = y + (2 * method_index - 1) * population_style["bar_offset"]
        left = np.zeros(len(dataset.cases))
        for kind, color in zip(
            transition_style["transition_kinds"], transition_style["transition_colors"], strict=True
        ):
            widths = []
            for case in dataset.cases:
                table = outlier_analyses[(dataset.name, case.label)].switch_counts
                widths.append(int(table.loc[table.kind.eq(kind), method].iloc[0]))
            widths = np.array(widths)
            ax.barh(centers, widths, left=left, color=color, height=population_style["bar_height"])
            left += widths
    method_ticks = np.column_stack(
        (y - population_style["bar_offset"], y + population_style["bar_offset"])
    ).ravel()
    method_labels = [
        f"{case.label} · {method}"
        for case in dataset.cases for method in ("Full", "Stitched")
    ]
    ax.set(
        yticks=method_ticks, yticklabels=method_labels,
        xlabel="Parent switches", title=dataset.label,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
transition_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(
        transition_style["transition_colors"], transition_style["transition_kinds"], strict=True
    )
]
fig.legend(handles=transition_handles, loc="outside lower center", ncol=2)
fig.suptitle("Outliers: switch position changes and parent sequence changes")
fig.savefig(outlier_output_dir / "outlier_parent_transition_partition.png")
plt.show()

# %% [markdown]
# Inspect every outlier's gap table and genomic switch distribution below.
# Switch-position rugs compare positions only; the parent-sequence tables and
# categorical zooms establish whether the parents also change. The cumulative
# curve accounts for switch gains and losses across the entire inference span.
# Gap plots label gaps as g0, g1, …, matching their table indices.

# %%
for dataset in outlier_case_datasets:
    case = dataset.primary
    key = (dataset.name, case.label)
    analysis = outlier_analyses[key]
    prefix = f"{dataset.name}_{case.summary['sample_id'].lower()}_{case.summary['ploidy_index']}"
    print(f"{dataset.label}: {case.label} ({case.summary['population']})")
    gap_columns = [
        "gap", "start_site", "end_site", "num_sites", "agreement_fraction",
        "full_gap_and_seam_switches", "stitched_gap_and_seam_switches",
        "extra_gap_and_seam_switches", "extra_interior_switches",
        "left_flank_parent_agrees", "right_flank_parent_agrees",
    ]
    ranked_gaps = analysis.gaps.sort_values(
        ["extra_switches_per_site", "differ_sites"], ascending=[False, False]
    )
    print("Gaps ranked by net extra switches per site (including their seams):")
    displayed_gaps = ranked_gaps[gap_columns].head(config["outliers"]["gaps_to_display"])
    ipython_display.display(displayed_gaps.round(3))
    print("Disagreement regions ranked by net extra switches, then number of sites:")
    ranked_regions = analysis.divergences.sort_values(
        ["extra_switches", "num_sites"], ascending=[False, False]
    )
    region_columns = [
        "difference", "start_site", "end_site", "num_sites", "gap_sites",
        "kind", "full_switches", "stitched_switches", "extra_switches",
        "full_parent_sequence", "stitched_parent_sequence",
    ]
    displayed_regions = ranked_regions[region_columns].head(
        config["outliers"]["divergences_to_display"]
    )
    ipython_display.display(displayed_regions)
    fig = plot_outlier_switch_genome(dataset, analysis)
    fig.savefig(outlier_output_dir / f"{prefix}_switch_distribution.png")
    plt.show()
    fig = plot_outlier_gap_properties(dataset, analysis)
    fig.savefig(outlier_output_dir / f"{prefix}_gap_properties.png")
    plt.show()

# %% [markdown]
# Choose detailed windows reproducibly: a short gap with high net extra switches
# per site, the gap with lowest parent agreement, the longest uninterrupted
# focal disagreement, a changed parent sequence around a gap, and a proven
# parent-preserving relocation when available. Duplicate displayed windows are
# shown once. Tables report the entire target and displayed coordinates.
# Short-gap zooms use inference-site indices to keep even one-site gaps visible;
# their genomic coordinates are reported in the tables. Other zooms use kb.
#
# Most long targets are clipped to the configured site limit around a relevant
# anchor. Relocation windows retain both switch positions, even when farther
# apart, so the before/after parents and the intervening disagreement are visible.
# Shared categorical rows identify the actual parents; black dashed is full HMM
# and solid purple is stitched. Dotted lines mark target boundaries.

# %%
outlier_window_rows = []
for dataset in outlier_case_datasets:
    case = dataset.primary
    key = (dataset.name, case.label)
    prefix = f"{dataset.name}_{case.summary['sample_id'].lower()}_{case.summary['ploidy_index']}"
    rows = []
    for index, window in enumerate(outlier_windows[key]):
        target = case.disagreement[window.target_start : window.target_end]
        row = {
            "dataset": dataset.name, "haplotype": case.label,
            **dataclasses.asdict(window),
            "target_agreement": float(np.mean(~target)),
            "display_start_bp": dataset.positions[window.start],
            "display_end_bp": dataset.site_edges[window.end],
        }
        rows.append(row)
    outlier_window_rows.extend(rows)
    print(f"Detailed windows: {dataset.label}, {case.label}")
    ipython_display.display(pd.DataFrame(rows).round(3))
    for index, window in enumerate(outlier_windows[key]):
        site_axis = (
            config["outliers"]["short_gap_site_axis"]
            and window.kind == "Short gap with extra switches"
        )
        fig = plot_detail(dataset, window, site_axis=site_axis)
        fig.savefig(outlier_output_dir / f"{prefix}_detail_{index + 1}.png")
        plt.show()
pd.DataFrame(outlier_window_rows).to_csv(outlier_output_dir / "outlier_detail_windows.csv", index=False)

# %% [markdown]
# Finally, compare disagreement with distance from the nearest seam separately
# for every outlier and source region. Distance is in inference-site steps;
# the sites immediately beside a seam have distance zero. These are descriptive
# profiles of selected outliers, with denominators exported alongside the plot.

# %%
fig, axes = plt.subplots(
    len(config["outliers"]["criteria"]), len(outlier_datasets),
    figsize=config["outliers"]["plot"]["distance_figsize"],
    layout="constrained", squeeze=False, sharey=True,
)
outlier_distance_rows = []
bins = np.array(config["distance_bin_edges"])
distance_labels = [f"{lo}–{hi - 1}" for lo, hi in zip(bins[:-1], bins[1:], strict=True)]
for column, dataset in enumerate(outlier_datasets):
    for row, criterion in enumerate(config["outliers"]["criteria"]):
        selected = outlier_selection.loc[
            outlier_selection.dataset.eq(dataset.name)
            & outlier_selection.criterion.eq(criterion["label"])
        ].iloc[0]
        case = next(
            case for case in dataset.cases
            if case.summary["sample_id"] == selected.sample_id
            and case.summary["ploidy_index"] == selected.ploidy_index
        )
        distance = seam_distances(case.covered)
        ax = axes[row, column]
        for focal, label, color in (
            (True, "Focal regions", style["category_colors"][1]),
            (False, "Gaps", style["category_colors"][3]),
        ):
            selected_sites = case.covered == focal
            totals = np.histogram(distance[selected_sites], bins=bins)[0]
            differences = np.histogram(distance[selected_sites & case.disagreement], bins=bins)[0]
            rate = np.divide(differences, totals, out=np.full(len(totals), np.nan), where=totals > 0)
            ax.plot(np.arange(len(rate)), rate, marker="o", color=color, label=label)
            for lo, hi, count, difference in zip(bins[:-1], bins[1:], totals, differences, strict=True):
                outlier_distance_rows.append(
                    {"dataset": dataset.name, "haplotype": case.label, "source": label,
                     "distance_start": lo, "distance_end": hi,
                     "num_sites": int(count), "disagree_sites": int(difference)}
                )
        ax.set(
            xticks=np.arange(len(distance_labels)), xticklabels=distance_labels,
            ylim=(0, 1), title=f"{case.label} ({case.summary['population']}) — {criterion['label']}",
        )
        ax.tick_params(axis="x", labelrotation=60)
        ax.grid(alpha=0.2)
        if column == 0:
            ax.set_ylabel("Parent disagreement fraction")
        if row == len(config["outliers"]["criteria"]) - 1:
            ax.set_xlabel("Distance from nearest seam (site steps)")
        if row == 0:
            ax.legend()
fig.suptitle("Outliers: disagreement inside focal regions and gaps")
fig.savefig(outlier_output_dir / "outlier_disagreement_by_seam_distance.png")
plt.show()
pd.DataFrame(outlier_distance_rows).to_csv(
    outlier_output_dir / "outlier_seam_distance_counts.csv", index=False
)


# %% [markdown]
# # Diagnosing stitched focal and gap matches
#
# Compare full HMM and stitched copying paths for a 10 Mb simulation and a
# 10 Mb region of chromosome 20 from 1KGP. Focal coverage is the union of
# zero-mismatch intervals anchored at each eligible ancestor's leftmost focal
# site, using 2 ≤ focal AC ≤ 10. Gaps are maximal uncovered site runs.
# Within coverage, a covering ancestor is kept until its interval ends; the
# next available interval extending furthest right is chosen, with original
# node ID breaking ties. Independent full-reference HMM matches fill the gaps.
# All parents are original-reference node IDs.
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


datasets = [load_dataset(settings) for settings in config["datasets"]]
summary = pd.DataFrame([case.summary for dataset in datasets for case in dataset.cases])
summary["focal_agreement_fraction"] = (
    1 - summary.focal_disagreement_sites / summary.focal_sites
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
            "focal_sites",
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
# provenance changes between focal coverage and a gap fill; it contributes
# a switch only if the parent changes there. The same edge categories are used for
# the full HMM path so their totals can be compared directly.


# %%


case_summary = plot_case_summary(
    datasets, f"Saved haplotypes at focal AC ≤ {experiment_config['ac_cutoff']}"
)
plt.show()
switch_table = case_summary.switches
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
        title=dataset.label,
    )
axes[0, 0].set_ylabel("Net extra switches / stitched switches")
plt.show()

# %% [markdown]
# ## Where covered focal pieces and gap fills agree with the full path
#
# Each haplotype has two tracks: provenance (covered focal or gap fill), then
# the four-way agreement classification. Genomic widths are a display convention:
# a site's classification extends to the next inference-site position, with one
# configured plotting width at the final site. This is **not** evidence of exact
# focal matching between sites or beyond ancestor support. All counts and rates
# remain site-based. The chromosome's unobserved flanks are excluded from plots.


# %%


fig = plot_case_tracks(
    datasets, "Each haplotype: provenance above, parent agreement below"
)
plt.show()

# %% [markdown]
# ## Whole-span paths for the selected simulated and real haplotypes
#
# The full HMM and stitched paths use the same original-node ID scale. Node IDs
# identify parents; their vertical spacing has no biological distance meaning.
# The source and agreement tracks show which sites have focal coverage and which were
# rematched. A gap's parent can agree with the full HMM for only part of that gap.


# %%


for dataset in datasets:
    plot_overview(dataset)
    plt.show()

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
# - **Focal disagreement:** the covered focal interval with the most disagreeing
#   sites. The zoom is anchored on its first disagreement.
#
# Short target intervals include their full extent and context on both sides.
# Long intervals are clipped to the configured maximum site count around the
# selected anchor, so a zoom may show only part of the target. Dashed lines mark
# target boundaries when they are inside the displayed window. The table reports
# both the complete target and the displayed half-open site window.


# %%


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
# ## Detailed parent paths: covered focal pieces, gap fills, agreement, and difference
#
# These zooms put both paths on **categorical rows labeled by original ancestor
# identity and node ID**. Rows are shared within each panel; matching parent IDs
# occupy exactly the same row. Solid purple is stitched; dashed black is full HMM.
# Markers locate inference sites. Background shading shows provenance, and the
# lower agreement strip distinguishes all four source/agreement combinations.
# The bottom rug marks full-path switches above and stitched-path switches below.
#
# Focal intervals are exact in generated ancestor calls. Parent agreement tests
# copying identity; the saved mutation records separately test copied alleles
# in the inferred reference.


# %%


for dataset in datasets:
    for window in detail_windows[dataset.name]:
        plot_detail(dataset, window)
        plt.show()

# %% [markdown]
# ## Does disagreement concentrate near focal/gap seams?
#
# Pool the saved haplotypes **within each dataset**, separating covered focal
# sites from gap-fill sites. Distance is the number of inference-site steps to
# the nearest provenance seam, with the immediately adjacent site on either side
# assigned distance zero. Fractions use the actual site counts in each distance
# bin; the accompanying table gives the denominators.
#
# These are selected diagnostic haplotypes, not a representative population
# sample. Proximity is descriptive and does not isolate a causal boundary effect.
# In particular, distant disagreement within covered focal pieces means that
# conditioning gap endpoints alone cannot recover this particular full parent path.


# %%


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
            distance = seam_distances(case.covered)
            selected = case.covered == focal
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
