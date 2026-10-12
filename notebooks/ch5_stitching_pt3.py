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
# # Reduced-ancestor stitching: comparison with pt2
#
# Reproduces pt2's plots except the outlier analysis, using the completed
# `tsinfer-match-eval` stitched pipeline introduced in commit `4694565`.
# The hyperparameter plots show every configured run; subsequent plots use
# the AC 10 / zero-mismatch case to compare with pt2's greedy interval cover.

# %%
import dataclasses
import itertools
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
import scipy.stats
import tskit
import yaml

logger = logging.getLogger(__name__)

# %%
notebook_dir = (
    pathlib.Path(__file__).resolve().parent
    if "__file__" in globals()
    else pathlib.Path.cwd()
)
repository_dir = notebook_dir
if not (repository_dir / "experiments").is_dir():
    repository_dir = notebook_dir.parent
diagnostic_config_path = repository_dir / "experiments/ch5/stitching_pt3_config.yaml"
with diagnostic_config_path.open() as file:
    config = yaml.safe_load(file)
experiment_config_path = (
    diagnostic_config_path.parent / config["experiment_config"]
).resolve()
with experiment_config_path.open() as file:
    experiment_config = yaml.safe_load(file)
input_dir = pathlib.Path(experiment_config["data_dir"]).expanduser()
if not input_dir.is_absolute():
    input_dir = experiment_config_path.parent / input_dir
output_dir = (diagnostic_config_path.parent / config["cohort"]["output_dir"]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
comparison = config["comparison"]
assert comparison["ac_cutoff"] in experiment_config["ac_cutoff"]
assert comparison["max_mismatches"] in experiment_config["max_mismatches"]
assert comparison["method"] in experiment_config["focal_methods"]
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
    covered: np.ndarray
    disagreement: np.ndarray
    categories: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.summary['sample_id']}/{self.summary['ploidy_index']}"


@dataclasses.dataclass
class DatasetDiagnostics:
    """The original ancestor reference and selected run's diagnostic haplotypes."""

    name: str
    label: str
    reference: tskit.TreeSequence
    positions: np.ndarray
    site_edges: np.ndarray
    cases: list[HaplotypeDiagnostics]
    primary: HaplotypeDiagnostics


def read_matches(path: pathlib.Path) -> dict:
    """Join native records by haplotype identity and order their genomic paths.

    Baseline JSONL paths are emitted right-to-left, while stitched paths are
    left-to-right. Canonical ordering also makes mutation comparisons explicit.
    """
    records = {}
    with path.open() as file:
        for line in file:
            record = json.loads(line)
            assert record["source"] == "samples"
            key = (record["sample_id"], record["ploidy_index"])
            assert key not in records
            record["path"].sort(key=lambda segment: segment["left"])
            record["mutations"].sort(key=lambda mutation: mutation["position"])
            records[key] = record
    return records


def parents_at_sites(record: dict, reference: tskit.TreeSequence) -> np.ndarray:
    """Evaluate a complete native path at the original reference inference sites."""
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
    """Reconstruct disjoint half-open inference-site intervals."""
    mask = np.zeros(num_sites, dtype=bool)
    for start, end in intervals:
        assert 0 <= start < end <= num_sites
        assert not np.any(mask[start:end])
        mask[start:end] = True
    return mask


def population_labels(settings: dict, dataset_config: dict) -> dict:
    """Read population identities from simulation truth or the 1KGP pedigree."""
    if settings["population_source"] == "simulation":
        truth_path = pathlib.Path(dataset_config["ts_path"]).expanduser()
        if not truth_path.is_absolute():
            truth_path = experiment_config_path.parent / truth_path
        truth = tskit.load(truth_path)
        labels = {}
        for individual in truth.individuals():
            for ploidy_index, node_id in enumerate(individual.nodes):
                node = truth.node(int(node_id))
                population = truth.population(node.population).metadata["name"]
                labels[(f"tsk_{individual.id}", ploidy_index)] = population
        return labels
    metadata_path = diagnostic_config_path.parent / settings["population_metadata"]
    pedigree = pd.read_csv(metadata_path)
    assert not pedigree.sample_id.duplicated().any()
    labels = {}
    for row in pedigree.itertuples(index=False):
        for ploidy_index in range(2):
            labels[(row.sample_id, ploidy_index)] = row.superpopulation
    return labels


@dataclasses.dataclass
class SavedResults:
    """Scalar results for every run and site arrays for the selected comparisons."""

    comparisons: pd.DataFrame
    datasets: dict[str, DatasetDiagnostics]


def summarise_case(case: HaplotypeDiagnostics) -> None:
    """Calculate pt2's site and switch metrics on the saved focal-mask partition."""
    summary = case.summary
    num_sites = len(case.covered)
    summary["num_sites"] = num_sites
    summary["focal_sites"] = int(np.count_nonzero(case.covered))
    summary["gap_sites"] = num_sites - summary["focal_sites"]
    summary["num_gaps"] = len(case.stitched["gap_site_intervals"])
    for method in ("full", "stitched"):
        parents = getattr(case, f"{method}_parents")
        counts = switch_breakdown(case, parents)
        summary[f"{method}_switches"] = int(counts.sum())
        assert summary[f"{method}_switches"] == len(getattr(case, method)["path"]) - 1
        for category, count in zip(("focal", "gap", "seam"), counts, strict=True):
            summary[f"{method}_{category}_switches"] = int(count)
        summary[f"{method}_mismatches"] = len(getattr(case, method)["mutations"])
    summary["num_extra_switches"] = (
        summary["stitched_switches"] - summary["full_switches"]
    )
    summary["extra_switch_fraction"] = (
        summary["num_extra_switches"] / summary["stitched_switches"]
        if summary["stitched_switches"] > 0
        else np.nan
    )
    summary["switch_increase_percentage"] = (
        100 * summary["num_extra_switches"] / summary["full_switches"]
        if summary["full_switches"] > 0
        else np.nan
    )
    summary["fraction_parent_agreement"] = float(np.mean(~case.disagreement))
    summary["parents_identical"] = bool(not np.any(case.disagreement))
    summary["mutations_identical"] = (
        case.full["mutations"] == case.stitched["mutations"]
    )
    for region, mask in (("focal", case.covered), ("gap", ~case.covered)):
        disagreement_count = int(np.count_nonzero(case.disagreement & mask))
        agreement_count = int(np.count_nonzero(~case.disagreement & mask))
        summary[f"{region}_disagreement_sites"] = disagreement_count
        summary[f"{region}_agree_fraction"] = agreement_count / num_sites
        summary[f"{region}_disagree_fraction"] = disagreement_count / num_sites


def load_saved_results() -> SavedResults:
    """Read every configured AC/k/method product without rerunning inference.

    Only configured diagnostic haplotypes retain site arrays. All cohort
    measurements use current JSONL parents and the matching mask, never pt2's
    summaries. Full-reference parents are read once per dataset and reused.
    """
    rows = []
    datasets = {}
    dataset_configs = {item["name"]: item for item in experiment_config["datasets"]}
    assert set(dataset_configs) == {item["name"] for item in config["datasets"]}
    for settings in config["datasets"]:
        name = settings["name"]
        reference = tskit.load(
            input_dir / "ancestors" / f"{name}_inferred_ancestors.trees"
        )
        positions = reference.sites_position
        terminal_edge = positions[-1] + style["terminal_site_width_bp"]
        site_edges = np.concatenate((positions, [terminal_edge]))
        full_records = read_matches(
            input_dir / "matches" / f"{name}_inferred_samples_matches.jsonl"
        )
        full_parents = {
            key: parents_at_sites(record, reference)
            for key, record in full_records.items()
        }
        labels = population_labels(settings, dataset_configs[name])
        with np.load(
            input_dir / "focal_ancestors" / f"{name}_inferred_focal_ancestors.npz",
            allow_pickle=False,
        ) as archive:
            focal = {field: archive[field] for field in archive.files}
        roster = list(
            zip(
                focal["sample_id"].tolist(), focal["ploidy_index"].tolist(), strict=True
            )
        )
        assert set(roster) == set(full_records)
        assert len(roster) == config["cohort"]["num_haplotypes"]
        selected = {
            (item["sample_id"], item["ploidy_index"]) for item in settings["haplotypes"]
        }
        for ac_cutoff in experiment_config["ac_cutoff"]:
            focal_counts = []
            for index in range(len(roster)):
                start, end = focal["offsets"][index : index + 2]
                columns = focal["ancestor_index"][start:end]
                count = np.count_nonzero(focal["derived_ac"][columns] <= ac_cutoff)
                focal_counts.append(int(count))
            for max_mismatches in experiment_config["max_mismatches"]:
                prefix = f"{name}_inferred_ac{ac_cutoff}_k{max_mismatches}"
                with np.load(
                    input_dir / "focal_masks" / f"{prefix}_focal_mask.npz",
                    allow_pickle=False,
                ) as archive:
                    mask = {field: archive[field] for field in archive.files}
                assert int(mask["num_sites"]) == len(positions)
                assert int(mask["ac_cutoff"]) == ac_cutoff
                assert int(mask["max_mismatches"]) == max_mismatches
                assert str(mask["focal_choice"]) == experiment_config["focal_choice"]
                assert np.array_equal(mask["sample_id"], focal["sample_id"])
                assert np.array_equal(mask["ploidy_index"], focal["ploidy_index"])
                for method in experiment_config["focal_methods"]:
                    match_path = (
                        input_dir
                        / "matches"
                        / f"{prefix}_{method}_stitched_samples_matches.jsonl"
                    )
                    stitched_records = read_matches(match_path)
                    assert set(stitched_records) == set(full_records)
                    current = {
                        "ac_cutoff": ac_cutoff,
                        "max_mismatches": max_mismatches,
                        "method": method,
                    }
                    cases = []
                    for index, key in enumerate(roster):
                        raw_stitched = stitched_records[key]
                        for field, value in current.items():
                            assert raw_stitched[field] == value
                        start, end = mask["offsets"][index : index + 2]
                        left = mask["left_site_index"][start:end]
                        right = mask["right_site_index"][start:end]
                        is_gap = mask["is_gap"][start:end]
                        assert left[0] == 0 and right[-1] == len(positions)
                        assert np.all(left < right) and np.array_equal(
                            right[:-1], left[1:]
                        )
                        assert np.all(is_gap[1:] != is_gap[:-1])
                        intervals = np.column_stack((left, right))
                        focal_intervals = intervals[~is_gap].tolist()
                        gap_intervals = intervals[is_gap].tolist()
                        covered = interval_mask(focal_intervals, len(positions))
                        gaps = interval_mask(gap_intervals, len(positions))
                        assert np.array_equal(gaps, ~covered)
                        identity = {
                            "dataset": name,
                            "source": "samples",
                            "sample_id": key[0],
                            "ploidy_index": key[1],
                            **current,
                        }
                        full = {**full_records[key], **identity}
                        stitched = {
                            **raw_stitched,
                            **identity,
                            "focal_site_intervals": focal_intervals,
                            "gap_site_intervals": gap_intervals,
                        }
                        stitched_parents = parents_at_sites(stitched, reference)
                        disagreement = full_parents[key] != stitched_parents
                        categories = (~covered).astype(
                            np.int8
                        ) * 2 + disagreement.astype(np.int8)
                        summary = {
                            **identity,
                            "population": labels[key],
                            "num_focal_ancestors": focal_counts[index],
                        }
                        case = HaplotypeDiagnostics(
                            summary,
                            full,
                            stitched,
                            full_parents[key],
                            stitched_parents,
                            covered,
                            disagreement,
                            categories,
                        )
                        summarise_case(case)
                        rows.append(summary)
                        if current == comparison and key in selected:
                            cases.append(case)
                    if current == comparison:
                        assert len(cases) == len(selected)
                        primary_key = (
                            settings["deep_dive"]["sample_id"],
                            settings["deep_dive"]["ploidy_index"],
                        )
                        primary = next(
                            case
                            for case in cases
                            if (case.summary["sample_id"], case.summary["ploidy_index"])
                            == primary_key
                        )
                        datasets[name] = DatasetDiagnostics(
                            name,
                            settings["label"],
                            reference,
                            positions,
                            site_edges,
                            cases,
                            primary,
                        )
    comparisons = pd.DataFrame(rows)
    run_keys = ["dataset", "ac_cutoff", "max_mismatches", "method"]
    identity_keys = [*run_keys, "sample_id", "ploidy_index"]
    assert not comparisons.duplicated(identity_keys).any()
    assert (
        comparisons.groupby(run_keys)
        .size()
        .eq(config["cohort"]["num_haplotypes"])
        .all()
    )
    assert (
        comparisons.groupby(run_keys)
        .sample_id.nunique()
        .eq(config["cohort"]["num_individuals"])
        .all()
    )
    assert comparisons.population.notna().all()
    for settings in config["datasets"]:
        populations = comparisons.loc[
            comparisons.dataset.eq(settings["name"]), "population"
        ]
        assert set(populations) == set(settings["populations"])
    return SavedResults(comparisons, datasets)


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
    for lo, hi in itertools.pairwise(run_bounds):
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
        x = (
            switches
            if site_axis
            else dataset.positions[switches] / style["detail_unit_bp"]
        )
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
            x = (
                boundary
                if site_axis
                else dataset.site_edges[boundary] / style["detail_unit_bp"]
            )
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


def plot_case_summary(
    datasets: list[DatasetDiagnostics], title: str
) -> CaseSummaryFigure:
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
            axes[0, column].barh(
                y, fractions, left=accumulated, color=color, label=label
            )
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
# # Extra switches vs hyperparameters
#
# Read the completed run in `tsinfer-match-eval/config.yaml`: AC cutoffs 3, 5,
# and 10, with zero or one mismatches allowed independently on each side of
# the focal seed. Each line is one dataset and focal matching method, averaged
# over its 200 haplotypes, with ± one SEM. Percentage increases exclude full
# paths with zero switches. The summary table reports defined counts.
#
# The revised pipeline matches once against a reduced ancestor tree sequence
# retaining eligible focal candidates, their internal ancestors, and roots.
# Its original-node parent path supplies covered sites; independent full-reference
# HMMs supply gaps. The saved masks define the partition for both methods.

# %%
saved_results = load_saved_results()
all_comparisons = saved_results.comparisons
all_comparisons.to_csv(output_dir / "all_run_comparisons.csv", index=False)
switch_counts = all_comparisons
hyperparameter_metrics = [
    ("num_extra_switches", "Mean extra switches"),
    ("switch_increase_percentage", "Mean percentage increase"),
]
run_keys = ["dataset", "method", "max_mismatches", "ac_cutoff"]
hyperparameter_summary = switch_counts.groupby(run_keys)[
    [metric for metric, _ in hyperparameter_metrics]
].agg(["mean", "sem", "count"])
hyperparameter_summary.to_csv(output_dir / "hyperparameter_summary.csv")
ipython_display.display(hyperparameter_summary.round(4))
requested_mismatch_settings = experiment_config["max_mismatches"]
fig, axes = plt.subplots(
    nrows=len(hyperparameter_metrics),
    ncols=len(requested_mismatch_settings),
    figsize=style["population_results"]["hyperparameter_figsize"],
    sharex=True,
    sharey="row",
    squeeze=False,
)
for column, mismatch_count in enumerate(requested_mismatch_settings):
    for row, (metric, ylabel) in enumerate(hyperparameter_metrics):
        ax = axes[row, column]
        for settings in config["datasets"]:
            for method in experiment_config["focal_methods"]:
                values = hyperparameter_summary.loc[
                    (settings["name"], method, mismatch_count), metric
                ]
                ax.errorbar(
                    values.index,
                    values["mean"],
                    yerr=values["sem"],
                    marker="o",
                    capsize=style["error_bar_capsize"],
                    label=f"{settings['label']} · {method}",
                )
        ax.set_title(f"{mismatch_count} mismatches allowed per side")
        ax.set_ylabel(ylabel)
        ax.set_xticks(experiment_config["ac_cutoff"])
        ax.grid(alpha=0.2)
        ax.legend(fontsize="small")
for ax in axes[-1]:
    ax.set_xlabel("AC cutoff")
fig.suptitle("Reduced-ancestor stitching: extra switches (mean ± SEM)")
fig.tight_layout()
fig.savefig(output_dir / "hyperparameter_extra_switches.png")
plt.show()

# %% [markdown]
# For side-by-side comparison with pt2, the remaining plots use **AC ≤ 10,
# zero mismatches, focal method HMM**, configured in `stitching_pt3_config.yaml`.
# This retains the same diagnostic haplotypes, populations, and plot styles.
# All summaries and figures are written to `data/ch5_stitching_pt3`.

# %%
selected_mask = (
    all_comparisons.ac_cutoff.eq(comparison["ac_cutoff"])
    & all_comparisons.max_mismatches.eq(comparison["max_mismatches"])
    & all_comparisons.method.eq(comparison["method"])
)
selected_comparisons = all_comparisons.loc[selected_mask].copy()
comparison_label = (
    f"AC ≤ {comparison['ac_cutoff']}, "
    f"k = {comparison['max_mismatches']}, {comparison['method']}"
)
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(hyperparameter_metrics),
    figsize=style["population_results"]["current_config_figsize"],
)
for ax, (metric, ylabel) in zip(axes, hyperparameter_metrics, strict=True):
    for settings in config["datasets"]:
        observations = selected_comparisons.loc[
            selected_comparisons.dataset.eq(settings["name"]), metric
        ]
        ax.errorbar(
            [comparison["ac_cutoff"]],
            [observations.mean()],
            yerr=[observations.sem()],
            marker="o",
            capsize=style["error_bar_capsize"],
            label=settings["label"],
        )
    ax.set_xlabel("AC cutoff")
    ax.set_ylabel(ylabel)
    ax.set_xticks([comparison["ac_cutoff"]])
    ax.legend(fontsize="small")
    if metric == "switch_increase_percentage":
        undefined_count = selected_comparisons[metric].isna().sum()
        ax.text(
            0.98,
            0.96,
            f"{undefined_count} undefined increases excluded",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
fig.suptitle(f"Selected comparison: {comparison_label} (mean ± SEM)")
fig.tight_layout()
fig.savefig(output_dir / "selected_configuration_extra_switches.png")
plt.show()

# %% [markdown]
# # Population-level results
#
# Compare all saved haplotypes, grouped by population, in the simulation and
# 1KGP datasets. Each observation is one haplotype. Settings and plot styles
# live in `experiments/ch5/stitching_pt3_config.yaml`.

# %%
population_style = style["population_results"]
population_output_dir = output_dir
population_comparisons = selected_comparisons.copy()
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
    population_comparisons[f"{region}_agreement_percentage"] = 100 * (
        1 - disagreeing_sites / region_sites
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
            observations = dataset_table.loc[
                dataset_table.population.eq(population), metric
            ]
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
                kde = scipy.stats.gaussian_kde(
                    values, bw_method=population_style["kde_bandwidth"]
                )
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
    "num_extra_switches",
    "Extra switches",
    "Extra switches by population",
    integer_bins=True,
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
    mpatches.Patch(color=population_style["population_colors"][population], label=label)
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
        1,
        len(config["datasets"]),
        figsize=population_style["gap_histogram_figsize"],
        layout="constrained",
        squeeze=False,
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
            values,
            bins=bins,
            density=True,
            color=population_style["gap_histogram_color"],
            alpha=population_style["histogram_alpha"],
            edgecolor="white",
        )
        if len(values) > 1 and np.ptp(values) > 0:
            kde = scipy.stats.gaussian_kde(
                values, bw_method=population_style["kde_bandwidth"]
            )
            density = kde(grid) + kde(-grid)
            ax.plot(
                grid,
                density,
                color=style["full_color"],
                linewidth=population_style["line_width"],
            )
            density_max = max(density_max, density.max())
        mean = values.mean()
        ax.axvline(
            mean,
            color=style["full_color"],
            linestyle="--",
            linewidth=population_style["line_width"],
        )
        ax.text(
            0.98,
            0.95,
            f"n = {len(values)}; excluded = {excluded}\nMean = {mean:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
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
                population_values,
                point_y,
                color=population_style["population_colors"][population],
                s=population_style["point_size"],
                alpha=population_style["point_alpha"],
                linewidths=0,
            )
        density_upper = density_max * 1.25
        tick_locator = mticker.MaxNLocator(nbins=population_style["density_tick_count"])
        density_ticks = tick_locator.tick_values(0, density_upper)
        density_ticks = density_ticks[
            (density_ticks >= 0) & (density_ticks <= density_upper)
        ]
        ax.set_yticks(density_ticks)
        ax.set(
            title=settings["label"],
            xlabel=xlabel,
            ylabel="Density",
            xlim=(bins[0], bins[-1]),
            ylim=(-strip_depth, density_upper),
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
    2,
    len(config["datasets"]),
    figsize=population_style["gap_scatter_figsize"],
    layout="constrained",
    squeeze=False,
    sharey="col",
)
for column, settings in enumerate(config["datasets"]):
    dataset_table = population_comparisons.loc[
        population_comparisons.dataset.eq(settings["name"])
    ]
    defined_table = dataset_table.dropna(subset=["mean_sites_per_gap"])
    excluded = len(dataset_table) - len(defined_table)
    for row, (metric, xlabel) in enumerate(
        (
            ("num_focal_ancestors", "Number of focal ancestors"),
            ("num_gaps", "Number of gaps"),
        )
    ):
        ax = axes[row, column]
        for population in settings["populations"]:
            observations = defined_table.loc[defined_table.population.eq(population)]
            ax.scatter(
                observations[metric],
                observations.mean_sites_per_gap,
                color=population_style["population_colors"][population],
                s=population_style["point_size"],
                alpha=population_style["point_alpha"],
                linewidths=0,
            )
        ax.set(xlabel=xlabel, ylabel="Mean sites per gap")
        ax.text(
            0.98,
            0.95,
            f"n = {len(defined_table)}; excluded = {excluded}",
            transform=ax.transAxes,
            ha="right",
            va="top",
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
    len(config["datasets"]),
    len(gap_predictors),
    figsize=population_style["extra_switch_scatter_figsize"],
    layout="constrained",
    squeeze=False,
    sharey="row",
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
            observations = defined_table.loc[defined_table.population.eq(population)]
            ax.scatter(
                observations[metric],
                observations.num_extra_switches,
                color=population_style["population_colors"][population],
                s=population_style["point_size"],
                alpha=population_style["point_alpha"],
                linewidths=0,
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
            0.98,
            0.95,
            f"n = {len(defined_table)}; excluded = {excluded}",
            transform=ax.transAxes,
            ha="right",
            va="top",
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
    assert np.allclose(
        method_table.loc[defined, population_switch_categories].sum(axis=1), 1
    )
    population_switch_tables.append(method_table)
population_switch_proportions = pd.concat(population_switch_tables, ignore_index=True)
population_switch_means = population_switch_proportions.groupby(
    ["dataset", "population", "method"]
)[population_switch_categories].mean()
population_switch_excluded = (
    population_switch_proportions.assign(
        excluded=population_switch_proportions.focal.isna()
    )
    .groupby(["dataset", "population", "method"])
    .excluded.sum()
)
excluded_compositions = population_switch_excluded.rename(
    "Undefined compositions excluded"
)
ipython_display.display(excluded_compositions.to_frame())

fig, axes = plt.subplots(
    1,
    len(config["datasets"]),
    figsize=population_style["composition_figsize"],
    layout="constrained",
    squeeze=False,
    sharex=True,
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
            ax.barh(
                centers,
                widths,
                left=left,
                height=population_style["bar_height"],
                color=color,
            )
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
        yticks=method_ticks,
        yticklabels=method_labels,
        xlim=(0, 100),
        xlabel="Mean proportion of switches (%)",
        title=settings["label"],
        xticks=population_style["percentage_ticks"],
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
population_switch_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(
        style["switch_colors"], population_style["switch_labels"], strict=True
    )
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
    1,
    len(config["datasets"]),
    figsize=population_style["ternary_figsize"],
    layout="constrained",
    squeeze=False,
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
                coordinates[:, 0],
                coordinates[:, 1],
                color="lightgray",
                linewidth=0.7,
                zorder=0,
            )
        ax.text(
            fraction,
            -0.025,
            f"{1 - fraction:.0%}",
            ha="center",
            va="top",
            fontsize="small",
        )
        ax.text(
            0.5 * fraction - 0.02,
            triangle_height * fraction,
            f"{fraction:.0%}",
            ha="right",
            fontsize="small",
        )
        ax.text(
            1 - 0.5 * fraction + 0.02,
            triangle_height * fraction,
            f"{1 - fraction:.0%}",
            ha="left",
            fontsize="small",
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
                coordinates[:, 0],
                coordinates[:, 1],
                color=color,
                marker=population_style["ternary_method_markers"][method],
                s=population_style["ternary_point_size"],
                alpha=population_style["ternary_point_alpha"],
                linewidths=0,
            )
            mean = population_switch_means.loc[
                (settings["name"], population, method)
            ].to_numpy()
            coordinate = ternary_coordinates(mean[np.newaxis, :])[0]
            ax.scatter(
                coordinate[0],
                coordinate[1],
                color=color,
                marker=population_style["ternary_method_markers"][method],
                s=population_style["ternary_mean_size"],
                edgecolors="white",
                zorder=3,
            )
            ax.annotate(
                population,
                coordinate,
                xytext=label_offset,
                textcoords="offset points",
                color=color,
                fontsize="small",
                ha="center",
                va="center",
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7},
            )
    ax.text(0, -0.07, "Within focal\ncoverage", ha="center", va="top")
    ax.text(1, -0.07, "Within gaps", ha="center", va="top")
    ax.text(0.5, triangle_height + 0.03, "At focal/gap seams", ha="center", va="bottom")
    ax.set(
        title=settings["label"],
        aspect="equal",
        xlim=(-0.15, 1.15),
        ylim=(-0.16, triangle_height + 0.1),
    )
    ax.set_axis_off()
population_method_handles = [
    mlines.Line2D(
        [],
        [],
        color=style["full_color"],
        linestyle="none",
        marker=population_style["ternary_method_markers"][method],
        label=label,
    )
    for method, label in (
        ("full", "Full HMM"),
        ("stitched", "Stitched HMM"),
    )
]
population_color_handles = [
    mpatches.Patch(color=population_style["population_colors"][population], label=label)
    for population, label in (
        ("YRI", "YRI / AFR"),
        ("CEU", "CEU / EUR"),
        ("CHB", "CHB / EAS"),
    )
]
fig.legend(
    handles=population_color_handles + population_method_handles,
    loc="outside lower center",
    ncol=5,
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
population_agreement_means = population_comparisons.groupby(["dataset", "population"])[
    population_agreement_columns
].mean()
fig, axes = plt.subplots(
    1,
    len(config["datasets"]),
    figsize=population_style["composition_figsize"],
    layout="constrained",
    squeeze=False,
    sharex=True,
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
        ax.barh(
            y, widths, left=left, height=population_style["bar_height"] * 2, color=color
        )
        left += widths
    ax.set(
        yticks=y,
        yticklabels=populations,
        xlim=(0, 100),
        xlabel="Mean proportion of all inference sites (%)",
        title=settings["label"],
        xticks=population_style["percentage_ticks"],
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
    "focal_agreement_percentage",
    "Parent agreement within focal regions (%)",
    "Parent agreement in focal regions by population",
    bounded_percentage=True,
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
    "gap_agreement_percentage",
    "Parent agreement within gaps (%)",
    "Parent agreement in gaps by population",
    bounded_percentage=True,
)
fig.savefig(population_output_dir / "population_gap_agreement_histograms.png")
plt.show()


# %% [markdown]
# # Diagnosing stitched focal and gap matches
#
# Compare full HMM and stitched copying paths for a 10 Mb simulation and a
# 10 Mb region of chromosome 20 from 1KGP. Focal coverage is the saved union
# of seed-anchored generated-ancestor intervals at the selected AC and mismatch
# allowance. Gaps are maximal uncovered site runs. Covered sites retain the
# HMM match against a haplotype-specific reduced ancestor tree sequence; gaps
# receive independent full-reference HMM fills. All parents have been mapped
# back to original-reference node IDs by the pipeline.
#
# The diagnostic identities and styles match pt2. Settings live in
# `experiments/ch5/stitching_pt3_config.yaml`. Inputs are the current pipeline's
# native JSONL and focal masks; this notebook does not rerun matching.
# Edit this `.py` file and use Jupytext to synchronize the paired notebook.

# %%


datasets = [saved_results.datasets[settings["name"]] for settings in config["datasets"]]
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
    datasets, f"Saved haplotypes at focal AC ≤ {comparison['ac_cutoff']}"
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
# Focal coverage comes from generated ancestor calls. The reduced-reference
# HMM chooses copying parents within it; those parents can differ from the
# interval-generating ancestors. Parent agreement tests copying identity, while
# saved mutations report mismatches against the final copied alleles.


# %%


for dataset in datasets:
    for window in detail_windows[dataset.name]:
        plot_detail(dataset, window)
        plt.show()

# %% [markdown]
# # Matched comparison: greedy cover versus reduced-reference HMM
#
# Compare the same haplotypes, reference nodes, focal masks, AC cutoff, and
# mismatch budget. **Greedy (pt2)** is the saved original implementation at
# AC 10 / k=0. **Greedy (same gap fills)** recomputes that cover on the current
# interval archives and retains exactly the pipeline's gap paths. This control
# is available at every AC/k combination, including cases not run in pt2.
# Its focal parents agree exactly with saved pt2 at AC 10 / k=0. Both datasets'
# old and current ancestor tree sequences also agree, including their topology.
#
# **Focal-only reduced** is a diagnostic control: use the same simplified
# reference and HMM parameters, but match each covered component independently,
# then reuse the pipeline's saved gap fills. It changes the optimization domain,
# while retaining the candidate/reference construction and observation model.
# Controls are run for all 200 haplotypes per dataset at the selected AC 10 / k=0.
#
# The analysis runs independently of notebook rendering. Rebuild its tables with:
#
# ```sh
# PYTHONPATH=src uv run --no-project \
#   --python ~/work/tsinfer-match-eval/.venv/bin/python \
#   python experiments/ch5/stitching_method_comparison.py \
#   --config experiments/ch5/stitching_pt3_config.yaml
# ```
#
# It uses queued multiprocessing across all available CPUs. Existing sections
# above remain as edited; the deleted sections are not restored.

# %%
comparison_style = config["method_comparison"]
comparison_output_dir = (
    diagnostic_config_path.parent / comparison_style["output_dir"]
).resolve()
implementation_results = pd.read_csv(comparison_output_dir / "comparisons.csv")
focal_controls = pd.read_csv(comparison_output_dir / "focal_controls.csv")
focal_components = pd.read_csv(comparison_output_dir / "focal_components.csv")
switch_distances = pd.read_csv(comparison_output_dir / "switch_distances.csv")
simulation_topology = pd.read_csv(comparison_output_dir / "simulation_topology.csv")
implementation_labels = comparison_style["implementation_labels"]
implementation_colors = comparison_style["implementation_colors"]
implementation_keys = [
    "dataset",
    "sample_id",
    "ploidy_index",
    "ac_cutoff",
    "max_mismatches",
    "implementation",
]
assert not implementation_results.duplicated(implementation_keys).any()
assert (
    implementation_results.groupby(
        ["dataset", "ac_cutoff", "max_mismatches", "implementation"]
    )
    .size()
    .eq(config["cohort"]["num_haplotypes"])
    .all()
)
matched_results = implementation_results.loc[
    implementation_results.ac_cutoff.eq(comparison["ac_cutoff"])
    & implementation_results.max_mismatches.eq(comparison["max_mismatches"])
].copy()
comparison_metrics = [
    "num_switches",
    "num_extra_switches",
    "switch_increase_percentage",
    "focal_switches",
    "gap_switches",
    "seam_switches",
    "num_mismatches",
    "full_model_log_likelihood_delta",
]
implementation_summary = matched_results.groupby(["dataset", "implementation"])[
    comparison_metrics
].agg(["mean", "sem"])
ipython_display.display(implementation_summary.xs("mean", axis=1, level=1).round(3))
# Match the new diagnostic's reduced outputs to the existing notebook summaries.
pipeline_check = matched_results.loc[
    matched_results.implementation.eq("reduced_hmm")
].set_index(["dataset", "sample_id", "ploidy_index"])
saved_check = selected_comparisons.set_index(["dataset", "sample_id", "ploidy_index"])
assert pipeline_check.num_switches.eq(saved_check.stitched_switches).all()
assert pipeline_check.num_mismatches.eq(saved_check.stitched_mismatches).all()

# %% [markdown]
# ## Where do the additional switches occur?
#
# Top: mean switch counts on the same focal/gap/seam partition. Bottom: paired
# differences from each haplotype's full HMM path, with ± one SEM. Negative
# bars represent switches removed relative to full HMM. Seam switches are counted
# only when parent IDs actually change, not simply when provenance changes.

# %%
switch_types = ["focal_switches", "gap_switches", "seam_switches"]
fig, axes = plt.subplots(
    2,
    len(config["datasets"]),
    figsize=comparison_style["switch_figsize"],
    squeeze=False,
    layout="constrained",
    sharey="row",
)
saved_methods = comparison_style["saved_methods"]
x = np.arange(len(switch_types))
width = style["bar_group_width"] / len(saved_methods)
for column, settings in enumerate(config["datasets"]):
    table = matched_results.loc[matched_results.dataset.eq(settings["name"])]
    paired = table.pivot(
        index=["sample_id", "ploidy_index"],
        columns="implementation",
        values=switch_types,
    )
    for index, method in enumerate(saved_methods):
        values = table.loc[table.implementation.eq(method), switch_types]
        centers = x + (index - (len(saved_methods) - 1) / 2) * width
        axes[0, column].bar(
            centers,
            values.mean(),
            width=width,
            yerr=values.sem(),
            capsize=style["error_bar_capsize"],
            color=implementation_colors[method],
            label=comparison_style["short_labels"][method],
        )
    for index, method in enumerate(saved_methods[1:]):
        differences = pd.DataFrame(
            {
                metric: paired[metric][method] - paired[metric]["full"]
                for metric in switch_types
            }
        )
        difference_width = style["bar_group_width"] / (len(saved_methods) - 1)
        centers = x + (index - (len(saved_methods) - 2) / 2) * difference_width
        axes[1, column].bar(
            centers,
            differences.mean(),
            width=difference_width,
            yerr=differences.sem(),
            capsize=style["error_bar_capsize"],
            color=implementation_colors[method],
            label=comparison_style["short_labels"][method],
        )
    for ax in axes[:, column]:
        ax.set_xticks(x, ["Focal interiors", "Gap interiors", "Seams"])
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize="small")
    axes[0, column].set_title(settings["label"])
axes[0, 0].set_ylabel("Mean switches per haplotype")
axes[1, 0].set_ylabel("Mean switch difference from full HMM")
fig.suptitle(f"Saved implementations at matched parameters: {comparison_label}")
fig.savefig(comparison_output_dir / "saved_switch_location_comparison.png")
plt.show()

# %% [markdown]
# ## Matched AC and mismatch budgets
#
# Solid lines use k=0; dashed lines use k=1. Both implementations use identical
# gaps in this comparison. Allowing a mismatch in interval extension does not
# make that mismatch free in the Li–Stephens likelihood: the greedy cover
# minimizes interval segments without considering mismatch penalties. Therefore
# compare its mismatch count alongside its switch count, especially for k=1.

# %%
parameter_metrics = [
    ("switch_increase_percentage", "Mean switch increase over full HMM (%)"),
    ("focal_switches", "Mean switches in focal interiors"),
    ("num_mismatches", "Mean mismatches in the final path"),
]
fig, axes = plt.subplots(
    len(parameter_metrics),
    len(config["datasets"]),
    figsize=comparison_style["parameter_figsize"],
    layout="constrained",
    squeeze=False,
    sharex=True,
    sharey="row",
)
for column, settings in enumerate(config["datasets"]):
    table = implementation_results.loc[
        implementation_results.dataset.eq(settings["name"])
    ]
    for row, (metric, ylabel) in enumerate(parameter_metrics):
        ax = axes[row, column]
        for method in ("greedy", "reduced_hmm"):
            for budget in experiment_config["max_mismatches"]:
                observations = table.loc[
                    table.implementation.eq(method) & table.max_mismatches.eq(budget)
                ]
                values = observations.groupby("ac_cutoff")[metric].agg(["mean", "sem"])
                ax.errorbar(
                    values.index,
                    values["mean"],
                    yerr=values["sem"],
                    marker="o",
                    linestyle=comparison_style["mismatch_line_styles"][budget],
                    color=implementation_colors[method],
                    capsize=style["error_bar_capsize"],
                    label=f"{comparison_style['short_labels'][method]}, k={budget}",
                )
        ax.set_ylabel(ylabel)
        ax.set_xticks(experiment_config["ac_cutoff"])
        ax.grid(alpha=0.2)
        if row == 0:
            ax.set_title(settings["label"])
        if row == len(parameter_metrics) - 1:
            ax.set_xlabel("Focal AC cutoff")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize="small")
fig.suptitle("Matched focal masks and gap fills: switch–mismatch trade-off")
fig.savefig(comparison_output_dir / "matched_hyperparameter_comparison.png")
plt.show()

# %% [markdown]
# ## Change the optimization domain while keeping the reduced reference
#
# The pipeline's reduced HMM matches the entire observed inference axis,
# including uncovered gaps, before discarding its gap path. The independent
# control resets the HMM at each covered component and scores only that component.
# All three controlled implementations below use the same saved gap fills.
#
# This directly tests whether extra focal switches are needed to reproduce focal
# alleles, or arise from optimizing a different domain and its boundary states.
# The diagonal in the paired scatter plots represents the greedy switch count.

# %%
control_summary = focal_controls.groupby("dataset").agg(
    haplotypes=("sample_id", "size"),
    mean_discarded_gap_mismatches=("global_reduced_gap_mismatches", "mean"),
    mean_pipeline_focal_switches=("global_reduced_focal_switches", "mean"),
    mean_greedy_focal_switches=("greedy_focal_switches", "mean"),
    mean_independent_focal_switches=("independent_focal_switches", "mean"),
    total_greedy_focal_mismatches=("greedy_focal_mismatches", "sum"),
    total_independent_focal_mismatches=("independent_focal_mismatches", "sum"),
    genotype_differences_after_simplify=("genotype_differences_after_simplify", "sum"),
)
ipython_display.display(control_summary.round(3))
assert focal_controls.genotype_differences_after_simplify.eq(0).all()
assert focal_controls.greedy_focal_mismatches.eq(0).all()
assert focal_controls.independent_focal_mismatches.eq(0).all()
assert focal_controls.independent_focal_switches.le(
    focal_controls.greedy_focal_switches
).all()
fig, axes = plt.subplots(
    2,
    len(config["datasets"]),
    figsize=comparison_style["control_figsize"],
    squeeze=False,
    layout="constrained",
    sharey="row",
)
controlled_methods = comparison_style["controlled_methods"]
width = style["bar_group_width"] / len(controlled_methods)
for column, settings in enumerate(config["datasets"]):
    table = matched_results.loc[matched_results.dataset.eq(settings["name"])]
    for index, method in enumerate(controlled_methods):
        values = table.loc[table.implementation.eq(method), switch_types]
        centers = x + (index - (len(controlled_methods) - 1) / 2) * width
        axes[0, column].bar(
            centers,
            values.mean(),
            width=width,
            yerr=values.sem(),
            capsize=style["error_bar_capsize"],
            color=implementation_colors[method],
            label=comparison_style["short_labels"][method],
        )
    controls = focal_controls.loc[focal_controls.dataset.eq(settings["name"])]
    for method, metric in (
        ("reduced_hmm", "global_reduced_focal_switches"),
        ("independent_hmm", "independent_focal_switches"),
    ):
        axes[1, column].scatter(
            controls.greedy_focal_switches,
            controls[metric],
            s=population_style["point_size"],
            alpha=population_style["point_alpha"],
            color=implementation_colors[method],
            label=comparison_style["short_labels"][method],
        )
    upper = (
        controls[["greedy_focal_switches", "global_reduced_focal_switches"]]
        .to_numpy()
        .max()
    )
    axes[1, column].plot([0, upper], [0, upper], "--", color="0.5")
    axes[1, column].set_xlabel("Greedy focal switches per haplotype")
    axes[0, column].set_xticks(x, ["Focal interiors", "Gap interiors", "Seams"])
    axes[0, column].set_title(settings["label"])
    for ax in axes[:, column]:
        ax.grid(alpha=0.2)
        ax.legend(fontsize="small")
axes[0, 0].set_ylabel("Mean switches per haplotype")
axes[1, 0].set_ylabel("Reduced-HMM focal switches per haplotype")
fig.suptitle(f"Controlled comparison: identical gap fills, {comparison_label}")
fig.savefig(comparison_output_dir / "independent_focal_hmm_control.png")
plt.show()

# %% [markdown]
# ## Are extra focal switches concentrated near seams?
#
# Each point is the pooled number of switches per 1,000 eligible edges in a
# distance bin. Both sites adjacent to an eligible edge must lie in the same
# source region; seam edges themselves are excluded. Distance is the smaller
# distance of the edge's two sites to the nearest seam, in inference-site steps.
# The exposure denominator is shared across implementations. Haplotypes without
# seams have undefined seam distances and do not enter this plot.

# %%
fig, axes = plt.subplots(
    2,
    len(config["datasets"]),
    figsize=comparison_style["distance_figsize"],
    layout="constrained",
    squeeze=False,
    sharex=True,
    sharey="row",
)
distance_methods = [*saved_methods, "independent_hmm"]
for column, settings in enumerate(config["datasets"]):
    table = switch_distances.loc[switch_distances.dataset.eq(settings["name"])]
    for row, region in enumerate(("focal", "gap")):
        ax = axes[row, column]
        for method in distance_methods:
            observations = table.loc[
                table.region.eq(region) & table.implementation.eq(method)
            ]
            values = observations.groupby(["distance_left", "distance_right"])[
                ["num_edges", "num_switches"]
            ].sum()
            rate = (
                comparison_style["switch_rate_scale"]
                * values.num_switches
                / values.num_edges.replace(0, np.nan)
            )
            labels = [f"{left}–{right - 1}" for left, right in values.index]
            ax.plot(
                np.arange(len(rate)),
                rate,
                marker="o",
                color=implementation_colors[method],
                label=comparison_style["short_labels"][method],
            )
        ax.set_xticks(np.arange(len(labels)), labels, rotation=60)
        ax.set_ylabel(f"{region.capitalize()} switches / 1,000 eligible edges")
        ax.grid(alpha=0.2)
        if row == 0:
            ax.set_title(settings["label"])
            ax.legend(fontsize="small")
        if row == 1:
            ax.set_xlabel("Distance to nearest seam (inference-site steps)")
fig.suptitle("Switch location at matched parameters, normalized by available edges")
fig.savefig(comparison_output_dir / "implementation_switch_distance.png")
plt.show()

# %% [markdown]
# ## Component-level evidence
#
# Top: switch differences from greedy within each maximal focal component.
# Bottom: mismatches in the reduced whole-span pass's gap regions, before they
# are overwritten. Those discarded observations still influence the retained
# focal path through the forward and traceback calculations.

# %%
focal_components["pipeline_minus_greedy"] = (
    focal_components.reduced_hmm_switches - focal_components.greedy_switches
)
focal_components["independent_minus_greedy"] = (
    focal_components.independent_hmm_switches - focal_components.greedy_switches
)
component_summary = focal_components.groupby("dataset").agg(
    components=("component", "size"),
    pipeline_extra_focal_switches=("pipeline_minus_greedy", "sum"),
    independent_difference=("independent_minus_greedy", "sum"),
)
ipython_display.display(component_summary)
assert focal_components.independent_minus_greedy.le(0).all()
fig, axes = plt.subplots(
    2,
    len(config["datasets"]),
    figsize=comparison_style["component_figsize"],
    layout="constrained",
    squeeze=False,
    sharey="row",
)
for column, settings in enumerate(config["datasets"]):
    components = focal_components.loc[focal_components.dataset.eq(settings["name"])]
    metrics = ["pipeline_minus_greedy", "independent_minus_greedy"]
    lower = int(components[metrics].to_numpy().min())
    upper = int(components[metrics].to_numpy().max())
    bins = np.arange(lower - 0.5, upper + 1.5, 1)
    for method, metric in zip(("reduced_hmm", "independent_hmm"), metrics, strict=True):
        axes[0, column].hist(
            components[metric],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=population_style["line_width"],
            color=implementation_colors[method],
            label=comparison_style["short_labels"][method],
        )
    axes[0, column].set(
        title=settings["label"],
        xlabel="Focal switches minus greedy, per component",
        xticks=np.arange(lower, upper + 1),
        ylabel="Fraction of focal components",
    )
    axes[0, column].legend(fontsize="small")
    controls = focal_controls.loc[focal_controls.dataset.eq(settings["name"])]
    values = controls.global_reduced_gap_mismatches
    axes[1, column].hist(
        values,
        bins=population_style["histogram_bins"],
        color=implementation_colors["reduced_hmm"],
        alpha=population_style["histogram_alpha"],
        edgecolor="white",
    )
    axes[1, column].axvline(
        values.mean(), color=implementation_colors["reduced_hmm"], linestyle="--"
    )
    axes[1, column].set(
        xlabel="Discarded reduced-pass gap mismatches per haplotype",
        ylabel="Number of haplotypes",
    )
    axes[1, column].text(
        0.98,
        0.95,
        f"Mean = {values.mean():.1f}",
        transform=axes[1, column].transAxes,
        ha="right",
        va="top",
    )
    for ax in axes[:, column]:
        ax.grid(axis="y", alpha=0.2)
fig.suptitle(
    "Local switch surplus and the observations discarded after global matching"
)
fig.savefig(comparison_output_dir / "focal_component_and_discarded_gap_diagnostics.png")
plt.show()

# %% [markdown]
# ## Inspect the same focal component under all three controlled methods
#
# These windows use the existing diagnostic haplotypes. Select the longest
# component with the largest positive pipeline-minus-greedy switch difference,
# then anchor the zoom at its first pipeline switch absent from greedy.
# Categorical node rows are shared across the methods. Matching alleles need
# not imply the same ancestor identity or topology.


# %%
def plot_implementation_window(
    dataset: DatasetDiagnostics, archive: dict, window: DetailWindow
) -> plt.Figure:
    """Show controlled parent paths on a common original-node categorical axis."""
    left, right = window.start, window.end
    methods = comparison_style["controlled_methods"]
    archive_fields = comparison_style["archive_fields"]
    parent_paths = {
        method: archive[archive_fields[method]][left:right] for method in methods
    }
    nodes = np.unique(np.concatenate(list(parent_paths.values())))
    edges = dataset.site_edges[left : right + 1] / style["detail_unit_bp"]
    labels = []
    for node_id in nodes:
        metadata = dataset.reference.node(int(node_id)).metadata
        labels.append(f"{metadata.get('sample_id', 'root')} (node {node_id})")
    figsize = list(comparison_style["window_figsize"])
    figsize[1] = max(
        figsize[1], len(nodes) * style["parent_row_height_in"] * len(methods)
    )
    fig, axes = plt.subplots(
        len(methods) + 1,
        1,
        figsize=figsize,
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [0.4, *[2 for _ in methods]]},
    )
    draw_strip(
        axes[0], edges, (~archive["covered"][left:right]).astype(int), source=True
    )
    axes[0].set_ylabel("Source", rotation=0, ha="right")
    for ax, method in zip(axes[1:], methods, strict=True):
        parent_rows = np.searchsorted(nodes, parent_paths[method])
        ax.stairs(
            parent_rows, edges, baseline=None, color=implementation_colors[method]
        )
        ax.set(
            yticks=np.arange(len(nodes)),
            yticklabels=labels,
            ylim=(-0.7, len(nodes) - 0.3),
            ylabel=comparison_style["short_labels"][method],
        )
        ax.tick_params(axis="y", labelsize="small")
        ax.grid(axis="y", alpha=0.2)
        for boundary in (window.target_start, window.target_end):
            if left <= boundary <= right:
                ax.axvline(
                    dataset.site_edges[boundary] / style["detail_unit_bp"],
                    color="0.4",
                    linestyle=":",
                )
    axes[-1].set_xlabel(f"Genomic position ({style['detail_unit_label']})")
    axes[-1].set_xlim(edges[0], edges[-1])
    fig.suptitle(
        f"{dataset.label}, {dataset.primary.label}: same focal component\n"
        f"Target [{window.target_start}, {window.target_end}); displayed [{left}, {right})"
    )
    return fig


for dataset in datasets:
    case = dataset.primary
    sample_id = case.summary["sample_id"]
    ploidy_index = case.summary["ploidy_index"]
    archive_path = (
        comparison_output_dir
        / f"{dataset.name}_{sample_id.lower()}_{ploidy_index}_paths.npz"
    )
    with np.load(archive_path, allow_pickle=False) as saved:
        archive = {field: saved[field] for field in saved.files}
    components = focal_components.loc[
        focal_components.dataset.eq(dataset.name)
        & focal_components.sample_id.eq(sample_id)
        & focal_components.ploidy_index.eq(ploidy_index)
        & focal_components.pipeline_minus_greedy.gt(0)
    ]
    target = components.sort_values(
        ["pipeline_minus_greedy", "num_sites", "component"],
        ascending=[False, False, True],
    ).iloc[0]
    target_left = int(target.left_site)
    target_right = int(target.right_site)
    pipeline_switches = (
        archive["reduced_stitched"][1:] != archive["reduced_stitched"][:-1]
    )
    greedy_switches = archive["greedy"][1:] != archive["greedy"][:-1]
    candidate_sites = np.flatnonzero(pipeline_switches & ~greedy_switches) + 1
    candidate_sites = candidate_sites[
        (candidate_sites > target_left) & (candidate_sites < target_right)
    ]
    anchor = int(candidate_sites[0])
    left = max(0, anchor - config["detail"]["context_sites"])
    right = min(len(dataset.positions), left + config["detail"]["max_sites"])
    window = DetailWindow(
        "Matched implementation comparison", target_left, target_right, left, right
    )
    plot_implementation_window(dataset, archive, window)
    plt.show()

# %% [markdown]
# ## Do the extra switches improve the simulated topology?
#
# Attach all 200 sample paths to the same original ancestor reference, preserving
# their observed genotypes and sample identities. Simplify, remove the unobserved
# flanks, and compare with the known simulation across the inference interval.
# The span-weighted Kendall–Colijn distance with λ=0 measures rooted topology;
# smaller is closer to truth. Branch lengths are excluded. See the
# [tskit metric documentation](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.kc_distance).
#
# Tree count describes fragmentation, not biological correctness. This is one
# simulated replicate and one topology metric; it cannot establish general
# superiority of an implementation, or assess accuracy in 1KGP.

# %%
full_topology = simulation_topology.loc[
    simulation_topology.implementation.eq("full")
].set_index("dataset")
simulation_topology["kc_difference_from_full"] = (
    simulation_topology.kc_topology_distance
    - simulation_topology.dataset.map(full_topology.kc_topology_distance)
)
simulation_topology["extra_marginal_trees"] = (
    simulation_topology.num_trees
    - simulation_topology.dataset.map(full_topology.num_trees)
)
ipython_display.display(simulation_topology.round(4))
fig, axes = plt.subplots(
    1,
    2,
    figsize=comparison_style["topology_figsize"],
    layout="constrained",
)
methods = comparison_style["topology_methods"]
values = simulation_topology.set_index("implementation").loc[methods]
colors = [implementation_colors[method] for method in methods]
labels = [comparison_style["short_labels"][method] for method in methods]
for ax, metric, ylabel in zip(
    axes,
    ("kc_difference_from_full", "extra_marginal_trees"),
    ("KC topology distance minus full HMM", "Marginal trees minus full HMM"),
    strict=True,
):
    ax.bar(np.arange(len(methods)), values[metric], color=colors)
    ax.set_xticks(np.arange(len(methods)), labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
fig.suptitle("Single simulated replicate: topology accuracy and fragmentation")
fig.savefig(comparison_output_dir / "simulation_topology_comparison.png")
plt.show()

# %% [markdown]
# ## Diagnosis: the HMM optimizes a different problem
#
# **The large focal switch surplus comes from the whole-span reduced pass, not
# from a failure of the greedy interval cover.** At AC 10 / k=0, the same
# reduced-reference HMM independently matched within focal components uses
# 63.91 versus greedy's 64.10 focal switches per simulated haplotype, and
# 53.265 versus 54.54 in 1KGP. The pipeline's whole-span pass uses 101.00 and
# 95.345 respectively. Both local methods have zero focal mismatches. Across
# all 21,258 covered components, the local HMM never needs more switches than
# greedy. The genotype-preservation checks also find no differences introduced
# by simplification in any ancestor used by greedy.
#
# The pipeline runs `match_stitched_haplotype`'s reduced HMM across **all observed
# sites**, including gaps that it cannot explain well with this candidate panel.
# Before replacement, those gap regions average 578.74 mismatches in the
# simulation and 1,631.33 in 1KGP. It then keeps only the covered parts of that
# globally optimized path and overwrites the gaps with independent full-reference
# paths. The retained focal segments inherit boundary states and switches from
# the discarded global solution. Restricting a global optimum to a subset of
# sites need not yield the optimum for independently matched components, and
# replacing its gaps does not preserve global optimality.
#
# Li–Stephens matching optimizes the joint copying/emission likelihood for its
# supplied panel and domain, not the masked focal switch count or a measure of
# true genealogical topology. This is consistent with the
# [tsinfer model description](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726478/).
# Here, a zero-mismatch focal component with free endpoint states has a better
# likelihood when it uses fewer switches. Its whole-span counterpart has
# additional constraints from both neighboring regions. The independently
# matched control therefore supplies the relevant like-for-like test of the
# supposed HMM-versus-greedy contradiction.
#
# **Those extra focal switches are not necessary to reproduce the focal alleles.**
# They also give no improvement in this simulation's mean topology-only KC
# distance: saved greedy 268.545, pipeline reduced 268.890, full HMM 267.605.
# Focal-only reduced is 268.645. These differences are small and do not imply
# identical trees; allele-exact paths may attach to different, equally matching
# ancestors. This experiment supports an optimization-domain explanation for
# the switch surplus, without establishing a universal topology ranking.
#
# The greedy heuristic has two real limitations, neither explaining this large
# k=0 surplus: it is restricted to generated seed-anchored intervals, so the
# local inferred-reference HMM occasionally finds a slightly shorter exact path;
# and it ignores copied-allele mismatch penalties when k>0. The matched k=1
# plots show that its lower switch counts come with many more mismatches. With
# μ=1e-20 these are expensive, so minimum switch count alone is not maximum
# likelihood. The exported `full_model_log_likelihood_delta` scores each final
# path with one common full-reference model, including both penalties.
#
# A direct correction to test in the pipeline is to match each maximal covered
# component independently on the reduced reference, then fill gaps as before.
# This produces mean total switch increases of 3.72% and 4.23% in these controls,
# close to greedy's 3.87% and 5.00%. It still has unconditioned seams; jointly
# optimizing focal/gap endpoint continuity would be a separate objective.
# Merely making gaps missing is not the same as resetting component endpoints.
# No upstream matching implementation is changed by this diagnostic notebook.
