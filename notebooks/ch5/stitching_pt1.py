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
# the scientific settings cell below. The notebook reads existing
# outputs and copied references; it does not rerun matching or change its inputs.
# Edit this `.py` file and use Jupytext to synchronize the paired notebook.

# %%
import dataclasses
import logging
import pathlib

import IPython.display as ipython_display
import numpy as np
import pandas as pd

from ch5 import nb_plot, nb_utils

repository_dir = pathlib.Path.cwd()
logger = logging.getLogger(__name__)

# %%
repository_dir = pathlib.Path.cwd()
diagnostic_config_path = {
    "datasets": [
        {
            "name": "out_of_africa_n100_10mbp",
            "label": "Simulation, 100 individuals, 10 Mb",
            "deep_dive": {"sample_id": "tsk_18", "ploidy_index": 0},
            "haplotypes": [
                {"sample_id": "tsk_18", "ploidy_index": 0},
                {"sample_id": "tsk_40", "ploidy_index": 1},
                {"sample_id": "tsk_96", "ploidy_index": 1},
            ],
            "populations": ["YRI", "CEU", "CHB"],
        },
        {
            "name": "tgp_chr20_n100",
            "label": "1KGP chromosome 20, 100 individuals, 10 Mb",
            "deep_dive": {"sample_id": "NA18582", "ploidy_index": 0},
            "haplotypes": [
                {"sample_id": "NA07019", "ploidy_index": 1},
                {"sample_id": "NA18582", "ploidy_index": 0},
                {"sample_id": "NA19150", "ploidy_index": 0},
            ],
            "populations": ["AFR", "EUR", "EAS"],
        },
    ],
    "detail": {"context_sites": 25, "max_sites": 240, "min_agreement_gap_sites": 5},
    "outliers": {
        "output_dir": repository_dir / "data/ch5/stitching_outliers",
        "criteria": [
            {
                "metric": "num_extra_switches",
                "label": "Most extra switches",
                "ascending": False,
            },
            {
                "metric": "focal_agreement_percentage",
                "label": "Lowest focal agreement",
                "ascending": True,
            },
            {
                "metric": "gap_agreement_percentage",
                "label": "Lowest gap agreement",
                "ascending": True,
            },
        ],
        "short_gap_max_sites": 10,
        "short_gap_site_axis": True,
        "gaps_to_display": 8,
        "divergences_to_display": 8,
        "max_windows_per_haplotype": 5,
    },
    "distance_bin_edges": [0, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000, 100000],
    "cohort": {
        "num_haplotypes": 200,
        "num_individuals": 100,
        "output_dir": repository_dir / "data/ch5/stitching_cohort",
    },
}
diagnostic_config_path["base_dir"] = repository_dir
config = diagnostic_config_path
experiment_config = {
    "input_dir": repository_dir / "data/ch5/match_eval",
    "coverage_input_dir": repository_dir.parent / "tsinfer-match-eval/data/match_eval",
    "output_dir": repository_dir / "data/ch5/chunk_matching_experiment",
    "ac_cutoff": 10,
    "min_focal_ac": 2,
    "workers": None,
    "score_tolerance": 1e-08,
    "datasets": [
        {"name": "out_of_africa_n100_10mbp", "haplotypes": None},
        {"name": "tgp_chr20_n100", "haplotypes": None},
    ],
}
input_dir = pathlib.Path(experiment_config["input_dir"]).expanduser()
output_dir = pathlib.Path(experiment_config["output_dir"]).expanduser()
if not input_dir.is_absolute():
    input_dir = repository_dir / input_dir
if not output_dir.is_absolute():
    output_dir = repository_dir / output_dir
coverage_input_dir = pathlib.Path(experiment_config["coverage_input_dir"]).expanduser()
gap_comparison_dir = (
    repository_dir / "data/ch5/stitching_gap_diagnostics"
).expanduser()
if not coverage_input_dir.is_absolute():
    coverage_input_dir = repository_dir / coverage_input_dir
if not gap_comparison_dir.is_absolute():
    gap_comparison_dir = repository_dir / gap_comparison_dir
diagnosing_stitched_focal_and_gap_matches_result = nb_plot.make_path_colormaps()
category_cmap = diagnosing_stitched_focal_and_gap_matches_result.category_cmap
category_norm = diagnosing_stitched_focal_and_gap_matches_result.category_norm
source_cmap = diagnosing_stitched_focal_and_gap_matches_result.source_cmap
source_norm = diagnosing_stitched_focal_and_gap_matches_result.source_norm

# %%
datasets = [
    nb_utils.load_dataset(
        settings,
        experiment_config=experiment_config,
        input_dir=input_dir,
        output_dir=output_dir,
        plot_terminal_site_width_bp=1,
    )
    for settings in config["datasets"]
]
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
coverage_agreement_and_switches_across_the_saved_haplotypes_result = (
    nb_plot.plot_coverage_agreement_and_switches_across_the_saved_haplotypes(
        datasets=datasets,
        experiment_config=experiment_config,
        figsize=[16, 8],
        xlim=(0, 1),
        xlabel="Fraction of inference sites",
        fontsize="small",
        height=0.35,
        xlabel_2="Parent switches (upper: full HMM; lower: stitched)",
        dpi=120,
        show_top_spine=True,
        show_right_spine=True,
    )
)
handles = coverage_agreement_and_switches_across_the_saved_haplotypes_result.handles
switch_rows = (
    coverage_agreement_and_switches_across_the_saved_haplotypes_result.switch_rows
)
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
nb_plot.plot_extra_switch_fraction(
    datasets=datasets,
    figsize=[16, 4],
    color="#6A3D9A",
    ylim=(0, 1),
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

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
coverage_handles = (
    nb_plot.plot_where_retained_focal_pieces_and_gap_fills_agree_with_the_full(
        category_cmap=category_cmap,
        category_norm=category_norm,
        datasets=datasets,
        source_cmap=source_cmap,
        source_norm=source_norm,
        figsize=[16, 7],
        dpi=120,
        show_top_spine=True,
        show_right_spine=True,
    )
)

# %% [markdown]
# ## Whole-span paths for the selected simulated and real haplotypes
#
# The full HMM and stitched paths use the same original-node ID scale. Node IDs
# identify parents; their vertical spacing has no biological distance meaning.
# The source and agreement tracks show which parts were retained and which were
# rematched. A gap's parent can agree with the full HMM for only part of that gap.


# %%
for dataset in datasets:
    nb_plot.plot_overview(
        dataset,
        category_cmap=category_cmap,
        category_norm=category_norm,
        coverage_handles=coverage_handles,
        plot_full_color="#222222",
        plot_genome_unit_bp=1000000,
        plot_genome_unit_label="Mb",
        plot_overview_figsize=[16, 7],
        plot_stitched_color="#6A3D9A",
        source_cmap=source_cmap,
        source_norm=source_norm,
        dpi=120,
        show_top_spine=True,
        show_right_spine=True,
    )

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
gap_tables = {}
detail_windows = {}
for dataset in datasets:
    gaps = nb_utils.gap_diagnostics(dataset)
    gap_tables[dataset.name] = gaps
    detail_windows[dataset.name] = nb_utils.select_windows(dataset, gaps, config=config)
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
for dataset in datasets:
    for window in detail_windows[dataset.name]:
        nb_plot.plot_detail(
            dataset,
            window,
            category_cmap=category_cmap,
            category_norm=category_norm,
            coverage_handles=coverage_handles,
            plot_detail_figsize=[16, 7],
            plot_detail_unit_bp=1000,
            plot_detail_unit_label="kb",
            plot_full_color="#222222",
            plot_parent_row_height_in=0.24,
            plot_source_colors=["#B8DED4", "#FFE3A3"],
            plot_stitched_color="#6A3D9A",
            source_cmap=source_cmap,
            source_norm=source_norm,
            dpi=120,
            show_top_spine=True,
            show_right_spine=True,
        )

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
distance_rows = []
bins = np.array(config["distance_bin_edges"])
nb_plot.plot_does_disagreement_concentrate_near_focal_gap_seams(
    bins=bins,
    datasets=datasets,
    distance_rows=distance_rows,
    figsize=[16, 4],
    marker="o",
    ylim=(0, 1),
    xlabel="Distance to nearest seam (inference-site steps)",
    alpha=0.2,
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)
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
gap_comparisons = {
    dataset.name: nb_utils.compare_gap_determination(
        dataset,
        config=config,
        coverage_input_dir=coverage_input_dir,
        experiment_config=experiment_config,
        gap_comparison_dir=gap_comparison_dir,
        input_dir=input_dir,
    )
    for dataset in datasets
}
gap_summary = pd.DataFrame(
    [result.summary for results in gap_comparisons.values() for result in results]
)
ipython_display.display(gap_summary.round(4))

# %%
nb_plot.plot_gap_determination(
    datasets=datasets,
    experiment_config=experiment_config,
    gap_comparisons=gap_comparisons,
    figsize=(12, 4),
    width=0.35,
    ylabel="Number of maximal gap intervals",
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %%
nb_plot.plot_gap_determination_2(
    datasets=datasets,
    gap_comparisons=gap_comparisons,
    figsize=(12, 4),
    width=0.35,
    ylabel="Total number of gap intervals across saved haplotypes",
    title="Total gaps: HMM extraction versus zero-mismatch coverage",
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %%
handles = nb_plot.plot_gap_determination_3(
    datasets=datasets,
    gap_comparisons=gap_comparisons,
    figsize=(12, 5),
    xlim=(0, 1),
    xlabel="Fraction of inference sites",
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

# %%
nb_plot.plot_gap_determination_4(
    datasets=datasets,
    gap_comparisons=gap_comparisons,
    handles=handles,
    figsize=(12, 6),
    dpi=120,
    show_top_spine=True,
    show_right_spine=True,
)

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
    hmm_count = sum((result.summary["hmm_num_gaps"] for result in results))
    zero_count = sum((result.summary["zero_coverage_num_gaps"] for result in results))
    intersection = sum((result.summary["overlap_sites"] for result in results))
    hmm_only = sum((result.summary["hmm_only_gap_sites"] for result in results))
    zero_only = sum((result.summary["zero_only_gap_sites"] for result in results))
    iou = intersection / (intersection + hmm_only + zero_only)
    print(
        f"{dataset.label}: {hmm_count} HMM gaps versus {zero_count} zero-coverage gaps; site-weighted gap IoU {iou:.1%}; {zero_only:,} accepted sites outside all anchored exact intervals."
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
    full_counts = nb_utils.switch_breakdown(case, case.full_parents)
    stitched_counts = nb_utils.switch_breakdown(case, case.stitched_parents)
    extra = stitched_counts - full_counts
    print(
        f"{dataset.label}, {case.label}: parent agreement {case.summary['fraction_parent_agreement']:.1%}; extra switches {int(extra.sum())} = focal {extra[0]} + gaps {extra[1]} + seams {extra[2]}; extra switch prop. {case.summary['extra_switch_fraction']:.1%}"
    )
