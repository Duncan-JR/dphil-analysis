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
# # Chapter 3 figures
#
# Change the paths, dataset prefixes and settings below to regenerate the chapter.
# Each call displays its figure here and saves it to dphil-analysis/figures/ch3.
# Only cells marked #include export their referenced figure files to
# dphil-thesis/figures/ch3. The tree illustration exports the PDF used by TeX;
# SVG source stays in memory for notebook display without creating SVG files.
# Simulation figures use the full alternative 0.5 aggregate in data/ch3, with
# separate focal-site caches in data/ch3/alt_version. The multiple-demography
# comparison uses the alternative Out-of-Africa aggregate and updated new-branch
# aggregates for the other three models.
# The plotting module is self-contained and does not import `tsinfer-paper`.
# Run src/ch3_extract_data.py on the machine holding the variant Zarr stores.
# Transfer PREFIX_zarr_data.csv alongside the aggregate CSV; the main plots
# need no Zarrs locally. Per-focal-site tables are written to data_folder.
# The separate phase-switch illustration still uses its chromosome 1 Zarrs.
# The empirical DAF comparison uses the local TGP chromosome 20 Zarr.

# %%
import shutil
import subprocess
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import polars as pl
import dataclasses
import pathlib
import importlib
import xml.etree.ElementTree as ET
import IPython.display as ipd
import numpy as np
import pandas as pd
import tskit
import ch3_plot
importlib.reload(ch3_plot)
import ch3_ancestor_extension
importlib.reload(ch3_ancestor_extension)

# %% [markdown]
# ## Ancestor extension: toy example
#
# Six samples and eight sites show both shared pruning and the shorter 0.5
# ancestor. The focal site has carriers {a,b,c,d}; site 2's doubleton {a,d}
# lies within that set. Sites 3–4 are older (five derived carriers) and remove b
# in both versions. Sites 5–6's doubletons {c,f} cross the sample-set boundary
# and remove c only in 0.5. Derived allele counts supply site times.
# The generator replays 0.4.1 and the corrected 0.5.x backport, including
# assignment of 0 at younger sites regardless of the current consensus.
# The replay follows `AncestorBuilder.make_ancestor` from tag 0.4.1 and the
# [0.5.x backport](https://github.com/Duncan-JR/tsinfer/tree/0.5.x)
# (commit `d694eea`). Both removal and flag updates use the same gate.
#
# Site 2 cannot mark samples for removal because both carriers are within S.
# Red conflicts mark and remove b at sites 3 and 4. Orange conflicts mark and
# remove c at sites 5 and 6. After b leaves, S={a,c,d}; f is a derived carrier
# outside S, making the c/f doubletons informative despite their younger ages.
# Removing c halves the original four-sample set, so 0.5 stops after site 6.
# The 0.4 ancestor continues through site 8, two sites longer in this window;
# its arrow indicates that its stopping criterion has still not been reached.
# Younger sites are assigned 0 in both ancestors regardless of their consensus.
# Both toy figures are also included in Chapter 3 of the thesis.

# %%
#include
figure_folder = pathlib.Path("~/work/dphil-analysis/figures/ch3").expanduser()
thesis_figure_folder = pathlib.Path("~/work/dphil-thesis/figures/ch3").expanduser()
figure_folder.mkdir(parents=True, exist_ok=True)
thesis_figure_folder.mkdir(parents=True, exist_ok=True)


def export_thesis_figure(filename):
    """Copy a referenced figure to the thesis from a cell marked #include."""
    source = figure_folder / filename
    destination = thesis_figure_folder / filename
    shutil.copy2(source, destination)

extension_config = ch3_ancestor_extension.FigureConfig(
    # Columns are one-indexed sites 1–8; site 1 is focal.
    genotypes=(
        (1, 1, 1, 1, 0, 0, 1, 1),  # a
        (1, 1, 1, 1, 0, 0, 1, 1),  # d
        (1, 0, 0, 0, 0, 0, 0, 0),  # b
        (1, 0, 1, 1, 1, 1, 1, 1),  # c
        (0, 0, 1, 1, 1, 1, 1, 1),  # f
        (0, 0, 1, 1, 0, 0, 1, 1),  # g
    ),
    samples=("a", "d", "b", "c", "f", "g"),
    focal_site=1,
    version_colors=ch3_plot.PlotConfig().version_colors,
    figsize=(9.0, 4.9),
    site_spacing=1.4,
    dpi=400,
)
ch3_ancestor_extension.plot_ancestor_extension(
    extension_config, figure_folder=figure_folder
);
export_thesis_figure("ancestor_extension_comparison.png")

# %% [markdown]
# ## Genealogical justification for ancestor extension
#
# Three trees reproduce **all 48 genotypes** above. At sites 1–2, E's doubleton
# {a,d} is nested within F's focal clade {a,b,c,d}. At sites 3–4, b has left F,
# while the two older mutations each cover everyone except b: both algorithms
# detect b's repeated disagreement. At sites 5–8, c joins f under G. The {c,f}
# doubletons at sites 5–6 overlap S={a,c,d} without being nested within it;
# they supply the extra evidence used by 0.5. The older mutations at sites 7–8
# are included in 0.4's longer ancestor. Every mutation occurs once, so the
# conflicting clades require recombination; the genotypes do not identify a
# unique recombination history.
#
# Red branches and bold red sample labels identify F's descendants in the first
# two trees and the surviving a/d clade in the last tree; orange
# branches connect G to c and f in the final tree. E remains purple to match
# its doubleton in the matrix, while its descendant branches are red because
# they lie within F. Circled labels identify the ancestors. Crosses mark 0→1
# mutations and are labelled by site. Ages are illustrative on a shared linear
# scale, rather than the derived allele counts used as inference site times.
# Sites have irregular illustrative positions across 100 bp; mutation and arrow
# labels retain the site numbers 1–8 used in the genotype matrix.
# F disappears at the second breakpoint (47 bp), where retaining it above E
# would create a unary node. E and G remain distinct coalescent ancestors.
# The unlabelled multifurcations are unresolved branches: a binary refinement
# permits b's departure at the first breakpoint and c joining f at the second,
# with no change to the labelled mutation nodes or any observed genotypes.
# This matches the non-unary reference representation used for ancestor
# evaluation: E, F and G are the mutation nodes for sites 2, 1 and 5–6,
# respectively, rather than persistent names for descendant sample sets.
#
# Using the evaluator's decoding convention, F's true haplotype is `1011----`:
# it is defined at sites 1–4 and missing thereafter. The inferred haplotypes
# are `10110011` (0.4) and `101100--` (0.5). Both recover the four known alleles
# exactly; 0.5 overshoots by two sites instead of four. Thus this example shows
# reduced span inflation, not improved allelic accuracy or an exact boundary.
# The true tree breakpoint (47 bp) and boundaries inferred from site coordinates
# are different quantities; the evaluator uses the next shared site as the
# right boundary of a contiguous stretch of known genotypes.
#
# Construction follows tskit's [tables and editing tutorial](https://tskit.dev/tutorials/tables_and_editing.html),
# and SVG styling follows its [visualization tutorial](https://tskit.dev/tutorials/viz.html).
# This plot is also included in Chapter 3 of the thesis.

# %%
extension_tree_config = {
    "sequence_length": 100,
    "plot_interval": (0, 100),
    "site_positions": (6, 17, 31, 39, 58, 65, 81, 96),
    "intervals": ((0, 24), (24, 47), (47, 100)),
    "node_ages": {"E": 2, "F": 4, "G": 2, "A": 6, "R": 10},
    "topologies": (
        {"A": ("F", "f", "g"), "F": ("E", "b", "c"), "E": ("a", "d")},
        {"R": ("A", "b"), "A": ("F", "f", "g"), "F": ("E", "c"), "E": ("a", "d")},
        {"R": ("A", "b"), "A": ("E", "G", "g"),
         "E": ("a", "d"), "G": ("c", "f")},
    ),
    # Site number, branch's child node, and mutation age. Positions are separate.
    "mutations": ((1, "F", 5.3), (2, "E", 3.1), (3, "A", 7.2), (4, "A", 8.6),
                  (5, "G", 3.2), (6, "G", 4.5), (7, "A", 7.2), (8, "A", 8.6)),
    "node_colors": {"F": extension_config.focal_color,
                    "E": extension_config.contained_color,
                    "G": extension_config.version_colors[1]},
    "descendant_colors": {"F": extension_config.focal_color,
                          "E": extension_config.focal_color,
                          "G": extension_config.version_colors[1]},
    "ink_color": extension_config.ink_color,
    "muted_color": extension_config.muted_color,
    "older_color": "#b59a66",
    "background_color": "#f1f4f6",
    "canvas_color": "white",
    "size": (680, 300),
    "axis_gap": 14,
    "y_ticks": (0, 2, 4, 6, 8, 10),
    "font_size": 14.5,
    "mutation_font_size": 14.8,
    "site_font_size": 13.8,
    "sample_font_size": 15.8,
    "symbol_size": 6,
    "label_radius": 11,
    "site_label_offset": -16,
    "svg_id": "ancestor_extension_trees",
}

# %%
# Build the nodes, edges, sites and mutations directly, without simulation.
extension_tables = tskit.TableCollection(extension_tree_config["sequence_length"])
extension_tables.time_units = "uncalibrated"
extension_node_ids = {}
for name in extension_config.samples:
    extension_node_ids[name] = extension_tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
for name, age in extension_tree_config["node_ages"].items():
    extension_node_ids[name] = extension_tables.nodes.add_row(time=age)

for interval, topology in zip(extension_tree_config["intervals"], extension_tree_config["topologies"]):
    left, right = interval
    for parent, children in topology.items():
        for child in children:
            extension_tables.edges.add_row(
                left=left, right=right,
                parent=extension_node_ids[parent], child=extension_node_ids[child],
            )
for site_number, node_name, age in extension_tree_config["mutations"]:
    position = extension_tree_config["site_positions"][site_number - 1]
    site_id = extension_tables.sites.add_row(position=position, ancestral_state="0")
    extension_tables.mutations.add_row(
        site=site_id, node=extension_node_ids[node_name], derived_state="1", time=age,
    )
extension_tables.sort()
extension_ts = extension_tables.tree_sequence()

# Verify compatibility with the existing haplotype illustration, including all
# background mutations, rather than just the highlighted doubletons.
expected_extension_genotypes = np.asarray(extension_config.genotypes)
np.testing.assert_array_equal(extension_ts.genotype_matrix().T, expected_extension_genotypes)
np.testing.assert_array_equal(extension_ts.sites_position, extension_tree_config["site_positions"])
assert extension_ts.num_trees == len(extension_tree_config["intervals"])
old_extension = ch3_ancestor_extension.trace_ancestor(extension_config, extension_config.versions[0])
new_extension = ch3_ancestor_extension.trace_ancestor(extension_config, extension_config.versions[1])
extension_b = extension_config.samples.index("b")
extension_c = extension_config.samples.index("c")
assert old_extension.marks[extension_b] == [2]  # Both mark b at site 3.
assert new_extension.marks[extension_b] == [2]
assert old_extension.removals == {extension_b: 3}  # Both remove b at site 4.
assert new_extension.marks[extension_c] == [4]  # Only 0.5 marks c at site 5.
assert new_extension.removals == {extension_b: 3, extension_c: 5}
assert old_extension.end == 8
assert new_extension.end == 6
assert old_extension.end - new_extension.end == 2

# Match get_true_ancestor's single-mutation rule, and check that these mutation
# nodes are coalescences rather than unary waypoints or sample-set labels.
for variant in extension_ts.variants():
    tree = extension_ts.at(variant.site.position)
    carriers = variant.samples[variant.genotypes == 1]
    true_node = variant.site.mutations[0].node
    assert true_node == tree.mrca(*carriers)
    assert tree.num_children(true_node) >= 2
for tree in extension_ts.trees():
    assert all(tree.num_children(node) != 1 for node in tree.nodes())
assert extension_node_ids["F"] not in set(extension_ts.last().nodes())

# process_ancestor_chunk marks all nodes as samples before decoding, so an
# ancestor absent from a tree is isolated and therefore missing at that site.
extension_truth_tables = extension_ts.dump_tables()
extension_truth_tables.nodes.flags = np.full(extension_ts.num_nodes, tskit.NODE_IS_SAMPLE, dtype=np.uint32)
extension_truth_ts = extension_truth_tables.tree_sequence()
extension_true_haplotype = extension_truth_ts.genotype_matrix(
    samples=[extension_node_ids["F"]], isolated_as_missing=True,
)[:, 0]
np.testing.assert_array_equal(extension_true_haplotype, [1, 0, 1, 1, -1, -1, -1, -1])
extension_true_sites = np.flatnonzero(extension_true_haplotype != tskit.MISSING_DATA)
np.testing.assert_array_equal(old_extension.haplotype[extension_true_sites], extension_true_haplotype[extension_true_sites])
np.testing.assert_array_equal(new_extension.haplotype[extension_true_sites], extension_true_haplotype[extension_true_sites])
assert old_extension.end - (extension_true_sites[-1] + 1) == 4
assert new_extension.end - (extension_true_sites[-1] + 1) == 2

# %%
extension_svg_id = extension_tree_config["svg_id"]
extension_ink = extension_tree_config["ink_color"]
extension_muted = extension_tree_config["muted_color"]
extension_background = extension_tree_config["background_color"]
# Scope every selector so this SVG cannot recolour other notebook figures.
extension_style = [
    f"#{extension_svg_id} text {{font-family: Arial, sans-serif; fill: {extension_ink};"
    f" font-size: {extension_tree_config['font_size']}px;}}",
    f"#{extension_svg_id} .edge {{stroke: {extension_muted}; stroke-width: 1.4;}}",
    f"#{extension_svg_id} .node > .sym {{fill: {extension_muted}; stroke: white; stroke-width: 1;}}",
    f"#{extension_svg_id} .sample > .lab {{font-style: italic;"
    f" font-size: {extension_tree_config['sample_font_size']}px;}}",
    f"#{extension_svg_id} .mut .sym {{stroke-width: 2;}}",
    f"#{extension_svg_id} .mut .lab {{font-size: {extension_tree_config['mutation_font_size']}px;}}",
    f"#{extension_svg_id} .background path {{fill: {extension_background}; fill-opacity: 1;}}",
    f"#{extension_svg_id} .background path:nth-child(even) {{fill: white;}}",
    f"#{extension_svg_id} .site > .lab {{font-size: {extension_tree_config['site_font_size']}px;"
    " text-anchor: middle; font-weight: bold;}",
]
extension_node_attrs = {}
for name, color in extension_tree_config["node_colors"].items():
    node_id = extension_node_ids[name]
    extension_node_attrs[node_id] = {"r": extension_tree_config["label_radius"]}
    extension_style.extend([
        f"#{extension_svg_id} .n{node_id} > .sym {{fill: white; stroke: {color}; stroke-width: 1.8;}}",
        f"#{extension_svg_id} .n{node_id} > .lab {{fill: {color}; font-weight: bold;"
        " text-anchor: middle; dominant-baseline: central;}",
    ])
for name, color in extension_tree_config["descendant_colors"].items():
    node_id = extension_node_ids[name]
    extension_style.extend([
        f"#{extension_svg_id} .n{node_id} .node > .edge {{stroke: {color}; stroke-width: 2;}}",
        f"#{extension_svg_id} .n{node_id} .sample > .lab {{fill: {color}; font-weight: bold;}}",
    ])
extension_color_by_node = {
    extension_node_ids[name]: color
    for name, color in extension_tree_config["node_colors"].items()
}
for mutation in extension_ts.mutations():
    color = extension_color_by_node.get(mutation.node, extension_tree_config["older_color"])
    extension_style.extend([
        f"#{extension_svg_id} .m{mutation.id} > .sym {{stroke: {color};}}",
        f"#{extension_svg_id} .m{mutation.id} > .lab {{fill: {color};}}",
        f"#{extension_svg_id} .site.s{mutation.site} > .lab {{fill: {color};}}",
    ])

extension_node_labels = {extension_node_ids[name]: name for name in extension_config.samples}
for name in extension_tree_config["node_colors"]:
    extension_node_labels[extension_node_ids[name]] = name
extension_mutation_labels = {}
for mutation in extension_ts.mutations():
    extension_mutation_labels[mutation.id] = str(mutation.site + 1)

extension_tree_svg = extension_ts.draw_svg(
    size=extension_tree_config["size"],
    canvas_size=(extension_tree_config["size"][0],
                 extension_tree_config["size"][1] + extension_tree_config["axis_gap"]),
    x_lim=extension_tree_config["plot_interval"],
    x_label="Genome position (bp)",
    time_scale="time", y_axis=True, y_gridlines=False,
    y_label="Age (generations ago)", y_ticks=extension_tree_config["y_ticks"],
    node_labels=extension_node_labels, node_attrs=extension_node_attrs,
    mutation_labels=extension_mutation_labels,
    symbol_size=extension_tree_config["symbol_size"],
    root_svg_attributes={"id": extension_svg_id},
    style="\n".join(extension_style),
    preamble=(
        f'<rect width="100%" height="100%" fill="{extension_tree_config["canvas_color"]}"/>'
    ),
)
# Centre the labels on their node symbols using SVG attributes so the white
# circles also render correctly in SVG viewers without CSS transform support.
ET.register_namespace("", "http://www.w3.org/2000/svg")
extension_svg_root = ET.fromstring(extension_tree_svg)
extension_x_axis = extension_svg_root.find(".//{http://www.w3.org/2000/svg}g[@class='x-axis']")
# Move the axis and its connecting background together, preserving the trees'
# alignment with the age scale while increasing the gap below the sample labels.
extension_axis_shift = f"translate(0 {extension_tree_config['axis_gap']})"
extension_x_axis.set("transform", extension_axis_shift)
extension_background_group = extension_svg_root.find(
    ".//{http://www.w3.org/2000/svg}g[@class='background']"
)
extension_background_group.set("transform", extension_axis_shift)
extension_circle_classes = {f"n{node_id}" for node_id in extension_node_attrs}
for group in extension_svg_root.iter("{http://www.w3.org/2000/svg}g"):
    group_classes = set(group.get("class", "").split())
    if group_classes.isdisjoint(extension_circle_classes):
        continue
    label = group.find("{http://www.w3.org/2000/svg}text")
    label.set("transform", "translate(0 0)")
# Add the site numbers above tskit's site arrows.
for site in extension_ts.sites():
    site_group = extension_svg_root.find(
        f".//{{http://www.w3.org/2000/svg}}g[@class='site s{site.id}']"
    )
    site_label = ET.SubElement(
        site_group, "{http://www.w3.org/2000/svg}text",
        {"class": "lab", "x": "0", "y": str(extension_tree_config["site_label_offset"])},
    )
    site_label.text = str(site.id + 1)
extension_tree_svg = ET.tostring(extension_svg_root, encoding="unicode")

#include
# PDF conversion uses rsvg-convert from librsvg.
subprocess.run(
    [
        "rsvg-convert", "--format=pdf", "--output",
        str(figure_folder / "ancestor_extension_trees.pdf"),
    ],
    input=extension_tree_svg, text=True, check=True,
)
export_thesis_figure("ancestor_extension_trees.pdf")
ipd.display(ipd.SVG(extension_tree_svg))

# %%
# Prefixes include the simulation region, sample size and seed, but not replicate
# or error-profile suffixes. All available replicates are pooled.
data_folder = pathlib.Path("~/work/dphil-analysis/data/ch3").expanduser()
aggregated_folder = pathlib.Path("~/work/tsinfer-anc-eval/data/anc_eval/dataframes").expanduser()
data_prefix = "OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1"
standard_aggregated_path = aggregated_folder / f"{data_prefix}-aggregated.csv"
aggregated_path = data_folder / f"{data_prefix}-aggregated_alt_version.csv"
# Separate caches keep the original and alternative algorithm results distinct.
focal_data_folder = data_folder / "alt_version"
focal_data_folder.mkdir(exist_ok=True)
# The preprocessing helper expects the canonical aggregate filename.
preprocessing_input = focal_data_folder / f"{data_prefix}-aggregated.csv"
if not preprocessing_input.exists():
    preprocessing_input.symlink_to(aggregated_path)
zarr_data_path = data_folder / f"{data_prefix}_zarr_data.csv"
phasing_folder = pathlib.Path("~/work/tsinfer-anc-eval/data/phasing").expanduser()
phasing_prefix = "OutOfAfrica_3G09-chr1-L0-R123400000-n20-s1"
tgp_zarr_path = pathlib.Path("~/work/dphil-analysis/data/zarr_vcfs/tgp/chr20/data.zarr").expanduser()

config = ch3_plot.PlotConfig(
    genotype_multipliers=(1, 2, 5, 10, 100),
    transition_multipliers=(1, 10),
    mispol_rates=(0, 0.01, 0.05, 0.1),
    daf_size=(5, 4.5),
    tgp_region_start=0,
    tgp_region_end=20_000_000,
    jitter_alpha=0.15,  # Default opacity for all jitter points (alpha can override).
    phase_ser=0.0003,
    phase_right_mb=123.4,
    phase_individuals=5,
    workers=None,  # Use available CPU cores for focal-site preprocessing.
    dpi=300,
)
# Rebuild automatically when the full alternative simulation or truth changes.
rebuild_focal_tables = False
focal_paths = ch3_plot.FocalPaths(
    focal_data_folder / f"{data_prefix}_focal_overshoots.csv",
    focal_data_folder / f"{data_prefix}_focal_spans.csv",
)
cached_truth_path = focal_data_folder / zarr_data_path.name
if not cached_truth_path.exists():
    cached_truth_path.symlink_to(zarr_data_path)
focal_cache_paths = (focal_paths.overshoots, focal_paths.spans)
focal_inputs_mtime = max(aggregated_path.stat().st_mtime, zarr_data_path.stat().st_mtime)
focal_cache_current = all(
    path.exists() and path.stat().st_mtime >= focal_inputs_mtime
    for path in focal_cache_paths
)
if rebuild_focal_tables or not focal_cache_current:
    processed_paths = ch3_plot.preprocess_data(
        preprocessing_input, zarr_data_path, focal_data_folder, config
    )
data = ch3_plot.load_data(focal_data_folder, data_prefix, config)
# Available pure phase-error rates are included in the exported CSV.
config.phase_rates = data.phase_rates

# %% [markdown]
# ## Error-free results
#
# The span and overshoot boxplots use the custom DAF cutoffs supplied below.
# Without cutoffs, five rounded quantile bins cover the full frequency range.
# Boxes show quartiles, medians and 1.5 IQR whiskers; outliers are hidden.
# All boxplots enable y_log=False. Spans use a standard log axis; signed
# overshoots use a symmetric log axis, linear between -1 kb and +1 kb.
# Zero values remain visible and the outer ticks mark powers of ten in Mb.
# Each focal site contributes its own true/inferred span comparison and one
# overshoot per side. Sites remain separate when they share a true or inferred
# ancestor, including when their polarisation states differ. Counts below the
# boxes show focal sites per bin. Bins use each focal allele's observed frequency.

# %%
# Focal-site counts use a logarithmic scale, with exact counts above each bar.
ch3_plot.plot_ancestor_spans(
    data,
    figure_folder=figure_folder,
    cutoffs=[0,0.001,0.01,0.5,1],
    figsize=(7, 4.5),
    y_log=True,
)

# %% [markdown]
# ### Mean ancestor span by focal allele frequency
#
# Use the same 20 equal-width DAF bins as the error-direction curve, including
# DAF 1 in the final bin. Calculate arithmetic means at bin midpoints, with
# each focal site counted separately even when sites share an ancestor.
# Simulation curves use only the error-free profile; truth is counted once.
# Separate plots use solid curves with the same version colours; simulation
# truth is dark red. Both mean-span panels share logarithmic y-axis limits.
# Each plot has a short focal-site count panel below it; counts are checked
# to match between versions within each dataset. Both are exported to the thesis.
#
# The corrected 1KGP spans run between the first and last included inference
# sites, without using sequence_length. The title reports the first-to-last
# inference-site extent from the CSV (26.0 Mb), rather than the nominal
# 28.1 Mb chr20p mask. The exact positions and width are displayed below.
# No spans are clipped or rescaled. Its `inferred_time` values are the benchmark's
# default focal-frequency proxies, rather than ages in generations (see the
# [tsinfer documentation](https://tskit.dev/tsinfer/docs/stable/api.html#tsinfer.VariantData)).

# %%
tgp_ancestor_spans_path = data_folder / "tgp_ancestor_spans.csv"
# The replacement export contains only 0.5; retain the corrected 0.4 baseline.
tgp_original_spans = pd.read_csv(data_folder / "tgp_ancestor_spans_0.4.1.csv")
tgp_alternative_spans = pd.read_csv(tgp_ancestor_spans_path)
assert tgp_original_spans.version.eq(config.versions[0]).all()
assert tgp_alternative_spans.version.eq(config.versions[1]).all()
tgp_combined_spans = pd.concat([tgp_original_spans, tgp_alternative_spans], ignore_index=True)
tgp_combined_spans_path = data_folder / "tgp_ancestor_spans_alt.csv"
tgp_combined_spans.to_csv(tgp_combined_spans_path, index=False)
tgp_ancestor_spans = ch3_plot.load_tgp_ancestor_spans(tgp_combined_spans_path)
ancestor_span_curve_summary = ch3_plot.summarise_ancestor_span_curves(data, tgp_ancestor_spans)
tgp_inference_site_extent = tgp_ancestor_spans.groupby("version").focal_position.agg(
    first_site_bp="min", last_site_bp="max",
)
tgp_inference_site_extent["width_bp"] = (
    tgp_inference_site_extent.last_site_bp - tgp_inference_site_extent.first_site_bp
)
tgp_inference_site_extent["width_mb"] = tgp_inference_site_extent.width_bp / config.bp_per_mb
assert tgp_inference_site_extent.width_bp.nunique() == 1
tgp_inference_width_mb = tgp_inference_site_extent.width_mb.iloc[0]
for version, extent in tgp_inference_site_extent.iterrows():
    version_spans = tgp_ancestor_spans.loc[tgp_ancestor_spans.version == version, "inferred_span"]
    assert version_spans.between(0, extent.width_bp).all()
ipd.display(tgp_inference_site_extent.style.format({
    "first_site_bp": "{:,.0f}", "last_site_bp": "{:,.0f}",
    "width_bp": "{:,.0f}", "width_mb": "{:.6f}",
}).set_caption("1KGP title width: (last inference-site position - first position) / 1,000,000"))

# %%
#include: panel A
ch3_plot.plot_ancestor_span_curves(
    ancestor_span_curve_summary, "Simulation", config=config,
    title="Out-of-Africa simulation (20mbp)",
    figure_folder=figure_folder, y_log=True,
)
export_thesis_figure("ancestor_span_frequency_simulation.png")

# %%
#include: panel B
ch3_plot.plot_ancestor_span_curves(
    ancestor_span_curve_summary, "1000 Genomes Project", config=config,
    title=f"1KGP chromosome 20p ({tgp_inference_width_mb:.1f}mbp)",
    figure_folder=figure_folder, y_log=True,
)
export_thesis_figure("ancestor_span_frequency_tgp.png")


# %%

#include
ch3_plot.plot_ancestor_overshoot(
    data,
    figure_folder=figure_folder,
    cutoffs=[0,0.001,0.003,0.02,0.5,1],
    conf=0.001,
    figsize=(7, 4.5),
    y_log=False,
)
export_thesis_figure("ancestor_overshoot_boxplot.png")

# %% [markdown]
# ### Pooled overlap error burden
#
# Use the same error-free profiles and frequency bins as the overshoot plot.
# Error counts and overlap lengths describe the whole shared overlap and are
# repeated on left/right rows, so use only the left row of each focal-site
# comparison. Comparisons sharing an ancestor remain separate, matching the
# overshoot plot. The top panel shows the proportion with zero ancestral-state
# errors. The middle panel pools `sum(num_errors) / sum(overlap_span)` within
# each version and bin, rather than averaging per-comparison error rates.
# Bars stack true 1 / observed 0 errors (solid) below true 0 / observed 1
# errors (white with diagonal lines in the version colour). Separate legends
# identify version colours and error directions.
# Orange labels above the pooled-burden bars show 100 × (0.5 / 0.4 - 1),
# using each version's total pooled error burden in that frequency bin.
# The bottom panel counts focal-site comparisons, shared by both versions.
# This figure is saved for the error-free results section of the thesis.

# %%
#include
ancestor_overlap_errors = ch3_plot.load_ancestor_overlap_errors(aggregated_path, config=config)
ancestor_error_summary = ch3_plot.summarise_ancestor_overlap_errors(
    ancestor_overlap_errors,
    config=dataclasses.replace(config, ancestor_error_label_decimals=4),
    cutoffs=[0,0.001,0.003,0.02,0.5,1],
)
ch3_plot.plot_ancestor_overlap_errors(
    ancestor_error_summary,
    figure_folder=figure_folder,
    config=dataclasses.replace(config, ancestor_error_label_decimals=4),
    figsize=config.ancestor_error_size,
    show_burden_change=True,
)
export_thesis_figure("ancestor_overlap_errors.png")

# %% [markdown]
# ### Comparison across demographic models
#
# Compare error-free profiles from four simulations, each with 6,000 diploid
# individuals across the first 20 Mb of chromosome 20. The American admixture
# simulation samples 1,500 individuals from each of its four populations.
# Both figures pool all focal allele frequencies within each demographic model.
# The usual overshoot boxes retain both sides per focal site, with the same
# jitter and 0.1% tail quantiles as the error-free overshoot plot above.
# The pooled burden uses each whole overlap once (left rows only), retaining
# the error-free proportion and stacked error-direction panels without counts.
# Both figures are exported to the thesis figure folder.
#
# Out of Africa uses the full alternative 0.5 aggregate in data/ch3; the
# other three models use the replacement aggregates generated with the new branch.

# %%
demography_labels = {
    "OutOfAfrica_3G09": "Out of Africa\n(3G09)",
    "AmericanAdmixture_4B18": "American admixture\n(4B18)",
    "Africa_1T12": "Africa\n(1T12)",
    "Zigzag_1S14": "Zigzag\n(1S14)",
}
demography_paths = {
    model: aggregated_folder / f"{model}-chr20-L0-R2e7-n6000-s1-aggregated.csv"
    for model in demography_labels
}
demography_paths["OutOfAfrica_3G09"] = aggregated_path
demography_comparisons = ch3_plot.load_demography_comparisons(
    demography_paths, config=config,
)

# %%
#include
ch3_plot.plot_demography_overshoot_box(
    demography_comparisons, demography_labels, config=config,
    figure_folder=figure_folder,
    conf=0.001, conf_secondary=0.01, figsize=(8.5, 4.5),
    y_log=False,
)
export_thesis_figure("demography_overshoot_box.png")

# %% [markdown]
# #### Verification: IQRs and tail quantiles used in the text
#
# These checks use exactly the error-free observations in the boxplot above:
# both sides of every focal-site comparison, with no frequency bins or
# deduplication of sites that share an ancestor. IDs are local to each model.
# Quantiles use linear interpolation on the untransformed overshoots in bp.
# IQR = Q75 - Q25; divide by 1,000 for kb. The common upper bound is the
# ceiling of the largest IQR in kb across all four models and both versions.

# %%
demography_check_settings = {
    "bp_per_kb": 1000,
    "primary_tail_probability": 0.001,
    "secondary_tail_probability": 0.01,
    "quantiles": [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999],
}
demography_site_keys = ["model", "version", "replicate", "true_focal_site"]
demography_side_keys = demography_site_keys + ["side"]
assert not demography_comparisons.duplicated(demography_side_keys).any()
assert set(demography_comparisons.side) == {"left", "right"}
demography_sides_per_site = demography_comparisons.groupby(demography_site_keys).size()
assert demography_sides_per_site.eq(2).all()
demography_group_order = pd.MultiIndex.from_product(
    [demography_labels, config.versions], names=["model", "version"],
)
demography_overshoot_groups = demography_comparisons.groupby(["model", "version"])
demography_quantiles_bp = demography_overshoot_groups.overshoot.quantile(
    demography_check_settings["quantiles"], interpolation="linear",
).unstack()
demography_quantiles_bp = demography_quantiles_bp.reindex(demography_group_order)
# Cross-check the grouped pandas results against the NumPy call used by the plot.
for group, frame in demography_overshoot_groups:
    numpy_quantiles = np.quantile(frame.overshoot, demography_check_settings["quantiles"])
    np.testing.assert_allclose(numpy_quantiles, demography_quantiles_bp.loc[group].to_numpy())
demography_iqr_table = pd.DataFrame(index=demography_group_order)
demography_iqr_table["num_focal_site_sides"] = demography_overshoot_groups.size()
demography_iqr_table["q25_kb"] = demography_quantiles_bp[0.25] / demography_check_settings["bp_per_kb"]
demography_iqr_table["median_kb"] = demography_quantiles_bp[0.5] / demography_check_settings["bp_per_kb"]
demography_iqr_table["q75_kb"] = demography_quantiles_bp[0.75] / demography_check_settings["bp_per_kb"]
demography_iqr_table["iqr_kb"] = demography_iqr_table.q75_kb - demography_iqr_table.q25_kb
demography_largest_iqr = demography_iqr_table.iqr_kb.idxmax()
demography_max_iqr_kb = demography_iqr_table.iqr_kb.max()
demography_iqr_bound_kb = int(np.ceil(demography_max_iqr_kb))
assert demography_iqr_table.iqr_kb.lt(demography_iqr_bound_kb).all()
ipd.display(demography_iqr_table.style.format({
    "num_focal_site_sides": "{:,.0f}", "q25_kb": "{:.3f}",
    "median_kb": "{:.3f}", "q75_kb": "{:.3f}", "iqr_kb": "{:.3f}",
}).set_caption("Quartiles, median and IQR (Q75 - Q25) for every model and version"))
demography_iqr_bound_table = pd.DataFrame([{
    "worst_model": demography_largest_iqr[0],
    "worst_version": demography_largest_iqr[1],
    "max_iqr_kb": demography_max_iqr_kb,
    "ceil_max_iqr_kb": demography_iqr_bound_kb,
}])
ipd.display(demography_iqr_bound_table.style.format({"max_iqr_kb": "{:.3f}"})
            .set_caption("Text bound: ceil(maximum IQR in kb)"))

# %% [markdown]
# The primary caps mark Q0.001 and Q0.999 (0.1st and 99.9th percentiles),
# so each nominal tail contains 0.1% of **focal-site-side observations**.
# They do not denote 0.05% of ancestors. The secondary caps mark Q0.01 and
# Q0.99. The counts below use a strict inequality; ties can make the empirical
# lower-tail percentage slightly smaller than its nominal value.

# %%
demography_primary_tail = demography_check_settings["primary_tail_probability"]
demography_secondary_tail = demography_check_settings["secondary_tail_probability"]
demography_tail_table = pd.DataFrame(index=demography_group_order)
demography_tail_table["lower_0_1_pct_mb"] = demography_quantiles_bp[demography_primary_tail] / config.bp_per_mb
demography_tail_table["upper_99_9_pct_mb"] = demography_quantiles_bp[1 - demography_primary_tail] / config.bp_per_mb
demography_tail_table["lower_1_pct_mb"] = demography_quantiles_bp[demography_secondary_tail] / config.bp_per_mb
demography_tail_table["upper_99_pct_mb"] = demography_quantiles_bp[1 - demography_secondary_tail] / config.bp_per_mb
demography_lower_tail_counts = []
for group, frame in demography_overshoot_groups:
    threshold = demography_quantiles_bp.loc[group, demography_primary_tail]
    num_below = int((frame.overshoot < threshold).sum())
    demography_lower_tail_counts.append({
        "model": group[0], "version": group[1],
        "num_sides": len(frame), "num_below_primary_lower_quantile": num_below,
        "observed_lower_tail_pct": 100 * num_below / len(frame),
    })
ipd.display(demography_tail_table.style.format("{:.5f}")
            .set_caption("Signed tail cutoffs in Mb; round to two decimals for the text"))
demography_lower_tail_count_table = pd.DataFrame(demography_lower_tail_counts)
demography_lower_tail_count_table = demography_lower_tail_count_table.set_index(["model", "version"])
demography_lower_tail_count_table = demography_lower_tail_count_table.reindex(demography_group_order)
ipd.display(demography_lower_tail_count_table.style.format({
    "num_sides": "{:,.0f}", "num_below_primary_lower_quantile": "{:,.0f}",
    "observed_lower_tail_pct": "{:.5f}",
}).set_caption("Tail check: 100 × count(overshoot < lower primary quantile) / number of sides"))

# %% [markdown]
# Pair versions **within each model** before averaging. A positive lower-tail
# change means the tsinfer 0.5 cutoff is less negative:
# `(Q0.001_0.5 - Q0.001_0.4) / 1,000` kb. Each of the four models receives
# equal weight; this is not a quantile of pooled observations. Positive upper
# reductions mean `Q0.999_0.4 - Q0.999_0.5` is positive. The same calculation
# is shown for the secondary lower quantile to distinguish the two cap pairs.

# %%
demography_old_version, demography_new_version = config.versions
demography_lower_primary = demography_quantiles_bp[demography_primary_tail].unstack("version")
demography_upper_primary = demography_quantiles_bp[1 - demography_primary_tail].unstack("version")
demography_lower_secondary = demography_quantiles_bp[demography_secondary_tail].unstack("version")
demography_tail_changes = pd.DataFrame(index=list(demography_labels))
demography_tail_changes.index.name = "model"
demography_tail_changes["lower_0_4_mb"] = demography_lower_primary[demography_old_version] / config.bp_per_mb
demography_tail_changes["lower_0_5_mb"] = demography_lower_primary[demography_new_version] / config.bp_per_mb
demography_tail_changes["primary_lower_change_kb"] = (
    demography_lower_primary[demography_new_version] - demography_lower_primary[demography_old_version]
) / demography_check_settings["bp_per_kb"]
demography_tail_changes["upper_0_4_mb"] = demography_upper_primary[demography_old_version] / config.bp_per_mb
demography_tail_changes["upper_0_5_mb"] = demography_upper_primary[demography_new_version] / config.bp_per_mb
demography_tail_changes["upper_reduction_kb"] = (
    demography_upper_primary[demography_old_version] - demography_upper_primary[demography_new_version]
) / demography_check_settings["bp_per_kb"]
demography_tail_changes["secondary_lower_change_kb"] = (
    demography_lower_secondary[demography_new_version] - demography_lower_secondary[demography_old_version]
) / demography_check_settings["bp_per_kb"]
assert demography_tail_changes.upper_reduction_kb.gt(0).all()
ipd.display(demography_tail_changes.style.format("{:.5f}")
            .set_caption("Paired changes: positive lower change = less negative; positive upper reduction = shorter upper tail"))
demography_lower_change_sum_kb = demography_tail_changes.primary_lower_change_kb.sum()
demography_num_models = len(demography_tail_changes)
demography_mean_lower_change_kb = demography_lower_change_sum_kb / demography_num_models
demography_lower_change_working = pd.DataFrame([{
    "sum_of_primary_lower_changes_kb": demography_lower_change_sum_kb,
    "num_equally_weighted_models": demography_num_models,
    "mean_primary_lower_change_kb": demography_mean_lower_change_kb,
}])
ipd.display(demography_lower_change_working.style.format({
    "sum_of_primary_lower_changes_kb": "{:.5f}", "mean_primary_lower_change_kb": "{:.5f}",
}).set_caption("Working for the text: sum of four paired lower-tail changes / 4"))


# %% [markdown]
# **Citation check.** The [stdpopsim catalog](https://github.com/popsim-consortium/stdpopsim/blob/main/stdpopsim/catalog/HomSap/demographic_models.py)
# identifies Zigzag_1S14 as a validation model with periodic growth and decline,
# citing Schiffels and Durbin (2014), [doi:10.1038/ng.3015](https://doi.org/10.1038/ng.3015).
# This matches the existing thesis bibliography key `schiffels2014inferring`.

# %%
#include
demography_error_summary = ch3_plot.summarise_demography_overlap_errors(
    demography_comparisons,
)
ch3_plot.plot_ancestor_overlap_errors(
    demography_error_summary, config=dataclasses.replace(config, ancestor_error_label_decimals=4), figsize=(8.5, 4.8),
    figure_folder=figure_folder,
    group_by="model", group_labels=demography_labels, show_counts=False,
    name="demography_overlap_errors", show_burden_change=True,
)
export_thesis_figure("demography_overlap_errors.png")

# %% [markdown]
# #### Verification: error-containing comparisons and pooled burden
#
# The whole-overlap error counts and lengths are duplicated on left/right
# rows. Keep only the **left** row, once per model, version, replicate and
# focal site. Comparisons sharing an ancestor remain separate.
# `error-containing fraction = count(num_errors > 0) / number of comparisons`;
# `pooled burden = sum(num_errors) / sum(overlap_span)`.
# The increase in the fraction containing errors is reported in **percentage
# points**, separately from the relative percentage increase.

# %%
demography_whole_overlaps = demography_comparisons.loc[demography_comparisons.side == "left"].copy()
assert not demography_whole_overlaps.duplicated(demography_site_keys).any()
demography_whole_overlaps["contains_error"] = demography_whole_overlaps.num_errors > 0
demography_error_checks = demography_whole_overlaps.groupby(["model", "version"]).agg(
    num_comparisons=("true_focal_site", "size"),
    num_with_errors=("contains_error", "sum"),
    total_errors=("num_errors", "sum"),
    total_overlap_bp=("overlap_span", "sum"),
    true_0_observed_1_errors=("num_should_be_1", "sum"),
    true_1_observed_0_errors=("num_should_be_0", "sum"),
)
demography_error_checks = demography_error_checks.reindex(demography_group_order)
demography_error_checks["num_error_free"] = demography_error_checks.num_comparisons - demography_error_checks.num_with_errors
demography_error_checks["fraction_with_errors"] = demography_error_checks.num_with_errors / demography_error_checks.num_comparisons
demography_error_checks["pooled_burden_errors_per_bp"] = demography_error_checks.total_errors / demography_error_checks.total_overlap_bp
demography_plotted_error_summary = demography_error_summary.set_index(["model", "version"])
demography_plotted_error_summary = demography_plotted_error_summary.reindex(demography_group_order)
np.testing.assert_array_equal(demography_error_checks.num_comparisons, demography_plotted_error_summary.num_ancestors)
np.testing.assert_array_equal(demography_error_checks.num_error_free, demography_plotted_error_summary.num_error_free)
np.testing.assert_allclose(demography_error_checks.pooled_burden_errors_per_bp, demography_plotted_error_summary.error_burden)
ipd.display(demography_error_checks[[
    "num_comparisons", "num_error_free", "num_with_errors", "fraction_with_errors",
    "total_errors", "total_overlap_bp", "pooled_burden_errors_per_bp",
]].style.format({
    "num_comparisons": "{:,.0f}", "num_error_free": "{:,.0f}",
    "num_with_errors": "{:,.0f}", "fraction_with_errors": "{:.6f}",
    "total_errors": "{:,.0f}", "total_overlap_bp": "{:,.0f}",
    "pooled_burden_errors_per_bp": "{:.8e}",
}).set_caption("Numerators and denominators: count of error-containing comparisons, and ratio-of-sums burden"))

# %%
demography_error_fractions = demography_error_checks.fraction_with_errors.unstack("version")
demography_error_changes = pd.DataFrame(index=list(demography_labels))
demography_error_changes.index.name = "model"
demography_error_changes["with_errors_0_4_pct"] = 100 * demography_error_fractions[demography_old_version]
demography_error_changes["with_errors_0_5_pct"] = 100 * demography_error_fractions[demography_new_version]
demography_error_changes["increase_percentage_points"] = (
    demography_error_changes.with_errors_0_5_pct - demography_error_changes.with_errors_0_4_pct
)
demography_error_changes["relative_increase_pct"] = 100 * (
    demography_error_fractions[demography_new_version] / demography_error_fractions[demography_old_version] - 1
)
ipd.display(demography_error_changes.style.format("{:.5f}")
            .set_caption("Version comparison: absolute percentage-point change versus relative percentage increase"))

# %% [markdown]
# **Error-direction check.** The aggregate writer's names are reversed:
# `num_should_be_0` counts true 1 / observed 0, and `num_should_be_1` counts
# true 0 / observed 1 (see `lib/utils.py` in tsinfer-anc-eval). Match the plot's
# mapping explicitly below and verify that the two directions sum to all
# errors. Calculate direction shares from the counts; the apparent segment
# heights on a logarithmic stacked-bar axis do not represent these shares.

# %%
demography_direction_checks = demography_error_checks[[
    "true_0_observed_1_errors", "true_1_observed_0_errors", "total_errors",
]].copy()
np.testing.assert_array_equal(
    demography_direction_checks.true_0_observed_1_errors + demography_direction_checks.true_1_observed_0_errors,
    demography_direction_checks.total_errors,
)
demography_direction_checks["true_0_error_share_pct"] = 100 * demography_direction_checks.true_0_observed_1_errors / demography_direction_checks.total_errors
demography_direction_checks["true_1_error_share_pct"] = 100 * demography_direction_checks.true_1_observed_0_errors / demography_direction_checks.total_errors
demography_direction_checks["true_0_errors_per_bp"] = demography_direction_checks.true_0_observed_1_errors / demography_error_checks.total_overlap_bp
demography_direction_checks["true_1_errors_per_bp"] = demography_direction_checks.true_1_observed_0_errors / demography_error_checks.total_overlap_bp
assert demography_direction_checks.true_1_error_share_pct.gt(50).all()
demography_derived_burdens = demography_direction_checks.true_1_errors_per_bp.unstack("version")
for version in config.versions:
    african_burden = demography_derived_burdens.loc["Africa_1T12", version]
    other_burdens = demography_derived_burdens.drop(index="Africa_1T12")[version]
    assert other_burdens.gt(african_burden).all()
ipd.display(demography_direction_checks.style.format({
    "true_0_observed_1_errors": "{:,.0f}", "true_1_observed_0_errors": "{:,.0f}",
    "total_errors": "{:,.0f}", "true_0_error_share_pct": "{:.3f}",
    "true_1_error_share_pct": "{:.3f}", "true_0_errors_per_bp": "{:.8e}",
    "true_1_errors_per_bp": "{:.8e}",
}).set_caption("Direction counts, shares and burdens; true 1 errors are the majority in every model/version"))

# %% [markdown]
# ### Direction of ancestral-state errors in tsinfer 0.5
#
# Split the error-free tsinfer 0.5 focal-site comparisons into 20 fixed-width
# DAF bins, each covering 5 percentage points. Plot rates at bin midpoints,
# pooling each error type over the bin's total overlapping bp. Include DAF 1
# in the final bin; empty bins have undefined rates.
# The aggregate writer's names are reversed: `num_should_be_0` counts true 1
# inferred as 0, and `num_should_be_1` counts true 0 inferred as 1. Legend labels
# describe the true and observed states of each error direction.
# Colours match the genotype-transition plot; zero burdens are gaps on the log
# scale. This exploratory figure is saved in dphil-analysis/figures/ch3.

# %%
# don't include
ancestor_error_types = ch3_plot.summarise_ancestor_error_types(ancestor_overlap_errors, config=config)
ch3_plot.plot_ancestor_error_types(
    ancestor_error_types,
    figure_folder=figure_folder,
    config=config,
    figsize=(6, 3.5),
)

# %% [markdown]
# ## Genotype errors
#
# Compare `call_genotype` in each pure genotype-error profile with the error-free
# store during HPC extraction. A diploid error changes genotype dosage (00, 01/10,
# or 11). Exported counts pool changed calls and denominators across replicates.
# The dashed curve scales the observed 1× rate linearly; both axes are logged.

# %%
#include
genotype_summary = ch3_plot.summarise_genotype_errors(data)
ch3_plot.plot_genotype_error_scaling(
    genotype_summary,
    figure_folder=figure_folder,
    config=config,
    figsize=(6, 3.5),
)
export_thesis_figure("genotype_error_scaling.png")

# %% [markdown]
# Transition rates divide 00 → 01/10 and 11 → 01/10 counts by the number of
# original 00 and 11 calls, respectively, in each true DAF bin. Empty denominator
# bins are gaps in the curves. The panels share a logarithmic vertical scale.

# %%
#include
ch3_plot.plot_genotype_error_transitions(
    genotype_summary,
    figure_folder=figure_folder,
    config=config,
    figsize=(7, 3.5),
)
export_thesis_figure("genotype_error_transitions.png")

# %% [markdown]
# Confirmation histogram layout: tsinfer 0.4 above, 0.5 below, mirrored
# logarithmic counts and common physical overshoot bins. Negative overshoot
# means stopping within the true ancestor. Colours run from 1× to 100×.
# Each histogram is followed by a separate grouped boxplot. White boxes with
# black borders and whiskers overlay sampled central jitter and all tail points.
# Every point has the same opacity, coloured by version. Dashed caps show the
# conf and 1-conf quantiles; the solid black line marks zero. Each comparison
# also has two tail panels with separate x ranges for negative and positive
# overshoot magnitudes. Each focal site contributes its larger negative and
# positive side magnitudes, floored at zero. Fractions count focal sites strictly
# above the magnitude cutoff by default. Set fraction_above=False for fractions
# at or below the cutoff. All focal sites, including those with zero overshoots,
# supply the denominator.

# %% [markdown]
# Bars show mean signed overshoot by genotype error, coloured by version,
# after averaging each focal site's two sides. The dashed black line marks zero.

# %%
#include: right of plot
ch3_plot.plot_genotype_overshoot_mean(
    data,
    figure_folder=figure_folder,
    figsize=(5, 4.5),
)
export_thesis_figure("genotype_error_overshoot_mean.png")

# %%
#include: left of plot
# Smaller legends keep the narrow histogram panels visible.
ch3_plot.plot_genotype_overshoot_hist(
    data,
    figure_folder=figure_folder,
    figsize=(5, 4.5),
    legend_fontsize=7.5,
)
export_thesis_figure("genotype_error_overshoot_hist.png")

# %%
ch3_plot.plot_genotype_overshoot_box(
    data,
    figure_folder=figure_folder,
    conf=0.001,
    figsize=(8.5, 3.5),
    y_log=False,
)

# %%
ch3_plot.plot_genotype_overshoot_tails(
    data,
    figure_folder=figure_folder,
    figsize=(8.5, 4.5),
    fraction_above=True,
)

# %% [markdown]
# ## Mispolarisation errors
#
# DAF is flipped at sites whose `variant_mispolarisation_mask` is False. Curves
# show absolute site counts in 200 equal-width bins, smoothed by a Gaussian
# kernel with a one-bin standard deviation. This keeps regeneration practical
# for larger datasets. Counts are pooled across replicates.
# Singletons are included in the simulation DAF plot, although the simulation
# error generator does not mispolarise them. Retain them in the TGP spectrum too.
# Both panels use the same bins, smoothing, x limits and logarithmic count scale.
# Counts are absolute: the simulation pools replicates and has 12,000 haplotypes;
# TGP has 6,404 haplotypes from all 3,202 individuals.

# %%
#include: left of plot
ch3_plot.plot_mispolarisation_daf(
    data,
    figure_folder=figure_folder,
    figsize=config.daf_size,
)
export_thesis_figure("mispolarisation_daf.png")

# %% [markdown]
# ### TGP: first 20 Mbp of chromosome 20
#
# Apply `variant_duplicate_position_mask`, `variant_not_biallelic_mask` and
# `variant_not_snps_mask`. Also exclude `variant_bad_ancestral_mask` and
# `variant_no_ancestral_allele_mask`, since DAF needs a valid ancestral allele
# present at the site. No low-allele-count, density or low-confidence ancestral
# allele mask is applied. Stored all-sample derived counts are already polarised;
# divide by ancestral plus derived counts to use called alleles as the denominator.
# This is the recorded TGP spectrum, with no additional simulated mispolarisation.

# %%
tgp_variants = ch3_plot.load_tgp_daf(tgp_zarr_path, config=config)
print(f"Retained TGP SNPs: {len(tgp_variants):,}")
print(f"Retained derived singletons: {(tgp_variants.derived_count == 1).sum():,}")

# %%
#include: right of plot
ch3_plot.plot_tgp_daf(
    tgp_variants,
    figure_folder=figure_folder,
    config=config,
    figsize=config.daf_size,
)
export_thesis_figure("tgp_daf.png")

# %% [markdown]
# Overshoot plots use true focal DAF from the error-free variant store, with
# 0.5 included only in the high-frequency figure. The all-frequency plots pool
# high and low DAFs. Pure mispolarisation profiles include the error-free
# reference; mixed error profiles are excluded. The low-count histogram includes
# only mispolarised focal sites with true derived count ≤ 5.

# %%
ch3_plot.plot_mispolarisation_overshoot_hist(
    data,
    figure_folder=figure_folder,
    frequency="high",
    figsize=(7, 4.5),
)

# %%
ch3_plot.plot_mispolarisation_overshoot_box(
    data,
    figure_folder=figure_folder,
    frequency="high",
    conf=0.001,
    figsize=(8.5, 3.5),
    y_log=False,
)

# %%
ch3_plot.plot_mispolarisation_overshoot_tails(
    data,
    figure_folder=figure_folder,
    frequency="high",
    figsize=(8.5, 4.5),
    show_counts=True,
)

# %%
ch3_plot.plot_mispolarisation_overshoot_hist(
    data,
    figure_folder=figure_folder,
    frequency="low",
    figsize=(7, 4.5),
)

# %%
ch3_plot.plot_mispolarisation_overshoot_box(
    data,
    figure_folder=figure_folder,
    frequency="low",
    conf=0.001,
    figsize=(8.5, 3.5),
    y_log=False,
)

# %%
ch3_plot.plot_mispolarisation_overshoot_tails(
    data,
    figure_folder=figure_folder,
    frequency="low",
    figsize=(8.5, 4.5),
    show_counts=True,
)

# %%
ch3_plot.plot_mispolarisation_overshoot_hist(
    data,
    figure_folder=figure_folder,
    frequency="all",
    figsize=(7, 4.5),
)

# %%
ch3_plot.plot_mispolarisation_overshoot_box(
    data,
    figure_folder=figure_folder,
    frequency="all",
    conf=0.001,
    figsize=(8.5, 3.5),
    y_log=False,
)

# %% [markdown]
# ### Mispolarisation overshoot across all frequencies
#
# Combine variants without splitting by true DAF. Keep the same
# negative/positive tail curves and focal-site count
# bars, pooling the underlying observations across both frequency groups.
# Export this combined figure to the thesis.

# %%
#include
ch3_plot.plot_mispolarisation_overshoot_tails(
    data,
    figure_folder=figure_folder,
    frequency="all",
    figsize=(8.5, 4.5),
    show_counts=True,
)
export_thesis_figure("mispolarisation_overshoot_all_frequency_tails.png")

# %%
#include
# Top rate legend and version label mirror the bottom panel.
ch3_plot.plot_mispolarisation_low_count_overshoot_hist(
    data,
    figure_folder=figure_folder,
    figsize=(7, 4.5),
)
export_thesis_figure("mispolarisation_overshoot_derived_count_le_5_hist.png")

# %% [markdown]
# ## Phase-switch errors
#
# Aligned chromosome pairs compare correct and erroneous phasing for the first
# five individuals on chromosome 1's p arm (0–123.4 Mb, SER = 3e-4).
# Colours track true haplotype identity using `call_genotype_phase`, including
# homozygous sites. The two overview columns share a haplotype legend on the right.
# Here SER applies to adjacent variant rows that are both heterozygous for the
# individual, as defined by the simulation code.

# %%
#include
ch3_plot.plot_phase_switch_illustration(
    phasing_folder, phasing_prefix,
    figure_folder=figure_folder,
    config=config,
    figsize=(8, 2.8),
)
export_thesis_figure("phase_switch_illustration.png")

# %% [markdown]
# Switch error rates are read from the exported profiles. The histograms use
# the same mirrored version layout as the genotype-error histogram, followed
# by separate boxplots and negative/positive tail fractions.

# %%
#include left
ch3_plot.plot_phasing_overshoot_hist(
    data,
    figure_folder=figure_folder,
    figsize=(5, 4.5),
    legend_fontsize=7.5,
)
export_thesis_figure("phasing_error_overshoot_hist.png")

# %%
#include right
ch3_plot.plot_phasing_overshoot_mean(
    data,
    figure_folder=figure_folder,
    figsize=(5, 4.5),
)
export_thesis_figure("phasing_error_overshoot_mean.png")

# %%
(0.0003+0.006)/2

# %%
# don't include anymore
ch3_plot.plot_phasing_overshoot_box(
    data,
    figure_folder=figure_folder,
    conf=0.001,
    figsize=(8.5, 3.5),
    y_log=False,
)

# %%
ch3_plot.plot_phasing_overshoot_tails(
    data,
    figure_folder=figure_folder,
    figsize=(8.5, 4.5),
)

# %%

# %%

# %%

# %% [markdown]
# ## Inference performance: tsinfer 0.4 versus 0.5
#
# These cells run independently of the simulation and Zarr analyses above.
# The aggregate contains one run per sample size, version and stage, all with
# 196 configured threads on the same host, over a 28.1 Mb region. No replicate
# uncertainty is estimated. Version labels follow the benchmark CSV: the run
# labelled 0.5.1 reports provenance version `0.5.2.dev1` (0.4.1 reports 0.4.1).
# Speedup is `elapsed_0.4 / elapsed_0.5`: above 1 means faster in 0.5.
# The elapsed-time stacks sum the three stages; each sample-size panel has its
# own linear y scale in minutes so smaller runs remain visible.
# Parentheses above the tsinfer 0.5 bars show speedup for the summed elapsed time.
# Add 1KGP chr20p as a fifth group (6,404 haplotypes from 3,202 individuals).
# The 1KGP comparison uses the threaded API and 504,149 filtered sites.
# Alternative 0.5 runs are used for both datasets; original 0.4 rows are retained.
# Both plots retain the original figure dimensions; the extra group narrows bars.

# %%
import pathlib

import ch3_anc_perf
import IPython.display as ipdisplay
import matplotlib.pyplot as plt
import pandas as pd

perf_config = ch3_anc_perf.PlotConfig(
    sample_labels={6404: "6,404 (1KGP)"},
    sample_titles={6404: "6,404 haplotypes\n(1KGP)"},
)
perf_folder = pathlib.Path("~/work/dphil-analysis/data/ch3/results").expanduser()
perf_sources = [perf_folder / "perf_0.4.1.csv", perf_folder / "perf_0.5_alt.csv"]
perf_frames = [pd.read_csv(path, comment="~") for path in perf_sources]
perf_combined_alt = pd.concat(perf_frames, ignore_index=True)
perf_combined_alt_path = perf_folder / "perf_combined_alt.csv"
perf_combined_alt.to_csv(perf_combined_alt_path, index=False)
original_tgp_perf = pd.read_csv(perf_folder.parent / "tgp_perf.csv")
alternative_tgp_perf = pd.read_csv(perf_folder.parent / "tgp_perf_0.5_alt.csv")
retained_tgp_perf = original_tgp_perf.loc[original_tgp_perf.version == perf_config.versions[0]]
tgp_perf_alt = pd.concat([retained_tgp_perf, alternative_tgp_perf], ignore_index=True)
tgp_perf_alt_path = perf_folder.parent / "tgp_perf_alt.csv"
tgp_perf_alt.to_csv(tgp_perf_alt_path, index=False)
perf_data = ch3_anc_perf.load_performance(perf_combined_alt_path, perf_config)
tgp_perf_data = ch3_anc_perf.load_performance(tgp_perf_alt_path, perf_config)
perf_comparison_data = pd.concat([perf_data, tgp_perf_data]).sort_index()

# %%
#include

perf_speedup_figure = ch3_anc_perf.plot_speedup(perf_comparison_data, perf_config)
perf_speedup_figure.savefig(
    figure_folder / "inference_performance_speedup.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.show()
export_thesis_figure("inference_performance_speedup.png")

# %%
#include
perf_elapsed_figure = ch3_anc_perf.plot_elapsed(perf_comparison_data, perf_config)
perf_elapsed_figure.savefig(
    figure_folder / "inference_performance_elapsed.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.show()
export_thesis_figure("inference_performance_elapsed.png")

# %% [markdown]
# ### Memory usage
#
# Peak resident memory from each stage's provenance, converted from bytes to
# GiB. Memory is not additive across stages. Each stage runs in a fresh process;
# its high-water mark includes setup and input loading. Log-log axes show
# scaling across the four sample sizes, with shared limits for all stages.

# %%
perf_memory_figure = ch3_anc_perf.plot_resources(perf_data, "memory_gib", perf_config)
perf_memory_figure.savefig(
    figure_folder / "inference_performance_memory.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.show()

# %% [markdown]
# ### Parallelism
#
# Average active CPU cores = `(user_time + sys_time) / elapsed_time` for each
# stage, including setup and I/O inside the inference API call. Top-level
# input loading, process startup and output writing are outside that timer.
# This measures achieved CPU concurrency, not
# speedup from threading: there is no single-threaded baseline here. Shared
# linear y axes allow comparison between stages; the dotted line marks the
# 196 configured threads.
# Triangles show 1KGP chr20p (6,404 haplotypes), with colours identifying
# tsinfer versions in both datasets. Both datasets use the threaded API.

# %%
#include
perf_parallelism_figure = ch3_anc_perf.plot_resources(
    perf_data, "parallelism", perf_config, reference_data=tgp_perf_data,
)
perf_parallelism_figure.savefig(
    figure_folder / "inference_performance_parallelism.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.show()
export_thesis_figure("inference_performance_parallelism.png")

# %% [markdown]
# ### Verification: performance numbers used in the text
#
# Ratios compare unchanged 0.4.1 runs with alternative 0.5 runs. CPU time
# includes user and system time; total elapsed time sums all three stages.
# These are single runs without uncertainty estimates.
# The old scheduler replay caches describe the original 0.5 ancestor files.
# New ancestor files and logs are needed before repeating that analysis.
# Their group statistics must not be combined with the alternative timings.

# %%
perf_text_metrics = perf_comparison_data.copy()
perf_text_metrics["cpu_seconds"] = perf_text_metrics.user_time + perf_text_metrics.sys_time
perf_text_checks = perf_text_metrics[[
    "elapsed_time", "elapsed_minutes", "cpu_seconds", "parallelism", "memory_gib",
    "num_sites", "num_edges",
]].unstack("version")
for metric, output_name in [
    ("elapsed_time", "elapsed_speedup"), ("cpu_seconds", "cpu_time_reduction"),
]:
    old_values = perf_text_checks[(metric, perf_config.versions[0])]
    new_values = perf_text_checks[(metric, perf_config.versions[1])]
    perf_text_checks[(output_name, "0.4 / 0.5")] = old_values / new_values
ipdisplay.display(perf_text_checks.style.format("{:.4f}"))
perf_total_seconds = perf_comparison_data.elapsed_time.groupby(["num_samples", "version"]).sum()
perf_totals = perf_total_seconds.unstack("version")
perf_total_checks = perf_totals / perf_config.seconds_per_minute
perf_total_checks.columns = [f"{version}_minutes" for version in perf_total_checks.columns]
perf_total_checks["total_speedup"] = perf_totals[perf_config.versions[0]] / perf_totals[perf_config.versions[1]]
ipdisplay.display(perf_total_checks.style.format("{:.4f}"))
perf_site_checks = perf_comparison_data.num_sites.unstack("version")
assert perf_site_checks[perf_config.versions[0]].equals(perf_site_checks[perf_config.versions[1]])

# %% [markdown]
# ### Verification: frequency patterns and remaining numerical placeholders
#
# Count focal sites separately, but ancestor IDs once within each replicate.
# Rare-tail cutoffs pool both focal-site sides, matching the overshoot plot.
# Overall error fractions count each focal-site overlap once; error burden
# divides summed errors by summed overlapping bp, without averaging bin rates.

# %%
text_check_settings = {"rare_frequency": 0.001, "tail_probability": 0.001, "percent": 100}
error_free_spans = ch3_plot._no_errors(data.spans)
error_free_overshoots = ch3_plot._no_errors(data.overshoots)
focal_count_rows = []
for version, rows in error_free_spans.groupby("version"):
    focal_count_rows.append({
        "version": version,
        "focal_sites": len(rows),
        "true_ancestors": len(rows.drop_duplicates(["replicate", "true_node"])),
        "inferred_ancestors": len(rows.drop_duplicates(["replicate", "inferred_node"])),
        "replicates": rows.replicate.nunique(),
    })
focal_count_checks = pd.DataFrame(focal_count_rows).set_index("version")
rare_overshoots = error_free_overshoots.loc[
    error_free_overshoots.allele_frequency < text_check_settings["rare_frequency"]
]
rare_tail_checks = rare_overshoots.groupby("version").overshoot.quantile(
    text_check_settings["tail_probability"],
).to_frame("lower_0_1_pct_bp")
rare_tail_checks["lower_0_1_pct_mb"] = rare_tail_checks.lower_0_1_pct_bp / config.bp_per_mb
whole_overlap_checks = ancestor_overlap_errors.groupby("version").agg(
    comparisons=("num_errors", "size"),
    with_errors=("num_errors", lambda values: values.gt(0).sum()),
    total_errors=("num_errors", "sum"),
    overlap_bp=("overlap_span", "sum"),
)
whole_overlap_checks["with_errors_pct"] = (
    text_check_settings["percent"] * whole_overlap_checks.with_errors / whole_overlap_checks.comparisons
)
whole_overlap_checks["pooled_error_burden_pct"] = (
    text_check_settings["percent"] * whole_overlap_checks.total_errors / whole_overlap_checks.overlap_bp
)
span_pattern_checks = ancestor_span_curve_summary.pivot(
    index=["dataset", "bin", "bin_left", "bin_right"], columns="version", values="mean_span_mb",
)
span_pattern_checks["new_minus_old_mb"] = (
    span_pattern_checks[config.versions[1]] - span_pattern_checks[config.versions[0]]
)
for table in [focal_count_checks, rare_tail_checks, whole_overlap_checks, span_pattern_checks]:
    ipdisplay.display(table)

# %% [markdown]
# ### Export numerical checks for the chapter review
#
# Regenerate the observations and units used in the figures with the notebook.

# %%
text_check_folder = data_folder / "text_checks_alt"
text_check_folder.mkdir(exist_ok=True)
text_check_tables = {
    "focal_counts": focal_count_checks,
    "rare_overshoot_tail": rare_tail_checks,
    "whole_overlap_errors": whole_overlap_checks,
    "span_patterns": span_pattern_checks,
    "ancestor_span_curves": ancestor_span_curve_summary,
    "tgp_inference_site_extent": tgp_inference_site_extent,
    "demography_quartiles": demography_iqr_table,
    "demography_tails": demography_tail_table,
    "demography_tail_changes": demography_tail_changes,
    "demography_mean_lower_change": demography_lower_change_working,
    "demography_error_changes": demography_error_changes,
    "demography_error_checks": demography_error_checks,
    "demography_error_directions": demography_direction_checks,
    "ancestor_overlap_errors": ancestor_error_summary,
    "genotype_errors": genotype_summary.rates,
    "genotype_transitions": genotype_summary.transitions,
    "performance_stages": perf_text_checks,
    "performance_totals": perf_total_checks,
}
for name, table in text_check_tables.items():
    table.to_csv(text_check_folder / f"{name}.csv")

# %% [markdown]
# ## Testing error buffering
#
# Compare the standard algorithms with the alternative tsinfer 0.5 algorithm,
# which skips sites with no derived carriers in the current sample set.
# Bright red denotes tsinfer 0.5 (alt) throughout. These comparisons retain
# original 0.5 results and save to dphil-analysis/figures/ch3/error_buffering.

# %%
buffering_figure_folder = figure_folder / "error_buffering"
buffering_figure_folder.mkdir(exist_ok=True)
buffering_settings = {
    "alt_version": "0.5 (alt)",
    "alt_color": "#ff0000",
    "tail_quantile": 0.01,
    "box_size": (9, 4.5),
    "box_multipliers": (0, 1, 10),
    "frequency_cutoffs": (0, 0.001, 0.003, 0.02, 0.5, 1),
    "box_y_log": False,
    "error_size": (9, 5.4),
}
buffering_config = dataclasses.replace(
    config,
    versions=(*config.versions, buffering_settings["alt_version"]),
    version_colors=(*config.version_colors, buffering_settings["alt_color"]),
)
buffering_multipliers = (0, *config.genotype_multipliers)
buffering_inputs = {
    "standard": standard_aggregated_path,
    "alternative": data_folder / f"{data_prefix}-aggregated_alt_version.csv",
}
buffering_overshoot_frames = []
buffering_span_frames = []
buffering_summary_frames = []
for algorithm, path in buffering_inputs.items():
    query = pl.scan_csv(path)
    query = query.filter(
        (pl.col("phase_ser") == 0)
        & (pl.col("mispol_rate") == 0)
        & pl.col("geno_multiplier").is_in(buffering_multipliers)
        & pl.col("version").is_in(config.versions)
    )
    if algorithm == "alternative":
        query = query.filter(pl.col("version") == config.versions[1])
        query = query.with_columns(pl.lit(buffering_settings["alt_version"]).alias("version"))
    # Whole-overlap errors occur on both side rows: count only the left copy.
    left = pl.col("side") == "left"
    grouped = query.group_by("version", "geno_multiplier").agg(
        left.sum().alias("num_ancestors"),
        (left & (pl.col("num_errors") == 0)).sum().alias("num_error_free"),
        pl.col("num_errors").filter(left).sum().alias("num_errors"),
        pl.col("num_should_be_1").filter(left).sum().alias("should_be_0_errors"),
        pl.col("num_should_be_0").filter(left).sum().alias("should_be_1_errors"),
        pl.col("overlap_span").filter(left).sum().alias("overlap_bp"),
    )
    table = grouped.collect(engine="streaming")
    buffering_summary_frames.append(pd.DataFrame(table.to_dict(as_series=False)))
    overshoots = query.filter(
        pl.col("geno_multiplier").is_in(buffering_settings["box_multipliers"])
    )
    overshoots = overshoots.select(
        "replicate", "version", "geno_multiplier", "true_focal_site", "side",
        "allele_frequency", "overshoot",
    )
    table = overshoots.collect(engine="streaming")
    buffering_overshoot_frames.append(pd.DataFrame(table.to_dict(as_series=False)))
    spans = query.filter(left & (pl.col("geno_multiplier") == 0))
    spans = spans.select(
        "replicate", "version", "geno_multiplier", "phase_ser", "mispol_rate",
        "true_focal_site", "allele_frequency", "inferred_span", "true_span",
    )
    table = spans.collect(engine="streaming")
    buffering_span_frames.append(pd.DataFrame(table.to_dict(as_series=False)))
buffering_summary = pd.concat(buffering_summary_frames, ignore_index=True)
buffering_overshoots = pd.concat(buffering_overshoot_frames, ignore_index=True)
buffering_spans = pd.concat(buffering_span_frames, ignore_index=True)
buffering_counts = buffering_summary.pivot(
    index="geno_multiplier", columns="version", values="num_ancestors",
)
assert set(buffering_counts.index) == set(buffering_multipliers)
assert set(buffering_counts.columns) == set(buffering_config.versions)
assert buffering_counts.eq(buffering_counts[config.versions[0]], axis="index").all().all()

# %% [markdown]
# ### Ancestor spans
#
# Use the error-free simulation profiles and the same frequency bins, truth
# curve and count panel as `plot_ancestor_span_curves` above.

# %%
buffering_data = dataclasses.replace(data, spans=buffering_spans, config=buffering_config)
buffering_span_summary = ch3_plot.summarise_ancestor_span_curves(
    buffering_data, tgp_ancestor_spans,
)
ch3_plot.plot_ancestor_span_curves(
    buffering_span_summary, "Simulation", config=buffering_config,
    title="Out-of-Africa simulation (20mbp)", figure_folder=buffering_figure_folder, y_log=True,
)


# %% [markdown]
# ### Overshoots by focal allele frequency
#
# Use the usual frequency cutoffs, boxplots, coloured jitter, tail quantile
# caps and focal-site count panel. Each plot selects one genotype error
# multiplier, with zero switch error and mispolarisation. The alternative
# implementation adds a third group in bright red within every frequency bin.

# %%
def plot_buffering_frequency_overshoots(multiplier):
    """Draw the usual overshoot layout for one selected genotype-error profile."""
    frame = buffering_overshoots.loc[
        buffering_overshoots.geno_multiplier == multiplier
    ].copy()
    edges = np.array(buffering_settings["frequency_cutoffs"], dtype=float)
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(edges[-1], np.inf)
    frame["frequency_bin"] = pd.cut(
        frame.allele_frequency, cut_edges, labels=False, right=False,
    )
    num_bins = len(edges) - 1
    counts = frame.loc[frame.side == "left"].groupby(
        ["frequency_bin", "version"], observed=True,
    ).size().unstack("version")
    counts = counts.reindex(range(num_bins), fill_value=0)
    reference_counts = counts[config.versions[0]]
    assert counts.eq(reference_counts, axis="index").all().all()
    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=buffering_settings["box_size"],
        layout="constrained", gridspec_kw={"height_ratios": [4, 1]},
    )
    rng = np.random.default_rng(config.jitter_seed)
    width = config.mean_bar_group_width / len(buffering_config.versions)
    for index, (version, color) in enumerate(zip(
        buffering_config.versions, buffering_config.version_colors,
    )):
        version_rows = frame.loc[frame.version == version]
        for bin_index in range(num_bins):
            rows = version_rows.loc[version_rows.frequency_bin == bin_index]
            if rows.empty:
                continue
            values = rows.overshoot.to_numpy() / config.bp_per_mb
            position = bin_index - config.mean_bar_group_width / 2
            position = position + width * (index + 0.5)
            ch3_plot._draw_overshoot_group(
                axes[0], values, position, width * 0.85, color, rng,
                buffering_config, buffering_settings["tail_quantile"], config.jitter_alpha,
            )
    ch3_plot._overshoot_legends(
        fig, buffering_config, buffering_settings["tail_quantile"], config.jitter_alpha,
    )
    if buffering_settings["box_y_log"]:
        ch3_plot._symlog_overshoot_axis(axes[0], buffering_config)
    ch3_plot._overshoot_zero_line(axes[0])
    axes[0].set_ylabel("Overshoot (Mb)")
    axes[0].set_title(f"Genotype error multiplier: {multiplier:g}×")
    axes[1].bar(
        np.arange(num_bins), reference_counts, width=width * 0.85 * 2,
        facecolor="white", edgecolor="black",
    )
    axes[1].set_ylabel("Count")
    axes[1].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))
    axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    labels = [f"{left:g}–{right:g}" for left, right in zip(edges[:-1], edges[1:])]
    axes[1].set_xticks(np.arange(num_bins), labels, rotation=25)
    axes[1].set_xlabel("Observed focal allele frequency")
    axes[0].set_xlim(-0.6, num_bins - 0.4)
    fig.savefig(
        buffering_figure_folder / f"overshoot_frequency_genotype_{multiplier:g}x.png",
        dpi=config.dpi, bbox_inches="tight", facecolor="white",
    )
    plt.show()
    plt.close(fig)

# %%
plot_buffering_frequency_overshoots(0)

# %%
plot_buffering_frequency_overshoots(1)

# %%
plot_buffering_frequency_overshoots(10)

# %% [markdown]
# ### Ancestor errors by genotype error multiplier
#
# Pool all focal frequencies within each multiplier. The top row shows the
# fraction of whole-overlap comparisons with no errors; the bottom row shows
# total errors divided by total overlapping bp, stacked by error direction.
# Each focal site contributes once, retaining the usual two-row format.

# %%
buffering_positions = np.arange(len(buffering_multipliers))
buffering_bar_width = config.mean_bar_group_width / len(buffering_config.versions)
buffering_group_labels = {rate: f"{rate:g}" for rate in buffering_multipliers}
buffering_error_summary = buffering_summary.copy()
buffering_error_summary["error_free_fraction"] = (
    buffering_error_summary.num_error_free / buffering_error_summary.num_ancestors
)
buffering_overlap_bp = buffering_error_summary.overlap_bp.replace(0, np.nan)
buffering_error_summary["error_burden"] = (
    buffering_error_summary.num_errors / buffering_overlap_bp
)
buffering_error_summary["should_be_0_burden"] = (
    buffering_error_summary.should_be_0_errors / buffering_overlap_bp
)
buffering_error_summary["should_be_1_burden"] = (
    buffering_error_summary.should_be_1_errors / buffering_overlap_bp
)
buffering_error_figure, buffering_error_axes = plt.subplots(
    2, 1, sharex=True, figsize=buffering_settings["error_size"], layout="constrained",
)
for index, (version, color) in enumerate(zip(
    buffering_config.versions, buffering_config.version_colors,
)):
    rows = buffering_error_summary.loc[buffering_error_summary.version == version]
    rows = rows.set_index("geno_multiplier").reindex(buffering_multipliers)
    offsets = buffering_positions - config.mean_bar_group_width / 2
    offsets = offsets + buffering_bar_width * (index + 0.5)
    label = f"tsinfer {'.'.join(version.split('.')[:2])}"
    bars = buffering_error_axes[0].bar(
        offsets, rows.error_free_fraction, width=buffering_bar_width,
        color=color, label=label,
    )
    buffering_error_axes[0].bar_label(
        bars, fmt=f"%.{config.ancestor_error_label_decimals}f", color=color,
        fontsize=config.ancestor_error_label_fontsize,
        padding=config.ancestor_error_label_padding,
    )
    buffering_error_axes[1].bar(
        offsets, rows.should_be_1_burden, width=buffering_bar_width,
        facecolor=color, edgecolor=color,
    )
    buffering_error_axes[1].bar(
        offsets, rows.should_be_0_burden, bottom=rows.should_be_1_burden,
        width=buffering_bar_width, facecolor="white", edgecolor=color,
        hatch=config.ancestor_error_hatch,
    )
buffering_error_axes[0].set_ylim(0, 1)
buffering_error_axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
buffering_error_axes[0].set_ylabel("Proportion of\nerror-free ancestors")
buffering_error_axes[1].set_ylabel("Pooled error burden\n(errors/bp)")
buffering_error_axes[1].set_yscale("log")
buffering_error_axes[1].set_xticks(buffering_positions, buffering_group_labels.values())
buffering_error_axes[1].set_xlabel("Genotype error multiplier")
buffering_method_handles, buffering_method_labels = buffering_error_axes[0].get_legend_handles_labels()
buffering_error_figure.legend(
    buffering_method_handles, buffering_method_labels,
    loc="outside upper center", ncol=len(buffering_config.versions), frameon=False,
)
buffering_direction_handles = [
    mpatches.Patch(facecolor="white", edgecolor="black", hatch=config.ancestor_error_hatch),
    mpatches.Patch(facecolor="black", edgecolor="black"),
]
buffering_error_axes[1].legend(
    buffering_direction_handles, config.ancestor_error_type_labels,
    frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2,
    fontsize=config.ancestor_error_legend_fontsize,
)
buffering_error_figure.savefig(
    buffering_figure_folder / "ancestor_errors_by_genotype_multiplier.png",
    dpi=config.dpi, bbox_inches="tight", facecolor="white",
)
plt.show()
plt.close(buffering_error_figure)

# %%

# %% [markdown]
# ### Negative overshoot proportions by focal allele frequency
#
# Each focal-site ancestor comparison contributes once, as in the boxplots
# above. A comparison has negative overshoot if either side is negative.
# Its negative magnitude is the larger underextension of its two sides,
# expressed as a positive length. Pool focal sites across replicates within
# the same frequency bins, retaining separate panels for 0×, 1× and 10×
# genotype errors. Switch error and mispolarisation remain zero.

# %%
buffering_line_settings = {
    "figsize": (12, 3.8),
    "marker": "o",
    "line_width": 1.8,
    "tick_rotation": 35,
}
buffering_focal_keys = ["replicate", "version", "geno_multiplier", "true_focal_site"]
buffering_negative_focals = buffering_overshoots.groupby(buffering_focal_keys).agg(
    allele_frequency=("allele_frequency", "first"),
    minimum_overshoot_bp=("overshoot", "min"),
    num_sides=("side", "size"),
).reset_index()
assert buffering_negative_focals.num_sides.eq(2).all()
buffering_negative_focals["has_negative_overshoot"] = (
    buffering_negative_focals.minimum_overshoot_bp < 0
)
buffering_negative_focals["negative_magnitude_mb"] = (
    -buffering_negative_focals.minimum_overshoot_bp.clip(upper=0) / config.bp_per_mb
)
buffering_negative_focals["conditional_negative_magnitude_mb"] = (
    buffering_negative_focals.negative_magnitude_mb.where(
        buffering_negative_focals.has_negative_overshoot
    )
)
buffering_line_edges = np.array(buffering_settings["frequency_cutoffs"], dtype=float)
buffering_line_cut_edges = buffering_line_edges.copy()
buffering_line_cut_edges[-1] = np.nextafter(buffering_line_edges[-1], np.inf)
buffering_negative_focals["frequency_bin"] = pd.cut(
    buffering_negative_focals.allele_frequency, buffering_line_cut_edges,
    labels=False, right=False,
)
buffering_negative_summary = buffering_negative_focals.groupby(
    ["version", "geno_multiplier", "frequency_bin"], observed=True,
).agg(
    num_focal_sites=("has_negative_overshoot", "size"),
    negative_proportion=("has_negative_overshoot", "mean"),
    mean_negative_magnitude_mb=("conditional_negative_magnitude_mb", "mean"),
    mean_magnitude_all_focals_mb=("negative_magnitude_mb", "mean"),
).reset_index()
# A populated bin with no negative overshoots has zero conditional magnitude.
buffering_negative_summary["mean_negative_magnitude_mb"] = (
    buffering_negative_summary.mean_negative_magnitude_mb.fillna(0)
)
buffering_negative_summary["weighted_negative_overshoot_score_mb"] = (
    buffering_negative_summary.negative_proportion
    * buffering_negative_summary.mean_negative_magnitude_mb
)
np.testing.assert_allclose(
    buffering_negative_summary.weighted_negative_overshoot_score_mb,
    buffering_negative_summary.mean_magnitude_all_focals_mb,
)
buffering_line_labels = [
    f"{left:g}–{right:g}"
    for left, right in zip(buffering_line_edges[:-1], buffering_line_edges[1:])
]


def plot_buffering_negative_lines(metric, ylabel, proportion=False):
    """Compare methods across frequency bins in three genotype-error panels."""
    fig, axes = plt.subplots(
        1, len(buffering_settings["box_multipliers"]), sharex=True, sharey=True,
        figsize=buffering_line_settings["figsize"], layout="constrained",
    )
    bins = np.arange(len(buffering_line_labels))
    for ax, multiplier in zip(axes, buffering_settings["box_multipliers"]):
        profile = buffering_negative_summary.loc[
            buffering_negative_summary.geno_multiplier == multiplier
        ]
        for version, color in zip(buffering_config.versions, buffering_config.version_colors):
            rows = profile.loc[profile.version == version]
            rows = rows.set_index("frequency_bin").reindex(bins)
            label = f"tsinfer {'.'.join(version.split('.')[:2])}"
            ax.plot(
                bins, rows[metric], color=color, label=label,
                marker=buffering_line_settings["marker"],
                linewidth=buffering_line_settings["line_width"],
            )
        ax.set_title(f"{multiplier:g}× genotype errors")
        ax.set_xticks(
            bins, buffering_line_labels, rotation=buffering_line_settings["tick_rotation"],
            ha="right",
        )
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel(ylabel)
    if proportion:
        axes[0].set_ylim(0, 1)
        axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    else:
        axes[0].set_ylim(bottom=0)
    fig.supxlabel("Observed focal allele frequency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(
        buffering_figure_folder / f"{metric}.png",
        dpi=config.dpi, bbox_inches="tight", facecolor="white",
    )
    plt.show()
    plt.close(fig)


plot_buffering_negative_lines(
    "negative_proportion", "Proportion with negative overshoot", proportion=True,
)

# %% [markdown]
# ### Weighted negative overshoot score
#
# Multiply the negative-overshoot proportion by the mean negative magnitude
# among affected focal-site comparisons in each bin. The score is in Mb;
# equivalently, it is the mean negative magnitude over all comparisons,
# assigning zero to comparisons with no negative overshoot. Higher scores
# indicate more frequent and/or larger underextension.

# %%
plot_buffering_negative_lines(
    "weighted_negative_overshoot_score_mb", "Weighted negative overshoot score (Mb)",
)
