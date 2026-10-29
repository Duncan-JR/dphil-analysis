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
# Each call displays its figure here. Only #include plots save a 300 dpi PNG
# to the thesis figure_folder; figure_folder=None displays without saving.
# The plotting module is self-contained and does not import `tsinfer-paper`.
# Run src/ch3_extract_data.py on the machine holding the variant Zarr stores.
# Transfer PREFIX_zarr_data.csv alongside the aggregate CSV; the main plots
# need no Zarrs locally. Per-focal-site tables are written to data_folder.
# The separate phase-switch illustration still uses its chromosome 1 Zarrs.
# The empirical DAF comparison uses the local TGP chromosome 20 Zarr.

# %%
import pathlib
import importlib
import xml.etree.ElementTree as ET
import IPython.display as ipd
import numpy as np
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
figure_folder = pathlib.Path("~/work/dphil-thesis/figures/ch3").expanduser()
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
    extension_config, figure_folder=None
);

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
ipd.display(ipd.SVG(extension_tree_svg))

# %%
# Prefixes include the simulation region, sample size and seed, but not replicate
# or error-profile suffixes. All available replicates are pooled.
data_folder = pathlib.Path("~/work/dphil-analysis/data/ch3").expanduser()
aggregated_folder = pathlib.Path("~/work/tsinfer-anc-eval/data/anc_eval/dataframes").expanduser()
data_prefix = "OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1"
aggregated_path = aggregated_folder / f"{data_prefix}-aggregated.csv"
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
# Build the focal-site tables when absent; rebuild explicitly when input data changes.
rebuild_focal_tables = False
focal_paths = ch3_plot.FocalPaths(
    data_folder / f"{data_prefix}_focal_overshoots.csv",
    data_folder / f"{data_prefix}_focal_spans.csv",
)
if rebuild_focal_tables or not focal_paths.overshoots.exists() or not focal_paths.spans.exists():
    processed_paths = ch3_plot.preprocess_data(
        aggregated_path, zarr_data_path, data_folder, config
    )
data = ch3_plot.load_data(data_folder, data_prefix, config)
# Available pure phase-error rates are included in the exported CSV.
config.phase_rates = data.phase_rates

# %% [markdown]
# ## Error-free results
#
# The span and overshoot boxplots use the custom DAF cutoffs supplied below.
# Without cutoffs, five rounded quantile bins cover the full frequency range.
# Boxes show quartiles, medians and 1.5 IQR whiskers; outliers are hidden.
# Each focal site contributes its own true/inferred span comparison and one
# overshoot per side. Sites remain separate when they share a true or inferred
# ancestor, including when their polarisation states differ. Counts below the
# boxes show focal sites per bin. Bins use each focal allele's observed frequency.

# %%
# include
# Focal-site counts use a logarithmic scale, with exact counts above each bar.
ch3_plot.plot_ancestor_spans(
    data,
 #   figure_folder=figure_folder,
    cutoffs=[0,0.001,0.01,0.5,1],
    figsize=(7, 4.5),
)

# %%

#include
ch3_plot.plot_ancestor_overshoot(
    data,
    figure_folder=figure_folder,
    cutoffs=[0,0.001,0.003,0.02,0.5,1],
    conf=0.001,
    figsize=(7, 4.5),
)

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
# The bottom panel counts focal-site comparisons, shared by both versions.
# This figure is displayed in the notebook only.

# %%
ancestor_overlap_errors = ch3_plot.load_ancestor_overlap_errors(aggregated_path, config=config)
ancestor_error_summary = ch3_plot.summarise_ancestor_overlap_errors(
    ancestor_overlap_errors,
    config=config,
    cutoffs=[0,0.001,0.003,0.02,0.5,1],
)
ch3_plot.plot_ancestor_overlap_errors(
    ancestor_error_summary,
    figure_folder=None,
    config=config,
    figsize=config.ancestor_error_size,
)

# %% [markdown]
# ### Direction of ancestral-state errors in tsinfer 0.5
#
# Split the error-free tsinfer 0.5 focal-site comparisons into 20 fixed-width
# DAF bins, each covering 5 percentage points. Plot rates at bin midpoints,
# pooling each error type over the bin's total overlapping bp. Include DAF 1
# in the final bin; empty bins have undefined rates.
# The aggregate writer's names are reversed: `num_should_be_0` counts true 1
# inferred as 0, and `num_should_be_1` counts true 0 inferred as 1. Legend labels
# describe the actual state: "Should be 1 (1 → 0)" and "Should be 0 (0 → 1)".
# Colours match the genotype-transition plot; zero burdens are gaps on the log
# scale. This figure is displayed in the notebook only.

# %%
ancestor_error_types = ch3_plot.summarise_ancestor_error_types(ancestor_overlap_errors, config=config)
ch3_plot.plot_ancestor_error_types(
    ancestor_error_types,
    figure_folder=None,
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
# include: right of plot
ch3_plot.plot_genotype_overshoot_mean(
    data,
    figure_folder=figure_folder,
    figsize=(5, 4.5),
)

# %%
#include: left of plot
# Smaller legends keep the narrow histogram panels visible.
ch3_plot.plot_genotype_overshoot_hist(
    data,
    figure_folder=figure_folder,
    figsize=(5, 4.5),
    legend_fontsize=7.5,
)

# %%
ch3_plot.plot_genotype_overshoot_box(
    data,
    figure_folder=None,
    conf=0.001,
    alpha=0.3,
    figsize=(8.5, 3.5),
)

# %%
ch3_plot.plot_genotype_overshoot_tails(
    data,
    figure_folder=None,
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

# %% [markdown]
# Overshoot plots use true focal DAF from the error-free variant store, with
# 0.5 included only in the high-frequency figure. The all-frequency plots pool
# high and low DAFs. Pure mispolarisation profiles include the error-free
# reference; mixed error profiles are excluded. The low-count histogram includes
# only mispolarised focal sites with true derived count ≤ 5.

# %%
ch3_plot.plot_mispolarisation_overshoot_hist(
    data,
    figure_folder=None,
    frequency="high",
    figsize=(7, 4.5),
)

# %%
ch3_plot.plot_mispolarisation_overshoot_box(
    data,
    figure_folder=None,
    frequency="high",
    conf=0.001,
    figsize=(8.5, 3.5),
)

# %%
ch3_plot.plot_mispolarisation_overshoot_tails(
    data,
    figure_folder=None,
    frequency="high",
    figsize=(8.5, 4.5),
    show_counts=True,
)

# %%
ch3_plot.plot_mispolarisation_overshoot_hist(
    data,
    figure_folder=None,
    frequency="low",
    figsize=(7, 4.5),
)

# %%
ch3_plot.plot_mispolarisation_overshoot_box(
    data,
    figure_folder=None,
    frequency="low",
    conf=0.001,
    figsize=(8.5, 3.5),
)

# %%
ch3_plot.plot_mispolarisation_overshoot_tails(
    data,
    figure_folder=None,
    frequency="low",
    figsize=(8.5, 4.5),
    show_counts=True,
)

# %%
ch3_plot.plot_mispolarisation_overshoot_hist(
    data,
    figure_folder=None,
    frequency="all",
    figsize=(7, 4.5),
)

# %%
ch3_plot.plot_mispolarisation_overshoot_box(
    data,
    figure_folder=None,
    frequency="all",
    conf=0.001,
    figsize=(8.5, 3.5),
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

# %%
#include
# Top rate legend and version label mirror the bottom panel.
ch3_plot.plot_mispolarisation_low_count_overshoot_hist(
    data,
    figure_folder=figure_folder,
    figsize=(7, 4.5),
)

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

# %%
#include right
ch3_plot.plot_phasing_overshoot_mean(
    data,
    figure_folder=figure_folder,
    figsize=(5, 4.5),
)

# %%
(0.0003+0.006)/2

# %%
# don't include anymore
ch3_plot.plot_phasing_overshoot_box(
    data,
    figure_folder=None,
    conf=0.001,
    figsize=(8.5, 3.5),
)

# %%
ch3_plot.plot_phasing_overshoot_tails(
    data,
    figure_folder=None,
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
# Both plots retain the original figure dimensions; the extra group narrows bars.

# %%
import pathlib

import ch3_anc_perf
import ch3_anc_perf_analysis
import IPython.display as ipdisplay
import matplotlib.pyplot as plt
import pandas as pd

perf_config = ch3_anc_perf.PlotConfig(
    sample_labels={6404: "6,404 (1KGP)"},
    sample_titles={6404: "6,404 haplotypes\n(1KGP)"},
)
perf_folder = pathlib.Path("~/work/dphil-analysis/data/ch3/results").expanduser()
perf_data = ch3_anc_perf.load_performance(perf_folder / "perf_combined.csv", perf_config)
tgp_perf_data = ch3_anc_perf.load_performance(perf_folder.parent / "tgp_perf.csv", perf_config)
perf_comparison_data = pd.concat([perf_data, tgp_perf_data]).sort_index()

# %%
# include

perf_speedup_figure = ch3_anc_perf.plot_speedup(perf_comparison_data, perf_config)
perf_speedup_figure.savefig(
    pathlib.Path("~/work/dphil-thesis/figures/ch3/inference_performance_speedup.png").expanduser(),
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.show()

# %%
# include
perf_elapsed_figure = ch3_anc_perf.plot_elapsed(perf_comparison_data, perf_config)
perf_elapsed_figure.savefig(
    pathlib.Path("~/work/dphil-thesis/figures/ch3/inference_performance_elapsed.png").expanduser(),
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.show()

# %% [markdown]
# ### Memory usage
#
# Peak resident memory from each stage's provenance, converted from bytes to
# GiB. Memory is not additive across stages. Each stage runs in a fresh process;
# its high-water mark includes setup and input loading. Log-log axes show
# scaling across the four sample sizes, with shared limits for all stages.

# %%
perf_memory_figure = ch3_anc_perf.plot_resources(perf_data, "memory_gib", perf_config)
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
# include
perf_parallelism_figure = ch3_anc_perf.plot_resources(
    perf_data, "parallelism", perf_config, reference_data=tgp_perf_data,
)
perf_parallelism_figure.savefig(
    pathlib.Path("~/work/dphil-thesis/figures/ch3/inference_performance_parallelism.png").expanduser(),
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.show()

# %% [markdown]
# ### Linesweep groups: scheduling capacity
#
# Each panel overlays both versions for one sample size. Each point is an
# actual matching group: x is its scheduling order (older to younger), y is
# the number of ancestors available concurrently. The empty virtual-root group
# is omitted from the log scale. The horizontal line marks 196 threads.
# These are the **hybrid** groups used in the benchmark: linesweep for older
# small epochs, followed by unchanged time epochs after the original cutoff.
# Group number is specific to each version; points at the same x need not
# contain the same ancestors.
#
# Reconstruct ancestor start/end/time from the matched ancestor TS; exclude
# path-compression nodes. The intervals were checked against all eight original
# LMDB ancestor files and match exactly. Replay the standalone grouping code
# from local tsinfer tag 0.5.1 (shared by both benchmark schedulers), preserving
# the median-size cutoff, group-ID gap, and virtual-root removal. No HMM is run.
# Every replayed group ID and count is checked against its benchmark log.
# The four sample sizes give 22,598 validated groups across the eight runs.
#
# Compact CSV caches are in `results/analysis`. Set `rebuild_perf_analysis=True`
# to regenerate them using multiprocessing queues across available CPU cores.
# The legacy grouping code is read from the configured local Git checkout;
# neither the installed tsinfer matcher nor a legacy Zarr API is needed.

# %%
perf_analysis_config = ch3_anc_perf_analysis.AnalysisConfig(
    tsinfer_source=pathlib.Path("~/work/tsinfer").expanduser(),
    grouping_ref="0.5.1",
    workers=None,
)
rebuild_perf_analysis = False
if rebuild_perf_analysis:
    perf_analysis = ch3_anc_perf_analysis.preprocess_results(
        perf_folder, perf_config, perf_analysis_config,
    )
else:
    perf_analysis = ch3_anc_perf_analysis.load_analysis(perf_folder, perf_config)
perf_linesweep_figure = ch3_anc_perf_analysis.plot_linesweep(
    perf_analysis, perf_config, perf_analysis_config,
)
plt.show()

# %% [markdown]
# Grouping improves as expected. The table also shows the fraction of measured
# group elapsed time spent in groups smaller than the thread count. Larger
# groups provide capacity; their size alone does not measure busy HMM workers.

# %%
perf_group_table = perf_analysis.summary[[
    "num_samples", "version", "num_groups", "median_group_size",
    "underfull_time_fraction",
]].copy()
perf_version_labels = dict(zip(perf_config.versions, perf_config.version_labels))
perf_group_table["version"] = perf_group_table.version.map(perf_version_labels)
perf_group_table = perf_group_table.rename(columns={
    "num_samples": "Haplotypes", "version": "Method", "num_groups": "Groups",
    "median_group_size": "Median ancestors/group",
    "underfull_time_fraction": "Time in groups <196",
})
ipdisplay.display(perf_group_table.style.hide(axis="index").format({
    "Haplotypes": "{:,.0f}", "Groups": "{:,.0f}",
    "Median ancestors/group": "{:.0f}", "Time in groups <196": "{:.1%}",
}))

# %% [markdown]
# ### Why average CPU concurrency falls
#
# At 100,000 haplotypes, ancestor matching falls from **124.6 to 65.2 average
# active cores**, even though groups fall from **8,596 to 6,487** and median
# group size rises from **12 to 18**. CPU time falls **7.86-fold**, whereas
# elapsed time falls **4.11-fold**. Since average cores = CPU time / elapsed
# time, these measurements imply the lower core count directly. The following
# plot shows the same relationship for both matching stages and all sizes.

# %%
perf_work_reduction_figure = ch3_anc_perf_analysis.plot_work_reduction(
    perf_analysis, perf_config,
)
plt.show()

# %% [markdown]
# **What the saved data establish.** At 100,000 haplotypes there are the same
# 1,085,897 ancestors and 1,086,350 inference sites in both versions. The older
# linesweep section covers 197,858 ancestors including the virtual root: its
# total covered sites fall about 25.4%, and its groups fall 8,439 → 6,330.
# The other 888,039 ancestors remain in the same 157 young time-epoch groups.
# Their total covered sites **increase 6.1%**, so the overall covered-site count
# increases 4.6%. Here "shorter ancestors" describes the older subset's aggregate
# span, not the entire ancestor set. The young epochs still account for 66.5%
# and 60.5% of group elapsed time in tsinfer 0.4 and tsinfer 0.5 respectively.
# The fraction of group time in groups smaller than 196 actually rises from
# 32.7% to 37.4%: the larger groups have become faster disproportionately.
#
# Sample matching is a useful comparison. It has no ancestor-group barriers,
# and the samples still span the full region. Yet its average cores also fall
# (188.5 → 167.3 at 100,000; 149.0 → 71.1 at 10,000). Its CPU time falls
# 9.88-fold at 100,000 while elapsed time falls 8.77-fold. Group counts alone
# therefore cannot explain the full change. The matched ancestor TS is similar
# in size: about 22.61M versus 22.52M edges (0.4% fewer) and 1.676M versus 1.615M
# nodes (3.6% fewer), despite the much greater CPU-work reduction.
#
# **Most plausible mechanism: cheaper HMM work exposes other costs.** The
# [legacy ancestor matcher](https://github.com/tskit-dev/tsinfer/blob/0.5.1/tsinfer/inference.py)
# consumes ancestor haplotypes through a producer, creates a thread pool for
# each group, waits for the group, and then inserts paths/mutations serially.
# `threading.local()` is created inside the per-ancestor callback, so the shown
# code allocates a new matcher for each task instead of reusing one per worker.
# Its C constructor retains the GIL, while `find_path` releases it. Full-length
# haplotype and match buffers also remain even when the matched interval is
# shorter. These costs do not shrink in proportion to interval length and can
# limit the rate at which useful HMM tasks reach the workers. Group barriers,
# task-length imbalance and serial insertion further limit realised occupancy.
# This code is shared by both versions; faster tasks make it more prominent.
#
# This version comparison also changes the HMM numerical update. In the tagged
# C sources, tsinfer 0.4 normalises every active likelihood using `tsk_round`,
# which calls `pow` and `round`; tsinfer 0.5 uses a threshold/max operation.
# The benchmark logs confirm the precision → likelihood-threshold switch.
# See the [tsinfer 0.4 matcher](https://github.com/tskit-dev/tsinfer/blob/0.4.1/lib/ancestor_matcher.c)
# and [tsinfer 0.5 matcher](https://github.com/tskit-dev/tsinfer/blob/0.5.1/lib/ancestor_matcher.c).
# This can reduce CPU work per site independently of ancestor spans, and is a
# strong candidate for the large reduction seen in both matching stages.
# Donor topology, traceback complexity and memory effects can also contribute.
#
# **Limits and next measurement.** These are eight single runs. The logs report
# total group duration, not per-group CPU time, HMM time, task-feed rate, or
# insertion time; provenance does not pin the exact source commit of the
# tsinfer 0.5 development build. Thus the dominant bottleneck and the share due
# to the numerical change are hypotheses, not measured attributions. Profile
# task production, matcher construction, `find_path`, and path insertion
# separately; record busy HMM workers over time. Reuse matchers and the pool,
# then repeat a thread-count sweep with fixed ancestors. A same-source comparison
# using the two ancestor sets would isolate the ancestor-length/grouping effect.
