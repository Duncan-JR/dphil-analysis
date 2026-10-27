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
# This self-contained figure needs no simulation data. Rows are samples a-h and
# columns are sites 1-10; the focal site is 2. Derived allele counts supply site
# times. The generator replays 0.4.1 and the corrected 0.5.x backport, including assignment of
# 0 at younger sites regardless of the current consensus. Its outputs have been
# checked against `AncestorBuilder.make_ancestor` from tag 0.4.1 and the
# [0.5.x backport](https://github.com/Duncan-JR/tsinfer/tree/0.5.x)
# (commit `d694eea`). Both removal and flag updates use the same gate.
#
# Samples b and e leave at sites 4 and 7 in both versions. The site-5 doubleton
# has both carriers in the current sample set and does not mark samples for
# removal. The c/f doubletons at sites 8-9 remove c only in 0.5.1, reducing the
# set from three to two and stopping extension after site 9. Arrows indicate
# that the stopping criterion was not reached within the displayed window.
# The included PNG and editable SVG are exported to the thesis figure folder.

# %%
# include
figure_folder = pathlib.Path("~/work/dphil-thesis/figures/ch3").expanduser()
extension_config = ch3_ancestor_extension.FigureConfig(
    # Columns are one-indexed sites 1-10; site 2 is focal.
    genotypes=(
        (0, 1, 1, 1, 1, 1, 1, 0, 0, 1),  # a
        (0, 1, 0, 0, 0, 1, 1, 0, 0, 1),  # b
        (0, 1, 1, 1, 0, 1, 1, 1, 1, 1),  # c
        (0, 1, 1, 1, 1, 1, 1, 0, 0, 1),  # d
        (0, 1, 1, 1, 0, 0, 0, 0, 0, 1),  # e
        (1, 0, 1, 1, 0, 1, 1, 1, 1, 1),  # f
        (1, 0, 1, 0, 0, 1, 1, 0, 0, 1),  # g
        (0, 0, 0, 1, 0, 1, 1, 0, 0, 0),  # h
    ),
    focal_site=2,
    version_colors=ch3_plot.PlotConfig().version_colors,
    figsize=(8.2, 5.6),
    dpi=400,
)
ch3_ancestor_extension.plot_ancestor_extension(
    extension_config, figure_folder=figure_folder
);

# %% [markdown]
# ## Genealogical justification for ancestor extension
#
# This hand-built tree sequence reproduces **every genotype at sites 4–10** of
# the example above, with three marginal trees: site 4, sites 5–7 and sites
# 8–10. Sample labels and the colours of **F** (focal ancestor, red), **E**
# (within-clade doubleton, purple) and **G** (conflicting doubleton, orange)
# match that example. Internal nodes without labels only complete the topology.
# Ages are illustrative, on a shared linear scale; they are chosen for legibility
# and are not the derived allele counts used as site times by the algorithm.
# Crosses mark single 0→1 mutations, labelled by site number. The focal mutation
# at site 2 lies outside this window, so F is identified by its red node.
#
# At site 5, E's carriers are a and d: both are still in the current sample set,
# so this nested doubleton supplies no evidence that either has left the focal
# clade. Between sites 7 and 8, c leaves F's descendants and joins f under G.
# The doubletons at sites 8 and 9 are therefore informative even though their
# derived allele count (2) is lower than the focal count (5): c is inside the
# current set S={a,c,d}, while f is outside it. Relative to focal-site membership,
# these sites exhibit all four gametes: 11 (c), 10 (a,b,d,e), 01 (f), and 00 (g,h).
# Under the single-mutation model used here, this requires recombination.
# Two successive disagreements with the sample-set consensus mark and then
# remove c, leaving S={a,d} and reaching the halving threshold at site 9.
# The tree is compatible with this history; the genotypes alone do not identify
# a unique recombination event. F is retained as a unary node in the final tree
# so the same ancestor can be followed across all three panels.
#
# Construction follows tskit's [tables and editing tutorial](https://tskit.dev/tutorials/tables_and_editing.html),
# and SVG styling follows its [visualization tutorial](https://tskit.dev/tutorials/viz.html).
# This plot is displayed in the notebook only.

# %%
extension_tree_config = {
    "sequence_length": 10.5,
    "plot_interval": (3.5, 10.5),
    "intervals": ((0, 4.5), (4.5, 7.5), (7.5, 10.5)),
    "regions": {(3.5, 4.5): "Site 4", (4.5, 7.5): "Sites 5–7", (7.5, 10.5): "Sites 8–10"},
    "node_ages": {"E": 2, "C": 3, "H": 3, "B": 4, "G": 3, "F": 5, "A": 8, "R": 10},
    "topologies": (
        {"R": ("A", "B"), "A": ("F", "H"), "F": ("E", "C"),
         "E": ("a", "d"), "C": ("c", "e"), "H": ("f", "h"), "B": ("b", "g")},
        {"R": ("A", "e"), "A": ("F", "B", "H"), "F": ("E", "c"),
         "E": ("a", "d"), "H": ("f", "h"), "B": ("b", "g")},
        {"R": ("A", "h"), "A": ("F", "B", "G", "e"), "F": ("E",),
         "E": ("a", "d"), "G": ("c", "f"), "B": ("b", "g")},
    ),
    # Each mutation is on the branch immediately above the named node.
    "mutations": ((4, "A", 9), (5, "E", 3.5), (6, "A", 8.6), (7, "A", 9.3),
                  (8, "G", 4), (9, "G", 5.5), (10, "A", 9)),
    "node_colors": {"F": extension_config.focal_color,
                    "E": extension_config.contained_color,
                    "G": extension_config.version_colors[1]},
    "mutation_colors": {5: extension_config.contained_color,
                        8: extension_config.version_colors[1],
                        9: extension_config.version_colors[1]},
    "ink_color": extension_config.ink_color,
    "muted_color": extension_config.muted_color,
    "older_color": "#b59a66",
    "background_color": "#f1f4f6",
    "canvas_color": "white",
    "size": (1100, 470),
    "y_ticks": (0, 2, 5, 8, 10),
    "symbol_size": 7,
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
for position, node_name, age in extension_tree_config["mutations"]:
    site_id = extension_tables.sites.add_row(position=position, ancestral_state="0")
    extension_tables.mutations.add_row(
        site=site_id, node=extension_node_ids[node_name], derived_state="1", time=age,
    )
extension_tables.sort()
extension_ts = extension_tables.tree_sequence()

# Verify compatibility with the existing haplotype illustration, including all
# background mutations, rather than just the highlighted doubletons.
expected_extension_genotypes = np.asarray(extension_config.genotypes)[:, 3:]
np.testing.assert_array_equal(extension_ts.genotype_matrix().T, expected_extension_genotypes)
assert extension_ts.num_trees == len(extension_tree_config["intervals"])

# %%
extension_svg_id = extension_tree_config["svg_id"]
extension_ink = extension_tree_config["ink_color"]
extension_muted = extension_tree_config["muted_color"]
extension_older = extension_tree_config["older_color"]
extension_background = extension_tree_config["background_color"]
# Scope every selector so this SVG cannot recolour other notebook figures.
extension_style = [
    f"#{extension_svg_id} text {{font-family: Arial, sans-serif; fill: {extension_ink}; font-size: 13px;}}",
    f"#{extension_svg_id} .edge {{stroke: {extension_muted}; stroke-width: 1.4;}}",
    f"#{extension_svg_id} .node > .sym {{fill: {extension_muted}; stroke: white; stroke-width: 1;}}",
    f"#{extension_svg_id} .sample > .lab {{font-style: italic;}}",
    f"#{extension_svg_id} .n{extension_node_ids['c']} > .lab {{font-weight: bold;}}",
    f"#{extension_svg_id} .mut .sym {{stroke: {extension_older}; stroke-width: 2;}}",
    f"#{extension_svg_id} .mut .lab {{fill: {extension_older}; font-size: 12px;}}",
    f"#{extension_svg_id} .y-axis .grid {{stroke: #e4e8eb; stroke-dasharray: 3 5;}}",
    f"#{extension_svg_id} .background path {{fill: {extension_background}; fill-opacity: 1;}}",
    f"#{extension_svg_id} .background path:nth-child(even) {{fill: white;}}",
    f"#{extension_svg_id} .x-axis .ticks .lab {{display: none;}}",
    f"#{extension_svg_id} .x-regions rect {{fill: #f0f3f5; stroke: white; stroke-width: 2;}}",
]
for name, color in extension_tree_config["node_colors"].items():
    node_id = extension_node_ids[name]
    extension_style.extend([
        f"#{extension_svg_id} .n{node_id} > .sym {{fill: {color};}}",
        f"#{extension_svg_id} .n{node_id} > .lab {{fill: {color}; font-size: 15px; font-weight: bold;}}",
        f"#{extension_svg_id} .n{node_id} > .edge {{stroke: {color}; stroke-width: 2;}}",
    ])
extension_g = extension_node_ids["G"]
extension_orange = extension_tree_config["node_colors"]["G"]
extension_style.extend([
    f"#{extension_svg_id} .n{extension_g} .edge {{stroke: {extension_orange}; stroke-width: 2;}}",
    f"#{extension_svg_id} .n{extension_g} .sample > .lab {{fill: {extension_orange};}}",
])
for mutation in extension_ts.mutations():
    position = extension_ts.site(mutation.site).position
    color = extension_tree_config["mutation_colors"].get(position, extension_older)
    extension_style.extend([
        f"#{extension_svg_id} .m{mutation.id} > .sym {{stroke: {color};}}",
        f"#{extension_svg_id} .m{mutation.id} > .lab {{fill: {color};}}",
    ])

extension_node_labels = {extension_node_ids[name]: name for name in extension_config.samples}
for name in extension_tree_config["node_colors"]:
    extension_node_labels[extension_node_ids[name]] = name
extension_mutation_labels = {}
for mutation in extension_ts.mutations():
    position = extension_ts.site(mutation.site).position
    extension_mutation_labels[mutation.id] = str(int(position))

extension_tree_svg = extension_ts.draw_svg(
    size=extension_tree_config["size"],
    x_lim=extension_tree_config["plot_interval"],
    x_regions=extension_tree_config["regions"],
    x_label="Genomic position · mutations labelled by site",
    time_scale="time", y_axis=True, y_gridlines=True,
    y_label="Illustrative age", y_ticks=extension_tree_config["y_ticks"],
    node_labels=extension_node_labels, mutation_labels=extension_mutation_labels,
    symbol_size=extension_tree_config["symbol_size"],
    root_svg_attributes={"id": extension_svg_id},
    style="\n".join(extension_style),
    preamble=(
        f'<rect width="100%" height="100%" fill="{extension_tree_config["canvas_color"]}"/>'
    ),
)
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

# %%
import ch3_anc_perf
import matplotlib.pyplot as plt

perf_config = ch3_anc_perf.PlotConfig()
perf_path = pathlib.Path("~/work/dphil-analysis/data/ch3/anc_perf/perf_combined.csv").expanduser()
perf_data = ch3_anc_perf.load_performance(perf_path, perf_config)

# %%
perf_speedup_figure = ch3_anc_perf.plot_speedup(perf_data, perf_config)
plt.show()

# %%
perf_elapsed_figure = ch3_anc_perf.plot_elapsed(perf_data, perf_config)
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
# stage, including setup and I/O. This measures achieved CPU concurrency, not
# speedup from threading: there is no single-threaded baseline here. Shared
# linear y axes allow comparison between stages; the dotted line marks the
# 196 configured threads.

# %%
perf_parallelism_figure = ch3_anc_perf.plot_resources(perf_data, "parallelism", perf_config)
plt.show()
