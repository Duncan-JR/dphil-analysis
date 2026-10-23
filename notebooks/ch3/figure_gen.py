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

# %%
import pathlib
import importlib
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

config = ch3_plot.PlotConfig(
    genotype_multipliers=(1, 2, 5, 10, 100),
    transition_multipliers=(1, 10),
    mispol_rates=(0, 0.01, 0.05, 0.1),
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
    figure_folder=figure_folder,
    cutoffs=[0,0.002,0.02,0.5,0.8,1],
    figsize=(7, 4.5),
)

# %%
ch3_plot.plot_ancestor_overshoot(
    data,
    figure_folder=None,
    cutoffs=[0,0.001,0.003,0.02,0.5,1],
    conf=0.001,
    figsize=(7, 4.5),
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

# %%
#include
ch3_plot.plot_mispolarisation_daf(
    data,
    figure_folder=figure_folder,
    figsize=(7, 3.5),
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
#include
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
#include
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

# %%
ch3_plot.plot_mispolarisation_overshoot_tails(
    data,
    figure_folder=None,
    frequency="all",
    figsize=(8.5, 4.5),
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
