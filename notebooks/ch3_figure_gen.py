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
# Each call displays its figure here and writes a 300 dpi PNG to `figure_folder`.
# The plotting module is self-contained and does not import `tsinfer-paper`.
# Preprocessing writes `data_prefix-overshoot.csv` and `data_prefix-spans.csv`
# beside the aggregate CSV; plots load these ancestor-level tables.

# %%
import pathlib
import importlib
import ch3_plot
importlib.reload(ch3_plot)

# %%
# Prefixes include the simulation region, sample size and seed, but not replicate
# or error-profile suffixes. All available replicates are pooled.
data_folder = pathlib.Path("~/work/tsinfer-anc-eval/data/anc_eval").expanduser()
data_prefix = "OutOfAfrica_3G09-chr20-L0-R1e7-n3000-s1"
phasing_folder = pathlib.Path("~/work/tsinfer-anc-eval/data/phasing").expanduser()
phasing_prefix = "OutOfAfrica_3G09-chr1-L0-R123400000-n20-s1"

# Works when Jupyter starts in either the repository or its notebooks directory.
repo_folder = pathlib.Path.cwd()
if repo_folder.name == "notebooks":
    repo_folder = repo_folder.parent
figure_folder = repo_folder / "figures" / "ch3"

config = ch3_plot.PlotConfig(
    genotype_multipliers=(1, 2, 5, 10, 100),
    transition_multipliers=(1, 10),
    mispol_rates=(0, 0.01, 0.05, 0.1),
    jitter_alpha=0.15,  # Default opacity for all jitter points (alpha can override).
    phase_ser=0.0003,
    phase_right_mb=123.4,
    phase_individuals=5,
    workers=None,  # Use available CPU cores for independent Zarr comparisons.
    dpi=300,
)
processed_paths = ch3_plot.preprocess_data(data_folder, data_prefix, config)
data = ch3_plot.load_data(data_folder, data_prefix, config)
# Discover the generated pure phase-switch profiles.
pure_phase_stores = [
    store for store in data.stores
    if store.geno_multiplier == 0 and store.mispol_rate == 0
]
phase_rates = sorted({store.phase_ser for store in pure_phase_stores})
config.phase_rates = tuple(phase_rates)

# %% [markdown]
# ## Error-free results
#
# The span and overshoot boxplots use the custom DAF cutoffs supplied below.
# Without cutoffs, five rounded quantile bins cover the full frequency range.
# Boxes show quartiles, medians and 1.5 IQR whiskers; outliers are hidden.
# Each inferred ancestor contributes one inferred span and its smallest true
# span. Its left and right overshoots are the minimum signed values over all
# eligible true/focal associations, chosen independently for each side. If any
# focal site is mispolarised, only mispolarised associations are eligible for
# both sides. Counts below
# the boxes show one shared ancestor count per bin. Frequency bins use the
# inferred ancestor's frequency, including polarisation in error profiles.

# %%
ch3_plot.plot_ancestor_spans(data, figure_folder, cutoffs=[0,0.001,0.003,0.02,0.5,1])

# %%
ch3_plot.plot_ancestor_overshoot(data, figure_folder, cutoffs=[0,0.001,0.003,0.02,0.5,1], conf=0.001)

# %% [markdown]
# ## Genotype errors
#
# Compare `call_genotype` in each pure genotype-error profile with the error-free
# store in variant chunks. A diploid error changes genotype dosage (00, 01/10,
# or 11). Rates pool changed calls and denominators across replicates. The dashed
# curve scales the observed 1× rate linearly; both axes are logged.

# %%
genotype_summary = ch3_plot.summarise_genotype_errors(data)
ch3_plot.plot_genotype_error_scaling(genotype_summary, figure_folder, config)

# %% [markdown]
# Transition rates divide 00 → 01/10 and 11 → 01/10 counts by the number of
# original 00 and 11 calls, respectively, in each true DAF bin. Empty denominator
# bins are gaps in the curves. The panels share a logarithmic vertical scale.

# %%
ch3_plot.plot_genotype_error_transitions(genotype_summary, figure_folder, config)

# %% [markdown]
# Confirmation histogram layout: tsinfer 0.4.1 above, 0.5.1 below, mirrored
# logarithmic counts and common physical overshoot bins. Negative overshoot
# means stopping within the true ancestor. Colours run from 1× to 100×.
# Each histogram is followed by a separate grouped boxplot. White boxes with
# black borders and whiskers overlay sampled central jitter and all tail points.
# Every point has the same opacity, coloured by version. Dashed caps show the
# conf and 1-conf quantiles; the solid black line marks zero. Each comparison
# also has two tail-probability panels: negative and positive overshoot magnitude.
# Fractions count each ancestor once if either retained side exceeds the bound;
# all ancestors in the group supply the denominator, including zero overshoots.

# %%
ch3_plot.plot_genotype_overshoot_hist(data, figure_folder)

# %%
ch3_plot.plot_genotype_overshoot_box(data, figure_folder, conf=0.0001, alpha=0.3)

# %%
ch3_plot.plot_genotype_overshoot_tails(data, figure_folder)

# %% [markdown]
# ## Mispolarisation errors
#
# DAF is flipped at sites whose `variant_mispolarisation_mask` is False. Curves
# show normalised histograms in 200 equal-width bins, smoothed by a Gaussian
# kernel with a one-bin standard deviation. This keeps regeneration practical
# for larger datasets. Each profile has its own unit-area density.

# %%
ch3_plot.plot_mispolarisation_daf(data, figure_folder)

# %% [markdown]
# Overshoot plots use true focal DAF from the error-free variant store, with
# 0.5 included only in the high-frequency figure. The all-frequency plots pool
# high and low DAFs. Pure mispolarisation profiles include the error-free
# reference; mixed error profiles are excluded. The low-count histogram includes
# only ancestors with mispolarised focal sites and true derived count ≤ 5.

# %%
ch3_plot.plot_mispolarisation_overshoot_hist(data, figure_folder, frequency="high")

# %%
ch3_plot.plot_mispolarisation_overshoot_box(data, figure_folder, frequency="high", conf=0.001)

# %%
ch3_plot.plot_mispolarisation_overshoot_tails(data, figure_folder, frequency="high")

# %%
ch3_plot.plot_mispolarisation_overshoot_hist(data, figure_folder, frequency="low")

# %%
ch3_plot.plot_mispolarisation_overshoot_box(data, figure_folder, frequency="low", conf=0.001)

# %%
ch3_plot.plot_mispolarisation_overshoot_tails(data, figure_folder, frequency="low")

# %%
ch3_plot.plot_mispolarisation_overshoot_hist(data, figure_folder, frequency="all")

# %%
ch3_plot.plot_mispolarisation_overshoot_box(data, figure_folder, frequency="all", conf=0.001)

# %%
ch3_plot.plot_mispolarisation_overshoot_tails(data, figure_folder, frequency="all")

# %%
ch3_plot.plot_mispolarisation_low_count_overshoot_hist(data, figure_folder)

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
ch3_plot.plot_phase_switch_illustration(
    phasing_folder, phasing_prefix, figure_folder, config
)

# %% [markdown]
# The 10 Mb dataset supplies SER 0, 3e-4, 6e-3 and 1e-2. The histograms use
# the same mirrored version layout as the genotype-error histogram, followed
# by separate boxplots and negative/positive tail fractions.

# %%
ch3_plot.plot_phasing_overshoot_hist(data, figure_folder)

# %%
ch3_plot.plot_phasing_overshoot_box(data, figure_folder, conf=0.001)

# %%
ch3_plot.plot_phasing_overshoot_tails(data, figure_folder)
