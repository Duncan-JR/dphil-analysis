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
# # Loading data and importing
#

# %%
# data is in ~/work/tsinfer-anc-eval/data/anc_eval

data_prefix = "OutOfAfrica_3G09-chr20-L0-R1e6"

#output all plots to ~/work/dphil-thesis/figures/ch3 with sensible titles in dpi 300 pngs. Make a helper function here to do this

# %% [markdown]
# # Error-free results

# %%
# Boxplot of ancestor spans for true, 
# 0.4.1 and 0.5.1 inferred ancestors from sim data (no error profile) divided into 8 equally sized frequency bins

# %%
# Boxplot of ancestor overshoot for true, 
# 0.4.1 and 0.5.1 inferred ancestors from sim data (no error profile) same freq bins

# %% [markdown]
# # Errors
#
# ## Genotype error 

# %%
# Actual genotype error rate (y) with modifiers 1x to 100x (log scale x axis) vs
# linear scaling 1x to 100x (i.e. straight line in non logged space that takes error rate at 1x and multiplies it)

# %%
# 00 genotype (homozygote anc) to 01/10 (hetero) and 11 (homozygote derived) to 01/10 error rate (y) against derived allele frequency of the variant (x).
# two columns, left 1x, right 10x error rate, colored curve for each type of error. Extract call_genotype matrix from zarr_vcf of none error profile
# and each error rate profile and compare to gather the data needed for the plot

# %%
# Confirmation plot (similar, except now we have multiple error rates): Overshoot histogram with colors for 
# Geno error 1x to 100x for tsinfer 0.4.1 (top) vs 0.5.1 (bottom)

# %% [markdown]
# ## Mispolarisation error

# %%
# DAF with a) no mispol, 0.01, 0.05,0.1 mispol (one set of axes, colored curves overlayed)

# %%
# Confirmation plot: mispol error rate 0, 0.01, 0.05, 0.1 overshoot for DAF >= 0.5 top 0.4.1 bottom 0.5.1

# %%
# Confirmation plot: same but for DAF < 0.5

# %% [markdown]
# ## Phase switch error

# %%
# Aligned pairs of chromosomes 
# correct vs. wrong phase with SER at 5e-3 
# using tsinfer-anc-eval/data/phasing
# use recommendations in ~/exo/archive/this_week_2026-10-04*.org under **** 11:19 Configure phase-switch illustration 

# %%
# Confirmation plot: phasing error overshoot 0.4.1 vs 0.5.1

# %%
