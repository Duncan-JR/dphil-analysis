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
# ## Pipeline intro

# %%
# Boxplot of ancestor spans (left) and ancestor overshoot (right) for true and 0.4 inferred ancestors

# %%

# %% [markdown]
# # Errors
#
# ## Genotype error 

# %%
# Actual genotype error rate vs linear scaling 1x to 15x

# %%
# 00 to 01/10 and 11 to 01/10 error rate vs DAF:
# two columns, left 1x, right 15x error rate

# %%
# Confirmation: Geno error 1x to 10x for tsinfer 0.4 vs 0.5

# %% [markdown]
# ## Mispolarisation error

# %%
# DAF with a) no mispol, human mispol, twice human mispol

# %%
# Confirmation plot: mispol error rate 0, 0.01, 0.05, 0.1 overshoot for DAF >= 0.5 top 0.4 bottom 0.5



# %%
# Confirmation plot: same but for f < 0.5

# %% [markdown]
# ## Phase switch error

# %%
# Random sample of x20 chromosome coloured by 
# correct vs. wrong phase with SER at human level

# %%
# Confirmation: phasing error overshoot 0.4 vs 0.5
