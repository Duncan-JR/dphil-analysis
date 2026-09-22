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
# # Initial error-rate estimation
#
# This notebook records the production-style call used for the chromosome 17
# error-estimation dataset. The estimator cell is intentionally not executed
# as part of repository checks.

# %%
from pathlib import Path

import error_estimation
import error_validation
import matplotlib.pyplot as plt
import pandas as pd


# %%
#zarr_dir = Path(
#    "~/work/tsinfer-anc-eval/data/error_eval/zarr_vcfs/"
#).expanduser()
#zarr_prefix = "OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0"
#ts_path = Path(
#    "~/work/tsinfer-anc-eval/data/error_eval/simulated/"
#    "OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0.trees"
#).expanduser()
zarr_dir = Path(
    "~/work/tsinfer-anc-eval/data/anc_eval/zarr_vcfs/"
).expanduser()
zarr_prefix = "OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0"
ts_path = Path(
    "~/work/tsinfer-anc-eval/data/anc_eval/simulated/"
    "OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0.trees"
).expanduser()

recombination = Path(
    "~/work/tsinfer-paper/data/HapMapII_GRCh38/"
    "genetic_map_Hg38_chr20.txt"
).expanduser()

config = error_estimation.EstimationConfig(
    window_sizes=[1e3, 5e3, 7e3, 1e4, 5e4, 7e4, 1e5, 5e5, 7e5, 1e6],
    num_doubletons=10_000,
    random_seed=42,
)


# %%
error_multipliers = [0, 1, 2, 5, 10]
true_diversity = error_validation.get_ts_diversity(ts_path)
rows = []

for multiplier in error_multipliers:
    zarr_path = zarr_dir / (
        f"{zarr_prefix}-geno-{multiplier:.1f}-phase0.0-mispol0.0.zarr"
    )
    estimate = error_estimation.estimate_error_rate(
        zarr_path,
        recombination=recombination,
        config=config,
    )
    rows.append(
        {
            "error_multiplier": multiplier,
            "diversity": estimate.diversity.global_pi_per_bp,
            "epsilon": estimate.fit.epsilon,
            "mu": estimate.fit.mu,
            "pi": estimate.fit.pi,
            "sigma_sq": estimate.fit.sigma_sq,
            "true_error": error_validation.get_true_error_rate(zarr_path),
            "true_diversity": true_diversity,
        }
    )

fit_df = pd.DataFrame(rows)
fit_df


# %%
fit_df.to_csv("../data/tmp/dphil-analysis-results.csv", index=None)

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

axes[0, 0].plot(
    fit_df["error_multiplier"],
    fit_df["true_error"],
    marker="o",
    label="True",
)
axes[0, 0].plot(
    fit_df["error_multiplier"],
    fit_df["epsilon"],
    marker="o",
    label="Estimated",
)
axes[0, 0].set_ylabel("Errors per haplotype-bp")
axes[0, 0].set_title("Genotype error rate")
axes[0, 0].legend()

axes[0, 1].plot(
    fit_df["error_multiplier"],
    fit_df["true_diversity"],
    marker="o",
    label="True",
)
axes[0, 1].plot(
    fit_df["error_multiplier"],
    fit_df["diversity"],
    marker="o",
    label="Estimated",
)
axes[0, 1].set_ylabel("Diversity per bp")
axes[0, 1].set_title("Diversity")
axes[0, 1].legend()

axes[1, 0].plot(
    fit_df["error_multiplier"],
    fit_df["mu"],
    marker="o",
)
axes[1, 0].set_ylabel(r"Estimated $\mu$")
axes[1, 0].set_title("Gamma mean")

axes[1, 1].plot(
    fit_df["error_multiplier"],
    fit_df["sigma_sq"],
    marker="o",
)
axes[1, 1].set_ylabel(r"Estimated $\sigma^2$")
axes[1, 1].set_title("Gamma variance")

for ax in axes.flat:
    ax.set_xlabel("Genotype error-rate multiplier")

fig

# %%
