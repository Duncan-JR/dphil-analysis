# %% [markdown]
# # Initial error-rate estimation
#
# This notebook records the production-style call used for the chromosome 17
# error-estimation dataset. The estimator cell is intentionally not executed
# as part of repository checks.

# %%
from pathlib import Path

import error_estimation


# %%
zarr_path = Path(
    "~/work/tsinfer-anc-eval/data/error_eval/zarr_vcfs/"
    "OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0-"
    "geno-1.0-phase0.0-mispol0.0.zarr"
).expanduser()
recombination = Path(
    "~/work/tsinfer-paper/data/HapMapII_GRCh38/"
    "genetic_map_Hg38_chr17.txt"
).expanduser()

config = error_estimation.EstimationConfig(
    window_sizes=[1_000, 5_000, 10_000, 50_000, 100_000, 250_000],
    num_doubletons=10_000,
    random_seed=42,
)


# %%
# This call is provided as the reproducible analysis entry point. It is not
# evaluated automatically because the chromosome 17 store is large.
result = error_estimation.estimate_error_rate(
    zarr_path,
    recombination=recombination,
    config=config,
)


# %%
result.fit.epsilon
