"""Validation quantities for simulated genotype-error analyses."""

import pathlib

import numpy as np
import tskit
import zarr


def get_true_error_rate(zarr_path):
    """Return realised genotype errors per haplotype-bp from a simulation Zarr."""
    path = pathlib.Path(zarr_path).expanduser()
    group = zarr.open_group(path, mode="r")
    genotype = group["call_genotype"]
    positions = np.asarray(group["variant_position"][:], dtype=float)
    num_haplotypes = genotype.shape[1] * genotype.shape[2]
    sequence_span = positions[-1] - positions[0]
    num_errors = np.asarray(group["variant_genotype_error_count"][:]).sum()
    return float(num_errors / (num_haplotypes * sequence_span))


def get_ts_diversity(ts_path):
    """Return tree-sequence diversity per bp over its first-to-last site span."""
    path = pathlib.Path(ts_path).expanduser()
    ts = tskit.load(path)
    positions = np.asarray(ts.sites_position, dtype=float)
    sequence_span = positions[-1] - positions[0]
    diversity = ts.diversity(mode="site", span_normalise=False)
    return float(diversity / sequence_span)
