"""Validation quantities for simulated genotype-error analyses."""

import csv
import pathlib

import error_estimation
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


def sample_doubletons(zarr_path, config):
    """Sample doubletons without running mismatch calculation or model fitting."""
    path = pathlib.Path(zarr_path).expanduser()
    group = zarr.open_group(path, mode="r")
    positions = np.asarray(group["variant_position"][:], dtype=float)
    G = np.asarray(group["call_genotype"][:])
    ac = G.sum(axis=(1, 2))
    return error_estimation._sample_doubletons(G, ac, positions, config)


def write_doubletons_csv(doubletons, path):
    """Write the complete :class:`error_estimation.Doubletons` state to CSV."""
    path = pathlib.Path(path).expanduser()
    fieldnames = [
        "site_index",
        "position",
        "sample_0",
        "sample_1",
        "ploidy_0",
        "ploidy_1",
        "num_eligible",
    ]
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(doubletons.site_indices)):
            writer.writerow(
                {
                    "site_index": doubletons.site_indices[index],
                    "position": doubletons.positions[index],
                    "sample_0": doubletons.sample_indices[index, 0],
                    "sample_1": doubletons.sample_indices[index, 1],
                    "ploidy_0": doubletons.ploidy_indices[index, 0],
                    "ploidy_1": doubletons.ploidy_indices[index, 1],
                    "num_eligible": doubletons.num_eligible,
                }
            )


def get_true_doubleton_mask(doubletons, truth_zarr_path):
    """Identify observed doubletons whose site and carriers match truth data."""
    path = pathlib.Path(truth_zarr_path).expanduser()
    group = zarr.open_group(path, mode="r")
    focal_G = np.asarray(
        group["call_genotype"].get_orthogonal_selection(
            (doubletons.site_indices, slice(None), slice(None))
        )
    )
    focal_ac = focal_G.sum(axis=(1, 2))
    row = np.arange(len(doubletons.site_indices))[:, None]
    carrier_alleles = focal_G[
        row,
        doubletons.sample_indices,
        doubletons.ploidy_indices,
    ]
    return (focal_ac == 2) & np.all(carrier_alleles == 1, axis=1)


def evaluate_mismatch_cutoffs(
    summary,
    true_doubleton_mask,
    *,
    cutoff_proportions,
    L_mismatch_trim_values,
):
    """Return classification metrics for high-mismatch cutoff combinations."""
    records = []
    num_doubletons = len(true_doubleton_mask)
    num_true = np.count_nonzero(true_doubleton_mask)
    num_false = num_doubletons - num_true
    for L_mismatch_trim in L_mismatch_trim_values:
        row = np.flatnonzero(summary.window_sizes == L_mismatch_trim)[0]
        order = np.argsort(summary.counts[row], kind="stable")
        for cutoff_proportion in cutoff_proportions:
            num_excluded = int(cutoff_proportion * num_doubletons)
            num_retained = num_doubletons - num_excluded
            retained = order[:num_retained]
            excluded = order[num_retained:]
            retained_true = np.count_nonzero(true_doubleton_mask[retained])
            retained_false = num_retained - retained_true
            excluded_false = np.count_nonzero(~true_doubleton_mask[excluded])
            excluded_true = num_excluded - excluded_false
            records.append(
                {
                    "L_mismatch_trim": L_mismatch_trim,
                    "cutoff_proportion": cutoff_proportion,
                    "num_doubletons": num_doubletons,
                    "num_true": num_true,
                    "num_false": num_false,
                    "num_retained": num_retained,
                    "retained_true": retained_true,
                    "retained_false": retained_false,
                    "excluded_true": excluded_true,
                    "excluded_false": excluded_false,
                    "true_proportion": num_true / num_doubletons,
                    "retained_true_proportion": retained_true / num_retained,
                    "true_retention_rate": retained_true / num_true,
                    "excluded_false_proportion": excluded_false / num_excluded,
                    "false_removal_rate": (
                        excluded_false / num_false if num_false > 0 else np.nan
                    ),
                    "accuracy": (
                        retained_true + excluded_false
                    ) / num_doubletons,
                }
            )
    return records
