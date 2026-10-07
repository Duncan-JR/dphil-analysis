"""Focal interval stitching and site-based comparisons of complete copying paths.

:func:`focal_parent_path` fixes parents only inside anchored zero-mismatch
intervals. The complement defines the independent HMM gaps. Both HMM paths use
that same partition in :func:`switch_counts` and :func:`comparison_metrics`.
"""

import dataclasses
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SwitchCounts:
    """Partition parent changes by the provenance of their two adjacent sites."""

    focal: int
    gap: int
    seam: int

    @property
    def total(self) -> int:
        return self.focal + self.gap + self.seam


def site_intervals(mask: np.ndarray) -> np.ndarray:
    """Return maximal true runs as half-open site-index intervals."""
    padded = np.pad(mask, (1, 1), constant_values=False)
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    return transitions.reshape(-1, 2)


def focal_parent_path(
    num_sites: int, left: np.ndarray, right: np.ndarray, parents: np.ndarray
) -> np.ndarray:
    """Cover the union of focal intervals with the fewest copying segments.

    At each uncovered position, choose the available interval extending furthest
    right, breaking ties by original node ID. Keep its parent until that interval
    ends. This greedy interval cover minimizes the number of focal segments in
    each connected covered component. Sites outside the union retain parent -1;
    no HMM output or unanchored exact islands enter gap determination.
    """
    if left.shape != right.shape or left.shape != parents.shape:
        raise ValueError("Focal interval bounds and parents must have the same shape")
    invalid_bounds = (left < 0) | (left >= right) | (right > num_sites)
    if np.any(invalid_bounds) or np.any(parents < 0):
        raise ValueError("Focal intervals require valid site bounds and parent IDs")
    order = np.lexsort((parents, -right, left))
    left = left[order]
    right = right[order]
    parents = parents[order]
    result = np.full(num_sites, -1, dtype=np.int32)
    index = 0
    site = 0
    while index < len(left):
        site = max(site, int(left[index]))
        best_end = site
        best_parent = -1
        while index < len(left) and left[index] <= site:
            end = int(right[index])
            parent = int(parents[index])
            if end > best_end or (end == best_end and parent < best_parent):
                best_end = end
                best_parent = parent
            index += 1
        if best_end > site:
            result[site:best_end] = best_parent
            site = best_end
    return result


def switch_counts(parents: np.ndarray, covered: np.ndarray) -> SwitchCounts:
    """Count focal interiors, gap interiors, and seams in one complete path."""
    if parents.shape != covered.shape or np.any(parents < 0):
        raise ValueError("Switch counting requires complete, partition-aligned parents")
    changes = parents[1:] != parents[:-1]
    focal = covered[1:] & covered[:-1]
    gap = ~covered[1:] & ~covered[:-1]
    seam = covered[1:] != covered[:-1]
    counts = SwitchCounts(
        int(np.count_nonzero(changes & focal)),
        int(np.count_nonzero(changes & gap)),
        int(np.count_nonzero(changes & seam)),
    )
    assert counts.total == np.count_nonzero(changes)
    return counts


def comparison_metrics(
    full: np.ndarray, stitched: np.ndarray, covered: np.ndarray
) -> dict:
    """Summarise switches and agreement without retaining cohort-wide site arrays.

    All four agreement fractions use the entire inference-site axis as their
    denominator. Agreement compares original-reference parent IDs, not alleles.
    Extra switch fraction is the net difference divided by stitched switches;
    it is undefined (None) when the stitched path has no parent changes.
    """
    full_counts = switch_counts(full, covered)
    stitched_counts = switch_counts(stitched, covered)
    disagreement = full != stitched
    focal_disagreement = int(np.count_nonzero(covered & disagreement))
    gap_disagreement = int(np.count_nonzero(~covered & disagreement))
    focal_sites = int(np.count_nonzero(covered))
    gap_sites = len(covered) - focal_sites
    extra = stitched_counts.total - full_counts.total
    metrics = {
        "full_switches": full_counts.total,
        "stitched_switches": stitched_counts.total,
        "num_extra_switches": extra,
        "extra_switch_fraction": extra / stitched_counts.total
        if stitched_counts.total > 0
        else None,
        "focal_sites": focal_sites,
        "gap_sites": gap_sites,
        "focal_disagreement_sites": focal_disagreement,
        "gap_disagreement_sites": gap_disagreement,
        "fraction_parent_agreement": float(np.mean(~disagreement)),
        "parents_identical": bool(not np.any(disagreement)),
        "focal_agree_fraction": (focal_sites - focal_disagreement) / len(covered),
        "focal_disagree_fraction": focal_disagreement / len(covered),
        "gap_agree_fraction": (gap_sites - gap_disagreement) / len(covered),
        "gap_disagree_fraction": gap_disagreement / len(covered),
    }
    for method, counts in (("full", full_counts), ("stitched", stitched_counts)):
        metrics[f"{method}_focal_switches"] = counts.focal
        metrics[f"{method}_gap_switches"] = counts.gap
        metrics[f"{method}_seam_switches"] = counts.seam
    return metrics
