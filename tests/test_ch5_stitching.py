"""Check anchored interval coverage, parent selection, and switch partitions."""

import numpy as np
import pytest

import ch5_stitching


@pytest.fixture
def overlapping_intervals():
    return {
        "left": np.array([1, 3, 4, 9, 9]),
        "right": np.array([5, 8, 7, 11, 11]),
        "parents": np.array([10, 20, 30, 50, 40]),
    }


class TestFocalParentPath:
    def test_overlap_gaps_and_deterministic_ties(self, overlapping_intervals):
        actual = ch5_stitching.focal_parent_path(12, **overlapping_intervals)
        np.testing.assert_array_equal(
            actual, [-1, 10, 10, 10, 10, 20, 20, 20, -1, 40, 40, -1]
        )
        np.testing.assert_array_equal(
            ch5_stitching.site_intervals(actual < 0), [[0, 1], [8, 9], [11, 12]]
        )

    def test_union_and_supported_parent_against_site_oracle(self):
        rng = np.random.default_rng(13)
        for num_sites in [1, 7, 40]:
            left = rng.integers(0, num_sites, size=30)
            right = rng.integers(left + 1, num_sites + 1)
            parents = np.arange(30)
            actual = ch5_stitching.focal_parent_path(num_sites, left, right, parents)
            covered = np.zeros(num_sites, dtype=bool)
            for start, stop in zip(left, right, strict=True):
                covered[start:stop] = True
            np.testing.assert_array_equal(actual >= 0, covered)
            for site in np.flatnonzero(covered):
                parent = actual[site]
                assert left[parent] <= site < right[parent]
            # Independent shortest-cover DP verifies the greedy segment count.
            for start, stop in ch5_stitching.site_intervals(covered):
                costs = {int(start): 0}
                for site in range(start, stop):
                    if site not in costs:
                        continue
                    eligible = (left <= site) & (right > site)
                    for end in right[eligible]:
                        cost = costs[site] + 1
                        costs[int(end)] = min(costs.get(int(end), num_sites + 1), cost)
                segments = 1 + np.count_nonzero(
                    actual[start + 1 : stop] != actual[start : stop - 1]
                )
                assert segments == costs[int(stop)]

    def test_no_candidates_and_touching_intervals(self):
        empty = np.array([], dtype=int)
        actual = ch5_stitching.focal_parent_path(3, empty, empty, empty)
        np.testing.assert_array_equal(actual, [-1, -1, -1])
        actual = ch5_stitching.focal_parent_path(
            4, np.array([0, 2]), np.array([2, 4]), np.array([3, 4])
        )
        np.testing.assert_array_equal(actual, [3, 3, 4, 4])
        assert ch5_stitching.site_intervals(actual < 0).shape == (0, 2)


class TestComparisonMetrics:
    def test_switch_partition_and_whole_axis_fractions(self):
        covered = np.array([True, True, False, False, True, True])
        full = np.array([1, 2, 2, 3, 4, 4])
        stitched = np.array([1, 1, 2, 2, 3, 5])
        counts = ch5_stitching.switch_counts(full, covered)
        assert counts == ch5_stitching.SwitchCounts(focal=1, gap=1, seam=1)
        metrics = ch5_stitching.comparison_metrics(full, stitched, covered)
        assert metrics["full_switches"] == metrics["stitched_switches"] == 3
        assert metrics["stitched_focal_switches"] == 1
        assert metrics["stitched_gap_switches"] == 0
        assert metrics["stitched_seam_switches"] == 2
        assert metrics["extra_switch_fraction"] == 0
        assert metrics["focal_agree_fraction"] == pytest.approx(1 / 6)
        assert metrics["focal_disagree_fraction"] == pytest.approx(3 / 6)
        assert metrics["gap_agree_fraction"] == pytest.approx(1 / 6)
        assert metrics["gap_disagree_fraction"] == pytest.approx(1 / 6)

    def test_negative_and_undefined_extra_fraction(self):
        covered = np.array([False, False, False])
        full = np.array([1, 2, 3])
        stitched = np.array([1, 1, 2])
        metrics = ch5_stitching.comparison_metrics(full, stitched, covered)
        assert metrics["extra_switch_fraction"] == -1
        metrics = ch5_stitching.comparison_metrics(full, np.ones(3, dtype=int), covered)
        assert metrics["extra_switch_fraction"] is None
        assert metrics["full_focal_switches"] == metrics["full_seam_switches"] == 0

    def test_incomplete_parent_path_is_rejected(self):
        with pytest.raises(ValueError, match="complete"):
            ch5_stitching.switch_counts(np.array([-1, 2]), np.array([True, False]))
