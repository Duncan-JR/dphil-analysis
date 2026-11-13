"""Small known-answer checks for calculations used by the final CH3 workflow."""

import numpy as np
import pandas as pd
import pytest

from ch3 import thesis_utils


class TestThesisCalculations:
    @pytest.fixture
    def overlap_comparisons(self):
        return pd.DataFrame(
            {
                "replicate": [0, 0, 0, 0],
                "version": ["0.4.1", "0.4.1", "0.5.1", "0.5.1"],
                "true_focal_site": [0, 1, 0, 1],
                "allele_frequency": [0.1, 0.2, 0.1, 0.2],
                "num_errors": [1, 9, 0, 20],
                "num_should_be_0": [1, 6, 0, 8],
                "num_should_be_1": [0, 3, 0, 12],
                "overlap_span": [100, 900, 100, 900],
                "overlap_site_span": [10, 990, 10, 990],
            }
        )

    def test_overlap_rates_pool_site_denominators(self, overlap_comparisons):
        summary = thesis_utils.summarise_ancestor_overlap_errors(
            overlap_comparisons, cutoffs=[0, 1]
        ).set_index("version")
        assert summary.loc["0.4.1", "mismatch_fraction"] == pytest.approx(10 / 1000)
        assert summary.loc["0.5.1", "mismatch_fraction"] == pytest.approx(20 / 1000)
        assert summary.loc["0.5.1", "error_free_fraction"] == 0.5
        # The source writer's direction names are reversed; output names use truth.
        assert summary.loc["0.4.1", "should_be_0_errors"] == 3
        assert summary.loc["0.4.1", "should_be_1_errors"] == 7

    @pytest.fixture
    def overshoots(self):
        return pd.DataFrame(
            {
                "version": ["0.4.1"] * 4 + ["0.5.1"] * 4,
                "geno_multiplier": [1] * 8,
                "side": ["left", "right"] * 4,
                "overshoot": [-11, -10, 0, 5, -10, 0, 2, 10],
            }
        )

    def test_overshoots_pool_sides_and_use_strict_tail(self, overshoots):
        config = thesis_utils.AnalysisConfig(overshoot_summary_threshold_bp=-10)
        summary = thesis_utils.summarise_overshoots(
            overshoots, "geno_multiplier", [1], config
        ).set_index("version")
        assert summary.loc["0.4.1", "num_sides"] == 4
        assert summary.loc["0.4.1", "median_bp"] == -5
        assert summary.loc["0.4.1", "q25_bp"] == -10.25
        assert summary.loc["0.4.1", "fraction_below_threshold"] == 0.25
        assert summary.loc["0.5.1", "median_bp"] == 1
        assert summary.loc["0.5.1", "fraction_below_threshold"] == 0

    @pytest.fixture
    def histogram_sides(self):
        return pd.DataFrame(
            {
                "version": ["0.4.1"] * 4 + ["0.5.1"] * 4,
                "geno_multiplier": [1] * 8,
                "overshoot": [-2, 1, 2, 2, -2, -2, -1, 2],
            }
        )

    def test_histogram_changes_are_relative_counts(self, histogram_sides):
        config = thesis_utils.AnalysisConfig(bp_per_mb=1)
        changes = thesis_utils.histogram_count_changes(
            histogram_sides, "geno_multiplier", [1], config, bins=4
        )
        assert changes.iloc[0].relative_increase_pct == 100
        assert changes.iloc[1].old_count == 0
        assert changes.iloc[1].new_count == 1
        assert np.isnan(changes.iloc[1].relative_increase_pct)
        assert changes.iloc[-1].relative_increase_pct == pytest.approx(-200 / 3)

    @pytest.fixture
    def performance_runs(self):
        stages = ("generate_ancestors", "match_ancestors", "match_samples")
        elapsed = {
            (0, "0.4"): [10, 20, 30],
            (0, "0.5"): [5, 5, 5],
            (1, "0.4"): [20, 20, 20],
            (1, "0.5"): [20, 20, 20],
        }
        rows = []
        for (rep, version), times in elapsed.items():
            for stage, seconds in zip(stages, times, strict=True):
                rows.append(
                    {
                        "num_samples": 4,
                        "version": version,
                        "rep": rep,
                        "command": stage,
                        "elapsed_time": seconds,
                    }
                )
        return pd.DataFrame(rows).set_index(
            ["num_samples", "version", "rep", "command"]
        )

    def test_performance_pairs_runs_before_averaging(self, performance_runs):
        summary = thesis_utils.summarise_performance(
            performance_runs, thesis_utils.PerformanceConfig()
        )
        stages = summary.stages.set_index("command")
        assert stages.loc["match_samples", "speedup"] == 3.5
        # Per-run total ratios are 4 and 1; ratio of the pooled means is 1.6.
        assert summary.totals.iloc[0].speedup == 2.5
        assert summary.totals.iloc[0]["0.4_hours"] == pytest.approx(60 / 3600)
