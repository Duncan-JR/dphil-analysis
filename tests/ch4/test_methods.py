"""Minimal known-answer checks for retained CH4 estimator/validation methods."""

import numpy as np
import pytest
import zarr

from ch4 import error_estimation, error_validation


class TestRetainedMethods:
    @pytest.fixture
    def genotype(self, tmp_path):
        group = zarr.open_group(tmp_path / "genotypes.zarr", mode="w")
        values = np.array(
            [[0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [1, 0]],
            dtype=np.int8,
        )
        return group.create_array(
            "call_genotype", data=values[:, :, None], chunks=(2, 2, 1)
        )

    def test_window_counts_exclude_focal_and_masked_sites(self, genotype):
        site_indices = np.array([0, 2, 3, 5, 6])
        left = np.array([[1], [0]])
        right = np.array([[4], [5]])
        focal = np.array([2])
        carriers = np.array([[0, 1]])
        contributions = []
        for chunk in range(3):
            contribution = error_estimation._mismatch_chunk(
                genotype, site_indices, chunk, left, right, focal, carriers
            )
            contributions.append(contribution.counts)
        total = np.sum(contributions, axis=0)
        np.testing.assert_array_equal(total, [[1], [3]])

    @pytest.fixture
    def tied_scores(self):
        return {
            "scores": np.array([0.9, 0.5, 0.5, 0.1]),
            "truth": np.array([True, True, False, False]),
        }

    def test_ranking_metrics_keep_score_ties_together(self, tied_scores):
        scores = tied_scores["scores"]
        truth = tied_scores["truth"]
        curves = error_validation.classifier_curves(scores, truth)
        assert curves["auroc"] == 0.875
        assert curves["average_precision"] == pytest.approx(5 / 6)
        np.testing.assert_array_equal(curves["tpr"], [0, 0.5, 1, 1])
        np.testing.assert_array_equal(curves["fpr"], [0, 0, 0.5, 1])
