"""Checks for classification direction and one-sided validation geometry."""

import types

import error_estimation
import error_validation
import numpy as np
import zarr


class TestMismatchClassification:
    def test_confusion_matrix_and_roc_direction(self):
        counts = np.array([[0, 1, 2, 3]])
        summary = error_estimation.MismatchSummary(
            np.array([100.0]), counts, np.zeros_like(counts)
        )
        truth = np.array([True, True, False, False])
        record = error_validation.evaluate_mismatch_cutoffs(
            summary,
            truth,
            cutoff_proportions=[0.5],
            L_mismatch_trim_values=[100.0],
        )[0]

        assert (record["tp"], record["fp"], record["fn"], record["tn"]) == (
            2, 0, 0, 2
        )
        assert record["tpr"] == 1
        assert record["fpr"] == 0
        assert record["tnr"] == 1
        assert record["precision"] == 1
        assert record["npv"] == 1
        assert record["accuracy"] == 1
        fpr, tpr, auroc = error_validation.mismatch_roc(counts[0], truth)
        assert fpr[0] == tpr[0] == 0
        assert fpr[-1] == tpr[-1] == 1
        assert auroc == 1


class TestCumulativeMismatchProfiles:
    def test_fixed_side_and_first_mismatch_censoring(self, tmp_path):
        positions = np.arange(0, 601, 20, dtype=float)
        genotype = np.zeros((len(positions), 2, 1), dtype=np.int8)
        focal = np.array([5, 15, 25])
        genotype[focal, :, 0] = 1
        genotype[[4, 8, 9, 14], 0, 0] = 1
        path = tmp_path / "simulation.zarr"
        group = zarr.open_group(path, mode="w")
        group.create_array("variant_position", data=positions)
        group.create_array("call_genotype", data=genotype, chunks=(4, 2, 1))

        doubletons = error_estimation.Doubletons(
            site_indices=focal,
            positions=positions[focal],
            sample_indices=np.array([[0, 1]] * 3),
            ploidy_indices=np.zeros((3, 2), dtype=int),
            num_eligible=3,
        )
        config = types.SimpleNamespace(window_sizes=np.array([20.0, 100.0]))
        estimate = types.SimpleNamespace(
            inference_path=str(path), config=config, doubletons=doubletons
        )
        profiles = error_validation.cumulative_mismatch_profiles(
            estimate, path, distances=np.array([20.0, 60.0, 100.0])
        )

        np.testing.assert_array_equal(profiles.left_count[:, 0], [1, 1, 1])
        np.testing.assert_array_equal(profiles.right_count[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(profiles.clean_count[:, 0], [1, 1, 1])
        np.testing.assert_array_equal(profiles.dirty_count[:, 0], [0, 1, 2])
        assert profiles.max_first_mismatch_distance[0] == 60
        assert not profiles.max_first_mismatch_censored[0]
        assert profiles.max_first_mismatch_distance[1] == 100
        assert profiles.max_first_mismatch_censored[1]
        assert profiles.left_is_clean[2]
        np.testing.assert_array_equal(profiles.is_true_doubleton, [True] * 3)
