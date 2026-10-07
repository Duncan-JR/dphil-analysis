"""Check sparse sweeps against explicit site coverage and raw interval loading."""

import ch5_analysis
import numpy as np
import polars as pl
import pytest
import zarr


@pytest.fixture
def coverage_data():
    haplotypes = pl.DataFrame(
        {
            "haplotype_index": [0, 1, 2],
            "sample_id": ["s0", "s0", "s1"],
            "ploidy_index": [0, 1, 0],
            "population": ["A", "A", "B"],
        }
    )
    return ch5_analysis.CoverageData(
        dataset="fixture",
        num_sites=10,
        haplotypes=haplotypes,
        offsets=np.array([0, 5, 6, 6], dtype=np.int64),
        focal_ac=np.array([6, 2, 3, 2, 1, 2], dtype=np.int64),
        left_site_index=np.array([[5, 4], [0, 0], [3, 2], [0, 0], [0, 0], [0, 0]]),
        right_site_index=np.array(
            [[10, 10], [4, 5], [5, 7], [4, 5], [10, 10], [10, 10]]
        ),
        max_ac=200,
        mismatch_budgets=[0, 1],
        first_position=10,
        last_position=10000,
    )


class TestCoverageSweep:
    @pytest.mark.parametrize("budget", [0, 1])
    def test_matches_explicit_per_site_oracle(self, coverage_data, budget):
        cutoffs = np.array([2, 3, 5, 6, 200])
        actual = ch5_analysis.summarise_coverage(
            coverage_data, cutoffs, max_mismatches=budget, workers=1
        )
        assert actual.height == 3 * len(cutoffs)
        for row in actual.iter_rows(named=True):
            sample = coverage_data.haplotypes.filter(
                (pl.col("sample_id") == row["sample_id"])
                & (pl.col("ploidy_index") == row["ploidy_index"])
            )
            index = sample["haplotype_index"].item()
            start = coverage_data.offsets[index]
            stop = coverage_data.offsets[index + 1]
            oracle = np.zeros(coverage_data.num_sites, dtype=int)
            for association in range(start, stop):
                if 2 <= coverage_data.focal_ac[association] <= row["focal_ac"]:
                    left = coverage_data.left_site_index[association, budget]
                    right = coverage_data.right_site_index[association, budget]
                    oracle[left:right] += 1
            assert row["mean_coverage"] == oracle.mean()
            assert row["min_coverage"] == oracle.min()
            assert row["max_coverage"] == oracle.max()
            assert row["q05_coverage"] == np.quantile(oracle, 0.05)
            assert row["q95_coverage"] == np.quantile(oracle, 0.95)
            assert row["fraction_covered"] == np.mean(oracle > 0)
            assert row["covered_sites"] == np.count_nonzero(oracle)
            assert row["population"] == sample["population"].item()

    def test_multiprocessing_matches_serial(self, coverage_data):
        cutoffs = np.array([2, 3, 6])
        serial = ch5_analysis.summarise_coverage(coverage_data, cutoffs, workers=1)
        parallel = ch5_analysis.summarise_coverage(coverage_data, cutoffs, workers=2)
        assert serial.equals(parallel)

    def test_all_empty_associations_preserve_haplotypes(self, coverage_data):
        coverage_data.offsets[:] = 0
        coverage_data.focal_ac = np.empty(0, dtype=np.int64)
        coverage_data.left_site_index = np.empty((0, 2), dtype=np.int64)
        coverage_data.right_site_index = np.empty((0, 2), dtype=np.int64)
        actual = ch5_analysis.summarise_coverage(
            coverage_data, np.array([2, 200]), workers=1
        )
        assert actual.height == 6
        assert actual["fraction_covered"].to_list() == [0] * 6
        assert actual["max_coverage"].to_list() == [0] * 6

    def test_random_intervals_and_quantiles(self):
        rng = np.random.default_rng(42)
        for num_sites in [1, 2, 37, 200]:
            left = rng.integers(0, num_sites, size=100)
            right = rng.integers(left + 1, num_sites + 1)
            ac = rng.integers(2, 201, size=100)
            order = np.argsort(ac)
            cutoffs = np.array([2, 3, 20, 200])
            task = ch5_analysis.SweepTask(
                0, num_sites, cutoffs, ac[order], left[order], right[order]
            )
            result = ch5_analysis._sweep_haplotype(task)
            for row in result.records:
                oracle = np.zeros(num_sites, dtype=int)
                for start, stop, count in zip(left, right, ac, strict=True):
                    if count <= row["focal_ac"]:
                        oracle[start:stop] += 1
                assert row["mean_coverage"] == oracle.mean()
                assert row["q05_coverage"] == np.quantile(oracle, 0.05)
                assert row["q95_coverage"] == np.quantile(oracle, 0.95)


@pytest.fixture
def dataset_dir(tmp_path):
    dataset = "fixture"
    for directory in [
        "ancestors",
        "focal_ancestors",
        "dataframes",
        "haplotype_intervals",
    ]:
        (tmp_path / directory).mkdir()
    panel_path = tmp_path / "ancestors" / f"{dataset}_inferred_ancestors.zarr"
    panel = zarr.open_group(panel_path, mode="w")
    panel.create_array("variant_position", data=np.array([10, 11, 500, 10000]))
    focal_path = tmp_path / "focal_ancestors" / f"{dataset}_inferred_focal_ancestors.npz"
    np.savez(focal_path, sample_id=["s0", "s0", "s1"], ploidy_index=[0, 1, 0])
    stats = pl.DataFrame(
        {
            "sample_id": ["s0", "s0", "s1", "s0"],
            "ploidy_index": [0, 1, 0, 0],
            "population": ["A", "A", "B", "A"],
        }
    )
    stats.write_csv(
        tmp_path / "dataframes" / f"{dataset}_inferred_focal_ancestor_stats.csv"
    )
    np.savez_compressed(
        tmp_path
        / "haplotype_intervals"
        / f"{dataset}_inferred_focal_ancestor_intervals.npz",
        sample_id=np.array(["s0", "s0", "s1"]),
        ploidy_index=np.array([0, 1, 0], dtype=np.int64),
        offsets=np.array([0, 2, 2, 2], dtype=np.int64),
        focal_ac=np.array([2, 3], dtype=np.int64),
        left_site_index=np.array([[0, 0], [1, 1]], dtype=np.int64),
        right_site_index=np.array([[1, 2], [3, 3]], dtype=np.int64),
        num_sites=np.int64(4),
        max_ac_cutoff=np.int64(200),
        max_mismatches=np.int64(1),
    )
    return tmp_path


class TestLoadCoverageData:
    def test_raw_associations_and_budget_columns(self, dataset_dir):
        data = ch5_analysis.load_coverage_data(dataset_dir, "fixture", 200, [0, 1])
        assert data.num_sites == 4
        assert data.haplotypes.height == 3
        np.testing.assert_array_equal(data.offsets, [0, 2, 2, 2])
        np.testing.assert_array_equal(data.focal_ac, [2, 3])
        np.testing.assert_array_equal(data.left_site_index, [[0, 0], [1, 1]])
        np.testing.assert_array_equal(data.right_site_index, [[1, 2], [3, 3]])
        assert data.mismatch_budgets == [0, 1]
        assert data.max_ac == 200
        subset = ch5_analysis.load_coverage_data(dataset_dir, "fixture", 2, [1])
        np.testing.assert_array_equal(subset.offsets, data.offsets)
        np.testing.assert_array_equal(subset.focal_ac, data.focal_ac)
        np.testing.assert_array_equal(subset.right_site_index, [[2], [3]])
        assert subset.mismatch_budgets == [1]
        with pytest.raises(ValueError, match="loaded analysis AC limit"):
            ch5_analysis.summarise_coverage(subset, np.array([2, 3]), 1, workers=1)
        actual = ch5_analysis.summarise_coverage(data, np.array([2, 3]), workers=1)
        sample = actual.filter(
            (pl.col("sample_id") == "s0") & (pl.col("ploidy_index") == 0)
        )
        assert sample["fraction_covered"].to_list() == [0.25, 0.75]
        assert sample["max_coverage"].to_list() == [1, 1]

    def test_excess_ac_is_reported(self, dataset_dir):
        with pytest.raises(ValueError, match="generation limit"):
            ch5_analysis.load_coverage_data(dataset_dir, "fixture", 201, [0])

    @pytest.mark.parametrize(
        "field,value,message",
        [
            ("offsets", np.array([0, 2, 1, 2]), "ragged offsets"),
            ("sample_id", np.array(["wrong", "s0", "s1"]), "roster"),
            ("num_sites", np.int64(5), "site count"),
            ("left_site_index", np.array([[-1, 0], [1, 1]]), "bounds must satisfy"),
            ("right_site_index", np.array([[1], [3]]), "dimensions"),
        ],
    )
    def test_invalid_archive_is_reported(self, dataset_dir, field, value, message):
        path = (
            dataset_dir
            / "haplotype_intervals"
            / "fixture_inferred_focal_ancestor_intervals.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays[field] = value
        np.savez_compressed(path, **arrays)
        with pytest.raises(ValueError, match=message):
            ch5_analysis.load_coverage_data(dataset_dir, "fixture", 200, [0])

    def test_missing_budget_is_reported(self, dataset_dir):
        with pytest.raises(ValueError, match="lacks mismatch budgets"):
            ch5_analysis.load_coverage_data(dataset_dir, "fixture", 200, [2])


class TestFocalAcCutoffs:
    @pytest.mark.parametrize("maximum", [2, 3, 10, 20, 200, 600])
    def test_integer_log_grid(self, maximum):
        cutoffs = ch5_analysis.focal_ac_cutoffs(maximum, num_bins=20)
        assert cutoffs[0] == 2
        assert cutoffs[-1] == maximum
        assert np.all(np.diff(cutoffs) > 0)
        assert len(cutoffs) >= min(20, maximum - 1)
        if maximum >= 3:
            assert cutoffs[1] == 3


class TestFocalAncestorSetSizes:
    def test_full_panel_denominator_and_empty_haplotype(self, tmp_path):
        focal_path = tmp_path / "focal.npz"
        np.savez(
            focal_path,
            sample_id=["s0", "s1", "s2"],
            ancestor_id=["a0", "a1", "a2", "a3", "a4"],
            derived_ac=[1, 2, 3, 5, 7],
            offsets=[0, 3, 3, 5],
            ancestor_index=[0, 1, 2, 1, 4],
        )
        cutoffs = np.array([2, 3, 5, 200])
        actual = ch5_analysis.summarise_focal_ancestor_set_sizes(focal_path, cutoffs)
        np.testing.assert_array_equal(
            actual["mean_num_focal_ancestors"], np.array([3, 4, 4, 5]) / 3
        )
        np.testing.assert_array_equal(
            actual["mean_proportion_included"], np.array([3, 4, 4, 5]) / 3 / 5
        )
        assert actual["num_haplotypes"].to_list() == [3] * 4
        assert actual["num_ancestors"].to_list() == [5] * 4
