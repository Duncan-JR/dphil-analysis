"""Check sparse sweeps against explicit site coverage and source deduplication."""

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
    chunks = pl.DataFrame(
        {
            "haplotype_index": [0, 0, 0, 1, 0, 0, 0, 1],
            "focal_ac": [2, 3, 6, 2, 2, 3, 6, 2],
            "max_mismatches": [0, 0, 0, 0, 1, 1, 1, 1],
            "left_site_index": [0, 3, 5, 0, 0, 2, 4, 0],
            "right_site_index": [4, 5, 10, 10, 5, 7, 10, 10],
        }
    )
    return ch5_analysis.CoverageData(
        "fixture", 10, haplotypes, chunks, [0, 1], 10, 10000
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
            chunks = coverage_data.chunks.filter(
                (pl.col("haplotype_index") == index)
                & (pl.col("max_mismatches") == budget)
                & (pl.col("focal_ac") <= row["focal_ac"])
            )
            oracle = np.zeros(coverage_data.num_sites, dtype=int)
            for chunk in chunks.iter_rows(named=True):
                oracle[chunk["left_site_index"] : chunk["right_site_index"]] += 1
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

    def test_all_empty_chunks_preserve_haplotypes(self, coverage_data):
        coverage_data.chunks = coverage_data.chunks.head(0)
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
    for directory in ["ancestors", "focal_ancestors", "dataframes"]:
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
    chunks = pl.DataFrame(
        {
            "sample_id": ["s0"] * 5,
            "ploidy_index": [0] * 5,
            "ancestor_index": [0, 0, 0, 0, 1],
            "focal_site_index": [2, 0, 2, 0, 1],
            "focal_ac": [2, 2, 2, 2, 3],
            "max_mismatches": [0, 0, 1, 1, 0],
            "left_site_index": [2, 0, 1, 0, 1],
            "right_site_index": [4, 1, 4, 2, 3],
        }
    )
    chunks.write_csv(
        tmp_path / "dataframes" / f"{dataset}_inferred_focal_ancestor_chunks.csv"
    )
    return tmp_path


class TestLoadCoverageData:
    def test_one_consistent_focal_per_ancestor_and_budget(self, dataset_dir):
        data = ch5_analysis.load_coverage_data(dataset_dir, "fixture", 200, [0, 1])
        assert data.num_sites == 4
        assert data.haplotypes.height == 3
        assert data.chunks.height == 3
        first_ancestor = data.chunks.filter(pl.col("ancestor_index") == 0)
        assert first_ancestor["focal_site_index"].to_list() == [0, 0]
        actual = ch5_analysis.summarise_coverage(data, np.array([2, 3]), workers=1)
        sample = actual.filter(
            (pl.col("sample_id") == "s0") & (pl.col("ploidy_index") == 0)
        )
        assert sample["fraction_covered"].to_list() == [0.25, 0.75]
        assert sample["max_coverage"].to_list() == [1, 1]

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
