"""Focal-polarisation groups retain one correctly weighted overlap per site."""

import ch3_plot
import matplotlib.pyplot as plt
import pandas as pd
import pytest


@pytest.fixture
def mispolarisation_csv(tmp_path):
    rows = []
    profiles = [
        (0, False, 0, 0, 10),
        (0, False, 1, 4, 990),
        (0.1, True, 1, 2, 100),
        (0.1, False, 0, 1, 900),
        (0.01, True, 1, 3, 200),
        (0.01, False, 0, 2, 800),
        (0.05, True, 1, 99, 100),
    ]
    for version in ("0.4.1", "0.5.1"):
        for rate, flipped, site, errors, overlap in profiles:
            for side in ("left", "right"):
                rows.append(dict(
                    version=version, replicate=0, true_focal_site=site,
                    inferred_node=42, side=side, geno_multiplier=0,
                    phase_ser=0, mispol_rate=rate, focal_site_mispolarised=flipped,
                    num_errors=errors, overlap_span=overlap,
                    allele_frequency=0.05 if site == 0 else 0.95,
                    focal_position=(site + 1) * 10,
                    overlap_site_span=10,
                    num_should_be_0=errors, num_should_be_1=0,
                ))
    for parameter in ("geno_multiplier", "phase_ser"):
        mixed = rows[0].copy()
        mixed[parameter] = 1
        rows.append(mixed)
    path = tmp_path / "aggregate.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestMispolarisationErrors:
    def test_true_count_table_counts_each_focal_once_in_each_pure_profile(self):
        rows = []
        for version in ("0.4.1", "0.5.1"):
            for rate, counts in [(0.01, [1, 2, 2, 3]), (0.1, [1, 1, 2, 2, 3])]:
                for count in counts:
                    rows.append(dict(
                        version=version, mispol_rate=rate, geno_multiplier=0,
                        phase_ser=0, focal_site_mispolarised=True, true_derived_count=count,
                    ))
        mixed = rows[-1].copy()
        mixed["geno_multiplier"] = 1
        unflipped = rows[-1].copy()
        unflipped["focal_site_mispolarised"] = False
        rows.extend([mixed, unflipped])
        config = ch3_plot.PlotConfig(mispol_rates=(0, 0.01, 0.1))
        data = ch3_plot.ChapterData(pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame(), config)
        table = ch3_plot.summarise_mispolarised_focal_counts(data)
        assert table.mispol_rate.tolist() == [0.01, 0.1]
        assert table.num_mispol_focal_sites.tolist() == [4, 5]
        assert table.num_true_singletons.tolist() == [1, 2]
        assert table.num_true_doubletons.tolist() == [2, 2]
        assert table.proportion_true_singletons.tolist() == pytest.approx([0.25, 0.4])
        assert table.proportion_true_doubletons.tolist() == pytest.approx([0.5, 0.4])

    def test_pure_profiles_split_by_own_focal_state_and_pool_once(self, mispolarisation_csv):
        comparisons = ch3_plot.load_mispolarisation_overlap_errors(mispolarisation_csv)
        assert len(comparisons) == 12
        summary = ch3_plot.summarise_mispolarisation_overlap_errors(comparisons)
        old = summary.loc[summary.version == "0.4.1"].set_index("group")
        assert old.num_ancestors.to_dict() == {
            "none": 2, "0.1_mispolarised": 1, "0.1_correctly_polarised": 1,
            "0.01_mispolarised": 1, "0.01_correctly_polarised": 1,
        }
        assert old.loc["none", "error_free_fraction"] == 0.5
        assert old.loc["none", "mismatch_fraction"] == pytest.approx(4 / 20)
        assert old.loc["0.1_mispolarised", "mismatch_fraction"] == pytest.approx(2 / 10)
        assert old.loc["0.1_correctly_polarised", "mismatch_fraction"] == pytest.approx(1 / 10)
        assert old.loc["0.01_mispolarised", "mismatch_fraction"] == pytest.approx(3 / 10)
        assert old.loc["0.01_correctly_polarised", "mismatch_fraction"] == pytest.approx(2 / 10)
        assert (summary.should_be_1_mismatch_fraction == summary.mismatch_fraction).all()

    def test_duplicate_overlap_is_rejected(self, mispolarisation_csv):
        comparisons = ch3_plot.load_mispolarisation_overlap_errors(mispolarisation_csv)
        repeated = pd.concat([comparisons, comparisons.iloc[:1]])
        with pytest.raises(ValueError, match="one overlap comparison"):
            ch3_plot.summarise_mispolarisation_overlap_errors(repeated)

    @pytest.mark.parametrize("num_bins", [5, 10])
    def test_daf_burden_pools_denominators_uses_truth_and_keeps_flipped_singletons(self, num_bins):
        comparisons = pd.DataFrame({
            "version": ["0.5.1"] * 7 + ["0.4.1"],
            "replicate": [0] * 8,
            "true_focal_site": [0, 1, 2, 0, 1, 2, 3, 0],
            "focal_position": [10, 20, 30, 10, 20, 30, 40, 10],
            "mispol_rate": [0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0],
            "focal_site_mispolarised": [False] * 5 + [True, True, False],
            "allele_frequency": [0.01, 0.02, 1, 0.8, 0.9, 0, 0.97, 0.01],
            "num_errors": [0, 2, 1, 1, 0, 1, 1, 99],
            "overlap_span": [10, 90, 100, 20, 80, 1, 2, 1],
            "overlap_site_span": [1, 9, 5, 2, 8, 1, 2, 1],
        })
        truth = pd.DataFrame({
            "replicate": [0] * 4,
            "focal_position": [10, 20, 30, 40],
            "true_allele_frequency": [0.01, 0.02, 1, 0.03],
        })
        summary = ch3_plot.summarise_mispolarisation_mismatch_by_daf(
            comparisons, truth, num_bins=num_bins,
        )
        binned = summary.set_index(["focal_group", "bin"])
        last_bin = num_bins - 1
        assert binned.loc[("none", 0), "mismatch_fraction"] == pytest.approx(2 / 10)
        assert binned.loc[("correctly_polarised", 0), "mismatch_fraction"] == pytest.approx(1 / 10)
        assert binned.loc[("none", 0), "mismatch_fraction"] == pytest.approx(2 / 10)
        assert binned.loc[("none", last_bin), "mismatch_fraction"] == pytest.approx(1 / 5)
        assert pd.isna(binned.loc[("correctly_polarised", last_bin), "mismatch_fraction"])
        assert binned.loc[("correctly_polarised", last_bin), "num_ancestors"] == 0
        assert binned.loc[("mispolarised", 0), "num_ancestors"] == 1
        assert binned.loc[("mispolarised", 0), "mismatch_fraction"] == pytest.approx(1 / 2)
        assert binned.loc[("mispolarised", last_bin), "num_ancestors"] == 1
        assert binned.num_ancestors.sum() == 7
        assert len(summary) == 3 * num_bins

    def test_per_site_plot_orders_three_groups_and_defaults_to_log(self, mispolarisation_csv, monkeypatch):
        comparisons = ch3_plot.load_mispolarisation_overlap_errors(mispolarisation_csv)
        truth = pd.DataFrame({
            "replicate": [0, 0], "focal_position": [10, 20],
            "true_allele_frequency": [0.05, 0.95],
        })
        summary = ch3_plot.summarise_mispolarisation_mismatch_by_daf(comparisons, truth, num_bins=5)
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: fig)
        fig = ch3_plot.plot_mispolarisation_mismatch_by_daf(summary)
        burden, counts = fig.axes
        assert burden.patches[4].get_height() == pytest.approx(4 / 10)
        assert burden.patches[9].get_height() == pytest.approx(2 / 10)
        assert burden.patches[10].get_height() == pytest.approx(1 / 10)
        assert burden.get_ylabel() == "Pooled mismatch fraction"
        assert burden.get_yscale() == counts.get_yscale() == "log"
        assert counts.get_xlabel() == "True focal derived allele frequency"
        assert len(counts.patches) == 15
        assert [label.get_text() for label in fig.legends[0].texts] == [
            "No error", "Mispol focal sites", "Non-mispol focal sites",
        ]
        assert burden.texts[0].get_text() == "A"
        assert counts.texts[0].get_text() == "B"
        plt.close(fig)
