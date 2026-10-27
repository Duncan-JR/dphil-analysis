"""Verify overlap-weighted error burden and whole-comparison denominators."""

import ch3_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def overlap_csv(tmp_path):
    rows = []
    for version, errors in zip(("0.4.1", "0.5.1"), ([1, 0, 2], [0, 0, 1])):
        for site, (frequency, overlap, num_errors) in enumerate(zip(
            [0.01, 0.02, 1.0], [1, 999, 100], errors,
        )):
            for side in ("left", "right"):
                rows.append(dict(
                    replicate=0, version=version, true_focal_site=site,
                    allele_frequency=frequency, num_errors=num_errors,
                    num_should_be_0=num_errors if site != 2 else 0,
                    num_should_be_1=num_errors if site == 2 else 0,
                    overlap_span=overlap, side=side,
                    geno_multiplier=0, phase_ser=0, mispol_rate=0,
                ))
    for profile in ("geno_multiplier", "phase_ser", "mispol_rate"):
        row = rows[0].copy()
        row[profile] = 1
        rows.append(row)
    path = tmp_path / "aggregate.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def overlap_comparisons(overlap_csv):
    return ch3_plot.load_ancestor_overlap_errors(overlap_csv)


@pytest.fixture
def directional_comparisons():
    return pd.DataFrame({
        "version": ["0.5.1"] * 5 + ["0.4.1"],
        "replicate": [0] * 6,
        "true_focal_site": range(6),
        "allele_frequency": [0.01, 0.01, 0.01, 0.5, 1, 0.9],
        "overlap_span": [1, 9, 90, 500, 500, 1],
        "num_errors": [1, 1, 1, 1, 2, 198],
        "num_should_be_0": [1, 1, 0, 1, 1, 99],
        "num_should_be_1": [0, 0, 1, 0, 1, 99],
    })


class TestAncestorOverlapErrors:
    def test_loader_counts_whole_overlap_once_and_selects_error_free_profiles(self, overlap_comparisons):
        assert len(overlap_comparisons) == 6
        assert overlap_comparisons.groupby("version").num_errors.sum().to_dict() == {
            "0.4.1": 3, "0.5.1": 1,
        }
        assert overlap_comparisons.groupby("version").overlap_span.sum().tolist() == [1100, 1100]

    def test_ratio_of_sums_empty_bins_and_inclusive_right_endpoint(self, overlap_comparisons):
        summary = ch3_plot.summarise_ancestor_overlap_errors(
            overlap_comparisons, cutoffs=[0, 0.1, 0.5, 1],
        )
        old = summary.loc[summary.version == "0.4.1"].set_index("bin")
        new = summary.loc[summary.version == "0.5.1"].set_index("bin")
        assert old.loc[0, "error_burden"] == pytest.approx(1 / 1000)
        assert old.loc[0, "error_free_fraction"] == 0.5
        assert new.loc[0, "error_free_fraction"] == 1
        assert old.num_ancestors.tolist() == [2, 0, 1]
        assert np.isnan(old.loc[1, "error_burden"])
        assert np.isnan(old.loc[1, "error_free_fraction"])
        assert old.loc[2, "error_burden"] == pytest.approx(2 / 100)
        assert old.loc[2, "error_free_fraction"] == 0

    def test_duplicate_side_rows_are_rejected(self, overlap_comparisons):
        repeated = pd.concat([overlap_comparisons, overlap_comparisons.iloc[:1]])
        with pytest.raises(ValueError, match="one overlap comparison"):
            ch3_plot.summarise_ancestor_overlap_errors(repeated)

    def test_three_compact_panels_use_grouped_rates_and_shared_counts(self, overlap_comparisons, monkeypatch):
        summary = ch3_plot.summarise_ancestor_overlap_errors(
            overlap_comparisons, cutoffs=[0, 0.1, 0.5, 1],
        )
        captured = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: captured.append(fig))
        ch3_plot.plot_ancestor_overlap_errors(summary)
        fig = captured[0]
        assert len(fig.axes) == 3
        assert len(fig.axes[0].patches) == 6
        assert len(fig.axes[1].patches) == 6
        assert [bar.get_height() for bar in fig.axes[2].patches] == [2, 0, 1]
        assert fig.axes[0].get_ylim() == (0, 1)
        assert fig.axes[1].get_yscale() == "log"
        assert fig.axes[1].patches[0].get_height() == pytest.approx(1 / 1000)
        plt.close(fig)


class TestAncestorErrorTypes:
    def test_direction_mapping_fixed_bins_and_pooled_denominators(self, directional_comparisons):
        config = ch3_plot.PlotConfig(ancestor_error_bins=2)
        summary = ch3_plot.summarise_ancestor_error_types(directional_comparisons, config)
        assert summary.num_ancestors.tolist() == [3, 2]
        np.testing.assert_allclose(summary.daf, [0.25, 0.75])
        assert summary.overlap_bp.tolist() == [100, 1000]
        # Stored num_should_be_0 is true 1 / inferred 0, so it should be 1.
        assert summary.should_be_1_errors.tolist() == [2, 2]
        assert summary.should_be_0_errors.tolist() == [1, 1]
        np.testing.assert_allclose(summary.should_be_1_burden, [0.02, 0.002])
        np.testing.assert_allclose(summary.should_be_0_burden, [0.01, 0.001])

    def test_20_fixed_bins_include_boundaries_and_leave_empty_rates_undefined(self, directional_comparisons):
        frame = pd.concat([directional_comparisons.iloc[:1]] * 5, ignore_index=True)
        frame["true_focal_site"] = np.arange(len(frame))
        frame["allele_frequency"] = [0, 0.05, 0.05, 0.95, 1]
        summary = ch3_plot.summarise_ancestor_error_types(frame)
        assert len(summary) == 20
        assert summary.num_ancestors.sum() == 5
        assert summary.loc[0, "num_ancestors"] == 1
        assert summary.loc[1, "num_ancestors"] == 2
        assert summary.loc[19, "num_ancestors"] == 2
        assert summary.loc[2, "num_ancestors"] == 0
        assert np.isnan(summary.loc[2, "should_be_0_burden"])
        assert np.isnan(summary.loc[2, "should_be_1_burden"])
        np.testing.assert_allclose(summary.bin_right - summary.bin_left, 0.05)
        np.testing.assert_allclose(summary.daf, np.arange(20) * 0.05 + 0.025)

    def test_plot_uses_genotype_transition_palette(self, directional_comparisons, monkeypatch):
        config = ch3_plot.PlotConfig(ancestor_error_bins=2)
        summary = ch3_plot.summarise_ancestor_error_types(directional_comparisons, config)
        captured = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: captured.append(fig))
        ch3_plot.plot_ancestor_error_types(summary, config=config)
        fig = captured[0]
        ax = fig.axes[0]
        colors = plt.get_cmap(config.genotype_transition_cmap)(config.genotype_color_positions)
        assert len(ax.lines) == 2
        for line, expected_color in zip(ax.lines, colors):
            np.testing.assert_allclose(line.get_color(), expected_color)
        assert ax.lines[0].get_label() == "Should be 0 (0 → 1)"
        assert ax.lines[1].get_label() == "Should be 1 (1 → 0)"
        assert ax.get_xlabel() == "Derived allele frequency"
        assert ax.get_yscale() == "log"
        plt.close(fig)
