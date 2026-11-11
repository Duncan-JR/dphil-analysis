"""Signed-side quantiles and strict negative-tail fractions in paired panels."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import ch3_plot


@pytest.fixture
def signed_sides():
    rows = []
    for version in ("0.4.1", "0.5.1"):
        for multiplier in (0, 1):
            for side_index, value in enumerate((-20e6, -10e6, 0, 11e6)):
                rows.append(dict(version=version, geno_multiplier=multiplier, overshoot=value,
                                 true_focal_site=side_index // 2,
                                 side="left" if side_index % 2 == 0 else "right"))
    return pd.DataFrame(rows)


class TestOvershootSummary:
    def test_signed_sides_are_pooled_and_cutoff_is_strict(self, signed_sides):
        summary = ch3_plot.summarise_overshoots(signed_sides, "geno_multiplier", (0, 1))
        assert summary.geno_multiplier.tolist() == [0, 0, 1, 1]
        assert summary.version.tolist() == ["0.4.1", "0.5.1"] * 2
        assert summary.num_sides.tolist() == [4] * 4
        assert summary.q25_bp.tolist() == [-12.5e6] * 4
        assert summary.median_bp.tolist() == [-5e6] * 4
        assert summary.q75_bp.tolist() == [2.75e6] * 4
        assert summary.fraction_below_threshold.tolist() == [0.25] * 4

    def test_panels_align_versions_and_bars_start_at_zero(self, signed_sides, monkeypatch):
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: fig)
        summary = ch3_plot.summarise_overshoots(signed_sides, "geno_multiplier", (0, 1))
        # Distinct fractions make ordering mistakes visible.
        summary["fraction_below_threshold"] = [0.25, 0.5, 0.1, 0.2]
        fig = ch3_plot.plot_overshoot_summary(
            summary, "geno_multiplier", (0, 1), {0: "No errors", 1: "1x"}, "test",
        )
        for row in range(2):
            central, tail = fig.axes[2 * row:2 * row + 2]
            intervals = central.lines[:4:2]
            medians = central.lines[1:4:2]
            for version_index, (bar, interval, median) in enumerate(zip(tail.patches, intervals, medians)):
                assert bar.get_x() == 0
                assert bar.get_width() == summary.fraction_below_threshold.iloc[2 * row + version_index]
                assert bar.get_y() + bar.get_height() / 2 == median.get_ydata()[0]
                assert interval.get_xdata().tolist() == [-12.5, 2.75]
                assert median.get_xdata().tolist() == [-5]
            assert tail.get_xlim()[0] == 0
            assert central.get_ylabel() == ["No errors", "1x"][row]
        plt.close(fig)
