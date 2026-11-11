"""Named combined-profile selection and the four-panel thesis comparison."""

import ch3_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def comparisons(tmp_path):
    rows = []
    for profile, multiplier in [("none", 1), ("all", 3), ("geno1x", 50)]:
        for version, errors in [("0.4.1", [1, 0]), ("0.5.1", [0, 0])]:
            for rep in [0, 1]:
                for site, (span, error) in enumerate(zip([1, 999], errors)):
                    for side in ["left", "right"]:
                        rows.append(dict(
                            error_profile=profile, version=version, replicate=rep,
                            true_focal_site=site, side=side,
                            geno_multiplier=int(profile != "none"),
                            space_per_switch=2e6 if profile == "all" else 0,
                            phase_ser=0.000761 if profile == "all" else 0,
                            mispol_rate=0.01 if profile == "all" else 0,
                            focal_site_mispolarised=False,
                            num_errors=error * multiplier, overlap_span=span,
                            overlap_site_span=10 if site == 0 else 90,
                            num_should_be_0=error * multiplier, num_should_be_1=0,
                            overshoot=(-1000000 if side == "left" else 10000) * multiplier,
                        ))
    flipped_rows = []
    for row in rows:
        if row["error_profile"] == "all":
            flipped = row.copy()
            flipped.update(true_focal_site=2, focal_site_mispolarised=True,
                           overshoot=20000000, num_errors=1000)
            flipped_rows.append(flipped)
    rows.extend(flipped_rows)
    path = tmp_path / "aggregate.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return ch3_plot.load_combined_error_comparisons(path)


class TestCombinedErrors:
    def test_named_profiles_and_both_sides_are_retained(self, comparisons):
        assert set(comparisons.error_profile) == {"none", "all"}
        assert len(comparisons) == 32
        assert not comparisons.focal_site_mispolarised.any()
        assert comparisons.groupby(["error_profile", "version", "side"]).size().eq(4).all()

    def test_whole_overlap_is_counted_once_and_rates_are_ratios_of_sums(self, comparisons):
        summary = ch3_plot.summarise_combined_overlap_errors(comparisons)
        assert summary.num_ancestors.tolist() == [4] * 4
        assert summary.overlap_bp.tolist() == [2000] * 4
        old = summary.loc[summary.version == "0.4.1"].set_index("error_profile")
        assert old.loc["none", "mismatch_fraction"] == pytest.approx(0.01)
        assert old.loc["all", "mismatch_fraction"] == pytest.approx(0.03)
        assert old.error_free_fraction.eq(0.5).all()
        repeated = pd.concat([comparisons, comparisons.iloc[:1]])
        with pytest.raises(ValueError, match="one overlap comparison per profile"):
            ch3_plot.summarise_combined_overlap_errors(repeated)

    def test_composite_panels_counts_colours_and_shared_bins(self, comparisons, monkeypatch):
        figures = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: figures.append(fig))
        ch3_plot.plot_combined_errors(comparisons, num_bins=7)
        fig = figures[0]
        assert len(fig.axes) == 4
        assert fig.axes[1].get_yscale() == "linear"
        assert fig.axes[1].get_ylim()[0] == 0
        assert fig.axes[1].get_ylabel() == "Pooled mismatch fraction"
        assert sum(bar.get_height() for bar in fig.axes[1].patches) == pytest.approx(0.04)
        for ax, label in zip(fig.axes, "ABCD"):
            assert label in [text.get_text() for text in ax.texts]
        palette = plt.get_cmap("viridis")(np.linspace(0, 0.8, 2))
        for ax in fig.axes[2:]:
            np.testing.assert_allclose([step.get_edgecolor() for step in ax.patches], palette)
        for step in fig.axes[2].patches:
            assert np.nansum(step.get_data().values) == 8
            assert len(step.get_data().values) == 7
        np.testing.assert_allclose(
            fig.axes[2].patches[0].get_data().edges,
            fig.axes[3].patches[1].get_data().edges,
        )
        fig.canvas.draw()
        plt.close(fig)
