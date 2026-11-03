"""Signed boxplot axes retain zeros and leave the underlying values unchanged."""

import ch3_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def chapter_data():
    rows = []
    for version in ("0.4.1", "0.5.1"):
        for geno, phase, mispol in [(0, 0, 0), (1, 0, 0), (0, 0.001, 0), (0, 0, 0.1)]:
            for site, frequency in enumerate([0.01, 0.8]):
                for side, overshoot in [("left", -10000), ("right", 0)]:
                    rows.append(dict(
                        version=version, replicate=0, true_focal_site=site,
                        geno_multiplier=geno, phase_ser=phase, mispol_rate=mispol,
                        allele_frequency=frequency, true_allele_frequency=frequency,
                        side=side, overshoot=overshoot, true_span=100000,
                        inferred_span=110000,
                    ))
    frame = pd.DataFrame(rows)
    config = ch3_plot.PlotConfig(
        genotype_multipliers=(0, 1), phase_rates=(0, 0.001), mispol_rates=(0, 0.1),
    )
    spans = frame.loc[frame.side == "left"].copy()
    return ch3_plot.ChapterData(frame, spans, pd.DataFrame(), config)


@pytest.fixture
def captured_figures(monkeypatch):
    figures = []
    monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: figures.append(fig))
    yield figures
    for fig in figures:
        plt.close(fig)


class TestBoxplotScales:
    @pytest.mark.parametrize("plot", [
        ch3_plot.plot_ancestor_overshoot,
        ch3_plot.plot_genotype_overshoot_box,
        ch3_plot.plot_mispolarisation_overshoot_box,
        ch3_plot.plot_phasing_overshoot_box,
    ])
    @pytest.mark.parametrize("y_log", [False, True])
    def test_overshoot_scale_retains_negative_and_zero_values(
        self, chapter_data, captured_figures, plot, y_log,
    ):
        original = chapter_data.overshoots.copy()
        plot(chapter_data, y_log=y_log)
        ax = captured_figures[0].axes[0]
        assert ax.get_yscale() == ("symlog" if y_log else "linear")
        assert ax.get_ylim()[0] < 0
        assert ax.get_ylim()[1] > 0
        assert np.isfinite(ax.yaxis.get_transform().transform([-0.01, 0, 0.01])).all()
        pd.testing.assert_frame_equal(chapter_data.overshoots, original)
        if y_log:
            assert ax.yaxis.get_transform().linthresh == pytest.approx(0.001)
            assert 0 in ax.get_yticks()

    def test_demography_uses_same_symlog_scale(self, chapter_data, captured_figures):
        frame = chapter_data.overshoots.copy()
        frame["model"] = "model"
        ch3_plot.plot_demography_overshoot_box(frame, {"model": "Model"}, y_log=True)
        assert captured_figures[0].axes[0].get_yscale() == "symlog"

    @pytest.mark.parametrize("y_log", [False, True])
    def test_positive_spans_use_standard_log_or_linear_scale(
        self, chapter_data, captured_figures, y_log,
    ):
        ch3_plot.plot_ancestor_spans(chapter_data, y_log=y_log)
        assert captured_figures[0].axes[0].get_yscale() == ("log" if y_log else "linear")
