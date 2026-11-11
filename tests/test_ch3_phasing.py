"""Check CSV-only phase illustrations and spacing-based phasing results."""

import ch3_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import zarr


@pytest.fixture
def captured(monkeypatch):
    figures = []
    monkeypatch.setattr(ch3_plot, "save_figure", lambda figure, *args: figures.append(figure))
    yield figures
    for figure in figures:
        plt.close(figure)


class TestPhaseIllustration:
    def test_six_csv_rows_are_grouped_and_need_no_zarr(self, tmp_path, captured, monkeypatch):
        rows = []
        for setting, spacing in enumerate([2e5, 2e6, 2e7]):
            for individual in range(2):
                for left, right, phase in [(0, 1e7, False), (1e7, 2e7, True)]:
                    rows.append(dict(plot_row=setting * 2 + individual, individual=individual,
                                     space_per_switch=spacing, phase_ser=1000 / spacing,
                                     left=left, right=right, phase=phase))
        path = tmp_path / "phase_runs.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        monkeypatch.setattr(zarr, "open_group", lambda *args, **kwargs: pytest.fail("Zarr read"))
        ch3_plot.plot_phase_switch_illustration(path)
        correct, observed = captured[0].axes
        assert [label.get_text() for label in observed.get_yticklabels()] == list("010101")
        assert correct.get_ylabel() == "Individual ID"
        distances = -np.diff(observed.get_yticks())
        assert distances[1] > distances[0]
        assert distances[3] > distances[2]
        assert len(observed.texts) == 3
        assert "0.2 Mb mean gap" in observed.texts[0].get_text()
        assert "20 Mb mean gap" in observed.texts[-1].get_text()
        for ax in [correct, observed]:
            assert ax.get_xlim() == (0, 20)
            assert ax.get_xlabel() == "Chromosome 20 position (Mb)"


class TestPhaseScatter:
    def test_correlates_log_ser_and_excludes_zero_rate(self, captured):
        rows = []
        gaps = [0, 2e5, 2e6, 2e7]
        means = [9, 1, 3, 4]
        for version in ["0.4.1", "0.5.1"]:
            for site, (gap, mean) in enumerate(zip(gaps, means)):
                for side in ["left", "right"]:
                    rows.append(dict(version=version, replicate=0, true_focal_site=site,
                                     geno_multiplier=0, mispol_rate=0, space_per_switch=gap,
                                     phase_ser=0 if gap == 0 else 1000 / gap,
                                     side=side, overshoot=mean * 1e6))
        frame = pd.DataFrame(rows)
        config = ch3_plot.PlotConfig(phase_spacings=tuple(gaps))
        data = ch3_plot.ChapterData(frame, frame.loc[frame.side == "left"], pd.DataFrame(), config)
        ch3_plot.plot_phasing_overshoot_scatter(data)
        fig = captured[0]
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_xlabel() == "Switch error rate"
        assert [text.get_text() for text in ax.texts[-3:]] == ["20 Mb", "2 Mb", "0.2 Mb"]
        curves = [line for line in ax.lines if line.get_marker() == "o"]
        for curve in curves:
            np.testing.assert_allclose(curve.get_xdata(), [5e-5, 5e-4, 5e-3])
            np.testing.assert_allclose(curve.get_ydata(), [4, 3, 1])
        expected = np.corrcoef(np.log10([5e-5, 5e-4, 5e-3]), [4, 3, 1])[0, 1]
        assert all(f"r = {expected:.3f}" in label.get_text() for label in fig.legends[0].get_texts())
        assert data.phase_spacings == tuple(gaps)
