"""Verify focal-site weighting, frequency boundaries and the five span curves."""

import ch3_plot
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def span_data():
    rows = []
    for version, spans in zip(("0.4.1", "0.5.1"), ([100, 100, 300, 600], [80, 80, 200, 400])):
        for site, (frequency, span, true_span) in enumerate(zip(
            [0.01, 0.01, 0.02, 1], spans, [50, 50, 200, 300],
        )):
            rows.append(dict(
                version=version, replicate=0, true_focal_site=site,
                inferred_node=2 if site < 2 else site + 2,
                allele_frequency=frequency, inferred_span=span, true_span=true_span,
                geno_multiplier=0, phase_ser=0, mispol_rate=0,
            ))
    erroneous = rows[0].copy()
    erroneous["geno_multiplier"] = 1
    erroneous["inferred_span"] = 1e9
    rows.append(erroneous)
    config = ch3_plot.PlotConfig(ancestor_error_bins=3)
    return ch3_plot.ChapterData(pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame(), config)


@pytest.fixture
def tgp_csv(tmp_path):
    rows = []
    for version in ("0.4.1", "0.5.1"):
        for site, (frequency, span) in enumerate(zip([0.01, 0.01, 0.02, 1], [400, 400, 1000, 1200])):
            rows.append(dict(
                version=version, inf_focal_site=site, inferred_node=2 if site < 2 else site + 2,
                inferred_time=frequency, inferred_span=span, num_samples=6404,
                focal_position=[5, 10, 40, 70][site],
            ))
    path = tmp_path / "tgp_ancestor_spans.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def summary(span_data, tgp_csv):
    empirical = ch3_plot.load_tgp_ancestor_spans(tgp_csv)
    return ch3_plot.summarise_ancestor_span_curves(span_data, empirical)


class TestAncestorSpanCurves:
    def test_means_weight_each_focal_site_and_truth_is_counted_once(self, summary):
        selected = summary.loc[(summary.dataset == "Simulation") & (summary.version == "0.4.1")]
        assert selected.num_focal_sites.tolist() == [3, 0, 1]
        assert selected.mean_span_bp.iloc[0] == pytest.approx(500 / 3)
        assert np.isnan(selected.mean_span_bp.iloc[1])
        assert selected.mean_span_bp.iloc[2] == 600
        assert selected.mean_span_mb.iloc[0] == pytest.approx(500 / 3 / 1e6)
        truth = summary.loc[(summary.dataset == "Simulation") & (summary.version == "True")]
        assert truth.num_focal_sites.tolist() == [3, 0, 1]
        assert truth.mean_span_bp.iloc[0] == 100
        empirical = summary.loc[(summary.dataset == "1000 Genomes Project") & (summary.version == "0.4.1")]
        assert empirical.num_focal_sites.tolist() == [3, 0, 1]
        assert empirical.mean_span_bp.iloc[0] == 600

    def test_bin_edges_include_zero_and_one_and_use_midpoints(self, span_data, tgp_csv):
        span_data.config.ancestor_error_bins = 2
        span_data.spans = span_data.spans.loc[span_data.spans.geno_multiplier == 0].copy()
        span_data.spans["allele_frequency"] = [0, 0.5, 0.5, 1] * 2
        empirical = ch3_plot.load_tgp_ancestor_spans(tgp_csv)
        empirical["allele_frequency"] = [0, 0.5, 0.5, 1] * 2
        table = ch3_plot.summarise_ancestor_span_curves(span_data, empirical)
        selected = table.loc[(table.dataset == "Simulation") & (table.version == "True")]
        assert selected.num_focal_sites.tolist() == [1, 3]
        np.testing.assert_allclose(selected.daf, [0.25, 0.75])

    def test_rejects_ages_in_place_of_frequency(self, span_data, tgp_csv):
        empirical = ch3_plot.load_tgp_ancestor_spans(tgp_csv)
        empirical.loc[0, "allele_frequency"] = 200
        with pytest.raises(ValueError, match="frequencies must be"):
            ch3_plot.summarise_ancestor_span_curves(span_data, empirical)

    @pytest.mark.parametrize("dataset, versions", [
        ("Simulation", ["tsinfer 0.4", "tsinfer 0.5", "True"]),
        ("1000 Genomes Project", ["tsinfer 0.4", "tsinfer 0.5"]),
    ])
    def test_each_plot_has_solid_curves_and_one_count_panel(
        self, summary, span_data, monkeypatch, dataset, versions,
    ):
        figures = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: figures.append(fig))
        ch3_plot.plot_ancestor_span_curves(summary, dataset, config=span_data.config, title="Title")
        fig = figures[0]
        assert len(fig.axes) == 2
        curve, counts = fig.axes
        assert [bar.get_height() for bar in counts.patches] == [3, 0, 1]
        assert curve.get_yscale() == "log"
        assert curve.get_title() == "Title"
        assert len(curve.lines) == len(versions)
        assert all(line.get_linestyle() == "-" for line in curve.lines)
        expected_colors = list(span_data.config.version_colors)
        if dataset == "Simulation":
            expected_colors.append(span_data.config.span_curve_true_color)
        for line, color in zip(curve.lines, expected_colors):
            assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(color)
        legend = curve.get_legend()
        assert legend.get_title().get_text() == ""
        assert [text.get_text() for text in legend.get_texts()] == versions
        assert all(text.get_fontsize() == 11 for text in legend.get_texts())
        assert span_data.config.span_curve_true_color == "#8b0000"
        plt.close(fig)

    def test_dataset_plots_share_the_same_mean_span_limits(self, summary, span_data, monkeypatch):
        figures = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: figures.append(fig))
        for dataset in ["Simulation", "1000 Genomes Project"]:
            ch3_plot.plot_ancestor_span_curves(summary, dataset, config=span_data.config)
        assert figures[0].axes[0].get_ylim() == figures[1].axes[0].get_ylim()
        for fig in figures:
            plt.close(fig)
