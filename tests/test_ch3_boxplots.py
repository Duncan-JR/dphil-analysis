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
                        space_per_switch=2e6 if phase > 0 else 0,
                        allele_frequency=frequency, true_allele_frequency=frequency,
                        side=side, overshoot=overshoot, true_span=100000,
                        inferred_span=110000,
                    ))
    frame = pd.DataFrame(rows)
    config = ch3_plot.PlotConfig(
        genotype_multipliers=(0, 1), phase_spacings=(0, 2e6), mispol_rates=(0, 0.1),
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


class TestBoxplotSampling:
    def test_disabled_sampling_shows_every_observation_at_configured_size(self):
        config = ch3_plot.PlotConfig(jitter_max_points=None, jitter_size=3.5)
        values = np.linspace(-2, 2, 10000)
        fig, ax = plt.subplots()
        ch3_plot._draw_overshoot_group(
            ax, values, 0, 0.2, "blue", np.random.default_rng(config.jitter_seed),
            config, 0.1, config.jitter_alpha,
        )
        points = ax.collections[0]
        np.testing.assert_array_equal(points.get_offsets()[:, 1], values)
        np.testing.assert_array_equal(points.get_sizes(), [3.5])
        assert points.get_alpha() == config.jitter_alpha
        plt.close(fig)

    def test_jitter_samples_only_within_one_mb_and_uses_full_data_quantiles(self):
        config = ch3_plot.PlotConfig()
        central = np.linspace(-1, 1, 10000)
        outside = np.array([-20, -5, -1.01, 1.01, 5, 20])
        values = np.concatenate([central, outside])
        fig, ax = plt.subplots()
        ch3_plot._draw_overshoot_group(
            ax, values, 0, 0.2, "blue", np.random.default_rng(config.jitter_seed),
            config, 0.1, config.jitter_alpha,
        )
        points, caps = ax.collections
        sampled = points.get_offsets()[:, 1]
        assert len(sampled) == config.jitter_max_points + len(outside)
        sampled_outside = sampled[np.abs(sampled) > 1]
        np.testing.assert_array_equal(np.sort(sampled_outside), outside)
        sampled_central = sampled[np.abs(sampled) <= 1]
        assert len(sampled_central) == config.jitter_max_points
        assert len(np.unique(sampled_central)) == config.jitter_max_points
        low, high = np.quantile(values, [0.1, 0.9])
        np.testing.assert_allclose([segment[0, 1] for segment in caps.get_segments()], [low, high])
        assert points.get_alpha() == config.jitter_alpha
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


class TestOvershootMeans:
    @pytest.mark.parametrize("plot", [
        ch3_plot.plot_genotype_overshoot_mean,
        ch3_plot.plot_phasing_overshoot_mean,
    ])
    def test_labels_show_relative_change_below_new_version_bars(
        self, chapter_data, captured_figures, plot,
    ):
        newer = chapter_data.overshoots.version == chapter_data.config.versions[1]
        chapter_data.overshoots.loc[newer, "overshoot"] *= 1.5
        plot(chapter_data)
        ax = captured_figures[0].axes[0]
        assert len(ax.texts) == 2
        for annotation in ax.texts:
            assert annotation.get_text() == "-50.0%"
            assert annotation.get_color() == chapter_data.config.version_colors[1]
            assert annotation.get_position()[1] < 0
            assert annotation.xy[1] == pytest.approx(-0.0075)


class TestOvershootDiff:
    def test_genotype_uses_even_viridis_colours_and_right_legend(
        self, chapter_data, captured_figures,
    ):
        extra_profile = chapter_data.overshoots.loc[
            chapter_data.overshoots.geno_multiplier == 1
        ].copy()
        extra_profile["geno_multiplier"] = 10
        chapter_data.overshoots = pd.concat([chapter_data.overshoots, extra_profile])
        chapter_data.config.genotype_multipliers = (0, 1, 10)
        ch3_plot.plot_genotype_overshoot_diff(chapter_data)
        fig = captured_figures[0]
        expected_colours = plt.get_cmap("viridis")(np.linspace(0, 0.8, 3))
        for ax in fig.axes:
            colours = [step.get_edgecolor() for step in ax.patches]
            np.testing.assert_allclose(colours, expected_colours)
        assert fig.legends[0].get_title().get_text() == "Error\nmultiplier"
        assert fig.legends[0]._outside_loc == "right"
        assert fig.get_layout_engine().get()["hspace"] == pytest.approx(0.15)

    def test_signed_count_changes_share_histogram_bins_and_leave_zero_reference_gaps(
        self, chapter_data, captured_figures,
    ):
        chapter_data.config.histogram_bins = 4
        chapter_data.config.genotype_multipliers = (0,)
        chapter_data.overshoots = pd.DataFrame({
            "version": ["0.4.1"] * 3 + ["0.5.1"] * 3,
            "geno_multiplier": 0, "phase_ser": 0, "mispol_rate": 0,
            "overshoot": [0, 0, 1500000, 0, 1000000, 3000000],
        })
        ch3_plot.plot_genotype_overshoot_hist(chapter_data)
        ch3_plot.plot_genotype_overshoot_diff(chapter_data)
        histogram_ax = captured_figures[0].axes[0]
        count_ax, change_ax = captured_figures[1].axes
        count_steps = count_ax.patches[0].get_data()
        change_steps = change_ax.patches[0].get_data()
        histogram_left_edges = [patch.get_x() for patch in histogram_ax.containers[0]]
        np.testing.assert_array_equal(count_steps.edges[:-1], histogram_left_edges)
        np.testing.assert_array_equal(change_steps.edges, count_steps.edges)
        np.testing.assert_allclose(count_steps.values, [2, np.nan, 1, np.nan])
        np.testing.assert_allclose(change_steps.values, [-50, np.nan, -100, np.nan])
        assert count_ax.get_yscale() == "log"
        assert change_ax.get_yscale() == "symlog"
        for ax, label in zip((count_ax, change_ax), ("A", "B")):
            assert ax.texts[0].get_text() == label
            assert ax.texts[0].get_transform() == ax.transAxes
            assert ax.texts[0].get_position() == (0.02, 0.96)
        assert count_ax.get_ylabel() == "Observation count\n(tsinfer 0.4)"
        assert change_ax.get_ylabel() == "Percentage increase in\ntsinfer 0.5"
        assert not count_ax.yaxis_inverted()
        assert not change_ax.yaxis_inverted()
        assert count_ax.get_position().height == pytest.approx(change_ax.get_position().height)
        assert not count_ax.patches[0].get_fill()
        assert not change_ax.patches[0].get_fill()

    @pytest.mark.parametrize("plot", [
        ch3_plot.plot_genotype_overshoot_diff,
        ch3_plot.plot_mispolarisation_overshoot_diff,
        ch3_plot.plot_mispolarisation_low_count_overshoot_diff,
        ch3_plot.plot_phasing_overshoot_diff,
    ])
    def test_error_types_share_step_layout(self, chapter_data, captured_figures, plot):
        chapter_data.overshoots["true_derived_count"] = 2
        chapter_data.overshoots["focal_site_mispolarised"] = chapter_data.overshoots.mispol_rate > 0
        plot(chapter_data, num_bins=7)
        count_ax, change_ax = captured_figures[0].axes
        assert count_ax.get_yscale() == "log"
        assert change_ax.get_yscale() == "symlog"
        expected_colours = plt.get_cmap("viridis")(np.linspace(0, 0.8, len(count_ax.patches)))
        for ax, label in zip((count_ax, change_ax), ("A", "B")):
            colours = [step.get_edgecolor() for step in ax.patches]
            np.testing.assert_allclose(colours, expected_colours)
            assert ax.texts[0].get_text() == label
        for step in change_ax.patches:
            values = step.get_data().values
            assert len(values) == 7
            assert np.isnan(values).any()
            assert (values[np.isfinite(values)] == 0).all()

    def test_mispolarised_histogram_has_two_upward_separated_panels(
        self, chapter_data, captured_figures,
    ):
        chapter_data.overshoots["true_derived_count"] = 2
        chapter_data.overshoots["focal_site_mispolarised"] = chapter_data.overshoots.mispol_rate > 0
        chapter_data.spans["focal_site_mispolarised"] = chapter_data.spans.mispol_rate > 0
        chapter_data.spans["true_derived_count"] = [1, 12000] * (len(chapter_data.spans) // 2)
        ch3_plot.plot_mispolarised_focal_overshoot_hist(chapter_data)
        upper, lower, dac = captured_figures[0].axes
        for ax, label, version in zip((upper, lower), ("A", "B"), ("0.4", "0.5")):
            assert ax.get_yscale() == "log"
            assert not ax.yaxis_inverted()
            assert ax.texts[0].get_text() == label
            assert ax.texts[0].get_position() == (0.02, 0.96)
            assert ax.get_legend().get_title().get_text() == rf"$\mathbf{{tsinfer\ {version}}}$" + "\nMispol. rate"
            assert not ax.containers
        assert upper.get_position().y0 > lower.get_position().y1
        line = dac.lines[0]
        np.testing.assert_array_equal(line.get_xdata(), [1, 12000])
        np.testing.assert_array_equal(line.get_ydata(), [1, 1])
        np.testing.assert_array_equal(line.get_markevery(), [0])
        assert line.get_marker() == "o"
        expected_color = ch3_plot._overshoot_colors([0.1], chapter_data.config.overshoot_cmaps[1])[0.1]
        np.testing.assert_allclose(line.get_color(), expected_color)
        assert dac.get_xscale() == dac.get_yscale() == "log"
        assert dac.get_title() == "True DAC spectrum\n(mispol. focal sites)"
        assert dac.texts[0].get_text() == "C"


class TestOvershootScatter:
    def test_measured_genotype_rates_means_and_correlations(
        self, chapter_data, captured_figures,
    ):
        chapter_data.zarr_data = pd.DataFrame({
            "record_type": ["genotype", "genotype"],
            "geno_multiplier": [1, 1], "changed": [1, 3], "total": [50, 150],
        })
        chapter_data.overshoots["overshoot"] = 1000000
        erroneous = chapter_data.overshoots.geno_multiplier == 1
        chapter_data.overshoots.loc[erroneous, "overshoot"] = -1000000
        newer = chapter_data.overshoots.version == "0.5.1"
        chapter_data.overshoots.loc[newer, "overshoot"] *= 2
        ch3_plot.plot_genotype_overshoot_scatter(chapter_data)
        fig = captured_figures[0]
        ax = fig.axes[0]
        assert ax.get_xscale() == "linear"
        version_lines = [line for line in ax.lines if line.get_marker() == "o"]
        assert len(version_lines) == 2
        for line in version_lines:
            np.testing.assert_allclose(line.get_xdata(), [0, 0.02])
            assert line.get_markerfacecolor() == line.get_color()
        np.testing.assert_allclose(version_lines[0].get_ydata(), [1, -1])
        np.testing.assert_allclose(version_lines[1].get_ydata(), [2, -2])
        assert [label.get_text() for label in ax.texts] == ["+1Mb", "-1Mb", "0X", "1X"]
        for annotation in ax.texts[:2]:
            assert annotation.get_color() == chapter_data.config.version_colors[1]
        assert ax.texts[0].get_position() == (-5, 4)
        assert ax.texts[0].get_verticalalignment() == "bottom"
        assert ax.texts[1].get_position() == (-5, -4)
        assert ax.texts[1].get_verticalalignment() == "top"
        guides = ax.texts[2:]
        assert guides[0].xy[1] == guides[1].xy[1]
        assert guides[0].xy[1] < 0
        assert all(guide.get_rotation() == 0 for guide in guides)
        np.testing.assert_allclose(ax.get_xticks(), [0, 0.02])
        assert ax.lines[-1].get_color() == "black"
        legend_labels = [label.get_text() for label in fig.legends[0].get_texts()]
        assert legend_labels == ["tsinfer 0.4 (r = -1.000)", "tsinfer 0.5 (r = -1.000)"]
