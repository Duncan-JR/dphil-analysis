"""Verify overlap-weighted error burden and whole-comparison denominators."""

import ch3_plot
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
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
                    overlap_span=overlap, overlap_site_span=10, side=side,
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
        "overlap_site_span": [1, 9, 10, 5, 5, 1],
        "num_errors": [1, 1, 1, 1, 2, 198],
        "num_should_be_0": [1, 1, 0, 1, 1, 99],
        "num_should_be_1": [0, 0, 1, 0, 1, 99],
    })


class TestAncestorOverlapErrors:
    def test_unsplit_burden_uses_total_rates_on_log_axis(self, overlap_comparisons, monkeypatch):
        summary = ch3_plot.summarise_ancestor_overlap_errors(
            overlap_comparisons, cutoffs=[0, 0.1, 0.5, 1],
        )
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: fig)
        fig = ch3_plot.plot_ancestor_overlap_errors(summary, split_error_types=False)
        burden = fig.axes[1]
        expected = summary.mismatch_fraction.to_numpy()
        np.testing.assert_allclose([bar.get_height() for bar in burden.patches], expected, equal_nan=True)
        assert burden.get_legend() is None
        assert burden.get_yscale() == "log"
        assert all(bar.get_hatch() is None for bar in burden.patches)
        plt.close(fig)

    def test_burden_change_labels_use_reference_pooled_rates(self, overlap_comparisons, monkeypatch):
        frame = overlap_comparisons.copy()
        increased = (frame.version == "0.5.1") & (frame.true_focal_site == 0)
        frame.loc[increased, ["num_errors", "num_should_be_0"]] = 2
        summary = ch3_plot.summarise_ancestor_overlap_errors(
            frame, cutoffs=[0, 0.1, 0.5, 1],
        )
        config = ch3_plot.PlotConfig()
        figures = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: figures.append(fig))
        ch3_plot.plot_ancestor_overlap_errors(summary, config=config, show_mismatch_change=True)
        labels = figures[0].axes[1].texts
        assert [label.get_text() for label in labels] == ["+100.00%", "", "-50.00%"]
        assert all(label.get_color() == config.version_colors[1] for label in labels)
        assert labels[0].xy[1] == pytest.approx(2 / 20)
        assert labels[2].xy[1] == pytest.approx(1 / 10)
        plt.close(figures[0])

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
        assert old.loc[0, "mismatch_fraction"] == pytest.approx(1 / 20)
        assert old.loc[0, "error_free_fraction"] == 0.5
        assert new.loc[0, "error_free_fraction"] == 1
        assert old.num_ancestors.tolist() == [2, 0, 1]
        assert np.isnan(old.loc[1, "mismatch_fraction"])
        assert np.isnan(old.loc[1, "error_free_fraction"])
        assert old.loc[2, "mismatch_fraction"] == pytest.approx(2 / 10)
        assert old.loc[2, "error_free_fraction"] == 0

    def test_physical_length_does_not_change_mismatch_fraction(self, overlap_comparisons):
        original = ch3_plot.summarise_ancestor_overlap_errors(overlap_comparisons)
        changed = overlap_comparisons.copy()
        changed["overlap_span"] *= 100
        actual = ch3_plot.summarise_ancestor_overlap_errors(changed)
        np.testing.assert_allclose(actual.mismatch_fraction, original.mismatch_fraction, equal_nan=True)
        changed["overlap_site_span"] *= 2
        actual = ch3_plot.summarise_ancestor_overlap_errors(changed)
        np.testing.assert_allclose(actual.mismatch_fraction, original.mismatch_fraction / 2, equal_nan=True)
        np.testing.assert_allclose(actual.error_free_fraction, original.error_free_fraction, equal_nan=True)

    def test_duplicate_side_rows_are_rejected(self, overlap_comparisons):
        repeated = pd.concat([overlap_comparisons, overlap_comparisons.iloc[:1]])
        with pytest.raises(ValueError, match="one overlap comparison"):
            ch3_plot.summarise_ancestor_overlap_errors(repeated)

    def test_proportion_labels_use_three_decimals_and_version_colors(
        self, overlap_comparisons, monkeypatch,
    ):
        summary = ch3_plot.summarise_ancestor_overlap_errors(
            overlap_comparisons, cutoffs=[0, 0.1, 0.5, 1],
        )
        config = ch3_plot.PlotConfig()
        figures = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: figures.append(fig))
        ch3_plot.plot_ancestor_overlap_errors(summary, config=config)
        labels = figures[0].axes[0].texts
        assert [label.get_text() for label in labels] == [
            "0.500", "", "0.000", "1.000", "", "0.000",
        ]
        assert [label.get_color() for label in labels] == [
            config.version_colors[0], config.version_colors[0], config.version_colors[0],
            config.version_colors[1], config.version_colors[1], config.version_colors[1],
        ]
        assert all(label.get_fontsize() == config.ancestor_error_label_fontsize for label in labels)
        plt.close(figures[0])

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
        assert len(fig.axes[1].patches) == 12
        assert [bar.get_height() for bar in fig.axes[2].patches] == [2, 0, 1]
        assert fig.axes[0].get_ylim() == (0, 1)
        assert fig.axes[1].get_yscale() == "log"
        assert fig.axes[1].patches[0].get_height() == pytest.approx(1 / 20)
        solid = fig.axes[1].patches[:3]
        hatched = fig.axes[1].patches[3:6]
        np.testing.assert_allclose(
            [bar.get_height() for bar in solid], [1 / 20, np.nan, 0], equal_nan=True,
        )
        np.testing.assert_allclose(
            [bar.get_height() for bar in hatched], [0, np.nan, 2 / 10], equal_nan=True,
        )
        for lower, upper in zip(solid, hatched):
            assert upper.get_hatch() == "///"
            assert upper.get_facecolor() == (1, 1, 1, 1)
            assert upper.get_edgecolor() == lower.get_facecolor()
        np.testing.assert_allclose(
            [bar.get_y() for bar in hatched],
            [bar.get_height() for bar in solid], equal_nan=True,
        )
        assert len(fig.legends) == 1
        error_labels = [text.get_text() for text in fig.axes[1].get_legend().get_texts()]
        assert error_labels == ["True = 0, Observed = 1", "True = 1, Observed = 0"]
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
        np.testing.assert_allclose(summary.should_be_1_mismatch_fraction, [0.1, 0.2])
        np.testing.assert_allclose(summary.should_be_0_mismatch_fraction, [0.05, 0.1])

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
        assert np.isnan(summary.loc[2, "should_be_0_mismatch_fraction"])
        assert np.isnan(summary.loc[2, "should_be_1_mismatch_fraction"])
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
        assert ax.lines[0].get_label() == "True = 0, Observed = 1"
        assert ax.lines[1].get_label() == "True = 1, Observed = 0"
        assert ax.get_xlabel() == "Derived focal allele frequency"
        assert ax.get_yscale() == "log"
        plt.close(fig)


@pytest.fixture
def demography_paths(overlap_csv, tmp_path):
    frame = pd.read_csv(overlap_csv)
    frame["overshoot"] = np.arange(len(frame)) - 5
    paths = {}
    for index, model in enumerate(["out_of_africa", "admixture", "africa", "zigzag"]):
        selected = frame.copy()
        selected["overlap_span"] *= index + 1
        selected["overlap_site_span"] *= index + 1
        path = tmp_path / f"{model}.csv"
        selected.to_csv(path, index=False)
        paths[model] = path
    return paths


class TestDemographyComparisons:
    def test_quantile_panels_use_full_data_and_linear_units(self, demography_paths, monkeypatch):
        comparisons = ch3_plot.load_demography_comparisons(demography_paths)
        labels = {model: model for model in demography_paths}
        config = ch3_plot.PlotConfig()
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: fig)
        fig = ch3_plot.plot_demography_overshoot_quantiles(comparisons, labels, config=config)
        assert len(fig.axes) == 8
        for row, model in enumerate(labels):
            central, tails = fig.axes[2 * row:2 * row + 2]
            assert central.get_xlim() == config.demography_quantile_center_limits_kb
            assert tails.get_xlim() == config.demography_quantile_tail_limits_mb
            assert central.get_xscale() == tails.get_xscale() == "linear"
            assert central.get_ylabel() == labels[model]
            for index, (version, color) in enumerate(zip(config.versions, config.version_colors)):
                values = comparisons.loc[
                    (comparisons.model == model) & (comparisons.version == version), "overshoot"
                ]
                expected = values.quantile([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
                iqr, median = central.lines[2 * index:2 * index + 2]
                outer, inner = tails.lines[2 * index:2 * index + 2]
                np.testing.assert_allclose(iqr.get_xdata(), expected.loc[[0.25, 0.75]] / config.bp_per_kb)
                np.testing.assert_allclose(median.get_xdata(), [expected.loc[0.5] / config.bp_per_kb])
                np.testing.assert_allclose(outer.get_xdata(), expected.loc[[0.001, 0.999]] / config.bp_per_mb)
                np.testing.assert_allclose(inner.get_xdata(), expected.loc[[0.01, 0.99]] / config.bp_per_mb)
                assert iqr.get_color() == median.get_color() == outer.get_color() == inner.get_color() == color
        plt.close(fig)

    def test_loader_selects_error_free_profiles_and_retains_both_sides(self, demography_paths):
        comparisons = ch3_plot.load_demography_comparisons(demography_paths)
        assert len(comparisons) == 48
        assert comparisons.model.unique().tolist() == list(demography_paths)
        assert comparisons.groupby(["model", "side"]).size().tolist() == [6] * 8

    def test_pooled_rates_keep_models_separate_and_count_overlap_once(self, demography_paths):
        comparisons = ch3_plot.load_demography_comparisons(demography_paths)
        summary = ch3_plot.summarise_demography_overlap_errors(comparisons)
        assert len(summary) == 8
        for index, model in enumerate(demography_paths):
            selected = summary.loc[summary.model == model].set_index("version")
            assert selected.num_ancestors.tolist() == [3, 3]
            assert selected.overlap_bp.tolist() == [1100 * (index + 1)] * 2
            assert selected.loc["0.4.1", "mismatch_fraction"] == pytest.approx(3 / (30 * (index + 1)))
            assert selected.loc["0.5.1", "mismatch_fraction"] == pytest.approx(1 / (30 * (index + 1)))
            assert selected.loc["0.4.1", "error_free_fraction"] == pytest.approx(1 / 3)
            np.testing.assert_allclose(
                selected.should_be_0_mismatch_fraction + selected.should_be_1_mismatch_fraction,
                selected.mismatch_fraction,
            )

    def test_categorical_plots_have_four_groups_and_no_count_panel(self, demography_paths, monkeypatch):
        comparisons = ch3_plot.load_demography_comparisons(demography_paths)
        labels = {model: model.replace("_", " ") for model in demography_paths}
        captured = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: captured.append(fig))
        ch3_plot.plot_demography_overshoot_box(comparisons, labels)
        box = captured[0]
        assert len(box.axes) == 1
        assert len(box.axes[0].patches) == 8
        assert [text.get_text() for text in box.axes[0].get_xticklabels()] == list(labels.values())
        summary = ch3_plot.summarise_demography_overlap_errors(comparisons)
        ch3_plot.plot_ancestor_overlap_errors(
            summary, group_by="model", group_labels=labels, show_counts=False,
        )
        burden = captured[1]
        assert len(burden.axes) == 2
        assert len(burden.axes[0].patches) == 8
        assert len(burden.axes[1].patches) == 16
        assert burden.axes[1].get_yscale() == "log"
        assert [text.get_text() for text in burden.axes[1].get_xticklabels()] == list(labels.values())
        assert burden.axes[1].patches[0].get_height() == pytest.approx(1 / 30)
        assert burden.axes[1].patches[3].get_height() == pytest.approx(1 / 120)
        for figure in captured:
            plt.close(figure)


    @pytest.mark.parametrize("conf_secondary", [0.01, 0.25])
    def test_secondary_quantiles_leave_scatter_and_primary_caps_unchanged(
        self, demography_paths, monkeypatch, conf_secondary,
    ):
        comparisons = ch3_plot.load_demography_comparisons(demography_paths)
        labels = {model: model for model in demography_paths}
        config = ch3_plot.PlotConfig(jitter_max_points=1)
        figures = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: figures.append(fig))
        ch3_plot.plot_demography_overshoot_box(
            comparisons, labels, config=config, conf=0.001, conf_secondary=None,
        )
        options = {} if conf_secondary == 0.01 else {"conf_secondary": conf_secondary}
        ch3_plot.plot_demography_overshoot_box(
            comparisons, labels, config=config, conf=0.001, **options,
        )
        baseline, secondary = figures
        baseline_dots = [c for c in baseline.axes[0].collections
                         if isinstance(c, mcollections.PathCollection)]
        secondary_dots = [c for c in secondary.axes[0].collections
                          if isinstance(c, mcollections.PathCollection)]
        assert len(baseline_dots) == len(secondary_dots) == 8
        for before, after in zip(baseline_dots, secondary_dots):
            np.testing.assert_array_equal(before.get_offsets(), after.get_offsets())
        baseline_caps = [c for c in baseline.axes[0].collections
                         if isinstance(c, mcollections.LineCollection)]
        secondary_caps = [c for c in secondary.axes[0].collections
                          if isinstance(c, mcollections.LineCollection)]
        assert len(baseline_caps) == 8
        assert len(secondary_caps) == 16
        for before, after in zip(baseline_caps, secondary_caps[::2]):
            np.testing.assert_array_equal(before.get_segments(), after.get_segments())
        selected = comparisons.loc[
            (comparisons.model == next(iter(labels))) & (comparisons.version == "0.4.1")
        ]
        expected = np.quantile(selected.overshoot, [conf_secondary, 1 - conf_secondary])
        expected /= config.bp_per_mb
        segments = secondary_caps[1].get_segments()
        np.testing.assert_allclose([segment[0, 1] for segment in segments], expected)
        primary_dash = secondary_caps[0].get_linestyles()[0][1]
        secondary_dash = secondary_caps[1].get_linestyles()[0][1]
        assert not np.array_equal(primary_dash, secondary_dash)
        expected_color = mcolors.to_rgba(config.quantile_secondary_color)
        np.testing.assert_allclose(secondary_caps[1].get_colors()[0], expected_color)
        assert secondary.legends[0].get_lines()[-1].get_color() == config.quantile_secondary_color
        legend = [text.get_text() for text in secondary.legends[0].get_texts()]
        assert legend[-2:] == ["0.1% tail quantiles", f"{conf_secondary * 100:g}% tail quantiles"]
        assert secondary.axes[0].get_yscale() == "linear"
        for fig in figures:
            plt.close(fig)
