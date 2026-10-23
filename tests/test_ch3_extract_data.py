"""Check transferable counts against direct diploid and polarisation calculations."""

import dataclasses
import pathlib
import sys

import ch3_extract_data
import ch3_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi
import zarr


@dataclasses.dataclass
class VariantFixture:
    folder: pathlib.Path
    prefix: str
    calls: np.ndarray
    observed: np.ndarray
    mask: np.ndarray


@dataclasses.dataclass
class ExportFixture:
    path: pathlib.Path
    frame: pd.DataFrame


@pytest.fixture
def variant_data(tmp_path):
    folder = tmp_path / "data/anc_eval"
    prefix = "test_chr20"
    calls = np.array([
        [[0, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 1], [0, 1]],
        [[1, 1], [0, 1]], [[1, 1], [1, 1]], [[0, 0], [1, 1]],
    ], dtype=np.int8)
    observed = calls.copy()
    observed[0, 0] = [0, 1]
    observed[1, 1] = [1, 0]  # A phase reversal preserves diploid dosage.
    observed[2, 0] = [1, 1]
    observed[3, 0] = [0, 1]
    observed[4, 1] = [0, 1]
    observed[5, 0] = [1, 1]
    mask = np.array([True, False, True, False, True, False])
    for replicate in (0, 1):
        for genotype, phase, mispol in [(0, 0, 0), (1, 0, 0), (0, 0, 0.1), (0, 0.01, 0)]:
            name = f"{prefix}-rep{replicate}-geno-{genotype}-phase{phase}-mispol{mispol}.zarr"
            group = zarr.open_group(folder / "zarr_vcfs" / name, mode="w")
            profile_calls = observed if genotype == 1 else calls
            group.create_array("call_genotype", data=profile_calls, chunks=(2, 2, 2))
            group.create_array("sample_id", data=np.arange(2))
            group.create_array("variant_position", data=np.arange(10, 70, 10))
            group.create_array("variant_allele_count", data=calls.sum(axis=(1, 2)))
            polarisation = mask if mispol > 0 else np.ones(6, dtype=bool)
            group.create_array("variant_mispolarisation_mask", data=polarisation)
    return VariantFixture(folder, prefix, calls, observed, mask)


@pytest.fixture
def exported_data(variant_data, tmp_path):
    folder = variant_data.folder
    prefix = variant_data.prefix
    output = tmp_path / "plot_data"
    config = ch3_extract_data.ExtractionConfig(
        data_folder=folder, output_folder=output, chunk_rows=2, workers=1,
    )
    path = ch3_extract_data.extract_data(prefix, config)
    frame = ch3_plot._read_zarr_data(path, prefix)
    return ExportFixture(path, frame)


class TestExtraction:
    def test_counts_match_direct_comparison_and_can_rebin(self, variant_data, exported_data):
        calls = variant_data.calls
        observed = variant_data.observed
        frame = exported_data.frame
        true_dosage = calls.sum(axis=2)
        observed_dosage = observed.sum(axis=2)
        for bins in (2, 3, 7):
            config = ch3_plot.PlotConfig(genotype_multipliers=(1,), transition_bins=bins)
            spans = pd.DataFrame({"replicate": [0, 1]})
            data = ch3_plot.ChapterData(pd.DataFrame(), spans, frame, config)
            summary = ch3_plot.summarise_genotype_errors(data)
            rate = summary.rates.iloc[0]
            assert rate.changed == 2 * np.count_nonzero(true_dosage != observed_dosage)
            assert rate.total == 2 * true_dosage.size
            frequencies = calls.sum(axis=(1, 2)) / 4
            edges = np.linspace(0, 1, bins + 1)
            for dosage, transition in [(0, "00 → 01/10"), (2, "11 → 01/10")]:
                homozygous = true_dosage == dosage
                to_het = homozygous & (observed_dosage == 1)
                expected_total = np.histogram(frequencies, edges, weights=homozygous.sum(axis=1))[0]
                expected_changed = np.histogram(frequencies, edges, weights=to_het.sum(axis=1))[0]
                actual = summary.transitions.loc[summary.transitions.transition == transition]
                np.testing.assert_array_equal(actual.total, expected_total * 2)
                np.testing.assert_array_equal(actual.changed, expected_changed * 2)
            assert data.phase_rates == (0, 0.01)

    def test_parallel_matches_serial(self, variant_data, exported_data, tmp_path):
        folder = variant_data.folder
        prefix = variant_data.prefix
        serial_path = exported_data.path
        config = ch3_extract_data.ExtractionConfig(
            data_folder=folder, output_folder=tmp_path / "parallel", chunk_rows=2, workers=2,
        )
        parallel = ch3_extract_data.extract_data(prefix, config)
        assert parallel.read_bytes() == serial_path.read_bytes()
        assert not parallel.with_suffix(".csv.tmp").exists()

    def test_cli_defaults_work_from_evaluation_root(self, variant_data, tmp_path, monkeypatch):
        folder = variant_data.folder
        prefix = variant_data.prefix
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["ch3_extract_data.py", prefix, "--workers", "1"])
        ch3_extract_data.main()
        assert (folder / "dataframes" / f"{prefix}_zarr_data.csv").is_file()

    def test_missing_prefix_fails_clearly(self, variant_data):
        folder = variant_data.folder
        config = ch3_extract_data.ExtractionConfig(data_folder=folder, workers=1)
        with pytest.raises(FileNotFoundError, match="No variant stores"):
            ch3_extract_data.extract_data("missing", config)


class TestCsvOnlyPlots:
    def test_preprocess_joins_focal_counts_across_chunks(self, variant_data, exported_data, tmp_path, monkeypatch):
        prefix = variant_data.prefix
        export = exported_data.path
        rows = []
        for version in ["0.4.1", "0.5.1"]:
            for side in ["left", "right"]:
                for focal_site, position, mispolarised in [(0, 10, False), (1, 20, True)]:
                    overshoots = [-7, 9] if focal_site == 0 else [4, -2]
                    side_index = 0 if side == "left" else 1
                    rows.append(dict(
                        replicate=0, version=version, geno_multiplier=0, phase_ser=0,
                        mispol_rate=0.1, inferred_node=0, true_node=42, side=side,
                        allele_frequency=0 if focal_site == 0 else 0.75,
                        inferred_span=100, true_span=50 + 25 * focal_site,
                        overshoot=overshoots[side_index],
                        true_focal_site=focal_site, focal_position=position,
                        focal_site_mispolarised=mispolarised,
                    ))
        aggregated = tmp_path / f"{prefix}-aggregated.csv"
        pd.DataFrame(rows).to_csv(aggregated, index=False)
        monkeypatch.setattr(zarr, "open_group", lambda *args, **kwargs: pytest.fail("Zarr read during plotting"))
        config = ch3_plot.PlotConfig(workers=1, preprocess_chunk_rows=1)
        output = export.parent
        paths = ch3_plot.preprocess_data(aggregated, export, output, config)
        reduced = pd.read_csv(paths.overshoots)
        assert len(reduced) == 8
        assert reduced.overshoot.tolist() == [-7, 4, 9, -2, -7, 4, 9, -2]
        assert reduced.true_derived_count.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
        assert reduced.focal_site_mispolarised.tolist() == [False, True] * 4
        assert reduced.allele_frequency.tolist() == [0, 0.75] * 4
        assert not reduced.duplicated(list(config.focal_keys) + ["side"]).any()
        data = ch3_plot.load_data(output, prefix, config)
        assert len(data.spans) == 4
        assert data.spans.true_span.tolist() == [50, 75, 50, 75]
        config.workers = 2
        parallel = ch3_plot.preprocess_data(aggregated, export, tmp_path / "parallel", config)
        assert parallel.overshoots.read_bytes() == paths.overshoots.read_bytes()
        assert parallel.spans.read_bytes() == paths.spans.read_bytes()

    def test_daf_plot_uses_exact_polarised_site_counts(self, variant_data, exported_data, monkeypatch):
        calls = variant_data.calls
        mask = variant_data.mask
        frame = exported_data.frame
        config = ch3_plot.PlotConfig(mispol_rates=(0.1,), daf_bins=4)
        data = ch3_plot.ChapterData(pd.DataFrame(), pd.DataFrame({"replicate": [0, 1]}), frame, config)
        captured = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: captured.append(fig))
        monkeypatch.setattr(zarr, "open_group", lambda *args, **kwargs: pytest.fail("Zarr read during plotting"))
        ch3_plot.plot_mispolarisation_daf(data)
        fig = captured[0]
        frequencies = calls.sum(axis=(1, 2)) / 4
        frequencies[~mask] = 1 - frequencies[~mask]
        edges = np.linspace(0, 1, 5)
        counts = np.histogram(frequencies, edges)[0]
        expected = ndi.gaussian_filter1d(
            counts.astype(float) * 2, config.daf_smoothing_bins, mode="reflect"
        )
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), expected)
        assert fig.axes[0].get_ylabel() == "Site count"
        assert fig.axes[0].lines[0].get_ydata().sum() == pytest.approx(2 * len(frequencies))
        plt.close(fig)


@pytest.fixture
def focal_plot_data():
    rows = []
    for version in ("0.4.1", "0.5.1"):
        for site, sides in enumerate(((-4, 2), (0, 0), (0, 8))):
            for side, overshoot in zip(("left", "right"), sides):
                rows.append(dict(
                    version=version, replicate=0, geno_multiplier=0, phase_ser=0,
                    mispol_rate=0.1, inferred_node=site // 2, true_node=42,
                    true_focal_site=site, side=side, overshoot=overshoot * 1e6,
                    allele_frequency=0.25, true_allele_frequency=0.25,
                    true_derived_count=1, focal_site_mispolarised=site == 1,
                    inferred_span=2e6, true_span=(site + 1) * 1e6,
                ))
    frame = pd.DataFrame(rows)
    spans = frame.loc[frame.side == "left"].copy()
    config = ch3_plot.PlotConfig(mispol_rates=(0.1,))
    return ch3_plot.ChapterData(frame, spans, pd.DataFrame(), config)


class TestFocalSitePlots:
    def test_mean_gives_each_focal_site_equal_weight(self, focal_plot_data, monkeypatch):
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: fig)
        fig = ch3_plot._overshoot_mean(
            focal_plot_data, focal_plot_data.overshoots, "mispol_rate", (0.1,),
            "Mispolarisation rate", "test", None,
        )
        heights = [bar.get_height() for bar in fig.axes[0].patches]
        np.testing.assert_allclose(heights, [1, 1])
        lower, upper = fig.axes[0].get_ylim()
        assert lower < 0 < upper
        plt.close(fig)

    @pytest.mark.parametrize("fraction_above", [True, False])
    def test_tails_count_sites_sharing_an_ancestor(self, focal_plot_data, monkeypatch, fraction_above):
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: fig)
        fig = ch3_plot.plot_mispolarisation_overshoot_tails(
            focal_plot_data, fraction_above=fraction_above,
        )
        negative = fig.axes[0].lines[0]
        positive = fig.axes[1].lines[0]
        expected = np.array([1 / 3, 2 / 3, 1 / 3])
        if not fraction_above:
            expected = 1 - expected
        at_two = np.flatnonzero(positive.get_xdata() == 2)[0]
        actual = [negative.get_ydata()[0], positive.get_ydata()[0], positive.get_ydata()[at_two]]
        np.testing.assert_allclose(actual, expected)
        direction = "above" if fraction_above else "below"
        assert fig.axes[0].get_ylabel() == f"Fraction of focal sites {direction} cutoff"
        plt.close(fig)

    def test_mispolarised_filter_does_not_include_neighbouring_focal_sites(self, focal_plot_data, monkeypatch):
        monkeypatch.setattr(ch3_plot, "_overshoot_histogram", lambda data, frame, *args, **kwargs: frame)
        frame = ch3_plot.plot_mispolarisation_low_count_overshoot_hist(focal_plot_data)
        assert frame.true_focal_site.tolist() == [1, 1, 1, 1]
        assert frame.overshoot.tolist() == [0, 0, 0, 0]

    def test_tail_count_bars_count_nonzero_sites_per_sign(self, focal_plot_data, monkeypatch):
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: fig)
        fig = ch3_plot.plot_mispolarisation_overshoot_tails(focal_plot_data, show_counts=True)
        assert len(fig.axes) == 5
        negative_counts, positive_counts = fig.axes[3:]
        assert [bar.get_height() for bar in negative_counts.patches] == [1, 1]
        assert [bar.get_height() for bar in positive_counts.patches] == [2, 2]
        assert negative_counts.get_ylabel() == "Count"
        assert positive_counts.get_ylabel() == "Count"
        assert fig.axes[0].get_ylabel() == "Fraction of focal sites\nabove cutoff"
        plt.close(fig)

    def test_span_counts_are_focal_sites(self, focal_plot_data, monkeypatch):
        focal_plot_data.spans["mispol_rate"] = 0
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: fig)
        fig = ch3_plot.plot_ancestor_spans(focal_plot_data, cutoffs=[0, 1])
        assert fig.axes[1].patches[0].get_height() == 3
        assert fig.axes[1].texts[0].get_text() == "3"
        assert fig.axes[1].get_ylabel() == "Count"
        plt.close(fig)
