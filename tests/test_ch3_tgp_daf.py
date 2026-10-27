"""Check empirical DAF selection and its agreement with simulation styling."""

import ch3_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi
import zarr


@pytest.fixture
def tgp_store(tmp_path):
    path = tmp_path / "tgp.zarr"
    group = zarr.open_group(path, mode="w")
    positions = np.arange(0, 120, 10)
    derived = np.array([0, 2, 2, 2, 2, 2, 1, 3, 7, 8, 2, 2])
    group.create_array("variant_position", data=positions)
    group.create_array("variant_1kgp_all_subset_derived_count", data=derived)
    group.create_array("variant_1kgp_all_subset_ancestral_count", data=8 - derived)
    for index, name in enumerate(ch3_plot.PlotConfig().tgp_variant_masks, start=1):
        mask = np.zeros(len(positions), dtype=np.int8)
        mask[index] = 1
        group.create_array(name, data=mask)
    return path


class TestTgpDaf:
    def test_masks_and_half_open_region_preserve_singletons(self, tgp_store):
        config = ch3_plot.PlotConfig(tgp_region_start=10, tgp_region_end=100)
        variants = ch3_plot.load_tgp_daf(tgp_store, config)
        assert variants.position.tolist() == [60, 70, 80, 90]
        assert variants.derived_count.tolist() == [1, 3, 7, 8]
        assert variants.num_alleles.tolist() == [8, 8, 8, 8]

    def test_counts_and_axes_match_simulation_conventions(self, tgp_store, monkeypatch):
        config = ch3_plot.PlotConfig(
            tgp_region_start=10, tgp_region_end=100, daf_bins=4,
        )
        variants = ch3_plot.load_tgp_daf(tgp_store, config)
        simulation = pd.DataFrame({
            "record_type": ["daf"], "mispol_rate": [0], "replicate": [0],
            "derived_count": [1], "num_alleles": [8], "mispolarised": [False],
            "num_sites": [4],
        })
        config.mispol_rates = (0,)
        data = ch3_plot.ChapterData(
            pd.DataFrame(), pd.DataFrame({"replicate": [0]}), simulation, config,
        )
        captured = []
        monkeypatch.setattr(ch3_plot, "save_figure", lambda fig, *args: captured.append(fig))
        ch3_plot.plot_mispolarisation_daf(data)
        ch3_plot.plot_tgp_daf(variants, config=config)
        simulation_ax = captured[0].axes[0]
        fig = captured[1]
        ax = fig.axes[0]
        # Explicit counts include the singleton and the rightmost endpoint.
        expected = ndi.gaussian_filter1d(
            np.array([1., 1., 0., 2.]), config.daf_smoothing_bins, mode="reflect"
        )
        tgp_line = ax.lines[-1]
        np.testing.assert_allclose(tgp_line.get_ydata(), expected)
        np.testing.assert_allclose(tgp_line.get_xdata(), [0.125, 0.375, 0.625, 0.875])
        assert tgp_line.get_ydata().sum() == pytest.approx(len(variants))
        assert tgp_line.get_color() == "red"
        assert tgp_line.get_linewidth() == simulation_ax.lines[0].get_linewidth()
        assert simulation_ax.get_legend().get_title().get_text() == "Simulation:\nMispol. rate"
        assert ax.get_legend().get_texts()[0].get_text() == "1000 Genomes Project\nn=3202"
        assert ax.get_xlim() == (0, 1)
        assert ax.get_yscale() == "log"
        assert ax.get_xlabel() == "Derived allele frequency"
        assert ax.get_ylabel() == "Site count"
        np.testing.assert_allclose(fig.get_size_inches(), config.daf_size)
        plt.close(fig)
        plt.close(captured[0])
