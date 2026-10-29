import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tskit

import ch3_anc_perf
import ch3_anc_perf_analysis as analysis


@pytest.fixture
def ancestors_path(tmp_path):
    tables = tskit.TableCollection(100)
    for time in [5, 4, 2, 1]:
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time)
    tables.nodes.add_row(time=3)
    for position in [5, 10, 40, 70]:
        tables.sites.add_row(position, ancestral_state="0")
    for left, right, parent, child in [
        (0, 100, 0, 1), (10, 70, 1, 4), (10, 40, 4, 2),
        (40, 70, 1, 2), (70, 100, 1, 3),
    ]:
        tables.edges.add_row(left, right, parent, child)
    tables.sort()
    path = tmp_path / 'ancestors.trees'
    tables.tree_sequence().dump(path)
    return path


class TestAncestorReplay:
    def test_intervals_include_all_copying_segments_and_exclude_pc_nodes(self, ancestors_path):
        inputs = analysis.load_ancestor_inputs(ancestors_path)
        np.testing.assert_array_equal(inputs.start, [0, 0, 1, 3])
        np.testing.assert_array_equal(inputs.end, [4, 4, 3, 4])
        np.testing.assert_array_equal(inputs.time, [5, 4, 2, 1])
        assert inputs.tree_stats['pc_ancestors'] == 1

    def test_epoch_handover_preserves_group_ids_and_removes_virtual_root(self):
        inputs = analysis.AncestorInputs(
            start=np.zeros(7, dtype=int), end=np.ones(7, dtype=int),
            time=np.array([6, 5, 4, 3, 2, 2, 2]),
            position=np.array([0, 1]), tree_stats={},
        )
        received = []

        def group(start, end, time):
            received.append(time)
            return {0: [0], 1: [1, 2], 2: [3]}

        grouping = types.SimpleNamespace(group_ancestors_by_linesweep=group)
        config = analysis.AnalysisConfig(epoch_cutoff_factor=1)
        replay = analysis.replay_grouping(inputs, config, grouping)
        np.testing.assert_array_equal(received[0], [6, 5, 4, 3])
        assert replay.epoch_first_group == 4
        assert list(replay.groups) == [0, 1, 2, 4]
        assert replay.groups[0] == []
        np.testing.assert_array_equal(replay.groups[4], [4, 5, 6])

    def test_incomplete_group_log_is_rejected(self, tmp_path):
        path = tmp_path / 'match_ancestors.log'
        path.write_text('Starting group 0 of 2 with 0 ancestors\n')
        with pytest.raises(ValueError, match='Incomplete'):
            analysis.load_group_timings(path)

    def test_group_ids_can_have_a_gap_at_epoch_handover(self, tmp_path):
        path = tmp_path / 'match_ancestors.log'
        path.write_text(
            'Starting group 0 of 2 with 0 ancestors\n'
            'Finished group 0 of 2 in 0.01 seconds\n'
            'Starting group 2 of 2 with 10 ancestors\n'
            'Finished group 2 of 2 in 2.25 seconds\n'
        )
        groups = analysis.load_group_timings(path)
        assert groups.index.tolist() == [0, 2]
        assert groups['size'].tolist() == [0, 10]
        assert groups.seconds.tolist() == [0.01, 2.25]


@pytest.fixture
def perf_data(tmp_path):
    rows = []
    config = ch3_anc_perf.PlotConfig()
    for version in config.versions:
        for stage in config.stages:
            rows.append(dict(
                num_samples=100, version=version, command=stage,
                elapsed_time=2, max_memory=1024**3, user_time=3,
                sys_time=1, num_threads=4,
            ))
    path = tmp_path / 'perf.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return ch3_anc_perf.load_performance(path, config)


class TestPerformanceLabels:
    def test_resource_titles_and_version_labels_are_distinct(self, perf_data):
        config = ch3_anc_perf.PlotConfig()
        fig = ch3_anc_perf.plot_resources(perf_data, 'parallelism', config)
        assert [ax.get_title() for ax in fig.axes] == list(config.stage_labels)
        assert [text.get_text() for text in fig.legends[0].texts][:2] == list(config.version_labels)
        plt.close(fig)

    def test_elapsed_stacks_use_short_version_labels(self, perf_data):
        config = ch3_anc_perf.PlotConfig()
        fig = ch3_anc_perf.plot_elapsed(perf_data, config)
        assert [text.get_text() for text in fig.axes[0].get_xticklabels()] == list(config.version_labels)
        plt.close(fig)

    def test_elapsed_annotation_uses_total_time_speedup(self, perf_data):
        config = ch3_anc_perf.PlotConfig()
        baseline = perf_data.index.get_level_values('version') == config.versions[0]
        perf_data.loc[baseline, 'elapsed_minutes'] *= 2
        fig = ch3_anc_perf.plot_elapsed(perf_data, config)
        assert [text.get_text() for text in fig.axes[0].texts] == ['0.2', '0.1\n(2.00×)']
        plt.close(fig)


@pytest.fixture
def perf_comparison_data(perf_data):
    frames = []
    for size in [100, 1000, 6404, 10000, 100000]:
        frame = perf_data.reset_index()
        frame['num_samples'] = size
        if size == 6404:
            baseline = frame.version == '0.4.1'
            frame.loc[baseline, 'elapsed_time'] *= 3
            frame.loc[baseline, 'elapsed_minutes'] *= 3
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined.set_index(['num_samples', 'version', 'command']).sort_index()


class Test1kgpPerformanceGroup:
    def test_speedup_adds_haplotype_group_and_preserves_figure_size(self, perf_comparison_data):
        config = ch3_anc_perf.PlotConfig(sample_labels={6404: '6,404 (1KGP)'})
        fig = ch3_anc_perf.plot_speedup(perf_comparison_data, config)
        ax = fig.axes[0]
        labels = [label.get_text() for label in ax.get_xticklabels()]
        assert labels == ['100', '1,000', '6,404 (1KGP)', '10,000', '100,000']
        for bars in ax.containers:
            assert len(bars) == 5
            assert bars[2].get_height() == 3
        np.testing.assert_allclose(fig.get_size_inches(), config.figsize)
        plt.close(fig)

    def test_elapsed_has_two_stacks_per_group_and_correct_total_ratio(self, perf_comparison_data):
        config = ch3_anc_perf.PlotConfig(
            sample_labels={6404: '6,404 (1KGP)'},
            sample_titles={6404: '6,404 haplotypes\n(1KGP)'},
        )
        fig = ch3_anc_perf.plot_elapsed(perf_comparison_data, config)
        assert len(fig.axes) == 5
        tgp_ax = fig.axes[2]
        assert tgp_ax.get_title() == '6,404 haplotypes\n(1KGP)'
        assert [text.get_text() for text in tgp_ax.texts] == ['0.3', '0.1\n(3.00×)']
        for bars in tgp_ax.containers:
            assert len(bars) == 2
            assert bars[0].get_width() == pytest.approx(config.elapsed_bar_width)
        np.testing.assert_allclose(fig.get_size_inches(), config.figsize)
        plt.close(fig)
