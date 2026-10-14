"""Check physical-window denominators and saved-run metric reconstruction."""

import json

import ch5_hyperparams
import numpy as np
import pandas as pd
import pytest
import tskit
import yaml


@pytest.fixture
def saved_run(tmp_path):
    for directory in ['ancestors', 'matches', 'focal_masks']:
        (tmp_path / directory).mkdir()
    tables = tskit.TableCollection(sequence_length=64)
    for _ in range(3):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for position in [5, 10, 20, 25, 50, 55]:
        tables.sites.add_row(position, ancestral_state='0')
    tables.tree_sequence().dump(tmp_path / 'ancestors/test_inferred_ancestors.trees')
    identity = {'source': 'samples', 'sample_id': 'tsk_0', 'ploidy_index': 0}
    full_path = [
        {'left': 50, 'right': 64, 'parent': 2},
        {'left': 20, 'right': 50, 'parent': 1},
        {'left': 0, 'right': 20, 'parent': 0},
    ]
    stitched_path = [
        {'left': 0, 'right': 10, 'parent': 0},
        {'left': 10, 'right': 25, 'parent': 1},
        {'left': 25, 'right': 64, 'parent': 2},
    ]
    full = {**identity, 'path': full_path}
    stitched = {**identity, 'path': stitched_path, 'ac_cutoff': 3,
                'max_mismatches': 0, 'method': 'hmm'}
    (tmp_path / 'matches/test_inferred_samples_matches.jsonl').write_text(json.dumps(full) + '\n')
    path = tmp_path / 'matches/test_inferred_ac3_k0_hmm_stitched_samples_matches.jsonl'
    path.write_text(json.dumps(stitched) + '\n')
    np.savez(tmp_path / 'focal_masks/test_inferred_ac3_k0_focal_mask.npz',
             sample_id=['tsk_0'], ploidy_index=[0], offsets=[0, 3],
             left_site_index=[0, 2, 4], right_site_index=[2, 4, 6],
             is_gap=[False, True, False], num_sites=6, ac_cutoff=3,
             max_mismatches=0, focal_choice='left')
    return ch5_hyperparams.RunInputs(tmp_path, 'test', {('tsk_0', 0): 'YRI'},
                                     'YRI', [0, 60], 3, 0, 'hmm', 'left')


@pytest.fixture
def reordered_trees():
    tables = tskit.TableCollection(sequence_length=100)
    tables.time_units = 'generations'
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    for _ in range(2):
        tables.individuals.add_row()
    for index in range(4):
        metadata = {'sample_id': f'tsk_{index // 2}', 'ploidy_index': index % 2}
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=index // 2,
                             metadata=metadata)
    for time in [1, 2, 3]:
        tables.nodes.add_row(time=time)
    for parent, child in [(4, 0), (4, 1), (5, 2), (5, 3), (6, 4), (6, 5)]:
        tables.edges.add_row(0, 100, parent, child)
    tables.sort()
    reference = tables.tree_sequence()
    sample_order = [2, 0, 3, 1]
    reordered = reference.simplify(samples=sample_order)
    comparison_tables = reordered.dump_tables()
    comparison_tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    for index, original in enumerate(sample_order):
        row = comparison_tables.nodes[index]
        metadata = {'sample_id': f'tsk_{original // 2}', 'ploidy_index': original % 2}
        comparison_tables.nodes[index] = row.replace(metadata=metadata)
    return ch5_hyperparams.AlignedTrees(comparison_tables.tree_sequence(), reference)


class TestSavedRun:
    def test_metrics_use_saved_partition_and_original_parents(self, saved_run):
        results = ch5_hyperparams.summarise_run(saved_run)
        row = results.comparisons.iloc[0]
        assert row.population_group == 'African'
        assert row.full_switches == row.stitched_switches == 2
        assert row.full_seam_switches == 2
        assert row.stitched_focal_switches == row.stitched_gap_switches == 1
        assert row.stitched_seam_switches == 0
        assert row.num_extra_switches == row.switch_increase_percentage == 0
        assert row.focal_agreement_percentage == 75
        assert row.gap_agreement_percentage == 50
        assert row.focal_coverage_percentage == row.gap_coverage_percentage == 50
        assert results.lengths.num_sites.tolist() == [2, 2, 2]
        assert results.lengths.region.tolist() == ['focal', 'gap', 'focal']

    def test_empty_focal_and_zero_full_switches_are_undefined(self, saved_run):
        path = saved_run.data_dir / 'matches/test_inferred_samples_matches.jsonl'
        record = json.loads(path.read_text())
        record['path'] = [{'left': 0, 'right': 64, 'parent': 0}]
        path.write_text(json.dumps(record) + '\n')
        mask_path = saved_run.data_dir / 'focal_masks/test_inferred_ac3_k0_focal_mask.npz'
        with np.load(mask_path) as archive:
            mask = {field: archive[field] for field in archive.files}
        mask.update(offsets=[0, 1], left_site_index=[0], right_site_index=[6], is_gap=[True])
        np.savez(mask_path, **mask)
        row = ch5_hyperparams.summarise_run(saved_run).comparisons.iloc[0]
        assert np.isnan(row.switch_increase_percentage)
        assert np.isnan(row.focal_agreement_percentage)
        assert row.focal_coverage_percentage == 0
        assert row.gap_coverage_percentage == 100


class TestPhysicalCoverage:
    def test_clips_both_chromosome_tails_to_selected_window(self):
        actual = ch5_hyperparams.coverage_percentage(
            np.array([0, 1, 2]), np.array([1, 2, 3]),
            np.array([False, True, False]), np.array([12, 20, 35]), [10, 40]
        )
        assert actual == 50


class TestArgComparison:
    def test_all_datasets_use_full_hmm_reference(self, reordered_trees):
        trees = ch5_hyperparams.TreeSequences(
            full={'simulation': reordered_trees.reference, '1kgp': reordered_trees.reference},
            stitched={('simulation', 3, 0): reordered_trees.comparison,
                      ('1kgp', 3, 1): reordered_trees.comparison},
            statistics=pd.DataFrame(),
        )
        settings = [{'name': name, 'analysis_span_bp': [10, 60]} for name in trees.full]
        metrics = ch5_hyperparams.arg_metrics(trees, settings, workers=1)
        assert set(metrics.dataset) == {'simulation', '1kgp'}
        assert metrics.reference_method.eq('full_hmm').all()
        assert metrics.arf.eq(0).all()
        assert metrics.tpr.eq(1).all()
        assert 'rmse' not in metrics.columns

    def test_window_preserves_missing_terminal_ancestry(self, reordered_trees):
        tables = reordered_trees.reference.dump_tables()
        tables.edges.left = np.full(tables.edges.num_rows, 20.0)
        tables.edges.right = np.full(tables.edges.num_rows, 50.0)
        ts = tables.tree_sequence()
        window = ch5_hyperparams.window_tree_sequence(ts, [10, 60])
        assert window.sequence_length == 50
        assert window.edges_left.min() == 10
        assert window.edges_right.max() == 40
        assert window.first().num_edges == window.last().num_edges == 0

    def test_aligns_biological_samples_and_clips_shared_window(self, reordered_trees):
        aligned = ch5_hyperparams.align_tree_sequences(
            reordered_trees.comparison, reordered_trees.reference, [10, 60]
        )
        assert aligned.comparison.sequence_length == aligned.reference.sequence_length == 50
        np.testing.assert_array_equal(aligned.comparison.samples(), aligned.reference.samples())
        job = ch5_hyperparams.ArgComparison('simulation', 3, 0, aligned.comparison, aligned.reference,
                                            'stitched_hmm', 'full_hmm')
        metrics = ch5_hyperparams.compare_arg(job)
        assert metrics['arf'] == 0
        assert metrics['tpr'] == 1
        assert 'rmse' not in metrics

    def test_arf_and_tpr_have_stitched_to_full_direction(self, reordered_trees):
        aligned = ch5_hyperparams.align_tree_sequences(
            reordered_trees.comparison, reordered_trees.reference, [10, 60]
        )
        tables = aligned.reference.dump_tables()
        unary = tables.nodes.add_row(time=0.5)
        edge_ids = [index for index, edge in enumerate(tables.edges) if edge.child == 0]
        assert len(edge_ids) == 1
        edge_id = edge_ids[0]
        edge = tables.edges[edge_id]
        tables.edges[edge_id] = edge.replace(child=unary)
        tables.edges.add_row(0, 50, unary, 0)
        tables.sort()
        reference_with_unary = tables.tree_sequence()
        job = ch5_hyperparams.ArgComparison('simulation', 3, 0, aligned.comparison, reference_with_unary,
                                            'stitched_hmm', 'full_hmm')
        metrics = ch5_hyperparams.compare_arg(job)
        assert metrics['arf'] == 0
        assert metrics['tpr'] == pytest.approx(7 / 8)
        assert 'rmse' not in metrics

    def test_aligns_truth_without_matched_node_metadata(self, reordered_trees):
        tables = reordered_trees.reference.dump_tables()
        for node_id in reordered_trees.reference.samples():
            row = tables.nodes[int(node_id)]
            tables.nodes[int(node_id)] = row.replace(metadata={})
        truth = tables.tree_sequence()
        reference_nodes = ch5_hyperparams.truth_sample_nodes(truth)
        aligned = ch5_hyperparams.align_tree_sequences(
            reordered_trees.comparison, truth, [10, 60], reference_nodes=reference_nodes
        )
        job = ch5_hyperparams.ArgComparison('simulation', 3, 0, aligned.comparison, aligned.reference,
                                            'stitched_hmm', 'true_arg')
        metrics = ch5_hyperparams.compare_arg(job)
        assert metrics['arf'] == 0
        assert metrics['tpr'] == 1
        assert metrics['reference_method'] == 'true_arg'

    def test_truth_comparisons_include_full_baseline_once(self, reordered_trees, tmp_path):
        truth_path = tmp_path / 'truth.trees'
        reordered_trees.reference.dump(truth_path)
        trees = ch5_hyperparams.TreeSequences(
            full={'simulation': reordered_trees.comparison, '1kgp': reordered_trees.comparison},
            stitched={('simulation', 3, 0): reordered_trees.comparison,
                      ('simulation', 3, 1): reordered_trees.comparison,
                      ('1kgp', 3, 0): reordered_trees.comparison},
            statistics=pd.DataFrame(),
        )
        experiment = {'datasets': [{'name': 'simulation', 'ts_path': str(truth_path)},
                                   {'name': '1kgp', 'ts_path': None}]}
        (tmp_path / 'experiment.yaml').write_text(yaml.safe_dump(experiment))
        config = {'experiment_config': 'experiment.yaml', 'truth_comparison': {'full_hmm_cutoff': 1},
                  'datasets': [{'name': name, 'analysis_span_bp': [10, 60]} for name in trees.full]}
        config_path = tmp_path / 'analysis.yaml'
        config_path.write_text(yaml.safe_dump(config))
        metrics = ch5_hyperparams.truth_arg_metrics(trees, config_path, workers=1)
        full = metrics.loc[metrics.comparison_method.eq('full_hmm')]
        assert len(metrics) == 3
        assert set(metrics.dataset) == {'simulation'}
        assert metrics.reference_method.eq('true_arg').all()
        assert full.ac_cutoff.tolist() == [1]
        assert full.max_mismatches.isna().all()
        assert metrics.arf.eq(0).all()
        assert metrics.tpr.eq(1).all()

    def test_rejects_duplicate_sample_identity(self, reordered_trees):
        tables = reordered_trees.comparison.dump_tables()
        row = tables.nodes[0]
        tables.nodes[0] = row.replace(metadata={'sample_id': 'tsk_0', 'ploidy_index': 0})
        with pytest.raises(ValueError, match='unique'):
            ch5_hyperparams.align_tree_sequences(tables.tree_sequence(), reordered_trees.reference, [0, 50])


class TestTreeStatisticDifferences:
    def test_signed_differences_use_each_datasets_full_counts(self):
        statistics = pd.DataFrame([
            {'dataset': 'simulation', 'method': 'full', 'num_edges': 20, 'num_trees': 10},
            {'dataset': '1kgp', 'method': 'full', 'num_edges': 5, 'num_trees': 4},
            {'dataset': 'simulation', 'method': 'stitched', 'num_edges': 21, 'num_trees': 8},
            {'dataset': '1kgp', 'method': 'stitched', 'num_edges': 3, 'num_trees': 6},
        ])
        differences = ch5_hyperparams.tree_statistic_differences(statistics).set_index('dataset')
        assert differences.loc['simulation', 'num_edges_difference'] == 1
        assert differences.loc['simulation', 'num_trees_difference'] == -2
        assert differences.loc['1kgp', 'num_edges_difference'] == -2
        assert differences.loc['1kgp', 'num_trees_difference'] == 2


class TestTreeSequenceLoading:
    def test_does_not_require_simulation_truth_file(self, saved_run):
        directory = saved_run.data_dir
        ts = tskit.load(directory / 'ancestors/test_inferred_ancestors.trees')
        ts.dump(directory / 'matches/test_inferred_samples_raw.trees')
        ts.dump(directory / 'matches/test_inferred_ac3_k0_hmm_stitched_samples_raw.trees')
        experiment = {'data_dir': str(directory), 'ac_cutoff': [3], 'max_mismatches': [0],
                      'datasets': [{'name': 'test', 'ts_path': 'missing_simulation_truth.trees'}]}
        experiment_path = directory / 'experiment.yaml'
        experiment_path.write_text(yaml.safe_dump(experiment))
        config = {'experiment_config': 'experiment.yaml', 'method': 'hmm',
                  'datasets': [{'name': 'test'}]}
        config_path = directory / 'analysis.yaml'
        config_path.write_text(yaml.safe_dump(config))
        trees = ch5_hyperparams.load_tree_sequences(config_path)
        assert set(trees.full) == {'test'}
        assert set(trees.stitched) == {('test', 3, 0)}
        assert len(trees.statistics) == 2
