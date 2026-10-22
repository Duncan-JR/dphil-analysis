"""Verify lossless length compression and the joint mutation intersection."""

import ch5_dating
import ch5_paramsearch
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
import tskit


@pytest.fixture
def length_observations():
    return np.array([1, 1, 2, 3, 3, 3, 10, 40, 100])


class TestCompressedLengths:
    def test_reads_pipeline_csv_and_reuses_complete_count_cache(self, tmp_path, length_observations):
        source = pd.DataFrame({'dataset': 'sim', 'af_cutoff': 0.01,
                               'min_focal_region_size': 1, 'is_gap': False,
                               'num_sites': length_observations})
        path = tmp_path / 'lengths.csv'
        source.to_csv(path, index=False)
        config = {'data_dir': tmp_path, 'cache_dir': tmp_path,
                  'manifest': {'tables': {'region_lengths': path.name}}}
        table = ch5_paramsearch.length_counts(config)
        assert table['count'].sum() == len(length_observations)
        assert (table.num_sites * table['count']).sum() == length_observations.sum()
        cached = ch5_paramsearch.length_counts(config)
        pd.testing.assert_frame_equal(cached, table, check_dtype=False)

    def test_quantiles_match_expanded_data(self, length_observations):
        values, counts = np.unique(length_observations, return_counts=True)
        table = pd.DataFrame({'num_sites': values, 'count': counts})
        for quantile in [0, 0.25, 0.5, 0.99, 1]:
            result = ch5_paramsearch.weighted_quantile(table, quantile)
            assert result == pytest.approx(np.quantile(length_observations, quantile))

    def test_kde_matches_pt6_expanded_scott_kde(self, length_observations):
        values, counts = np.unique(length_observations, return_counts=True)
        table = pd.DataFrame({'num_sites': values, 'count': counts})
        grid = np.linspace(0, 150, 100)
        kde = stats.gaussian_kde(length_observations)
        expected = kde(grid) + kde(-grid)
        result = ch5_paramsearch.length_density(table, grid)
        np.testing.assert_allclose(result, expected)


@pytest.fixture
def mutation_runs(tmp_path):
    tables = tskit.TableCollection(100)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(time=100)
    tables.edges.add_row(0, 100, parent=2, child=0)
    tables.edges.add_row(0, 100, parent=2, child=1)
    tables.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
    for position in [10, 20, 30, 40]:
        site = tables.sites.add_row(position, '0')
        tables.mutations.add_row(site, node=0, derived_state='1', time=position,
                                 metadata={'mn': position * 2})
    tables.sort()
    full = tables.tree_sequence()
    full_path = tmp_path / 'full.trees'
    full.dump(full_path)
    # Stitched loses position 30; truth loses 20. Both references must use 10/40.
    stitched = full.delete_sites([2])
    stitched_path = tmp_path / 'stitched.trees'
    stitched.dump(stitched_path)
    truth = ch5_dating.mutation_ages(full.delete_sites([1]), posterior=False)
    runs = pd.DataFrame([
        {'dataset': 'sim', 'comparison_method': 'full_hmm', 'af_cutoff': None,
         'min_focal_region_size': None, 'dated_path': full_path},
        {'dataset': 'sim', 'comparison_method': 'stitched_greedy', 'af_cutoff': 0.01,
         'min_focal_region_size': 1, 'dated_path': stitched_path},
    ])
    return runs, truth


class TestJointMutationIntersection:
    def test_both_references_use_truth_full_stitched_intersection(self, mutation_runs):
        runs, truth = mutation_runs
        settings = {'cutoffs': [], 'labels': ['all_variants']}
        result = ch5_paramsearch.shared_mutation_correlations(runs, truth, settings)
        assert result.matched_mutations.eq(2).all()
        assert set(result.reference_method) == {'true_arg', 'full_hmm'}
        assert set(result.af_cutoff.dropna()) == {0.01}
        assert result.pearson_r.eq(1).all()
        assert result.loc[result.reference_method.eq('true_arg'), 'mean_reference_age'].eq(25).all()
        assert result.loc[result.reference_method.eq('full_hmm'), 'mean_reference_age'].eq(50).all()
