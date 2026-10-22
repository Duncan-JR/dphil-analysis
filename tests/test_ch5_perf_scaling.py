import ch5_perf_scaling
import ch5_gap_parents
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
import tskit


class TestWeightedRegionDistributions:
    @pytest.fixture
    def counts(self):
        return pd.DataFrame({'length': [1.0, 4.0, 11.0, 27.0], 'count': [9, 4, 2, 1]})

    def test_quantiles_match_full_observations(self, counts):
        expanded = np.repeat(counts.length, counts['count'])
        for quantile in [0, 0.25, 0.5, 0.99, 1]:
            result = ch5_perf_scaling.weighted_quantile(counts, quantile)
            assert result == pytest.approx(np.quantile(expanded, quantile))

    def test_density_matches_full_observations(self, counts):
        expanded = np.repeat(counts.length, counts['count'])
        kde = stats.gaussian_kde(expanded)
        grid = np.linspace(0, 40, 80)
        expected = kde(grid) + kde(-grid)
        actual = ch5_perf_scaling.length_density(counts, grid)
        np.testing.assert_allclose(actual, expected)


class TestGapParentClassification:
    @pytest.fixture
    def trees(self):
        tables = tskit.TableCollection(60)
        for time in [2, 0.5, 0.05, 0.05]:
            tables.nodes.add_row(time=time)
        for child in [1, 2, 3]:
            tables.edges.add_row(0, 60, 0, child)
        for position, parent in [(10, 2), (20, 1), (40, 2), (50, 3)]:
            site = tables.sites.add_row(position, ancestral_state='0')
            tables.mutations.add_row(site, parent, derived_state='1')
        tables.sort()
        reference = tables.tree_sequence()
        for sample in [4, 5]:
            node = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
            assert node == sample
            tables.edges.add_row(0, 15, 2, sample)
            tables.edges.add_row(15, 60, 1 if sample == 4 else 3, sample)
        tables.mutations.add_row(0, 5, derived_state='0')
        tables.mutations.add_row(3, 4, derived_state='1')
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        return {'reference': reference, 'full': tables.tree_sequence()}

    def test_selected_seed_and_terminal_mutations(self, trees):
        reference = trees['reference']
        seeds = ch5_gap_parents.focal_seed_sites(reference, 'left')
        assert seeds.tolist() == [-1, 1, 0, 3]
        right_seeds = ch5_gap_parents.focal_seed_sites(reference, 'right')
        assert right_seeds[2] == 2
        samples = np.array([0, 1, 0, 1])
        parents = np.array([2, 2, 3, 3])
        carried = ch5_gap_parents.carried_focal_seeds(trees['full'], reference, seeds, samples, parents)
        assert carried.tolist() == [True, False, True, True]

    def test_gap_overlap_merges_touching_parent_runs(self):
        fragments = ch5_gap_parents.gap_fragments(
            np.array([0.0, 12.0, 15.0]), np.array([12.0, 15.0, 60.0]),
            np.array([2, 2, 1]), np.array([0.0]), np.array([50.0]),
            np.array([10.0, 20.0, 40.0, 50.0]), [[10.0, 50.0]])
        rows = fragments.set_index('parent')
        assert rows.loc[2, 'sites'] == 1
        assert rows.loc[2, 'span_bp'] == 5
        assert rows.loc[2, 'segments'] == 1
        assert rows.loc[1, 'sites'] == 2
        assert rows.loc[1, 'span_bp'] == 35
        assert fragments.distinct_parents.sum() == 2

    def test_final_site_has_zero_analysis_span(self):
        fragments = ch5_gap_parents.gap_fragments(
            np.array([0.0]), np.array([60.0]), np.array([1]),
            np.array([50.0]), np.array([60.0]),
            np.array([10.0, 20.0, 40.0, 50.0]), [[10.0, 50.0]])
        assert fragments.sites.item() == 1
        assert fragments.span_bp.item() == 0
