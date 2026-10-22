"""Check the exact bitset adapter against tscompare's reference implementation."""

import ch5_arg_metrics
import msprime
import numpy as np
import pytest
import tscompare
import tscompare.methods
import tskit


@pytest.fixture
def recombining_pair():
    comparison = msprime.sim_ancestry(40, sequence_length=1000, population_size=100,
                                     recombination_rate=0.0001, random_seed=14)
    reference = msprime.sim_ancestry(40, sequence_length=1000, population_size=100,
                                    recombination_rate=0.0001, random_seed=15)
    return comparison, reference


@pytest.fixture
def unary_missing_pair():
    tables = tskit.TableCollection(100)
    for _ in range(3):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for time in [1, 2, 3]:
        tables.nodes.add_row(time=time)
    tables.edges.add_row(0, 50, parent=3, child=0)
    tables.edges.add_row(0, 50, parent=3, child=1)
    tables.edges.add_row(0, 50, parent=4, child=3)
    tables.edges.add_row(0, 50, parent=5, child=4)
    tables.edges.add_row(0, 50, parent=5, child=2)
    tables.edges.add_row(50, 90, parent=3, child=1)
    tables.edges.add_row(50, 90, parent=3, child=2)
    tables.edges.add_row(50, 90, parent=5, child=3)
    tables.edges.add_row(50, 90, parent=5, child=0)
    tables.sort()
    comparison = tables.tree_sequence()
    reference = comparison.simplify(keep_unary=False)
    return comparison, reference


class TestBitsetArgMetrics:
    @pytest.mark.parametrize('pair_fixture', ['recombining_pair', 'unary_missing_pair'])
    def test_matches_every_reference_metric_and_restores_library(self, request, pair_fixture):
        comparison, reference = request.getfixturevalue(pair_fixture)
        original = tscompare.methods.CladeMap
        expected = tscompare.haplotype_arf(comparison, reference)
        actual = ch5_arg_metrics.haplotype_arf(comparison, reference)
        assert tscompare.methods.CladeMap is original
        for name in ['arf', 'tpr', 'rmse', 'matched_span', 'total_span']:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name))

    def test_self_comparison_is_exact(self, recombining_pair):
        comparison, _ = recombining_pair
        actual = ch5_arg_metrics.haplotype_arf(comparison, comparison)
        assert actual.arf == 0
        assert actual.tpr == 1

    def test_restores_library_on_failure(self, recombining_pair):
        comparison, reference = recombining_pair
        original = tscompare.methods.CladeMap
        reference = reference.simplify(samples=reference.samples()[:2])
        with pytest.raises(ValueError, match='Samples'):
            ch5_arg_metrics.haplotype_arf(comparison, reference)
        assert tscompare.methods.CladeMap is original
