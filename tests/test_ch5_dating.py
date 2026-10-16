"""Verify mutation-age extraction and mutation-key alignment."""

import ch5_dating
import numpy as np
import pandas as pd
import pytest
import tsinfer
import tskit


@pytest.fixture
def dated_mutations():
    tables = tskit.TableCollection(sequence_length=100)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(time=100)
    tables.edges.add_row(0, 100, parent=2, child=0)
    tables.edges.add_row(0, 100, parent=2, child=1)
    tables.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
    for position, mean, time in zip([10, 20, 30], [10, 40, 90], [2, 4, 6], strict=True):
        site = tables.sites.add_row(position, ancestral_state='0')
        tables.mutations.add_row(site, node=0, derived_state='1', time=time,
                                 metadata={'mn': mean})
    site = tables.sites.add_row(40, ancestral_state='0')
    first = tables.mutations.add_row(site, node=0, derived_state='1', time=8,
                                     metadata={'mn': 80})
    tables.mutations.add_row(site, node=0, derived_state='0', time=7,
                             parent=first, metadata={'mn': 70})
    tables.sort()
    return tables.tree_sequence()


class TestMutationAges:
    def test_uses_posterior_means_and_excludes_recurrent_sites(self, dated_mutations):
        ages = ch5_dating.mutation_ages(dated_mutations, posterior=True)
        assert ages.index.get_level_values('position').tolist() == [10, 20, 30]
        assert ages.age.tolist() == [10, 40, 90]

    def test_truth_uses_recorded_mutation_times(self, dated_mutations):
        ages = ch5_dating.mutation_ages(dated_mutations, posterior=False)
        assert ages.age.tolist() == [2, 4, 6]

    def test_derived_frequency_counts_sample_descendants(self, dated_mutations):
        ages = ch5_dating.mutation_ages(dated_mutations, posterior=True, include_frequency=True)
        assert ages.derived_frequency.tolist() == [0.5, 0.5, 0.5]

    def test_frequency_bin_boundaries_enter_upper_bin(self):
        frequencies = pd.Series([0.001, 0.009, 0.01, 0.49, 0.5, 0.9])
        settings = {'cutoffs': [0.01, 0.5], 'labels': ['rare', 'intermediate', 'high']}
        bins = ch5_dating.frequency_bins(frequencies, settings)
        assert bins.tolist() == ['rare', 'rare', 'intermediate', 'intermediate', 'high', 'high']

    def test_calculates_pearson_on_corresponding_mutations(self, dated_mutations):
        inferred = ch5_dating.mutation_ages(dated_mutations, posterior=True).age
        truth = ch5_dating.mutation_ages(dated_mutations, posterior=False).age
        expected = np.corrcoef([10, 40, 90], [2, 4, 6])[0, 1]
        assert ch5_dating.correlate_ages(inferred, truth) == pytest.approx(expected)

    def test_rejects_misaligned_mutations(self):
        first = pd.Series([1, 2], index=[10, 20])
        second = pd.Series([1, 2], index=[20, 10])
        with pytest.raises(ValueError, match='identical ordered mutation keys'):
            ch5_dating.correlate_ages(first, second)


@pytest.fixture(params=[
    {'include': 'POS >= 1000', 'exclude': 'INFO/match_eval_derived_ac == 1'},
    {'include': '(POS >= 1000) & (INFO/match_eval_derived_ac != 1)', 'exclude': None},
])
def native_config(request):
    source = tsinfer.config.Source(path='observed.zarr', name='samples', samples='a,b',
                                    **request.param)
    return tsinfer.config.Config(
        sources={'samples': source}, ancestors=[],
        match=tsinfer.config.MatchConfig(sources={}, output='unused.trees'),
        ancestral_state=tsinfer.config.AncestralState(path='observed.zarr', is_reference=True))


class TestAugmentationConfig:
    def test_restores_dataset_filters_and_preserves_sample_selection(self, native_config):
        restored = ch5_dating.augmentation_config(native_config, 'POS >= 1000', None)
        assert restored.sources['samples'].include == 'POS >= 1000'
        assert restored.sources['samples'].exclude is None
        assert restored.sources['samples'].samples == 'a,b'
        assert restored.augment_sites.sources == ['samples']


@pytest.fixture
def matched_grid(tmp_path, dated_mutations, monkeypatch):
    tables = dated_mutations.dump_tables()
    for _ in range(198):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    boundary_mutation = tables.mutations[1]
    tables.mutations[1] = boundary_mutation.replace(node=2, time=101)
    full = tables.tree_sequence()
    full_path = tmp_path / 'full.trees'
    full.dump(full_path)
    truth_path = tmp_path / 'truth.trees'
    full.dump(truth_path)
    mutation = tables.mutations[2]
    tables.mutations[2] = mutation.replace(derived_state='2')
    mutation = tables.mutations[0]
    tables.mutations[0] = mutation.replace(metadata={'mn': 20})
    stitched_path = tmp_path / 'stitched.trees'
    tables.tree_sequence().dump(stitched_path)
    runs = pd.DataFrame([
        {'dataset': 'simulation', 'comparison_method': method, 'dated_path': path,
         'ac_cutoff': cutoff, 'max_mismatches': budget, 'min_focal_region_size': None}
        for method, path, cutoff, budget in [
            ('full_hmm', full_path, None, None), ('stitched_greedy', stitched_path, 10, 0)]
    ])
    config = {'experiment_path': tmp_path / 'experiment.yaml',
              'experiment': {'datasets': [{'name': 'simulation', 'ts_path': str(truth_path)}]},
              'datasets': [{'name': 'simulation', 'analysis_span_bp': [0, 100]}],
              'dating': {'compare_full_datasets': ['simulation']}}
    monkeypatch.setattr(ch5_dating, 'load_configuration', lambda path: config)
    return runs


class TestStitchedMutationPairs:
    def test_pairs_posterior_means_at_all_frequencies_without_truth(self, matched_grid, tmp_path):
        (tmp_path / 'truth.trees').unlink()
        pairs = ch5_dating.stitched_mutation_age_pairs(matched_grid, tmp_path / 'config.yaml')
        assert pairs.comparison_method.tolist() == ['stitched_greedy', 'stitched_greedy']
        assert pairs.position.tolist() == [10, 20]
        assert pairs.full_hmm_age.tolist() == [10, 40]
        assert pairs.stitched_age.tolist() == [20, 40]
