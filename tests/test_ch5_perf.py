"""Check per-sample coverage and worker-time averaging."""

import ch5_perf
import numpy as np
import pandas as pd
import pytest
import tskit


@pytest.fixture
def sample_inputs(tmp_path):
    tables = tskit.TableCollection(sequence_length=100)
    for position in [10, 20, 40, 80]:
        tables.sites.add_row(position, ancestral_state='0')
    reference_path = tmp_path / 'reference.trees'
    tables.tree_sequence().dump(reference_path)
    mask_path = tmp_path / 'mask.npz'
    np.savez(mask_path, sample_id=['a', 'a'], ploidy_index=[0, 1],
             offsets=[0, 2, 4], left_site_index=[0, 2, 0, 1],
             right_site_index=[2, 4, 1, 4], is_gap=[False, True, False, True])
    population_path = tmp_path / 'populations.csv'
    pd.DataFrame({'sample_id': ['a'], 'population': ['CEU']}).to_csv(
        population_path, index=False)
    return {'mask_path': mask_path, 'reference_path': reference_path,
            'population_path': population_path,
            'settings': {'analysis_span_bp': [10, 90], 'population_column': 'population'},
            'resident_runs': pd.DataFrame({'method': ['greedy', 'greedy'], 'repeat': [0, 1]})}


@pytest.fixture
def worker_times():
    return pd.DataFrame({'method': ['greedy'] * 4, 'repeat': [0, 0, 1, 1],
                         'sample_id': ['a'] * 4, 'ploidy_index': [0, 1, 0, 1],
                         'service_s': [0.1, 0.2, 0.3, 0.6]})


class TestPerSampleData:
    def test_averages_trials_and_haplotypes(self, sample_inputs, worker_times):
        samples = ch5_perf.per_sample_data(worker_times, **sample_inputs)
        assert len(samples) == 1
        sample = samples.iloc[0]
        assert sample.sample_id == 'a'
        assert sample.population == 'CEU'
        assert sample.worker_time_ms == pytest.approx(300)
        assert sample.focal_coverage_percentage == pytest.approx(25)

    def test_rejects_missing_haplotype_trial(self, sample_inputs, worker_times):
        with pytest.raises(AssertionError):
            ch5_perf.per_sample_data(worker_times.iloc[:-1], **sample_inputs)
