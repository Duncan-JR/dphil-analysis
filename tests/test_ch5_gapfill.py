import ch5_gapfill
import ch5_perf_scaling
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
import tskit


class TestRecurrentSites:
    @pytest.fixture
    def trees(self):
        tables = tskit.TableCollection(50)
        root = tables.nodes.add_row(time=10)
        sample = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.edges.add_row(0, 50, root, sample)
        for position in [10, 20, 30, 40]:
            tables.sites.add_row(position, ancestral_state='0')
        tables.mutations.add_row(0, sample, derived_state='1')
        first = tables.mutations.add_row(1, root, derived_state='1')
        tables.mutations.add_row(1, sample, derived_state='0', parent=first)
        first = tables.mutations.add_row(2, root, derived_state='1')
        second = tables.mutations.add_row(2, root, derived_state='0', parent=first)
        tables.mutations.add_row(2, sample, derived_state='1', parent=second)
        return tables.tree_sequence()

    def test_counts_include_back_mutations_and_zero_mutation_sites(self, trees):
        result = ch5_gapfill.recurrent_site_counts(trees, trees.sites_position)
        assert result['num_sites'] == 4
        assert result['recurrent_sites'] == 2
        assert result['recurrent_site_percentage'] == 50
        assert result['mutations_at_recurrent_sites'] == 5
        assert result['mean_mutations_per_recurrent_site'] == 2.5
        assert result['additional_mutations'] == 3

    def test_restricts_truth_to_inference_positions(self, trees):
        result = ch5_gapfill.recurrent_site_counts(trees, np.array([10, 20]))
        assert result['recurrent_sites'] == 1
        assert result['recurrent_site_percentage'] == 50
        assert result['mean_mutations_per_recurrent_site'] == 2

    def test_no_recurrent_sites_has_undefined_conditional_mean(self, trees):
        result = ch5_gapfill.recurrent_site_counts(trees, np.array([10, 40]))
        assert result['recurrent_sites'] == 0
        assert result['recurrent_site_percentage'] == 0
        assert np.isnan(result['mean_mutations_per_recurrent_site'])


class TestStageAccounting:
    @pytest.fixture
    def inputs(self, tmp_path):
        root = tmp_path / 'performance' / 'example'
        root.mkdir(parents=True)
        stages = pd.DataFrame([
            {'stage': 'setup', 'wall_s': 2},
            {'stage': 'drain', 'wall_s': 5},
            {'stage': 'nested_wait', 'wall_s': 4},
            {'stage': 'extension', 'service_s': 3, 'thread_cpu_s': 2},
            {'stage': 'gap_native', 'service_s': 10, 'thread_cpu_s': 8},
            {'stage': 'gap_conversion', 'service_s': 2, 'thread_cpu_s': 1},
            {'stage': 'gap_total', 'service_s': 13, 'thread_cpu_s': 10},
        ])
        stages['run_id'] = 'greedy_run'
        stages.to_csv(root / 'stages.csv', index=False)
        config = {'data_dir': tmp_path, 'cache_dir': tmp_path,
                  'experiment': {'performance': {'experiment_id': 'example'}},
                  'plot': {'wall_stages': {'setup': {}, 'drain': {}, 'other': {}},
                           'worker_stages': {'extension': {}, 'gap_native': {},
                                             'gap_conversion': {}, 'gap_other': {}}},
                  'stages': {'gap_parent': 'gap_total',
                             'gap_children': ['gap_native', 'gap_conversion']}}
        runs = pd.DataFrame([{'run_id': 'greedy_run', 'dataset': 'example',
                              'num_samples': 2, 'matched_haplotypes': 4,
                              'method': 'greedy', 'af_cutoff': 0.1, 'wall_s': 8}])
        return {'config': config, 'runs': runs}

    def test_wall_stack_excludes_nested_timers(self, inputs):
        result = ch5_gapfill.stage_breakdown(**inputs)
        assert result.wall.wall_s.sum() == 8
        assert result.wall.wall_percentage.sum() == 100
        assert result.wall.set_index('stage').loc['other', 'wall_s'] == 1
        assert 'nested_wait' not in set(result.wall.stage)

    def test_gap_children_replace_inclusive_parent(self, inputs):
        result = ch5_gapfill.stage_breakdown(**inputs)
        totals = result.worker.groupby('metric').seconds.sum()
        assert totals['service_s'] == 16
        assert totals['thread_cpu_s'] == 12
        averages = result.worker.groupby('metric').ms_per_haplotype.sum()
        assert averages['service_s'] == 4000
        assert averages['thread_cpu_s'] == 3000
        residuals = result.worker.loc[result.worker.stage.eq('gap_other')]
        assert residuals.seconds.eq(1).all()


class TestMutationRankCorrelation:
    @pytest.fixture
    def inputs(self, tmp_path):
        full_ages = [1, 2, 3, 9]
        greedy_ages = [4, 3, 2, 1]
        rows = []
        for method, ages in [('full_hmm', full_ages), ('greedy', greedy_ages)]:
            tables = tskit.TableCollection(50)
            tables.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
            root = tables.nodes.add_row(time=10)
            sample = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
            tables.edges.add_row(0, 50, root, sample)
            for position, age in zip([10, 20, 30, 40], ages, strict=True):
                site = tables.sites.add_row(position, ancestral_state='0')
                tables.mutations.add_row(site, sample, derived_state='1', metadata={'mn': age})
            path = tmp_path / f'{method}.trees'
            tables.tree_sequence().dump(path)
            rows.append({'dataset': 'example', 'num_samples': 1, 'method': method,
                         'af_cutoff': np.nan if method == 'full_hmm' else 0.1,
                         'dated_path': str(path)})
        config = {'data_dir': tmp_path, 'cache_dir': tmp_path,
                  'frequency_bins': {'cutoffs': [0.5], 'labels': ['low', 'high']}}
        return {'config': config, 'dataset': {'name': 'example', 'truth': None},
                'dated': pd.DataFrame(rows)}

    def test_spearman_uses_ranks_on_same_mutation_intersection(self, inputs):
        result = ch5_perf_scaling.age_comparison(**inputs)
        overall = result.loc[result.frequency_bin.eq('all')].iloc[0]
        assert overall.matched_mutations == 4
        assert overall.spearman_r == pytest.approx(-1)
        assert overall.pearson_r == pytest.approx(stats.pearsonr([1, 2, 3, 9], [4, 3, 2, 1]).statistic)
        assert np.isnan(result.loc[result.frequency_bin.eq('low'), 'spearman_r'].item())


class TestDatasetSpecificFilters:
    @pytest.fixture
    def config(self, tmp_path):
        table = pd.DataFrame([
            {'dataset': 'small', 'af_cutoff': 0.1, 'min_region_size_kb': 25,
             'min_focal_region_size': 84, 'num_samples': 1, 'unmatched_sites_mean': 0},
            {'dataset': 'large', 'af_cutoff': 0.1, 'min_region_size_kb': 100,
             'min_focal_region_size': 1751, 'num_samples': 2, 'unmatched_sites_mean': 0},
        ])
        table.to_csv(tmp_path / 'summary.csv', index=False)
        datasets = []
        for row in table.itertuples():
            cohort = [{'source': 'samples', 'sample_id': f'sample_{index}', 'ploidy_index': phase}
                      for index in range(row.num_samples) for phase in [0, 1]]
            run = {'af_cutoff': row.af_cutoff, 'min_region_size_kb': row.min_region_size_kb,
                   'min_focal_region_size': row.min_focal_region_size}
            datasets.append({'name': row.dataset, 'cohort': cohort, 'num_sites': 2000,
                             'analysis_span_bp': 1000000, 'truth': None, 'runs': [run]})
        return {'data_dir': tmp_path, 'cache_dir': tmp_path,
                'manifest': {'datasets': datasets, 'tables': {'run_summary': 'summary.csv'}}}

    def test_matches_each_dataset_to_its_archived_filter(self, config):
        result = ch5_perf_scaling.diagnostic_summary(config)
        assert result.min_region_size_kb.tolist() == [25, 100]

    def test_rejects_mismatched_filter(self, config):
        config['manifest']['datasets'][0]['runs'][0]['min_region_size_kb'] = 50
        with pytest.raises(AssertionError):
            ch5_perf_scaling.diagnostic_summary(config)
