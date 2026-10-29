import csv
import importlib.util
import json
import pathlib
import sys

import pytest
import tskit


@pytest.fixture(scope="module")
def extractor():
    folder = pathlib.Path(__file__).resolve().parents[1]
    path = folder / "scripts/ch3_tgp_perf/extract_1kgp_perf.py"
    spec = importlib.util.spec_from_file_location("extract_1kgp_perf", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completed_runs(tmp_path, extractor):
    config = extractor.Config()
    for index, (folder, version) in enumerate(config.versions.items()):
        tables = tskit.TableCollection(100)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(time=1)
        tables.edges.add_row(0, 100, parent=2, child=0)
        tables.edges.add_row(0, 100, parent=2, child=1)
        site = tables.sites.add_row(50, ancestral_state="0")
        tables.mutations.add_row(site, node=0, derived_state="1")
        actual_version = version if index == 0 else "0.5.2.dev1"
        for stage, command in enumerate(config.commands, start=1):
            record = {
                "software": {"name": "tsinfer", "version": actual_version},
                "parameters": {"command": command},
                "resources": {
                    "elapsed_time": stage * 10 + index,
                    "max_memory": 1024**3,
                    "sys_time": stage,
                    "user_time": stage * 20,
                },
                "environment": {"os": {"node": "benchmark_node"}},
            }
            tables.provenances.add_row(record=json.dumps(record))
        raw = tmp_path / folder / 'trees/cohort/cohort-raw.trees'
        raw.parent.mkdir(parents=True)
        ts = tables.tree_sequence()
        ts.dump(raw)
        # The dated output must not be counted as a second benchmark run.
        ts.dump(raw.with_name('cohort-dated.trees'))
    return tmp_path


class TestExtract1kgpPerf:
    def test_stage_metrics_and_backport_version_are_preserved(self, completed_runs, extractor):
        config = extractor.Config()
        output = completed_runs / config.output_name
        extractor.extract_runs(completed_runs, output, config)
        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 6
        assert [row['command'] for row in rows] == list(config.commands) * 2
        assert [row['elapsed_time'] for row in rows] == ['10', '20', '30', '11', '21', '31']
        assert rows[3]['version'] == '0.5.1'
        assert rows[3]['provenance_version'] == '0.5.2.dev1'
        assert rows[0]['num_samples'] == '2'
        assert rows[0]['num_sites'] == '1'
        assert rows[0]['num_mutations'] == '1'
        assert rows[0]['num_edges'] == '2'
        assert rows[0]['sequence_length'] == '100.0'
        assert rows[0]['max_memory'] == str(1024**3)
        assert rows[0]['num_threads'] == '196'
        assert rows[0]['hostname'] == 'benchmark_node'

    def test_missing_run_does_not_write_a_partial_csv(self, tmp_path, extractor):
        output = tmp_path / 'perf.csv'
        with pytest.raises(ValueError, match='Expected one raw sample TS'):
            extractor.extract_runs(tmp_path, output, extractor.Config())
        assert not output.exists()
