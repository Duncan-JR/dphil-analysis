import importlib.util
import pathlib
import types
import unittest.mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def extractor():
    path = pathlib.Path(__file__).parents[1] / "scripts/ch3_anc_perf/extract_spans.py"
    spec = importlib.util.spec_from_file_location("extract_spans", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ancestor_outputs(tmp_path, monkeypatch, extractor):
    stores = {}
    for version, start in [("0.4.1", 0), ("0.5.1", 1)]:
        path = tmp_path / version / "n6404/ancestors.ancestors"
        path.parent.mkdir(parents=True)
        path.touch()
        focal_sites = np.empty(4, dtype=object)
        focal_sites[:] = [np.array([], dtype=int), np.array([], dtype=int),
                          np.array([1, 2]), np.array([3])]
        stores[str(path)] = types.SimpleNamespace(
            num_ancestors=4,
            sites_position=np.array([5, 10, 40, 70]),
            ancestors_start=np.array([0, 0, start, 3]),
            ancestors_end=np.array([4, 4, 3, 4]),
            ancestors_time=np.array([5, 4, 2, 1]),
            ancestors_focal_sites=focal_sites,
        )

    def load(path):
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = stores[path]
        return context

    # The analysis environment uses modern tsinfer; the script runs against
    # the legacy benchmark checkout whose metadata interface is represented here.
    monkeypatch.setattr(extractor.tsinfer, "AncestorData",
                        types.SimpleNamespace(load=load), raising=False)
    return tmp_path


class TestExtractSpans:
    def test_spans_use_first_and_last_included_inference_sites(self, extractor, ancestor_outputs):
        frame = extractor.extract_spans(ancestor_outputs)
        expected = pd.DataFrame({
            "version": ["0.4.1"] * 3 + ["0.5.1"] * 3,
            "num_samples": [6404] * 6,
            "inferred_node": [2, 2, 3, 2, 2, 3],
            "inf_focal_site": [1, 2, 3, 1, 2, 3],
            "focal_position": [10, 40, 70, 10, 40, 70],
            "inferred_time": [2, 2, 1, 2, 2, 1],
            "inferred_span": [35, 35, 0, 30, 30, 0],
            "inferred_site_span": [3, 3, 1, 2, 2, 1],
        })
        pd.testing.assert_frame_equal(frame, expected)

    def test_missing_stores(self, extractor, tmp_path):
        with pytest.raises(FileNotFoundError, match="No generated ancestor stores"):
            extractor.extract_spans(tmp_path)

    def test_cli_writes_dataframe(self, extractor, ancestor_outputs, monkeypatch):
        monkeypatch.setattr("sys.argv", ["extract_spans.py", str(ancestor_outputs)])
        extractor.main()
        frame = pd.read_csv(ancestor_outputs / "ancestor_spans.csv")
        expected = extractor.extract_spans(ancestor_outputs)
        pd.testing.assert_frame_equal(frame, expected)
