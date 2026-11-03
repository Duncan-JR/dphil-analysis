"""Unit contracts for pt12's calibration and error attribution."""

import numpy as np
import pytest
import tskit
import zarr

import ch5_conditioning


class TestMismatchCalibration:
    def test_per_haplotype_conversion_has_no_diploid_factor(self):
        probability = ch5_conditioning.calibrate_mismatch(1e-5, 500_000, 5_000)
        assert probability == pytest.approx(0.001)

    def test_density_is_invariant_to_window_replication(self):
        small = ch5_conditioning.calibrate_mismatch(1e-5, 500_000, 5_000)
        large = ch5_conditioning.calibrate_mismatch(1e-5, 1_000_000, 10_000)
        assert small == large


class TestRecurrentErrorAttribution:
    @pytest.fixture
    def store(self, tmp_path):
        store = zarr.open_group(tmp_path / "input.zarr", mode="w", zarr_format=3)
        store.create_array("variant_position", data=np.array([10, 20, 30, 40]))
        store.create_array("sample_id", data=np.array(["person"], dtype="T"))
        store.create_array("variant_allele", data=np.array([["0", "1"]] * 4, dtype="T"))
        calls = np.array([[[0, 1]], [[1, 0]], [[1, 0]], [[0, 0]]], dtype=np.int8)
        store.create_array("call_genotype", data=calls)
        # One altered pre-phasing call at site 20, haplotype 0. Phasing moves
        # that altered call to observed haplotype 1. Site 40 has a second error.
        errors = np.zeros((4, 1, 2), dtype=bool)
        errors[1, 0, 0] = True
        errors[3, 0, 0] = True
        phase = np.zeros((4, 1), dtype=bool)
        phase[1, 0] = True
        store.create_array("call_genotype_error_mask", data=errors)
        store.create_array("call_genotype_phase", data=phase)
        return store

    @pytest.fixture
    def trees(self):
        tables = tskit.TableCollection(50)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        root = tables.nodes.add_row(time=10)
        other = tables.nodes.add_row(time=5)
        metadata = {"sample_id": "person", "ploidy_index": 0}
        first = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, metadata=metadata)
        metadata = {"sample_id": "person", "ploidy_index": 1}
        second = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, metadata=metadata)
        tables.edges.add_row(0, 50, root, other)
        tables.edges.add_row(0, 25, root, first)
        tables.edges.add_row(25, 50, other, first)
        tables.edges.add_row(0, 50, root, second)
        for position in [10, 20, 30, 40]:
            tables.sites.add_row(position, ancestral_state="0")
        # Sites 10 and 20 are recurrent. Only site 20 has an injected error.
        ancestral = tables.mutations.add_row(0, root, derived_state="1")
        tables.mutations.add_row(0, first, derived_state="0", parent=ancestral)
        ancestral = tables.mutations.add_row(1, root, derived_state="1")
        tables.mutations.add_row(1, second, derived_state="0", parent=ancestral)
        tables.mutations.add_row(2, first, derived_state="1")
        tables.sort()
        return tables.tree_sequence()

    def test_preserves_error_polarity_and_aligns_phasing(self, store):
        errors = ch5_conditioning.aligned_errors(store, np.array([1, 2]))
        assert errors.tolist() == [[[False, True]], [[False, False]]]

    def test_recurrence_overlap_counts_sites_and_sample_records_separately(self, trees, store):
        result = ch5_conditioning.error_overlap(trees, store)
        assert result["recurrent_only_sites"] == 1
        assert result["overlap_sites"] == 1
        assert result["error_only_sites"] == 1
        assert result["neither_sites"] == 1
        assert result["recurrent_sample_mutations"] == 2
        assert result["recurrent_sample_mutations_at_error_calls"] == 1
        assert result["recurrent_sample_mutations_at_other_calls"] == 1
        assert result["recurrent_reference_mutations"] == 2

    def test_copying_switches_and_mutations_remain_per_haplotype(self, trees):
        result = ch5_conditioning.copying_events(trees)
        assert result.sample_id.tolist() == ["person", "person"]
        assert result.ploidy_index.tolist() == [0, 1]
        assert result.switches.tolist() == [1, 0]
        assert result.mutations.tolist() == [2, 1]

    def test_called_alleles_are_matched_by_individual_and_ploidy(self, trees, store):
        result = ch5_conditioning.called_allele_check(trees, store)
        assert result == {"called_alleles_checked": 8, "allele_disagreements": 0}

    def test_reports_a_reconstruction_failure(self, trees, store):
        store["call_genotype"][3, 0, 1] = 1
        result = ch5_conditioning.called_allele_check(trees, store)
        assert result == {"called_alleles_checked": 8, "allele_disagreements": 1}
