"""Small map-calibrated matching experiments for stitching pt12.

Prepare compact VCZ inputs, run the existing tsinfer-match-eval Snakefile, and
summarise recurrent sites and sample copying events. The calibration site axis
precedes selected-cohort singleton and polymorphism filtering: see
:func:`calibrate_mismatch` and the discussion linked from the notebook.
"""

import dataclasses
import json
import logging
import os
import pathlib
import subprocess

import numpy as np
import pandas as pd
import tsinfer
import tskit
import yaml
import zarr

import ch5_gapfill

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentTables:
    """Tables with shared dataset/method keys, saved alongside the experiment."""

    calibration: pd.DataFrame
    recurrence: pd.DataFrame
    overlap: pd.DataFrame
    haplotypes: pd.DataFrame
    sample_means: pd.DataFrame
    allele_checks: pd.DataFrame


def load_config(path: pathlib.Path) -> dict:
    """Resolve experiment paths relative to its YAML file."""
    config = yaml.safe_load(path.read_text())
    for key in ("pipeline_dir", "output_dir", "recombination_map"):
        config[key] = (path.parent / config[key]).resolve()
    for dataset in config["datasets"]:
        for key in ("zarr_path", "sample_list", "truth_path", "raw_zarr_path"):
            if dataset[key] is not None:
                dataset[key] = (path.parent / dataset[key]).resolve()
    config["config_path"] = path.resolve()
    return config


def calibrate_mismatch(epsilon: float, span_bp: int, num_sites: int) -> float:
    """Convert one-haplotype errors/bp to errors per included variant site.

    ``num_sites`` counts the fixed, quality-filtered regional source axis,
    before the matching cohort's singleton/polymorphism filters. Transferring
    this average to inference sites assumes the same per-site error profile.
    This is the linear expectation conversion, not a Poisson event probability.
    """
    probability = epsilon * span_bp / num_sites
    if not 0 < probability < 0.5:
        raise ValueError("The calibrated biallelic mismatch must be between 0 and 0.5")
    return probability


def _write_array(group, name, values, dimensions, config):
    chunks = []
    for size, dimension in zip(values.shape, dimensions, strict=True):
        chunk = size
        if dimension == "variants":
            chunk = min(size, config["storage"]["variant_chunk"])
        elif dimension == "samples":
            chunk = min(size, config["storage"]["sample_chunk"])
        chunks.append(chunk)
    array = group.create_array(name, data=values, chunks=tuple(chunks))
    array.attrs["_ARRAY_DIMENSIONS"] = dimensions


def _input_selection(dataset):
    sample_ids = None
    if dataset["sample_list"] is not None:
        sample_ids = pd.read_csv(dataset["sample_list"], header=None)[0].tolist()
    return {"source": str(dataset["zarr_path"]), "sample_ids": sample_ids,
            "region": dataset["region"], "ancestral_field": dataset["ancestral_field"],
            "exclude_masks": dataset["exclude_masks"]}


def prepare_dataset(config: dict, dataset: dict) -> dict:
    """Subset the source once, retaining coordinates and simulation error flags.

    Read only regional rows and selected individuals. All variant-dimensioned
    arrays in the compact store share one physical chunk size, as required by
    native vcztools readers. Source stores are opened read-only.
    """
    root = config["output_dir"] / dataset["name"]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "input_manifest.json"
    selection = _input_selection(dataset)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["input_selection"] != selection:
            raise ValueError("Input selection changed; use a new output_dir for this experiment")
        return manifest
    source = zarr.open_group(dataset["zarr_path"], mode="r")
    positions = source["variant_position"][:]
    left, right = dataset["region"]
    selected = (positions >= left) & (positions < right)
    for name in dataset["exclude_masks"]:
        selected &= ~source[name][:].astype(bool)
    rows = np.flatnonzero(selected)
    alleles = source["variant_allele"].oindex[rows, :]
    ancestral = source[dataset["ancestral_field"]].oindex[rows]
    known_ancestral = np.any(alleles == ancestral[:, None], axis=1)
    rows = rows[known_ancestral]
    alleles = alleles[known_ancestral]
    ancestral = ancestral[known_ancestral]
    selected_positions = positions[rows]
    assert np.all(np.diff(selected_positions) > 0)
    identifiers = source["sample_id"][:]
    if dataset["sample_list"] is None:
        columns = np.arange(len(identifiers))
    else:
        names = pd.read_csv(dataset["sample_list"], header=None)[0].astype(str)
        lookup = {name: index for index, name in enumerate(identifiers)}
        columns = np.array([lookup[name] for name in names])
    calls = source["call_genotype"].oindex[rows, columns, :]
    contigs = source["variant_contig"].oindex[rows]
    assert np.all(contigs == contigs[0])
    contig = int(contigs[0])
    output = root / "input.zarr"
    store = zarr.open_group(output, mode="w", zarr_format=3)
    store.attrs.update({"vcf_zarr_version": "0.4", "source": str(dataset["zarr_path"])})
    arrays = {
        "variant_position": (selected_positions, ["variants"]),
        "variant_allele": (alleles, ["variants", "alleles"]),
        dataset["ancestral_field"]: (ancestral, ["variants"]),
        "variant_contig": (np.zeros(len(rows), dtype=np.int8), ["variants"]),
        "contig_id": (source["contig_id"][contig:contig + 1], ["contigs"]),
        "contig_length": (source["contig_length"][contig:contig + 1], ["contigs"]),
        "sample_id": (identifiers[columns], ["samples"]),
        "call_genotype": (calls, ["variants", "samples", "ploidy"]),
        "call_genotype_phased": (source["call_genotype_phased"].oindex[rows, columns].astype(bool),
                                 ["variants", "samples"]),
    }
    if dataset["truth_path"] is not None:
        for name, dimensions in (
            ("call_genotype_error_mask", ["variants", "samples", "ploidy"]),
            ("call_genotype_phase", ["variants", "samples"]),
        ):
            selection = (rows, columns, slice(None)) if len(dimensions) == 3 else (rows, columns)
            arrays[name] = (source[name].oindex[selection].astype(bool), dimensions)
    for name, (values, dimensions) in arrays.items():
        _write_array(store, name, np.asarray(values), dimensions, config)
    zarr.consolidate_metadata(output)
    probability = calibrate_mismatch(
        config["epsilon_per_haplotype_bp"], right - left, len(rows)
    )
    derived_code = np.where(ancestral == alleles[:, 0], 1, 0)
    derived_count = (calls == derived_code[:, None, None]).sum(axis=(1, 2))
    called_count = (calls >= 0).sum(axis=(1, 2))
    segregating = (derived_count > 0) & (derived_count < called_count)
    inference = segregating & (derived_count > 1)
    manifest = {
        "input_selection": selection,
        "dataset": dataset["name"], "label": dataset["label"],
        "source": str(dataset["zarr_path"]),
        "sample_list": str(dataset["sample_list"]) if dataset["sample_list"] is not None else None,
        "left_bp": left, "right_bp": right, "span_bp": right - left,
        "num_individuals": len(columns), "num_haplotypes": calls.shape[1] * calls.shape[2],
        "calibration_sites": len(rows), "segregating_sites": int(segregating.sum()),
        "expected_inference_sites": int(inference.sum()),
        "epsilon_per_haplotype_bp": config["epsilon_per_haplotype_bp"],
        "sites_per_bp": len(rows) / (right - left), "mismatch_probability": probability,
    }
    np.savez(root / "source_indices.npz", rows=rows, columns=columns)
    pd.Series(identifiers[columns]).to_csv(root / "selected_samples.csv", index=False, header=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare(config: dict) -> pd.DataFrame:
    """Write one pipeline configuration per independently calibrated source."""
    records = []
    settings = config["pipeline"]
    for dataset in config["datasets"]:
        record = prepare_dataset(config, dataset)
        root = config["output_dir"] / dataset["name"]
        record["epsilon_per_haplotype_bp"] = config["epsilon_per_haplotype_bp"]
        record["mismatch_probability"] = calibrate_mismatch(
            config["epsilon_per_haplotype_bp"], record["span_bp"], record["calibration_sites"]
        )
        (root / "input_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        pipeline = dict(settings)
        del pipeline["af_cutoff"], pipeline["min_region_size_kb"]
        pipeline.update({
            "config_name": dataset["name"], "data_dir": str(root),
            "progress_dir": str(root / "progress"), "temp_dir": str(root / "temporary"),
            "hmm": {"recombination": None, "recombination_rate": str(config["recombination_map"]),
                    "mismatch": record["mismatch_probability"]},
            "performance": None,
            "datasets": [{
                "name": dataset["name"], "zarr_path": str(root / "input.zarr"),
                "ts_path": None, "ancestral_state": {"field": dataset["ancestral_field"]},
                "af_cutoff": settings["af_cutoff"],
                "min_region_size_kb": settings["min_region_size_kb"],
            }],
        })
        (root / "temporary").mkdir(exist_ok=True)
        (root / "config.yaml").write_text(yaml.safe_dump(pipeline, sort_keys=False))
        records.append({key: value for key, value in record.items() if key != "input_selection"})
    calibration = pd.DataFrame(records)
    calibration.to_csv(config["output_dir"] / "calibration.csv", index=False)
    return calibration


def run_pipeline(config: dict) -> None:
    """Run the unmodified matching-evaluation DAG inside each experiment folder.

    The sibling project's environment supplies Snakemake and its own libraries.
    Logs, native configs, inputs, references and outputs stay in this repository.
    Snakemake schedules native threads and greedy multiprocessing workers.
    """
    pipeline_dir = config["pipeline_dir"]
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(config["output_dir"] / "uv_cache")
    environment["NUMBA_CACHE_DIR"] = str(config["output_dir"] / "numba_cache")
    environment["PYTHONPATH"] = str(pipeline_dir)
    revision = {}
    for name, path in (("tsinfer_match_eval", pipeline_dir),
                       ("tsinfer", pathlib.Path(tsinfer.__file__).resolve().parents[1])):
        revision[name] = {
            "commit": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
            "diff": subprocess.check_output(["git", "-C", str(path), "diff"], text=True),
        }
    (config["output_dir"] / "code_versions.json").write_text(json.dumps(revision, indent=2) + "\n")
    for dataset in config["datasets"]:
        root = config["output_dir"] / dataset["name"]
        source_cache = root / "source_cache"
        source_cache.mkdir(exist_ok=True)
        command = ["uv", "run", "--no-sync", "--project", str(pipeline_dir),
                   "snakemake", "--snakefile", str(pipeline_dir / "Snakefile"),
                   "--directory", str(root), "--cores", "all",
                   "--runtime-source-cache-path", str(source_cache)]
        logger.info("Running %s; log: %s", dataset["name"], root / "pipeline.log")
        with (root / "pipeline.log").open("w") as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)


def aligned_errors(store: zarr.Group, rows: np.ndarray) -> np.ndarray:
    """Return injected genotype errors on the observed, post-phasing call axis.

    This snapshot stores True for altered calls, before phase switching, as
    verified against its raw companion by :func:`verify_error_mask`. Swap the
    ploidy columns wherever the stored phase is 1. Phase switches themselves
    are not labelled as injected genotype errors.
    """
    errors = store["call_genotype_error_mask"].oindex[rows, :, :].astype(bool)
    phase = store["call_genotype_phase"].oindex[rows, :].astype(bool)
    return np.where(phase[:, :, None], errors[:, :, ::-1], errors)


def verify_error_mask(config: dict, dataset: dict) -> None:
    """Check this saved snapshot's mask polarity and phase axis against raw calls.

    The current generator's mask convention differs from this October snapshot.
    Use the saved data, not today's generator implementation, as the evidence.
    This read-only check runs on the compact selected region and cohort.
    """
    root = config["output_dir"] / dataset["name"]
    store = zarr.open_group(root / "input.zarr", mode="r")
    raw = zarr.open_group(dataset["raw_zarr_path"], mode="r")
    source = zarr.open_group(dataset["zarr_path"], mode="r")
    with np.load(root / "source_indices.npz") as indices:
        rows = indices["rows"]
        columns = indices["columns"]
    np.testing.assert_array_equal(raw["variant_position"].oindex[rows], store["variant_position"][:])
    np.testing.assert_array_equal(raw["sample_id"].oindex[columns], store["sample_id"][:])
    expected = raw["call_genotype"].oindex[rows, columns, :]
    phase = store["call_genotype_phase"][:].astype(bool)
    expected_phased = np.where(phase[:, :, None], expected[:, :, ::-1], expected)
    observed = store["call_genotype"][:]
    errors = aligned_errors(store, np.arange(len(rows)))
    np.testing.assert_array_equal(errors, observed != expected_phased)
    source_errors = source["call_genotype_error_mask"].oindex[rows, :, :].astype(bool)
    counts = source["variant_genotype_error_count"].oindex[rows]
    np.testing.assert_array_equal(source_errors.sum(axis=(1, 2)), counts)
    report = {"mask_true_means": "injected_genotype_error_before_phasing",
              "verified_against": str(dataset["raw_zarr_path"]),
              "selected_error_calls": int(errors.sum()),
              "phase_swapped_diplotypes": int(phase.sum()),
              "error_mask_matches_observed_minus_phased_truth": True}
    (root / "error_mask_check.json").write_text(json.dumps(report, indent=2) + "\n")


def copying_events(ts: tskit.TreeSequence) -> pd.DataFrame:
    """Count sample mutations and parent changes at consecutive inference sites.

    Boundary edges in the unobserved chromosome flanks do not count. Means are
    formed per diploid individual by summing its two haplotypes first.
    """
    samples = ts.samples()
    sample_index = {int(node): index for index, node in enumerate(samples)}
    parents = np.full((len(samples), ts.num_sites), tskit.NULL, dtype=np.int32)
    for edge in ts.edges():
        if edge.child in sample_index:
            left = np.searchsorted(ts.sites_position, edge.left)
            right = np.searchsorted(ts.sites_position, edge.right)
            parents[sample_index[edge.child], left:right] = edge.parent
    mutations = np.bincount(ts.mutations_node, minlength=ts.num_nodes)
    result = []
    for index, node_id in enumerate(samples):
        metadata = ts.node(node_id).metadata
        path = parents[index]
        called = path >= 0
        adjacent = called[1:] & called[:-1]
        switches = adjacent & (path[1:] != path[:-1])
        result.append({"sample_id": metadata["sample_id"],
                       "ploidy_index": metadata["ploidy_index"],
                       "switches": int(switches.sum()), "mutations": int(mutations[node_id])})
    return pd.DataFrame(result)


def called_allele_check(ts: tskit.TreeSequence, store: zarr.Group) -> dict:
    """Count output/input allele differences without adding a validation plot.

    Match individuals and haplotypes by node metadata, not table row order.
    A failed reconstruction is retained as an experimental finding rather than
    silently treating a matching mutation count as a complete representation.
    """
    rows = np.searchsorted(store["variant_position"][:], ts.sites_position)
    calls = store["call_genotype"].oindex[rows, :, :]
    alleles = store["variant_allele"].oindex[rows, :]
    lookup = {name: column for column, name in enumerate(store["sample_id"][:])}
    nodes = [ts.node(node_id) for node_id in ts.samples()]
    columns = [lookup[node.metadata["sample_id"]] for node in nodes]
    ploidies = [node.metadata["ploidy_index"] for node in nodes]
    expected = calls[:, columns, ploidies]
    evaluated = 0
    disagreements = 0
    for variant in ts.variants(isolated_as_missing=False):
        site = variant.site.id
        called = expected[site] >= 0
        expected_alleles = alleles[site, expected[site, called]]
        states = np.asarray(variant.alleles, dtype=object)
        observed_alleles = states[variant.genotypes[called]]
        disagreements += int(np.count_nonzero(observed_alleles != expected_alleles))
        evaluated += int(called.sum())
    return {"called_alleles_checked": evaluated, "allele_disagreements": disagreements}


def error_overlap(ts: tskit.TreeSequence, store: zarr.Group) -> dict:
    """Intersect recurrence with injected errors at site and sample-mutation levels.

    Internal/reference mutation records cannot be assigned to a sample call.
    Report them separately rather than calling every event at an error-bearing
    site an error. Even a direct overlap is an association, not causal proof.
    """
    positions = store["variant_position"][:]
    rows = np.searchsorted(positions, ts.sites_position)
    np.testing.assert_array_equal(positions[rows], ts.sites_position)
    errors = aligned_errors(store, rows)
    error_sites = np.any(errors, axis=(1, 2))
    counts = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    recurrent = counts > 1
    sample_ids = store["sample_id"][:]
    sample_columns = {name: index for index, name in enumerate(sample_ids)}
    recurrent_sample = 0
    recurrent_sample_error = 0
    recurrent_internal = 0
    for mutation in ts.mutations():
        if not recurrent[mutation.site]:
            continue
        node = ts.node(mutation.node)
        if not node.is_sample():
            recurrent_internal += 1
            continue
        recurrent_sample += 1
        column = sample_columns[node.metadata["sample_id"]]
        ploidy = node.metadata["ploidy_index"]
        recurrent_sample_error += int(errors[mutation.site, column, ploidy])
    return {
        "num_sites": ts.num_sites,
        "recurrent_only_sites": int(np.count_nonzero(recurrent & ~error_sites)),
        "overlap_sites": int(np.count_nonzero(recurrent & error_sites)),
        "error_only_sites": int(np.count_nonzero(~recurrent & error_sites)),
        "neither_sites": int(np.count_nonzero(~recurrent & ~error_sites)),
        "recurrent_sample_mutations": recurrent_sample,
        "recurrent_sample_mutations_at_error_calls": recurrent_sample_error,
        "recurrent_sample_mutations_at_other_calls": recurrent_sample - recurrent_sample_error,
        "recurrent_reference_mutations": recurrent_internal,
    }


def analyse(config: dict) -> ExperimentTables:
    """Summarise raw trees on identical inference positions, without dating."""
    calibration = []
    recurrence = []
    overlaps = []
    haplotypes = []
    allele_checks = []
    settings = config["pipeline"]
    for dataset in config["datasets"]:
        name = dataset["name"]
        root = config["output_dir"] / name
        manifest = json.loads((root / "input_manifest.json").read_text())
        del manifest["input_selection"]
        full = tskit.load(root / "matches" / f"{name}_inferred_samples_raw.trees")
        greedy_dir = root / "matches" / f"af_{settings['af_cutoff']}"
        greedy_dir /= f"min_region_kb_{settings['min_region_size_kb']}"
        greedy = tskit.load(greedy_dir / f"{name}_inferred_greedy_samples_raw.trees")
        np.testing.assert_array_equal(full.sites_position, greedy.sites_position)
        assert full.num_sites == manifest["expected_inference_sites"]
        assert full.num_samples == greedy.num_samples == manifest["num_haplotypes"]
        native_path = root / "configs" / f"{name}_ancestor_inference.toml"
        native = tsinfer.config.Config.from_toml(native_path)
        hmm = tsinfer.matching.hmm_parameters(native.match.sources["samples"], full.sites_position)
        np.testing.assert_allclose(hmm.mismatch, manifest["mismatch_probability"])
        map_path = native.match.sources["samples"].recombination_rate
        genetic_positions = tsinfer.matching.read_genetic_map(map_path, full.sites_position)
        genetic_distance = np.diff(genetic_positions)
        expected_recombination = -np.expm1(-2 * genetic_distance) / 2
        np.testing.assert_allclose(hmm.recombination[1:], expected_recombination)
        manifest.update({"inference_sites": full.num_sites,
                         "median_genetic_distance_morgans": float(np.median(genetic_distance)),
                         "median_recombination_probability": float(np.median(hmm.recombination[1:])),
                         "min_recombination_probability": float(np.min(hmm.recombination[1:])),
                         "max_recombination_probability": float(np.max(hmm.recombination[1:]))})
        calibration.append(manifest)
        store = zarr.open_group(root / "input.zarr", mode="r")
        if dataset["raw_zarr_path"] is not None:
            verify_error_mask(config, dataset)
        trees = {"full_hmm": full, "greedy": greedy}
        if dataset["truth_path"] is not None:
            trees["truth"] = tskit.load(dataset["truth_path"])
        for method, ts in trees.items():
            identity = {"dataset": name, "label": dataset["label"], "method": method}
            recurrence.append({**identity, **ch5_gapfill.recurrent_site_counts(ts, full.sites_position)})
            if method == "truth":
                continue
            events = copying_events(ts).assign(**identity)
            haplotypes.append(events)
            allele_checks.append({**identity, **called_allele_check(ts, store)})
            if dataset["truth_path"] is not None:
                overlaps.append({**identity, **error_overlap(ts, store)})
    haplotype_table = pd.concat(haplotypes, ignore_index=True)
    keys = ["dataset", "label", "method"]
    per_sample = haplotype_table.groupby([*keys, "sample_id"], sort=False)[["switches", "mutations"]].sum()
    sample_means = per_sample.groupby(keys, sort=False).mean().reset_index()
    tables = ExperimentTables(pd.DataFrame(calibration), pd.DataFrame(recurrence),
                              pd.DataFrame(overlaps), haplotype_table, sample_means,
                              pd.DataFrame(allele_checks))
    for field in dataclasses.fields(tables):
        getattr(tables, field.name).to_csv(config["output_dir"] / f"{field.name}.csv", index=False)
    return tables
