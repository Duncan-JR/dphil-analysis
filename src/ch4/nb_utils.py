"""Chapter 4 notebook preparation; methods live in the retained libraries."""

from __future__ import annotations

import logging
import concurrent.futures as cf
import contextlib
import dataclasses
import json
import multiprocessing
import pathlib
import time

import numpy as np
import msprime
import pandas as pd
import tskit
import zarr

from tsinfer import ancestors as tsinfer_ancestors
from tsinfer import config as tsinfer_config
from tsinfer import matching as tsinfer_matching
from tsinfer import pipeline as tsinfer_pipeline

from . import error_estimation, error_validation

logger = logging.getLogger(__name__)


def fit_error_model(summary, *, pi, config):
    """Delegate to :func:`error_estimation.fit_error_model`."""
    return error_estimation.fit_error_model(summary=summary, pi=pi, config=config)


def estimate_error_rate(
    zarr_path,
    *,
    recombination,
    config,
    pi=None,
    fixed_doubletons_path=None,
    variant_mask_name=None,
):
    """Delegate to :func:`error_estimation.estimate_error_rate`."""
    return error_estimation.estimate_error_rate(
        zarr_path=zarr_path,
        recombination=recombination,
        config=config,
        pi=pi,
        fixed_doubletons_path=fixed_doubletons_path,
        variant_mask_name=variant_mask_name,
    )


def doubleton_features(profiles):
    """Delegate to :func:`error_validation.doubleton_features`."""
    return error_validation.doubleton_features(profiles=profiles)


def fit_doubleton_classifier(tier, training_profiles, *, target_recall=0.95):
    """Delegate to :func:`error_validation.fit_doubleton_classifier`."""
    return error_validation.fit_doubleton_classifier(
        tier=tier, training_profiles=training_profiles, target_recall=target_recall
    )


def doubleton_scores(classifier, features):
    """Delegate to :func:`error_validation.doubleton_scores`."""
    return error_validation.doubleton_scores(classifier=classifier, features=features)


def classify_doubletons(classifier, profiles, summary, *, baseline_excluded=0.25):
    """Delegate to :func:`error_validation.classify_doubletons`."""
    return error_validation.classify_doubletons(
        classifier=classifier,
        profiles=profiles,
        summary=summary,
        baseline_excluded=baseline_excluded,
    )


def classifier_curves(scores, truth):
    """Delegate to :func:`error_validation.classifier_curves`."""
    return error_validation.classifier_curves(scores=scores, truth=truth)


def classifier_metrics(scores, truth, retained):
    """Delegate to :func:`error_validation.classifier_metrics`."""
    return error_validation.classifier_metrics(
        scores=scores, truth=truth, retained=retained
    )


def fit_error_rate_with_selected_doubletons(estimate, retained):
    """Delegate to :func:`error_validation.fit_error_rate_with_selected_doubletons`."""
    return error_validation.fit_error_rate_with_selected_doubletons(
        estimate=estimate, retained=retained
    )


def get_true_error_rate(zarr_path):
    """Delegate to :func:`error_validation.get_true_error_rate`."""
    return error_validation.get_true_error_rate(zarr_path=zarr_path)


def get_ts_diversity(ts_path):
    """Delegate to :func:`error_validation.get_ts_diversity`."""
    return error_validation.get_ts_diversity(ts_path=ts_path)


def sample_doubletons(zarr_path, config):
    """Delegate to :func:`error_validation.sample_doubletons`."""
    return error_validation.sample_doubletons(zarr_path=zarr_path, config=config)


def write_doubletons_csv(doubletons, path):
    """Delegate to :func:`error_validation.write_doubletons_csv`."""
    return error_validation.write_doubletons_csv(doubletons=doubletons, path=path)


def get_true_doubleton_mask(doubletons, truth_zarr_path):
    """Delegate to :func:`error_validation.get_true_doubleton_mask`."""
    return error_validation.get_true_doubleton_mask(
        doubletons=doubletons, truth_zarr_path=truth_zarr_path
    )


def cumulative_mismatch_profiles(estimate, truth_zarr_path=None, *, distances=None):
    """Delegate to :func:`error_validation.cumulative_mismatch_profiles`."""
    return error_validation.cumulative_mismatch_profiles(
        estimate=estimate, truth_zarr_path=truth_zarr_path, distances=distances
    )


def evaluate_mismatch_cutoffs(
    summary, true_doubleton_mask, *, cutoff_proportions, L_mismatch_trim_values
):
    """Delegate to :func:`error_validation.evaluate_mismatch_cutoffs`."""
    return error_validation.evaluate_mismatch_cutoffs(
        summary=summary,
        true_doubleton_mask=true_doubleton_mask,
        cutoff_proportions=cutoff_proportions,
        L_mismatch_trim_values=L_mismatch_trim_values,
    )


def mismatch_roc(mismatch_rate, is_true_doubleton):
    """Delegate to :func:`error_validation.mismatch_roc`."""
    return error_validation.mismatch_roc(
        mismatch_rate=mismatch_rate, is_true_doubleton=is_true_doubleton
    )


def make_estimation_config(
    *,
    window_sizes,
    num_doubletons=10000,
    random_seed=None,
    exclude_high_mismatch_proportion=0.25,
    L_mismatch_trim=None,
    num_workers=None,
    queue_depth=2,
    worker_poll_seconds=0.2,
    optimizer_bounds=((-30, 30), (-30, 30), (-50, float(np.log(0.5)))),
    start_alpha=(0.25, 0.5, 0.75),
    start_log_beta=(-12, -9, -6, -3, 0),
    start_log_epsilon=(-50, -30, -20, -12),
    objective_penalty=1e100,
    residual_guard=1e50,
):
    """Initialise notebook settings for the estimator."""
    return error_estimation.EstimationConfig(
        window_sizes=window_sizes,
        num_doubletons=num_doubletons,
        random_seed=random_seed,
        exclude_high_mismatch_proportion=exclude_high_mismatch_proportion,
        L_mismatch_trim=L_mismatch_trim,
        num_workers=num_workers,
        queue_depth=queue_depth,
        worker_poll_seconds=worker_poll_seconds,
        optimizer_bounds=optimizer_bounds,
        start_alpha=start_alpha,
        start_log_beta=start_log_beta,
        start_log_epsilon=start_log_epsilon,
        objective_penalty=objective_penalty,
        residual_guard=residual_guard,
    )


def make_doubleton_classifier(*, tier, threshold):
    """Construct the retained classifier for a notebook control."""
    return error_validation.DoubletonClassifier(tier=tier, threshold=threshold)


@dataclasses.dataclass(frozen=True)
class EpsilonDataset:
    """One noisy simulation and its biological truth; truth is never a donor panel."""

    name: str
    zarr_path: pathlib.Path
    truth_path: pathlib.Path
    left: int
    right: int


@dataclasses.dataclass(frozen=True)
class EpsilonSweepConfig:
    """Explicit scientific inputs and resource limits for the sanity check."""

    datasets: tuple[EpsilonDataset, ...]
    epsilons: tuple[float, ...]
    genetic_map: pathlib.Path
    output_dir: pathlib.Path
    num_workers: int = 4
    threads_per_match: int = 8
    cache_size_mb: int = 65536
    ancestor_exclude: str = "INFO/observed_derived_ac == 1"
    sample_likelihood_threshold: float = 0.0


def prepare_epsilon_dataset(dataset, config):
    """Infer noisy-data ancestors once and align site truth by physical position.

    Epsilon is errors per haplotype-bp. Its biallelic emission probability is
    epsilon * simulated_span / source_site_count, before inference-site filtering.
    The main classifier is restricted to the sites actually used by the HMM.
    """
    output = config.output_dir / dataset.name
    output.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(dataset.zarr_path, mode="r")
    positions = np.asarray(store["variant_position"][:])
    inside = (positions >= dataset.left) & (positions < dataset.right)
    if not np.all(inside):
        raise ValueError(
            "The configured simulation interval must contain all source sites"
        )
    truth = tskit.load(dataset.truth_path)
    np.testing.assert_array_equal(positions, truth.sites_position)
    genotype_shape = store["call_genotype"].shape
    sample_size = genotype_shape[1] * genotype_shape[2]
    if sample_size != truth.num_samples:
        raise ValueError("Noisy source and truth have different haploid cohort sizes")
    errors = ~np.asarray(store["call_genotype_error_mask"][:], dtype=bool)
    error_sites = np.any(errors, axis=(1, 2))
    recorded_counts = np.asarray(store["variant_genotype_error_count"][:])
    np.testing.assert_array_equal(errors.sum(axis=(1, 2)), recorded_counts)
    observed = np.asarray(store["call_genotype"][:]).reshape(
        len(positions), sample_size
    )
    np.testing.assert_array_equal(
        errors.reshape(len(positions), sample_size), observed != truth.genotype_matrix()
    )
    true_counts = np.bincount(truth.mutations_site, minlength=truth.num_sites)
    inference_input = output / "ancestor_input.zarr"
    if not inference_input.exists():
        compact = zarr.open_group(inference_input, mode="w", zarr_format=3)
        compact.attrs.update(dict(store.attrs))
        arrays = {
            "call_genotype": (
                observed.reshape(genotype_shape),
                ("variants", "samples", "ploidy"),
                (10000, min(50, genotype_shape[1]), genotype_shape[2]),
            ),
            "variant_position": (positions, ("variants",), (10000,)),
            "variant_allele": (
                np.asarray(store["variant_allele"][:], dtype=np.dtypes.StringDType()),
                ("variants", "alleles"),
                (10000, 2),
            ),
            "variant_ancestral_state": (
                np.asarray(
                    store["variant_ancestral_state"][:], dtype=np.dtypes.StringDType()
                ),
                ("variants",),
                (10000,),
            ),
            "variant_contig": (
                np.asarray(store["variant_contig"][:]),
                ("variants",),
                (10000,),
            ),
            "variant_observed_derived_ac": (
                observed.sum(axis=1),
                ("variants",),
                (10000,),
            ),
            "sample_id": (
                np.asarray(store["sample_id"][:], dtype=np.dtypes.StringDType()),
                ("samples",),
                (min(50, genotype_shape[1]),),
            ),
            "contig_id": (
                np.asarray(store["contig_id"][:], dtype=np.dtypes.StringDType()),
                ("contigs",),
                (1,),
            ),
            "contig_length": (
                np.asarray(store["contig_length"][:]),
                ("contigs",),
                (1,),
            ),
        }
        for name, (values, dimensions, chunks) in arrays.items():
            array = compact.create_array(
                name, data=values, chunks=chunks, dimension_names=dimensions
            )
            array.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)
    source = tsinfer_config.Source(
        path=inference_input, name="samples", exclude=config.ancestor_exclude
    )
    ancestor_path = output / "inferred_ancestors.zarr"
    ancestor_config = tsinfer_config.AncestorsConfig(
        name="ancestors", path=ancestor_path, sources=["samples"]
    )
    ancestral_state = tsinfer_config.AncestralState(
        path=dataset.zarr_path, field="variant_ancestral_state"
    )
    started = time.monotonic()
    if not ancestor_path.exists():
        tsinfer_ancestors.infer_ancestors(
            [source],
            ancestor_config,
            ancestral_state,
            num_threads=config.threads_per_match,
            progress=False,
        )
    ancestor_store = zarr.open_group(ancestor_path, mode="r")
    inference_positions = np.asarray(ancestor_store["variant_position"][:])
    rows = np.searchsorted(positions, inference_positions)
    np.testing.assert_array_equal(positions[rows], inference_positions)
    np.savez_compressed(
        output / "site_truth.npz",
        positions=positions,
        error_sites=error_sites,
        true_mutation_counts=true_counts,
        inference_rows=rows,
    )
    span = dataset.right - dataset.left
    manifest = {
        "dataset": dataset.name,
        "zarr_path": str(dataset.zarr_path),
        "truth_path": str(dataset.truth_path),
        "left": dataset.left,
        "right": dataset.right,
        "span_bp": span,
        "header_sequence_length": float(truth.sequence_length),
        "individuals": genotype_shape[1],
        "sample_size": sample_size,
        "ancestor_exclude": config.ancestor_exclude,
        "observed_singleton_sites": int(np.sum(observed.sum(axis=1) == 1)),
        "source_sites": len(positions),
        "inference_sites": len(rows),
        "inferred_ancestors": ancestor_store["call_genotype"].shape[1],
        "error_alleles": int(errors.sum()),
        "error_sites": int(error_sites.sum()),
        "biological_recurrent_sites": int(np.sum(true_counts > 1)),
        "error_or_recurrent_sites": int(np.sum(error_sites | (true_counts > 1))),
        "source_site_density_per_bp": len(positions) / span,
        "realised_epsilon": float(errors.sum() / (sample_size * span)),
        "ancestor_preparation_seconds": time.monotonic() - started,
    }
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_epsilon_sweep(config):
    """Derive a cumulative map from the exact cached simulation rate map."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rate_map = msprime.RateMap.read_hapmap(config.genetic_map, rate_col=2)
    positions = rate_map.position
    rates = np.nan_to_num(rate_map.rate, nan=0.0)
    cumulative_morgans = np.concatenate(([0.0], np.cumsum(np.diff(positions) * rates)))
    derived_map = config.output_dir / "genetic_map_hmm.txt"
    with derived_map.open("w") as stream:
        stream.write("chromosome position rate_cm_mb map_cm\n")
        for index, position in enumerate(positions):
            rate = rates[index] if index < len(rates) else 0.0
            stream.write(
                f"chr20 {position:.0f} {rate * 1e8:.15g} {cumulative_morgans[index] * 100:.15g}\n"
            )
    table = []
    for dataset in config.datasets:
        manifest = prepare_epsilon_dataset(dataset, config)
        source_positions = np.load(config.output_dir / dataset.name / "site_truth.npz")[
            "positions"
        ]
        finite_start = positions[np.flatnonzero(np.isfinite(rate_map.rate))[0]]
        expected = rate_map.get_cumulative_mass(source_positions)
        expected -= rate_map.get_cumulative_mass(finite_start)
        observed = tsinfer_matching.read_genetic_map(derived_map, source_positions)
        np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)
        manifest["map_max_abs_difference_morgans"] = float(
            np.max(np.abs(observed - expected))
        )
        table.append(manifest)
    frame = pd.DataFrame(table)
    frame.to_csv(config.output_dir / "datasets.csv", index=False)
    return frame


def epsilon_confusion(truth, predicted):
    """Site-level confusion counts and operating-point metrics, without pseudocounts."""
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    fpr = fp / (fp + tn) if fp + tn > 0 else np.nan
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else np.nan,
        "mcc": (tp * tn - fp * fn) / denominator if denominator > 0 else np.nan,
        "balanced_accuracy": (recall + 1 - fpr) / 2,
        "truth_prevalence": float(np.mean(truth)),
    }


def evaluate_epsilon_arg(ts, *, dataset_dir, epsilon, mismatch, match_seconds):
    """Score recurrence as >=2 mutation records, including back mutations.

    Scores are site decisions, not event localisation or allele-level error calls.
    The all-source scope also contains sites added by parsimony after matching.
    """
    truth = np.load(dataset_dir / "site_truth.npz")
    manifest = json.loads((dataset_dir / "dataset.json").read_text())
    source_positions = truth["positions"]
    positions = ts.sites_position
    source_rows = np.searchsorted(source_positions, positions)
    np.testing.assert_array_equal(source_positions[source_rows], positions)
    inference_positions = source_positions[truth["inference_rows"]]
    inference_rows = np.searchsorted(positions, inference_positions)
    np.testing.assert_array_equal(positions[inference_rows], inference_positions)
    true_counts = truth["true_mutation_counts"][source_rows]
    counts = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    inferred_recurrent = counts > 1
    error_sites = truth["error_sites"][source_rows]
    biological_recurrent = true_counts > 1
    positive = error_sites | biological_recurrent
    sample_nodes = np.zeros(ts.num_nodes, dtype=bool)
    sample_nodes[ts.samples()] = True
    recurrent_records = inferred_recurrent[ts.mutations_site]
    sample_records = sample_nodes[ts.mutations_node]
    sample_mutation_sites = ts.mutations_site[recurrent_records & sample_records]
    internal_mutation_sites = ts.mutations_site[recurrent_records & ~sample_records]
    np.savez_compressed(
        dataset_dir / f"eps_{epsilon:.0e}" / "predictions.npz",
        mutation_counts=counts,
        recurrent=inferred_recurrent,
        source_rows=source_rows,
        positions=positions,
    )
    rows = []
    scopes = {
        "inference_sites": inference_rows,
        "covered_source_sites": np.arange(len(positions)),
    }
    for scope, selection in scopes.items():
        predicted = inferred_recurrent[selection]
        actual = positive[selection]
        selected_counts = counts[selection]
        recurrent_counts = selected_counts[predicted]
        error = error_sites[selection]
        biological = biological_recurrent[selection]
        row = {
            "dataset": manifest["dataset"],
            "individuals": manifest["individuals"],
            "scope": scope,
            "epsilon": epsilon,
            "mismatch_probability": mismatch,
            "match_seconds": match_seconds,
            "sites": len(selection),
            "source_sites_outside_flanks": len(source_positions) - len(positions),
            "recurrent_sites": int(predicted.sum()),
            "recurrent_site_percentage": float(100 * predicted.mean()),
            "mean_mutations_per_recurrent_site": float(recurrent_counts.mean())
            if len(recurrent_counts) > 0
            else np.nan,
            "additional_mutations": int(np.maximum(selected_counts - 1, 0).sum()),
            "total_mutations": int(selected_counts.sum()),
            "error_site_recall": float(predicted[error].mean())
            if np.any(error)
            else np.nan,
            "biological_recurrence_recall": float(predicted[biological].mean())
            if np.any(biological)
            else np.nan,
            "error_only_recall": float(predicted[error & ~biological].mean())
            if np.any(error & ~biological)
            else np.nan,
            "biological_only_recall": float(predicted[biological & ~error].mean())
            if np.any(biological & ~error)
            else np.nan,
            "true_recurrent_site_percentage": float(100 * biological.mean()),
            "true_mean_mutations_per_recurrent_site": float(
                true_counts[selection][biological].mean()
            ),
            "recurrent_records_on_samples": int(
                np.isin(sample_mutation_sites, selection).sum()
            ),
            "recurrent_records_on_internal_nodes": int(
                np.isin(internal_mutation_sites, selection).sum()
            ),
        }
        row.update(epsilon_confusion(actual, predicted))
        rows.append(row)
    return rows


@dataclasses.dataclass(frozen=True)
class EpsilonMatchJob:
    dataset: EpsilonDataset
    config: EpsilonSweepConfig
    epsilon: float


@contextlib.contextmanager
def epsilon_inference_log(path):
    """Retain tsinfer diagnostics separately for each inference stage."""
    with path.open("a") as stream:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        inference_logger = logging.getLogger("tsinfer")
        inference_logger.addHandler(handler)
        inference_logger.setLevel(logging.INFO)
        try:
            yield
        finally:
            inference_logger.removeHandler(handler)
            handler.close()


def match_epsilon_ancestors(dataset, config):
    """Build one fixed ancestor ARG with stock recombination and mismatch settings.

    Neither epsilon nor the recombination map enters this stage. The full
    haploid cohort size is still supplied to the new TSLS transition model.
    """
    output = config.output_dir / dataset.name
    result_path = output / "ancestor_reference.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    manifest = json.loads((output / "dataset.json").read_text())
    ancestor_path = output / "inferred_ancestors.zarr"
    cfg = tsinfer_config.Config(
        sources={
            "ancestors": tsinfer_config.Source(
                name="ancestors", path=ancestor_path, sample_time="sample_time"
            )
        },
        ancestors=[
            tsinfer_config.AncestorsConfig(
                name="ancestors", path=ancestor_path, sources=["ancestors"]
            )
        ],
        match=tsinfer_config.MatchConfig(
            sources={
                "ancestors": tsinfer_config.MatchSourceConfig(
                    node_flags=0,
                    create_individuals=False,
                    sample_size=manifest["sample_size"],
                )
            },
            output=output / "ancestor_reference.trees",
            path_compression=False,
        ),
        ancestral_state=tsinfer_config.AncestralState(
            path=dataset.zarr_path, field="variant_ancestral_state"
        ),
    )
    (output / "ancestor_config.txt").write_text(cfg.format())
    started = time.monotonic()
    with epsilon_inference_log(output / "ancestor_matching.log"):
        ts = tsinfer_pipeline.match(
            cfg, num_threads=config.threads_per_match, cache_size=config.cache_size_mb
        )
    ts.dump(output / "ancestor_reference.trees")
    counts = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    result = {
        "dataset": dataset.name,
        "sample_size": manifest["sample_size"],
        "recombination": tsinfer_matching.DEFAULT_RECOMBINATION,
        "mismatch": tsinfer_matching.DEFAULT_MISMATCH,
        "recombination_map": None,
        "epsilon": None,
        "seconds": time.monotonic() - started,
        "nodes": ts.num_nodes,
        "sites": ts.num_sites,
        "mutations": ts.num_mutations,
        "recurrent_sites": int(np.sum(counts > 1)),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def prepare_epsilon_references(config):
    """Match the two ancestor panels independently, once each, before the sweep."""
    context = multiprocessing.get_context("spawn")
    workers = min(config.num_workers, len(config.datasets))
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [
            pool.submit(match_epsilon_ancestors, dataset, config)
            for dataset in config.datasets
        ]
        records = [future.result() for future in cf.as_completed(futures)]
    frame = pd.DataFrame(records).sort_values("dataset")
    frame.to_csv(config.output_dir / "ancestor_references.csv", index=False)
    return frame


def match_epsilon(job):
    """Match samples to a fixed stock-matched ancestor ARG using calibrated eps."""
    dataset = job.dataset
    config = job.config
    output = config.output_dir / dataset.name
    run_dir = output / f"eps_{job.epsilon:.0e}"
    run_dir.mkdir(exist_ok=True)
    result_path = run_dir / "metrics.csv"
    if result_path.exists():
        return pd.read_csv(result_path)
    manifest = json.loads((output / "dataset.json").read_text())
    mismatch = job.epsilon / manifest["source_site_density_per_bp"]
    if not 0 < mismatch < 0.5:
        raise ValueError(
            "Calibrated biallelic mismatch probability must lie in (0, 0.5)"
        )
    cfg = tsinfer_config.Config(
        sources={
            "samples": tsinfer_config.Source(name="samples", path=dataset.zarr_path)
        },
        ancestors=[],
        match=tsinfer_config.MatchConfig(
            sources={
                "samples": tsinfer_config.MatchSourceConfig(
                    recombination_rate=config.output_dir / "genetic_map_hmm.txt",
                    mismatch=mismatch,
                    sample_size=manifest["sample_size"],
                    likelihood_threshold=config.sample_likelihood_threshold,
                )
            },
            reference_ts=output / "ancestor_reference.trees",
            output=run_dir / "inferred.trees",
            path_compression=False,
        ),
        ancestral_state=tsinfer_config.AncestralState(
            path=dataset.zarr_path, field="variant_ancestral_state"
        ),
        post_process=tsinfer_config.PostProcessConfig(),
        augment_sites=tsinfer_config.AugmentSitesConfig(sources=["samples"]),
    )
    (run_dir / "config.txt").write_text(cfg.format())
    settings = {
        "epsilon": job.epsilon,
        "mismatch_probability": mismatch,
        "sample_size": manifest["sample_size"],
        "stage": "samples_only",
        "likelihood_threshold": config.sample_likelihood_threshold,
        "matching_threads": config.threads_per_match,
        "cache_size_mb": config.cache_size_mb,
    }
    (run_dir / "parameters.json").write_text(json.dumps(settings, indent=2) + "\n")
    started = time.monotonic()
    with epsilon_inference_log(run_dir / "matching.log"):
        ts = tsinfer_pipeline.match(
            cfg, num_threads=config.threads_per_match, cache_size=config.cache_size_mb
        )
        ts = tsinfer_pipeline.post_process(ts, cfg)
        ts = tsinfer_pipeline.augment_sites(ts, cfg)
    ts.dump(run_dir / "inferred.trees")
    elapsed = time.monotonic() - started
    rows = evaluate_epsilon_arg(
        ts,
        dataset_dir=output,
        epsilon=job.epsilon,
        mismatch=mismatch,
        match_seconds=elapsed,
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(result_path, index=False)
    logger.info("Completed %s epsilon=%g in %.1fs", dataset.name, job.epsilon, elapsed)
    return frame


def run_epsilon_sweep(config):
    """Distribute independent epsilon runs to a bounded multiprocessing queue."""
    jobs = [
        EpsilonMatchJob(dataset, config, epsilon)
        for dataset in config.datasets
        for epsilon in config.epsilons
    ]
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(
        max_workers=config.num_workers, mp_context=context
    ) as pool:
        futures = [pool.submit(match_epsilon, job) for job in jobs]
        frames = []
        for future in cf.as_completed(futures):
            frames.append(future.result())
            frame = pd.concat(frames, ignore_index=True)
            frame.to_csv(config.output_dir / "metrics.csv", index=False)
    frame = frame.sort_values(["individuals", "scope", "epsilon"])
    frame.to_csv(config.output_dir / "metrics.csv", index=False)
    return frame


def epsilon_recommendations(frame, *, scope="inference_sites"):
    """Compare site F1 and MCC optima; both are tuning results on these same seeds."""
    selected = frame[frame.scope == scope]
    rows = []
    for dataset, group in selected.groupby("dataset", sort=False):
        for metric in ["f1", "mcc", "balanced_accuracy"]:
            best = group.loc[group[metric].idxmax()]
            rows.append({"dataset": dataset, "criterion": metric, **best.to_dict()})
    return pd.DataFrame(rows)


def epsilon_macro_scores(frame, *, scope="inference_sites"):
    """Give the two datasets equal weight when comparing a shared tuning epsilon."""
    selected = frame[frame.scope == scope]
    grouped = selected.groupby("epsilon")
    columns = ["precision", "recall", "f1", "mcc", "balanced_accuracy"]
    summary = grouped[columns].mean()
    return summary.reset_index()
