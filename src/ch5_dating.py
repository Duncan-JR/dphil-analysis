"""Post-process saved ARG grids and compare corresponding mutation ages.

Dating restores non-inference variants from the observed VCZ sample source.
Correlations use posterior mutation means stored by tsdate, never node times or
the branch-midpoint mutation times in the output mutation table.
"""

import concurrent.futures as cf
import dataclasses
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import logging
import multiprocessing
import pathlib
import time

import ch5_hyperparams
import numpy as np
import numcodecs
import pandas as pd
import scipy.stats as stats
import tsdate
import tsinfer
import tskit
import zarr
import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatingCase:
    """A saved inferred ARG and the observed sample source used for dating."""

    dataset: str
    method: str
    ac_cutoff: int | None
    minimum_size: int | None
    raw_path: pathlib.Path
    native_config_path: pathlib.Path
    output_dir: pathlib.Path
    span: list[float]
    mutation_rate: float
    settings: dict
    source_include: str | None
    source_exclude: str | None
    sample_store_signature: str
    max_mismatches: int | None = None

    @property
    def dated_path(self) -> pathlib.Path:
        return self.output_dir / f'{self.raw_path.stem}_dated.trees'


def store_signature(path: pathlib.Path) -> str:
    """Detect changes to observed VCZ metadata or genotype chunks for caching."""
    digest = hashlib.sha256()
    for item in sorted(path.rglob('*')):
        if item.is_file():
            stat = item.stat()
            content = [str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns]
            digest.update(json.dumps(content).encode())
    return digest.hexdigest()


def load_configuration(config_path: pathlib.Path) -> dict:
    """Read either the legacy mismatch grid or the minimum-size experiment."""
    with config_path.open() as source:
        config = yaml.safe_load(source)
    experiment_path = ch5_hyperparams.resolve_path(config['experiment_config'], config_path.parent)
    with experiment_path.open() as source:
        config['experiment'] = yaml.safe_load(source)
    config['experiment_path'] = experiment_path
    config['data_dir'] = ch5_hyperparams.resolve_path(config['experiment']['data_dir'], experiment_path.parent)
    return config


def dating_cases(config_path: pathlib.Path) -> list[DatingCase]:
    """Select both Full HMM outputs and the configured stitched parameter grid."""
    config = load_configuration(config_path)
    experiment = config['experiment']
    settings = config['dating']
    experiment_datasets = {item['name']: item for item in experiment['datasets']}
    output_dir = ch5_hyperparams.resolve_path(config['output_dir'], config_path.parent)
    output_dir /= settings['output_subdir']
    cases = []
    for dataset in config['datasets']:
        name = dataset['name']
        native_path = config['data_dir'] / 'configs' / f'{name}_ancestor_inference.toml'
        native = tsinfer.config.Config.from_toml(native_path)
        signature = store_signature(pathlib.Path(native.sources['samples'].path))
        grid = [(None, None, None)]
        minimums = experiment.get('min_focal_region_size', [None])
        grid.extend(itertools.product(experiment['ac_cutoff'], experiment['max_mismatches'], minimums))
        for cutoff, budget, minimum in grid:
            prefix = f'{name}_inferred'
            method = 'full_hmm'
            if cutoff is not None:
                prefix += f'_ac{cutoff}_k{budget}'
                if minimum is not None:
                    prefix += f'_min{minimum}'
                prefix += f'_{config["method"]}_stitched'
                method = f'stitched_{config["method"]}'
            cases.append(DatingCase(
                name, method, cutoff, minimum,
                config['data_dir'] / 'matches' / f'{prefix}_samples_raw.trees',
                native_path, output_dir, dataset['analysis_span_bp'],
                dataset['mutation_rate'], settings, experiment_datasets[name].get('include'),
                experiment_datasets[name].get('exclude'), signature, max_mismatches=budget))
    return cases


def augmentation_config(native: tsinfer.config.Config, include: str | None,
                        exclude: str | None) -> tsinfer.config.Config:
    """Restore dataset filters, removing matching-only singleton exclusions."""
    sample_source = dataclasses.replace(native.sources['samples'], include=include, exclude=exclude)
    return dataclasses.replace(
        native, sources={'samples': sample_source},
        augment_sites=tsinfer.config.AugmentSitesConfig(sources=['samples']))


def date_case(case: DatingCase) -> dict:
    """Save a dated ARG, reusing it only when its inputs and settings match."""
    with case.raw_path.open('rb') as source:
        raw_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    signature = {
        'raw_sha256': raw_hash,
        'native_config': case.native_config_path.read_text(),
        'sample_store_signature': case.sample_store_signature,
        'span': case.span,
        'mutation_rate': case.mutation_rate,
        'augmentation_filters': {'include': case.source_include, 'exclude': case.source_exclude},
        'post_process': case.settings['post_process'],
        'preprocess': case.settings['preprocess'],
        'estimation': case.settings['estimation'],
        'versions': {name: importlib.metadata.version(name) for name in ['tsinfer', 'tsdate', 'tskit']},
        'tsinfer_pipeline_sha256': hashlib.sha256(pathlib.Path(tsinfer.pipeline.__file__).read_bytes()).hexdigest(),
        'analysis_sha256': hashlib.sha256(inspect.getsource(date_case).encode()).hexdigest(),
    }
    manifest_path = case.dated_path.with_suffix('.json')
    if case.dated_path.exists() and manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['signature'] == signature:
            return saved['measurement']
    case.output_dir.mkdir(parents=True, exist_ok=True)
    numcodecs.blosc.set_nthreads(case.settings['helper_threads'])
    zarr.config.set({'async.concurrency': case.settings['helper_threads'],
                     'threading.max_workers': case.settings['helper_threads']})
    raw = tskit.load(case.raw_path)
    native = tsinfer.config.Config.from_toml(case.native_config_path)
    native = augmentation_config(native, case.source_include, case.source_exclude)
    started = time.perf_counter()
    processed = tsinfer.post_process(raw, **case.settings['post_process'])
    processed_at = time.perf_counter()
    augmented = tsinfer.augment_sites(processed, native)
    augmented_at = time.perf_counter()
    window = ch5_hyperparams.window_tree_sequence(augmented, case.span)
    prepared = tsdate.preprocess_ts(window, **case.settings['preprocess'])
    prepared_at = time.perf_counter()
    dated = tsdate.date(prepared, mutation_rate=case.mutation_rate, **case.settings['estimation'])
    dated_at = time.perf_counter()
    assert dated.num_samples == raw.num_samples
    assert dated.time_units == case.settings['estimation']['time_units']
    dated.dump(case.dated_path)
    measurement = {
        'dataset': case.dataset, 'comparison_method': case.method,
        'ac_cutoff': case.ac_cutoff, 'min_focal_region_size': case.minimum_size,
        'dated_path': str(case.dated_path), 'raw_nodes': raw.num_nodes,
        'dated_nodes': dated.num_nodes, 'raw_sites': raw.num_sites,
        'dated_sites': dated.num_sites, 'dated_mutations': dated.num_mutations,
        'post_process_s': processed_at - started,
        'augment_sites_s': augmented_at - processed_at,
        'preprocess_s': prepared_at - augmented_at, 'dating_s': dated_at - prepared_at,
    }
    manifest_path.write_text(json.dumps({'signature': signature, 'measurement': measurement}, indent=2) + '\n')
    return measurement


def date_all(config_path: pathlib.Path) -> pd.DataFrame:
    """Distribute the saved ARGs across available CPU cores."""
    config = load_configuration(config_path)
    cases = dating_cases(config_path)
    workers = config['dating']['workers']
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(cases))
    context = multiprocessing.get_context('spawn')
    rows = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(date_case, case): case for case in cases}
        for future in cf.as_completed(futures):
            row = future.result()
            row['max_mismatches'] = futures[future].max_mismatches
            rows.append(row)
            logger.info('Dated %s: %s, AC %s, minimum %s', row['dataset'],
                        row['comparison_method'], row['ac_cutoff'], row['min_focal_region_size'])
    return pd.DataFrame(rows).sort_values(['dataset', 'comparison_method', 'ac_cutoff', 'min_focal_region_size'])


def mutation_ages(ts: tskit.TreeSequence, posterior: bool,
                  include_frequency: bool = False) -> pd.DataFrame:
    """Return finite ages at sites containing exactly one mutation.

    Keys include position and ancestral/derived states to exclude polarity or
    allele mismatches. Recurrent and back-mutation sites have no unique pairing.
    """
    rows = []
    tree = ts.first() if include_frequency else None
    for site in ts.sites():
        if len(site.mutations) != 1:
            continue
        mutation = site.mutations[0]
        age = mutation.metadata['mn'] if posterior else mutation.time
        if np.isfinite(age):
            row = {'position': site.position, 'ancestral_state': site.ancestral_state,
                   'derived_state': mutation.derived_state, 'age': age}
            if tree is not None:
                tree.seek(site.position)
                row['derived_frequency'] = tree.num_samples(mutation.node) / ts.num_samples
            rows.append(row)
    return pd.DataFrame(rows).set_index(['position', 'ancestral_state', 'derived_state'])


def correlate_ages(comparison: pd.Series, reference: pd.Series) -> float:
    """Calculate Pearson r on aligned mutation ages with a defined variance."""
    if not comparison.index.equals(reference.index):
        raise ValueError('Mutation-age series must have identical ordered mutation keys')
    if len(comparison) < 2 or comparison.nunique() < 2 or reference.nunique() < 2:
        raise ValueError('Mutation-age correlation requires at least two distinct ages')
    return float(stats.pearsonr(comparison.to_numpy(), reference.to_numpy()).statistic)


def frequency_bins(frequencies: pd.Series, settings: dict) -> pd.Series:
    """Use half-open derived-frequency bins, placing boundaries in the upper bin."""
    indices = np.searchsorted(settings['cutoffs'], frequencies.to_numpy(), side='right')
    labels = np.asarray(settings['labels'])
    return pd.Series(labels[indices], index=frequencies.index)


def correlation_rows(runs: pd.DataFrame, ages: list[pd.DataFrame], reference: pd.Series,
                     frequencies: pd.Series, reference_method: str, settings: dict) -> list[dict]:
    """Use fixed mutation keys and Full HMM frequencies for every case in a bin."""
    bins = frequency_bins(frequencies.loc[reference.index], settings)
    rows = []
    groups = [('all', reference.index)]
    groups.extend((label, bins.index[bins.eq(label)]) for label in settings['labels'])
    for label, keys in groups:
        reference_ages = reference.loc[keys]
        for index, run in runs.iterrows():
            if reference_method == 'full_hmm' and run.comparison_method == 'full_hmm':
                continue
            inferred = ages[index]['age'].loc[keys]
            errors = inferred - reference_ages
            rows.append({
                'dataset': run.dataset, 'comparison_method': run.comparison_method,
                'ac_cutoff': run.ac_cutoff, 'min_focal_region_size': run.min_focal_region_size,
                'max_mismatches': run.max_mismatches,
                'reference_method': reference_method, 'frequency_bin': label,
                'matched_mutations': len(keys), 'pearson_r': correlate_ages(inferred, reference_ages),
                'mean_estimated_age': inferred.mean(), 'mean_reference_age': reference_ages.mean(),
                'mean_error_generations': errors.mean(),
                'median_absolute_error_generations': errors.abs().median(),
            })
    return rows


@dataclasses.dataclass
class MatchedMutationAges:
    """Aligned inferred ages, observed frequencies, and optional simulation truth."""

    inferred: list[pd.DataFrame]
    common: pd.MultiIndex
    frequencies: pd.Series
    baseline: pd.Series
    truth: pd.Series | None


def matched_mutation_ages(runs: pd.DataFrame, dataset: dict, config: dict,
                          include_truth: bool = True) -> MatchedMutationAges:
    """Hold mutation identity and Full HMM frequency fixed across a parameter grid."""
    ages = []
    for run in runs.itertuples():
        ts = tskit.load(run.dated_path)
        values = mutation_ages(ts, posterior=True,
                               include_frequency=run.comparison_method == 'full_hmm')
        ages.append(values)
    common = ages[0].index
    for values in ages[1:]:
        common = common.intersection(values.index, sort=False)
    common = common.sort_values()
    baseline_index = runs.index[runs.comparison_method.eq('full_hmm')].item()
    baseline = ages[baseline_index]['age'].loc[common]
    frequencies = ages[baseline_index]['derived_frequency']
    source = next(item for item in config['experiment']['datasets'] if item['name'] == dataset['name'])
    true_ages = None
    if include_truth and source['ts_path'] is not None:
        truth_path = ch5_hyperparams.resolve_path(source['ts_path'], config['experiment_path'].parent)
        truth = tskit.load(truth_path)
        truth = ch5_hyperparams.window_tree_sequence(truth, dataset['analysis_span_bp'])
        true_ages = mutation_ages(truth, posterior=False)['age']
        truth_common = common.intersection(true_ages.index, sort=False).sort_values()
        true_ages = true_ages.loc[truth_common]
    return MatchedMutationAges(ages, common, frequencies, baseline, true_ages)


def stitched_mutation_age_pairs(dated_runs: pd.DataFrame, config_path: pathlib.Path) -> pd.DataFrame:
    """Pair stitched and Full HMM posterior ages across all frequencies.

    Each dataset uses the same common mutation keys as its Full HMM correlation
    panel. Simulation truth is neither required nor used for these comparisons.
    """
    config = load_configuration(config_path)
    frames = []
    for dataset in config['datasets']:
        name = dataset['name']
        if name not in config['dating']['compare_full_datasets']:
            continue
        runs = dated_runs.loc[dated_runs.dataset.eq(name)].reset_index(drop=True)
        matched = matched_mutation_ages(runs, dataset, config, include_truth=False)
        keys = matched.common
        for index, run in runs.iterrows():
            if run.comparison_method == 'full_hmm':
                continue
            pairs = pd.DataFrame({'full_hmm_age': matched.baseline,
                                  'stitched_age': matched.inferred[index]['age'].loc[keys]})
            for column in ['dataset', 'comparison_method', 'ac_cutoff',
                           'max_mismatches', 'min_focal_region_size']:
                pairs[column] = run[column]
            frames.append(pairs.reset_index())
    return pd.concat(frames, ignore_index=True)


def mutation_correlations(dated_runs: pd.DataFrame, config_path: pathlib.Path) -> pd.DataFrame:
    """Compare dates overall and in fixed observed-derived-frequency bins.

    Frequencies come from the dated Full HMM sample genealogy for both panels.
    Within a panel, mutation keys and bin membership are held fixed across cases.
    """
    config = load_configuration(config_path)
    settings = config['dating']['frequency_bins']
    output_rows = []
    for dataset in config['datasets']:
        name = dataset['name']
        runs = dated_runs.loc[dated_runs.dataset.eq(name)].reset_index(drop=True)
        matched = matched_mutation_ages(runs, dataset, config)
        if name in config['dating']['compare_full_datasets']:
            output_rows.extend(correlation_rows(
                runs, matched.inferred, matched.baseline, matched.frequencies, 'full_hmm', settings))
        if matched.truth is not None:
            output_rows.extend(correlation_rows(
                runs, matched.inferred, matched.truth, matched.frequencies, 'true_arg', settings))
    return pd.DataFrame(output_rows)
