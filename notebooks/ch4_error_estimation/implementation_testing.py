# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dphil-analysis
#     language: python
#     name: dphil_analysis
# ---

# %% [markdown]
# # Setup and reproducibility
# These are software smoke tests, not cohort error estimates. Focal doubletons
# are ascertained in the observed calls of 200 selected samples. All source
# stores are opened read-only; test stores use Zarr 3 in a temporary directory.
# The simulation's contig label is `1`; the explicit chr17 map follows its
# source filename and the documented simulation provenance, not that label.
# With user approval, all records at duplicated coordinates are excluded from
# fixtures. Exclusions are reported; the production estimator rejects duplicates.

# %%
import atexit
import dataclasses
import importlib.metadata
import pathlib
import tempfile
import time

import msprime
import numba
import numpy as np
import scipy
import zarr

import dphil_analysis
from dphil_analysis import error_estimation


@dataclasses.dataclass
class SmokeConfig:
    seed: int = 42
    num_samples: int = 200
    interval_width: int = 4_000_000
    window_sizes: tuple = (1000, 5000, 10000, 50000, 100000, 250000)
    num_doubletons: int = 100
    num_workers: int = 1


def make_masked_store(source, output_path, interval, sample_indices):
    """Copy a contiguous site interval and sample subset in bounded blocks."""
    positions = source['variant_position'][:]
    start, stop = np.searchsorted(positions, interval, side='left')
    selected_positions = positions[start:stop]
    # Drop every record at repeated coordinates, including across interval edges.
    duplicate = np.zeros(len(positions), dtype=bool)
    repeated = np.diff(positions) == 0
    duplicate[:-1] |= repeated
    duplicate[1:] |= repeated
    keep = ~duplicate[start:stop]
    selected_positions = selected_positions[keep]
    assert np.all(np.diff(selected_positions) > 0)
    target = zarr.open_group(output_path, mode='w', zarr_format=3)
    names = ['call_genotype', 'variant_position', 'sample_id', 'variant_contig', 'contig_id']
    for optional in ['call_genotype_phased', 'call_genotype_mask', 'contig_length']:
        if optional in source:
            names.append(optional)
    for name in names:
        array = source[name]
        dimensions = tuple(array.attrs['_ARRAY_DIMENSIONS'])
        shape = list(array.shape)
        if 'variants' in dimensions:
            shape[dimensions.index('variants')] = len(selected_positions)
        if 'samples' in dimensions:
            shape[dimensions.index('samples')] = len(sample_indices)
        chunks = tuple(min(n, c) for n, c in zip(shape, array.chunks))
        copied = target.create_array(
            name, shape=tuple(shape), dtype=array.dtype, chunks=chunks,
            dimension_names=dimensions, attributes=dict(array.attrs),
        )
        if 'variants' in dimensions:
            output_offset = 0
            for offset in range(start, stop, source['call_genotype'].chunks[0]):
                end = min(offset + source['call_genotype'].chunks[0], stop)
                selection = [slice(None)] * array.ndim
                selection[0] = slice(offset, end)
                if 'samples' in dimensions:
                    selection[dimensions.index('samples')] = sample_indices
                block = array.oindex[tuple(selection)]
                block = block[keep[offset - start:end - start]]
                copied[output_offset:output_offset + len(block)] = block
                output_offset += len(block)
        elif 'samples' in dimensions:
            copied[:] = array.oindex[sample_indices]
        else:
            copied[:] = array[:]
    return {
        'source': str(source.store), 'source_shape': source['call_genotype'].shape,
        'interval': tuple(float(x) for x in interval),
        'sample_ids': target['sample_id'][:].tolist(),
        'excluded_duplicate_records': int(np.count_nonzero(~keep)),
        'interval_source_sites': int(stop - start),
        'fixture_shape': target['call_genotype'].shape,
        'represented_span': float(selected_positions[-1] - selected_positions[0]),
        'output': str(output_path),
    }


def source_mask(path, map_path, chrom, smoke):
    """Inspect source metadata and choose the prescribed seeded mask."""
    source = zarr.open_group(path, mode='r')
    for name in ['call_genotype', 'variant_position', 'sample_id', 'variant_contig', 'contig_id']:
        assert name in source
    positions = source['variant_position'][:]
    assert np.all(np.diff(positions) >= 0)
    assert positions[-1] - positions[0] >= smoke.interval_width
    start = np.median(positions) - smoke.interval_width / 2
    start = np.clip(start, positions[0], positions[-1] - smoke.interval_width)
    interval = np.array([start, start + smoke.interval_width])
    rate_map = msprime.RateMap.read_hapmap(map_path, position_col=1, rate_col=2)
    with map_path.open() as handle:
        next(handle)
        assert all(line.split()[0] == f'chr{chrom}' for line in handle if line.strip())
    covered = (rate_map.left < interval[1]) & (rate_map.right > interval[0])
    assert rate_map.position[0] <= interval[0]
    assert rate_map.position[-1] >= interval[1]
    assert np.all(np.isfinite(rate_map.rate[covered]))
    assert np.all(rate_map.rate[covered] >= 0)
    rng = np.random.default_rng(smoke.seed)
    samples = rng.choice(source['call_genotype'].shape[1], smoke.num_samples, replace=False)
    samples.sort()
    return {'source': source, 'interval': interval, 'samples': samples}


def make_tiny_store(genotypes, positions, *, path=None, sample_ids=None, phased=None):
    """Create only ephemeral fixtures; ordinary calls use in-memory Zarr 3."""
    group = zarr.open_group(path, mode='w', zarr_format=3)
    genotypes = np.asarray(genotypes, dtype=np.int8)
    if sample_ids is None:
        sample_ids = np.asarray([f'sample_{i}' for i in range(genotypes.shape[1])], dtype='T')
    arrays = {
        'call_genotype': (genotypes, ('variants', 'samples', 'ploidy')),
        'variant_position': (np.asarray(positions, dtype=float), ('variants',)),
        'sample_id': (np.asarray(sample_ids, dtype='T'), ('samples',)),
        'variant_contig': (np.zeros(len(positions), dtype=np.int8), ('variants',)),
        'contig_id': (np.asarray(['tiny'], dtype='T'), ('contigs',)),
    }
    if phased is not None:
        arrays['call_genotype_phased'] = (np.asarray(phased), ('variants', 'samples'))
    for name, (data, dimensions) in arrays.items():
        chunks = data.shape
        if dimensions[0] == 'variants':
            chunks = (min(2, len(data)), *data.shape[1:])
        group.create_array(name, data=data, chunks=chunks, dimension_names=dimensions)
    return group


def assert_sampling(path, doubletons, config):
    group = zarr.open_group(path, mode='r')
    positions = group['variant_position'][:]
    gt = group['call_genotype']
    d = doubletons.num_doubletons
    assert d == min(config.num_doubletons, doubletons.num_eligible)
    assert np.all(np.diff(doubletons.site_indices) > 0)
    assert doubletons.sample_indices.shape == (d, 2)
    assert doubletons.ploidy_indices.shape == (d, 2)
    assert np.all(doubletons.sample_indices >= 0)
    assert np.all(doubletons.sample_indices < gt.shape[1])
    assert np.all(doubletons.ploidy_indices >= 0)
    assert np.all(doubletons.ploidy_indices < gt.shape[2])
    assert np.all(doubletons.positions - config.window_sizes[-1] >= positions[0])
    assert np.all(doubletons.positions + config.window_sizes[-1] <= positions[-1])
    chunk_indices = doubletons.site_indices // gt.chunks[0]
    for chunk in np.unique(chunk_indices):
        offset = chunk * gt.chunks[0]
        block = gt[offset:offset + gt.chunks[0]]
        for row in np.flatnonzero(chunk_indices == chunk):
            site = block[doubletons.site_indices[row] - offset]
            assert np.all((site == 0) | (site == 1))
            assert np.count_nonzero(site == 1) == 2
            assert np.all(site[doubletons.sample_indices[row], doubletons.ploidy_indices[row]] == 1)
            addresses = doubletons.sample_indices[row] * gt.shape[2] + doubletons.ploidy_indices[row]
            assert addresses[0] != addresses[1]
    print('Sampling passed:', d, 'selected;', doubletons.num_eligible, 'eligible', flush=True)


def assert_same_doubletons(first, second):
    for field in ['site_indices', 'positions', 'sample_indices', 'sample_ids', 'ploidy_indices']:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert first.num_eligible == second.num_eligible


def assert_value_error(function, *args, match, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert match.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError(f'Expected ValueError containing {match!r}')


def assert_summaries(path, doubletons, diversity, summary, config):
    group = zarr.open_group(path, mode='r')
    assert diversity.num_sites == group['call_genotype'].shape[0]
    assert diversity.pair_difference_counts.shape == (doubletons.num_doubletons,)
    assert np.isfinite(diversity.global_pi_per_site)
    assert 0 <= diversity.global_pi_per_site <= 1
    assert np.all(np.isfinite(diversity.pair_pi_per_site))
    assert np.all((diversity.pair_pi_per_site >= 0) & (diversity.pair_pi_per_site <= 1))
    density = diversity.num_sites / diversity.span_bp
    np.testing.assert_allclose(diversity.global_pi_per_bp, diversity.global_pi_per_site * density, rtol=1e-12)
    np.testing.assert_allclose(diversity.pair_pi_per_bp, diversity.pair_pi_per_site * density, rtol=1e-12)
    shape = (len(config.window_sizes), doubletons.num_doubletons)
    for values in [summary.left_counts, summary.right_counts, summary.left_rates, summary.right_rates]:
        assert values.shape == shape
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0)
    assert np.all(np.diff(summary.left_counts, axis=0) >= 0)
    assert np.all(np.diff(summary.right_counts, axis=0) >= 0)
    positions = group['variant_position'][:]
    for row in range(min(3, doubletons.num_doubletons)):
        focal = doubletons.site_indices[row]
        carriers = doubletons.sample_indices[row] * group['call_genotype'].shape[2] + doubletons.ploidy_indices[row]
        for window_index, length in enumerate(config.window_sizes):
            left = np.searchsorted(positions, positions[focal] - length, side='left')
            right = np.searchsorted(positions, positions[focal] + length, side='right')
            block = group['call_genotype'][left:right].reshape(right - left, -1)
            differences = block[:, carriers[0]] != block[:, carriers[1]]
            assert summary.left_counts[window_index, row] == differences[:focal-left].sum()
            assert summary.right_counts[window_index, row] == differences[focal-left+1:].sum()
    print(path.name, 'diversity units, shapes, monotonicity and direct windows passed', flush=True)


if __name__ == '__main__':
    repo = pathlib.Path.home() / 'work' / 'dphil-analysis'
    sim_name = (
        'OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0-'
        'geno-1.0-phase0.0-mispol0.0.zarr'
    )
    sim_source_path = repo / 'data' / 'zarr_vcfs' / 'sim' / sim_name
    tgp_source_path = repo / 'data' / 'zarr_vcfs' / 'tgp' / 'chr20' / 'data.zarr'
    map_dir = pathlib.Path.home() / 'work' / 'tsinfer-paper' / 'data' / 'HapMapII_GRCh38'
    sim_map = map_dir / 'genetic_map_Hg38_chr17.txt'
    tgp_map = map_dir / 'genetic_map_Hg38_chr20.txt'
    smoke = SmokeConfig()
    config = error_estimation.EstimationConfig(
        window_sizes=smoke.window_sizes, num_doubletons=smoke.num_doubletons,
        random_seed=smoke.seed, num_workers=smoke.num_workers,
    )
    temporary = tempfile.TemporaryDirectory(prefix='error_estimation_')
    atexit.register(temporary.cleanup)
    sim_smoke_path = pathlib.Path(temporary.name) / 'sim.zarr'
    tgp_smoke_path = pathlib.Path(temporary.name) / 'tgp.zarr'
    for package in ['dphil-analysis', 'numpy', 'numba', 'scipy', 'zarr', 'msprime']:
        print(package, importlib.metadata.version(package))
    assert int(zarr.__version__.split('.')[0]) >= 3
    print('Installed package:', dphil_analysis.__file__)

# %% [markdown]
# # Simulated data — chr17

# %%
if __name__ == '__main__':
    started = time.perf_counter()
    sim_mask = source_mask(sim_source_path, sim_map, 17, smoke)
    sim_provenance = make_masked_store(
        sim_mask['source'], sim_smoke_path, sim_mask['interval'], sim_mask['samples'],
    )
    assert sim_provenance['fixture_shape'][1] == smoke.num_samples
    assert sim_provenance['represented_span'] >= 2 * max(smoke.window_sizes)
    print('Resolved source:', sim_source_path.resolve())
    print('Map:', sim_map)
    print({key: value for key, value in sim_provenance.items() if key != 'sample_ids'})
    print('Fixture wall seconds:', time.perf_counter() - started, flush=True)

# %%
if __name__ == '__main__':
    started = time.perf_counter()
    sim_doubletons = error_estimation.sample_doubletons(sim_smoke_path, config=config)
    assert_sampling(sim_smoke_path, sim_doubletons, config)
    print('sim sampling seconds:', time.perf_counter() - started, 'workers:', config.num_workers)

# %%
if __name__ == '__main__':
    started = time.perf_counter()
    sim_diversity = error_estimation.compute_diversity(
        sim_smoke_path, sim_doubletons, num_workers=config.num_workers,
    )
    sim_summary = error_estimation.summarise_mismatches(
        sim_smoke_path, sim_doubletons, config=config, recombination=sim_map,
    )
    assert_summaries(sim_smoke_path, sim_doubletons, sim_diversity, sim_summary, config)
    print('sim diversity/windows seconds:', time.perf_counter() - started, 'workers:', config.num_workers)
    print('Global pi per site/bp:', sim_diversity.global_pi_per_site, sim_diversity.global_pi_per_bp)

# %% [markdown]
# # 1000 Genomes Project (tgp) — chr20

# %%
if __name__ == '__main__':
    started = time.perf_counter()
    tgp_mask = source_mask(tgp_source_path, tgp_map, 20, smoke)
    tgp_provenance = make_masked_store(
        tgp_mask['source'], tgp_smoke_path, tgp_mask['interval'], tgp_mask['samples'],
    )
    assert tgp_provenance['fixture_shape'][1] == smoke.num_samples
    assert tgp_provenance['represented_span'] >= 2 * max(smoke.window_sizes)
    print('Resolved source:', tgp_source_path.resolve())
    print('Map:', tgp_map)
    print({key: value for key, value in tgp_provenance.items() if key != 'sample_ids'})
    print('Fixture wall seconds:', time.perf_counter() - started, flush=True)

# %%
if __name__ == '__main__':
    started = time.perf_counter()
    tgp_doubletons = error_estimation.sample_doubletons(tgp_smoke_path, config=config)
    assert_sampling(tgp_smoke_path, tgp_doubletons, config)
    print('tgp sampling seconds:', time.perf_counter() - started, 'workers:', config.num_workers)

# %%
if __name__ == '__main__':
    started = time.perf_counter()
    tgp_diversity = error_estimation.compute_diversity(
        tgp_smoke_path, tgp_doubletons, num_workers=config.num_workers,
    )
    tgp_summary = error_estimation.summarise_mismatches(
        tgp_smoke_path, tgp_doubletons, config=config, recombination=tgp_map,
    )
    assert_summaries(tgp_smoke_path, tgp_doubletons, tgp_diversity, tgp_summary, config)
    print('tgp diversity/windows seconds:', time.perf_counter() - started, 'workers:', config.num_workers)
    print('Global pi per site/bp:', tgp_diversity.global_pi_per_site, tgp_diversity.global_pi_per_bp)

# %% [markdown]
# # Focused numerical and indexing checks
# The initial checks establish fixture schema, completeness and phasing.

# %%
if __name__ == '__main__':
    for path, provenance in [(sim_smoke_path, sim_provenance), (tgp_smoke_path, tgp_provenance)]:
        group = zarr.open_group(path, mode='r')
        gt = group['call_genotype']
        positions = group['variant_position'][:]
        assert np.all(np.diff(positions) > 0)
        assert provenance['interval'][1] - provenance['interval'][0] == smoke.interval_width
        assert np.isclose(provenance['represented_span'], smoke.interval_width, rtol=0.001)
        assert gt.shape[0] == provenance['interval_source_sites'] - provenance['excluded_duplicate_records']
        for offset in range(0, gt.shape[0], gt.chunks[0]):
            stop = min(offset + gt.chunks[0], gt.shape[0])
            block = gt[offset:stop]
            assert np.all(block >= 0)
            heterozygous = np.any(block != block[:, :, :1], axis=2)
            phased = group['call_genotype_phased'][offset:stop].astype(bool)
            assert np.all(phased[heterozygous])
            assert not np.any(group['call_genotype_mask'][offset:stop])
        print(path.name, 'schema, completeness and phasing passed', flush=True)

# %%
if __name__ == '__main__':
    small_config = dataclasses.replace(config, num_doubletons=10)
    first = error_estimation.sample_doubletons(sim_smoke_path, config=small_config)
    second = error_estimation.sample_doubletons(sim_smoke_path, config=small_config)
    assert_same_doubletons(first, second)
    assert error_estimation.EstimationConfig(window_sizes=[1]).num_doubletons == 10000
    for invalid in [0, -1, 1.5, True]:
        assert_value_error(error_estimation.EstimationConfig, window_sizes=[1],
                           num_doubletons=invalid, match='positive integer')
    for invalid in [[], [0], [2, 1], [1, 1], [np.nan]]:
        assert_value_error(error_estimation.EstimationConfig, window_sizes=invalid,
                           match='window_sizes')
    tiny_genotypes = np.array([
        [[0, 0], [0, 0], [0, 0]],
        [[1, 1], [0, 0], [0, 0]],
        [[1, 0], [0, 0], [0, 0]],  # A singleton must survive diversity scans.
        [[1, 0], [1, 0], [0, 0]],
        [[0, 2], [0, 0], [0, 0]],  # Sum == 2 is not a doubleton.
        [[0, 0], [0, 0], [0, 0]],
    ], dtype=np.int8)
    tiny_positions = np.arange(6) * 100
    tiny = make_tiny_store(tiny_genotypes, tiny_positions)
    tiny_config = error_estimation.EstimationConfig(
        window_sizes=[50, 100], num_doubletons=10, random_seed=42, num_workers=1,
    )
    tiny_doubletons = error_estimation.sample_doubletons(tiny, config=tiny_config)
    assert tiny_doubletons.num_doubletons == tiny_doubletons.num_eligible == 2
    np.testing.assert_array_equal(tiny_doubletons.site_indices, [1, 3])
    parallel = error_estimation.sample_doubletons(
        tiny, config=dataclasses.replace(tiny_config, num_workers=2),
    )
    assert_same_doubletons(tiny_doubletons, parallel)
    empty = make_tiny_store(np.zeros_like(tiny_genotypes), tiny_positions)
    assert_value_error(error_estimation.sample_doubletons, empty,
                       config=tiny_config, match='No eligible doubletons')
    duplicates = make_tiny_store(tiny_genotypes, [0, 100, 100, 300, 400, 500])
    assert_value_error(error_estimation.sample_doubletons, duplicates,
                       config=tiny_config, match='strictly increasing')
    print('Determinism, worker dispatch, cap and invalid-input checks passed', flush=True)

# %%
if __name__ == '__main__':
    tiny_diversity = error_estimation.compute_diversity(tiny, tiny_doubletons, num_workers=1)
    flat = tiny_genotypes.reshape(len(tiny_positions), -1)
    pair_differences = []
    for a in range(flat.shape[1]):
        for b in range(a + 1, flat.shape[1]):
            pair_differences.append(np.count_nonzero(flat[:, a] != flat[:, b]))
    np.testing.assert_allclose(tiny_diversity.global_difference_sum, np.mean(pair_differences), rtol=1e-12)
    # Site 2 is a singleton and contributes a mismatch to both carrier pairs.
    for row, carriers in enumerate(tiny_doubletons.sample_indices * 2 + tiny_doubletons.ploidy_indices):
        expected = np.count_nonzero(flat[:, carriers[0]] != flat[:, carriers[1]])
        assert tiny_diversity.pair_difference_counts[row] == expected
        assert flat[2, carriers[0]] != flat[2, carriers[1]]
    order = np.array([2, 0, 1])
    reordered = make_tiny_store(tiny_genotypes[:, order], tiny_positions,
                                sample_ids=tiny['sample_id'][:][order])
    remapped = error_estimation.compute_diversity(tiny, tiny_doubletons, zarr_path=reordered, num_workers=1)
    np.testing.assert_array_equal(remapped.pair_difference_counts, tiny_diversity.pair_difference_counts)
    np.testing.assert_allclose(remapped.global_difference_sum, tiny_diversity.global_difference_sum, rtol=1e-12)
    override_path = pathlib.Path(temporary.name) / 'reordered.zarr'
    make_tiny_store(tiny_genotypes[:, order], tiny_positions * 2, path=override_path,
                    sample_ids=tiny['sample_id'][:][order])
    overridden = error_estimation.compute_diversity(tiny, tiny_doubletons, zarr_path=override_path, num_workers=1)
    np.testing.assert_array_equal(overridden.pair_difference_counts, tiny_diversity.pair_difference_counts)
    assert overridden.span_bp == tiny_diversity.span_bp * 2
    np.testing.assert_allclose(overridden.pair_pi_per_bp, tiny_diversity.pair_pi_per_bp / 2, rtol=1e-12)
    absent = make_tiny_store(tiny_genotypes, tiny_positions, sample_ids=['absent', 'sample_1', 'sample_2'])
    assert_value_error(error_estimation.compute_diversity, tiny, tiny_doubletons,
                       zarr_path=absent, num_workers=1, match='absent')
    negative_genotypes = tiny_genotypes.copy()
    negative_genotypes[2, 0, 0] = -1
    negative = make_tiny_store(negative_genotypes, tiny_positions)
    assert_value_error(error_estimation.compute_diversity, negative, tiny_doubletons,
                       num_workers=1, match='Negative allele code at site 2')
    assert_value_error(error_estimation.sample_doubletons, negative,
                       config=dataclasses.replace(tiny_config, num_workers=2), match='Negative allele code')
    unphased = make_tiny_store(tiny_genotypes, tiny_positions, phased=np.zeros((6, 3), dtype=bool))
    assert_value_error(error_estimation.summarise_mismatches, unphased, tiny_doubletons,
                       config=tiny_config, recombination=0, match='Unphased heterozygote')
    zero_summary = error_estimation.summarise_mismatches(tiny, tiny_doubletons, config=tiny_config, recombination=0)
    assert np.all(zero_summary.left_rates == 0)
    assert np.all(zero_summary.right_rates == 0)
    synthetic_rate = 1e-8  # Synthetic constant for a units/endpoint check, not a data fit.
    constant_map = msprime.RateMap(position=[0, 500], rate=[synthetic_rate])
    scalar_summary = error_estimation.summarise_mismatches(tiny, tiny_doubletons, config=tiny_config, recombination=synthetic_rate)
    map_summary = error_estimation.summarise_mismatches(tiny, tiny_doubletons, config=tiny_config, recombination=constant_map)
    np.testing.assert_array_equal(scalar_summary.total_counts, map_summary.total_counts)
    np.testing.assert_allclose(scalar_summary.left_rates, map_summary.left_rates, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(scalar_summary.right_rates, map_summary.right_rates, rtol=1e-12, atol=1e-20)
    assert np.all(scalar_summary.left_rates[0] == 0)  # Empty 50 bp sides.
    assert np.all(scalar_summary.right_rates[0] == 0)
    np.testing.assert_allclose(scalar_summary.left_rates[1], synthetic_rate, rtol=1e-12)
    np.testing.assert_allclose(scalar_summary.right_rates[1], synthetic_rate, rtol=1e-12)
    np.testing.assert_array_equal(scalar_summary.left_counts, [[0, 0], [0, 1]])
    np.testing.assert_array_equal(scalar_summary.right_counts, [[0, 0], [1, 0]])
    parallel_diversity = error_estimation.compute_diversity(tiny, tiny_doubletons, num_workers=2)
    np.testing.assert_array_equal(parallel_diversity.pair_difference_counts, tiny_diversity.pair_difference_counts)
    np.testing.assert_allclose(parallel_diversity.global_difference_sum, tiny_diversity.global_difference_sum, rtol=1e-12)
    invalid_map = msprime.RateMap(position=[0, 250, 500], rate=[synthetic_rate, np.nan])
    assert_value_error(error_estimation.summarise_mismatches, tiny, tiny_doubletons,
                       config=tiny_config, recombination=invalid_map, match='undefined')
    short_map = msprime.RateMap(position=[0, 400], rate=[synthetic_rate])
    assert_value_error(error_estimation.summarise_mismatches, tiny, tiny_doubletons,
                       config=tiny_config, recombination=short_map, match='cover')
    assert_value_error(error_estimation.summarise_mismatches, tiny, tiny_doubletons,
                       config=tiny_config, recombination=-1, match='nonnegative')
    print('Singleton, explicit diversity, carrier remapping, error propagation and map endpoint checks passed', flush=True)

# %% [markdown]
# # Execution commands and results
# From the repository root:
# ```sh
# uv sync
# uv run python -c 'import dphil_analysis, numpy, numba, scipy, zarr, msprime; assert int(zarr.__version__.split(".")[0]) >= 3'
# uv run python notebooks/ch4_error_estimation/implementation_testing.py
# ```

# %%
if __name__ == '__main__':
    temporary.cleanup()
