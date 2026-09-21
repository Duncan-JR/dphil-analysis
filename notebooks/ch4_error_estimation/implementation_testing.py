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
