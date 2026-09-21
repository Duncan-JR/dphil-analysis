"""Estimate additive errors per haplotype-bp from observed doubletons.

The gamma-time expectation and multistart fit follow ``FitErrorRate`` in the
historical tsinfer-paper analysis. Ascertainment here uses observed alternate
allele-1 counts, not simulation truth or inferred ancestral alleles. Calls must
be complete and phased; absent phase flags imply an explicit phased-ordering
assumption. Singletons and multiallelic sites remain in mismatch and diversity
calculations. No tree sequence or error truth is used.
"""

import dataclasses
import logging
import multiprocessing
import operator
import pathlib
import queue

import msprime
import numba
import numpy as np
import scipy.optimize
import zarr

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class _ExecutionConfig:
    num_workers: int | None = None
    queue_depth: int = 2
    worker_poll_seconds: float = 0.2

    def __post_init__(self):
        if self.num_workers is not None:
            self.num_workers = _positive_integer(self.num_workers, 'num_workers')
        self.queue_depth = _positive_integer(self.queue_depth, 'queue_depth')
        if not np.isfinite(self.worker_poll_seconds) or self.worker_poll_seconds <= 0:
            raise ValueError('worker_poll_seconds must be positive and finite')


@dataclasses.dataclass(kw_only=True)
class EstimationConfig(_ExecutionConfig):
    """Sampling, physical windows and numerical optimisation settings.

    Parameters
    ----------
    window_sizes : array_like
        Strictly increasing positive one-sided lengths, in bp.
    num_doubletons : int
        Maximum sample size; :func:`sample_doubletons` uses all eligible sites
        when fewer are available.
    random_seed : int or None
        Local generator seed. :func:`estimate_error_rate` records a generated
        seed when None is supplied.
    num_workers : int or None
        Number of processes. None resolves to available CPUs at dispatch only.
    """

    window_sizes: np.ndarray
    num_doubletons: int = 10000
    random_seed: int | None = None
    optimizer_bounds: tuple = ((-30, 30), (-30, 30), (-50, float(np.log(0.5))))
    start_alpha: tuple = (0.25, 0.5, 0.75)
    start_log_beta: tuple = (-12, -9, -6, -3, 0)
    start_log_epsilon: tuple = (-50, -30, -20, -12)
    objective_penalty: float = 1e100
    residual_guard: float = 1e50

    def __post_init__(self):
        super().__post_init__()
        self.window_sizes = np.array(self.window_sizes, dtype=float, copy=True)
        windows = self.window_sizes
        if windows.ndim != 1 or windows.size == 0:
            raise ValueError('window_sizes must be a nonempty one-dimensional array')
        if np.any(~np.isfinite(windows)) or np.any(windows <= 0):
            raise ValueError('window_sizes must be positive and finite')
        if np.any(np.diff(windows) <= 0):
            raise ValueError('window_sizes must be strictly increasing')
        self.num_doubletons = _positive_integer(self.num_doubletons, 'num_doubletons')
        if self.random_seed is not None:
            self.random_seed = operator.index(self.random_seed)
            if self.random_seed < 0:
                raise ValueError('random_seed must be nonnegative')
        bounds = np.asarray(self.optimizer_bounds)
        if bounds.shape != (3, 2) or np.any(~np.isfinite(bounds)):
            raise ValueError('optimizer_bounds must be a finite (3, 2) array')
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError('optimizer_bounds must have lower < upper')
        for starts in (self.start_alpha, self.start_log_beta, self.start_log_epsilon):
            values = np.asarray(starts)
            if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
                raise ValueError('optimizer start grids must be nonempty finite vectors')
        if np.any(np.asarray(self.start_alpha) <= 0) or np.any(np.asarray(self.start_alpha) >= 1):
            raise ValueError('start_alpha must lie strictly between zero and one')
        if not np.isfinite(self.objective_penalty) or self.objective_penalty <= 0:
            raise ValueError('objective_penalty must be positive and finite')
        if not np.isfinite(self.residual_guard) or self.residual_guard <= 0:
            raise ValueError('residual_guard must be positive and finite')


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a positive integer')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


@dataclasses.dataclass
class Doubletons:
    """Stable genomic row order from :func:`sample_doubletons`.

    Carriers have (D, 2) sample IDs and ploidy slots. Numerical sample indices
    refer only to the inference store. ``source_ploidy`` checks compatibility
    when :func:`compute_diversity` resolves identities in another store.
    """

    site_indices: np.ndarray
    positions: np.ndarray
    sample_indices: np.ndarray
    sample_ids: np.ndarray
    ploidy_indices: np.ndarray
    num_eligible: int
    source_ploidy: int

    @property
    def num_doubletons(self):
        return len(self.site_indices)


@dataclasses.dataclass
class Diversity:
    """All-site diversity returned by :func:`compute_diversity`.

    Per-site values are conditional on stored sites; per-bp values divide by
    the diversity store's first-to-last position span. Carrier rows follow the
    supplied :class:`Doubletons`, including repeated pairs at distinct sites.
    """

    num_sites: int
    num_haplotypes: int
    span_bp: float
    global_difference_sum: float
    pair_difference_counts: np.ndarray
    source_path: str
    sample_ids: np.ndarray
    ploidy_indices: np.ndarray

    @property
    def global_pi_per_site(self):
        return self.global_difference_sum / self.num_sites

    @property
    def pair_pi_per_site(self):
        return self.pair_difference_counts / self.num_sites

    @property
    def global_pi_per_bp(self):
        return self.global_difference_sum / self.span_bp

    @property
    def pair_pi_per_bp(self):
        return self.pair_difference_counts / self.span_bp


@dataclasses.dataclass
class MismatchSummary:
    """(W, D) side counts and rates from :func:`summarise_mismatches`.

    Windows include their physical endpoints and exclude the focal site.
    Rates are genetic distance between included site endpoints divided by the
    one-sided window length, in bp, per generation.
    """

    window_sizes: np.ndarray
    left_counts: np.ndarray
    right_counts: np.ndarray
    left_rates: np.ndarray
    right_rates: np.ndarray

    @property
    def total_counts(self):
        return self.left_counts + self.right_counts

    @property
    def window_mean_rate(self):
        return (self.left_rates + self.right_rates) / 2

    @property
    def observed_means(self):
        return np.mean(self.total_counts, axis=1)

    @property
    def observed_side_means(self):
        return self.observed_means / 2


@dataclasses.dataclass
class ErrorRateFit:
    """Numerical outcome of :func:`fit_error_model`, including convergence.

    Epsilon is additive errors per haplotype-bp, not per-genotype probability.
    ``pi`` is explicitly per bp; ``fit_mode`` is aggregate or per_doubleton.
    """

    epsilon: float
    mu: float
    sigma_sq: float
    objective: float
    success: bool
    message: str
    observed_means: np.ndarray
    fitted_means: np.ndarray
    pi: float | np.ndarray
    fit_mode: str


@dataclasses.dataclass
class ErrorRateEstimate:
    """Grouped results and reproducibility settings of :func:`estimate_error_rate`."""

    doubletons: Doubletons
    diversity: Diversity
    mismatches: MismatchSummary
    fit: ErrorRateFit
    inference_path: str
    config: EstimationConfig


@dataclasses.dataclass
class _Store:
    group: zarr.Group
    positions: np.ndarray
    sample_ids: np.ndarray
    source_path: str

    @property
    def ploidy(self):
        return self.group['call_genotype'].shape[2]


@dataclasses.dataclass
class _ChunkResult:
    index: int
    value: object = None
    error: Exception | None = None


def _open_store(zarr_path):
    """Open a path or Zarr group read-only and validate the fixed-ploidy schema."""
    if isinstance(zarr_path, zarr.Group):
        group = zarr.open_group(zarr_path.store, path=zarr_path.path, mode='r')
        source_path = str(zarr_path.store)
    else:
        source_path = str(pathlib.Path(zarr_path).resolve())
        group = zarr.open_group(source_path, mode='r')
    for name in ('call_genotype', 'variant_position', 'sample_id', 'variant_contig', 'contig_id'):
        if name not in group:
            raise ValueError(f'Missing required array: {name}')
    genotype = group['call_genotype']
    if genotype.ndim != 3 or not np.issubdtype(genotype.dtype, np.integer):
        raise ValueError('call_genotype must be a three-dimensional integer array')
    sites, samples, ploidy = genotype.shape
    if sites < 2 or samples < 1 or ploidy < 1 or samples * ploidy < 2:
        raise ValueError('Require at least two sites and two fixed-ploidy haplotypes')
    positions = np.asarray(group['variant_position'][:], dtype=float)
    if positions.shape != (sites,) or np.any(~np.isfinite(positions)):
        raise ValueError('variant_position must be finite with one value per site')
    if np.any(positions < 0) or np.any(np.diff(positions) <= 0):
        raise ValueError('Positions must be nonnegative and strictly increasing; duplicates unsupported')
    sample_ids = np.asarray(group['sample_id'][:], dtype='T')
    if sample_ids.shape != (samples,) or len(np.unique(sample_ids)) != samples:
        raise ValueError('sample_id must contain one unique ID per sample')
    contigs = group['variant_contig'][:]
    if contigs.shape != (sites,) or len(np.unique(contigs)) != 1:
        raise ValueError('Exactly one represented contig is required')
    if not np.issubdtype(contigs.dtype, np.integer):
        raise ValueError('variant_contig must contain integer contig indices')
    if group['contig_id'].ndim != 1 or not 0 <= contigs[0] < group['contig_id'].shape[0]:
        raise ValueError('variant_contig is outside contig_id bounds')
    if 'call_genotype_phased' in group:
        if group['call_genotype_phased'].shape != (sites, samples):
            raise ValueError('call_genotype_phased must have shape (sites, samples)')
    if 'call_genotype_mask' in group:
        if group['call_genotype_mask'].shape != genotype.shape:
            raise ValueError('call_genotype_mask must match call_genotype shape')
    return _Store(group, positions, sample_ids, source_path)


def _read_genotype_chunk(group, index):
    genotype = group['call_genotype']
    offset = index * genotype.chunks[0]
    stop = min(offset + genotype.chunks[0], genotype.shape[0])
    block = np.asarray(genotype[offset:stop])
    negative = np.argwhere(block < 0)
    if len(negative) > 0:
        site = offset + negative[0, 0]
        raise ValueError(f'Negative allele code at site {site} in chunk {index}')
    if 'call_genotype_mask' in group:
        if np.any(group['call_genotype_mask'][offset:stop]):
            raise ValueError(f'Masked genotype in chunk {index}, sites [{offset}, {stop})')
    if 'call_genotype_phased' in group:
        phase = np.asarray(group['call_genotype_phased'][offset:stop])
        if np.any((phase != 0) & (phase != 1)):
            raise ValueError(f'Invalid phase flag in chunk {index}')
        heterozygous = np.any(block != block[:, :, :1], axis=2)
        invalid = np.argwhere(heterozygous & (phase == 0))
        if len(invalid) > 0:
            site = offset + invalid[0, 0]
            raise ValueError(f'Unphased heterozygote at site {site} in chunk {index}')
    return block.reshape(len(block), -1)


def _iter_genotype_chunks(store):
    """Yield offsets and validated contiguous blocks at the physical site chunk size."""
    genotype = store.group['call_genotype']
    for offset in range(0, genotype.shape[0], genotype.chunks[0]):
        yield offset, _read_genotype_chunk(store.group, offset // genotype.chunks[0])


def _resolve_carriers(store, doubletons):
    if store.ploidy != doubletons.source_ploidy:
        raise ValueError('Diversity and inference stores must have compatible ploidy')
    shape = (doubletons.num_doubletons, 2)
    if doubletons.sample_ids.shape != shape or doubletons.ploidy_indices.shape != shape:
        raise ValueError('Carrier IDs and ploidy indices must have shape (D, 2)')
    slots = doubletons.ploidy_indices
    if not np.issubdtype(slots.dtype, np.integer) or np.any(slots < 0) or np.any(slots >= store.ploidy):
        raise ValueError('Carrier ploidy index out of bounds')
    lookup = {name: index for index, name in enumerate(store.sample_ids)}
    indices = np.empty(shape, dtype=np.int64)
    for row in range(doubletons.num_doubletons):
        for side in range(2):
            name = doubletons.sample_ids[row, side]
            if name not in lookup:
                raise ValueError(f'Carrier sample ID absent from target store: {name}')
            indices[row, side] = lookup[name]
    return indices * store.ploidy + slots


@numba.njit(cache=True)
def _scan_doubletons(block, offset, bounds):
    candidates = np.empty((len(block), 3), dtype=np.int64)
    count = 0
    for row in range(len(block)):
        site = offset + row
        if site < bounds[0] or site >= bounds[1]:
            continue
        ones = 0
        first = -1
        second = -1
        biallelic = True
        for hap in range(block.shape[1]):
            allele = block[row, hap]
            if allele > 1:
                biallelic = False
                break
            if allele == 1:
                ones += 1
                if ones == 1:
                    first = hap
                elif ones == 2:
                    second = hap
        if biallelic and ones == 2:
            candidates[count, 0] = site
            candidates[count, 1] = first
            candidates[count, 2] = second
            count += 1
    return candidates[:count].copy()


def _chunk_task(block, offset, operation, payload):
    if operation == 'sample':
        return _scan_doubletons(block, offset, payload)
    if operation == 'reduce':
        return _reduce_chunk(block, offset, payload)
    raise ValueError(f'Unknown chunk operation: {operation}')


def _chunk_worker(store_reference, operation, payload, jobs, results):
    """Spawn-safe queue worker: open locally once, return compact partials."""
    try:
        store, path = store_reference
        group = zarr.open_group(store, path=path, mode='r')
        chunk_size = group['call_genotype'].chunks[0]
        while True:
            index = jobs.get()
            if index is None:
                return
            block = _read_genotype_chunk(group, index)
            value = _chunk_task(block, index * chunk_size, operation, payload)
            results.put(_ChunkResult(index, value))
    except Exception as exc:
        results.put(_ChunkResult(-1, error=exc))


def _execute_chunks(store, operation, payload, execution):
    """Yield compact results in genomic order with bounded outstanding jobs.

    Worker failures, including abnormal exits, terminate and join all children.
    Queues carry chunk indices rather than genotype blocks.
    """
    genotype = store.group['call_genotype']
    num_chunks = (genotype.shape[0] + genotype.chunks[0] - 1) // genotype.chunks[0]
    workers = execution.num_workers
    if workers is None:
        workers = multiprocessing.cpu_count()
    workers = min(workers, num_chunks)
    if workers == 1:
        for offset, block in _iter_genotype_chunks(store):
            yield _chunk_task(block, offset, operation, payload)
        return
    context = multiprocessing.get_context('spawn')
    capacity = workers * execution.queue_depth
    jobs = context.Queue(maxsize=capacity)
    results = context.Queue(maxsize=capacity)
    reference = (store.group.store, store.group.path)
    processes = [
        context.Process(target=_chunk_worker, args=(reference, operation, payload, jobs, results))
        for _ in range(workers)
    ]
    completed = False
    try:
        for process in processes:
            process.start()
        submitted = 0
        next_index = 0
        pending = {}
        while next_index < num_chunks:
            while submitted < num_chunks and submitted - next_index < capacity:
                jobs.put(submitted)
                submitted += 1
            try:
                result = results.get(timeout=execution.worker_poll_seconds)
            except queue.Empty:
                failed = [process for process in processes if process.exitcode is not None]
                if len(failed) > 0:
                    raise RuntimeError(f'Chunk worker exited unexpectedly: {failed[0].exitcode}')
                continue
            if result.error is not None:
                raise result.error
            pending[result.index] = result.value
            while next_index in pending:
                yield pending.pop(next_index)
                next_index += 1
        for _ in processes:
            jobs.put(None)
        for process in processes:
            process.join()
        completed = True
    finally:
        if not completed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                if process.pid is not None:
                    process.join()
        for channel in (jobs, results):
            channel.cancel_join_thread()
            channel.close()


def sample_doubletons(zarr_path, *, config):
    """Uniformly sample eligible observed allele-1 doubletons without replacement.

    Parameters
    ----------
    zarr_path : pathlib.Path, str or zarr.Group
        Complete phased fixed-ploidy genotype store, opened read-only.
    config : EstimationConfig
        Physical windows, reservoir capacity, local seed and worker settings.

    Returns
    -------
    Doubletons
        Genomically ordered sites and their two haplotype addresses. Homozygous
        carriers contribute two distinct slots. Zero eligible sites raises
        ValueError. See :func:`compute_diversity` for carrier-ID remapping.
    """
    store = _open_store(zarr_path)
    positions = store.positions
    largest = config.window_sizes[-1]
    lower = np.searchsorted(positions, positions[0] + largest, side='left')
    upper = np.searchsorted(positions, positions[-1] - largest, side='right')
    bounds = np.array([lower, upper], dtype=np.int64)
    rng = np.random.default_rng(config.random_seed)
    reservoir = np.empty((config.num_doubletons, 3), dtype=np.int64)
    eligible = 0
    for candidates in _execute_chunks(store, 'sample', bounds, config):
        for candidate in candidates:
            eligible += 1
            index = eligible - 1
            if eligible > config.num_doubletons:
                index = rng.integers(eligible)
            if index < config.num_doubletons:
                reservoir[index] = candidate
    if eligible == 0:
        raise ValueError('No eligible doubletons have complete windows')
    selected = reservoir[:min(config.num_doubletons, eligible)]
    selected = selected[np.argsort(selected[:, 0])]
    sites = selected[:, 0]
    sample_indices = selected[:, 1:] // store.ploidy
    ploidy_indices = selected[:, 1:] % store.ploidy
    logger.info('Sampled %d of %d eligible doubletons', len(sites), eligible)
    return Doubletons(
        site_indices=sites, positions=positions[sites], sample_indices=sample_indices,
        sample_ids=store.sample_ids[sample_indices], ploidy_indices=ploidy_indices,
        num_eligible=eligible, source_ploidy=store.ploidy,
    )


@dataclasses.dataclass
class _ReductionTask:
    carriers: np.ndarray
    diversity: bool
    boundaries: np.ndarray | None


@dataclasses.dataclass
class _ReductionResult:
    global_difference_sum: float
    pair_difference_counts: np.ndarray
    left_counts: np.ndarray
    right_counts: np.ndarray


@dataclasses.dataclass
class _WindowSetup:
    boundaries: np.ndarray
    left_rates: np.ndarray
    right_rates: np.ndarray


@numba.njit(cache=True)
def _accumulate_diversity(block, carriers, pair_counts):
    """Accumulate distinct-pair diversity without materialising haplotype pairs."""
    haplotypes = block.shape[1]
    denominator = haplotypes * (haplotypes - 1)
    allele_counts = np.zeros(int(np.max(block)) + 1, dtype=np.int64)
    difference_sum = 0.0
    for site in range(len(block)):
        allele_counts[:] = 0
        for haplotype in range(haplotypes):
            allele_counts[block[site, haplotype]] += 1
        equal_pairs = 0
        for count in allele_counts:
            equal_pairs += count * (count - 1)
        difference_sum += (denominator - equal_pairs) / denominator
        for pair in range(len(carriers)):
            pair_counts[pair] += block[site, carriers[pair, 0]] != block[site, carriers[pair, 1]]
    return difference_sum


@numba.njit(cache=True)
def _accumulate_windows(block, offset, boundaries, carriers, pair_counts, left_counts, right_counts):
    """Query chunk-local pair prefixes at clipped, inclusive physical endpoints."""
    prefix = np.empty(len(block) + 1, dtype=np.int64)
    for pair in range(len(carriers)):
        prefix[0] = 0
        for site in range(len(block)):
            unequal = block[site, carriers[pair, 0]] != block[site, carriers[pair, 1]]
            prefix[site + 1] = prefix[site] + unequal
        pair_counts[pair] = prefix[-1]
        for window in range(boundaries.shape[1]):
            left, focal, right = boundaries[:, window, pair]
            start = min(max(left - offset, 0), len(block))
            stop = min(max(focal - offset, 0), len(block))
            left_counts[window, pair] = prefix[stop] - prefix[start]
            start = min(max(focal + 1 - offset, 0), len(block))
            stop = min(max(right - offset, 0), len(block))
            right_counts[window, pair] = prefix[stop] - prefix[start]


def _reduce_chunk(block, offset, task):
    pairs = len(task.carriers)
    windows = 0 if task.boundaries is None else task.boundaries.shape[1]
    result = _ReductionResult(
        global_difference_sum=0.0,
        pair_difference_counts=np.zeros(pairs, dtype=np.int64),
        left_counts=np.zeros((windows, pairs), dtype=np.int64),
        right_counts=np.zeros((windows, pairs), dtype=np.int64),
    )
    if task.diversity:
        carriers = task.carriers
        if task.boundaries is not None:
            # Window prefixes also give whole-chunk pair totals in the fused pass.
            carriers = np.empty((0, 2), dtype=np.int64)
        result.global_difference_sum = _accumulate_diversity(
            block, carriers, result.pair_difference_counts,
        )
    if task.boundaries is not None:
        _accumulate_windows(
            block, offset, task.boundaries, task.carriers,
            result.pair_difference_counts, result.left_counts, result.right_counts,
        )
    return result


def _reduce_chunks(store, doubletons, *, execution, diversity, boundaries=None):
    carriers = _resolve_carriers(store, doubletons)
    task = _ReductionTask(carriers, diversity, boundaries)
    pairs = len(carriers)
    windows = 0 if boundaries is None else boundaries.shape[1]
    result = _ReductionResult(
        0.0, np.zeros(pairs, dtype=np.int64),
        np.zeros((windows, pairs), dtype=np.int64),
        np.zeros((windows, pairs), dtype=np.int64),
    )
    for partial in _execute_chunks(store, 'reduce', task, execution):
        result.global_difference_sum += partial.global_difference_sum
        result.pair_difference_counts += partial.pair_difference_counts
        result.left_counts += partial.left_counts
        result.right_counts += partial.right_counts
    return result


def _site_recombination_cumsum(positions, recombination):
    """Integrate a literal HapMap path, RateMap or finite nonnegative scalar.

    No extrapolation over undefined map intervals is permitted. Distances are
    later differenced at included site endpoints, not physical window edges.
    """
    if isinstance(recombination, (str, pathlib.Path)):
        recombination = msprime.RateMap.read_hapmap(
            pathlib.Path(recombination), position_col=1, rate_col=2,
        )
    if isinstance(recombination, msprime.RateMap):
        if positions[0] < recombination.position[0] or positions[-1] > recombination.position[-1]:
            raise ValueError('Recombination map does not cover the represented positions')
        overlap = (recombination.left < positions[-1]) & (recombination.right > positions[0])
        rates = recombination.rate[overlap]
        if np.any(~np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError('Recombination map has undefined or invalid rates over the store span')
        cumulative = recombination.get_cumulative_mass(positions)
    else:
        rate = np.asarray(recombination, dtype=float)
        if rate.ndim != 0 or not np.isfinite(rate) or rate < 0:
            raise ValueError('Recombination rate must be a finite nonnegative scalar or map')
        cumulative = positions * float(rate)
    if np.any(~np.isfinite(cumulative)):
        raise ValueError('Nonfinite cumulative recombination distance')
    return cumulative


def _prepare_windows(store, doubletons, config, recombination):
    positions = store.positions
    sites = doubletons.site_indices
    if sites.ndim != 1 or len(sites) == 0 or not np.issubdtype(sites.dtype, np.integer):
        raise ValueError('Doubleton sites must be a nonempty integer vector')
    if np.any(sites < 0) or np.any(sites >= len(positions)) or np.any(np.diff(sites) <= 0):
        raise ValueError('Doubleton site indices must be ordered, unique and in bounds')
    if not np.array_equal(positions[sites], doubletons.positions):
        raise ValueError('Doubleton positions do not match the inference store')
    maximum = config.window_sizes[-1]
    if np.any(doubletons.positions - maximum < positions[0]) or np.any(doubletons.positions + maximum > positions[-1]):
        raise ValueError('Doubletons must have complete windows for every requested length')
    windows = config.window_sizes[:, None]
    focal_positions = doubletons.positions[None, :]
    left = np.searchsorted(positions, focal_positions - windows, side='left')
    right = np.searchsorted(positions, focal_positions + windows, side='right')
    focal = np.broadcast_to(sites, left.shape)
    boundaries = np.stack([left, focal, right])
    cumulative = _site_recombination_cumsum(positions, recombination)
    left_distance = cumulative[focal] - cumulative[left]
    right_distance = cumulative[right - 1] - cumulative[focal]
    left_rates = left_distance / windows
    right_rates = right_distance / windows
    return _WindowSetup(boundaries, left_rates, right_rates)


def _diversity_result(store, doubletons, reduction):
    return Diversity(
        num_sites=len(store.positions),
        num_haplotypes=len(store.sample_ids) * store.ploidy,
        span_bp=float(store.positions[-1] - store.positions[0]),
        global_difference_sum=reduction.global_difference_sum,
        pair_difference_counts=reduction.pair_difference_counts,
        source_path=store.source_path,
        sample_ids=doubletons.sample_ids.copy(),
        ploidy_indices=doubletons.ploidy_indices.copy(),
    )


def _mismatch_result(config, setup, reduction):
    return MismatchSummary(
        window_sizes=config.window_sizes.copy(),
        left_counts=reduction.left_counts, right_counts=reduction.right_counts,
        left_rates=setup.left_rates, right_rates=setup.right_rates,
    )


def compute_diversity(inference_zarr_path, doubletons, *, zarr_path=None, num_workers=None):
    """Compute global and carrier-pair diversity in one complete genotype scan.

    Parameters
    ----------
    inference_zarr_path : pathlib.Path, str or zarr.Group
        Default source of diversity, used only when zarr_path is None.
    doubletons : Doubletons
        Carrier identities in the stable order from :func:`sample_doubletons`.
    zarr_path : pathlib.Path, str, zarr.Group or None
        Optional alternative source. Sample IDs, not source row indices, resolve
        the carriers. Cross-chromosome ploidy slots are an indexing convention,
        not biological evidence of homolog correspondence. Extra samples still
        contribute to global diversity.
    num_workers : int or None
        Parallel chunk workers; None uses available CPUs at dispatch.

    Returns
    -------
    Diversity
        Both per-stored-site and per-bp normalisations using this source's span.
        Every site, including singletons, contributes. :func:`fit_error_model`
        requires the per-bp normalisation.
    """
    source = inference_zarr_path if zarr_path is None else zarr_path
    store = _open_store(source)
    execution = _ExecutionConfig(num_workers=num_workers)
    reduction = _reduce_chunks(store, doubletons, execution=execution, diversity=True)
    return _diversity_result(store, doubletons, reduction)


def summarise_mismatches(zarr_path, doubletons, *, config, recombination):
    """Count two-sided physical-window mismatches in one streamed scan.

    Parameters
    ----------
    zarr_path : pathlib.Path, str or zarr.Group
        Inference genotype store used by :func:`sample_doubletons`.
    doubletons : Doubletons
        Sampled focal sites and carriers, retaining their genomic row order.
    config : EstimationConfig
        Increasing positive window lengths, in bp, and worker settings.
    recombination : float, msprime.RateMap, pathlib.Path or str
        Explicit per-bp per-generation scalar rate or map; paths are literal
        HapMap inputs with position column 1 and rate column 2.

    Returns
    -------
    MismatchSummary
        Exact integer counts and side rates for :func:`fit_error_model`.
        Focal sites are excluded; empty sides have zero genetic distance.
    """
    store = _open_store(zarr_path)
    setup = _prepare_windows(store, doubletons, config, recombination)
    reduction = _reduce_chunks(
        store, doubletons, execution=config, diversity=False, boundaries=setup.boundaries,
    )
    return _mismatch_result(config, setup, reduction)
