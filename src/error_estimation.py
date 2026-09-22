"""Estimate additive errors per haplotype-bp using the aggregate FitErrorRate model.

Inputs are complete, phased, fixed-ploidy, biallelic 0/1 genotypes on one
sequence with sorted, unique positions. All stored sites contribute to diversity
and mismatches; observed allele counts determine doubleton ascertainment.
"""

import dataclasses
import logging
import multiprocessing
import pathlib
import queue

import msprime
import numpy as np
import scipy.optimize
import zarr

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class EstimationConfig:
    """One-sided physical windows, sampling and aggregate optimisation settings.

    ``num_workers=None`` uses available CPUs for mismatch chunks only.
    A generated ``random_seed`` is recorded in :class:`ErrorRateEstimate`.
    """

    window_sizes: np.ndarray
    num_doubletons: int = 10000
    random_seed: int | None = None
    num_workers: int | None = None
    queue_depth: int = 2
    worker_poll_seconds: float = 0.2
    optimizer_bounds: tuple = ((-30, 30), (-30, 30), (-50, float(np.log(0.5))))
    start_alpha: tuple = (0.25, 0.5, 0.75)
    start_log_beta: tuple = (-12, -9, -6, -3, 0)
    start_log_epsilon: tuple = (-50, -30, -20, -12)
    objective_penalty: float = 1e100
    residual_guard: float = 1e50

    def __post_init__(self):
        self.window_sizes = np.array(self.window_sizes, dtype=float, copy=True)
        windows = self.window_sizes
        if windows.ndim != 1 or windows.size == 0:
            raise ValueError('window_sizes must be a nonempty one-dimensional array')
        if np.any(~np.isfinite(windows)) or np.any(windows <= 0) or np.any(np.diff(windows) <= 0):
            raise ValueError('window_sizes must be positive, finite and increasing')
        if self.num_doubletons <= 0:
            raise ValueError('num_doubletons must be positive')


@dataclasses.dataclass
class Doubletons:
    """Selected sites in genomic order and their two carrier coordinates."""

    site_indices: np.ndarray
    positions: np.ndarray
    sample_indices: np.ndarray
    ploidy_indices: np.ndarray
    num_eligible: int


@dataclasses.dataclass
class Diversity:
    """Global diversity over all stored sites, normalised by sites or bp."""

    num_sites: int
    num_haplotypes: int
    span_bp: float
    global_difference_sum: float

    @property
    def global_pi_per_site(self):
        return self.global_difference_sum / self.num_sites

    @property
    def global_pi_per_bp(self):
        return self.global_difference_sum / self.span_bp


@dataclasses.dataclass
class MismatchSummary:
    """Full-window counts and per-bp recombination rates, each shaped (W, D).

    Physical endpoints are included, the focal site is excluded. Genetic
    distances run between the first and last included sites and are divided
    by the full physical window length, 2L.
    """

    window_sizes: np.ndarray
    counts: np.ndarray
    rates: np.ndarray

    @property
    def observed_means(self):
        return np.mean(self.counts, axis=1)

    @property
    def mean_rates(self):
        return np.mean(self.rates, axis=1)


@dataclasses.dataclass
class ErrorRateFit:
    """Aggregate fit; epsilon is additive errors per haplotype-bp, pi is per bp."""

    epsilon: float
    mu: float
    sigma_sq: float
    objective: float
    success: bool
    message: str
    observed_means: np.ndarray
    fitted_means: np.ndarray
    pi: float


@dataclasses.dataclass
class ErrorRateEstimate:
    """Results and resolved settings from :func:`estimate_error_rate`."""

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


def _open_store(zarr_path):
    group = zarr.open_group(zarr_path, mode='r')
    positions = np.asarray(group['variant_position'][:], dtype=float)
    if np.any(np.diff(positions) == 0):
        raise ValueError('Duplicate variant positions are unsupported')
    return _Store(group, positions)


def _sample_doubletons(G, ac, positions, config):
    dbtn_sites = np.flatnonzero(ac == 2)
    dbtn_pos = positions[dbtn_sites]
    max_L = config.window_sizes[-1]
    keep = (dbtn_pos - max_L >= positions[0]) & (dbtn_pos + max_L <= positions[-1])
    dbtn_sites = dbtn_sites[keep]
    dbtn_pos = dbtn_pos[keep]
    num_eligible = len(dbtn_sites)
    if num_eligible == 0:
        raise ValueError('No doubletons have complete max-L windows')
    rng = np.random.default_rng(config.random_seed)
    n = min(config.num_doubletons, num_eligible)
    if n < num_eligible:
        selected = rng.choice(num_eligible, size=n, replace=False)
        dbtn_sites = dbtn_sites[selected]
        dbtn_pos = dbtn_pos[selected]
    order = np.argsort(dbtn_sites)
    dbtn_sites = dbtn_sites[order]
    dbtn_pos = dbtn_pos[order]
    dbtn_G = G[dbtn_sites]
    _, samples_long, ploidies_long = np.nonzero(dbtn_G)
    return Doubletons(
        dbtn_sites, dbtn_pos, samples_long.reshape(n, 2),
        ploidies_long.reshape(n, 2), num_eligible,
    )


@dataclasses.dataclass
class _MismatchContribution:
    first: int
    last: int
    counts: np.ndarray


def _mismatch_chunk(genotype, index, left_sites, right_sites, focal_sites, carriers):
    """Compare active pairs in a storage chunk; clip every count to its window.

    Only chunks intersecting a pair's max-L interval compare that pair.
    Vectorisation adds at most one chunk of excess work at either endpoint.
    """
    start = index * genotype.chunks[0]
    stop = min(start + genotype.chunks[0], genotype.shape[0])
    first = np.searchsorted(right_sites[-1], start, side='right')
    last = np.searchsorted(left_sites[-1], stop, side='left')
    if first == last:
        return None
    block = np.asarray(genotype[start:stop]).reshape(stop - start, -1)
    active = carriers[first:last]
    mismatch = block[:, active[:, 0]] != block[:, active[:, 1]]
    focal = focal_sites[first:last] - start
    inside = (focal >= 0) & (focal < len(block))
    columns = np.arange(last - first)
    mismatch[focal[inside], columns[inside]] = False
    cumulative = np.zeros((len(block) + 1, last - first), dtype=np.int64)
    np.cumsum(mismatch, axis=0, out=cumulative[1:])
    left = np.clip(left_sites[:, first:last] - start, 0, len(block))
    right = np.clip(right_sites[:, first:last] - start, 0, len(block))
    counts = cumulative[right, columns] - cumulative[left, columns]
    return _MismatchContribution(first, last, counts)


def _mismatch_worker(path, left, right, focal, carriers, jobs, results):
    """Open once and consume mismatch chunk indices from the work queue."""
    try:
        genotype = zarr.open_group(path, mode='r')['call_genotype']
        while True:
            index = jobs.get()
            if index is None:
                return
            result = _mismatch_chunk(genotype, index, left, right, focal, carriers)
            results.put(result)
    except Exception as exc:
        results.put(exc)


def _mismatch_counts(path, genotype, left, right, focal, carriers, config):
    counts = np.zeros(left.shape, dtype=np.int64)
    num_chunks = (genotype.shape[0] + genotype.chunks[0] - 1) // genotype.chunks[0]
    workers = config.num_workers
    if workers is None:
        workers = multiprocessing.cpu_count()
    workers = min(workers, num_chunks)
    if workers == 1:
        for index in range(num_chunks):
            result = _mismatch_chunk(genotype, index, left, right, focal, carriers)
            if result is not None:
                counts[:, result.first:result.last] += result.counts
        return counts
    context = multiprocessing.get_context('spawn')
    capacity = workers * config.queue_depth
    jobs = context.Queue(maxsize=capacity)
    results = context.Queue(maxsize=capacity)
    processes = [
        context.Process(
            target=_mismatch_worker,
            args=(path, left, right, focal, carriers, jobs, results),
        )
        for _ in range(workers)
    ]
    try:
        for process in processes:
            process.start()
        submitted = 0
        completed = 0
        while completed < num_chunks:
            while submitted < num_chunks and submitted - completed < capacity:
                jobs.put(submitted)
                submitted += 1
            try:
                result = results.get(timeout=config.worker_poll_seconds)
            except queue.Empty:
                if any(process.exitcode is not None for process in processes):
                    raise RuntimeError('Mismatch worker exited unexpectedly')
                continue
            if isinstance(result, Exception):
                raise result
            if result is not None:
                counts[:, result.first:result.last] += result.counts
            completed += 1
        for _ in processes:
            jobs.put(None)
        for process in processes:
            process.join()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        jobs.close()
        results.close()
    return counts


def _summarise_mismatches(path, store, doubletons, recombination, config):
    """Prepare full windows and map distances, then run bounded mismatch work."""
    windows = config.window_sizes[:, None]
    positions = doubletons.positions[None, :]
    left = np.searchsorted(store.positions, positions - windows, side='left')
    right = np.searchsorted(store.positions, positions + windows, side='right')
    if isinstance(recombination, (str, pathlib.Path)):
        recombination = msprime.RateMap.read_hapmap(
            recombination, position_col=1, rate_col=2,
        )
    if isinstance(recombination, msprime.RateMap):
        cumulative = recombination.get_cumulative_mass(store.positions)
    else:
        cumulative = store.positions * recombination
    distances = cumulative[right - 1] - cumulative[left]
    rates = distances / (2 * windows)
    genotype = store.group['call_genotype']
    carriers = doubletons.sample_indices * genotype.shape[2] + doubletons.ploidy_indices
    counts = _mismatch_counts(
        path, genotype, left, right, doubletons.site_indices, carriers, config,
    )
    return MismatchSummary(config.window_sizes.copy(), counts, rates)


def _power_integral_from_one(z, exponent):
    """Return integral from 1 to z of x**exponent, including its log limit."""
    log_z = np.log(z)
    power = exponent + 1
    if np.isclose(power, 0):
        return log_z
    return np.expm1(power * log_z) / power


def _expected_haplotype_mismatches(mu, sigma_sq, pi, length, rate):
    """One-side gamma-time mismatch expectation, with pi in per-bp units."""
    alpha = mu**2 / sigma_sq
    beta = mu / sigma_sq
    lengths = np.asarray(length, dtype=float)
    rates = np.asarray(rate, dtype=float)
    expected = np.zeros(lengths.shape, dtype=float)
    nonzero = rates != 0
    q = 2 * rates[nonzero]
    z = 1 + q * lengths[nonzero] / beta
    survival_integral = beta / q * _power_integral_from_one(z, -alpha)
    values = pi * (lengths[nonzero] - survival_integral)
    expected[nonzero] = np.maximum(values, 0)
    return expected


def _fitted_means(mu, sigma_sq, epsilon, lengths, rates, pi):
    haplotype = _expected_haplotype_mismatches(mu, sigma_sq, pi, lengths, rates)
    return 2 * haplotype + 4 * epsilon * lengths


def _objective(params, lengths, observed, mean_r, pi, config):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        mu = np.exp(params[0])
        sigma_sq = mu**2 + np.exp(params[1])
        epsilon = np.exp(params[2])
        fitted = _fitted_means(mu, sigma_sq, epsilon, lengths, mean_r, pi)
    # A rounded-away excess no longer satisfies the strict variance constraint.
    if sigma_sq <= mu**2:
        return config.objective_penalty
    if not np.all(np.isfinite([mu, sigma_sq, epsilon])) or np.any(~np.isfinite(fitted)):
        return config.objective_penalty
    scale = np.maximum(np.abs(observed), 1)
    residuals = (observed - fitted) / scale
    if np.any(~np.isfinite(residuals)) or np.any(np.abs(residuals) > config.residual_guard):
        return config.objective_penalty
    value = np.sum(residuals**2)
    if not np.isfinite(value):
        return config.objective_penalty
    return float(value)


def fit_error_model(summary, *, pi, config):
    """Fit aggregate means with scalar per-bp diversity and historical multistarts.

    Retains FitErrorRate's gamma-time expectation, transformed parameters,
    bounds, L-BFGS-B and scaled least-squares objective. Numerical guards reject
    invalid trials; the returned success flag reports optimiser convergence.
    """
    pi = float(pi)
    observed = summary.observed_means
    mean_r = summary.mean_rates
    best = None
    for alpha in config.start_alpha:
        for log_beta in config.start_log_beta:
            beta = np.exp(log_beta)
            mu = alpha / beta
            sigma_sq = alpha / beta**2
            excess = max(sigma_sq - mu**2, np.finfo(float).tiny)
            for log_epsilon in config.start_log_epsilon:
                initial = np.array([np.log(mu), np.log(excess), log_epsilon])
                result = scipy.optimize.minimize(
                    _objective, x0=initial, args=(summary.window_sizes, observed, mean_r, pi, config),
                    method='L-BFGS-B', bounds=config.optimizer_bounds,
                )
                if np.isfinite(result.fun) and result.fun < config.objective_penalty:
                    if best is None or result.fun < best.fun:
                        best = result
    if best is None:
        raise RuntimeError('Error-model fitting failed: every optimisation trial was invalid or penalised')
    mu = float(np.exp(best.x[0]))
    sigma_sq = float(mu**2 + np.exp(best.x[1]))
    epsilon = float(np.exp(best.x[2]))
    fitted = _fitted_means(mu, sigma_sq, epsilon, summary.window_sizes, mean_r, pi)
    return ErrorRateFit(
        epsilon=epsilon, mu=mu, sigma_sq=sigma_sq, objective=float(best.fun),
        success=bool(best.success), message=str(best.message),
        observed_means=observed, fitted_means=fitted, pi=pi,
    )


def estimate_error_rate(zarr_path, *, recombination, config, pi=None):
    """Estimate additive errors per haplotype-bp from a complete genotype store.

    ``recombination`` is an explicit HapMap path, msprime RateMap or scalar
    per-bp per-generation rate. ``pi`` optionally overrides global per-bp
    diversity with a scalar. Genotypes are released before mismatch workers
    open the store; only that phase uses chunking and multiprocessing.
    """
    path = str(pathlib.Path(zarr_path).expanduser().resolve())
    seed = config.random_seed
    if seed is None:
        seed = np.random.SeedSequence().entropy
    config = dataclasses.replace(config, random_seed=seed)
    store = _open_store(path)
    G = np.asarray(store.group['call_genotype'][:])
    ac = G.sum(axis=(1, 2))
    doubletons = _sample_doubletons(G, ac, store.positions, config)
    H = G.shape[1] * G.shape[2]
    site_diversity = 2 * ac * (H - ac) / (H * (H - 1))
    diversity = Diversity(
        num_sites=len(store.positions), num_haplotypes=H,
        span_bp=float(store.positions[-1] - store.positions[0]),
        global_difference_sum=float(site_diversity.sum()),
    )
    del G
    del ac
    mismatches = _summarise_mismatches(path, store, doubletons, recombination, config)
    if pi is None:
        pi = diversity.global_pi_per_bp
    fit = fit_error_model(mismatches, pi=pi, config=config)
    return ErrorRateEstimate(doubletons, diversity, mismatches, fit, path, config)
