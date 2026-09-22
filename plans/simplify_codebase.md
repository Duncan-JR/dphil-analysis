# Plan: simplify error estimation and restore max-L-bounded mismatch computation

## Goal

Reduce `src/error_estimation.py` to the minimum implementation required to reproduce the intended aggregate `FitErrorRate` workflow from `tsinfer-paper`, while fixing the scaling problem in mismatch calculation.

The workflow should be:

```text
load genotype data
    ↓
simple NumPy allele counts
    ├── global diversity
    └── doubleton discovery + sampling + carrier identification
    ↓
release full genotype array
    ↓
calculate window boundaries + recombination distances
    ↓
bounded chunked mismatch calculation within ±max_L only
    ↓
aggregate FitErrorRate model
```

There should be:

* no pairwise diversity mode;
* no `per_doubleton` mode;
* no left/right mismatch fitting;
* no generic chunk-reduction framework;
* no chunked doubleton sampling;
* no subsetting/fixture generation;
* no permanent smoke-test notebook;
* minimal defensive validation.

Prefer short, direct analysis code over reusable abstractions or defensive library-style interfaces.

---

# 1. Remove failed and unnecessary functionality

Delete support for all of the following.

## Pairwise diversity

Remove:

* `diversity_mode`;
* `pair_difference_counts`;
* `pair_pi_per_site`;
* `pair_pi_per_bp`;
* `(D,)` `pi` values;
* alternative diversity chromosomes;
* `diversity_zarr_path`;
* sample-ID carrier remapping between stores;
* whole-chromosome carrier-pair diversity.

The model uses one scalar global `pi`.

## `per_doubleton`

Remove the `per_doubleton` argument everywhere.

Remove:

* per-doubleton fitting branches;
* side-specific fitting;
* `fit_mode`;
* concatenated left/right observations;
* pair-specific recombination expectations.

This was an unsuccessful experiment and is no longer part of the method.

There should be exactly one fitting path, equivalent to the old `per_doubleton=False` behaviour.

## Left/right mismatch summaries

Do not expose or retain:

```text
left_counts
right_counts
left_rates
right_rates
```

Instead, for each `L` and doubleton calculate:

* one total mismatch count across the full symmetric window;
* one recombination rate for the full window.

The focal doubleton site is excluded from the mismatch interval, although its two carrier haplotypes necessarily agree there so its contribution would be zero.

---

# 2. Remove the existing validation notebook entirely

Delete:

```text
notebooks/ch4_error_estimation/implementation_testing.py
notebooks/ch4_error_estimation/implementation_testing.ipynb
```

If the directory is then empty, delete:

```text
notebooks/ch4_error_estimation/
```

Remove README instructions referring to:

* `implementation_testing.py`;
* Jupytext conversion;
* notebook execution;
* notebook kernel installation.

If nothing else uses them, remove the notebook-only dependencies from `pyproject.toml`:

```text
jupyterlab
jupytext
ipykernel
nbconvert
nbformat
```

Do not replace this with another permanent test harness or notebook.

Temporary implementation checks may be run while fixing the repository but should not become part of the maintained code.

---

# 3. Use only one dataset during implementation validation

During this refactor, do not evaluate the method on any dataset except:

```text
~/work/tsinfer-anc-eval/data/anc_eval/zarr_vcfs/
OutOfAfrica_4J17-chr20-L0-R1e6-n300-s1-rep0-
geno-1.0-phase0.0-mispol0.0.zarr/
```

Use the complete store directly.

Do not:

* subset samples;
* subset sites;
* construct a temporary Zarr;
* select a smaller interval;
* add duplicate-removal preprocessing.

Duplicate coordinates have already been removed from this dataset.

Use the corresponding explicit chr20 HapMap file for recombination.

This small dataset can also be loaded fully into memory for temporary direct validation calculations.

---

# 4. Remove unnecessary defensive checks

The code is analysis code operating on controlled input stores, not a general-purpose public library.

Reduce defensive validation substantially.

Prefer:

* clear assumptions;
* simple code;
* natural NumPy/Zarr exceptions when an assumption is violated;

over explicit checks for every possible malformed input.

Do not retain validation simply because the previous implementation had it.

## `_open_store`

`_open_store()` should essentially just:

```python
group = zarr.open_group(path, mode="r")
positions = np.asarray(group["variant_position"][:])
```

and check that there are no duplicate positions:

```python
if np.any(np.diff(positions) == 0):
    raise ValueError("Duplicate variant positions are unsupported")
```

This duplicate check is worth retaining because duplicate coordinates make the physical-window/searchsorted semantics ambiguous while potentially producing superficially plausible results.

Remove the other current `_open_store` checks, including explicit checking of:

* required array names;
* genotype dimensionality;
* integer dtype;
* minimum sample/site counts;
* finite positions;
* nonnegative positions;
* strictly increasing order beyond the duplicate check;
* unique sample IDs;
* single represented contig;
* `variant_contig` dtype/bounds;
* phase-array shape;
* mask-array shape.

The analysis inputs are assumed to satisfy the expected schema.

If an array is absent or has an incompatible shape, ordinary indexing or NumPy/Zarr operations can fail naturally.

## Other functions

Apply the same principle throughout the module.

Remove validation whose only purpose is to provide a friendlier error for a condition that cannot occur in the intended workflow.

Examples to simplify or remove include:

* repeated shape checks on arrays produced internally;
* checking that sampled site indices are sorted and in bounds;
* checking that positions selected by `site_indices` equal cached positions;
* repeated finite/nonnegative checks on internally calculated mismatch counts;
* carrier sample/ploidy bounds checks after carriers were derived directly from `G`;
* input-array shape checks in internal helpers;
* extensive worker-result consistency checks.

Do not defensively verify invariants immediately after the code that establishes those invariants.

## Configuration

Keep configuration validation minimal.

It is reasonable to retain only checks that prevent the code from having an ambiguous meaning, such as:

* `window_sizes` being non-empty and increasing;
* `num_doubletons` being positive.

Do not build a large validation layer around optimiser constants or internal parameters.

## Fitting

Keep only numerical handling that is part of the actual optimisation algorithm, such as the existing objective penalty for invalid transformed parameter values.

Do not confuse numerical optimisation guards with API/schema validation: the former are part of the method and should remain.

---

# 5. Assume the intended simple genotype representation

The workflow assumes:

* complete data;
* fixed ploidy;
* phased haplotypes;
* biallelic 0/1 genotype coding;
* one represented genomic sequence;
* positions suitable for `searchsorted`;
* duplicate positions absent.

Do not add code attempting to generalise beyond these assumptions.

In particular, do not maintain machinery for:

* missing genotypes;
* multiallelic focal variants;
* variable ploidy;
* cross-store sample identity;
* multiple contigs.

The validation dataset already has the appropriate representation.

---

# 6. Replace chunked doubleton discovery with ordinary NumPy

Delete the current:

* `_scan_doubletons`;
* chunked `"sample"` operation;
* reservoir sampling;
* multiprocessing sampling code.

Load the genotype array:

```python
G = np.asarray(group["call_genotype"][:])
positions = np.asarray(group["variant_position"][:], dtype=float)
```

Calculate allele counts directly:

```python
ac = G.sum(axis=(1, 2))
```

Find observed doubletons:

```python
dbtn_sites = np.flatnonzero(ac == 2)
```

No custom chunk code is required for this step.

---

# 7. Require complete max-L windows

Calculate:

```python
max_L = config.window_sizes[-1]

dbtn_pos = positions[dbtn_sites]

keep = (
    (dbtn_pos - max_L >= positions[0])
    & (dbtn_pos + max_L <= positions[-1])
)

dbtn_sites = dbtn_sites[keep]
dbtn_pos = dbtn_pos[keep]
```

Therefore every selected doubleton is at least `max_L` from both represented sequence boundaries.

There must be no shortened edge windows.

Every requested smaller window is consequently complete as well.

Record:

```python
num_eligible = len(dbtn_sites)
```

If no doubletons remain, allow the estimator to raise a simple clear error rather than trying to continue.

---

# 8. Sample doubletons with NumPy

Use:

```python
rng = np.random.default_rng(config.random_seed)

n = min(config.num_doubletons, len(dbtn_sites))

if n < len(dbtn_sites):
    selected = rng.choice(
        len(dbtn_sites),
        size=n,
        replace=False,
    )
    dbtn_sites = dbtn_sites[selected]
    dbtn_pos = dbtn_pos[selected]
```

Then restore genomic order:

```python
order = np.argsort(dbtn_sites)

dbtn_sites = dbtn_sites[order]
dbtn_pos = dbtn_pos[order]
```

This is the complete sampling implementation.

Do not use reservoir sampling.

---

# 9. Identify carriers directly

For the selected sites:

```python
dbtn_G = G[dbtn_sites]

_, samples_long, ploidies_long = np.nonzero(dbtn_G)

dbtn_samples = samples_long.reshape(n, 2)
dbtn_ploidies = ploidies_long.reshape(n, 2)
```

Because sites were selected using:

```python
ac == 2
```

under the expected 0/1 genotype representation, each selected site has exactly two carrier haplotypes.

Do not add another explicit validation pass to prove this.

`Doubletons` only needs:

```text
site_indices
positions
sample_indices
ploidy_indices
num_eligible
```

Remove:

```text
sample_ids
source_ploidy
```

---

# 10. Calculate global diversity from the same allele-count vector

Do not perform another genotype pass for global diversity.

For:

```python
H = G.shape[1] * G.shape[2]
```

calculate:

```python
site_diversity = (
    2 * ac * (H - ac)
    / (H * (H - 1))
)

global_difference_sum = site_diversity.sum()

span_bp = positions[-1] - positions[0]
```

Then:

```python
global_pi_per_site = (
    global_difference_sum / len(positions)
)

global_pi_per_bp = (
    global_difference_sum / span_bp
)
```

This provides the scalar `pi` needed by the aggregate model.

There is no separate `compute_diversity()` scan.

A minimal result is sufficient:

```python
@dataclasses.dataclass
class Diversity:
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
```

Once:

* diversity;
* doubletons;
* carrier locations

have been obtained, release:

```python
del G
del ac
```

before starting the chunked mismatch phase.

---

# 11. Prepare physical windows simply

For each one-sided `L`:

```python
left_sites = np.searchsorted(
    positions,
    doubleton_positions[None, :] - window_sizes[:, None],
    side="left",
)

right_sites = np.searchsorted(
    positions,
    doubleton_positions[None, :] + window_sizes[:, None],
    side="right",
)
```

These have shape:

```text
(W, D)
```

The last row defines the maximum region ever needed for each doubleton:

```python
max_left = left_sites[-1]
max_right = right_sites[-1]
```

A carrier pair must never be compared outside:

```text
[max_left[d], max_right[d])
```

---

# 12. Calculate one recombination rate per full window

Remove all left/right rate state.

Continue to compute cumulative recombination distance for each stored site.

For each `(L, doubleton)`:

```python
window_distance = (
    cumulative[right_sites - 1]
    - cumulative[left_sites]
)
```

Then:

```python
rates = window_distance / (
    2 * config.window_sizes[:, None]
)
```

This gives:

```text
rates.shape == (W, D)
```

and corresponds to historical aggregate `r_window`.

No side-specific recombination arrays are needed.

Keep recombination-map handling simple. Do not add redundant coverage/finite checks beyond what is required for the intended HapMap input; allow `msprime`/NumPy failures to surface naturally where appropriate.

---

# 13. Make `MismatchSummary` minimal

Use:

```python
@dataclasses.dataclass
class MismatchSummary:
    window_sizes: np.ndarray
    counts: np.ndarray
    rates: np.ndarray

    @property
    def observed_means(self):
        return np.mean(self.counts, axis=1)

    @property
    def mean_rates(self):
        return np.mean(self.rates, axis=1)
```

Both:

```text
counts
rates
```

have shape:

```text
(W, D)
```

Remove all left/right summary fields and properties.

---

# 14. Restrict mismatch work to max-L

This is the main scaling fix.

For each doubleton `d`, mismatch comparisons are permitted only over:

```text
[max_left[d], max_right[d])
```

A doubleton must never be compared against sites elsewhere on the chromosome.

This should emulate the essential scaling behaviour of the older `mismatch.py` implementation:

```python
region = slice(left, right)
local_haps = G[region, samples, ploidies]
```

while calculating all requested `L` values from the single largest region.

---

# 15. Keep chunk scheduling very small

Only the mismatch phase needs custom chunk logic.

For genotype chunk:

```text
[chunk_start, chunk_stop)
```

a doubleton is active iff:

```text
max_right[d] > chunk_start
and
max_left[d] < chunk_stop
```

Because the sampled doubletons are sorted, the active doubletons form a contiguous range.

Find that range with `searchsorted`; do not build a separate scheduling framework.

Conceptually:

```python
first = np.searchsorted(
    max_right,
    chunk_start,
    side="right",
)

last = np.searchsorted(
    max_left,
    chunk_stop,
    side="left",
)
```

Use boundary choices that exactly implement:

```text
max_right > chunk_start
max_left < chunk_stop
```

If:

```python
first == last
```

skip the chunk entirely without loading its genotype data.

Do not build:

* chunk × doubleton masks;
* persistent per-chunk pair lists;
* scheduling dataclasses;
* a generic task graph.

---

# 16. Vectorise each active chunk

Flatten a loaded genotype chunk:

```python
block = np.asarray(
    genotype[chunk_start:chunk_stop]
).reshape(chunk_length, -1)
```

Precompute flattened carrier addresses once:

```python
carrier_0 = (
    sample_indices[:, 0] * ploidy
    + ploidy_indices[:, 0]
)

carrier_1 = (
    sample_indices[:, 1] * ploidy
    + ploidy_indices[:, 1]
)
```

For active pairs only:

```python
mismatch = (
    block[:, carrier_0[first:last]]
    != block[:, carrier_1[first:last]]
)
```

This creates a temporary:

```text
chunk_sites × active_doubletons
```

array.

It must never create:

```text
all_sites × all_doubletons
```

Use a cumulative sum along the site dimension and accumulate each requested full-window count.

For a particular `L`, count:

```text
[left_site, focal_site)
+
(focal_site, right_site)
```

or equivalently the full `[left_site, right_site)` range with the focal contribution removed explicitly.

Do not calculate separate left/right outputs.

---

# 17. Prefer NumPy; remove Numba unless demonstrably necessary

Implement the active-chunk comparison with ordinary NumPy first.

The operations are naturally vectorised:

* carrier-column selection;
* inequality comparison;
* cumulative sum;
* indexed subtraction.

Remove Numba-specific kernels that become unnecessary.

Do not preserve Numba merely to minimise the diff.

Only retain or reintroduce a specialised kernel if actual profiling on the intended production workflow shows a meaningful need.

If Numba is no longer used anywhere after simplification, remove it from project dependencies.

---

# 18. Multiprocessing should be mismatch-specific only

If multiprocessing remains useful, keep only a dedicated mismatch worker.

Remove generic dispatch such as:

```text
operation == "sample"
operation == "reduce"
```

A mismatch worker should:

1. open the Zarr store once;
2. receive chunk indices;
3. find the active doubleton slice;
4. skip inactive chunks;
5. read the genotype chunk;
6. calculate mismatch contributions;
7. return them.

Keep a simple direct serial path when:

```python
num_workers == 1
```

No multiprocessing is needed for:

* allele counting;
* doubleton sampling;
* carrier extraction;
* global diversity;
* recombination calculation;
* fitting.

Do not over-engineer worker-health/error-handling beyond what is necessary for this controlled analysis workflow.

---

# 19. Simplify fitting to one aggregate model

`fit_error_model()` should be:

```python
fit_error_model(
    summary,
    *,
    pi,
    config,
)
```

`pi` is one scalar global diversity per bp.

Calculate:

```python
observed = summary.observed_means
mean_r = summary.mean_rates
```

Retain the historical aggregate expectation:

```text
fitted =
    2 * H(mu, sigma_sq, pi, L, mean_r)
    + 4 * epsilon * L
```

Do not change:

* gamma-time expectation;
* transformed parameterisation;
* `sigma_sq > mu²`;
* L-BFGS-B;
* bounds;
* starting grid;
* objective scaling;
* numerical penalty behaviour.

Remove all `per_doubleton` branching.

Reduce `_validate_fit_inputs()` substantially or remove it if its checks only duplicate assumptions already established by the implementation.

The optimiser's own numerical guards should remain because they are part of fitting, not defensive schema checking.

---

# 20. Simplify result objects

`ErrorRateFit` needs only:

```text
epsilon
mu
sigma_sq
objective
success
message
observed_means
fitted_means
pi
```

Remove:

```text
fit_mode
```

`ErrorRateEstimate` can continue to group:

```text
doubletons
diversity
mismatches
fit
inference_path
config
```

Do not add provenance fields unless they are directly useful to the analysis.

---

# 21. Simplify the top-level API

The main interface becomes:

```python
estimate_error_rate(
    zarr_path,
    *,
    recombination,
    config,
    pi=None,
) -> ErrorRateEstimate
```

If:

```python
pi is None
```

use:

```python
diversity.global_pi_per_bp
```

Otherwise use the supplied scalar override.

There is no:

```text
diversity_mode
per_doubleton
diversity_zarr_path
```

A normal production call is:

```python
config = error_estimation.EstimationConfig(
    window_sizes=[
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
        250_000,
    ],
    num_doubletons=10_000,
    random_seed=42,
)

estimate = error_estimation.estimate_error_rate(
    zarr_path,
    recombination=recombination_map,
    config=config,
)
```

---

# 22. Remove obsolete machinery rather than leaving it dormant

Expected removals include most or all of:

```text
_ExecutionConfig
_ChunkResult
_ReductionTask
_ReductionResult
_scan_doubletons
_chunk_task
generic _execute_chunks operation dispatch
_accumulate_diversity
_reduce_chunk
_reduce_chunks
_resolve_carriers
compute_diversity
_validate_fit_inputs
```

Keep only the smallest mismatch-specific multiprocessing helpers actually required.

Do not retain unused generalized code for hypothetical future use.

---

# 23. Temporary internal validation only

Do not add regression testing against the current implementation.

Do not add a new notebook.

While implementing, use only:

```text
~/work/tsinfer-anc-eval/data/anc_eval/zarr_vcfs/
OutOfAfrica_4J17-chr20-L0-R1e6-n300-s1-rep0-
geno-1.0-phase0.0-mispol0.0.zarr/
```

Because it is small, direct whole-array calculations are acceptable as temporary checks.

Useful temporary checks are:

### Doubletons

Verify the selected focal sites correspond to:

```python
ac == 2
```

and satisfy complete `max_L` windows.

### Mismatch counts

For each selected carrier pair, temporarily calculate:

```python
pair_mm = (
    G[:, carrier_0]
    != G[:, carrier_1]
)
```

and compare each production count against:

```python
pair_mm[left:focal].sum()
+ pair_mm[focal + 1:right].sum()
```

This whole-array reference exists only during development and should not be retained in production code.

### Diversity

Compare the stored global diversity result with the direct formula based on `ac`.

### Recombination

Compare the aggregate window rate against direct cumulative-map differences.

### Fit

Check that the resulting fit is finite and that the optimiser succeeds.

These checks may use ordinary assertions in temporary development code.

Do not build generalized validation helpers around them.

---

# 24. README after simplification

The README should contain only what is useful for running the estimator:

* `uv sync`;
* basic input assumptions;
* a minimal usage example;
* interpretation of `epsilon`.

For example:

```python
import error_estimation

config = error_estimation.EstimationConfig(
    window_sizes=[1_000, 5_000, 10_000, 50_000, 100_000, 250_000],
    num_doubletons=10_000,
    random_seed=42,
)

result = error_estimation.estimate_error_rate(
    "data.zarr",
    recombination="genetic_map.txt",
    config=config,
)

print(result.fit.epsilon)
```

Document:

```text
epsilon = additive errors per haplotype-bp
```

Do not describe it as a per-genotype probability.

---

# 25. Intended complexity

The estimator should contain no `O(SD)` whole-chromosome carrier-pair operation.

Initial NumPy work:

```text
allele counts / global diversity:
O(SH)
```

Mismatch work:

```text
O(sum of sites lying within each sampled doubleton's ±max_L interval)
```

approximately:

```text
O(D × S_L)
```

where `S_L` is the number of stored sites within a `2 × max_L` region.

Chunk boundaries cause only local excess work at window edges.

There is no:

```text
D × S mismatch matrix
```

and no comparison of a carrier pair to genomic regions outside its complete max-L window.

---

# 26. Definition of done

The refactor is complete when:

1. `per_doubleton` has been removed everywhere.
2. Pairwise diversity has been removed.
3. Left/right mismatch and recombination outputs have been removed.
4. Each `(L, doubleton)` has one total mismatch count and one aggregate recombination rate.
5. Doubleton discovery uses `G.sum(axis=(1, 2))`, NumPy filtering and `rng.choice`.
6. Only doubletons with a complete `±max_L` window are eligible.
7. Global diversity is derived directly from the same allele-count vector.
8. Sampling and diversity use no custom chunking or multiprocessing.
9. Mismatch calculation is the only substantial chunked operation.
10. Every carrier pair is compared only within chunks intersecting its `±max_L` interval.
11. No `D × S` mismatch array is created.
12. The generic chunk/reduction framework has been removed.
13. Defensive validation has been reduced to the minimum; `_open_store()` checks duplicate positions and otherwise trusts the expected store schema.
14. Internal helpers do not repeatedly validate invariants created by upstream code.
15. The existing implementation-testing notebook and `.ipynb` are deleted.
16. No replacement smoke-test notebook or fixture/subsetting machinery is added.
17. Development evaluation uses only the complete specified 1 Mb / 300-sample chr20 dataset.
18. The aggregate fitting equations and optimiser remain unchanged.
19. The resulting module is materially shorter and structurally easy to compare directly with the corresponding `tsinfer-paper` implementation.

