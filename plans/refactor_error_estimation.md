# Plan: estimate error rates directly from genotype Zarr data

## Deliverable and repository boundary

Implement a small importable package in `~/work/dphil-analysis`, with the estimator
at `src/dphil_analysis/error_estimation.py`. All implementation, dependencies,
notebooks, results and commits belong to that repository. Follow its `AGENTS.md`
and the placement advice in
`~/work/tsinfer-paper/plans/dphil_analysis_directory.md`. Do not modify
`tsinfer-paper`, move its `error_analysis.py` notebook, or migrate its environment.
Source files there are read-only references for extracting the existing method.

This document is an implementation plan. Writing it does not execute the phases
or authorise changes to DPhil project/task records. Preserve unrelated local work
and data, including existing symlinks. Commit only files belonging to each phase;
do not use an indiscriminate `git add .`. No Co-authored-By commit trailers.

The application is a Python package with thin notebook clients; no CLI, service,
plotting framework or extra pipeline is needed. Its inputs are genotype Zarr
arrays, physical window lengths and an explicit recombination rate/map. No tree
sequence, simulation provenance or error truth enters estimation. The core must
not import tsinfer-paper, tsinfer, sgkit, pandas or plotting libraries.

```text
dphil-analysis/
├── pyproject.toml
├── uv.lock
├── src/dphil_analysis/
│   ├── __init__.py
│   └── error_estimation.py
├── notebooks/ch4_error_estimation/
│   ├── implementation_testing.py
│   └── implementation_testing.ipynb
├── plans/refactor_error_estimation.md
├── data/                       # existing local data and symlinks; ignored
└── results/                    # generated outputs; ignored
```

Use a fresh package environment: runtime dependencies `numpy>=2`, `numba`,
`scipy`, `zarr>=3` and `msprime` (for the existing HapMap reader). Development
notebook dependencies are `jupytext`, `ipykernel`, `nbconvert` and `nbformat`.
Use an ordinary build backend with a `src` package layout; `uv sync` must install
it so notebooks can import it without `sys.path` manipulation. Commit the lock
file. Do not create an environment inside another repository or add sgkit.

## Source evidence and implementation inputs

Read-only sources inspected in `~/work/tsinfer-paper`:

- `lib/modeling.py`: `FitErrorRate` (line 1866), doubleton selection (1945),
  mismatch kernels (1662, 1679), `mean_differences` (901), `expected_mm` (1751),
  error terms (1757, 1764) and `_fit_ls_parameters` (2144).
- `lib/mismatch.py`: `_get_subset_doubletons` (61) and recombination helpers
  (118–156). Extract computations without copying imports or path guessing.
- `jupytext/error_analysis.py`: physical windows, default aggregate fitting,
  HapMap input and per-haplotype-bp epsilon. Leave this notebook in place.

Also inspected `~/work/tsinfer/CLAUDE.md`, `tsinfer/config.py` and
`tsinfer/vcz.py`: dataclass configuration, explicit functions, read-only store
opening and chunk-aligned genotype loading. Emulate these patterns without
importing tsinfer classes. Recheck the named functions at implementation time
if source line numbers have moved; do not use other notebooks as method sources.

Required source datasets for implementation validation:

| Dataset | Input path | Genotype shape | Physical chunk shape |
| --- | --- | --- | --- |
| Simulated chr17, error multiplier 1.0 | `~/work/tsinfer-anc-eval/data/error_eval/zarr_vcfs/OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0-geno-1.0-phase0.0-mispol0.0.zarr/` | 373894 × 1500 × 2 | 1000 × 1500 × 2 |
| 1000 Genomes Project (tgp), chr20, GRCh38 | `~/work/dphil-analysis/data/zarr_vcfs/tgp/chr20/data.zarr` | 1644384 × 3202 × 2 | 20000 × 3202 × 2 |

The directories and JSON metadata were inspected on 2026-09-21; genotype
completeness has not been established. Both use disk format Zarr 2. Read these
with the Zarr 3 library and its supported APIs. Treat the source stores as
read-only. The notebook smoke tests create small temporary Zarr stores from
deterministic masks of these inputs, as specified below; they do not run the
estimator over either complete chromosome.
`dphil-analysis/data/zarr_vcfs/sim` is a symlink to the simulation directory above.
Use that local alias in notebook code and show its resolved path as provenance.
Because sample subsetting changes allele counts, focal doubletons in a fixture
are doubletons within its 200 selected samples. Parameters fitted to these
fixtures are software smoke-test outputs only and must not be reported as
estimates for either complete source cohort.

Local GRCh38 HapMap files for chromosomes 17 and 20 exist at
`~/work/tsinfer-paper/data/HapMapII_GRCh38/genetic_map_Hg38_chr{chrom}.txt`.
Read them as explicit inputs for smoke runs, after checking chromosome, physical
coordinate coverage and finite rates. Record these paths in the notebook; reading
input files there does not modify or introduce an import dependency on that repo.
Do not infer a rate from trees or silently invent one. Scalar-rate support can
be checked with a clearly labelled synthetic constant, separate from data fits.

### Implementation clarification approved on 2026-09-21

The prescribed tgp interval contains repeated positions. With user approval,
`make_masked_store` excludes **all** records at duplicated coordinates (testing
neighbours across interval boundaries too), records the number excluded, and
otherwise preserves the prescribed contiguous 4 Mb interval and sample mask.
The production estimator still rejects duplicates. Fixture site-count assertions
refer to all retained records. This is a smoke-test preprocessing choice, not a
change to the estimator's ascertainment or a claim about the full cohort.
The simulated input labels its contig `1`; the chr17 map is explicitly assigned
from the documented simulation/filename provenance, without rewriting its label.
Map coverage is checked over each fixture's interval, not the unused chromosome
flanks (the tgp source extends beyond the supplied map).

## Scientific contracts and decisions

### Preserve the fitted model and sampling population

The inspected implementation is a gamma-time model with a **linear** additive
error term, not an exponential error-saturation model. Preserve these equations; changing the error model is outside this extraction.

Keep physical windows, exclusion of the focal site from mismatch counts,
full-window eligibility, recombination-distance conventions, objective scaling,
parameter transformation, multistart search and bounds. Remove truth-based
selection: identify focal doubletons and their carriers from the observed Zarr.
That ascertainment difference is necessary and must be documented when comparing
against old fits, which select using `truth_G` even for error-bearing data.

Sample uniformly without replacement from doubletons eligible for every requested
window. `EstimationConfig.num_doubletons` is a positive integer hyperparameter,
defaulting to 10000. Use `min(num_doubletons, num_eligible)` observations. All
required implementation smoke runs use `num_doubletons=100` on temporary
masked stores; smaller values are permitted for focused checks. Reject zero eligible doubletons with a
clear error rather than attempt a fit. Keep singletons in diversity and all
mismatch calculations, as well as other nonfocal sites.

The minimal focal-site definition is a completely called biallelic site with
exactly two copies of allele index 1. This matches the existing 0/1 calculation;
never sum arbitrary allele indices or silently reinterpret a reference allele
as ancestral. A homozygous carrier contributes its two distinct haplotypes.
Multiallelic sites can contribute ordinary allele inequalities to diversity and
window mismatches, but are not focal doubletons. This is alternate-allele
ascertainment on real VCZ data, not a claim to infer derived doubletons without
ancestral information. Changing to folded/minor or ancestrally polarised
ascertainment would be a separate method choice.

Require phased, fixed-ploidy, completely called genotypes for this MVP. Detect
negative allele codes while streaming and fail with a site/chunk location;
missing-vs-called must not become a mismatch, and missing-vs-missing must not
become a match. If phase flags exist, reject unphased heterozygous calls. If
absent, phased haplotype ordering is an explicit input assumption. Supporting
missingness via new denominators/exposures is outside the unchanged method.
Genotype completeness and phase validity of the examples have not been verified.

### Diversity units: report per site, fit per bp

The prompt proposes the number of stored sites as the diversity denominator.
Implement that reported quantity. It is diversity conditional on the sites
represented in the store, not nucleotide diversity across unrecorded invariant
bases. A store with a different site ascertainment scheme can give a different
per-site value for the same biological population.

Let S be the number of stored sites, H the number of haplotypes, and n[s,a]
the count of allele a at site s. For complete calls:

```
d[s] = (H * (H - 1) - sum_a n[s,a] * (n[s,a] - 1)) / (H * (H - 1))
global_difference_sum = sum_s d[s]
global_pi_per_site = global_difference_sum / S
pair_difference_count[k] = sum_s (G[s, carrier0[k]] != G[s, carrier1[k]])
pair_pi_per_site[k] = pair_difference_count[k] / S
```

Thus the global statistic averages over distinct haplotype pairs, without
constructing H² pairs. Pair values include the focal site when it belongs to the
diversity store (its contribution is normally zero); only window mismatch counts
exclude it. Use every stored site and every sample for global diversity.

The current expectation multiplies pi by physical L, so substituting per-site
pi directly would change units and the fitted model. Preserve the model by
also returning per-bp values:

```
span_bp = position[-1] - position[0]
site_density = S / span_bp
pi_per_bp = pi_per_site * site_density
```

Use the **diversity store's** span and site density, even when it is a different
chromosome. This reproduces the old numerator/span convention for complete
matching data. Require at least two sites spanning a positive distance. The
first-to-last-site span is chosen to match `FitErrorRate`, not the full contig
length including unknown or unrepresented flanks. Physical inference windows
likewise remain inside the inference store's first and last positions.

Expose both normalisations explicitly; `fit_error_model(pi=...)` always takes
per-bp pi, whether scalar or an array of length D. Epsilon remains errors per
haplotype-bp in the existing additive approximation. Do not describe it as a
per-genotype-cell probability or introduce a calibration to that probability.

### Carrier identity across stores

Represent carriers by `sample_id` plus ploidy index, with numerical sample indices
cached only for the inference store. Resolve another store by sample ID, never
by sample order. Require unique IDs, all requested carriers present and compatible
ploidy; additional target samples contribute to that store's global diversity.

A same-sample haplotype slot on a different chromosome is a chosen indexing
convention, not evidence of biological homolog correspondence across chromosomes
or independently phased datasets. Document this limitation with the alternative
chromosome option. Preserve the requested two-haplotype statistic; do not silently
replace it with an average over all four diploid cross-comparisons.

## Proposed interface and data structures

Use module imports (`import dataclasses`, `import scipy.optimize`, `import numba`),
PEP 604 annotations, NumPy-style docstrings cross-linking the relevant functions,
one module logger and straightforward control flow. No large mutable fitter
class is necessary. Arrays use D = sampled doubletons and W = window lengths.

| Dataclass | Fields and meaning |
| --- | --- |
| `EstimationConfig` | Required `window_sizes`: positive finite increasing lengths in bp; `num_doubletons`: positive integer, default 10000; `random_seed`: integer or None; `num_workers`: integer or None, resolved to available CPUs in the worker-launching function only. Optimiser bounds, start grids, penalty and queue depth are named configuration fields holding the existing numerical choices. |
| `Doubletons` | `site_indices`, `positions`: (D,); `sample_indices`, `sample_ids`, `ploidy_indices`: (D, 2). `num_eligible`: count before sampling. A `num_doubletons` property derives D from the arrays. All arrays share a single stable row order. |
| `Diversity` | `num_sites`, `num_haplotypes`, `span_bp`, `global_difference_sum`, `pair_difference_counts`: (D,); properties `global_pi_per_site`, `pair_pi_per_site`, `global_pi_per_bp`, `pair_pi_per_bp`. Store source path and carrier identity/order for provenance. |
| `MismatchSummary` | `window_sizes`: (W,); `left_counts`, `right_counts`, `left_rates`, `right_rates`: (W, D). Properties for total counts, window mean rate, and observed means. Recombination rates are per bp per generation. No full sites × doubletons matrix. |
| `ErrorRateFit` | `epsilon`, `mu`, `sigma_sq`, `objective`, `success`, `message`, observed and fitted means (W,), pi used and fit mode. Expose optimiser outcome without falsely treating a finite objective as convergence. |
| `ErrorRateEstimate` | Groups `doubletons`, `diversity`, `mismatches`, `fit`, inference path and resolved seed/configuration for reproducibility. No truth or plotting attributes. |

Public functions:

```python
sample_doubletons(zarr_path, *, config) -> Doubletons
compute_diversity(inference_zarr_path, doubletons, *, zarr_path=None,
                  num_workers=None) -> Diversity
summarise_mismatches(zarr_path, doubletons, *, config,
                    recombination) -> MismatchSummary
fit_error_model(summary, *, pi, config,
                per_doubleton=False) -> ErrorRateFit
estimate_error_rate(zarr_path, *, recombination, config,
                    diversity_zarr_path=None, pi=None,
                    diversity_mode="global", per_doubleton=False) -> ErrorRateEstimate
```

`compute_diversity` resolves its optional path only where it opens the source.
It can be called separately for another chromosome without resampling doubletons.
`recombination` is a nonnegative finite scalar, an `msprime.RateMap`, or a HapMap
path. Resolve a path literally relative to the working directory; no repository
path heuristics. A rate/map is mandatory because simulation metadata is gone.

The convenience function computes both diversity summaries. With `pi=None`, it
selects the computed global or pairwise per-bp value according to
`diversity_mode`. A supplied pi overrides that choice and must be a finite,
nonnegative scalar or (D,) vector aligned to the returned doubleton order;
for externally prepared pair vectors prefer the separate sampling/diversity/
fitting functions, where that order is already known. Reject an array with
`per_doubleton=False` rather than collapse it silently; pairwise mode in the
convenience function selects `per_doubleton=True`. Scalar pi supports either
existing fitting mode. Retain the old aggregate mode as the default.

Private helpers have one responsibility each:

- `_open_store(zarr_path)`: read-only Zarr group, required arrays and dimensional
  checks. Require one represented contig and strictly increasing positions;
  detect unsupported duplicates/multiple contigs rather than guessing alignment.
- `_iter_genotype_chunks(store)`: site offsets and contiguous genotype blocks
  using the physical site chunk size; flatten only in memory. NumPy conversion
  is limited to each block, positions and small identity arrays.
- `_resolve_carriers(store, doubletons)`: target sample indices by ID and ploidy.
- `_scan_doubletons(block, offset, bounds)`: Numba allele counts and two carrier
  locations; output compact candidates, not copies of full genotypes.
- `_accumulate_diversity(block, carriers)`: Numba global allele-count statistic
  and one difference counter per carrier pair.
- `_accumulate_windows(block, offset, boundaries, carriers)`: exact integer
  window counts using the scheme below.
- `_site_recombination_cumsum(positions, recombination)`: adapted existing map
  integration with explicit coverage and finite-rate checks.
- `_power_integral_from_one`, `_expected_haplotype_mismatches`,
  `_expected_error_mismatches`, `_fitted_means`, `_objective`: extracted numerical
  model with deliberate scalar/vector pi broadcasting.

Sampling and numerical fitting settings belong in `EstimationConfig`, keeping
the module self-contained and allowing a simple future move into tsinfer's
configuration layer. `num_doubletons` means the requested maximum; the
`Doubletons.num_doubletons` property reports the realised count.

## Streaming algorithms and performance

### Pass 1: observed doubletons and uniform sampling

Read positions and sample IDs once. Compute maximum requested L and reject focal
sites whose full largest window crosses the first/last stored position. Scan
genotypes in site chunks to identify valid biallelic allele-1 count-two sites.
Record the two (sample, ploidy) addresses, including two slots from one sample.

Use reservoir sampling with capacity K = config.num_doubletons over eligible
sites in genomic order. For eligible site number t (one-based), keep the first K;
subsequently draw an integer uniformly from [0, t) and replace the corresponding
reservoir entry only when it is below K. Use a local NumPy Generator and the configured seed.
Sort selected records by site index at the end, applying the same permutation
to every field. This uses O(D) selection memory and does not bias towards chunks
or common carrier pairs. Do not deduplicate doubletons sharing a carrier pair:
those are separate fitting observations.

Candidate discovery can run in processes for independent site chunks. Return
chunk IDs and consume candidates in genomic order so the seed's result does not
depend on scheduling. Bound outstanding work and reorder buffers.

### Pass 2: one diversity scan, with optional fused window accumulation

For each genotype chunk, count alleles across samples for the global statistic
and loop over the D pairs inside a Numba kernel. Sum integer pair mismatches and
the global floating statistic; divide only after reduction. There is no Python
loop that rereads the store once per doubleton, no H × H matrix, and no retained
D × S mismatch matrix. Multiallelic equality is handled by allele index within
each site; no ancestral polarisation is needed for pair differences.

Compute both diversity quantities in this same pass. For the default inference
store, additionally accumulate window counts from each loaded block. For an
alternative diversity chromosome, make one diversity scan there and one window
scan of the inference store. Sampling necessarily precedes pair diversity: the
promise is one scan **for diversity**, not one total scan including discovery.

Standalone `compute_diversity` and `summarise_mismatches` delegate to a shared
private chunk reducer; the convenience wrapper fuses their accumulation when
both paths identify the same source. Do not introduce a general pipeline framework.

### Window counts without the full mismatch matrix

For each L and doubleton, precompute the existing searchsorted boundaries:

```
left = searchsorted(positions, focal_position - L, side="left")
right = searchsorted(positions, focal_position + L, side="right")
left interval = [left, focal_site)
right interval = [focal_site + 1, right)
```

Within each chunk and pair, build a temporary prefix sum of 0/1 inequalities.
For each intersecting window side, clip its boundaries to the chunk and add
`prefix[stop] - prefix[start]`. This exactly reproduces the previous inclusive
physical endpoints and exclusion of the focal site. Discard the prefix before
advancing to the next pair. Whole-chromosome pair counts can use its final value.
No rescan of the genotype store per window length is required.

Compute left genetic distance from the leftmost included site's map coordinate
to the focal coordinate, and right distance from focal to `right - 1`, matching
`_fit_error_rate_mismatch_summaries`. Empty sides have distance zero. Divide by
physical L to obtain each side's rate. Do not replace these site-endpoint
conventions with exact physical-window-endpoint map integration in this extraction.
For scalar r, use cumulative coordinates `positions * r` and the same endpoints.
For HapMap paths, preserve `position_col=1`, `rate_col=2` and cumulative mass.
Reject undefined map spans; do not extrapolate or substitute a guessed rate.

### Multiprocessing and resource bounds

Use multiprocessing queues for independent chunk work, defaulting to available
CPU cores as required by `AGENTS.md`. Each worker opens a local read-only store
once and receives chunk-index jobs; each assigned chunk is decoded once per
pass. Return compact partial counts and chunk IDs, not full genotype blocks.
The parent reduces results in a defined order and bounds queued work/results.
Propagate worker exceptions and join/terminate workers on failure; a crashed
worker must not leave the parent waiting forever. Use a spawn-safe module worker
function and no nested process pools or Numba parallel kernel inside each worker.
Resolve `num_workers=None` only in this execution helper. Limit active workers
to available chunks, and allow one worker for constrained machines.

If C is the sites per chunk, H haplotypes, D <= config.num_doubletons, W windows, and P active
workers, memory is approximately O(P(CH + WD) + S + WD), including a small
per-worker prefix and bounded candidate/result queues. Pair comparisons cost
O(SD); all-sample allele counts cost O(SH), with O(number_of_chunks × WD)
window-prefix queries. The full tgp store would still require a large genotype
scan even when D is small. It is therefore outside the smoke-test scope. The
normal default can use available CPUs; callers can bound workers for large
chunks. Never materialise a full D × S mismatch array or copy all pair genotypes
into a large temporary. Lowering K reduces pair work but does not eliminate the
whole-store discovery and all-sample diversity scans, which is why the notebook
also subsets sites and samples.

## Fitting algorithm retained from the inspected code

Set alpha = mu² / sigma_sq, beta = mu / sigma_sq and q = 2r. The one-side
haplotype expectation is:

```
H(mu, sigma_sq, pi, L, r) = pi * (L - integral_0^L (1 + q*x/beta)^(-alpha) dx)
```

Keep the existing stable power-integral calculation with `log`/`expm1` and its
logarithmic limit near alpha = 1, zero-r result and nonnegative roundoff clamp.
Broadcast pi to the same shape as L/r **before** masking zero rates; copying
`mean_differences` unchanged would not correctly handle a per-doubleton pi array.

The aggregate mode fits, for each L:

```
observed = mean(left_counts + right_counts)
fitted = 2 * H(mu, sigma_sq, scalar_pi, L, mean(window_rate)) + 4 * epsilon * L
```

The existing per-doubleton mode fits the mean over the 2D sides, not a new
likelihood for individual pair counts:

```
side_rates = concatenate([left_rates, right_rates], axis=1)
side_pi = concatenate([pair_pi, pair_pi])  # or broadcast a scalar
observed = mean(concatenate([left_counts, right_counts]), axis=1)
fitted[L] = mean(H(mu, sigma_sq, side_pi, L, side_rates[L]) + 2 * epsilon * L)
```

Never average pi and r separately in pairwise mode. Their shared indexing is
part of the contract. Require finite input summaries instead of masking away a
rate while retaining its mismatches. Both-side weighting remains uniform because
all selected doubletons have all requested complete windows.

Preserve the existing optimisation:

- Transform `mu = exp(a)`, `sigma_sq = mu**2 + exp(b)`, `epsilon = exp(c)`.
  In particular sigma_sq > mu² (alpha < 1) is an existing constraint; do not
  silently replace it by an arbitrary positive variance.
- Minimise sum of squared residuals divided by `max(abs(observed), 1)` squared.
  Retain nonfinite/overflow penalty behavior (penalty 1e100 and the current
  residual magnitude guard 1e50).
- L-BFGS-B bounds: a,b in [-30,30], c in [-50,log(0.5)].
- Starts: alpha in {0.25,0.5,0.75}, log(beta) in {-12,-9,-6,-3,0},
  log(epsilon) in {-50,-30,-20,-12}; initialise excess variance as in the source.
- Select the lowest objective result and retain its success/message. If every
  trial is invalid/penalised, raise an explicit fitting failure. Do not add
  confidence intervals, a different optimiser or an identifiability correction.

No direct gamma fit to true TMRCAs, truth error counts, no-error comparator,
first-recombination traversal, excess mismatch diagnostics or plotting survives
in the new module. Keep old modules and their notebook consumers untouched.

## Execution phases and planned commits

Execute these four phases in order, in `~/work/dphil-analysis`. Keep each phase
small and commit only after its stated checks pass. During implementation, run
smoke checks with `uv run` as functions become available; record the final
reproducible calls and assertions in the paired notebook source. Do not delay
all validation until the end. Notebook checks are the initial test coverage;
a separate pytest suite is not required for this MVP. The user's requested
notebook calls and direct assertions are the testing format here.

### Phase 1 — package, dependencies and notebook scaffold

Planned commit: `Set up dphil-analysis package and error estimation notebook`

Changes:

- Create `pyproject.toml` with the package/dependencies specified above, a
  `src/dphil_analysis/__init__.py`, and a brief README showing `uv sync` and
  notebook execution from the repository root. Keep `__init__.py` minimal.
- Add `.gitignore` entries for `.venv/`, Python/Numba caches, notebook checkpoints,
  `.DS_Store`, `data/` and `results/`. Preserve every existing data file/symlink.
- Create the Jupytext percent-format `implementation_testing.py` scaffold with
  pairing metadata `formats: ipynb,py:percent` and Python 3 kernel metadata.
  Include setup, simulated-data and tgp headings. The `.py` is the editing source;
  materialise the executed `.ipynb` in phase 4.
- Put imports, paths, seed, windows and smoke configuration in the setup cells.
  Initially just check dependencies and read-only metadata, without calling
  not-yet-implemented functions or leaving deliberately failing placeholders.
- Establish that both stores open, required arrays exist, position arrays are
  ordered and input map coverage is adequate. Inspect phase/missingness encoding
  to implement the actual schema, rather than assuming a flag's name/shape.
- Add a notebook-only `make_masked_store` helper. It takes a source group, an
  output path, a physical interval and sample indices, and writes the minimal
  estimator schema (`call_genotype`, `variant_position`, `sample_id`, contig
  metadata and phase data when present) to a temporary Zarr 3 store. Select the
  site interval with `searchsorted`, and select samples with Zarr orthogonal
  indexing in bounded site blocks. Preserve array dimension metadata. This is a
  materialised test fixture, because Zarr has no general lazy masked-group view;
  do not add masks or subset options to the production estimator API.
- Build one fixture per source under `tempfile.TemporaryDirectory`: choose 200
  sample indices uniformly without replacement with seed 42, sort the indices,
  and take a contiguous 4 Mb interval centred on the source's median variant
  position. Clip or shift the interval to the represented position bounds while
  retaining its width. Assert the resulting span is at least twice the largest
  smoke-test window. Keep the temporary-directory context alive for all notebook
  sections and delete it automatically when execution ends.

Commands from the new repository (include these in the notebook command log):

```sh
uv sync
uv run python -c 'import dphil_analysis, numpy, numba, scipy, zarr, msprime; assert int(zarr.__version__.split(".")[0]) >= 3'
uv run python notebooks/ch4_error_estimation/implementation_testing.py
```

Outcome: the package imports from `src`, Zarr 3 reads both inputs without sgkit,
the notebook creates bounded temporary fixtures from both real schemas, paths
and maps are explicit, and `uv.lock` captures the fresh environment. Source
genotype stores are unchanged. Commit the scaffold, dependencies, lock file,
README, ignore file and plan; leave unrelated existing files alone.

### Phase 2 — dataclasses, input contracts and doubleton sampling

Planned commit: `Add estimator data structures and chunked doubleton sampling`

Changes:

- Add `error_estimation.py` with the six dataclasses and fields described above,
  config validation, read-only metadata/chunk helpers, and carrier ID resolution.
- Implement `sample_doubletons` with observed biallelic allele-1 counts,
  full-window eligibility, reservoir selection and deterministic row ordering.
- Implement queue execution for chunk tasks once, only to the extent needed by
  sampling and later reductions. Keep workers importable and spawn-safe.
- Extend both notebook dataset sections with sampling calls on their temporary
  fixtures using `num_doubletons=100`, seed 42, one worker and the smoke window
  lengths specified below. Record realised and eligible counts.
- Include focused checks on the simulated input: repeat a small seeded sample
  (e.g. K=10), compare all selected sites/carriers, and use a small in-memory
  Zarr 3 store for the fewer-than-K and zero-eligible cases. A tiny optional
  one-versus-two-worker comparison checks dispatch determinism without making
  either full smoke workflow parallel.
  This temporary fixture is notebook-only and contains enough flanking positions
  for the chosen windows. Do not create persistent fake input data.

Run `uv run python notebooks/ch4_error_estimation/implementation_testing.py`.
Direct assertions must establish on both inputs that D equals
`min(100, num_eligible)`, selected sites are unique and ordered, windows fit,
each focal site has exactly two allele-1 copies and carriers match the actual
calls. Group selected validation reads by chunk, not one read per doubleton.
Check positive-integer K validation, default K=10000 and carrier-index bounds.
A no-doubleton or invalid-input case must raise the documented error.

Outcome: dataclass fields have the specified shapes/units; reproducible sampling
works single-threaded on masked data originating from both datasets; the cap is
configurable and source genotype arrays are never loaded in full. Commit the
module and updated notebook source.

### Phase 3 — diversity, window counts and recombination summaries

Planned commit: `Compute streamed diversity and doubleton mismatch summaries`

Changes:

- Implement standalone `compute_diversity`, `summarise_mismatches`, the compiled
  chunk accumulators and a narrowly scoped internal reducer that can compute
  both in one pass. Reuse queue infrastructure, not a second execution framework.
- Add map/scalar integration with the exact endpoint rules stated above.
- Extend both notebook sections with whole-fixture diversity and window-summary
  calls on up to 100 sampled doubletons. Store the returned objects in memory
  for phase 4 fitting; rerunning a fit should not rescan genotypes unnecessarily.
- On the simulated input, check selected pairs/window counts against direct
  inequalities on small slices, and the global numerator against explicit
  distinct-pair comparisons on a tiny hand-constructed genotype array. Include
  a singleton so an accidental singleton filter fails visibly.
- Exercise sample-ID remapping with a small, in-memory copy whose sample axis is
  reordered; compare against the same tiny source before reordering. Check the
  explicit `zarr_path` override with a temporary on-disk Zarr under a temporary
  directory. Verify absent carriers fail instead of being matched by row number.
  These focused fixtures stay in the notebook; no third production dataset is
  needed. Resolve different paths by sample ID and ploidy as specified above.

Run `uv run python notebooks/ch4_error_estimation/implementation_testing.py`.
On each masked input, require finite nonnegative global and pair diversity,
per-site values in [0,1], `(D,)` pair arrays and `(W,D)` counts/rates. Assert the
per-site/per-bp conversion identity, complete fixture site count, window-count
monotonicity across increasing L and exact count agreement on inspected windows.
Check a zero-rate synthetic input, scalar/map equivalence for a constant map,
and same-pair results after remapping. Negative calls/unphased heterozygotes
must give explicit errors on tiny fixtures rather than plausible estimates.

Outcome: every site in each fixture contributes to the intended diversity calculation, pair
ordering is preserved, window counts match direct comparisons and recombination
units/endpoints are correct. Each fixture scans once per call regardless of K.
Record wall time and the single-worker setting for both datasets. Commit module
and notebook.

### Phase 4 — numerical fitting and complete notebook validation

Planned commit: `Fit Zarr error rates and validate simulated and tgp workflows`

Changes:

- Extract numerical expectations and the exact objective/multistart optimiser.
  Extend pi broadcasting as described, retaining the aggregate and per-doubleton
  modes. Implement `estimate_error_rate`, including the fused default-source
  diversity/window pass and alternative-source routing.
- Call `fit_error_model` with global scalar pi and pairwise pi on both masked
  datasets; run the convenience entry point on both with up to 100 doubletons
  and the same seed.
  Compare staged and convenience outputs for equal settings and selected rows.
- On one dataset only, compare scalar pi and a constant vector in the same
  per-doubleton mode. Separately check zero-r expectations, zero-error terms,
  the logarithmic limit and expectations against direct numerical integration
  on small parameter examples. Do not import legacy modules merely for a check:
  their plotting/sgkit dependencies are unnecessary. Tiny direct formula
  calculations in notebook cells are sufficient independent references.
- Verify private kernels/helpers through the public calls and the targeted
  formula/fixture checks; exercise all five public functions on both datasets.
- Complete the `.py` notebook with successful command history, paths, versions,
  parameters, assertions and concise displayed results. Generate and execute the
  paired `.ipynb` using the commands below. Commit both paired files, keeping
  large genotype/count arrays out of cell outputs.

Require finite parameters and objective, `mu > 0`, `sigma_sq > mu**2`,
`epsilon > 0`, finite predicted means and optimiser success. Display objective,
status/message, D and timing for each fit. If a run fails or reaches an
unusable solution, investigate and report the actual cause; do not suppress
assertions, weaken the method or label a failed run as passed. A fitted value on
real data is a functionality result, not proof of calibrated error-rate accuracy.

Outcome: temporary stores masked from both named inputs pass the staged and
integrated workflows with 100 requested doubletons, scalar and pairwise pi are exercised, and the paired
notebook runs from a fresh kernel without hidden state. Commit only after these
checks and notebook synchronisation succeed. No migration of historical code or
notebooks is part of this phase.

## Notebook contents and exact validation workflow

Edit `notebooks/ch4_error_estimation/implementation_testing.py`. Use headings:

1. `Setup and reproducibility`
2. `Simulated data — chr17`
3. `1000 Genomes Project (tgp) — chr20`
4. `Focused numerical and indexing checks`
5. `Execution commands and results`

The two dataset headings must each show the source-to-fixture mask and visible
direct estimator calls, not just one opaque workflow helper invoked twice. The
small `make_masked_store` helper itself is shared because duplicating Zarr-copy
logic would obscure the test. Use short shared assertion helpers only when they
improve readability. Notebook expressions display scalar summaries and small
arrays; never print full genotypes or all pairwise summaries. Keep the Python
source executable with plain Python: avoid magics and shell escapes.

Setup, using `pathlib` imported at the top:

```python
repo = pathlib.Path.home() / "work" / "dphil-analysis"
sim_name = (
    "OutOfAfrica_4J17-chr17-L0-R22.7e6-n1500-s1-rep0-"
    "geno-1.0-phase0.0-mispol0.0.zarr"
)
sim_source_path = repo / "data" / "zarr_vcfs" / "sim" / sim_name
tgp_source_path = repo / "data" / "zarr_vcfs" / "tgp" / "chr20" / "data.zarr"
map_dir = pathlib.Path.home() / "work" / "tsinfer-paper" / "data" / "HapMapII_GRCh38"
sim_map = map_dir / "genetic_map_Hg38_chr17.txt"
tgp_map = map_dir / "genetic_map_Hg38_chr20.txt"
config = error_estimation.EstimationConfig(
    window_sizes=[1000, 5000, 10000, 50000, 100000, 250000],
    num_doubletons=100,
    random_seed=42,
    num_workers=1,
)
```

Within the guarded setup cell, create `TemporaryDirectory`, then materialise
`sim_smoke_path` and `tgp_smoke_path` from their respective source paths. The
selection helper returns a small provenance dataclass or dictionary containing
the source path, source site/sample counts, chosen interval, selected sample IDs,
fixture site/sample counts and output path. Display those fields in each dataset
section. Assert exactly 200 samples and a 4 Mb represented interval apart from
minor endpoint/site-position differences. If a deterministic mask has no eligible
doubletons, move the interval to the densest of a small fixed set of candidate
4 Mb intervals and record the chosen interval; do not increase data size or scan
the whole estimator workflow repeatedly.

Import `from dphil_analysis import error_estimation` normally from the installed
package. Put process-launching script cells under `if __name__ == "__main__":`
so the percent-format source also runs safely on macOS with spawn; keep worker
functions in the installed module, never in notebook cells. This guard is true
in notebook execution and false in spawned script imports. Initialise fixtures
and other executable checks inside guarded cells too; module imports stay at
the top. Avoid nested multiprocessing from notebook helpers.

Simulated-data section, with these calls inside a guarded cell:

```python
sim_doubletons = error_estimation.sample_doubletons(sim_smoke_path, config=config)
sim_diversity = error_estimation.compute_diversity(
    sim_smoke_path, sim _doubletons, num_workers=config.num_workers,
)
sim_summary = error_estimation.summarise_mismatches(
    sim_smoke_path, sim_doubletons, config=config, recombination=sim_map,
)
sim_global_fit = error_estimation.fit_error_model(
    sim_summary, pi=sim_diversity.global_pi_per_bp, config=config,
)
sim_pair_fit = error_estimation.fit_error_model(
    sim_summary, pi=sim_diversity.pair_pi_per_bp,
    config=config, per_doubleton=True,
)
sim_estimate = error_estimation.estimate_error_rate(
    sim_smoke_path, recombination=sim_map, config=config,
)
```

The tgp section must contain the corresponding calls explicitly:

```python
tgp_doubletons = error_estimation.sample_doubletons(tgp_smoke_path, config=config)
tgp_diversity = error_estimation.compute_diversity(
    tgp_smoke_path, tgp_doubletons, num_workers=config.num_workers,
)
tgp_summary = error_estimation.summarise_mismatches(
    tgp_smoke_path, tgp_doubletons, config=config, recombination=tgp_map,
)
tgp_global_fit = error_estimation.fit_error_model(
    tgp_summary, pi=tgp_diversity.global_pi_per_bp, config=config,
)
tgp_pair_fit = error_estimation.fit_error_model(
    tgp_summary, pi=tgp_diversity.pair_pi_per_bp,
    config=config, per_doubleton=True,
)
tgp_estimate = error_estimation.estimate_error_rate(
    tgp_smoke_path, recombination=tgp_map, config=config,
)
```

Use numerical tolerances for floating reductions/fits and exact equality for
selected rows/integer counts. Record actual tolerances and outcomes in cells.
Keep the seed fixed between staged and convenience runs. Exercise integrated
pairwise and alternative-source options on a small fixture rather than repeat
the fixture scans for every option. Do not run either complete chromosome as an
implementation requirement. The default remains 10000 for later analyses; the
smoke notebook explicitly uses at most 100.

During phases 1–3 and fitting development, use the current executable `.py` as
the smoke script, or run focused snippets with `uv run python`. Record any
additional successful focused commands/assertions in the final notebook. The
notebook should preserve the reproducible validation sequence, not failed shell
attempts or transcripts of unrelated investigation.

At the end of phase 4, from `~/work/dphil-analysis`, generate and execute the
notebook using the project's own Python kernel:

```sh
uv run python -m ipykernel install --prefix .venv --name dphil_analysis --display-name "dphil-analysis"
uv run jupytext --to ipynb notebooks/ch4_error_estimation/implementation_testing.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dphil_analysis --ExecutePreprocessor.timeout=-1 notebooks/ch4_error_estimation/implementation_testing.ipynb
uv run jupytext --sync notebooks/ch4_error_estimation/implementation_testing.ipynb
```

Set source kernel metadata to `dphil_analysis` before conversion so the pair stays
consistent. These commands operate only inside the new repository and its `.venv`.
Long validation runs should produce progress/timings and be polled without
blocking user updates. The final fresh-kernel execution is the end-to-end check;
do not rerun it once more without a subsequent code change or unresolved failure.

Completion means the four implementation commits exist in `dphil-analysis`,
the source module and dependencies are self-contained, all required notebook
checks pass on both datasets, and the committed `.py`/`.ipynb` pair records the
commands and observed results. No claim of scientific calibration or edits to
external project/task records follows from these smoke checks.
