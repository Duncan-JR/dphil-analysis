# Refactor validation evidence

## Phase 3

- All chapter helpers import under the installed direct packages. Hatch installs
  `ch3`, `ch4`, `ch5`; original top-level chapter modules are removed.
- Compilation and Ruff undefined-reference checks pass across all chapter
  modules and notebook sources. A one-time AST/`inspect.signature.bind` audit
  of explicit helper calls reports zero issues; this is migration validation,
  not a retained automated architecture or rough-code test suite.
- The first five CH3 code cells ran with figure output redirected to
  `/private/tmp/dphil_refactor_validation/ch3/`: extension replay, expected
  genotypes, non-unary trees and both illustrated haplotypes verified. Matrix
  PNG and tree PDF/SVG rendered. No original scientific figure was replaced.
- The first eight perf5/pt13 code cells ran with caches and exports redirected
  to `/private/tmp/dphil_refactor_validation/ch5/`: archived cohort/quality
  checks, wall/throughput, memory, speedup/parallelism, sequential wall stages
  and worker stages. Speedup/parallelism and worker-stage PNGs inspected;
  labels, whole-cohort lines, stage definitions and units remain correct.
- All 19 saved notebook output/attachment/execution-count hashes match the
  phase 1 inventory. Cell counts and types are unchanged. Historical outputs
  remain historical; the selected validation cells did not replace them.
- Current dependency code is pinned to tsinfer
  `e2f8de7c950179215ece27e1d31c7b096308b4c8` and tsls
  `c560b405761ae09625e266d39c2356ccc4b54b69`. `uv sync` resolved the missing
  tsls dependency. Python is 3.12.13. The committed lockfile records installed
  packages. Dependency backup/provenance is completed with the thesis workflow.

The older pt1 source referenced presentation keys no longer present in the
current diagnostics YAML. Its missing geometry/colour/label choices were
made explicit in its plot calls; no scientific classification was changed.

## Phase 4

- `experiments/` is absent. All 13,895 experiment scientific files (including
  copied Zarr chunks, result tables, JSONL paths and population metadata) match
  their inventory sizes/SHA-256 hashes after relocation. All 22 moved data
  units match file counts and byte totals; see `refactor_data_moves.json`.
- Four data symlinks were recorded and still resolve. Shared stores/maps and
  pedigree metadata are under `data/general/`; chapter 4 validation outputs
  and pig maps are under `data/ch4/`. The before/after tree files were identified
  from their tsinfer provenance as chr20p ancestor-grouping states (groups
  83/84) and retained in `data/ch5/ancestor_grouping/`.
- Both exporters remain independent of chapter imports and accept CLI options.
  All four scripts' `--help` succeeds. Driver numerical work and multiprocessing
  workers moved to CH5 preparation; script bodies retain CLI/orchestration.
- Native matching jobs were not launched. The historical drivers require
  `tsinfer.matching.MatcherIndexes`/`AncestorMatcher`, absent from the current
  pinned tsinfer revision; this pre-existing upstream API/environment gap
  prevents claiming a matching rerun. Saved inputs/results are retained.
- The retained pt3 upstream grid YAML is unchanged and selected with an explicit
  original relative path base. No notebook-parameter YAML collection remains.

## Phase 5

All 19 notebooks passed their recorded selected-cell checks. The one-time
validation used spawn-process workers and queues across available CPUs;
CSV/figure writes were isolated to a temporary validation tree. Each worker
restored its plotting/write hooks before taking another notebook. Initialisation
and explicit source paths were audited independently (all selected paths
resolve). Retained chapter 4 fitted results were rendered separately, without
claiming a refit. Three contact sheets covering all figure-producing notebook
checks were inspected for axes, labels, legends and layout. The HTML-only pt10
review and its evidence tables were read directly. See the per-notebook JSON
and chapter README tables for exact code cells and scientific limitations.

Actual-data checks identified and resolved remaining source-contract issues:
CH3 nested presentation-dataclass copies became keyword arguments; the shared
minimum-size switch plot takes an explicit Full HMM partition; pt1 uses its
original retained-piece JSONL archive; and pt6 initialises the historical grid
from retained run-summary/raw-file identities rather than the changed sibling
configuration. No scientific methods were replaced to accommodate tests.

Every notebook retains its original output/attachment/execution-count hash.
The saved visible outputs remain from historical runs; separate migrated-cell
checks do not alter that provenance. No rough-work automated tests were added.

## Phase 6

The root Snakemake workflow generated all 23 final CH3 figures, TeX values and
claim/check JSONs first under `/private/tmp/dphil_thesis_build` with fresh
prepared tables under `/private/tmp/dphil_thesis_prepared`, then at the default
`../dphil-thesis` destination. All six preparation jobs have declared inputs and
outputs. Final modules have no exploratory imports. Zarr's consumed metadata
and chunks are individually declared; large genotype arrays are not read.

Nine independently retained numerical check tables match within 1e-9 relative
and absolute tolerance, including focal counts, rare tails, span curves, pooled
overlap errors, genotype errors/transitions, combined overlaps, demographic
quartiles and paired total performance. Counts/denominators and units are
preserved. Six side-by-side contact sheets covered every final figure; the tree
PDF was rendered separately. All 22 PNGs remain byte-identical to the tracked
thesis exports. Only the PDF container changed. Visual inspection caught and
corrected an initially selected exploratory high-frequency subset in the
correctly polarised mispolarisation panel before final generation.

The statistics explicitly distinguish relative percentages, percentage points,
per-site and per-bp denominators, sides versus focal sites, and haploid versus
diploid sample counts. The [claim review](thesis_claims.md) records numerical
rounding corrections. Upper bounds round outwards; the quoted severe histogram
claim uses severe bins, rather than every negative bin. The phase-summary
interface now retains SER metadata in both final and exploratory helpers;
its actual-data notebook-style call was separately rendered successfully.

Input-change validation altered only a temporary copied performance CSV. The
statistics rule reran on the changed input and the largest paired sample speedup
changed from 9.7695794382 to 8.8814358529 as predicted by multiplying the new
elapsed time by 1.1. The real input stayed unchanged; original configuration and
outputs were rebuilt. Config parameter changes and actual maintained helper
edits trigger regeneration. Both the temporary and default completed workflows
report nothing to do on unchanged follow-up dry runs. A fresh build with a
missing aggregate fails at `ch3_prepare_comparisons`, naming the missing CSV.

Mechanical macro substitutions in the sibling thesis are reversible back to
its exact original source; no prose was rewritten. All generated macros compile
in a standalone TeX smoke check. Thesis commit `691af19` contains that integration
and generated statistics. The input manifest records file/chunk SHA-256 values,
producer commands, known revisions, historical provenance limits and retention
locations. Pinned dependency Git bundles are committed for recovery, including
the unpublished tsls commit. This reproduces maintained analyses from retained
inputs; it does not claim to have rerun upstream inference or to have recovered
unrecorded historical producer invocations.

## Phase 7

Pytest discovery is restricted to `tests/`. Collection contains exactly six
class-based tests with pytest fixtures: four CH3 final calculation cases for
pooled inference-site denominators/error directions, pooled sides/strict tail
thresholds, relative histogram counts with zero-baseline gaps, and paired
stage/total speedup aggregation; two retained CH4 checks for masked/focal
exclusion across Zarr v3 chunks and tied-score ROC/precision-recall metrics.
All six pass locally in under two seconds. Inputs are tiny synthetic arrays
and frames; there are no research-data, sibling-checkout or HPC requirements.
No rough helper, notebook, plotting-artist/style, chapter 5 or old-suite tests
were added or retained. Production methods were not complicated to fit tests.

## Phase 8

The final acceptance audit passed on 13 November 2026. The permanent root
ownership/maintenance guidance matches the plan's required content verbatim.
The source tree has exactly the fourteen specified chapter files and five flat
drivers. All active notebook and script consumers were checked against the
consolidated helper signatures, with zero mismatches. Twelve unused rough
helpers were removed. Shared plotting remains specific to each geometry;
benchmark panels return a dataclass, and notebook DPI, spine, colour and label
choices are explicit plot-call arguments. Updated plotting calls were exercised
on retained data and inspected against the earlier figures.

All nineteen notebook pairs retain their original visible-record hashes and
cell counts. Their selected-cell checks pass in separate temporary destinations;
this does not relabel historical saved outputs as new runs. Eighteen ignored
autosave source/notebook records were also preserved byte-for-byte in three
chapter-owned recovery ZIPs. Their member hashes and the final artifact outcomes
are recorded in `refactor_checkpoint_preservation.json` and
`refactor_final_retirements.json`. Historical console/config/run logs moved from
`tmp/` into chapter 3 provenance; superseded notebook/thesis-writing drivers were
retired after protecting their sources externally. Neither `tmp/` nor an active
checkpoint directory remains.

The final data audit rechecked all 13,895 moved scientific files by SHA-256 and
all five external symlinks. Simulated error truth, consumed by both chapters 4
and 5, now belongs under `data/general/`; both affected notebooks passed their
selected checks after the move. The only data root owners are `ch3`, `ch4`,
`ch5` and `general`.

The tie-breaking notebook's optional native rerun now invokes the fifth flat
driver, `compare_tie_breaking.py`, rather than archived research source. Its
preparation, workers and summary calculations belong to CH5 `nb_utils`.
Regenerating summaries from retained paths and references reproduced all four
saved result tables. All five drivers pass import/help checks. The pinned
tsinfer lacks the historical `MatcherIndexes`/`AncestorMatcher` API and has a
different C build layout; native reruns fail clearly before compilation. The
original native matching experiments were not launched or claimed reproduced.

Installation no longer requires a machine-specific sibling tsls checkout. A
committed source distribution contains the same pinned commit/version and
149 verified Python/C/header files, including its pinned tskit source. A
same-version UV override replaces the transitive absolute Git URL; dependency
versions are unchanged. The retained Git bundles contain the pinned histories
without unrelated branches/stashes. A fresh locked installation containing only
source, tests, packaging files and that archive succeeded after downloading the
locked package wheels; all six tests then passed without scientific data or
sibling checkouts. The local locked offline sync and all six tests also pass.

The final default thesis build regenerated all 23 figures and 60 TeX values.
Nine retained numerical tables match again at 1e-9 tolerance. Two diagnostic
tables now have stable row ordering; forcing the statistics target a second time
produced an identical `ch3_checks.json`. No calculated values changed. An
unchanged final dry run reports nothing to do. The relative dependency archive,
lockfile and project metadata are explicit rule inputs. All 22 PNGs remain
byte-identical to the earlier thesis exports; the tree PDF container timestamp
changes on regeneration. Final generated artifacts are committed in the sibling
thesis as `4d3b1d5`, following the mechanical integration in `691af19`.

Final collection contains exactly the six permitted class-based cases; all pass
(0.78 seconds locally, 7.38 seconds in the fresh installation). Source
compilation, undefined-name checks, signature checks, output preservation,
provenance hashes, final layout and working documentation commands were checked.
`refactor_final_audit.json` records the acceptance results. There is no required
CH3 build blocker. Historical producer-invocation gaps and unavailable native
rerun environments remain documented limitations, not unfinished migration.

## Follow-up simplification, 13 November 2026

Read the tsinfer-match-eval and tsinfer-anc-eval workflows before simplifying
the maintained rules. The thesis config is now 77 lines, down from 1,024. It
contains directories, scientific settings, figure filenames and only three
non-default argument blocks for shared renderers. The fixed eight-site
extension example, its trees, unit conversions and numerical formatting are
code. Plot defaults reproduce the finished outputs. The Snakefile has only
three directory variables before its rules; raw paths and source dependencies
are declared in rule inputs, and the configured figure names expand to
individual output files without a second figure list.

The full workflow passed first in an isolated temporary destination, then at
the default destinations. Statistics are now `statistics/ch3.tex`,
`statistics/ch3_claims.json` and `statistics/ch3_checks.json` in dphil-analysis.
The sibling thesis has only a mechanical change to its generated-TeX input
path; no prose was changed. All 22 PNGs match the earlier finished exports
byte-for-byte. Rendering the old and new tree PDFs produces identical pixels.
All 60 macro names, values, displayed text and units match the previous build;
every diagnostic numerical table matches, as do all nine retained reference
tables at 1e-9 tolerance. All 60 macros compile in a standalone TeX check.

A dry run with a renamed filename in a copied config declares the renamed
figure output without a code change. A copied rule pointing at a missing
aggregate fails with MissingInputException naming the precise missing path.
The maintained six tests pass in 0.78 seconds; compilation and undefined-name
checks pass. The histogram fixture now supplies bp input values directly
instead of overriding a conversion factor. No new tests were added.

All nineteen notebook pairs were inspected and protected externally before
the explicitly authorised Jupytext sync. Only three paired text files needed
round-trip whitespace reconciliation. Every original output, attachment and
execution-count hash is unchanged. Jupytext reports no differences for any
pair, and its actual contents manager opens all nineteen output-bearing
notebooks successfully. Notebooks were not rerun. The reappeared experiments
README duplicated chapter 5's scientific documentation and obsolete commands;
its original bytes are protected externally, its useful content is already in
`notebooks/ch5/README.md`, and the experiments tree is absent.

All follow-up source, documentation, notebook and generated-statistics changes
are left uncommitted, as requested.
