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
