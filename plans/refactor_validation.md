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
