# Standalone commands

Run commands from the analysis repository root, after `uv sync`. Drivers use
`ch5.nb_utils` for reusable calculations and spawn-safe workers. No driver
reads a repository notebook-parameter YAML file. Use `--help` for every option.

## Chapter 3 transferable exports

`extract_zarr_validation_data.py` is self-contained (NumPy, pandas, Zarr v3).
Copy it to the data-holding machine; no installed `ch3` package is needed.
From the upstream ancestor-evaluation root:

```sh
uv run python /path/to/extract_zarr_validation_data.py \
  OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1 \
  --data-folder data/anc_eval_new --output-folder data/anc_eval_new/dataframes
```

It streams variant chunks, distributes independent error profiles to CPU
workers, and atomically replaces its completed CSV. Record types retain raw
profile, truth, diploid genotype-transition and DAF counts; local plots choose
their own frequency bins. Transfer the resulting `PREFIX_zarr_data.csv` and
matching `PREFIX-aggregated.csv` to `data/ch3/`.

`extract_phase_data.py` is also self-contained, with the same three packages.
It exports phase segments for the same individuals across switch spacings:

```sh
uv run python /path/to/extract_phase_data.py \
  --data-prefix OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1 \
  --data-folder data/anc_eval_new --output-folder data/anc_eval_new/dataframes \
  --n-per-sim 2 --spacing-labels 2e5 2e6 2e7
```

Transfer `PREFIX_phase_runs.csv` to `data/ch3/`. These exports read existing
stores; they do not simulate data or run inference.

## Chapter 5 zero-coverage matching

This command **starts HMM matching** from completed upstream products. It reads
local copied trees, panel/sample Zarrs, native parameter TOMLs and focal NPZs;
TOML source paths are not followed. Missing historical inputs or native matcher
APIs require the identified original upstream environment, rather than a new
upstream inference run just to verify migration.

```sh
uv run python scripts/chunk_matching.py \
  --input-dir data/ch5/match_eval \
  --coverage-input-dir ../tsinfer-match-eval/data/match_eval \
  --output-dir data/ch5/chunk_matching_zero_coverage \
  --dataset out_of_africa_n100_10mbp --dataset tgp_chr20_n100 \
  --ac-cutoff 10 --min-focal-ac 2
```

Omit `--workers` to use available cores, or set it explicitly. An optional
`--haplotypes selection.json` maps dataset names to lists of
`{"sample_id": "…", "ploidy_index": 0}`; otherwise process the whole roster.
Each dataset writes `full_matches.jsonl`, `stitched_matches.jsonl` and
`comparison.csv`. Choose a fresh output directory for a new scientific run.

## Chapter 5 controlled stitching comparison

This command starts controlled focal HMM and topology calculations while
reusing completed gap paths. Inputs are a saved upstream parameter grid,
current and earlier reference/sample products, and optional simulated truth.
It writes comparisons, controls, component and distance tables, per-method
JSONL paths and `simulation_topology.csv`.

```sh
uv run python scripts/stitching_method_comparison.py \
  --pipeline-config ../tsinfer-match-eval/config.yaml \
  --input-dir ../tsinfer-match-eval/data/match_eval \
  --old-input-dir data/ch5/match_eval \
  --old-output-dir data/ch5/chunk_matching_zero_coverage \
  --output-dir data/ch5/stitching_pt3/method_comparison \
  --dataset out_of_africa_n100_10mbp=out_of_africa_n100_10mbp \
  --dataset tgp_chr20_n100_10mbp=tgp_chr20_n100 \
  --simulation out_of_africa_n100_10mbp --ac-cutoff 10
```

Use `--pipeline-base-dir` when a historical configuration has moved away from
its original relative path base. `--archive-haplotypes` uses the same optional
JSON selection format for detailed path archives. Diagnostic distance-bin
edges, worker count and logging interval are explicit CLI options. The driver
retains the historical matcher calculations; CLI help/import checks were run,
while expensive matching/topology jobs were not launched during migration.

## Chapter 5 isolated native tie-breaking comparison

`compare_tie_breaking.py` replaces the notebook's direct execution of archived
source. The historical sources remain unchanged as research provenance in
`data/ch5/tie_breaking/source/`; active preparation, matching workers, toy
calculations and summaries live in `ch5.nb_utils`.

```sh
uv run python scripts/compare_tie_breaking.py \
  --config data/ch5/tie_breaking/experiment_config.json \
  --output data/ch5/tie_breaking/reproduced \
  --tsinfer-repository /path/to/recorded/tsinfer
```

This command **compiles and runs native matching experiments**. Use a fresh
output directory. It reads existing `hpc_perf2_diagnostics` archives and builds
only in temporary directories. `--workers` defaults to available CPUs;
`--worker-python` selects an interpreter with this workspace installed and the
recorded historical matching API. Native variants are selected in separate
processes through their compiled-extension directory; no notebook/package
`sys.path` edits or compatibility modules are used. `--stage` exposes the toy,
matching and summary stages for explicitly prepared inputs.

The current pinned tsinfer revision lacks `MatcherIndexes`/`AncestorMatcher`
and has changed its C build layout. The driver fails before native builds with
a clear historical-environment requirement. CLI import/help and saved-result
summary checks were performed; the compiler/matcher experiment was not rerun.
