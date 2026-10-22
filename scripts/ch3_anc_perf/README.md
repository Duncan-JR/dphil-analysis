# Ancestor inference performance

Run from this folder inside an exclusive 196-CPU, 1-TB allocation:

```sh
uv run --project /well/kelleher/users/uuc395/tsinfer-anc-eval --no-sync \
    snakemake --cores 196 --resources mem_mb=950000 --printshellcmds
```

Every rule reserves all configured cores and memory, so the command above runs
one job at a time. Keep the command's totals equal to `config.yaml` resources.
These are scheduling reservations, not an enforced RAM limit. `time_min` is
metadata for cluster submission; the local executor uses the allocation's walltime.

The workflow uses the existing tsinfer-anc-eval environment because these
releases require Zarr 2. Do not run the worker script in dphil-analysis's Zarr 3
environment. Each stage uses a fresh process with the configured release checkout
on `PYTHONPATH`; the Git release tag is checked before inference. Both checkouts
must be built for the environment's Python and NumPy.

`datasets` lists haploid sample counts. The input filename's `n` is half
that count (diploid individuals). The input glob must match exactly one Zarr
dataset per `n`. Default datasets contain 100,
1,000, 10,000 and 100,000 haplotypes, with the same input used for both versions.
Edit paths and inference settings in `config.yaml` before starting. For a rerun
with changed scientific settings, choose a new `output_dir`.

Outputs are retained under `results/`:

- `<version>/n<count>/ancestors.ancestors`: generated ancestors (an LMDB file).
- `<version>/n<count>/ancestors.trees`: matched ancestors.
- `<version>/n<count>/inferred.trees`: matched samples, without post-processing.
- `<version>/n<count>/*.log`: logs for each inference stage.
- `<version>/n<count>/perf.csv`: three provenance rows, one per stage.
- `perf_0.4.1.csv`, `perf_0.5.1.csv`, `perf_combined.csv`: aggregate tables.

INFO logging reports ancestor-building and ancestor-matching activity. During
sample matching, the tsinfer inference logger is set to WARNING to avoid its two
INFO messages per haplotype; stage start/end messages remain. Progress bars and
DEBUG logging are disabled. Helper library threads are limited to one, while
`num_threads` is passed to all three inference calls.

Performance columns match tsinfer-paper's `data/eval/perf_combined.csv`, with
additional site count, thread count, hostname and original provenance version.
Times are seconds and `max_memory` is bytes. Metrics come directly from the
inherited provenance in the final tree sequence. Timers cover the inference API
calls, not all input loading, script startup or output writing. Peak memory is
the process lifetime maximum, including input loading; separate processes prevent
an earlier inference stage's peak from contaminating the next stage.

Some source checkouts report `undefined` in provenance. `version` uses the
verified Git tag and `provenance_version` retains the original string.
The paper's `viz.plot_perf` currently hardcodes 0.5.0/1.0 as the new version;
update that selection to 0.5.1 and select the two matching commands to reproduce
its original comparison. The CSV also includes `generate_ancestors` for a
three-stage comparison.
