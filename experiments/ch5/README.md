# Zero-coverage focal stitching

`chunk_matching.py` reads the copied reference, panel, sample calls, and focal
candidates in `chunk_matching_config.yaml`. It reads anchored focal intervals
and population labels from the configured `coverage_input_dir`. Paths inside
the upstream TOML files are not followed.

Focal coverage is the union of the budget-zero intervals with
`min_focal_ac <= focal_ac <= ac_cutoff`. The intervals extend from each
ancestor's leftmost focal seed until a mismatch, missing call, or support bound.
Gaps are maximal runs outside that union, regardless of any HMM path.

Within each covered component, select an available interval extending furthest
right and keep that parent until the interval ends. Break equal-endpoint ties
by original reference node ID. This minimizes focal copying segments; it does
not optimize gap endpoint continuity. Copied alleles come from the inferred
reference, so differences from generated ancestor calls remain visible as
mutations. Each gap is independently matched against the full reference, and a
separate whole-span HMM provides the comparison path.

Run from the repository root using the existing native tsinfer environment:

```sh
PYTHONPATH=src uv run --no-project \
  --python ~/work/tsinfer-match-eval/.venv/bin/python \
  python experiments/ch5/chunk_matching.py \
  --config experiments/ch5/chunk_matching_config.yaml
```

A null `haplotypes` selection evaluates the complete focal roster. A null
`workers` value uses all available CPU cores, with queued whole-haplotype jobs
and reusable matcher indexes in each worker. Results stream to separate full
and stitched JSONL files and `comparison.csv`, including population labels,
switches within focal coverage, within gaps and at seams, total extra switch
fraction, and four site-weighted parent-agreement fractions. Agreement fractions
use all inference sites; extra switch fraction uses stitched switches.

Open `notebooks/ch5_stitching_pt2.ipynb`, or generate it from the tracked source:

```sh
uv run jupytext --to notebook notebooks/ch5_stitching_pt2.py
```

The notebook reads the existing results, displays selected-haplotype diagnostics,
and builds an 800-row table covering both methods for the two 200-haplotype
cohorts. Its final section exports tables and population plots under the
configured cohort output directory. Error bars show one sample standard
deviation across haplotypes. Both haplotypes of each diploid individual are
included, with three population groups per dataset.
