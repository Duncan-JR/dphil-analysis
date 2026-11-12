# Chapter 4 error estimation

`initial_testing` records validation, doubleton classification, fitted error
models and empirical 1KGP/pig analyses. Run from the repository root. Notebook
calls go through `ch4.nb_utils` and `ch4.nb_plot`; estimator and validation
methods remain in their own chapter libraries. Settings are initialised in
cells. Saved outputs are preserved; full upstream estimation was not rerun
as a migration check. Original simulation stores are external inputs.

The estimator assumes complete, phased, fixed-ploidy, biallelic 0/1 genotypes
on one sequence, with sorted, unique variant positions. It uses observed
doubletons with complete windows and global diversity from the same store.
Recombination is supplied explicitly, for example as a HapMap file with
position and rate in columns 1 and 2 (zero-indexed).

Example script caller:

```python
from ch4 import error_estimation


if __name__ == "__main__":
    config = error_estimation.EstimationConfig(
        window_sizes=[1_000, 5_000, 10_000, 50_000, 100_000, 250_000],
        num_doubletons=10_000,
        random_seed=42,
        exclude_high_mismatch_proportion=0.25,
        L_mismatch_trim=1_000,
    )
    result = error_estimation.estimate_error_rate(
        "data.zarr",
        recombination="genetic_map.txt",
        config=config,
    )
    print(result.fit.epsilon)
    print(result.fit.success, result.fit.message)
```

The main guard supports multiprocessing; set `num_workers=1` for serial use.
`epsilon` means additive errors per haplotype-bp, not a per-genotype probability.
Before fitting, the estimator ranks doubletons by their mismatch counts at
`L_mismatch_trim` and excludes the configured highest-count proportion from
every window. `L_mismatch_trim` defaults to the smallest configured window. The
full mismatch arrays remain available in the result. Pass
`fixed_doubletons_path="doubletons.csv"` to use a fixed set created with
`ch4.error_validation.write_doubletons_csv()` instead of sampling from the input
Zarr.

Pass `variant_mask_name="variant_mask"` to exclude sites where that boolean
Zarr array is true. A sequence of mask names excludes the union of their true
values. Doubleton ascertainment, diversity, region span, mismatch counts and
recombination summaries all use the resulting included sites.

