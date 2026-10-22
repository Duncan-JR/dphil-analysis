# DPhil analysis

Install from the repository root:

```sh
uv sync
```

The development dependencies include JupyterLab and Jupytext. Open the paired
notebook with:

```sh
uv run jupyter lab notebooks/ch4/initial_testing.ipynb
```

Keep the Python and notebook forms synchronized with:

```sh
uv run jupytext --sync notebooks/ch4/initial_testing.py
```

The estimator assumes complete, phased, fixed-ploidy, biallelic 0/1 genotypes
on one sequence, with sorted, unique variant positions. It uses observed
doubletons with complete windows and global diversity from the same store.
Recombination is supplied explicitly, for example as a HapMap file with
position and rate in columns 1 and 2 (zero-indexed).

Run a script with `uv run python estimate.py`:

```python
import error_estimation


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
`error_validation.write_doubletons_csv()` instead of sampling from the input
Zarr.

Pass `variant_mask_name="variant_mask"` to exclude sites where that boolean
Zarr array is true. A sequence of mask names excludes the union of their true
values. Doubleton ascertainment, diversity, region span, mismatch counts and
recombination summaries all use the resulting included sites.

Chapter 3 plots can use aggregate CSVs and an exported Zarr-summary CSV without
transferring the variant stores. Copy `src/ch3_extract_data.py` to the HPC and
run it from the `tsinfer-anc-eval` root (or invoke it by its full path):

```sh
uv run python ch3_extract_data.py OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1
```

This reads `data/anc_eval/zarr_vcfs` and writes
`data/anc_eval/dataframes/OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1_zarr_data.csv`.
The script is standalone and needs only NumPy, pandas and Zarr. It streams
variant chunks and processes independent profiles across available CPU cores;
`--workers`, `--chunk-rows`, `--data-folder` and `--output-folder` are optional.
The output is replaced only after extraction finishes successfully.

Transfer that export to `dphil-analysis/data/ch3/`, together with access to the
matching `PREFIX-aggregated.csv`. In `notebooks/ch3/figure_gen.py`, set
`aggregated_folder` to the directory containing the aggregate and `data_prefix`
to the chosen prefix. Preprocessing writes reduced ancestor tables into
`data_folder` (`dphil-analysis/data/ch3/` by default). The notebook currently
uses the validated local `OutOfAfrica_3G09-chr20-L0-R1e7-n3000-s1` dataset.

The exported CSV contains `profile` rows describing available error profiles,
`truth` rows with unflipped allele counts and positions, `genotype` rows with
error/transition counts grouped by true allele count, and `daf` rows with site
counts grouped by allele count and polarisation. These are raw counts, so local
frequency bins and smoothing settings remain adjustable. The phase-switch
illustration is excluded and retains its separate chromosome 1 Zarr input.
