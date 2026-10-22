"""Export the Zarr inputs for chapter 3 plots to one transferable CSV per prefix.

Run from tsinfer-anc-eval: ``uv run python /path/to/ch3_extract_data.py PREFIX``.
The default input is data/anc_eval/zarr_vcfs and the output is
``data/anc_eval/dataframes/PREFIX_zarr_data.csv``. This script is standalone:
only numpy, pandas and zarr are required. The phase-switch illustration is
excluded; it continues to use its separate chromosome 1 Zarr dataset.

Record types are profile metadata, truth (unflipped counts at each variant),
genotype (diploid error/transition counts by true allele count), and daf
(site counts by allele count and polarisation). Counts remain unbinned so
``ch3_plot`` can change frequency bins without repeating the extraction.
"""

import argparse
import dataclasses
import logging
import multiprocessing
import pathlib
import re

import numpy as np
import pandas as pd
import zarr

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ExtractionConfig:
    """Extraction paths, bounded variant batches and CSV schema."""

    data_folder: pathlib.Path = pathlib.Path("data/anc_eval")
    output_folder: pathlib.Path | None = None
    chunk_rows: int = 1000
    workers: int | None = None
    columns: tuple[str, ...] = (
        "record_type", "data_prefix", "replicate", "geno_multiplier", "phase_ser",
        "mispol_rate", "num_alleles", "focal_position", "derived_count",
        "mispolarised", "num_sites", "changed", "total", "homozygous_0",
        "homozygous_2", "to_heterozygous_0", "to_heterozygous_2",
    )


@dataclasses.dataclass(frozen=True)
class VariantStore:
    """One replicate/error profile parsed by :func:`discover_stores`."""

    path: pathlib.Path
    replicate: int
    geno_multiplier: float
    phase_ser: float
    mispol_rate: float


@dataclasses.dataclass
class StoreTask:
    store: VariantStore
    truth: VariantStore
    prefix: str
    config: ExtractionConfig


def discover_stores(folder, prefix):
    """Find error-profile stores, excluding the raw simulation stores."""
    folder = pathlib.Path(folder)
    pattern = re.compile(
        r"(.+)-rep(\d+)-geno-([0-9.eE+-]+)-phase([0-9.eE+-]+)"
        r"-mispol([0-9.eE+-]+)\.zarr"
    )
    stores = []
    for path in sorted(folder.glob(f"{prefix}-rep*.zarr")):
        match = pattern.fullmatch(path.name)
        if match is None or match[1] != prefix:
            continue
        stores.append(VariantStore(
            path, int(match[2]), float(match[3]), float(match[4]), float(match[5])
        ))
    if len(stores) == 0:
        raise FileNotFoundError(f"No variant stores for {prefix} in {folder}")
    keys = [(s.replicate, s.geno_multiplier, s.phase_ser, s.mispol_rate) for s in stores]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate error profiles for {prefix} in {folder}")
    return stores


def find_store(stores, replicate, geno_multiplier=0, phase_ser=0, mispol_rate=0):
    """Require a unique store for a numeric error profile and replicate."""
    matches = [
        store for store in stores
        if store.replicate == replicate and store.geno_multiplier == geno_multiplier
        and store.phase_ser == phase_ser and store.mispol_rate == mispol_rate
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one store for rep={replicate}, geno={geno_multiplier}, "
            f"phase={phase_ser}, mispol={mispol_rate}; found {len(matches)}"
        )
    return matches[0]


def _genotype_counts(truth_store, observed, config):
    """Accumulate exact diploid counts by true derived allele count, in chunks."""
    truth = zarr.open_group(truth_store.path, mode="r")
    true_calls = truth["call_genotype"]
    observed_calls = observed["call_genotype"]
    if true_calls.shape != observed_calls.shape:
        raise ValueError("Genotype shapes differ between the observed and true stores")
    if true_calls.shape[2] != 2:
        raise ValueError("Genotype-error plots require diploid genotypes")
    if not np.array_equal(truth["sample_id"][:], observed["sample_id"][:]):
        raise ValueError("Sample order differs between the observed and true stores")
    num_individuals = true_calls.shape[1]
    num_alleles = num_individuals * true_calls.shape[2]
    names = ["changed", "total", "homozygous_0", "homozygous_2",
             "to_heterozygous_0", "to_heterozygous_2"]
    counts = {name: np.zeros(num_alleles + 1, dtype=np.int64) for name in names}
    for start in range(0, true_calls.shape[0], config.chunk_rows):
        region = slice(start, start + config.chunk_rows)
        if not np.array_equal(truth["variant_position"][region], observed["variant_position"][region]):
            raise ValueError("Variant order differs between the observed and true stores")
        true_dosage = true_calls[region].sum(axis=2)
        observed_dosage = observed_calls[region].sum(axis=2)
        allele_counts = true_dosage.sum(axis=1).astype(np.int64)
        sites = np.bincount(allele_counts, minlength=num_alleles + 1)
        counts["total"] += sites * num_individuals
        changed = (true_dosage != observed_dosage).sum(axis=1)
        counts["changed"] += np.bincount(
            allele_counts, weights=changed, minlength=num_alleles + 1
        ).astype(np.int64)
        for dosage in (0, 2):
            homozygous = true_dosage == dosage
            to_het = homozygous & (observed_dosage == 1)
            denominators = homozygous.sum(axis=1)
            numerators = to_het.sum(axis=1)
            counts[f"homozygous_{dosage}"] += np.bincount(
                allele_counts, weights=denominators, minlength=num_alleles + 1
            ).astype(np.int64)
            counts[f"to_heterozygous_{dosage}"] += np.bincount(
                allele_counts, weights=numerators, minlength=num_alleles + 1
            ).astype(np.int64)
    selected = counts["total"] > 0
    frame = pd.DataFrame({name: values[selected] for name, values in counts.items()})
    frame["derived_count"] = np.flatnonzero(selected)
    frame["record_type"] = "genotype"
    return frame


def _extract_store(task):
    """Read one profile; the parent process writes the returned rows to CSV."""
    store = task.store
    group = zarr.open_group(store.path, mode="r")
    shape = group["call_genotype"].shape
    num_alleles = shape[1] * shape[2]
    frames = [pd.DataFrame({
        "record_type": ["profile"], "data_prefix": [task.prefix], "num_sites": [shape[0]],
    })]
    if store.geno_multiplier == 0 and store.phase_ser == 0:
        allele_counts = group["variant_allele_count"][:].astype(np.int64)
        flipped = ~group["variant_mispolarisation_mask"][:].astype(bool)
        frequencies = pd.DataFrame({"derived_count": allele_counts, "mispolarised": flipped})
        frequencies = frequencies.value_counts(sort=False).rename("num_sites").reset_index()
        frequencies["record_type"] = "daf"
        frames.append(frequencies)
        if store.mispol_rate == 0:
            frames.append(pd.DataFrame({
                "record_type": "truth", "focal_position": group["variant_position"][:],
                "derived_count": allele_counts,
            }))
    if store.geno_multiplier > 0 and store.phase_ser == 0 and store.mispol_rate == 0:
        frames.append(_genotype_counts(task.truth, group, task.config))
    frame = pd.concat(frames, ignore_index=True)
    frame["replicate"] = store.replicate
    frame["geno_multiplier"] = store.geno_multiplier
    frame["phase_ser"] = store.phase_ser
    frame["mispol_rate"] = store.mispol_rate
    frame["num_alleles"] = num_alleles
    return frame.reindex(columns=task.config.columns)


def extract_data(prefix, config=None):
    """Write ``PREFIX_zarr_data.csv`` atomically; see the module's record schema."""
    if config is None:
        config = ExtractionConfig()
    if config.chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    data_folder = pathlib.Path(config.data_folder).expanduser()
    stores = discover_stores(data_folder / "zarr_vcfs", prefix)
    tasks = []
    for store in stores:
        truth = find_store(stores, store.replicate)
        tasks.append(StoreTask(store, truth, prefix, config))
    output_folder = config.output_folder
    if output_folder is None:
        output_folder = data_folder / "dataframes"
    output_folder = pathlib.Path(output_folder).expanduser()
    output_folder.mkdir(parents=True, exist_ok=True)
    output = output_folder / f"{prefix}_zarr_data.csv"
    temporary = output.with_suffix(".csv.tmp")
    workers = config.workers
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(tasks))
    if workers < 1:
        raise ValueError("workers must be positive")
    with temporary.open("w") as file:
        if workers == 1:
            for index, task in enumerate(tasks):
                frame = _extract_store(task)
                frame.to_csv(file, index=False, header=index == 0)
                logger.info("Extracted %s (%s rows)", task.store.path.name, len(frame))
        else:
            context = multiprocessing.get_context("spawn")
            with context.Pool(processes=workers) as pool:
                for index, frame in enumerate(pool.imap(_extract_store, tasks)):
                    frame.to_csv(file, index=False, header=index == 0)
                    logger.info("Extracted %s (%s rows)", tasks[index].store.path.name, len(frame))
    temporary.replace(output)
    logger.info("Wrote %s", output)
    return output


def main():
    """Command-line entry point; no plotting dependencies or repository imports."""
    defaults = ExtractionConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="Dataset prefix, without replicate/error suffixes")
    parser.add_argument("--data-folder", type=pathlib.Path, default=defaults.data_folder)
    parser.add_argument("--output-folder", type=pathlib.Path, default=defaults.output_folder)
    parser.add_argument("--chunk-rows", type=int, default=defaults.chunk_rows)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = ExtractionConfig(
        data_folder=args.data_folder, output_folder=args.output_folder,
        chunk_rows=args.chunk_rows, workers=args.workers,
    )
    extract_data(args.prefix, config)


if __name__ == "__main__":
    main()
