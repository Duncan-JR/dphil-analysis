"""Chapter 5: run zero-coverage gap matching on existing saved pipeline inputs.

This launches HMM matching; it does not merely load result tables. Run from the
repository root with explicit input, coverage and output directories. Inputs
include saved trees, Zarr panels, focal NPZs and native TOML parameter records.
"""

import argparse
import json
import logging
import pathlib

from ch5 import nb_utils

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--coverage-input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Saved dataset name; repeat for several datasets",
    )
    parser.add_argument(
        "--haplotypes",
        type=pathlib.Path,
        help="Optional JSON mapping dataset names to sample_id/ploidy_index selections",
    )
    parser.add_argument("--ac-cutoff", type=int, default=10)
    parser.add_argument("--min-focal-ac", type=int, default=2)
    parser.add_argument("--score-tolerance", type=float, default=1e-8)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if not 1 <= args.min_focal_ac <= args.ac_cutoff:
        parser.error("require 1 <= min-focal-ac <= ac-cutoff")
    if args.workers is not None and args.workers < 1:
        parser.error("workers must be positive")
    if args.score_tolerance < 0:
        parser.error("score-tolerance must be nonnegative")
    selection = (
        {} if args.haplotypes is None else json.loads(args.haplotypes.read_text())
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in args.dataset:
        settings = nb_utils.ChunkDatasetSettings(
            name,
            args.input_dir.resolve(),
            args.coverage_input_dir.resolve(),
            args.ac_cutoff,
            args.min_focal_ac,
            args.score_tolerance,
        )
        nb_utils.chunk_run_dataset(
            settings,
            selection.get(name),
            args.workers,
            args.output_dir.resolve() / name,
        )


if __name__ == "__main__":
    main()
