"""Chapter 5: compare saved stitching and controlled focal HMM matches.

This launches controlled HMM matching and topology calculations from saved
pipeline products. It does not run the upstream inference pipeline. Scientific
selectors and all input/output paths are explicit CLI arguments. The pipeline
configuration identifies the saved parameter grid and simulated truth inputs.
"""

import argparse
import json
import logging
import multiprocessing
import pathlib

import numpy as np
import pandas as pd
import tskit
import yaml

from ch5 import nb_utils

logger = logging.getLogger(__name__)


def run(settings, *, workers=None, progress_every=20) -> None:
    assert settings["comparison"]["max_mismatches"] == 0
    assert min(settings["pipeline"]["max_mismatches"]) == 0
    settings["output_dir"].mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in settings["datasets"]:
        reference = tskit.load(
            settings["input_dir"] / "ancestors" / f"{name}_inferred_ancestors.trees"
        )
        focal = nb_utils.comparison_read_archive(
            settings["input_dir"]
            / "focal_ancestors"
            / f"{name}_inferred_focal_ancestors.npz"
        )
        observations = nb_utils.comparison_read_observations(
            settings["input_dir"], name, reference, focal
        )
        np.save(settings["output_dir"] / f"{name}_observations.npy", observations)
        jobs.extend(((name, index) for index in range(len(focal["sample_id"]))))
        logger.info("Prepared %s (%s haplotypes)", name, observations.shape[1])
    comparisons = []
    controls = []
    components = []
    distances = []
    all_matches = {
        name: {method: [] for method in settings["topology_methods"]}
        for name in settings["datasets"]
    }
    if workers is None:
        workers = multiprocessing.cpu_count()
    pool_context = multiprocessing.get_context("spawn")
    with pool_context.Pool(
        workers, initializer=nb_utils.comparison_initialize_worker, initargs=(settings,)
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(nb_utils.comparison_analyse_haplotype, jobs), start=1
        ):
            comparisons.extend(result.comparisons)
            controls.append(result.controls)
            components.extend(result.components)
            distances.extend(result.distances)
            name = result.controls["dataset"]
            for method in all_matches[name]:
                all_matches[name][method].append(result.matches[method])
            if completed % progress_every == 0:
                logger.info("Compared %s / %s haplotypes", completed, len(jobs))
    for filename, records in {
        "comparisons": comparisons,
        "focal_controls": controls,
        "focal_components": components,
        "switch_distances": distances,
    }.items():
        pd.DataFrame(records).to_csv(
            settings["output_dir"] / f"{filename}.csv", index=False
        )
    for name, methods in all_matches.items():
        for method, records in methods.items():
            records.sort(
                key=lambda record: (record["sample_id"], record["ploidy_index"])
            )
            with (settings["output_dir"] / f"{name}_{method}_matches.jsonl").open(
                "w"
            ) as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")
    topology = nb_utils.comparison_topology_comparison(settings, all_matches)
    topology.to_csv(settings["output_dir"] / "simulation_topology.csv", index=False)
    logger.info("Wrote comparisons to %s", settings["output_dir"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-config",
        type=pathlib.Path,
        required=True,
        help="Upstream run configuration, retained as provenance",
    )
    parser.add_argument(
        "--pipeline-base-dir",
        type=pathlib.Path,
        help="Original base for relative paths in a retained run snapshot",
    )
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--old-input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--old-output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset", action="append", required=True, metavar="NAME=SAVED_GREEDY_NAME"
    )
    parser.add_argument(
        "--simulation",
        action="append",
        default=[],
        help="Dataset names with simulated truth",
    )
    parser.add_argument(
        "--archive-haplotypes",
        type=pathlib.Path,
        help="Optional JSON dataset-to-haplotype selection for path archives",
    )
    parser.add_argument("--ac-cutoff", type=int, default=10)
    parser.add_argument(
        "--distance-bin-edges",
        type=int,
        nargs="+",
        default=[0, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000, 100000],
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()
    pipeline_path = args.pipeline_config.resolve()
    pipeline = yaml.safe_load(pipeline_path.read_text())
    archive = (
        {}
        if args.archive_haplotypes is None
        else json.loads(args.archive_haplotypes.read_text())
    )
    datasets = {}
    for item in args.dataset:
        name, old_name = item.split("=", 1)
        datasets[name] = {
            "name": name,
            "saved_greedy_name": old_name,
            "population_source": "simulation"
            if name in args.simulation
            else "empirical",
            "haplotypes": archive.get(name, []),
        }
    settings = {
        "pipeline": pipeline,
        "pipeline_dir": pipeline_path.parent
        if args.pipeline_base_dir is None
        else args.pipeline_base_dir.resolve(),
        "input_dir": args.input_dir.resolve(),
        "comparison": {
            "ac_cutoff": args.ac_cutoff,
            "max_mismatches": 0,
            "method": "hmm",
        },
        "datasets": datasets,
        "old_input_dir": args.old_input_dir.resolve(),
        "old_output_dir": args.old_output_dir.resolve(),
        "output_dir": args.output_dir.resolve(),
        "distance_bin_edges": args.distance_bin_edges,
        "topology_methods": [
            "full",
            "saved_greedy",
            "greedy",
            "reduced_hmm",
            "independent_hmm",
        ],
    }
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(settings, workers=args.workers, progress_every=args.progress_every)


if __name__ == "__main__":
    main()
