"""Chapter 5: reproduce the isolated native HMM tie-breaking comparison.

Read the retained experiment JSON and hpc_perf2_diagnostics archives. Build
control, youngest-state and last-stored variants in temporary directories,
then write a fresh result directory. Requires the recorded historical tsinfer
source/API and a worker interpreter with this workspace installed. Normal
notebook execution reads saved results and does not start these native jobs.

From the repository root:
  uv run python scripts/compare_tie_breaking.py --config CONFIG --output NEW_DIR
"""

import argparse
import concurrent.futures as cf
import dataclasses
import json
import logging
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import _tsinfer
import numpy as np
import pandas as pd
import tsinfer.matching

from ch5 import nb_utils

logger = logging.getLogger(__name__)


def check_native_variant(scratch, mode):
    """Reject a stage labelled with a different compiled native variant."""
    expected = scratch / mode
    if pathlib.Path(_tsinfer.__file__).parent != expected:
        raise RuntimeError(
            f"Expected native tie-breaking variant in {expected}; loaded {_tsinfer.__file__}"
        )


def run_matches(scratch, settings, mode):
    """Distribute independent haplotypes to spawned workers and save their paths."""
    check_native_variant(scratch, mode)
    jobs = [
        nb_utils.TieMatchTask(case, haplotype, settings)
        for case in settings["datasets"]
        for haplotype in case["haplotypes"]
    ]
    workers = settings["workers"]
    if workers is None:
        workers = multiprocessing.cpu_count()
    context = multiprocessing.get_context("spawn")
    results = {}
    before = time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for count, result in enumerate(
            pool.map(nb_utils.tie_match_one, jobs, chunksize=4), 1
        ):
            result.row["mode"] = mode
            results.setdefault(result.row["dataset"], []).append(result)
            if count % 64 == 0:
                logger.info(
                    "%s: %s/%s haplotypes, %.1fs",
                    mode,
                    count,
                    len(jobs),
                    time.monotonic() - before,
                )
    for name, rows in results.items():
        pd.DataFrame([result.row for result in rows]).to_csv(
            scratch / f"{name}_{mode}.csv", index=False
        )
        lengths = [len(result.left) for result in rows]
        offsets = np.r_[0, np.cumsum(lengths)]
        np.savez_compressed(
            scratch / f"{name}_{mode}_paths.npz",
            offsets=offsets,
            haplotype_slot=np.array([result.row["haplotype_slot"] for result in rows]),
            left=np.concatenate([result.left for result in rows]),
            right=np.concatenate([result.right for result in rows]),
            parent=np.concatenate([result.parent for result in rows]),
        )


def run_toys(scratch, settings, mode):
    check_native_variant(scratch, mode)
    cases = [nb_utils.TieToyCase(**case) for case in settings["toy_cases"]]
    rows = [nb_utils.tie_toy_case(case, mode) for case in cases]
    pd.DataFrame(rows).to_csv(scratch / f"toys_{mode}.csv", index=False)


def run_comparison(settings, output, worker_python):
    """Build only in scratch and orchestrate isolated native worker processes."""
    missing = [
        name
        for name in ("MatcherIndexes", "AncestorMatcher")
        if not hasattr(tsinfer.matching, name)
    ]
    if missing:
        raise RuntimeError(
            "The retained tie-breaking experiment requires the historical tsinfer matching API: "
            + ", ".join(missing)
            + ". Restore its recorded source/environment before starting native builds."
        )
    if output.exists():
        raise FileExistsError(f"Choose a fresh tie-breaking output directory: {output}")
    repo = pathlib.Path(settings["tsinfer_repository"]).expanduser().resolve()
    script = pathlib.Path(__file__).resolve()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="tsinfer_tie_breaking_") as directory:
        scratch = pathlib.Path(directory)
        nb_utils.tie_prepare_builds(repo, scratch, settings["modes"])
        nb_utils.tie_prepare_data(scratch, settings)
        prepared_config = scratch / "experiment_config.json"
        for mode in settings["modes"]:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "--project",
                    str(repo),
                    "python",
                    "setup.py",
                    "build_ext",
                    "--inplace",
                ],
                cwd=scratch / mode,
                env=env,
                check=True,
            )
            # Select the experimental native extension in a fresh process.
            # Python package imports remain installed; no sys.path edits occur.
            mode_env = env.copy()
            mode_env["PYTHONPATH"] = str(scratch / mode)
            command = [
                "uv",
                "run",
                "--no-project",
                "--python",
                worker_python,
                "python",
                str(script),
            ]
            for stage in ("toys", "matches"):
                subprocess.run(
                    command
                    + [
                        "--stage",
                        stage,
                        "--config",
                        str(prepared_config),
                        "--output",
                        str(scratch),
                        "--mode",
                        mode,
                    ],
                    env=mode_env,
                    check=True,
                )
        effective = json.loads(prepared_config.read_text())
        nb_utils.tie_summarise(scratch, effective)
        output.mkdir(parents=True)
        for pattern in ("*.csv", "*.diff", "*_paths.npz"):
            for path in scratch.glob(pattern):
                shutil.copy2(path, output / path.name)
        shutil.copy2(prepared_config, output / "experiment_config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Retained upstream experiment JSON; scientific settings remain explicit",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Fresh result directory; internal stages use their prepared scratch directory",
    )
    parser.add_argument("--workers", type=int, help="Available CPU count when omitted")
    parser.add_argument(
        "--analysis-root", type=pathlib.Path, default=pathlib.Path.cwd()
    )
    parser.add_argument(
        "--tsinfer-repository",
        type=pathlib.Path,
        help="Historical source checkout; otherwise use its retained config path",
    )
    parser.add_argument(
        "--worker-python",
        default=sys.executable,
        help="Interpreter with ch5 installed and the recorded matcher API",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "toys", "matches", "summary"],
        default="all",
        help="Internal native stages are also callable for explicit prepared inputs",
    )
    parser.add_argument("--mode", choices=["control", "youngest", "last_tie"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = json.loads(args.config.read_text())
    if args.stage == "all":
        settings["workers"] = args.workers
        settings["analysis_root"] = str(args.analysis_root.resolve())
        if args.tsinfer_repository is not None:
            settings["tsinfer_repository"] = str(args.tsinfer_repository.resolve())
        chain = [(0, 1), (1, 2), (2, 3)]
        star = [(0, 1), (1, 2), (1, 3)]
        cases = [
            nb_utils.TieToyCase(
                "root_tie_explicit_young_alternative",
                [10, 9, 5, 1],
                star,
                [3, 2, 2],
                [0, 1, 0],
            ),
            nb_utils.TieToyCase(
                "root_tie_young_alternative_compressed",
                [10, 9, 5, 1],
                chain,
                [None],
                [0],
            ),
            nb_utils.TieToyCase(
                "derived_identical_chain", [10, 9, 5, 1], chain, [2, 2, 2], [1, 1, 1]
            ),
            nb_utils.TieToyCase(
                "age_vs_last_stored_tie",
                [10, 9, 1, 5],
                star,
                [2, 3],
                [1, 1],
                rho=0,
                mu=0.125,
            ),
        ]
        settings["toy_cases"] = [dataclasses.asdict(case) for case in cases]
        run_comparison(settings, args.output, args.worker_python)
    elif args.stage == "summary":
        nb_utils.tie_summarise(args.output, settings)
    else:
        if args.mode is None:
            parser.error("--mode is required for native toy and matching stages")
        if args.stage == "toys":
            run_toys(args.output, settings, args.mode)
        else:
            run_matches(args.output, settings, args.mode)


if __name__ == "__main__":
    main()
