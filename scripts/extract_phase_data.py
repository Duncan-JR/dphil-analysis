"""Export phase runs for six illustration rows from the ancestor-evaluation data.

Copy this standalone script to tsinfer-anc-eval and run
``uv run python extract_phase_data.py``. Pass CLI arguments to select the input dataset and individuals per phase setting. The same
first individuals are selected for each setting to allow direct comparison.
Only numpy, pandas and zarr are required; no plotting code is imported.
"""

import argparse
import dataclasses
import pathlib

import numpy as np
import pandas as pd
import zarr


@dataclasses.dataclass
class ExtractionConfig:
    output_folder: pathlib.Path | None = None
    data_prefix: str = "OutOfAfrica_3G09-chr20-L0-R2e7-n6000-s1"
    data_folder: pathlib.Path = pathlib.Path("data/anc_eval_new")
    n_per_sim: int = 2
    replicate: int = 0
    spacing_labels: tuple[str, ...] = ("2e5", "2e6", "2e7")
    left: float = 0
    right: float = 20_000_000


def extract_data(config=None):
    """Write one CSV of phase segments, grouped by the six plot_row IDs."""
    if config is None:
        config = ExtractionConfig()
    if config.n_per_sim < 1:
        raise ValueError("n_per_sim must be positive")
    data_folder = pathlib.Path(config.data_folder).expanduser()
    frames = []
    for setting, label in enumerate(config.spacing_labels):
        name = (
            f"{config.data_prefix}-rep{config.replicate}-geno-0"
            f"-phase{label}-mispol0.0.zarr"
        )
        group = zarr.open_group(data_folder / "zarr_vcfs" / name, mode="r")
        phase_array = group["call_genotype_phase"]
        if phase_array.shape[1] < config.n_per_sim:
            raise ValueError(f"{name} has fewer than {config.n_per_sim} individuals")
        positions = group["variant_position"][:]
        phase = phase_array[:, : config.n_per_sim].astype(bool)
        sample_ids = group["sample_id"][: config.n_per_sim]
        for individual in range(config.n_per_sim):
            states = phase[:, individual]
            changes = np.flatnonzero(states[1:] != states[:-1]) + 1
            indices = np.r_[0, changes]
            boundaries = np.r_[config.left, positions[changes], config.right]
            frames.append(
                pd.DataFrame(
                    {
                        "data_prefix": config.data_prefix,
                        "replicate": config.replicate,
                        "plot_row": setting * config.n_per_sim + individual,
                        "individual": individual,
                        "sample_id": sample_ids[individual],
                        "space_per_switch": group.attrs["space_per_switch"],
                        "phase_ser": group.attrs["phase_ser"],
                        "left": boundaries[:-1],
                        "right": boundaries[1:],
                        "phase": states[indices],
                    }
                )
            )
    frame = pd.concat(frames, ignore_index=True)
    output_folder = config.output_folder
    if output_folder is None:
        output_folder = data_folder / "dataframes"
    output = pathlib.Path(output_folder) / f"{config.data_prefix}_phase_runs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-prefix", default=ExtractionConfig.data_prefix)
    parser.add_argument(
        "--data-folder", type=pathlib.Path, default=ExtractionConfig.data_folder
    )
    parser.add_argument("--output-folder", type=pathlib.Path)
    parser.add_argument("--n-per-sim", type=int, default=2)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--spacing-labels", nargs="+", default=["2e5", "2e6", "2e7"])
    parser.add_argument("--left", type=float, default=0)
    parser.add_argument("--right", type=float, default=20_000_000)
    args = parser.parse_args()
    config = ExtractionConfig(
        data_prefix=args.data_prefix,
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        n_per_sim=args.n_per_sim,
        replicate=args.replicate,
        spacing_labels=tuple(args.spacing_labels),
        left=args.left,
        right=args.right,
    )
    print(extract_data(config))


if __name__ == "__main__":
    main()
