"""Export phase runs for six illustration rows from the ancestor-evaluation data.

Copy this standalone script to tsinfer-anc-eval and run
``uv run python ch3_extract_phase_data.py``. Edit ExtractionConfig to change
the input dataset or the number of individuals per phase setting. The same
first individuals are selected for each setting to allow direct comparison.
Only numpy, pandas and zarr are required; no plotting code is imported.
"""

import dataclasses
import pathlib

import numpy as np
import pandas as pd
import zarr


@dataclasses.dataclass
class ExtractionConfig:
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
    output = data_folder / "dataframes" / f"{config.data_prefix}_phase_runs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(extract_data())
