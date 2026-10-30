"""Extract ancestor spans between their first and last inference sites."""

import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
import tsinfer

logger = logging.getLogger(__name__)


def extract_spans(output_dir):
    """Return focal-site spans for every version and sample count in output_dir.

    Read the generated ``<version>/n<count>/ancestors.ancestors`` stores using
    the benchmark's legacy tsinfer runtime (see README.md). Only ancestor
    metadata is loaded; haplotypes and matched tree sequences are not needed.

    Column names follow tsinfer-anc-eval. ``inf_focal_site`` indexes the
    ancestor store's inference sites, and ``inferred_node`` is its ancestor ID.
    ``inferred_span`` is in bp and ``inferred_site_span`` counts inference
    sites. For the half-open site interval [start, end), the span is
    positions[end - 1] - positions[start]: the distance between the first
    and last included inference sites. A one-site ancestor has span zero.
    The sequence length is not used to calculate these spans.
    Ancestors with multiple focal sites contribute one row for each; ancestors
    without focal sites (including the two artificial roots) contribute none.
    """
    output_dir = pathlib.Path(output_dir)
    paths = sorted(output_dir.glob("*/n*/ancestors.ancestors"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No generated ancestor stores in {output_dir}")
    frames = []
    for path in paths:
        version = path.parent.parent.name
        num_samples = int(path.parent.name[1:])
        logger.info("Reading tsinfer %s, n=%s: %s", version, num_samples, path)
        with tsinfer.AncestorData.load(str(path)) as ancestors:
            positions = ancestors.sites_position[:]
            start = ancestors.ancestors_start[:]
            end = ancestors.ancestors_end[:]
            times = ancestors.ancestors_time[:]
            focal_sites = ancestors.ancestors_focal_sites[:]
            counts = np.array([len(sites) for sites in focal_sites], dtype=int)
            ancestor_ids = np.arange(ancestors.num_ancestors)
            inferred_node = np.repeat(ancestor_ids, counts)
            inf_focal_site = np.concatenate(focal_sites).astype(int)
            last_site = end - 1
            spans = positions[last_site] - positions[start]
            site_spans = end - start
            frame = pd.DataFrame({
                "version": version,
                "num_samples": num_samples,
                "inferred_node": inferred_node,
                "inf_focal_site": inf_focal_site,
                "focal_position": positions[inf_focal_site],
                "inferred_time": times[inferred_node],
                "inferred_span": spans[inferred_node],
                "inferred_site_span": site_spans[inferred_node],
            })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=pathlib.Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    frame = extract_spans(args.output_dir)
    path = args.output_dir / "ancestor_spans.csv"
    frame.to_csv(path, index=False)
    logger.info("Wrote %s focal-site rows to %s", len(frame), path)


if __name__ == "__main__":
    main()
