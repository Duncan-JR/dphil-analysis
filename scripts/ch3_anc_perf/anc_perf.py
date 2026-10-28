"""Run one inference stage, or extract/aggregate its provenance metrics."""

import argparse
import json
import logging
import pathlib

import pandas as pd
import tsinfer
import tskit
import yaml

logger = logging.getLogger(__name__)


def extract_performance(path, version, num_samples, num_threads):
    """Read all three stage metrics inherited by the final tree sequence."""
    ts = tskit.load(path)
    rows = []
    for prov in ts.provenances():
        record = json.loads(prov.record)
        if record["software"]["name"] == "tsinfer":
            command = record["parameters"]["command"]
            if command in ["generate_ancestors", "match_ancestors", "match_samples"]:
                row = dict(record["resources"])
                row.update(
                    command=command,
                    num_samples=ts.num_samples,
                    num_sites=ts.num_sites,
                    num_mutations=ts.num_mutations,
                    num_edges=ts.num_edges,
                    version=version,
                    provenance_version=record["software"]["version"],
                    sequence_length=ts.sequence_length,
                    num_threads=num_threads,
                    hostname=record["environment"]["os"]["node"],
                )
                rows.append(row)
    commands = [row["command"] for row in rows]
    assert commands == ["generate_ancestors", "match_ancestors", "match_samples"]
    assert ts.num_samples == num_samples
    return pd.DataFrame(rows)


def run_inference(args, config):
    """Use the configured tsinfer checkout in a fresh process for each stage."""
    installation = pathlib.Path(tsinfer.__file__).resolve().parents[1]
    logger.info("Using tsinfer %s from %s", args.version, installation)
    data = tsinfer.VariantData(
        args.zarr,
        ancestral_state=config["ancestral_state"],
        sequence_length=config["sequence_length"],
        site_mask=config["site_mask"],
        sample_mask=config["sample_mask"],
    )
    assert data.num_samples == args.num_samples
    logger.info(
        "%s: %s haplotypes, %s sites", args.command, data.num_samples, data.num_sites
    )
    matching = {
        "num_threads": args.threads,
        "recombination_rate": config["recombination_rate"],
        "mismatch_ratio": config["mismatch_ratio"],
        "precision": config["precision"],
        "progress_monitor": False,
    }
    if args.command == "generate_ancestors":
        ancestors = tsinfer.generate_ancestors(
            data,
            path=args.output,
            num_threads=args.threads,
            genotype_encoding=config["genotype_encoding"],
            progress_monitor=False,
        )
        ancestors.close()
    elif args.command == "match_ancestors":
        with tsinfer.AncestorData.load(args.input[0]) as ancestors:
            ts = tsinfer.match_ancestors(data, ancestors, **matching)
        ts.dump(args.output)
    elif args.command == "match_samples":
        ancestors_ts = tskit.load(args.input[0])
        ts = tsinfer.match_samples(
            data, ancestors_ts, post_process=config["post_process"], **matching
        )
        ts.dump(args.output)
    logger.info("Completed %s: %s", args.command, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "generate_ancestors",
            "match_ancestors",
            "match_samples",
            "extract",
            "aggregate",
        ],
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--zarr")
    parser.add_argument("--version")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    logging.basicConfig(
        level=config["log_level"],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "match_samples":
        # Released versions log twice per haplotype at INFO in the worker loop.
        logging.getLogger("tsinfer.inference").setLevel(
            config["sample_matching_log_level"]
        )
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.command == "extract":
        df = extract_performance(
            args.input[0], args.version, args.num_samples, args.threads
        )
        df.to_csv(args.output, index=False)
    elif args.command == "aggregate":
        frames = [pd.read_csv(path) for path in args.input]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["num_samples", "version", "command"])
        df.to_csv(args.output, index=False)
    else:
        run_inference(args, config)


if __name__ == "__main__":
    main()
