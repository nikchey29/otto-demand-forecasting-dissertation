from __future__ import annotations

import argparse
import json
from pathlib import Path

from otto_forecasting.config import load_config
from otto_forecasting.data import (
    aggregate_jsonl,
    audit_frame,
    generate_synthetic_hourly,
    load_processed,
    save_processed,
)
from otto_forecasting.pipeline import run_training
from otto_forecasting.research import run_ablation_suite, run_research_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="otto-forecast")
    subcommands = parser.add_subparsers(dest="command", required=True)

    aggregate = subcommands.add_parser("aggregate", help="Aggregate raw OTTO JSONL data")
    aggregate.add_argument("--input", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--frequency", default="1h")

    audit = subcommands.add_parser("audit-data", help="Validate and fingerprint processed data")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output")

    synthetic = subcommands.add_parser(
        "make-smoke-data", help="Generate synthetic hourly data for a smoke test"
    )
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--hours", type=int, default=960)
    synthetic.add_argument("--seed", type=int, default=42)

    train = subcommands.add_parser("train", help="Run the fast single-holdout experiment")
    train.add_argument("--config", default="configs/default.yaml")

    research = subcommands.add_parser(
        "research", help="Run rolling-origin, repeated-seed research evaluation"
    )
    research.add_argument("--config", default="configs/research.yaml")

    ablate = subcommands.add_parser(
        "ablate", help="Run Transformer lookback, feature and depth ablations"
    )
    ablate.add_argument("--config", default="configs/research.yaml")

    run_all = subcommands.add_parser("run-all", help="Aggregate raw data and train")
    run_all.add_argument("--config", default="configs/default.yaml")

    return parser


def _print_paths(outputs: dict[str, Path]) -> None:
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "aggregate":
        frame = aggregate_jsonl(arguments.input, arguments.frequency)
        output = save_processed(frame, arguments.output)
        print(json.dumps({"rows": len(frame), "output": str(output)}, indent=2))
        return

    if arguments.command == "audit-data":
        frame = load_processed(arguments.input)
        report = audit_frame(frame, arguments.input)
        if arguments.output:
            destination = Path(arguments.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if arguments.command == "make-smoke-data":
        frame = generate_synthetic_hourly(arguments.hours, arguments.seed)
        output = save_processed(frame, arguments.output)
        print(json.dumps({"rows": len(frame), "output": str(output)}, indent=2))
        return

    config = load_config(arguments.config)
    if arguments.command == "run-all":
        frame = aggregate_jsonl(config.data.raw_path, config.data.frequency)
        save_processed(frame, config.data.processed_path)
        _print_paths(run_training(config))
        return
    if arguments.command == "train":
        _print_paths(run_training(config))
        return
    if arguments.command == "research":
        _print_paths(run_research_suite(config))
        return
    if arguments.command == "ablate":
        _print_paths(run_ablation_suite(config))
        return
    raise RuntimeError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    main()
