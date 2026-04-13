from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_CONFIGS: List[Dict[str, str]] = [
    {
        "name": "lq_fast",
        "mode": "realtime",
        "quality": "fast",
        "max_context_tokens": "256",
        "hvp_samples": "1",
    },
    {
        "name": "lq_mid",
        "mode": "realtime",
        "quality": "balanced",
        "max_context_tokens": "384",
        "hvp_samples": "1",
    },
    {
        "name": "lq_high",
        "mode": "fidelity",
        "quality": "accurate",
        "max_context_tokens": "512",
        "hvp_samples": "1",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed latency-quality sweep and optionally aggregate the results."
    )
    parser.add_argument("--input_jsonl", required=True, help="Path to converted HETA-style JSONL.")
    parser.add_argument("--output_root", default="outputs", help="Directory that stores run folders.")
    parser.add_argument("--model", default="Qwen2.5-3B", help="Model label or raw HF id.")
    parser.add_argument("--seed", type=int, default=0, help="Seed suffix used in run folder names.")
    parser.add_argument("--max_examples", type=int, default=30, help="Examples per latency point.")
    parser.add_argument("--save_every", type=int, default=10, help="Checkpoint cadence per run.")
    parser.add_argument(
        "--extra_baseline_csv",
        action="append",
        default=[],
        help="Optional baseline CSV(s) forwarded to aggregate_phase2.py.",
    )
    parser.add_argument(
        "--aggregate_output_dir",
        default="outputs/latency_sweep_aggregate",
        help="Aggregate output directory for the final two plots plus summary.csv.",
    )
    parser.add_argument(
        "--skip_aggregate",
        action="store_true",
        help="Run sweep folders only and skip aggregate step.",
    )
    parser.add_argument(
        "--strict_methods",
        action="store_true",
        help="Forward --strict_methods to aggregate if extra baselines are provided.",
    )
    parser.add_argument(
        "--title_prefix",
        default="Latency Sweep",
        help="Plot title prefix for aggregate output.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[cmd] {}".format(" ".join(cmd)), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs: List[str] = []
    for cfg in DEFAULT_CONFIGS:
        run_dir = output_root / f"{cfg['name']}_s{args.seed}_{args.max_examples}"
        run_dirs.append(run_dir.name)
        cmd = [
            sys.executable,
            "scripts/run_faithfulness_hotpot.py",
            "--mode",
            cfg["mode"],
            "--input_jsonl",
            args.input_jsonl,
            "--output_dir",
            str(run_dir),
            "--model",
            args.model,
            "--max_examples",
            str(args.max_examples),
            "--seed",
            str(args.seed),
            "--save_every",
            str(args.save_every),
            "--quality",
            cfg["quality"],
            "--max_context_tokens",
            cfg["max_context_tokens"],
            "--hvp_samples",
            cfg["hvp_samples"],
        ]
        run_cmd(cmd, repo_root)

    if args.skip_aggregate:
        return

    aggregate_cmd = [
        sys.executable,
        "scripts/aggregate_phase2.py",
        "--input_root",
        str(output_root),
        "--run_glob",
        f"lq_*_s{args.seed}_{args.max_examples}",
        "--output_dir",
        args.aggregate_output_dir,
        "--no_timestamp",
        "--title_prefix",
        args.title_prefix,
    ]
    for path in args.extra_baseline_csv:
        aggregate_cmd.extend(["--extra_baseline_csv", path])
    if args.strict_methods:
        aggregate_cmd.append("--strict_methods")
    run_cmd(aggregate_cmd, repo_root)


if __name__ == "__main__":
    main()
