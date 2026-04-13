from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

METHOD_ORDER = [
    "Random",
    "Attention-Rollout",
    "Integrated Gradients",
    "Gradient×Input",
    "HETA-Lite",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate phase-2 runs and export baseline comparison + latency-quality CSV/PNG outputs."
        )
    )
    parser.add_argument("--input_root", default="outputs", help="Root directory containing run folders.")
    parser.add_argument("--run_glob", default="phase2_*", help="Glob under input_root to select run folders.")
    parser.add_argument(
        "--output_dir",
        default="outputs/phase2_aggregate",
        help="Base output directory. A timestamped child folder is created by default.",
    )
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Write directly to output_dir without timestamp subfolder.",
    )
    parser.add_argument(
        "--group_regex",
        default=r"^(.*)_s\d+(.*)$",
        help="Regex used to normalize run group names across seeds.",
    )
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.20,
        help="Removal ratio used for single-number bar summary and latency-quality mean_drop_r20.",
    )
    parser.add_argument(
        "--extra_baseline_csv",
        action="append",
        default=[],
        help=(
            "Optional external baseline tidy CSV(s) with columns: run_id,example_id,method,"
            "removal_ratio,p0,pr,drop,target_token,seq_len,timestamp"
        ),
    )
    parser.add_argument(
        "--title_prefix",
        default="HETA-Lite Phase2",
        help="Prefix for plot titles.",
    )
    parser.add_argument(
        "--strict_methods",
        action="store_true",
        help="Fail if Random/Attention-Rollout/Integrated Gradients/HETA-Lite are not all present.",
    )
    parser.add_argument(
        "--skip_legacy_outputs",
        action="store_true",
        help="Skip legacy phase-2 CSV/PNG compatibility outputs.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def ratio_key(x: Any) -> str:
    return f"{safe_float(x, 0.0):.2f}"


def ratio_value(x: Any) -> float:
    return safe_float(x, 0.0)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_run_dirs(root: Path, run_glob: str) -> List[Path]:
    runs: List[Path] = []
    for p in sorted(root.glob(run_glob)):
        if p.is_dir() and (p / "summary.json").exists() and (p / "results.jsonl").exists():
            runs.append(p)
    if runs:
        return runs
    for summary in sorted(root.rglob("summary.json")):
        run_dir = summary.parent
        if (run_dir / "results.jsonl").exists():
            runs.append(run_dir)
    return sorted(set(runs))


def normalize_group_name(run_name: str, group_regex: str) -> str:
    try:
        m = re.match(group_regex, run_name)
        if m:
            return (m.group(1) + m.group(2)).strip("_")
    except re.error:
        pass
    return run_name


def extract_seed(run_name: str, fallback_seed: Any = "") -> str:
    m = re.search(r"_s(\d+)", run_name)
    if m:
        return m.group(1)
    if fallback_seed not in (None, ""):
        return str(fallback_seed)
    return ""


def normalize_method_name(method: str) -> str:
    m = (method or "").strip().lower()
    if m in {"random", "rand"}:
        return "Random"
    if m in {"heta", "heta-lite", "heta_lite", "topk", "ours"}:
        return "HETA-Lite"
    if m in {"attn", "attn_roll", "attention-rollout", "attention_rollout", "rollout"}:
        return "Attention-Rollout"
    if m in {"ig", "integrated gradients", "integrated_gradients"}:
        return "Integrated Gradients"
    if m in {"gradxinput", "gradientxinput", "gradientxinput", "grad_x_input"}:
        return "Gradient×Input"
    return method.strip() if method else "Unknown"


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def parse_drop_map(d: Dict[str, Any] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        rk = ratio_key(k)
        out[rk] = safe_float(v, 0.0)
    return out


def parse_prob_map(d: Dict[str, Any] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        rk = ratio_key(k)
        out[rk] = safe_float(v, 0.0)
    return out


def collect_method_curves(rec: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    curves: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    p0 = safe_float(rec.get("target_prob_orig", 0.0))

    topk_drop = parse_drop_map(rec.get("prob_drop_topk"))
    topk_prob = parse_prob_map(rec.get("target_prob_topk"))
    rand_drop = parse_drop_map(rec.get("prob_drop_random"))
    rand_prob = parse_prob_map(rec.get("target_prob_random"))

    for rk, drop in topk_drop.items():
        pr = topk_prob.get(rk, p0 - drop)
        curves["HETA-Lite"][rk] = {"drop": drop, "pr": pr}
    for rk, drop in rand_drop.items():
        pr = rand_prob.get(rk, p0 - drop)
        curves["Random"][rk] = {"drop": drop, "pr": pr}

    # Optional generic container.
    generic_drop = rec.get("prob_drop_methods", {})
    generic_prob = rec.get("target_prob_methods", {})
    if isinstance(generic_drop, dict):
        for method, ratio_map in generic_drop.items():
            mname = normalize_method_name(method)
            drop_map = parse_drop_map(ratio_map if isinstance(ratio_map, dict) else {})
            prob_map = parse_prob_map(
                (generic_prob.get(method) if isinstance(generic_prob, dict) else {})
            )
            for rk, drop in drop_map.items():
                pr = prob_map.get(rk, p0 - drop)
                curves[mname][rk] = {"drop": drop, "pr": pr}

    # Optional explicit keys.
    explicit = {
        "Attention-Rollout": rec.get("prob_drop_attn_roll"),
        "Integrated Gradients": rec.get("prob_drop_ig"),
        "Gradient×Input": rec.get("prob_drop_gradxinput"),
    }
    explicit_pr = {
        "Attention-Rollout": rec.get("target_prob_attn_roll"),
        "Integrated Gradients": rec.get("target_prob_ig"),
        "Gradient×Input": rec.get("target_prob_gradxinput"),
    }
    for method, drop_raw in explicit.items():
        drop_map = parse_drop_map(drop_raw if isinstance(drop_raw, dict) else {})
        pr_map = parse_prob_map(explicit_pr.get(method) if isinstance(explicit_pr.get(method), dict) else {})
        for rk, drop in drop_map.items():
            pr = pr_map.get(rk, p0 - drop)
            curves[method][rk] = {"drop": drop, "pr": pr}

    return curves


def aggregate_baseline_curves(
    baseline_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_method_ratio: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    random_lookup: Dict[Tuple[str, str, str], float] = {}

    for row in baseline_rows:
        method = row["method"]
        ratio = ratio_key(row["removal_ratio"])
        drop = safe_float(row["drop"], 0.0)
        by_method_ratio[(method, ratio)].append(drop)
        if method == "Random":
            key = (row["run_id"], row["example_id"], ratio)
            random_lookup[key] = drop

    gap_by_method_ratio: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in baseline_rows:
        method = row["method"]
        if method == "Random":
            continue
        ratio = ratio_key(row["removal_ratio"])
        key = (row["run_id"], row["example_id"], ratio)
        if key in random_lookup:
            gap = safe_float(row["drop"], 0.0) - random_lookup[key]
            gap_by_method_ratio[(method, ratio)].append(gap)

    out: List[Dict[str, Any]] = []
    for (method, ratio), drops in sorted(by_method_ratio.items(), key=lambda x: (x[0][0], float(x[0][1]))):
        gaps = gap_by_method_ratio.get((method, ratio), [])
        out.append(
            {
                "method": method,
                "removal_ratio": ratio,
                "mean_drop": float(mean(drops)) if drops else 0.0,
                "median_drop": float(median(drops)) if drops else 0.0,
                "std_drop": float(pstdev(drops)) if len(drops) > 1 else 0.0,
                "mean_gap_vs_random": float(mean(gaps)) if gaps else 0.0,
                "n_examples": len(drops),
            }
        )
    return out


def aggregate_latency_quality(
    run_rows: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    target_ratio: float,
) -> List[Dict[str, Any]]:
    ratio_key_target = ratio_key(target_ratio)

    # mean_drop at target ratio for HETA-Lite per run_id
    per_run_drop: Dict[str, List[float]] = defaultdict(list)
    # strict faithfulness at target ratio: drop(HETA) > drop(Random)
    per_example_target: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for row in baseline_rows:
        if ratio_key(row["removal_ratio"]) != ratio_key_target:
            continue
        run_id = row["run_id"]
        example_id = row["example_id"]
        method = row["method"]
        drop = safe_float(row["drop"], 0.0)

        if method == "HETA-Lite":
            per_run_drop[run_id].append(drop)
            per_example_target[(run_id, example_id)]["heta"] = drop
        elif method == "Random":
            per_example_target[(run_id, example_id)]["random"] = drop

    per_run_faithfulness_strict: Dict[str, List[int]] = defaultdict(list)
    per_run_gap: Dict[str, List[float]] = defaultdict(list)
    for (run_id, _example_id), vals in per_example_target.items():
        if "heta" in vals and "random" in vals:
            gap = vals["heta"] - vals["random"]
            per_run_gap[run_id].append(gap)
            per_run_faithfulness_strict[run_id].append(1 if gap > 0 else 0)

    out: List[Dict[str, Any]] = []
    for rr in run_rows:
        run_id = rr["run_name"]
        drops = per_run_drop.get(run_id, [])
        strict_flags = per_run_faithfulness_strict.get(run_id, [])
        gaps = per_run_gap.get(run_id, [])
        quality_label = rr.get("quality", "")
        config_id = rr.get("config_id", "") or rr.get("group", "") or run_id
        out.append(
            {
                "run_id": run_id,
                "config_id": config_id,
                "mode": rr.get("mode", ""),
                "quality": quality_label,
                "hessian_samples": rr.get("hessian_samples", ""),
                "layer_sampling": rr.get("layer_sampling", "default"),
                "window_mode": rr.get("window_mode", "no_window"),
                "kl_budget": rr.get("kl_budget", "full"),
                "n_examples": rr.get("n_examples", 0),
                "latency_p50_ms": rr.get("latency_p50_ms", 0.0),
                "latency_p95_ms": rr.get("latency_p95_ms", 0.0),
                "success_rate_strict": float(mean(strict_flags)) if strict_flags else 0.0,
                "mean_drop_r20": float(mean(drops)) if drops else 0.0,
                "mean_gap_vs_random_r20": float(mean(gaps)) if gaps else 0.0,
                "success_rate_alignment": rr.get("success_rate_strict", 0.0),
                "mean_alignment": rr.get("mean_alignment", 0.0),
                "success_rate_dsa": rr.get("success_rate_dsa", 0.0),
                "evidence_mass_share": rr.get("mean_evidence_share_en", 0.0),
                "timestamp": rr.get("timestamp", ""),
                "notes": rr.get("notes", ""),
            }
        )
    return out


def collapse_latency_quality_by_config(
    sweep_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_cfg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in sweep_rows:
        cfg = str(row.get("config_id", "")).strip() or str(row.get("run_id", "")).strip()
        by_cfg[cfg].append(row)

    out: List[Dict[str, Any]] = []
    for cfg, rows in sorted(by_cfg.items()):
        out.append(
            {
                "config_id": cfg,
                "n_runs": len(rows),
                "latency_p95_ms": float(mean([safe_float(r.get("latency_p95_ms", 0.0), 0.0) for r in rows])),
                "success_rate_strict": float(mean([safe_float(r.get("success_rate_strict", 0.0), 0.0) for r in rows])),
                "mean_alignment": float(mean([safe_float(r.get("mean_alignment", 0.0), 0.0) for r in rows])),
            }
        )
    return out


def build_summary_rows(
    run_rows: List[Dict[str, Any]],
    example_rows: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    baseline_agg_rows: List[Dict[str, Any]],
    latency_quality_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    run_lookup = {r["run_name"]: r for r in run_rows}
    ex_lookup = {(e["run_id"], e["example_id"]): e for e in example_rows}

    random_lookup: Dict[Tuple[str, str, str], float] = {}
    for row in baseline_rows:
        if row["method"] == "Random":
            random_lookup[(row["run_id"], row["example_id"], ratio_key(row["removal_ratio"]))] = safe_float(
                row["drop"], 0.0
            )

    out: List[Dict[str, Any]] = []

    for row in baseline_rows:
        run_id = row["run_id"]
        example_id = row["example_id"]
        rkey = ratio_key(row["removal_ratio"])
        rand_drop = random_lookup.get((run_id, example_id, rkey), 0.0)
        gap = safe_float(row["drop"], 0.0) - rand_drop
        rr = run_lookup.get(run_id, {})
        ex = ex_lookup.get((run_id, example_id), {})
        out.append(
            {
                "row_type": "example_baseline",
                "run_id": run_id,
                "config_id": rr.get("group", ""),
                "seed": rr.get("seed", ""),
                "mode": rr.get("mode", ""),
                "example_id": example_id,
                "method": row["method"],
                "removal_ratio": rkey,
                "p0": safe_float(row.get("p0", 0.0), 0.0),
                "pr": safe_float(row.get("pr", 0.0), 0.0),
                "drop": safe_float(row.get("drop", 0.0), 0.0),
                "gap_vs_random": gap,
                "seq_len": safe_int(row.get("seq_len", 0), 0),
                "target_token": row.get("target_token", ""),
                "latency_ms_total": safe_float(row.get("latency_ms_total", 0.0), 0.0),
                "success_strict_alignment": safe_int(ex.get("success_strict", 0), 0),
                "success_dsa": safe_int(ex.get("success_dsa", 0), 0),
                "mean_alignment_run": safe_float(rr.get("mean_alignment", 0.0), 0.0),
                "success_rate_strict_run": safe_float(rr.get("success_rate_strict", 0.0), 0.0),
                "success_rate_dsa_run": safe_float(rr.get("success_rate_dsa", 0.0), 0.0),
                "latency_p95_ms_run": safe_float(rr.get("latency_p95_ms", 0.0), 0.0),
                "timestamp": row.get("timestamp", ""),
            }
        )

    for row in baseline_agg_rows:
        out.append(
            {
                "row_type": "baseline_aggregate",
                "run_id": "",
                "config_id": "",
                "seed": "",
                "mode": "",
                "example_id": "",
                "method": row.get("method", ""),
                "removal_ratio": ratio_key(row.get("removal_ratio", 0.0)),
                "p0": "",
                "pr": "",
                "drop": safe_float(row.get("mean_drop", 0.0), 0.0),
                "gap_vs_random": safe_float(row.get("mean_gap_vs_random", 0.0), 0.0),
                "seq_len": "",
                "target_token": "",
                "latency_ms_total": "",
                "success_strict_alignment": "",
                "success_dsa": "",
                "mean_alignment_run": "",
                "success_rate_strict_run": "",
                "success_rate_dsa_run": "",
                "latency_p95_ms_run": "",
                "timestamp": "",
            }
        )

    for row in latency_quality_rows:
        out.append(
            {
                "row_type": "latency_quality_config",
                "run_id": row.get("run_id", ""),
                "config_id": row.get("config_id", ""),
                "seed": "",
                "mode": row.get("mode", ""),
                "example_id": "",
                "method": "HETA-Lite",
                "removal_ratio": "0.20",
                "p0": "",
                "pr": "",
                "drop": safe_float(row.get("mean_drop_r20", 0.0), 0.0),
                "gap_vs_random": safe_float(row.get("mean_gap_vs_random_r20", 0.0), 0.0),
                "seq_len": "",
                "target_token": "",
                "latency_ms_total": "",
                "success_strict_alignment": "",
                "success_dsa": safe_float(row.get("success_rate_dsa", 0.0), 0.0),
                "mean_alignment_run": safe_float(row.get("mean_alignment", 0.0), 0.0),
                "success_rate_strict_run": safe_float(row.get("success_rate_strict", 0.0), 0.0),
                "success_rate_dsa_run": safe_float(row.get("success_rate_dsa", 0.0), 0.0),
                "latency_p95_ms_run": safe_float(row.get("latency_p95_ms", 0.0), 0.0),
                "timestamp": row.get("timestamp", ""),
            }
        )

    return out


def cleanup_extra_outputs(out_dir: Path) -> None:
    keep = {
        "summary.csv",
        "baseline_comparison_plot.png",
        "latency_quality_curve.png",
    }
    for p in out_dir.glob("*"):
        if not p.is_file():
            continue
        if p.name in keep:
            continue
        # Remove previous aggregate artifacts to keep output minimal and avoid confusion.
        if p.suffix.lower() in {".csv", ".png", ".json"}:
            try:
                p.unlink()
            except OSError:
                pass


def render_baseline_plots(
    agg_rows: List[Dict[str, Any]],
    output_dir: Path,
    title_prefix: str,
    _target_ratio: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable, skipping baseline plots: {exc}")
        return

    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in agg_rows)]
    if not methods:
        return

    # Single required baseline plot: mean drop vs removal ratio.
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        m_rows = sorted(
            [r for r in agg_rows if r["method"] == method], key=lambda r: ratio_value(r["removal_ratio"])
        )
        xs = [ratio_value(r["removal_ratio"]) for r in m_rows]
        ys = [safe_float(r["mean_drop"], 0.0) for r in m_rows]
        stds = [safe_float(r["std_drop"], 0.0) for r in m_rows]
        ns = [max(1, safe_int(r["n_examples"], 1)) for r in m_rows]
        ci95 = [1.96 * s / np.sqrt(n) for s, n in zip(stds, ns)]
        ax.errorbar(xs, ys, yerr=ci95, marker="o", capsize=3, label=method)
    ax.set_xlabel("Removal Ratio")
    ax.set_ylabel("Mean Probability Drop")
    ax.set_title(f"{title_prefix}: Baseline Comparison")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_comparison_plot.png", dpi=180)
    plt.close(fig)


def render_latency_quality_plots(
    sweep_rows: List[Dict[str, Any]], output_dir: Path, title_prefix: str
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable, skipping latency-quality plots: {exc}")
        return

    if not sweep_rows:
        return

    cfg_rows = collapse_latency_quality_by_config(sweep_rows)
    if len(cfg_rows) < 2:
        only = cfg_rows[0]["config_id"] if cfg_rows else "none"
        print(
            f"[warn] latency-quality curve has {len(cfg_rows)} config point(s); need >=2 configs. current={only}"
        )

    # Single required latency-quality tradeoff plot.
    fig, ax = plt.subplots(figsize=(9, 5))
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    for row in cfg_rows:
        x = safe_float(row.get("latency_p95_ms", 0.0), 0.0)
        y = safe_float(row.get("success_rate_strict", 0.0), 0.0)
        label = row.get("config_id", "")
        xs.append(x)
        ys.append(y)
        labels.append(label)
        ax.scatter([x], [y], s=55)
    if len(xs) > 1:
        order = np.argsort(np.asarray(xs, dtype=np.float64))
        xs_ord = [xs[i] for i in order]
        ys_ord = [ys[i] for i in order]
        ax.plot(xs_ord, ys_ord, linestyle="--", alpha=0.6)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Latency p95 (ms)")
    ax.set_ylabel("Quality (Success Rate Strict)")
    ax.set_title(f"{title_prefix}: Latency-Quality Curve")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "latency_quality_curve.png", dpi=180)
    plt.close(fig)


def build_legacy_plot_rows(
    run_rows: List[Dict[str, Any]], baseline_rows: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Legacy summary tables used by prior report scripts.
    by_group_f: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_group_a: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rr in run_rows:
        group = rr.get("group", "")
        by_group_f[group]["success_rate_strict"].append(safe_float(rr.get("success_rate_strict", 0.0), 0.0))
        by_group_f[group]["success_rate_dsa"].append(safe_float(rr.get("success_rate_dsa", 0.0), 0.0))
        by_group_a[group]["mean_alignment"].append(safe_float(rr.get("mean_alignment", 0.0), 0.0))
        by_group_a[group]["mean_alignment_dsa"].append(safe_float(rr.get("mean_alignment_dsa", 0.0), 0.0))

    plot_faithfulness_rows: List[Dict[str, Any]] = []
    for group, metric_map in sorted(by_group_f.items()):
        for metric, values in metric_map.items():
            if not values:
                continue
            plot_faithfulness_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "mean": float(mean(values)),
                    "std": float(pstdev(values)) if len(values) > 1 else 0.0,
                    "n_runs": len(values),
                }
            )

    plot_alignment_rows: List[Dict[str, Any]] = []
    for group, metric_map in sorted(by_group_a.items()):
        for metric, values in metric_map.items():
            if not values:
                continue
            plot_alignment_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "mean": float(mean(values)),
                    "std": float(pstdev(values)) if len(values) > 1 else 0.0,
                    "n_runs": len(values),
                }
            )

    drops_by_key: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
    topk_by_group_ratio: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    random_by_group_ratio: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    group_lookup = {rr["run_name"]: rr.get("group", rr["run_name"]) for rr in run_rows}

    for br in baseline_rows:
        group = group_lookup.get(br["run_id"], br["run_id"])
        ratio = ratio_key(br["removal_ratio"])
        method = br["method"]
        drop = safe_float(br["drop"], 0.0)
        if method == "HETA-Lite":
            topk_by_group_ratio[(group, ratio)].append(drop)
            drops_by_key[(br["run_id"], br["example_id"], ratio)]["heta"] = drop
        elif method == "Random":
            random_by_group_ratio[(group, ratio)].append(drop)
            drops_by_key[(br["run_id"], br["example_id"], ratio)]["random"] = drop

    gap_by_group_ratio: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for (run_id, example_id, ratio), vals in drops_by_key.items():
        if "heta" in vals and "random" in vals:
            group = group_lookup.get(run_id, run_id)
            gap_by_group_ratio[(group, ratio)].append(vals["heta"] - vals["random"])

    plot_probdrop_rows: List[Dict[str, Any]] = []
    groups_ratios = sorted(set(topk_by_group_ratio.keys()) | set(random_by_group_ratio.keys()))
    for group, ratio in groups_ratios:
        topk_vals = topk_by_group_ratio.get((group, ratio), [])
        rand_vals = random_by_group_ratio.get((group, ratio), [])
        gap_vals = gap_by_group_ratio.get((group, ratio), [])
        n_examples = min(len(topk_vals), len(rand_vals)) if (topk_vals and rand_vals) else max(len(topk_vals), len(rand_vals))
        plot_probdrop_rows.append(
            {
                "group": group,
                "ratio": ratio,
                "topk_mean": float(mean(topk_vals)) if topk_vals else 0.0,
                "topk_std": float(pstdev(topk_vals)) if len(topk_vals) > 1 else 0.0,
                "random_mean": float(mean(rand_vals)) if rand_vals else 0.0,
                "random_std": float(pstdev(rand_vals)) if len(rand_vals) > 1 else 0.0,
                "gap_mean": float(mean(gap_vals)) if gap_vals else 0.0,
                "gap_std": float(pstdev(gap_vals)) if len(gap_vals) > 1 else 0.0,
                "n_examples": n_examples,
            }
        )

    plot_probdrop_rows.sort(key=lambda r: (r["group"], ratio_value(r["ratio"])))
    return plot_faithfulness_rows, plot_probdrop_rows, plot_alignment_rows


def render_legacy_plots(
    plot_faithfulness_rows: List[Dict[str, Any]],
    plot_probdrop_rows: List[Dict[str, Any]],
    plot_alignment_rows: List[Dict[str, Any]],
    output_dir: Path,
    title_prefix: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable, skipping legacy plots: {exc}")
        return

    if plot_faithfulness_rows:
        groups = sorted({r["group"] for r in plot_faithfulness_rows})
        metrics = ["success_rate_strict", "success_rate_dsa"]
        width = 0.35
        x = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, metric in enumerate(metrics):
            vals = []
            errs = []
            for g in groups:
                row = next((r for r in plot_faithfulness_rows if r["group"] == g and r["metric"] == metric), None)
                vals.append(safe_float(row["mean"], 0.0) if row else 0.0)
                errs.append(safe_float(row["std"], 0.0) if row else 0.0)
            ax.bar(x + (i - 0.5) * width, vals, width=width, yerr=errs, capsize=3, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Rate")
        ax.set_title(f"{title_prefix}: Attribution Faithfulness Comparison")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "attribution_faithfulness_comparison.png", dpi=180)
        plt.close(fig)

    if plot_probdrop_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        groups = sorted({r["group"] for r in plot_probdrop_rows})
        for group in groups:
            g_rows = [r for r in plot_probdrop_rows if r["group"] == group]
            g_rows = sorted(g_rows, key=lambda r: ratio_value(r["ratio"]))
            xs = [ratio_value(r["ratio"]) for r in g_rows]
            topk = [safe_float(r["topk_mean"], 0.0) for r in g_rows]
            rand = [safe_float(r["random_mean"], 0.0) for r in g_rows]
            ax.plot(xs, topk, marker="o", label=f"{group} Top-K")
            ax.plot(xs, rand, marker="s", linestyle="--", label=f"{group} Random")
        ax.set_xlabel("Removal Ratio")
        ax.set_ylabel("Mean Probability Drop")
        ax.set_title(f"{title_prefix}: Probability Drop vs Removal Ratio")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "probability_drop_vs_removal_ratio.png", dpi=180)
        plt.close(fig)

    if plot_alignment_rows:
        groups = sorted({r["group"] for r in plot_alignment_rows})
        metrics = ["mean_alignment", "mean_alignment_dsa"]
        width = 0.35
        x = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, metric in enumerate(metrics):
            vals = []
            errs = []
            for g in groups:
                row = next((r for r in plot_alignment_rows if r["group"] == g and r["metric"] == metric), None)
                vals.append(safe_float(row["mean"], 0.0) if row else 0.0)
                errs.append(safe_float(row["std"], 0.0) if row else 0.0)
            ax.bar(x + (i - 0.5) * width, vals, width=width, yerr=errs, capsize=3, label=metric)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.set_ylabel("Alignment")
        ax.set_title(f"{title_prefix}: Alignment Comparison")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "alignment_comparison_bar.png", dpi=180)
        plt.close(fig)


def read_external_baseline_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = normalize_method_name(row.get("method", ""))
            rows.append(
                {
                    "run_id": row.get("run_id", ""),
                    "example_id": row.get("example_id", ""),
                    "method": method,
                    "removal_ratio": ratio_value(row.get("removal_ratio", 0.0)),
                    "p0": safe_float(row.get("p0", 0.0)),
                    "pr": safe_float(row.get("pr", 0.0)),
                    "drop": safe_float(row.get("drop", 0.0)),
                    "target_token": row.get("target_token", ""),
                    "seq_len": safe_int(row.get("seq_len", 0)),
                    "timestamp": row.get("timestamp", ""),
                    "latency_ms_total": safe_float(row.get("latency_ms_total", 0.0)),
                    "latency_ms_attribution": safe_float(row.get("latency_ms_attribution", 0.0)),
                    "latency_ms_forward": safe_float(row.get("latency_ms_forward", 0.0)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.no_timestamp else Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(root, args.run_glob)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found under {root} with pattern {args.run_glob}")

    run_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        group = normalize_group_name(run_name, args.group_regex)
        summary_path = run_dir / "summary.json"
        results_path = run_dir / "results.jsonl"

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metric_config = summary.get("metric_config", {}) or {}
        rows = list(iter_jsonl(results_path))
        fallback_seed = rows[0].get("seed", "") if rows else ""
        seed = extract_seed(run_name, fallback_seed)

        lat_values = [safe_float(r.get("latency_ms", 0.0), 0.0) for r in rows]
        p50 = percentile(lat_values, 50.0)
        p95 = percentile(lat_values, 95.0)

        first = rows[0] if rows else {}
        quality = str(first.get("quality", "")).strip()
        hessian_samples = first.get("hvp_samples", metric_config.get("hvp_samples", ""))
        window_mode = first.get(
            "window_mode",
            f"ctx{first.get('max_context_tokens', metric_config.get('max_context_tokens', 'na'))}",
        )
        config_id = (
            f"{metric_config.get('mode', first.get('mode', ''))}"
            f"_q{quality or 'na'}"
            f"_h{hessian_samples}"
            f"_w{window_mode}"
        )
        run_rows.append(
            {
                "run_name": run_name,
                "group": group,
                "config_id": config_id,
                "seed": seed,
                "mode": metric_config.get("mode", first.get("mode", "")),
                "quality": quality,
                "target_source": metric_config.get("target_source", first.get("target_source", "")),
                "fusion_mode": metric_config.get("fusion_mode", first.get("fusion_mode", "")),
                "important_selection": metric_config.get(
                    "important_selection", first.get("important_selection", "")
                ),
                "filter_structural_tokens": metric_config.get(
                    "filter_structural_tokens", first.get("filter_structural_tokens", "")
                ),
                "n_examples": safe_int(summary.get("n_examples", len(rows))),
                "success_rate_strict": safe_float(summary.get("success_rate_strict", 0.0)),
                "success_rate_dsa": safe_float(summary.get("success_rate_dsa", 0.0)),
                "success_rate_faithfulness": safe_float(
                    summary.get("success_rate_faithfulness", 0.0)
                ),
                "mean_alignment": safe_float(summary.get("mean_alignment", 0.0)),
                "median_alignment": safe_float(summary.get("median_alignment", 0.0)),
                "mean_alignment_dsa": safe_float(summary.get("mean_alignment_dsa", 0.0)),
                "median_alignment_dsa": safe_float(summary.get("median_alignment_dsa", 0.0)),
                "mean_evidence_share_en": safe_float(summary.get("mean_evidence_share_en", 0.0)),
                "mean_narrative_share_en": safe_float(summary.get("mean_narrative_share_en", 0.0)),
                "valid_context_rate": safe_float(summary.get("valid_context_rate", 0.0)),
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "hessian_samples": hessian_samples,
                "layer_sampling": first.get("layer_sampling", metric_config.get("layer_sampling", "default")),
                "window_mode": window_mode,
                "kl_budget": first.get("kl_budget", metric_config.get("kl_budget", "full")),
                "timestamp": datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(timespec="seconds"),
                "notes": f"mode={metric_config.get('mode', '')}",
            }
        )

        for rec in rows:
            seq_counts = rec.get("segment_token_counts", {}) or {}
            seq_len = sum(safe_int(v, 0) for v in seq_counts.values())
            if seq_len <= 0:
                seq_len = safe_int(rec.get("seq_len", 0), 0)

            example_rows.append(
                {
                    "run_id": run_name,
                    "group": group,
                    "seed": seed,
                    "example_id": rec.get("id", ""),
                    "mode": rec.get("mode", metric_config.get("mode", "")),
                    "model": rec.get("model", ""),
                    "success_strict": int(bool(rec.get("success_strict", False))),
                    "success_dsa": int(bool(rec.get("success_dsa", False))),
                    "faithfulness_pass_0.10": int(bool(rec.get("faithfulness_pass_0.10", False))),
                    "alignment": safe_float(rec.get("alignment", 0.0)),
                    "alignment_dsa": safe_float(rec.get("alignment_dsa", 0.0)),
                    "evidence_share_en": safe_float(rec.get("evidence_share_en", 0.0)),
                    "narrative_share_en": safe_float(rec.get("narrative_share_en", 0.0)),
                    "latency_ms_total": safe_float(rec.get("latency_ms", 0.0)),
                    "onset_token_text": rec.get("onset_token_text", ""),
                    "target_prob_orig": safe_float(rec.get("target_prob_orig", 0.0)),
                    "faithfulness_auc_gap": safe_float(rec.get("faithfulness_auc_gap", 0.0)),
                    "truncated_tokens": safe_int(rec.get("truncated_tokens", 0)),
                    "timestamp": datetime.fromtimestamp(results_path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )

            p0 = safe_float(rec.get("target_prob_orig", 0.0))
            target_token = rec.get("onset_token_text", "")
            curves = collect_method_curves(rec)
            for method, ratio_map in curves.items():
                for rk, vals in ratio_map.items():
                    baseline_rows.append(
                        {
                            "run_id": run_name,
                            "example_id": rec.get("id", ""),
                            "method": method,
                            "removal_ratio": ratio_value(rk),
                            "p0": p0,
                            "pr": safe_float(vals.get("pr", p0 - safe_float(vals.get("drop", 0.0)))),
                            "drop": safe_float(vals.get("drop", 0.0)),
                            "target_token": target_token,
                            "seq_len": seq_len,
                            "timestamp": datetime.fromtimestamp(results_path.stat().st_mtime).isoformat(timespec="seconds"),
                            "latency_ms_total": safe_float(rec.get("latency_ms", 0.0)),
                            "latency_ms_attribution": safe_float(rec.get("latency_ms", 0.0)),
                            "latency_ms_forward": 0.0,
                        }
                    )

    # Merge extra external baseline curves.
    for extra in args.extra_baseline_csv:
        extra_path = Path(extra)
        if extra_path.exists():
            baseline_rows.extend(read_external_baseline_csv(extra_path))
        else:
            print(f"[warn] extra baseline CSV not found: {extra_path}")

    if not baseline_rows:
        raise RuntimeError(
            "No baseline removal data found. Ensure run results contain prob_drop_* fields or pass --extra_baseline_csv."
        )

    method_present = {row["method"] for row in baseline_rows}
    required_methods = {"Random", "Attention-Rollout", "Integrated Gradients", "HETA-Lite"}
    missing = sorted(required_methods - method_present)
    if missing:
        msg = f"Missing baseline methods in aggregated data: {missing}"
        if args.strict_methods:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")

    baseline_agg_rows = aggregate_baseline_curves(baseline_rows)
    latency_quality_rows = aggregate_latency_quality(run_rows, baseline_rows, args.target_ratio)

    summary_rows = build_summary_rows(
        run_rows=run_rows,
        example_rows=example_rows,
        baseline_rows=baseline_rows,
        baseline_agg_rows=baseline_agg_rows,
        latency_quality_rows=latency_quality_rows,
    )

    cleanup_extra_outputs(out_dir)
    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        [
            "row_type",
            "run_id",
            "config_id",
            "seed",
            "mode",
            "example_id",
            "method",
            "removal_ratio",
            "p0",
            "pr",
            "drop",
            "gap_vs_random",
            "seq_len",
            "target_token",
            "latency_ms_total",
            "success_strict_alignment",
            "success_dsa",
            "mean_alignment_run",
            "success_rate_strict_run",
            "success_rate_dsa_run",
            "latency_p95_ms_run",
            "timestamp",
        ],
    )

    render_baseline_plots(baseline_agg_rows, out_dir, args.title_prefix, args.target_ratio)
    render_latency_quality_plots(latency_quality_rows, out_dir, args.title_prefix)

    print(f"[done] runs={len(run_dirs)}")
    print(f"[done] output_dir={out_dir}")
    print(f"[done] summary.csv={out_dir / 'summary.csv'}")
    print(f"[done] baseline_plot={out_dir / 'baseline_comparison_plot.png'}")
    print(f"[done] latency_plot={out_dir / 'latency_quality_curve.png'}")


if __name__ == "__main__":
    main()
