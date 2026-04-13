from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import signal
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heta_batch_runner import MODEL_OPTIONS, get_model_id, load_attributor, run_one_example


STOP_REQUESTED = False
EPS = 1e-8

# Keep only representative configurations to reduce experiment-mode sprawl.
EXPERIMENT_MODES: Dict[str, Dict[str, Any]] = {
    "fidelity": {
        "quality": "accurate",
        "masking": "drop",
        "target_source": "generated",
        "fusion_mode": "paper",
        "important_selection": "combined",
        "packing_mode": "evidence_first",
        "mt_floor": 0.0,
        "disable_structural_filter": False,
        "max_context_tokens": 1024,
        "hvp_samples": 1,
    },
    "curated_eval": {
        "quality": "accurate",
        "masking": "drop",
        "target_source": "generated",
        "fusion_mode": "paper",
        "important_selection": "combined",
        "packing_mode": "evidence_first",
        "mt_floor": 0.0,
        "disable_structural_filter": False,
        "max_context_tokens": 1024,
        "hvp_samples": 1,
    },
    "realtime": {
        "quality": "balanced",
        "masking": "drop",
        "target_source": "generated",
        "fusion_mode": "log",
        "important_selection": "balanced",
        "packing_mode": "evidence_first",
        "mt_floor": 0.15,
        "disable_structural_filter": False,
        "max_context_tokens": 512,
        "hvp_samples": 1,
    },
}


def compute_alignment_ratio(evidence_mass: float, narrative_mass: float) -> float:
    return float(evidence_mass / (narrative_mass + EPS))


def compute_alignment_logratio(evidence_mass: float, narrative_mass: float) -> float:
    return float(math.log((evidence_mass + EPS) / (narrative_mass + EPS)))


def compute_dynamic_delta(evidence_token_count: int, coeff: float, floor: float) -> float:
    count = max(1, int(evidence_token_count))
    return float(max(float(floor), float(coeff) / math.sqrt(count)))


def parse_removal_ratios(raw: str) -> List[float]:
    ratios: List[float] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = float(part)
        except ValueError:
            continue
        if 0.0 < val < 1.0:
            ratios.append(val)
    ratios = sorted(set(ratios))
    return ratios or [0.05, 0.10, 0.20]


def _request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[signal] received {signum}; will stop after current example.", flush=True)


def install_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _request_stop)
        except Exception:
            pass


def write_checkpoint(
    checkpoint_path: Path,
    input_jsonl: Path,
    output_dir: Path,
    processed_total: int,
    processed_new: int,
    success_total: int,
    alignment_sum_total: float,
    last_id: str,
) -> None:
    payload = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "processed_total": int(processed_total),
        "processed_new_this_run": int(processed_new),
        "success_total": int(success_total),
        "alignment_sum_total": float(alignment_sum_total),
        "last_id": last_id,
    }
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HETA faithfulness batch test on HotpotQA.")
    parser.add_argument(
        "--mode",
        choices=sorted(EXPERIMENT_MODES.keys()),
        default="fidelity",
        help="Experiment profile. Keeps runs consistent and avoids mode sprawl.",
    )
    parser.add_argument("--input_jsonl", required=True, help="Path to converted HETA-style JSONL.")
    parser.add_argument("--output_dir", required=True, help="Directory for JSONL/CSV/summary outputs.")
    parser.add_argument(
        "--model",
        default="",
        help="Model label or model id. If omitted, project default model is used.",
    )
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument(
        "--masking",
        choices=["drop", "unk", "zero_embed"],
        default="zero_embed",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
    )
    parser.add_argument("--target_k", type=int, default=1)
    parser.add_argument(
        "--target_source",
        choices=["generated", "gold_answer", "auto"],
        default="generated",
        help="Target token source for k=1. Paper setting uses generated onset token.",
    )
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument(
        "--hvp_samples",
        type=int,
        default=1,
        help="Hutchinson samples for Hessian term S (lower = less VRAM).",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=0,
        help="Max context tokens for attribution. 0 = auto by quality (fast=256, balanced=384, accurate=512).",
    )
    parser.add_argument(
        "--important_top_fraction",
        type=float,
        default=0.2,
        help="Top fraction of paragraph tokens used as model-selected important tokens for DSA-style mass.",
    )
    parser.add_argument(
        "--important_selection",
        choices=["combined", "balanced"],
        default="combined",
        help="How top important tokens are selected across narrative/evidence segments.",
    )
    parser.add_argument(
        "--fusion_mode",
        choices=["paper", "log"],
        default="paper",
        help="Final fusion mode for combining MT/S/KL.",
    )
    parser.add_argument(
        "--mt_floor",
        type=float,
        default=0.05,
        help="Lower bound for MT in paper_floor/log fusion to avoid hard zeroing.",
    )
    parser.add_argument(
        "--margin_delta",
        type=float,
        default=0.05,
        help="Relaxed success margin delta: evidence >= narrative + delta.",
    )
    parser.add_argument(
        "--ratio_threshold",
        type=float,
        default=1.2,
        help="Relaxed success ratio threshold: evidence/(narrative+eps) >= ratio.",
    )
    parser.add_argument(
        "--valid_evidence_min_tokens",
        type=int,
        default=20,
        help="Minimum evidence token count for valid_by_token_count.",
    )
    parser.add_argument(
        "--dynamic_delta_coeff",
        type=float,
        default=0.1,
        help="Adaptive delta coefficient in coeff/sqrt(evidence_token_count).",
    )
    parser.add_argument(
        "--dynamic_delta_floor",
        type=float,
        default=0.02,
        help="Adaptive delta floor lower bound.",
    )
    parser.add_argument(
        "--packing_mode",
        choices=["evidence_first"],
        default="evidence_first",
        help="Context truncation packing mode before attribution.",
    )
    parser.add_argument(
        "--narrative_cap_ratio",
        type=float,
        default=1.0,
        help="Evidence-first mode: narrative kept tokens <= ratio * kept evidence tokens.",
    )
    parser.add_argument(
        "--narrative_cap_min",
        type=int,
        default=32,
        help="Evidence-first mode: minimum narrative cap when truncation is active.",
    )
    parser.add_argument(
        "--narrative_cap_max",
        type=int,
        default=192,
        help="Evidence-first mode: maximum narrative cap when truncation is active.",
    )
    parser.add_argument(
        "--question_min_tokens",
        type=int,
        default=24,
        help="Evidence-first mode: reserve at least this many question tokens when truncating.",
    )
    parser.add_argument(
        "--disable_structural_filter",
        action="store_true",
        help="Disable filtering of structural marker/punctuation tokens during important-token selection.",
    )
    parser.add_argument(
        "--disable_removal_eval",
        action="store_true",
        help="Disable Top-K vs random removal probability-drop evaluation.",
    )
    parser.add_argument(
        "--removal_ratios",
        default="0.05,0.10,0.20",
        help="Comma-separated removal ratios used for faithfulness curve.",
    )
    parser.add_argument(
        "--random_removal_trials",
        type=int,
        default=5,
        help="Random trials per ratio for random-removal baseline.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] Skip bad JSON at line {line_no}: {exc}")


def load_completed_ids(results_jsonl: Path) -> Tuple[Set[str], int, int, float]:
    ids: Set[str] = set()
    n = 0
    n_success = 0
    sum_alignment = 0.0
    if not results_jsonl.exists():
        return ids, n, n_success, sum_alignment
    for rec in iter_jsonl(results_jsonl):
        rec_id = str(rec.get("id", ""))
        if rec_id:
            ids.add(rec_id)
        has_success = ("success_strict" in rec) or ("success" in rec)
        if has_success and "alignment" in rec:
            n += 1
            success_flag = bool(rec.get("success_strict", rec.get("success", False)))
            if success_flag:
                n_success += 1
            sum_alignment += float(rec.get("alignment", 0.0))
    return ids, n, n_success, sum_alignment


def build_top_tokens(tokens: List[str], final_scores: List[float], top_n: int = 8) -> List[Dict[str, Any]]:
    if not tokens or not final_scores:
        return []
    arr = np.asarray(final_scores, dtype=np.float64)
    if arr.size == 0:
        return []
    top_idx = np.argsort(arr)[::-1][:top_n]
    out: List[Dict[str, Any]] = []
    for idx in top_idx.tolist():
        score = float(arr[idx])
        if score <= 0:
            continue
        out.append({"index": int(idx), "token": tokens[idx], "score": score})
    return out


def build_segment_token_counts(segment_token_spans: Dict[str, List[int] | Tuple[int, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k in ("narrative", "evidence", "question"):
        span = segment_token_spans.get(k, (0, 0))
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            start, end = 0, 0
        else:
            start, end = int(span[0]), int(span[1])
        out[k] = max(0, end - start)
    return out


def segment_mass_from_scores(
    scores: List[float], segment_spans: Dict[str, List[int] | Tuple[int, int]]
) -> Dict[str, float]:
    arr = np.abs(np.asarray(scores, dtype=np.float64))
    masses: Dict[str, float] = {}
    for name in ("narrative", "evidence", "question"):
        span = segment_spans.get(name, (0, 0))
        start, end = int(span[0]), int(span[1])
        lo = max(0, start)
        hi = min(arr.shape[0], end)
        masses[name] = float(arr[lo:hi].sum()) if hi > lo else 0.0
    denom = masses["narrative"] + masses["evidence"] + masses["question"]
    if denom > 1e-12:
        masses = {k: float(v / denom) for k, v in masses.items()}
    return masses


def is_structural_token(token_text: str) -> bool:
    text = (token_text or "").strip()
    if not text:
        return True
    if re.search(r"[A-Za-z0-9]", text) is None:
        return True
    lower = text.lower()
    if lower in {"<s>", "</s>", "[narrativeqa]", "[sciq]", "[question]"}:
        return True
    if text in {"[", "]", "[N", "[E", "[Q", "[T"}:
        return True
    if text.startswith("[") and len(text) <= 3:
        return True
    return False


def build_candidate_indices(
    segment_spans: Dict[str, List[int] | Tuple[int, int]],
    arr_len: int,
    token_texts: List[str] | None = None,
    filter_structural_tokens: bool = True,
) -> List[int]:
    candidates: List[int] = []
    for name in ("narrative", "evidence"):
        span = segment_spans.get(name, (0, 0))
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        lo = max(0, start)
        hi = min(arr_len, end)
        if hi <= lo:
            continue
        raw_indices = list(range(lo, hi))
        if filter_structural_tokens and token_texts:
            filtered = [
                idx
                for idx in raw_indices
                if idx < len(token_texts) and not is_structural_token(token_texts[idx])
            ]
            candidates.extend(filtered if filtered else raw_indices)
        else:
            candidates.extend(raw_indices)
    return sorted(set(candidates))


def select_important_indices(
    scores: List[float],
    segment_spans: Dict[str, List[int] | Tuple[int, int]],
    top_fraction: float,
    selection_mode: str = "balanced",
    token_texts: List[str] | None = None,
    filter_structural_tokens: bool = True,
) -> List[int]:
    arr = np.abs(np.asarray(scores, dtype=np.float64))
    if arr.size == 0:
        return []

    segment_indices: Dict[str, List[int]] = {"narrative": [], "evidence": []}
    for name in ("narrative", "evidence"):
        span = segment_spans.get(name, (0, 0))
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        lo = max(0, start)
        hi = min(arr.shape[0], end)
        if hi <= lo:
            continue
        indices = build_candidate_indices(
            {name: (lo, hi)},
            arr_len=arr.shape[0],
            token_texts=token_texts,
            filter_structural_tokens=filter_structural_tokens,
        )
        segment_indices[name] = indices

    narrative_candidates = segment_indices["narrative"]
    evidence_candidates = segment_indices["evidence"]
    all_candidates = narrative_candidates + evidence_candidates
    if not all_candidates:
        return []

    frac = min(1.0, max(0.01, float(top_fraction)))
    k_total = max(1, int(round(len(all_candidates) * frac)))
    mode = (selection_mode or "balanced").strip().lower()

    if mode != "balanced" or not narrative_candidates or not evidence_candidates:
        cand_arr = np.asarray(all_candidates, dtype=np.int64)
        cand_scores = arr[cand_arr]
        order = np.argsort(cand_scores)[::-1][:k_total]
        selected = cand_arr[order].tolist()
        selected.sort()
        return selected

    k_evidence = min(len(evidence_candidates), max(1, k_total // 2))
    k_narrative = min(len(narrative_candidates), max(1, k_total - k_evidence))
    while (k_evidence + k_narrative) < k_total:
        add_evidence = (len(evidence_candidates) - k_evidence) >= (
            len(narrative_candidates) - k_narrative
        )
        if add_evidence and k_evidence < len(evidence_candidates):
            k_evidence += 1
        elif k_narrative < len(narrative_candidates):
            k_narrative += 1
        elif k_evidence < len(evidence_candidates):
            k_evidence += 1
        else:
            break

    e_arr = np.asarray(evidence_candidates, dtype=np.int64)
    n_arr = np.asarray(narrative_candidates, dtype=np.int64)
    e_order = np.argsort(arr[e_arr])[::-1][:k_evidence]
    n_order = np.argsort(arr[n_arr])[::-1][:k_narrative]
    selected = e_arr[e_order].tolist() + n_arr[n_order].tolist()
    if len(selected) < k_total:
        remain_pool = sorted(
            set(all_candidates).difference(selected), key=lambda idx: arr[idx], reverse=True
        )
        selected.extend(remain_pool[: k_total - len(selected)])
    selected.sort()
    return selected


def segment_mass_on_selected(
    scores: List[float],
    segment_spans: Dict[str, List[int] | Tuple[int, int]],
    selected_indices: List[int],
) -> Dict[str, float]:
    arr = np.abs(np.asarray(scores, dtype=np.float64))
    sel = set(int(i) for i in selected_indices if 0 <= int(i) < arr.shape[0])
    masses: Dict[str, float] = {}
    for name in ("narrative", "evidence", "question"):
        span = segment_spans.get(name, (0, 0))
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            masses[name] = 0.0
            continue
        start, end = int(span[0]), int(span[1])
        lo = max(0, start)
        hi = min(arr.shape[0], end)
        if hi <= lo:
            masses[name] = 0.0
            continue
        masses[name] = float(sum(arr[idx] for idx in range(lo, hi) if idx in sel))
    denom = masses["narrative"] + masses["evidence"] + masses["question"]
    if denom > 1e-12:
        masses = {k: float(v / denom) for k, v in masses.items()}
    return masses


def selected_token_counts(
    segment_spans: Dict[str, List[int] | Tuple[int, int]], selected_indices: List[int]
) -> Dict[str, int]:
    selected = set(int(i) for i in selected_indices)
    counts: Dict[str, int] = {}
    for name in ("narrative", "evidence", "question"):
        span = segment_spans.get(name, (0, 0))
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            counts[name] = 0
            continue
        start, end = int(span[0]), int(span[1])
        lo = max(0, start)
        hi = max(lo, int(end))
        counts[name] = int(sum(1 for idx in range(lo, hi) if idx in selected))
    return counts


def _sentence_token_spans(tokens: List[str], start: int, end: int) -> List[Tuple[int, int]]:
    lo = max(0, int(start))
    hi = min(len(tokens), int(end))
    if hi <= lo:
        return []
    spans: List[Tuple[int, int]] = []
    cur = lo
    for idx in range(lo, hi):
        tok = tokens[idx] if idx < len(tokens) else ""
        if re.search(r"[.!?]", tok):
            spans.append((cur, idx + 1))
            cur = idx + 1
    if cur < hi:
        spans.append((cur, hi))
    # Merge tiny punctuation-only chunks into previous sentence.
    merged: List[Tuple[int, int]] = []
    for span in spans:
        s, e = span
        text = "".join(tokens[s:e]).strip()
        if merged and (e - s) <= 1 and re.search(r"[A-Za-z0-9]", text or "") is None:
            ps, pe = merged[-1]
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged or [(lo, hi)]


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def compute_dsa_like_metrics(
    scores: List[float],
    tokens: List[str],
    segment_spans: Dict[str, List[int] | Tuple[int, int]],
    answer_text: str,
) -> Dict[str, Any]:
    arr = np.abs(np.asarray(scores, dtype=np.float64))
    if arr.size == 0 or not tokens:
        return {
            "success_dsa": False,
            "alignment_dsa": 0.0,
            "evidence_anchor_mass": 0.0,
            "narrative_anchor_mass": 0.0,
            "evidence_anchor_source": "none",
            "evidence_sentence_count": 0,
            "narrative_sentence_count": 0,
        }

    def sentence_masses(name: str) -> List[Tuple[Tuple[int, int], float, str]]:
        span = segment_spans.get(name, (0, 0))
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            return []
        start, end = int(span[0]), int(span[1])
        out: List[Tuple[Tuple[int, int], float, str]] = []
        for s, e in _sentence_token_spans(tokens, start, end):
            mass = float(arr[s:e].sum()) if e > s else 0.0
            text = "".join(tokens[s:e])
            out.append(((s, e), mass, text))
        return out

    evidence_sents = sentence_masses("evidence")
    narrative_sents = sentence_masses("narrative")
    narrative_anchor_mass = max((m for _, m, _ in narrative_sents), default=0.0)

    answer_norm = _normalize_match_text(answer_text)
    matched_masses: List[float] = []
    if answer_norm:
        for _, mass, text in evidence_sents:
            if answer_norm in _normalize_match_text(text):
                matched_masses.append(float(mass))

    if matched_masses:
        evidence_anchor_mass = float(max(matched_masses))
        evidence_anchor_source = "answer_match"
    else:
        evidence_anchor_mass = float(max((m for _, m, _ in evidence_sents), default=0.0))
        evidence_anchor_source = "max_sentence_fallback"

    alignment_dsa = float(evidence_anchor_mass - narrative_anchor_mass)
    return {
        "success_dsa": bool(alignment_dsa > 0.0),
        "alignment_dsa": alignment_dsa,
        "evidence_anchor_mass": evidence_anchor_mass,
        "narrative_anchor_mass": float(narrative_anchor_mass),
        "evidence_anchor_source": evidence_anchor_source,
        "evidence_sentence_count": int(len(evidence_sents)),
        "narrative_sentence_count": int(len(narrative_sents)),
    }


def _predict_target_probability(model: Any, token_ids: List[int], target_token_id: int) -> float:
    if not token_ids:
        return 0.0
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1, :]
        probs = torch.softmax(logits.to(torch.float32), dim=-1)
    return float(probs[int(target_token_id)].item())


def compute_removal_faithfulness(
    model: Any,
    context_token_ids: List[int],
    target_token_id: int,
    scores: List[float],
    candidate_indices: List[int],
    removal_ratios: List[float],
    random_trials: int,
    rng: random.Random,
) -> Dict[str, Any]:
    clean_context = [int(t) for t in context_token_ids]
    if not clean_context or not candidate_indices or int(target_token_id) < 0:
        return {
            "enabled": False,
            "target_prob_orig": 0.0,
            "ratios": [],
            "topk_drop": {},
            "random_drop": {},
            "drop_gap": {},
            "topk_prob": {},
            "random_prob": {},
            "faithfulness_pass_0.10": False,
            "faithfulness_auc_gap": 0.0,
        }

    arr = np.asarray(scores, dtype=np.float64)
    ranked = sorted(candidate_indices, key=lambda idx: arr[idx], reverse=True)
    ratios = [r for r in removal_ratios if 0.0 < r < 1.0]
    if not ratios:
        ratios = [0.05, 0.10, 0.20]

    p_orig = _predict_target_probability(model, clean_context, target_token_id)
    topk_drop: Dict[str, float] = {}
    random_drop: Dict[str, float] = {}
    drop_gap: Dict[str, float] = {}
    topk_prob: Dict[str, float] = {}
    random_prob: Dict[str, float] = {}

    for ratio in ratios:
        key = f"{ratio:.2f}"
        k = max(1, int(round(len(candidate_indices) * ratio)))
        k = min(k, len(candidate_indices))

        top_remove = set(ranked[:k])
        kept_top = [tok for idx, tok in enumerate(clean_context) if idx not in top_remove]
        if not kept_top:
            kept_top = clean_context[:1]
        p_top = _predict_target_probability(model, kept_top, target_token_id)
        topk_prob[key] = p_top
        topk_drop[key] = float(p_orig - p_top)

        trial_probs: List[float] = []
        for _ in range(max(1, int(random_trials))):
            rand_remove = set(rng.sample(candidate_indices, k=k))
            kept_rand = [tok for idx, tok in enumerate(clean_context) if idx not in rand_remove]
            if not kept_rand:
                kept_rand = clean_context[:1]
            trial_probs.append(_predict_target_probability(model, kept_rand, target_token_id))
        p_rand = float(sum(trial_probs) / len(trial_probs))
        random_prob[key] = p_rand
        random_drop[key] = float(p_orig - p_rand)
        drop_gap[key] = float(topk_drop[key] - random_drop[key])

    sorted_keys = sorted(topk_drop.keys(), key=float)
    auc_gap = 0.0
    if len(sorted_keys) >= 2:
        xs = [float(k) for k in sorted_keys]
        ys = [drop_gap[k] for k in sorted_keys]
        auc_gap = float(np.trapz(np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)))
    elif len(sorted_keys) == 1:
        auc_gap = float(drop_gap[sorted_keys[0]] * float(sorted_keys[0]))

    pass_010 = False
    if "0.10" in drop_gap:
        pass_010 = bool(drop_gap["0.10"] > 0.0)
    elif sorted_keys:
        pass_010 = bool(drop_gap[sorted_keys[0]] > 0.0)

    return {
        "enabled": True,
        "target_prob_orig": float(p_orig),
        "ratios": sorted_keys,
        "topk_drop": topk_drop,
        "random_drop": random_drop,
        "drop_gap": drop_gap,
        "topk_prob": topk_prob,
        "random_prob": random_prob,
        "faithfulness_pass_0.10": pass_010,
        "faithfulness_auc_gap": auc_gap,
    }


def write_summary_and_csv(
    results_jsonl: Path,
    summary_path: Path,
    csv_path: Path,
    margin_delta: float,
    ratio_threshold: float,
    valid_evidence_min_tokens: int,
    dynamic_delta_coeff: float,
    dynamic_delta_floor: float,
    important_top_fraction: float,
    important_selection: str,
    fusion_mode: str,
    mt_floor: float,
    packing_mode: str,
    narrative_cap_ratio: float,
    narrative_cap_min: int,
    narrative_cap_max: int,
    question_min_tokens: int,
    target_source: str,
    filter_structural_tokens: bool,
    mode: str,
    removal_ratios: List[float],
    random_removal_trials: int,
    removal_eval_enabled: bool,
) -> None:
    alignments: List[float] = []
    dsa_alignments: List[float] = []
    logratios: List[float] = []
    ratios: List[float] = []
    n = 0
    n_success_strict = 0
    n_success_margin = 0
    n_success_ratio = 0
    n_success_margin_dynamic = 0
    n_success_margin_005 = 0
    n_success_ratio_12 = 0
    n_success_per_token = 0
    n_success_dsa = 0
    n_success_faithfulness = 0
    n_valid = 0
    n_valid_success_strict = 0
    n_valid_context = 0
    n_valid_context_success_strict = 0
    sum_alignment = 0.0
    sum_ev_share = 0.0
    sum_na_share = 0.0
    sum_ev = 0.0
    sum_na = 0.0
    sum_q = 0.0

    with csv_path.open("w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(
            csv_f,
            fieldnames=[
                "id",
                "mode",
                "seed",
                "model",
                "quality",
                "masking",
                "target_k",
                "target_source",
                "beta",
                "gamma",
                "fusion_mode",
                "mt_floor",
                "max_context_tokens",
                "hvp_samples",
                "success",
                "success_strict",
                "success_margin",
                "success_ratio",
                "success_margin_dynamic",
                "success_margin_0.05",
                "success_ratio_1.2",
                "alignment",
                "alignment_raw",
                "alignment_logratio",
                "alignment_ratio",
                "alignment_dsa",
                "faithfulness_pass_0.10",
                "faithfulness_auc_gap",
                "target_prob_orig",
                "drop_topk_0.05",
                "drop_topk_0.10",
                "drop_topk_0.20",
                "drop_random_0.05",
                "drop_random_0.10",
                "drop_random_0.20",
                "drop_gap_0.05",
                "drop_gap_0.10",
                "drop_gap_0.20",
                "evidence_mass",
                "narrative_mass",
                "question_mass",
                "evidence_share_en",
                "narrative_share_en",
                "success_dsa",
                "evidence_anchor_mass",
                "narrative_anchor_mass",
                "evidence_anchor_source",
                "success_per_token",
                "evidence_mass_per_token",
                "narrative_mass_per_token",
                "valid_by_token_count",
                "valid_context",
                "evidence_token_count",
                "dynamic_margin_delta",
                "margin_delta",
                "ratio_threshold",
                "latency_ms",
                "onset_token_text",
            ],
        )
        writer.writeheader()

        for rec in iter_jsonl(results_jsonl):
            segment_mass = rec.get("segment_mass", {}) or {}
            ev_raw = float(segment_mass.get("evidence", 0.0))
            na_raw = float(segment_mass.get("narrative", 0.0))
            q_raw = float(segment_mass.get("question", 0.0))
            en_denom = ev_raw + na_raw + EPS
            ev = float(rec.get("evidence_share_en", ev_raw / en_denom))
            na = float(rec.get("narrative_share_en", na_raw / en_denom))
            q = float(q_raw)

            alignment = float(rec.get("alignment", ev - na))
            alignment_ratio = float(rec.get("alignment_ratio", compute_alignment_ratio(ev, na)))
            alignment_logratio = float(
                rec.get("alignment_logratio", compute_alignment_logratio(ev, na))
            )

            counts = rec.get("segment_token_counts", {}) or {}
            if not counts:
                counts = build_segment_token_counts(rec.get("segment_token_spans", {}) or {})
            evidence_token_count = int(counts.get("evidence", 0))
            dynamic_margin_delta = float(
                rec.get(
                    "dynamic_margin_delta",
                    compute_dynamic_delta(
                        evidence_token_count,
                        coeff=dynamic_delta_coeff,
                        floor=dynamic_delta_floor,
                    ),
                )
            )
            valid_by_token_count = bool(
                rec.get(
                    "valid_by_token_count",
                    evidence_token_count >= int(valid_evidence_min_tokens),
                )
            )
            valid_context = bool(rec.get("valid_context", True))
            evidence_mass_per_token = float(
                rec.get(
                    "evidence_mass_per_token",
                    ev_raw / max(1, evidence_token_count),
                )
            )
            narrative_token_count = int(counts.get("narrative", 0))
            narrative_mass_per_token = float(
                rec.get(
                    "narrative_mass_per_token",
                    na_raw / max(1, narrative_token_count),
                )
            )
            success_per_token = bool(
                rec.get(
                    "success_per_token",
                    evidence_mass_per_token > narrative_mass_per_token,
                )
            )

            success_strict = bool(rec.get("success_strict", rec.get("success", ev > na)))
            success_margin = bool(rec.get("success_margin", ev >= na + float(margin_delta)))
            success_ratio = bool(rec.get("success_ratio", alignment_ratio >= float(ratio_threshold)))
            success_margin_dynamic = bool(
                rec.get("success_margin_dynamic", ev >= na + dynamic_margin_delta)
            )
            success_margin_005 = bool(rec.get("success_margin_0.05", ev >= na + 0.05))
            success_ratio_12 = bool(rec.get("success_ratio_1.2", alignment_ratio >= 1.2))
            alignment_dsa = float(rec.get("alignment_dsa", 0.0))
            success_dsa = bool(rec.get("success_dsa", alignment_dsa > 0.0))
            faithfulness_pass = bool(rec.get("faithfulness_pass_0.10", False))
            faithfulness_auc_gap = float(rec.get("faithfulness_auc_gap", 0.0))
            target_prob_orig = float(rec.get("target_prob_orig", 0.0))
            topk_drop = rec.get("prob_drop_topk", {}) or {}
            rand_drop = rec.get("prob_drop_random", {}) or {}
            gap_drop = rec.get("prob_drop_gap", {}) or {}

            writer.writerow(
                {
                    "id": rec.get("id", ""),
                    "mode": rec.get("mode", ""),
                    "seed": rec.get("seed", ""),
                    "model": rec.get("model", ""),
                    "quality": rec.get("quality", ""),
                    "masking": rec.get("masking", ""),
                    "target_k": rec.get("target_k", ""),
                    "target_source": rec.get("target_source", ""),
                    "beta": rec.get("beta", ""),
                    "gamma": rec.get("gamma", ""),
                    "fusion_mode": rec.get("fusion_mode", ""),
                    "mt_floor": rec.get("mt_floor", ""),
                    "max_context_tokens": rec.get("max_context_tokens", ""),
                    "hvp_samples": rec.get("hvp_samples", ""),
                    "success": int(success_strict),
                    "success_strict": int(success_strict),
                    "success_margin": int(success_margin),
                    "success_ratio": int(success_ratio),
                    "success_margin_dynamic": int(success_margin_dynamic),
                    "success_margin_0.05": int(success_margin_005),
                    "success_ratio_1.2": int(success_ratio_12),
                    "alignment": alignment,
                    "alignment_raw": float(rec.get("alignment_raw", ev_raw - na_raw)),
                    "alignment_logratio": alignment_logratio,
                    "alignment_ratio": alignment_ratio,
                    "alignment_dsa": alignment_dsa,
                    "faithfulness_pass_0.10": int(faithfulness_pass),
                    "faithfulness_auc_gap": faithfulness_auc_gap,
                    "target_prob_orig": target_prob_orig,
                    "drop_topk_0.05": float(topk_drop.get("0.05", 0.0)),
                    "drop_topk_0.10": float(topk_drop.get("0.10", 0.0)),
                    "drop_topk_0.20": float(topk_drop.get("0.20", 0.0)),
                    "drop_random_0.05": float(rand_drop.get("0.05", 0.0)),
                    "drop_random_0.10": float(rand_drop.get("0.10", 0.0)),
                    "drop_random_0.20": float(rand_drop.get("0.20", 0.0)),
                    "drop_gap_0.05": float(gap_drop.get("0.05", 0.0)),
                    "drop_gap_0.10": float(gap_drop.get("0.10", 0.0)),
                    "drop_gap_0.20": float(gap_drop.get("0.20", 0.0)),
                    "evidence_mass": ev_raw,
                    "narrative_mass": na_raw,
                    "question_mass": q_raw,
                    "evidence_share_en": ev,
                    "narrative_share_en": na,
                    "success_dsa": int(success_dsa),
                    "evidence_anchor_mass": float(rec.get("evidence_anchor_mass", 0.0)),
                    "narrative_anchor_mass": float(rec.get("narrative_anchor_mass", 0.0)),
                    "evidence_anchor_source": rec.get("evidence_anchor_source", ""),
                    "success_per_token": int(success_per_token),
                    "evidence_mass_per_token": evidence_mass_per_token,
                    "narrative_mass_per_token": narrative_mass_per_token,
                    "valid_by_token_count": int(valid_by_token_count),
                    "valid_context": int(valid_context),
                    "evidence_token_count": evidence_token_count,
                    "dynamic_margin_delta": dynamic_margin_delta,
                    "margin_delta": float(margin_delta),
                    "ratio_threshold": float(ratio_threshold),
                    "latency_ms": rec.get("latency_ms", ""),
                    "onset_token_text": rec.get("onset_token_text", ""),
                }
            )

            n += 1
            n_success_strict += int(success_strict)
            n_success_margin += int(success_margin)
            n_success_ratio += int(success_ratio)
            n_success_margin_dynamic += int(success_margin_dynamic)
            n_success_margin_005 += int(success_margin_005)
            n_success_ratio_12 += int(success_ratio_12)
            n_success_per_token += int(success_per_token)
            n_success_dsa += int(success_dsa)
            n_success_faithfulness += int(faithfulness_pass)
            n_valid += int(valid_by_token_count)
            if valid_by_token_count:
                n_valid_success_strict += int(success_strict)
            n_valid_context += int(valid_context)
            if valid_context:
                n_valid_context_success_strict += int(success_strict)
            sum_alignment += alignment
            sum_ev_share += ev
            sum_na_share += na
            sum_ev += ev_raw
            sum_na += na_raw
            sum_q += q_raw
            alignments.append(alignment)
            dsa_alignments.append(alignment_dsa)
            logratios.append(alignment_logratio)
            ratios.append(alignment_ratio)

    if n == 0:
        summary = {
            "n_examples": 0,
            "success_rate": 0.0,
            "success_rate_strict": 0.0,
            "success_rate_margin": 0.0,
            "success_rate_ratio": 0.0,
            "success_rate_margin_dynamic": 0.0,
            "success_rate_margin_0.05": 0.0,
            "success_rate_ratio_1.2": 0.0,
            "success_rate_per_token": 0.0,
            "success_rate_dsa": 0.0,
            "success_rate_faithfulness": 0.0,
            "valid_rate": 0.0,
            "success_rate_strict_valid": 0.0,
            "valid_context_rate": 0.0,
            "success_rate_strict_valid_context": 0.0,
            "mean_alignment": 0.0,
            "median_alignment": 0.0,
            "mean_alignment_dsa": 0.0,
            "median_alignment_dsa": 0.0,
            "mean_alignment_logratio": 0.0,
            "median_alignment_logratio": 0.0,
            "mean_alignment_ratio": 0.0,
            "median_alignment_ratio": 0.0,
            "mean_evidence_mass": 0.0,
            "mean_narrative_mass": 0.0,
            "mean_question_mass": 0.0,
            "mean_evidence_share_en": 0.0,
            "mean_narrative_share_en": 0.0,
            "alignment_p10": 0.0,
            "alignment_p50": 0.0,
            "alignment_p90": 0.0,
            "alignment_percentiles": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
            "metric_config": {
                "margin_delta": float(margin_delta),
                "ratio_threshold": float(ratio_threshold),
                "valid_evidence_min_tokens": int(valid_evidence_min_tokens),
                "dynamic_delta_coeff": float(dynamic_delta_coeff),
                "dynamic_delta_floor": float(dynamic_delta_floor),
                "important_top_fraction": float(important_top_fraction),
                "important_selection": str(important_selection),
                "fusion_mode": str(fusion_mode),
                "mt_floor": float(mt_floor),
                "packing_mode": str(packing_mode),
                "narrative_cap_ratio": float(narrative_cap_ratio),
                "narrative_cap_min": int(narrative_cap_min),
                "narrative_cap_max": int(narrative_cap_max),
                "question_min_tokens": int(question_min_tokens),
                "target_source": str(target_source),
                "filter_structural_tokens": bool(filter_structural_tokens),
                "mode": str(mode),
                "removal_ratios": [float(x) for x in removal_ratios],
                "random_removal_trials": int(random_removal_trials),
                "removal_eval_enabled": bool(removal_eval_enabled),
            },
        }
    else:
        p10, p50, p90 = np.percentile(np.asarray(alignments, dtype=np.float64), [10, 50, 90]).tolist()
        summary = {
            "n_examples": n,
            "success_rate": n_success_strict / n,
            "success_rate_strict": n_success_strict / n,
            "success_rate_margin": n_success_margin / n,
            "success_rate_ratio": n_success_ratio / n,
            "success_rate_margin_dynamic": n_success_margin_dynamic / n,
            "success_rate_margin_0.05": n_success_margin_005 / n,
            "success_rate_ratio_1.2": n_success_ratio_12 / n,
            "success_rate_per_token": n_success_per_token / n,
            "success_rate_dsa": n_success_dsa / n,
            "success_rate_faithfulness": n_success_faithfulness / n,
            "valid_rate": n_valid / n,
            "success_rate_strict_valid": (n_valid_success_strict / n_valid) if n_valid > 0 else 0.0,
            "valid_context_rate": (n_valid_context / n),
            "success_rate_strict_valid_context": (
                (n_valid_context_success_strict / n_valid_context) if n_valid_context > 0 else 0.0
            ),
            "mean_alignment": sum_alignment / n,
            "median_alignment": statistics.median(alignments),
            "mean_alignment_dsa": statistics.mean(dsa_alignments),
            "median_alignment_dsa": statistics.median(dsa_alignments),
            "mean_alignment_logratio": (sum(logratios) / n),
            "median_alignment_logratio": statistics.median(logratios),
            "mean_alignment_ratio": (sum(ratios) / n),
            "median_alignment_ratio": statistics.median(ratios),
            "mean_evidence_mass": sum_ev / n,
            "mean_narrative_mass": sum_na / n,
            "mean_question_mass": sum_q / n,
            "mean_evidence_share_en": sum_ev_share / n,
            "mean_narrative_share_en": sum_na_share / n,
            "alignment_p10": p10,
            "alignment_p50": p50,
            "alignment_p90": p90,
            "alignment_percentiles": {"p10": p10, "p50": p50, "p90": p90},
            "metric_config": {
                "margin_delta": float(margin_delta),
                "ratio_threshold": float(ratio_threshold),
                "valid_evidence_min_tokens": int(valid_evidence_min_tokens),
                "dynamic_delta_coeff": float(dynamic_delta_coeff),
                "dynamic_delta_floor": float(dynamic_delta_floor),
                "important_top_fraction": float(important_top_fraction),
                "important_selection": str(important_selection),
                "fusion_mode": str(fusion_mode),
                "mt_floor": float(mt_floor),
                "packing_mode": str(packing_mode),
                "narrative_cap_ratio": float(narrative_cap_ratio),
                "narrative_cap_min": int(narrative_cap_min),
                "narrative_cap_max": int(narrative_cap_max),
                "question_min_tokens": int(question_min_tokens),
                "target_source": str(target_source),
                "filter_structural_tokens": bool(filter_structural_tokens),
                "mode": str(mode),
                "removal_ratios": [float(x) for x in removal_ratios],
                "random_removal_trials": int(random_removal_trials),
                "removal_eval_enabled": bool(removal_eval_enabled),
            },
        }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    global STOP_REQUESTED
    args = parse_args()
    install_signal_handlers()

    cli_tokens = sys.argv[1:]

    def _flag_passed(flag: str) -> bool:
        return any(token == flag or token.startswith(flag + "=") for token in cli_tokens)

    preset = EXPERIMENT_MODES.get(args.mode, {})
    preset_flag_map = {
        "quality": "--quality",
        "masking": "--masking",
        "target_source": "--target_source",
        "fusion_mode": "--fusion_mode",
        "important_selection": "--important_selection",
        "packing_mode": "--packing_mode",
        "mt_floor": "--mt_floor",
        "disable_structural_filter": "--disable_structural_filter",
        "max_context_tokens": "--max_context_tokens",
        "hvp_samples": "--hvp_samples",
    }
    for key, value in preset.items():
        flag = preset_flag_map.get(key)
        if flag and not _flag_passed(flag):
            setattr(args, key, value)

    filter_structural_tokens = not bool(args.disable_structural_filter)
    removal_eval_enabled = not bool(args.disable_removal_eval)
    removal_ratios = parse_removal_ratios(args.removal_ratios)
    random_removal_trials = max(1, int(args.random_removal_trials))
    if args.num_workers != 1:
        print("[warn] Only num_workers=1 is currently supported; forcing serial execution.")
    if args.model and args.model not in MODEL_OPTIONS and args.model not in MODEL_OPTIONS.values():
        print("[warn] Model is not in known dropdown options; trying it as raw model id.")

    set_seed(args.seed)
    quality_context_budget = {"fast": 256, "balanced": 384, "accurate": 512}
    effective_context_tokens = (
        int(args.max_context_tokens)
        if int(args.max_context_tokens) > 0
        else int(quality_context_budget[args.quality])
    )
    effective_hvp_samples = max(1, int(args.hvp_samples))
    print(
        "[info] mode={} runtime controls: max_context_tokens={} hvp_samples={} important_top_fraction={} important_selection={} fusion_mode={} mt_floor={} packing_mode={} target_source={} filter_structural_tokens={} removal_eval={} ratios={} random_trials={}".format(
            args.mode,
            effective_context_tokens,
            effective_hvp_samples,
            float(args.important_top_fraction),
            args.important_selection,
            args.fusion_mode,
            float(args.mt_floor),
            args.packing_mode,
            args.target_source,
            filter_structural_tokens,
            removal_eval_enabled,
            ",".join(f"{x:.2f}" for x in removal_ratios),
            random_removal_trials,
        ),
        flush=True,
    )

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    results_jsonl = output_dir / "results.jsonl"
    summary_json = output_dir / "summary.json"
    results_csv = output_dir / "results.csv"
    checkpoint_json = output_dir / "checkpoint.json"

    completed_ids, base_n, base_success, base_alignment_sum = load_completed_ids(results_jsonl)
    max_examples = max(0, int(args.max_examples))
    print(
        "[info] input={} output={} resume_ids={} max_examples={}".format(
            input_jsonl,
            output_dir,
            len(completed_ids),
            max_examples,
        ),
        flush=True,
    )
    if checkpoint_json.exists():
        print(f"[info] found checkpoint file: {checkpoint_json}", flush=True)
    if max_examples and len(completed_ids) >= max_examples:
        print(
            f"[info] Resume found {len(completed_ids)} completed ids; "
            f"already >= max_examples={max_examples}. Rebuilding summary/csv only.",
            flush=True,
        )
        write_summary_and_csv(
            results_jsonl=results_jsonl,
            summary_path=summary_json,
            csv_path=results_csv,
            margin_delta=args.margin_delta,
            ratio_threshold=args.ratio_threshold,
            valid_evidence_min_tokens=args.valid_evidence_min_tokens,
            dynamic_delta_coeff=args.dynamic_delta_coeff,
            dynamic_delta_floor=args.dynamic_delta_floor,
            important_top_fraction=args.important_top_fraction,
            important_selection=args.important_selection,
            fusion_mode=args.fusion_mode,
            mt_floor=args.mt_floor,
            packing_mode=args.packing_mode,
            narrative_cap_ratio=args.narrative_cap_ratio,
            narrative_cap_min=args.narrative_cap_min,
            narrative_cap_max=args.narrative_cap_max,
            question_min_tokens=args.question_min_tokens,
            target_source=args.target_source,
            filter_structural_tokens=filter_structural_tokens,
            mode=args.mode,
            removal_ratios=removal_ratios,
            random_removal_trials=random_removal_trials,
            removal_eval_enabled=removal_eval_enabled,
        )
        return

    processed_new = 0
    success_new = 0
    alignment_sum_new = 0.0
    last_id = ""
    eval_model = None
    if removal_eval_enabled:
        eval_model = load_attributor(get_model_id(args.model)).model

    with results_jsonl.open("a", encoding="utf-8", buffering=1) as out_f:
        for rec in iter_jsonl(input_jsonl):
            if STOP_REQUESTED:
                print("[info] stop requested; exiting loop cleanly.", flush=True)
                break
            if max_examples and len(completed_ids) >= max_examples:
                break

            rec_id = str(rec.get("id", "")).strip()
            if not rec_id:
                rec_id = f"row_{len(completed_ids) + processed_new}"
            if rec_id in completed_ids:
                continue

            segments = rec.get("segments", {}) or {}
            narrative = (segments.get("narrative") or "").strip()
            evidence = (segments.get("evidence") or "").strip()
            question = (segments.get("question") or "").strip()
            if not (narrative and evidence and question):
                print(
                    f"[warn] Skip id={rec_id}: missing narrative/evidence/question.",
                    flush=True,
                )
                continue

            target_k = int(rec.get("meta", {}).get("target_k", args.target_k))
            print(
                "[run] id={} processed={} resume_n={}".format(
                    rec_id, processed_new + 1, base_n + processed_new
                ),
                flush=True,
            )
            one: Dict[str, Any] | None = None
            run_hvp_samples = effective_hvp_samples
            run_context_tokens = effective_context_tokens
            attempt_plan: List[Tuple[int, int]] = []
            # Gradual context backoff keeps more evidence/question tokens than hard-jumping to 256.
            context_candidates: List[int] = [run_context_tokens]
            for c in (768, 512, 384, 256, 192):
                if c < run_context_tokens:
                    context_candidates.append(c)
            seen_ctx: Set[int] = set()
            for ctx in context_candidates:
                ctx_i = max(64, int(ctx))
                if ctx_i in seen_ctx:
                    continue
                seen_ctx.add(ctx_i)
                attempt_plan.append((1, ctx_i))

            last_error: Exception | None = None
            for attempt_idx, (attempt_hvp, attempt_ctx) in enumerate(attempt_plan, start=1):
                try:
                    one = run_one_example(
                        model_name=args.model,
                        narrative=narrative,
                        evidence=evidence,
                        question=question,
                        target_k=target_k,
                        beta=args.beta,
                        gamma=args.gamma,
                        masking=args.masking,
                        quality=args.quality,
                        hvp_samples=attempt_hvp,
                        max_context_tokens=attempt_ctx,
                        answer_text=str(rec.get("answer", "")),
                        fusion_mode=args.fusion_mode,
                        mt_floor=args.mt_floor,
                        target_source_mode=args.target_source,
                        packing_mode=args.packing_mode,
                        narrative_cap_ratio=args.narrative_cap_ratio,
                        narrative_cap_min=args.narrative_cap_min,
                        narrative_cap_max=args.narrative_cap_max,
                        question_min_tokens=args.question_min_tokens,
                    )
                    break
                except KeyboardInterrupt:
                    STOP_REQUESTED = True
                    print("[signal] keyboard interrupt; stopping after checkpoint.", flush=True)
                    break
                except RuntimeError as exc:
                    last_error = exc
                    if "out of memory" not in str(exc).lower():
                        print(f"[warn] id={rec_id} failed: {exc}", flush=True)
                        break
                    clear_cuda_memory()
                    if attempt_idx < len(attempt_plan):
                        next_hvp, next_ctx = attempt_plan[attempt_idx]
                        print(
                            "[warn] id={} OOM on attempt {}/{} (hvp={}, ctx={}); retry hvp={} ctx={}.".format(
                                rec_id,
                                attempt_idx,
                                len(attempt_plan),
                                attempt_hvp,
                                attempt_ctx,
                                next_hvp,
                                next_ctx,
                            ),
                            flush=True,
                        )
                        continue
                    print(
                        "[warn] id={} failed after OOM retries (last hvp={}, ctx={}).".format(
                            rec_id, attempt_hvp, attempt_ctx
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    print(f"[warn] id={rec_id} failed: {exc}", flush=True)
                    break

            if STOP_REQUESTED:
                clear_cuda_memory()
                break

            if one is None:
                clear_cuda_memory()
                continue

            candidate_pool = build_candidate_indices(
                one["segment_token_spans"],
                arr_len=len(one["component_scores"]["final"]),
                token_texts=one["tokens"],
                filter_structural_tokens=filter_structural_tokens,
            )
            important_indices = select_important_indices(
                one["component_scores"]["final"],
                one["segment_token_spans"],
                top_fraction=args.important_top_fraction,
                selection_mode=args.important_selection,
                token_texts=one["tokens"],
                filter_structural_tokens=filter_structural_tokens,
            )
            segment_mass_raw = one["segment_mass"]
            segment_mass = segment_mass_on_selected(
                one["component_scores"]["final"],
                one["segment_token_spans"],
                important_indices,
            )
            if not important_indices:
                segment_mass = segment_mass_raw
            evidence_mass = float(segment_mass.get("evidence", 0.0))
            narrative_mass = float(segment_mass.get("narrative", 0.0))
            question_mass = float(segment_mass.get("question", 0.0))
            alignment_raw = evidence_mass - narrative_mass
            en_denom = evidence_mass + narrative_mass + EPS
            evidence_share_en = evidence_mass / en_denom
            narrative_share_en = narrative_mass / en_denom
            alignment = evidence_share_en - narrative_share_en
            alignment_ratio = compute_alignment_ratio(evidence_share_en, narrative_share_en)
            alignment_logratio = compute_alignment_logratio(evidence_share_en, narrative_share_en)
            success_strict = evidence_share_en > 0.5
            success_margin = evidence_share_en >= (narrative_share_en + float(args.margin_delta))
            success_ratio = alignment_ratio >= float(args.ratio_threshold)

            segment_token_counts = build_segment_token_counts(one["segment_token_spans"])
            selected_counts = selected_token_counts(one["segment_token_spans"], important_indices)
            evidence_token_count = int(segment_token_counts.get("evidence", 0))
            narrative_token_count = int(segment_token_counts.get("narrative", 0))
            dynamic_margin_delta = compute_dynamic_delta(
                evidence_token_count,
                coeff=args.dynamic_delta_coeff,
                floor=args.dynamic_delta_floor,
            )
            success_margin_dynamic = evidence_share_en >= (
                narrative_share_en + dynamic_margin_delta
            )
            valid_by_token_count = evidence_token_count >= int(args.valid_evidence_min_tokens)
            segment_coverage = (
                one.get("run_meta", {}).get("segment_coverage", {})
                if isinstance(one.get("run_meta", {}), dict)
                else {}
            )
            valid_context = bool(one.get("run_meta", {}).get("valid_context", True))

            evidence_mass_per_token = float(
                segment_mass_raw.get("evidence", 0.0) / max(1, evidence_token_count)
            )
            narrative_mass_per_token = float(
                segment_mass_raw.get("narrative", 0.0) / max(1, narrative_token_count)
            )
            success_per_token = bool(evidence_mass_per_token > narrative_mass_per_token)
            dsa_metrics = compute_dsa_like_metrics(
                scores=one["component_scores"]["final"],
                tokens=one["tokens"],
                segment_spans=one["segment_token_spans"],
                answer_text=str(rec.get("answer", "")),
            )
            if removal_eval_enabled and eval_model is not None:
                local_seed = (
                    int(args.seed)
                    + int(processed_new)
                    + sum(ord(ch) for ch in str(rec_id))
                )
                removal_metrics = compute_removal_faithfulness(
                    model=eval_model,
                    context_token_ids=one.get("context_token_ids", []),
                    target_token_id=int(one["run_meta"].get("target_token_id", -1)),
                    scores=one["component_scores"]["final"],
                    candidate_indices=candidate_pool,
                    removal_ratios=removal_ratios,
                    random_trials=random_removal_trials,
                    rng=random.Random(local_seed),
                )
            else:
                removal_metrics = {
                    "enabled": False,
                    "target_prob_orig": 0.0,
                    "topk_drop": {},
                    "random_drop": {},
                    "drop_gap": {},
                    "topk_prob": {},
                    "random_prob": {},
                    "faithfulness_pass_0.10": False,
                    "faithfulness_auc_gap": 0.0,
                }

            out_record = {
                "id": rec_id,
                "model": one["run_meta"]["model_name"],
                "mode": str(args.mode),
                "seed": int(args.seed),
                "removal_eval_enabled": bool(removal_eval_enabled),
                "removal_ratios": [f"{x:.2f}" for x in removal_ratios],
                "random_removal_trials": int(random_removal_trials),
                "beta": float(args.beta),
                "gamma": float(args.gamma),
                "masking": args.masking,
                "quality": args.quality,
                "target_k": int(one["target_k"]),
                "onset_token_text": one["onset_token_text"],
                "generated_onset_token_text": one.get(
                    "generated_onset_token_text", one["onset_token_text"]
                ),
                "target_source": one["run_meta"].get("target_source", "generated"),
                "target_source_mode": one["run_meta"].get("target_source_mode", args.target_source),
                "target_token_id": int(one["run_meta"].get("target_token_id", -1)),
                "fusion_mode": one["run_meta"].get("fusion_mode", args.fusion_mode),
                "mt_floor": float(one["run_meta"].get("mt_floor", args.mt_floor)),
                "packing_mode": str(args.packing_mode),
                "narrative_cap_ratio": float(args.narrative_cap_ratio),
                "narrative_cap_min": int(args.narrative_cap_min),
                "narrative_cap_max": int(args.narrative_cap_max),
                "question_min_tokens": int(args.question_min_tokens),
                "hvp_samples": int(one["run_meta"].get("s_hvp_samples", run_hvp_samples)),
                "max_context_tokens": int(
                    one["run_meta"].get("max_context_tokens", run_context_tokens)
                ),
                "truncated_tokens": int(one["run_meta"].get("truncated_tokens", 0)),
                "important_top_fraction": float(args.important_top_fraction),
                "important_selection": str(args.important_selection),
                "filter_structural_tokens": bool(filter_structural_tokens),
                "important_token_count": int(len(important_indices)),
                "important_token_indices": important_indices,
                "candidate_token_count": int(len(candidate_pool)),
                "segment_mass_scope": "important_tokens_top_fraction",
                "context_packing": str(one["run_meta"].get("context_packing", args.packing_mode)),
                "segment_token_spans": one["segment_token_spans"],
                "segment_token_counts": segment_token_counts,
                "selected_token_counts": selected_counts,
                "segment_coverage": {
                    "narrative": float(segment_coverage.get("narrative", 0.0)),
                    "evidence": float(segment_coverage.get("evidence", 0.0)),
                    "question": float(segment_coverage.get("question", 0.0)),
                },
                "segment_mass_raw": {
                    "narrative": float(segment_mass_raw.get("narrative", 0.0)),
                    "evidence": float(segment_mass_raw.get("evidence", 0.0)),
                    "question": float(segment_mass_raw.get("question", 0.0)),
                },
                "segment_mass": {
                    "narrative": narrative_mass,
                    "evidence": evidence_mass,
                    "question": question_mass,
                },
                "evidence_share_en": float(evidence_share_en),
                "narrative_share_en": float(narrative_share_en),
                "component_segment_mass": {
                    "MT": segment_mass_from_scores(
                        one["component_scores"]["MT"], one["segment_token_spans"]
                    ),
                    "S": segment_mass_from_scores(
                        one["component_scores"]["S"], one["segment_token_spans"]
                    ),
                    "KL": segment_mass_from_scores(
                        one["component_scores"]["KL"], one["segment_token_spans"]
                    ),
                    "final": segment_mass_from_scores(
                        one["component_scores"]["final"], one["segment_token_spans"]
                    ),
                },
                "component_segment_mass_selected": {
                    "MT": segment_mass_on_selected(
                        one["component_scores"]["MT"],
                        one["segment_token_spans"],
                        important_indices,
                    ),
                    "S": segment_mass_on_selected(
                        one["component_scores"]["S"],
                        one["segment_token_spans"],
                        important_indices,
                    ),
                    "KL": segment_mass_on_selected(
                        one["component_scores"]["KL"],
                        one["segment_token_spans"],
                        important_indices,
                    ),
                    "final": segment_mass_on_selected(
                        one["component_scores"]["final"],
                        one["segment_token_spans"],
                        important_indices,
                    ),
                },
                "success": bool(success_strict),
                "success_strict": bool(success_strict),
                "success_margin": bool(success_margin),
                "success_ratio": bool(success_ratio),
                "success_margin_dynamic": bool(success_margin_dynamic),
                "success_margin_0.05": bool(evidence_share_en >= (narrative_share_en + 0.05)),
                "success_ratio_1.2": bool(alignment_ratio >= 1.2),
                "success_per_token": bool(success_per_token),
                "success_dsa": bool(dsa_metrics["success_dsa"]),
                "alignment": float(alignment),
                "alignment_raw": float(alignment_raw),
                "alignment_ratio": float(alignment_ratio),
                "alignment_logratio": float(alignment_logratio),
                "alignment_dsa": float(dsa_metrics["alignment_dsa"]),
                "faithfulness_pass_0.10": bool(removal_metrics["faithfulness_pass_0.10"]),
                "faithfulness_auc_gap": float(removal_metrics["faithfulness_auc_gap"]),
                "margin_delta": float(args.margin_delta),
                "ratio_threshold": float(args.ratio_threshold),
                "dynamic_margin_delta": float(dynamic_margin_delta),
                "valid_by_token_count": bool(valid_by_token_count),
                "valid_context": bool(valid_context),
                "target_prob_orig": float(removal_metrics["target_prob_orig"]),
                "prob_drop_topk": {
                    str(k): float(v) for k, v in (removal_metrics["topk_drop"] or {}).items()
                },
                "prob_drop_random": {
                    str(k): float(v) for k, v in (removal_metrics["random_drop"] or {}).items()
                },
                "prob_drop_gap": {
                    str(k): float(v) for k, v in (removal_metrics["drop_gap"] or {}).items()
                },
                "target_prob_topk": {
                    str(k): float(v) for k, v in (removal_metrics["topk_prob"] or {}).items()
                },
                "target_prob_random": {
                    str(k): float(v) for k, v in (removal_metrics["random_prob"] or {}).items()
                },
                "evidence_anchor_mass": float(dsa_metrics["evidence_anchor_mass"]),
                "narrative_anchor_mass": float(dsa_metrics["narrative_anchor_mass"]),
                "evidence_anchor_source": str(dsa_metrics["evidence_anchor_source"]),
                "evidence_sentence_count": int(dsa_metrics["evidence_sentence_count"]),
                "narrative_sentence_count": int(dsa_metrics["narrative_sentence_count"]),
                "evidence_mass_per_token": float(evidence_mass_per_token),
                "narrative_mass_per_token": float(narrative_mass_per_token),
                "selected_evidence_mass_per_token": float(
                    evidence_mass / max(1, int(selected_counts.get("evidence", 0)))
                ),
                "selected_narrative_mass_per_token": float(
                    narrative_mass / max(1, int(selected_counts.get("narrative", 0)))
                ),
                "latency_ms": int(one["run_meta"]["latency_ms"]),
                "top_tokens": build_top_tokens(
                    one["tokens"], one["component_scores"]["final"], top_n=8
                ),
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()

            if out_record["segment_token_counts"]["evidence"] == 0:
                print(
                    f"[warn] id={rec_id} evidence token span is empty; check segment mapping.",
                    flush=True,
                )
            if not out_record["valid_context"]:
                print(
                    f"[warn] id={rec_id} invalid_context evidence_coverage={out_record['segment_coverage']['evidence']:.3f}",
                    flush=True,
                )

            processed_new += 1
            success_new += int(success_strict)
            alignment_sum_new += alignment
            completed_ids.add(rec_id)
            last_id = rec_id

            if processed_new % max(1, args.save_every) == 0:
                out_f.flush()
                os.fsync(out_f.fileno())
                write_checkpoint(
                    checkpoint_path=checkpoint_json,
                    input_jsonl=input_jsonl,
                    output_dir=output_dir,
                    processed_total=base_n + processed_new,
                    processed_new=processed_new,
                    success_total=base_success + success_new,
                    alignment_sum_total=base_alignment_sum + alignment_sum_new,
                    last_id=last_id,
                )

            running_n = base_n + processed_new
            if running_n > 0 and running_n % 10 == 0:
                running_success = base_success + success_new
                running_alignment_sum = base_alignment_sum + alignment_sum_new
                print(
                    "[progress] n={} success_rate={:.4f} mean_alignment={:.6f}".format(
                        running_n,
                        running_success / running_n,
                        running_alignment_sum / running_n,
                    ),
                    flush=True,
                )

            clear_cuda_memory()

        out_f.flush()
        os.fsync(out_f.fileno())

    write_checkpoint(
        checkpoint_path=checkpoint_json,
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        processed_total=base_n + processed_new,
        processed_new=processed_new,
        success_total=base_success + success_new,
        alignment_sum_total=base_alignment_sum + alignment_sum_new,
        last_id=last_id,
    )
    write_summary_and_csv(
        results_jsonl=results_jsonl,
        summary_path=summary_json,
        csv_path=results_csv,
        margin_delta=args.margin_delta,
        ratio_threshold=args.ratio_threshold,
        valid_evidence_min_tokens=args.valid_evidence_min_tokens,
        dynamic_delta_coeff=args.dynamic_delta_coeff,
        dynamic_delta_floor=args.dynamic_delta_floor,
        important_top_fraction=args.important_top_fraction,
        important_selection=args.important_selection,
        fusion_mode=args.fusion_mode,
        mt_floor=args.mt_floor,
        packing_mode=args.packing_mode,
        narrative_cap_ratio=args.narrative_cap_ratio,
        narrative_cap_min=args.narrative_cap_min,
        narrative_cap_max=args.narrative_cap_max,
        question_min_tokens=args.question_min_tokens,
        target_source=args.target_source,
        filter_structural_tokens=filter_structural_tokens,
        mode=args.mode,
        removal_ratios=removal_ratios,
        random_removal_trials=random_removal_trials,
        removal_eval_enabled=removal_eval_enabled,
    )

    print(f"[done] results_jsonl={results_jsonl}", flush=True)
    print(f"[done] summary_json={summary_json}", flush=True)
    print(f"[done] results_csv={results_csv}", flush=True)
    if STOP_REQUESTED:
        print("[done] exited early by signal; resume is safe.", flush=True)


if __name__ == "__main__":
    main()
