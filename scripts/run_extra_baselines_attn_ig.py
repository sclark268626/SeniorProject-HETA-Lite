from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heta_batch_runner import (
    ANSWER_CUE,
    _first_generated_content_index,
    _first_gold_answer_token_id,
    _pack_context_ids,
    build_segmented_prompt,
    generate_answer_tokens,
    get_model_id,
    load_attributor,
    load_tokenizer,
)
from scripts.run_faithfulness_hotpot import (
    build_candidate_indices,
    compute_removal_faithfulness,
    parse_removal_ratios,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only Attention-Rollout and Integrated-Gradients baselines for existing examples."
    )
    parser.add_argument("--input_jsonl", required=True, help="Converted HETA-style input JSONL.")
    parser.add_argument(
        "--base_results_jsonl",
        required=True,
        help="Existing HETA run results.jsonl used to select example IDs and base settings.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output tidy CSV for aggregate_phase2 --extra_baseline_csv.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model label or HF model id. If empty, uses model from base results row.",
    )
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--masking",
        choices=["drop", "unk", "zero_embed"],
        default="drop",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "accurate"],
        default="accurate",
    )
    parser.add_argument("--target_k", type=int, default=1)
    parser.add_argument("--target_source", choices=["generated", "gold_answer", "auto"], default="generated")
    parser.add_argument("--hvp_samples", type=int, default=1)
    parser.add_argument("--max_context_tokens", type=int, default=1024)
    parser.add_argument("--packing_mode", choices=["evidence_first"], default="evidence_first")
    parser.add_argument("--narrative_cap_ratio", type=float, default=1.0)
    parser.add_argument("--narrative_cap_min", type=int, default=32)
    parser.add_argument("--narrative_cap_max", type=int, default=192)
    parser.add_argument("--question_min_tokens", type=int, default=24)
    parser.add_argument("--filter_structural_tokens", action="store_true")
    parser.add_argument("--removal_ratios", default="0.05,0.10,0.20")
    parser.add_argument("--random_removal_trials", type=int, default=5)
    parser.add_argument("--ig_steps", type=int, default=16)
    parser.add_argument("--max_examples", type=int, default=0, help="0 means all examples in base results.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run_id_override",
        default="",
        help="Override run_id written to CSV. Default: parent folder name of base_results_jsonl.",
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


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_base_records(path: Path, max_examples: int) -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if max_examples and max_examples > 0:
        rows = rows[: max_examples]
    return rows


def load_examples_by_id(path: Path, wanted_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in iter_jsonl(path):
        rec_id = str(rec.get("id", ""))
        if not rec_id or rec_id not in wanted_ids:
            continue
        out[rec_id] = rec
        if len(out) >= len(wanted_ids):
            break
    return out


def compute_integrated_gradients_scores(
    model: Any,
    context_token_ids: List[int],
    target_token_id: int,
    steps: int,
) -> List[float]:
    if not context_token_ids or target_token_id < 0:
        return []
    device = next(model.parameters()).device
    target_pos = len(context_token_ids)
    full_ids = context_token_ids + [int(target_token_id)]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    model_dtype = next(model.parameters()).dtype

    embedding = model.get_input_embeddings()
    with torch.no_grad():
        input_embeds = embedding(input_ids).detach().to(model_dtype)
    baseline_embeds = torch.zeros_like(input_embeds)

    total_grads = torch.zeros_like(input_embeds, dtype=torch.float32)
    pred_pos = max(0, target_pos - 1)
    steps = max(1, int(steps))

    for step in range(1, steps + 1):
        alpha = float(step) / float(steps)
        interp = (baseline_embeds + alpha * (input_embeds - baseline_embeds)).detach()
        interp.requires_grad_(True)
        outputs = model(inputs_embeds=interp, use_cache=False)
        logits = outputs.logits[0, pred_pos, :].to(torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = log_probs[int(target_token_id)]
        grad = torch.autograd.grad(loss, interp, retain_graph=False, create_graph=False)[0]
        total_grads += grad.detach().to(torch.float32)
        del outputs, logits, log_probs, loss, grad, interp

    avg_grads = total_grads / float(steps)
    ig = (input_embeds.to(torch.float32) - baseline_embeds.to(torch.float32)) * avg_grads
    token_scores = ig.norm(dim=-1).squeeze(0)[:target_pos]
    token_scores = token_scores.clamp(min=0.0)
    denom = token_scores.sum().item()
    if denom > 1e-12:
        token_scores = token_scores / denom
    return [float(x) for x in token_scores.detach().cpu().tolist()]


def is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def prepare_baseline_payload(
    attributor: Any,
    tokenizer: Any,
    narrative: str,
    evidence: str,
    question: str,
    answer_text: str,
    args: argparse.Namespace,
    max_context_tokens: int,
) -> Optional[Dict[str, Any]]:
    started = time.perf_counter()
    prompt_bundle = build_segmented_prompt(narrative, evidence, question, tokenizer)
    base_prompt_text = prompt_bundle["full_text"]
    generation_prompt_text = f"{base_prompt_text}{ANSWER_CUE}"
    answer_bundle = generate_answer_tokens(attributor.model, tokenizer, generation_prompt_text)
    answer_ids = answer_bundle["answer_token_ids"]
    answer_tokens = answer_bundle["answer_tokens"]
    if not answer_ids:
        return None

    requested_k = max(1, int(args.target_k))
    generated_k = max(1, min(requested_k, len(answer_ids)))
    if requested_k == 1:
        generated_k = _first_generated_content_index(answer_tokens) + 1
    generated_target_token_id = int(answer_ids[generated_k - 1])

    source_mode = (args.target_source or "generated").strip().lower()
    if source_mode not in {"generated", "gold_answer", "auto"}:
        source_mode = "generated"
    target_token_id = generated_target_token_id
    if requested_k == 1 and source_mode in {"gold_answer", "auto"}:
        gold_target_token_id = _first_gold_answer_token_id(tokenizer, answer_text)
        if gold_target_token_id is not None:
            target_token_id = int(gold_target_token_id)

    answer_prefix_ids = answer_ids[: generated_k - 1]
    pack_info = _pack_context_ids(
        tokenizer=tokenizer,
        narrative=narrative,
        evidence=evidence,
        question=question,
        answer_prefix_ids=answer_prefix_ids,
        max_context_tokens=int(max_context_tokens),
        packing_mode=args.packing_mode,
        narrative_cap_ratio=float(args.narrative_cap_ratio),
        narrative_cap_min=int(args.narrative_cap_min),
        narrative_cap_max=int(args.narrative_cap_max),
        question_min_tokens=int(args.question_min_tokens),
    )
    context_ids = [int(x) for x in pack_info["context_ids"]]
    if not context_ids:
        return None

    device = attributor.device
    input_with_target = torch.tensor(
        [context_ids + [int(target_token_id)]],
        dtype=torch.long,
        device=device,
    )
    target_pos = len(context_ids)
    attn_scores = attributor.compute_attention_rollout(input_with_target, target_pos)
    mt_scores = attn_scores.detach().to(torch.float32).cpu().numpy()[:target_pos].tolist()
    tokens = [tokenizer.decode([tok], skip_special_tokens=False) for tok in context_ids]

    return {
        "context_token_ids": context_ids,
        "tokens": tokens,
        "target_token_id": int(target_token_id),
        "onset_token_text": tokenizer.decode([int(target_token_id)], skip_special_tokens=False),
        "segment_token_spans": {
            "narrative": [int(pack_info["segment_ranges"]["narrative"][0]), int(pack_info["segment_ranges"]["narrative"][1])],
            "evidence": [int(pack_info["segment_ranges"]["evidence"][0]), int(pack_info["segment_ranges"]["evidence"][1])],
            "question": [int(pack_info["segment_ranges"]["question"][0]), int(pack_info["segment_ranges"]["question"][1])],
        },
        "attn_rollout_scores": [float(x) for x in mt_scores],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def prepare_example_with_backoff(
    attributor: Any,
    tokenizer: Any,
    narrative: str,
    evidence: str,
    question: str,
    answer_text: str,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    start_ctx = max(1, int(args.max_context_tokens))
    context_candidates: List[int] = []
    for v in [start_ctx, 384, 256, 192, 128]:
        iv = min(start_ctx, int(v))
        if iv > 0 and iv not in context_candidates:
            context_candidates.append(iv)

    for max_ctx in context_candidates:
        try:
            return prepare_baseline_payload(
                attributor=attributor,
                tokenizer=tokenizer,
                narrative=narrative,
                evidence=evidence,
                question=question,
                answer_text=answer_text,
                args=args,
                max_context_tokens=max_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc):
                print(f"[warn] OOM; retry with smaller context. tried={max_ctx}", flush=True)
                clear_cuda_memory()
                continue
            raise
    return None


def compute_ig_with_backoff(
    model: Any,
    context_token_ids: List[int],
    target_token_id: int,
    ig_steps: int,
) -> Optional[List[float]]:
    start_steps = max(1, int(ig_steps))
    step_candidates: List[int] = []
    for v in [start_steps, max(1, start_steps // 2), max(1, start_steps // 4), 1]:
        iv = max(1, int(v))
        if iv not in step_candidates:
            step_candidates.append(iv)

    for steps in step_candidates:
        try:
            return compute_integrated_gradients_scores(
                model=model,
                context_token_ids=context_token_ids,
                target_token_id=target_token_id,
                steps=steps,
            )
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc):
                print(f"[warn] IG OOM; retry with fewer steps. tried={steps}", flush=True)
                clear_cuda_memory()
                continue
            raise
    return None


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    base_results_path = Path(args.base_results_jsonl)
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_records = load_base_records(base_results_path, max_examples=int(args.max_examples))
    if not base_records:
        raise RuntimeError(f"No records found in {base_results_path}")

    run_id = (
        args.run_id_override.strip()
        if args.run_id_override.strip()
        else base_results_path.parent.name
    )

    wanted_ids = {str(r.get("id", "")) for r in base_records if r.get("id")}
    example_map = load_examples_by_id(input_path, wanted_ids)
    missing = sorted(wanted_ids - set(example_map.keys()))
    if missing:
        print(f"[warn] missing {len(missing)} ids in input_jsonl; those rows will be skipped.", flush=True)

    first_model = str(base_records[0].get("model", "")).strip()
    model_name = args.model.strip() or first_model
    if not model_name:
        raise RuntimeError("Model name is empty. Pass --model explicitly.")

    # Load once for IG/removal forward passes.
    eval_attributor = load_attributor(get_model_id(model_name))
    eval_model = eval_attributor.model
    eval_model.eval()
    for param in eval_model.parameters():
        param.requires_grad_(False)
    tokenizer = load_tokenizer(get_model_id(model_name))

    ratios = parse_removal_ratios(args.removal_ratios)
    rng = random.Random(int(args.seed))

    fieldnames = [
        "run_id",
        "example_id",
        "method",
        "removal_ratio",
        "p0",
        "pr",
        "drop",
        "target_token",
        "seq_len",
        "timestamp",
        "latency_ms_total",
        "latency_ms_attribution",
        "latency_ms_forward",
    ]

    processed = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in base_records:
            rec_id = str(rec.get("id", ""))
            if not rec_id or rec_id not in example_map:
                continue

            row = example_map[rec_id]
            seg = row.get("segments", {}) or {}
            narrative = str(seg.get("narrative", ""))
            evidence = str(seg.get("evidence", ""))
            question = str(seg.get("question", ""))
            answer_text = str(row.get("answer", ""))

            try:
                one = prepare_example_with_backoff(
                    attributor=eval_attributor,
                    tokenizer=tokenizer,
                    narrative=narrative,
                    evidence=evidence,
                    question=question,
                    answer_text=answer_text,
                    args=args,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] id={rec_id} baseline prep failed: {exc}", flush=True)
                clear_cuda_memory()
                continue
            if one is None:
                print(f"[warn] id={rec_id} baseline prep failed after OOM backoff", flush=True)
                clear_cuda_memory()
                continue

            token_ids = [int(x) for x in (one.get("context_token_ids") or [])]
            token_texts = [str(t) for t in (one.get("tokens") or [])]
            target_token_id = _safe_int(one.get("target_token_id", -1), -1)
            target_token_text = str(one.get("onset_token_text", ""))
            spans = one.get("segment_token_spans", {}) or {}

            candidate_indices = build_candidate_indices(
                segment_spans=spans,
                arr_len=len(token_ids),
                token_texts=token_texts,
                filter_structural_tokens=bool(args.filter_structural_tokens),
            )
            if not candidate_indices:
                print(f"[warn] id={rec_id} no candidate indices", flush=True)
                clear_cuda_memory()
                continue

            mt_scores = [float(x) for x in (one.get("attn_rollout_scores") or [])]
            if len(mt_scores) != len(token_ids):
                print(f"[warn] id={rec_id} MT length mismatch", flush=True)
                clear_cuda_memory()
                continue

            try:
                ig_scores = compute_ig_with_backoff(
                    model=eval_model,
                    context_token_ids=token_ids,
                    target_token_id=target_token_id,
                    ig_steps=int(args.ig_steps),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] id={rec_id} IG failed: {exc}", flush=True)
                clear_cuda_memory()
                continue
            if ig_scores is None:
                print(f"[warn] id={rec_id} IG failed after OOM backoff", flush=True)
                clear_cuda_memory()
                continue
            if len(ig_scores) != len(token_ids):
                print(f"[warn] id={rec_id} IG length mismatch", flush=True)
                clear_cuda_memory()
                continue

            method_to_scores = {
                "Attention-Rollout": mt_scores,
                "Integrated Gradients": ig_scores,
            }

            for method, scores in method_to_scores.items():
                metrics = compute_removal_faithfulness(
                    model=eval_model,
                    context_token_ids=token_ids,
                    target_token_id=target_token_id,
                    scores=scores,
                    candidate_indices=candidate_indices,
                    removal_ratios=ratios,
                    random_trials=max(1, int(args.random_removal_trials)),
                    rng=rng,
                )
                p0 = _safe_float(metrics.get("target_prob_orig", 0.0), 0.0)
                topk_drop = metrics.get("topk_drop", {}) or {}
                topk_prob = metrics.get("topk_prob", {}) or {}
                for ratio_key, drop in topk_drop.items():
                    ratio_val = _safe_float(ratio_key, 0.0)
                    pr = _safe_float(topk_prob.get(ratio_key, p0 - _safe_float(drop, 0.0)), 0.0)
                    writer.writerow(
                        {
                            "run_id": run_id,
                            "example_id": rec_id,
                            "method": method,
                            "removal_ratio": ratio_val,
                            "p0": p0,
                            "pr": pr,
                            "drop": _safe_float(drop, 0.0),
                            "target_token": target_token_text,
                            "seq_len": len(token_ids),
                            "timestamp": "",
                            "latency_ms_total": _safe_float(one.get("latency_ms", 0.0), 0.0),
                            "latency_ms_attribution": _safe_float(one.get("latency_ms", 0.0), 0.0),
                            "latency_ms_forward": 0.0,
                        }
                    )

            processed += 1
            if processed % 10 == 0:
                print(f"[progress] processed={processed}", flush=True)
            clear_cuda_memory()

    print(f"[done] rows written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
