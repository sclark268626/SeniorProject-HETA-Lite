from __future__ import annotations

import time
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from heta_demo import HETAAttributor, MODEL_OPTIONS

SEGMENT_SEPARATOR = ""
MAX_ANSWER_TOKENS = 16
DEFAULT_MODEL_LABEL = next(iter(MODEL_OPTIONS.keys()))
DEFAULT_MODEL_ID = MODEL_OPTIONS[DEFAULT_MODEL_LABEL]
DEFAULT_HVP_SAMPLES = 1
DEFAULT_MAX_CONTEXT_TOKENS = 384
DEFAULT_PACKING_MODE = "evidence_first"
DEFAULT_NARRATIVE_CAP_RATIO = 1.0
DEFAULT_NARRATIVE_CAP_MIN = 32
DEFAULT_NARRATIVE_CAP_MAX = 192
DEFAULT_QUESTION_MIN_TOKENS = 24
ANSWER_CUE = "\nAnswer:"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_compute_dtype(device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def get_model_id(model_choice: str) -> str:
    if model_choice in MODEL_OPTIONS:
        return MODEL_OPTIONS[model_choice]
    if model_choice in MODEL_OPTIONS.values():
        return model_choice
    return DEFAULT_MODEL_ID


def get_model_label(model_choice: str) -> str:
    if model_choice in MODEL_OPTIONS:
        return model_choice
    for label, model_id in MODEL_OPTIONS.items():
        if model_id == model_choice:
            return label
    return DEFAULT_MODEL_LABEL


@lru_cache(maxsize=8)
def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=1)
def load_attributor(model_name: str) -> HETAAttributor:
    device = get_device()
    dtype = get_compute_dtype(device)
    tokenizer = load_tokenizer(model_name)
    common_kwargs = {
        "device_map": device,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            **common_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **common_kwargs,
        )
    model.eval()
    return HETAAttributor(model, tokenizer, device)


def build_segmented_prompt(
    narrative: str, evidence: str, question: str, tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    narrative = (narrative or "").strip()
    evidence = (evidence or "").strip()
    question = (question or "").strip()
    prefix_text = {
        "narrative": "",
        "evidence": "",
        "question": "",
    }
    segment_content = {
        "narrative": narrative,
        "evidence": evidence,
        "question": question,
    }
    segment_text = {
        key: (
            "{} {}".format(prefix_text[key], segment_content[key]).strip()
            if segment_content[key]
            else prefix_text[key]
        )
        for key in prefix_text
    }
    full_text = SEGMENT_SEPARATOR.join(
        [segment_text["narrative"], segment_text["evidence"], segment_text["question"]]
    )

    sep_len = len(SEGMENT_SEPARATOR)
    narrative_seg_start_char = 0
    narrative_seg_end_char = len(segment_text["narrative"])
    evidence_seg_start_char = narrative_seg_end_char + sep_len
    evidence_seg_end_char = evidence_seg_start_char + len(segment_text["evidence"])
    question_seg_start_char = evidence_seg_end_char + sep_len
    question_seg_end_char = question_seg_start_char + len(segment_text["question"])

    narrative_content_start_char = narrative_seg_start_char + len(prefix_text["narrative"])
    evidence_content_start_char = evidence_seg_start_char + len(prefix_text["evidence"])
    question_content_start_char = question_seg_start_char + len(prefix_text["question"])
    if narrative:
        narrative_content_start_char += 1
    if evidence:
        evidence_content_start_char += 1
    if question:
        question_content_start_char += 1
    narrative_content_end_char = narrative_content_start_char + len(narrative)
    evidence_content_end_char = evidence_content_start_char + len(evidence)
    question_content_end_char = question_content_start_char + len(question)

    def approx_spans() -> Dict[str, Tuple[int, int]]:
        separator_ids = tokenizer(SEGMENT_SEPARATOR, add_special_tokens=False).input_ids
        narrative_ids = tokenizer(segment_text["narrative"], add_special_tokens=False).input_ids
        evidence_ids = tokenizer(segment_text["evidence"], add_special_tokens=False).input_ids
        question_ids = tokenizer(segment_text["question"], add_special_tokens=False).input_ids

        narrative_prefix_ids = tokenizer(
            "{} ".format(prefix_text["narrative"]), add_special_tokens=False
        ).input_ids
        evidence_prefix_ids = tokenizer(
            "{} ".format(prefix_text["evidence"]), add_special_tokens=False
        ).input_ids
        question_prefix_ids = tokenizer(
            "{} ".format(prefix_text["question"]), add_special_tokens=False
        ).input_ids

        narrative_content_ids = tokenizer(narrative, add_special_tokens=False).input_ids
        evidence_content_ids = tokenizer(evidence, add_special_tokens=False).input_ids
        question_content_ids = tokenizer(question, add_special_tokens=False).input_ids

        narrative_segment_start = 0
        narrative_segment_end = len(narrative_ids)
        evidence_segment_start = len(narrative_ids) + len(separator_ids)
        evidence_segment_end = evidence_segment_start + len(evidence_ids)
        question_segment_start = evidence_segment_start + len(evidence_ids) + len(separator_ids)
        question_segment_end = question_segment_start + len(question_ids)

        narrative_start = narrative_segment_start + len(narrative_prefix_ids)
        narrative_end = narrative_start + len(narrative_content_ids)
        evidence_start = evidence_segment_start + len(evidence_prefix_ids)
        evidence_end = evidence_start + len(evidence_content_ids)
        question_start = question_segment_start + len(question_prefix_ids)
        question_end = question_start + len(question_content_ids)

        narrative_start = min(narrative_start, narrative_segment_end)
        narrative_end = min(max(narrative_start, narrative_end), narrative_segment_end)
        evidence_start = min(evidence_start, evidence_segment_end)
        evidence_end = min(max(evidence_start, evidence_end), evidence_segment_end)
        question_start = min(question_start, question_segment_end)
        question_end = min(max(question_start, question_end), question_segment_end)
        return {
            "narrative": (narrative_start, narrative_end),
            "evidence": (evidence_start, evidence_end),
            "question": (question_start, question_end),
        }

    spans: Dict[str, Tuple[int, int]]
    use_offsets = bool(getattr(tokenizer, "is_fast", False))
    if use_offsets:
        try:
            encoded = tokenizer(
                full_text, add_special_tokens=False, return_offsets_mapping=True
            )
            offsets = encoded.get("offset_mapping", [])

            def char_to_token_span(start_char: int, end_char: int) -> Tuple[int, int]:
                if end_char <= start_char:
                    return (0, 0)
                idxs = [
                    idx
                    for idx, (s, e) in enumerate(offsets)
                    if int(e) > int(start_char) and int(s) < int(end_char)
                ]
                if not idxs:
                    return (0, 0)
                return (int(idxs[0]), int(idxs[-1]) + 1)

            spans = {
                "narrative": char_to_token_span(
                    narrative_content_start_char, narrative_content_end_char
                ),
                "evidence": char_to_token_span(
                    evidence_content_start_char, evidence_content_end_char
                ),
                "question": char_to_token_span(
                    question_content_start_char, question_content_end_char
                ),
            }
        except Exception:
            spans = approx_spans()
    else:
        spans = approx_spans()

    return {
        "full_text": full_text,
        "segments": spans,
    }


def _encode_segment_token_parts(
    tokenizer: AutoTokenizer, tag: str, content: str
) -> Tuple[List[int], List[int], List[int]]:
    marker_ids = tokenizer(tag, add_special_tokens=False).input_ids
    clean_content = (content or "").strip()
    content_ids = (
        tokenizer(f" {clean_content}", add_special_tokens=False).input_ids
        if clean_content
        else []
    )
    return marker_ids, content_ids, marker_ids + content_ids


def _build_segment_ranges(
    narrative_marker_ids: List[int],
    narrative_content_ids: List[int],
    evidence_marker_ids: List[int],
    evidence_content_ids: List[int],
    question_marker_ids: List[int],
    question_content_ids: List[int],
    sep_ids: List[int],
) -> Dict[str, Tuple[int, int]]:
    narrative_start = len(narrative_marker_ids)
    narrative_end = narrative_start + len(narrative_content_ids)

    evidence_offset = len(narrative_marker_ids) + len(narrative_content_ids) + len(sep_ids)
    evidence_start = evidence_offset + len(evidence_marker_ids)
    evidence_end = evidence_start + len(evidence_content_ids)

    question_offset = evidence_offset + len(evidence_marker_ids) + len(evidence_content_ids) + len(
        sep_ids
    )
    question_start = question_offset + len(question_marker_ids)
    question_end = question_start + len(question_content_ids)

    return {
        "narrative": (int(narrative_start), int(narrative_end)),
        "evidence": (int(evidence_start), int(evidence_end)),
        "question": (int(question_start), int(question_end)),
    }


def _segment_coverage_from_ranges(
    ranges: Dict[str, Tuple[int, int]],
    original_counts: Dict[str, int],
    context_len: int,
) -> Dict[str, float]:
    coverage: Dict[str, float] = {}
    for name in ("narrative", "evidence", "question"):
        start, end = ranges.get(name, (0, 0))
        lo = max(0, int(start))
        hi = min(int(context_len), int(end))
        kept = max(0, hi - lo)
        orig = max(0, int(original_counts.get(name, 0)))
        coverage[name] = float(kept / orig) if orig > 0 else 1.0
    return coverage


def _pack_context_ids(
    tokenizer: AutoTokenizer,
    narrative: str,
    evidence: str,
    question: str,
    answer_prefix_ids: List[int],
    max_context_tokens: int,
    packing_mode: str,
    narrative_cap_ratio: float,
    narrative_cap_min: int,
    narrative_cap_max: int,
    question_min_tokens: int,
) -> Dict[str, Any]:
    sep_ids = tokenizer(SEGMENT_SEPARATOR, add_special_tokens=False).input_ids
    cue_ids = tokenizer(ANSWER_CUE, add_special_tokens=False).input_ids

    n_marker, n_content, n_full = _encode_segment_token_parts(
        tokenizer, "[NarrativeQA]", narrative
    )
    e_marker, e_content, e_full = _encode_segment_token_parts(tokenizer, "[SciQ]", evidence)
    q_marker, q_content, q_full = _encode_segment_token_parts(tokenizer, "[Question]", question)

    full_prompt_ids = n_full + sep_ids + e_full + sep_ids + q_full + cue_ids
    full_context_ids = full_prompt_ids + list(answer_prefix_ids)
    full_ranges = _build_segment_ranges(
        n_marker,
        n_content,
        e_marker,
        e_content,
        q_marker,
        q_content,
        sep_ids,
    )
    original_counts = {
        "narrative": len(n_content),
        "evidence": len(e_content),
        "question": len(q_content),
    }

    keep = int(max_context_tokens)
    if keep <= 0 or len(full_context_ids) <= keep:
        coverage = _segment_coverage_from_ranges(full_ranges, original_counts, len(full_context_ids))
        return {
            "context_ids": full_context_ids,
            "prompt_token_count": len(full_prompt_ids),
            "segment_ranges": full_ranges,
            "segment_counts_original": original_counts,
            "segment_counts_kept": {
                "narrative": len(n_content),
                "evidence": len(e_content),
                "question": len(q_content),
            },
            "segment_coverage": coverage,
            "truncated_tokens": 0,
            "original_context_len": len(full_context_ids),
            "context_packing": "full",
            "valid_context": bool(
                coverage["evidence"] >= 0.999 and coverage["question"] > 0.0
            ),
        }

    mode = (packing_mode or DEFAULT_PACKING_MODE).strip().lower()
    if mode not in {"evidence_first", "tail"}:
        mode = DEFAULT_PACKING_MODE

    if mode == "tail":
        context_ids = full_context_ids[-keep:]
        truncated = len(full_context_ids) - len(context_ids)
        shifted_ranges: Dict[str, Tuple[int, int]] = {}
        for name, (start, end) in full_ranges.items():
            new_start = max(0, int(start) - truncated)
            new_end = max(0, int(end) - truncated)
            new_start = min(new_start, len(context_ids))
            new_end = min(new_end, len(context_ids))
            if new_end < new_start:
                new_end = new_start
            shifted_ranges[name] = (new_start, new_end)
        coverage = _segment_coverage_from_ranges(shifted_ranges, original_counts, len(context_ids))
        kept_counts = {
            "narrative": max(0, shifted_ranges["narrative"][1] - shifted_ranges["narrative"][0]),
            "evidence": max(0, shifted_ranges["evidence"][1] - shifted_ranges["evidence"][0]),
            "question": max(0, shifted_ranges["question"][1] - shifted_ranges["question"][0]),
        }
        prompt_len_after_tail = max(0, len(full_prompt_ids) - truncated)
        return {
            "context_ids": context_ids,
            "prompt_token_count": prompt_len_after_tail,
            "segment_ranges": shifted_ranges,
            "segment_counts_original": original_counts,
            "segment_counts_kept": kept_counts,
            "segment_coverage": coverage,
            "truncated_tokens": truncated,
            "original_context_len": len(full_context_ids),
            "context_packing": "tail",
            "valid_context": bool(
                coverage["evidence"] >= 0.999 and coverage["question"] > 0.0
            ),
        }

    fixed_len = (
        len(n_marker)
        + len(sep_ids)
        + len(e_marker)
        + len(sep_ids)
        + len(q_marker)
        + len(cue_ids)
        + len(answer_prefix_ids)
    )
    if fixed_len >= keep:
        context_ids = full_context_ids[-keep:]
        truncated = len(full_context_ids) - len(context_ids)
        shifted_ranges = {}
        for name, (start, end) in full_ranges.items():
            new_start = max(0, int(start) - truncated)
            new_end = max(0, int(end) - truncated)
            new_start = min(new_start, len(context_ids))
            new_end = min(new_end, len(context_ids))
            if new_end < new_start:
                new_end = new_start
            shifted_ranges[name] = (new_start, new_end)
        coverage = _segment_coverage_from_ranges(shifted_ranges, original_counts, len(context_ids))
        kept_counts = {
            "narrative": max(0, shifted_ranges["narrative"][1] - shifted_ranges["narrative"][0]),
            "evidence": max(0, shifted_ranges["evidence"][1] - shifted_ranges["evidence"][0]),
            "question": max(0, shifted_ranges["question"][1] - shifted_ranges["question"][0]),
        }
        prompt_len_after_tail = max(0, len(full_prompt_ids) - truncated)
        return {
            "context_ids": context_ids,
            "prompt_token_count": prompt_len_after_tail,
            "segment_ranges": shifted_ranges,
            "segment_counts_original": original_counts,
            "segment_counts_kept": kept_counts,
            "segment_coverage": coverage,
            "truncated_tokens": truncated,
            "original_context_len": len(full_context_ids),
            "context_packing": "tail_fallback",
            "valid_context": bool(
                coverage["evidence"] >= 0.999 and coverage["question"] > 0.0
            ),
        }

    avail = keep - fixed_len
    reserve_q = min(len(q_content), max(0, int(question_min_tokens)))
    e_keep_len = min(len(e_content), max(0, avail - reserve_q))
    avail_after_e = avail - e_keep_len
    q_keep_len = min(len(q_content), max(0, avail_after_e))
    if q_keep_len < reserve_q and e_keep_len > 0:
        need = reserve_q - q_keep_len
        give_back = min(need, e_keep_len)
        e_keep_len -= give_back
        q_keep_len += give_back
        avail_after_e += give_back

    avail_after_eq = avail - e_keep_len - q_keep_len
    e_keep = e_content[:e_keep_len]
    q_keep = q_content[:q_keep_len]

    dynamic_cap = int(max(float(narrative_cap_min), float(narrative_cap_ratio) * max(1, e_keep_len)))
    dynamic_cap = min(dynamic_cap, int(max(0, narrative_cap_max)))
    n_keep_len = min(len(n_content), max(0, dynamic_cap), max(0, avail_after_eq))
    n_keep = n_content[-n_keep_len:] if n_keep_len > 0 else []

    n_full_keep = n_marker + n_keep
    e_full_keep = e_marker + e_keep
    q_full_keep = q_marker + q_keep
    prompt_context_ids = n_full_keep + sep_ids + e_full_keep + sep_ids + q_full_keep + cue_ids
    context_ids = prompt_context_ids + list(answer_prefix_ids)

    # Final guardrail in case the heuristic still overshoots.
    if len(context_ids) > keep:
        context_ids = context_ids[-keep:]
        truncated = len(full_context_ids) - len(context_ids)
        shifted_ranges = {}
        for name, (start, end) in full_ranges.items():
            new_start = max(0, int(start) - truncated)
            new_end = max(0, int(end) - truncated)
            new_start = min(new_start, len(context_ids))
            new_end = min(new_end, len(context_ids))
            if new_end < new_start:
                new_end = new_start
            shifted_ranges[name] = (new_start, new_end)
        coverage = _segment_coverage_from_ranges(shifted_ranges, original_counts, len(context_ids))
        kept_counts = {
            "narrative": max(0, shifted_ranges["narrative"][1] - shifted_ranges["narrative"][0]),
            "evidence": max(0, shifted_ranges["evidence"][1] - shifted_ranges["evidence"][0]),
            "question": max(0, shifted_ranges["question"][1] - shifted_ranges["question"][0]),
        }
        prompt_len_after_tail = max(0, len(full_prompt_ids) - truncated)
        return {
            "context_ids": context_ids,
            "prompt_token_count": prompt_len_after_tail,
            "segment_ranges": shifted_ranges,
            "segment_counts_original": original_counts,
            "segment_counts_kept": kept_counts,
            "segment_coverage": coverage,
            "truncated_tokens": truncated,
            "original_context_len": len(full_context_ids),
            "context_packing": "evidence_first_tail_guardrail",
            "valid_context": bool(
                coverage["evidence"] >= 0.999 and coverage["question"] > 0.0
            ),
        }

    kept_ranges = _build_segment_ranges(
        n_marker,
        n_keep,
        e_marker,
        e_keep,
        q_marker,
        q_keep,
        sep_ids,
    )
    kept_counts = {
        "narrative": len(n_keep),
        "evidence": len(e_keep),
        "question": len(q_keep),
    }
    coverage = {
        name: (float(kept_counts[name] / original_counts[name]) if original_counts[name] > 0 else 1.0)
        for name in ("narrative", "evidence", "question")
    }
    truncated = len(full_context_ids) - len(context_ids)
    return {
        "context_ids": context_ids,
        "prompt_token_count": len(prompt_context_ids),
        "segment_ranges": kept_ranges,
        "segment_counts_original": original_counts,
        "segment_counts_kept": kept_counts,
        "segment_coverage": coverage,
        "truncated_tokens": max(0, truncated),
        "original_context_len": len(full_context_ids),
        "context_packing": "evidence_first",
        "valid_context": bool(coverage["evidence"] >= 0.999 and coverage["question"] > 0.0),
    }


def generate_answer_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_text: str,
    max_new_tokens: int = MAX_ANSWER_TOKENS,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids[0].tolist()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer_token_ids = generated[0, inputs.input_ids.shape[1] :].tolist()
    if not answer_token_ids:
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits[0, -1, :]
            answer_token_ids = [int(torch.argmax(logits).item())]
    answer_tokens = [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in answer_token_ids
    ]
    answer_text = tokenizer.decode(answer_token_ids, skip_special_tokens=True)
    return {
        "prompt_token_ids": prompt_ids,
        "answer_token_ids": answer_token_ids,
        "answer_tokens": answer_tokens,
        "answer_text": answer_text.strip(),
    }


def _is_structural_token(token_text: str) -> bool:
    t = (token_text or "").strip()
    if not t:
        return True
    return re.search(r"[A-Za-z0-9]", t) is None


def _first_generated_content_index(answer_tokens: List[str]) -> int:
    for idx, tok in enumerate(answer_tokens):
        if not _is_structural_token(tok):
            return idx
    return 0


def _first_gold_answer_token_id(
    tokenizer: AutoTokenizer, answer_text: str, with_prefix_space: bool = True
) -> int | None:
    answer = (answer_text or "").strip()
    if not answer:
        return None
    variants = [answer]
    if with_prefix_space:
        variants = [f" {answer}", answer]
    for text in variants:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        tok = tokenizer.decode([int(ids[0])], skip_special_tokens=False)
        if _is_structural_token(tok):
            continue
        return int(ids[0])
    return None


def compute_kl_information(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    target_pos: int,
    mask_strategy: str,
) -> np.ndarray:
    seq_len = input_ids.shape[1]
    kl_scores = torch.zeros(seq_len, device=input_ids.device)
    if target_pos <= 0:
        return kl_scores.cpu().numpy()

    pred_pos = target_pos - 1
    with torch.no_grad():
        orig_logits = model(input_ids, use_cache=False).logits[0, pred_pos, :]
        orig_probs = torch.softmax(orig_logits, dim=-1)

    replace_id = tokenizer.unk_token_id
    if replace_id is None:
        replace_id = tokenizer.mask_token_id
    if replace_id is None:
        replace_id = tokenizer.pad_token_id
    if replace_id is None:
        replace_id = tokenizer.eos_token_id
    if replace_id is None:
        replace_id = 0

    for pos in range(target_pos):
        if mask_strategy == "drop":
            masked_ids = torch.cat([input_ids[:, :pos], input_ids[:, pos + 1 :]], dim=1)
            masked_target_pos = target_pos - 1
            masked_pred_pos = masked_target_pos - 1
            if masked_pred_pos < 0:
                continue
            with torch.no_grad():
                masked_logits = model(masked_ids, use_cache=False).logits[0, masked_pred_pos, :]
        elif mask_strategy == "zero_embed":
            with torch.no_grad():
                embeds = model.get_input_embeddings()(input_ids).clone()
                embeds[:, pos, :] = 0
                masked_logits = model(inputs_embeds=embeds, use_cache=False).logits[
                    0, pred_pos, :
                ]
        else:
            masked_ids = input_ids.clone()
            masked_ids[0, pos] = replace_id
            with torch.no_grad():
                masked_logits = model(masked_ids, use_cache=False).logits[0, pred_pos, :]

        masked_probs = torch.softmax(masked_logits, dim=-1)
        kl = (
            orig_probs
            * (torch.log(orig_probs + 1e-10) - torch.log(masked_probs + 1e-10))
        ).sum()
        kl_scores[pos] = kl.clamp(min=0)

    kl_scores[target_pos:] = 0
    score_sum = kl_scores[:target_pos].sum()
    if score_sum > 1e-10:
        kl_scores[:target_pos] = kl_scores[:target_pos] / score_sum
    return kl_scores.cpu().numpy()


def compute_semantic_flow_mt(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    target_pos: int,
    target_token_id: int | None = None,
) -> np.ndarray:
    seq_len = input_ids.shape[1]
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )

    attentions = outputs.attentions
    hidden_states = outputs.hidden_states
    if not attentions:
        mt = torch.ones(seq_len, device=input_ids.device, dtype=torch.float32)
        mt = mt / mt.sum().clamp(min=1e-12)
        return mt.detach().cpu().numpy()

    def _get_decoder_layers(model_obj: AutoModelForCausalLM) -> List[Any]:
        if hasattr(model_obj, "model") and hasattr(model_obj.model, "layers"):
            return list(model_obj.model.layers)
        if hasattr(model_obj, "transformer") and hasattr(model_obj.transformer, "h"):
            return list(model_obj.transformer.h)
        return []

    def _get_attn_module(layer_obj: Any) -> Any:
        for name in ("self_attn", "attn", "attention"):
            if hasattr(layer_obj, name):
                return getattr(layer_obj, name)
        return None

    def _get_linear(mod: Any, names: List[str]) -> Any:
        for name in names:
            if hasattr(mod, name):
                obj = getattr(mod, name)
                if hasattr(obj, "weight"):
                    return obj
        return None

    def _projected_value_norms(layer_idx: int, head_count: int) -> torch.Tensor:
        # Try to follow paper term ||V_i^{(l,h)} W_O^{(l,h)}||_1. If module access fails,
        # fallback to per-token hidden-state norm replicated across heads.
        if hidden_states is None or layer_idx >= len(hidden_states):
            base = torch.ones(seq_len, device=input_ids.device, dtype=torch.float32)
            return base.unsqueeze(0).repeat(head_count, 1)

        layers = _get_decoder_layers(model)
        if not layers or layer_idx >= len(layers):
            base = hidden_states[layer_idx].squeeze(0).to(torch.float32).norm(p=1, dim=-1)
            return base.unsqueeze(0).repeat(head_count, 1)

        attn_mod = _get_attn_module(layers[layer_idx])
        if attn_mod is None:
            base = hidden_states[layer_idx].squeeze(0).to(torch.float32).norm(p=1, dim=-1)
            return base.unsqueeze(0).repeat(head_count, 1)

        v_proj = _get_linear(attn_mod, ["v_proj", "value", "v"])
        o_proj = _get_linear(attn_mod, ["o_proj", "out_proj", "c_proj", "dense"])
        if v_proj is None or o_proj is None:
            base = hidden_states[layer_idx].squeeze(0).to(torch.float32).norm(p=1, dim=-1)
            return base.unsqueeze(0).repeat(head_count, 1)
        with torch.no_grad():
            h_in = hidden_states[layer_idx].to(v_proj.weight.dtype)
            v_all = v_proj(h_in)  # [1, T, D_v]
            if v_all.ndim != 3 or v_all.shape[1] != seq_len:
                base = hidden_states[layer_idx].squeeze(0).to(torch.float32).norm(p=1, dim=-1)
                return base.unsqueeze(0).repeat(head_count, 1)

            v_dim = int(v_all.shape[-1])
            num_heads = int(
                getattr(attn_mod, "num_heads", 0)
                or getattr(attn_mod, "n_heads", 0)
                or getattr(attn_mod, "num_attention_heads", 0)
                or head_count
            )
            num_kv_heads = int(getattr(attn_mod, "num_key_value_heads", num_heads))
            num_kv_heads = max(1, num_kv_heads)
            if v_dim % num_kv_heads != 0:
                base = hidden_states[layer_idx].squeeze(0).to(torch.float32).norm(p=1, dim=-1)
                return base.unsqueeze(0).repeat(head_count, 1)

            head_dim = v_dim // num_kv_heads
            v_heads = v_all.view(1, seq_len, num_kv_heads, head_dim)
            if num_kv_heads != num_heads:
                repeat = max(1, num_heads // num_kv_heads)
                v_heads = v_heads.repeat_interleave(repeat, dim=2)
            if v_heads.shape[2] < num_heads:
                repeat_extra = max(1, (num_heads + v_heads.shape[2] - 1) // v_heads.shape[2])
                v_heads = v_heads.repeat_interleave(repeat_extra, dim=2)
            v_heads = v_heads[:, :, :num_heads, :].squeeze(0)  # [T, H, Dh]

            o_w = o_proj.weight.detach().to(torch.float32)  # [D_out, D_in]
            in_dim = int(o_w.shape[1])
            heads_from_o = max(1, in_dim // head_dim)
            h_eff = min(int(v_heads.shape[1]), heads_from_o, head_count)
            v_heads = v_heads[:, :h_eff, :].to(torch.float32)
            o_slice = o_w[:, : h_eff * head_dim]
            o_heads = o_slice.view(o_w.shape[0], h_eff, head_dim).permute(1, 2, 0)  # [H, Dh, D_out]
            projected = torch.einsum("thd,hdm->thm", v_heads, o_heads)
            norms = projected.abs().sum(dim=-1).transpose(0, 1)  # [H, T]
            if h_eff < head_count:
                pad = norms.mean(dim=0, keepdim=True).repeat(head_count - h_eff, 1)
                norms = torch.cat([norms, pad], dim=0)
            return norms[:head_count, :]

    head_count = int(attentions[0].shape[1])
    eye = torch.eye(seq_len, device=input_ids.device, dtype=torch.float32).unsqueeze(0)
    rollout = eye.repeat(head_count, 1, 1)
    mt = torch.zeros(seq_len, device=input_ids.device, dtype=torch.float32)

    for layer_idx, layer_attn in enumerate(attentions):
        attn_h = layer_attn.squeeze(0).to(torch.float32)
        if attn_h.ndim != 3:
            continue
        if attn_h.shape[0] != head_count:
            # Align unexpected head count to stable shape.
            h_cur = int(attn_h.shape[0])
            if h_cur <= 0:
                continue
            if h_cur < head_count:
                rep = max(1, (head_count + h_cur - 1) // h_cur)
                attn_h = attn_h.repeat(rep, 1, 1)[:head_count]
            else:
                attn_h = attn_h[:head_count]
        attn_h = 0.5 * attn_h + 0.5 * eye
        attn_h = attn_h / attn_h.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        rollout = torch.matmul(attn_h, rollout)
        path_mass_h = rollout[:, target_pos, :].clamp(min=0)  # [H, T]
        value_norm_h = _projected_value_norms(layer_idx, head_count).clamp(min=1e-10)
        mt = mt + (path_mass_h * value_norm_h).sum(dim=0) / float(head_count)

    mt[target_pos:] = 0
    mt = mt.clamp(min=0)
    mt_sum = mt.sum()
    if mt_sum > 1e-12:
        mt = mt / mt_sum
    return mt.detach().cpu().numpy()


def compute_hessian_sensitivity_hvp(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    target_token_id: int,
    pred_pos: int,
    num_samples: int = DEFAULT_HVP_SAMPLES,
) -> np.ndarray:
    seq_len = input_ids.shape[1]
    scores = torch.zeros(seq_len, device=input_ids.device, dtype=torch.float32)
    if seq_len <= 1 or num_samples <= 0:
        return scores.cpu().numpy()

    target_token = int(target_token_id)
    model_dtype = model.get_input_embeddings().weight.dtype
    embeds = model.get_input_embeddings()(input_ids).detach().to(model_dtype)
    embeds.requires_grad_(True)
    logits = model(inputs_embeds=embeds, use_cache=False).logits[0, pred_pos, :].to(torch.float32)
    log_prob = F.log_softmax(logits, dim=-1)[target_token]
    grad = torch.autograd.grad(log_prob, embeds, create_graph=True)[0]

    accum = torch.zeros_like(embeds, dtype=torch.float32)
    for k in range(int(num_samples)):
        r = torch.randint_like(embeds, low=0, high=2, dtype=torch.int32).to(model_dtype)
        r = r * 2 - 1  # Rademacher {+1,-1}
        hvp = torch.autograd.grad(
            (grad * r).sum(),
            embeds,
            retain_graph=(k < int(num_samples) - 1),
            create_graph=False,
        )[0]
        accum = accum + (hvp.to(torch.float32) * r.to(torch.float32)).abs()

    scores = accum.sum(dim=-1).squeeze(0).to(torch.float32) / float(num_samples)
    scores[pred_pos + 1 :] = 0
    total = scores.sum()
    if total > 1e-10:
        scores = scores / total
    return scores.cpu().numpy()


def combine_attr(
    mt_gate: np.ndarray,
    hessian_s: np.ndarray,
    kl_i: np.ndarray,
    beta: float,
    gamma: float,
    fusion_mode: str = "paper",
    mt_floor: float = 0.0,
) -> np.ndarray:
    mt = np.clip(np.asarray(mt_gate, dtype=np.float64), 0.0, None)
    s = np.clip(np.asarray(hessian_s, dtype=np.float64), 0.0, None)
    i = np.clip(np.asarray(kl_i, dtype=np.float64), 0.0, None)

    mode = (fusion_mode or "paper").strip().lower()
    if mode == "log":
        eps = 1e-12
        floor = max(0.0, float(mt_floor))
        mt_safe = np.maximum(mt, floor)
        s_safe = np.maximum(s, eps)
        i_safe = np.maximum(i, eps)
        logits = np.log(mt_safe + eps) + float(beta) * np.log(s_safe) + float(gamma) * np.log(i_safe)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        denom = exp_logits.sum()
        if denom > 1e-12:
            final = exp_logits / denom
        else:
            final = np.zeros_like(exp_logits, dtype=np.float64)
    elif mode == "paper_floor":
        floor = max(0.0, float(mt_floor))
        final = np.maximum(mt, floor) * (beta * s + gamma * i)
    else:
        final = mt * (beta * s + gamma * i)

    normalizer = final.sum()
    if normalizer > 1e-12:
        final = final / normalizer
    return final.astype(np.float64)


def normalize_on_indices(scores: np.ndarray, keep_indices: List[int]) -> np.ndarray:
    score_vec = np.asarray(scores, dtype=np.float64).copy()
    if score_vec.size == 0:
        return score_vec
    mask = np.zeros(score_vec.shape[0], dtype=bool)
    for idx in keep_indices:
        if 0 <= int(idx) < score_vec.shape[0]:
            mask[int(idx)] = True
    score_vec[~mask] = 0.0
    total = score_vec.sum()
    if total > 1e-12:
        score_vec = score_vec / total
    return score_vec


def _build_content_indices(
    segment_ranges: Dict[str, Tuple[int, int]], prompt_len: int, context_len: int
) -> List[int]:
    indices: List[int] = []
    for start, end in segment_ranges.values():
        lo = max(0, int(start))
        hi = min(context_len, int(end))
        if hi > lo:
            indices.extend(range(lo, hi))
    if context_len > prompt_len:
        indices.extend(range(prompt_len, context_len))
    return sorted(set(indices))


def _segment_mass_abs(
    final_scores: np.ndarray, segment_ranges: Dict[str, Tuple[int, int]]
) -> Dict[str, float]:
    score_vec = np.abs(np.asarray(final_scores, dtype=np.float64))
    masses: Dict[str, float] = {}
    for name in ("narrative", "evidence", "question"):
        start, end = segment_ranges.get(name, (0, 0))
        lo = max(0, int(start))
        hi = min(score_vec.shape[0], int(end))
        masses[name] = float(score_vec[lo:hi].sum()) if hi > lo else 0.0
    denom = masses["narrative"] + masses["evidence"] + masses["question"]
    if denom > 1e-12:
        masses = {k: float(v / denom) for k, v in masses.items()}
    return masses


def run_one_example(
    model_name: str,
    narrative: str,
    evidence: str,
    question: str,
    target_k: int,
    beta: float,
    gamma: float,
    masking: str,
    quality: str,
    hvp_samples: int = DEFAULT_HVP_SAMPLES,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    answer_text: str = "",
    fusion_mode: str = "paper",
    mt_floor: float = 0.0,
    target_source_mode: str = "generated",
    packing_mode: str = DEFAULT_PACKING_MODE,
    narrative_cap_ratio: float = DEFAULT_NARRATIVE_CAP_RATIO,
    narrative_cap_min: int = DEFAULT_NARRATIVE_CAP_MIN,
    narrative_cap_max: int = DEFAULT_NARRATIVE_CAP_MAX,
    question_min_tokens: int = DEFAULT_QUESTION_MIN_TOKENS,
) -> Dict[str, Any]:
    """
    Run one end-to-end attribution request using the backend flow from gradio_app.py.
    """
    start = time.perf_counter()
    model_id = get_model_id(model_name)
    model_label = get_model_label(model_name)

    tokenizer = load_tokenizer(model_id)
    attributor = load_attributor(model_id)
    prompt_bundle = build_segmented_prompt(narrative, evidence, question, tokenizer)
    base_prompt_text = prompt_bundle["full_text"]
    generation_prompt_text = f"{base_prompt_text}{ANSWER_CUE}"

    answer_bundle = generate_answer_tokens(attributor.model, tokenizer, generation_prompt_text)
    answer_ids = answer_bundle["answer_token_ids"]
    answer_tokens = answer_bundle["answer_tokens"]
    if not answer_ids:
        raise RuntimeError("Generation produced no answer tokens.")

    requested_k = int(target_k) if target_k is not None else 1
    requested_k = max(1, requested_k)
    generated_k = max(1, min(requested_k, len(answer_ids)))
    if requested_k == 1:
        generated_k = _first_generated_content_index(answer_tokens) + 1
    generated_target_token_id = int(answer_ids[generated_k - 1])

    source_mode = (target_source_mode or "generated").strip().lower()
    if source_mode not in {"generated", "gold_answer", "auto"}:
        source_mode = "generated"
    target_source = "generated"
    target_token_id = generated_target_token_id
    if requested_k == 1 and source_mode in {"gold_answer", "auto"}:
        gold_target_token_id = _first_gold_answer_token_id(tokenizer, answer_text)
        if gold_target_token_id is not None:
            target_token_id = int(gold_target_token_id)
            target_source = "gold_answer"
    if source_mode == "gold_answer" and target_source != "gold_answer":
        target_source = "generated_fallback"

    k = generated_k

    pack_info = _pack_context_ids(
        tokenizer=tokenizer,
        narrative=narrative,
        evidence=evidence,
        question=question,
        answer_prefix_ids=answer_ids[: k - 1],
        max_context_tokens=int(max_context_tokens),
        packing_mode=packing_mode,
        narrative_cap_ratio=float(narrative_cap_ratio),
        narrative_cap_min=int(narrative_cap_min),
        narrative_cap_max=int(narrative_cap_max),
        question_min_tokens=int(question_min_tokens),
    )
    context_ids = list(pack_info["context_ids"])
    original_context_len = int(pack_info["original_context_len"])
    truncated_tokens = int(pack_info["truncated_tokens"])
    prompt_token_count = int(pack_info["prompt_token_count"])
    segment_ranges = {
        "narrative": tuple(pack_info["segment_ranges"]["narrative"]),
        "evidence": tuple(pack_info["segment_ranges"]["evidence"]),
        "question": tuple(pack_info["segment_ranges"]["question"]),
    }

    input_ids = torch.tensor([context_ids], dtype=torch.long, device=attributor.device)
    logical_target_pos = len(context_ids)
    pred_pos = max(0, input_ids.shape[1] - 1)

    mt_input_ids = torch.tensor(
        [context_ids + [target_token_id]],
        dtype=torch.long,
        device=attributor.device,
    )
    mt_gate_full = compute_semantic_flow_mt(
        attributor.model,
        mt_input_ids,
        target_pos=len(context_ids),
        target_token_id=target_token_id,
    )
    mt_gate = mt_gate_full[: len(context_ids)]
    hessian_s = compute_hessian_sensitivity_hvp(
        attributor.model,
        input_ids,
        target_token_id=target_token_id,
        pred_pos=pred_pos,
        num_samples=max(1, int(hvp_samples)),
    )
    kl_i = compute_kl_information(
        attributor.model,
        tokenizer,
        input_ids,
        logical_target_pos,
        masking,
    )

    paragraph_indices = _build_content_indices(
        {
            "narrative": segment_ranges.get("narrative", (0, 0)),
            "evidence": segment_ranges.get("evidence", (0, 0)),
        },
        prompt_len=max(0, prompt_token_count),
        context_len=len(context_ids),
    )
    norm_indices = paragraph_indices
    if not norm_indices:
        norm_indices = _build_content_indices(
            segment_ranges,
            prompt_len=max(0, prompt_token_count),
            context_len=len(context_ids),
        )

    mt_gate = normalize_on_indices(mt_gate, norm_indices)
    hessian_s = normalize_on_indices(hessian_s, norm_indices)
    kl_i = normalize_on_indices(kl_i, norm_indices)
    final_scores = combine_attr(
        mt_gate,
        hessian_s,
        kl_i,
        float(beta),
        float(gamma),
        fusion_mode=fusion_mode,
        mt_floor=float(mt_floor),
    )
    final_scores = normalize_on_indices(final_scores, norm_indices)

    tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in context_ids]
    segment_mass = _segment_mass_abs(final_scores, segment_ranges)

    return {
        "prompt_full_text": generation_prompt_text,
        "prompt_base_text": base_prompt_text,
        "tokens": tokens,
        "context_token_ids": [int(x) for x in context_ids],
        "answer_tokens": answer_tokens,
        "onset_token_text": tokenizer.decode([target_token_id], skip_special_tokens=False),
        "generated_onset_token_text": answer_tokens[k - 1],
        "target_k": int(requested_k),
        "generated_target_k": int(k),
        "segment_token_spans": {
            "narrative": list(segment_ranges["narrative"]),
            "evidence": list(segment_ranges["evidence"]),
            "question": list(segment_ranges["question"]),
        },
        "component_scores": {
            "MT": [float(x) for x in mt_gate.tolist()],
            "S": [float(x) for x in hessian_s.tolist()],
            "KL": [float(x) for x in kl_i.tolist()],
            "final": [float(x) for x in final_scores.tolist()],
        },
        "segment_mass": segment_mass,
        "run_meta": {
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "model_name": model_label,
            "beta": float(beta),
            "gamma": float(gamma),
            "masking": masking,
            "quality": quality,
            "mt_variant": "target_answer_rollout_headwise_value_projection",
            "s_variant": "hvp_hutchinson",
            "s_hvp_samples": int(max(1, int(hvp_samples))),
            "max_context_tokens": int(max_context_tokens),
            "original_context_len": int(original_context_len),
            "truncated_tokens": int(truncated_tokens),
            "context_packing": str(pack_info["context_packing"]),
            "valid_context": bool(pack_info["valid_context"]),
            "segment_counts_original": {
                "narrative": int(pack_info["segment_counts_original"]["narrative"]),
                "evidence": int(pack_info["segment_counts_original"]["evidence"]),
                "question": int(pack_info["segment_counts_original"]["question"]),
            },
            "segment_counts_kept": {
                "narrative": int(pack_info["segment_counts_kept"]["narrative"]),
                "evidence": int(pack_info["segment_counts_kept"]["evidence"]),
                "question": int(pack_info["segment_counts_kept"]["question"]),
            },
            "segment_coverage": {
                "narrative": float(pack_info["segment_coverage"]["narrative"]),
                "evidence": float(pack_info["segment_coverage"]["evidence"]),
                "question": float(pack_info["segment_coverage"]["question"]),
            },
            "segment_norm_scope": "paragraph_only",
            "target_source": target_source,
            "target_source_mode": source_mode,
            "generated_target_token_id": int(generated_target_token_id),
            "target_token_id": int(target_token_id),
            "fusion_mode": str(fusion_mode),
            "mt_floor": float(mt_floor),
            "packing_mode_requested": str(packing_mode),
            "narrative_cap_ratio": float(narrative_cap_ratio),
            "narrative_cap_min": int(narrative_cap_min),
            "narrative_cap_max": int(narrative_cap_max),
            "question_min_tokens": int(question_min_tokens),
        },
    }


__all__ = ["run_one_example", "MODEL_OPTIONS"]