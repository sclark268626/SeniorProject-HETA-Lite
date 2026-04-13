import json
import random
import sys
from typing import Dict, List, Tuple

def build_title_to_sents(context: List[List]) -> Dict[str, List[str]]:
    # context: [[title, [sent1, sent2, ...]], ...]
    mp = {}
    for title, sents in context:
        mp[title] = sents
    return mp

def build_supporting_map(supporting_facts: List[List]) -> Dict[str, List[int]]:
    # supporting_facts: [[title, sent_idx], ...]
    sup = {}
    for title, idx in supporting_facts:
        sup.setdefault(title, []).append(int(idx))
    # keep unique + sorted
    for t in sup:
        sup[t] = sorted(list(set(sup[t])))
    return sup

def extract_evidence_text(title_to_sents: Dict[str, List[str]],
                          sup_map: Dict[str, List[int]],
                          window: int = 0) -> str:
    chunks = []
    for title, idxs in sup_map.items():
        sents = title_to_sents.get(title, [])
        picked = []
        for i in idxs:
            lo = max(0, i - window)
            hi = min(len(sents) - 1, i + window)
            for j in range(lo, hi + 1):
                picked.append(sents[j].strip())
        # dedup while preserving order
        seen = set()
        picked2 = []
        for s in picked:
            if s and s not in seen:
                picked2.append(s)
                seen.add(s)
        if picked2:
            chunks.append(f"[{title}] " + " ".join(picked2))
    return "\n".join(chunks).strip()

def extract_narrative_text(context: List[List],
                           sup_titles: set,
                           max_distractors: int = 3,
                           seed: int = 0) -> str:
    rnd = random.Random(seed)
    distractor_titles = [title for title, _ in context if title not in sup_titles]
    rnd.shuffle(distractor_titles)
    distractor_titles = distractor_titles[:max_distractors]

    title_to_sents = build_title_to_sents(context)
    chunks = []
    for title in distractor_titles:
        sents = title_to_sents.get(title, [])
        para = " ".join([s.strip() for s in sents if s.strip()])
        if para:
            chunks.append(f"[{title}] {para}")
    return "\n".join(chunks).strip()

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 convert_hotpot_to_heta.py <hotpot_json> <out_jsonl> [max_distractors=3] [window=0] [seed=0]")
        sys.exit(1)

    hotpot_path = sys.argv[1]
    out_path = sys.argv[2]
    max_distractors = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    window = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    base_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    with open(hotpot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(out_path, "w", encoding="utf-8") as w:
        for idx, ex in enumerate(data):
            q = ex.get("question", "").strip()
            ans = ex.get("answer", "").strip()
            context = ex.get("context", [])
            supporting = ex.get("supporting_facts", [])

            title_to_sents = build_title_to_sents(context)
            sup_map = build_supporting_map(supporting)
            sup_titles = set(sup_map.keys())

            evidence = extract_evidence_text(title_to_sents, sup_map, window=window)
            narrative = extract_narrative_text(context, sup_titles, max_distractors=max_distractors, seed=base_seed + idx)

            # HETA-style structured sample
            out = {
                "id": ex.get("_id", f"hotpot_{idx}"),
                "source": "hotpotqa_fullwiki",
                "segments": {
                    "narrative": narrative,
                    "evidence": evidence,
                    "question": q
                },
                "answer": ans,
                "meta": {
                    "max_distractors": max_distractors,
                    "evidence_window": window,
                    # fill these after running your model:
                    "target_k": 1,                 # default onset token (k=1)
                    "onset_token_text": None,
                    "onset_token_index": None
                }
            }
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[DONE] Wrote {out_path} as JSONL")

if __name__ == "__main__":
    main()
