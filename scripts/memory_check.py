#!/usr/bin/env python3
"""GPU memory profiling for HETA Lite.

Loads the model, runs attribution on sample prompts, and reports peak GPU memory.
Fails with exit code 1 if peak memory exceeds the threshold (default 14GB).
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Adjust if testing a different model
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MEMORY_LIMIT_GB = 14.0

SAMPLE_PROMPTS = [
    "What causes the seasons on Earth?",
    "If a train travels 120 miles in 2 hours, what is the average speed?",
    (
        "In the early 1900s, scientists debated how the brain stores memory. "
        "Some believed memory was localized to specific regions, while others argued "
        "it was distributed across networks. Today, evidence shows memory involves "
        "both localized and distributed processes, depending on the type and time scale."
    ),
]


def format_bytes(b: int) -> str:
    return f"{b / (1024 ** 3):.2f} GB"


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device available. Run this on a GPU machine.")
        return 0

    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")
    print(f"Memory limit: {MEMORY_LIMIT_GB} GB")
    print()

    torch.cuda.reset_peak_memory_stats()

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    mem_after_load = torch.cuda.max_memory_allocated()
    print(f"Memory after model load: {format_bytes(mem_after_load)}")
    print()

    # Import and create attributor
    sys.path.insert(0, ".")
    from heta_demo import HETAAttributor

    attributor = HETAAttributor(model, tokenizer, device)

    # Run attribution on each sample prompt
    for i, prompt in enumerate(SAMPLE_PROMPTS, 1):
        torch.cuda.reset_peak_memory_stats()
        preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
        print(f"[{i}/{len(SAMPLE_PROMPTS)}] {preview}")

        tokens, scores, target_pos = attributor.attribute(prompt)

        peak = torch.cuda.max_memory_allocated()
        print(f"  Tokens: {len(tokens)} | Peak memory: {format_bytes(peak)}")

    # Final check
    overall_peak = torch.cuda.max_memory_allocated()
    peak_gb = overall_peak / (1024 ** 3)

    print()
    print(f"Overall peak memory: {format_bytes(overall_peak)}")

    if peak_gb > MEMORY_LIMIT_GB:
        print(f"FAIL: Peak memory {peak_gb:.2f} GB exceeds limit of {MEMORY_LIMIT_GB} GB")
        return 1

    print(f"PASS: Peak memory is within the {MEMORY_LIMIT_GB} GB limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
