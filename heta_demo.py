#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

MODEL_OPTIONS = {
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "GPT-J-6B": "EleutherAI/gpt-j-6B",
    "Phi-3-Medium-4K-Instruct (14B)": "microsoft/Phi-3-medium-4k-instruct",
    "Llama-3.1-70B": "meta-llama/Llama-3.1-70B-Instruct",
}


class HETAAttributor:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compute_attention_rollout(self, input_ids, target_pos):
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True, use_cache=False)

        attentions = outputs.attentions
        seq_len = input_ids.shape[1]

        if not attentions:
            return torch.ones(seq_len, device=self.device) / seq_len

        attn_avg = torch.stack([a.squeeze(0).mean(dim=0) for a in attentions], dim=0)

        eye = torch.eye(seq_len, device=self.device)
        attn_residual = 0.5 * attn_avg + 0.5 * eye.unsqueeze(0)
        attn_residual = attn_residual / attn_residual.sum(dim=-1, keepdim=True).clamp(
            min=1e-10
        )

        rollout = attn_residual[0]
        for layer_attn in attn_residual[1:]:
            rollout = torch.matmul(layer_attn, rollout)

        result = rollout[target_pos, :].clone()
        result[target_pos:] = 0

        if result.sum() > 1e-10:
            result = result / result.sum()
        return result

    def compute_gradient_sensitivity(self, input_ids, target_pos):
        seq_len = input_ids.shape[1]
        embeddings = self.model.get_input_embeddings()
        input_embeds = embeddings(input_ids).clone().detach().requires_grad_(True)

        outputs = self.model(inputs_embeds=input_embeds, use_cache=False)
        logits = outputs.logits

        pred_pos = max(0, target_pos - 1)
        target_token = input_ids[0, target_pos]
        log_probs = F.log_softmax(logits[0, pred_pos, :], dim=-1)
        target_log_prob = log_probs[target_token]

        self.model.zero_grad()
        target_log_prob.backward()

        grad_norms = input_embeds.grad.norm(dim=-1).squeeze(0) ** 2

        grad_norms[target_pos:] = 0

        if grad_norms.sum() > 1e-10:
            grad_norms = grad_norms / grad_norms.sum()
        return grad_norms.detach()

    def compute_kl_divergence(self, input_ids, target_pos):
        seq_len = input_ids.shape[1]
        kl_scores = torch.zeros(seq_len, device=self.device)
        pred_pos = max(0, target_pos - 1)

        with torch.no_grad():
            orig_logits = self.model(input_ids, use_cache=False).logits[0, pred_pos, :]
            orig_probs = F.softmax(orig_logits, dim=-1)

        mask_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        for pos in range(target_pos):
            masked_ids = input_ids.clone()
            masked_ids[0, pos] = mask_token_id

            with torch.no_grad():
                masked_logits = self.model(masked_ids, use_cache=False).logits[
                    0, pred_pos, :
                ]
                masked_probs = F.softmax(masked_logits, dim=-1)

            kl = (
                orig_probs
                * (torch.log(orig_probs + 1e-10) - torch.log(masked_probs + 1e-10))
            ).sum()
            kl_scores[pos] = kl.clamp(min=0)

        if kl_scores.sum() > 1e-10:
            kl_scores = kl_scores / kl_scores.sum()
        return kl_scores

    def attribute(self, text, target_pos=None):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]

        if target_pos is None:
            target_pos = seq_len - 1
        target_pos = min(max(1, target_pos), seq_len - 1)

        attn = self.compute_attention_rollout(input_ids, target_pos)
        grad = self.compute_gradient_sensitivity(input_ids, target_pos)
        kl = self.compute_kl_divergence(input_ids, target_pos)

        scores = 0.2 * attn + 0.4 * grad + 0.4 * kl
        scores[target_pos:] = 0
        if scores.sum() > 1e-10:
            scores = scores / scores.sum()

        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]
        return tokens, scores.cpu().numpy(), target_pos


def main():
    parser = argparse.ArgumentParser(description="HETA Token Attribution Demo")
    parser.add_argument(
        "--model",
        default="Qwen2.5-3B",
        choices=list(MODEL_OPTIONS.keys()),
        help="Model option",
    )
    parser.add_argument("--text", default=None, help="Input text")
    parser.add_argument(
        "--target-pos", type=int, default=None, help="Target token position"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    model_id = MODEL_OPTIONS[args.model]
    print(f"Loading {args.model} ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded.\n")

    attributor = HETAAttributor(model, tokenizer, args.device)

    # Interactive loop
    while True:
        if args.text:
            text = args.text
        else:
            text = input("Enter text (or 'quit'): ").strip()
            if text.lower() == "quit":
                break
            if not text:
                continue

        # Show tokens
        inputs = tokenizer(text, return_tensors="pt")
        tokens = [tokenizer.decode([tid]) for tid in inputs.input_ids[0]]
        print(f"\nTokens ({len(tokens)}):")
        for i, t in enumerate(tokens):
            print(f"  {i}: {repr(t)}")

        # Get target position
        if args.target_pos is not None:
            target_pos = args.target_pos
        elif args.text:
            target_pos = (
                len(tokens) - 1
            )  # Default to last token in non-interactive mode
        else:
            pos_str = input(
                f"Target position [0-{len(tokens)-1}, default={len(tokens)-1}]: "
            ).strip()
            target_pos = int(pos_str) if pos_str else len(tokens) - 1

        # Compute attribution
        print()
        tokens, scores, target_pos = attributor.attribute(text, target_pos)

        # Display results
        print(f"\nTarget: {repr(tokens[target_pos])} (position {target_pos})")
        print("\nAttribution scores:")
        for i, (tok, score) in enumerate(zip(tokens, scores)):
            if i == target_pos:
                print(f"  {i}: {repr(tok):20} {score*100:5.1f}%  <-- TARGET")
            elif score > 0.01:
                bar = "*" * int(score * 50)
                print(f"  {i}: {repr(tok):20} {score*100:5.1f}%  {bar}")
            else:
                print(f"  {i}: {repr(tok):20} {score*100:5.1f}%")

        # Show top tokens
        top_k = min(5, target_pos)
        top_idx = np.argsort(scores)[::-1][:top_k]
        print(f"\nTop {top_k} contributors:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank}. {repr(tokens[idx]):20} ({scores[idx]*100:.1f}%)")

        if args.text:
            break
        print()


if __name__ == "__main__":
    main()
