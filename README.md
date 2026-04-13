# Project 1: HETA Lite  
**Real-Time Token Attribution for Decoder-Only LLMs**

This project implements **HETA Lite**, a public-facing system for explaining why a decoder-only large language model predicts a specific token.

The system is based on **HETA (Hessian Enhanced Token Attribution)**:  
https://vishalpramanik.github.io/hetaproject.html

This is a **deployment-focused engineering project**, not a research project. The core HETA algorithm is provided in prototype form; the goal of this project is to make it **fast, reliable, and usable** as a real-time web application.

---

## Overview

HETA Lite provides **token-level explanations** for language model predictions.  
Given an input prompt and a selected target token, the system produces a heatmap showing which input tokens most strongly influenced the prediction.

The underlying HETA method combines:
- Attention flow tracing
- Curvature-based sensitivity (Hessian estimation)
- Information-theoretic masking (KL divergence)

Users do **not** need to understand the mathematical details to use the system.

---

## User Workflow

A typical user interaction follows these steps:

1. Paste a prompt (e.g., a question or paragraph)
2. Select a target token position  
   (“Why did the model predict *this* word?”)
3. View:
   - A token-level attribution heatmap
   - A ranked list of the most influential input tokens
4. Export a structured JSON report for debugging or auditing purposes

---

## System Constraints

The system must satisfy the following **non-negotiable constraints**:

- **Single GPU deployment** with ≤ 16GB VRAM  
  (e.g., T4, RTX 3060, RTX 4060)
- **Model size ≈ 3B parameters**  
  - Qwen2.5 3B recommended  
  - Fallback to 1–2B models if required
- **Interactive latency**  
  - Responses in seconds, not minutes  
  - p95 latency target defined during development
- **Reproducible Docker build**  
  - Must run without manual edits by a TA or reviewer

---

## Technical Scope

### A. HETA Core Integration

The provided prototype implements the three main HETA components:

- Semantic transition gate (attention value rollout)
- Curvature term (Hessian-based sensitivity)
- Information gain via masking (KL divergence)

Project responsibilities include:
- Wiring these components into a single inference pipeline
- Handling edge cases (empty input, long prompts, timeouts)
- Adding caching where applicable

---

### B. Performance Optimization

To support a public demo, the system must be optimized for speed and stability:

- Implement a **quality vs. latency control**  
  (e.g., fewer Hessian samples for faster responses)
- Cache:
  - Tokenization results
  - KV states when possible
- Enforce strict input limits
- Profile GPU memory usage and fix memory leaks

---

### C. Frontend & Visualization

The system includes a lightweight web interface (Gradio or Streamlit) with:

- Token-level heatmap visualization
- Hover details (raw score, percentile)
- Ranked list of influential tokens
- Export functionality:
  - JSON attribution report
  - PNG heatmap snapshot
- Graceful handling of request queuing and slow responses

---

### D. Deployment & Hardening

The system is deployed as a production-style GPU service:

- Dockerized backend
- Reverse proxy and request queue
- Rate limiting
- Telemetry:
  - Latency metrics
  - Error counts  
  - **No prompt logging**

---

### E. Optional: Quantization Exploration

Optional extension work includes:

- Running the model in 4-bit or 8-bit quantized form
  (e.g., bitsandbytes or GPTQ)
- Measuring:
  - Speed improvements
  - Memory savings
  - Any degradation in attribution quality
- Exploring whether Hessian-related computations remain stable under quantization

---

## Evaluation Criteria

The system is evaluated along four dimensions:

### Correctness
- Removing the top-K attributed tokens should reduce the target token’s probability more than removing random tokens.

### Performance
- The system must meet its defined p95 latency target on a standard prompt length.

### Reliability
- The service must survive a **2-hour soak test** with scripted traffic.

### Usability
- A non-expert user should be able to obtain results in under **60 seconds**.

---

## Team Requirements

- Team size: **4–5 students**
- Minimum GPA:
  - 3.5 overall
  - 3.75 in major
- Required skill coverage:
  - Transformer internals (hooks, forward passes, GPU debugging)
  - Web deployment (Docker, reverse proxy, rate limiting)
  - Evaluation and benchmarking (metrics, reproducibility)

---

## Project Timeline

| Weeks | Milestone |
|------|-----------|
| 1–3  | Design Review (Architecture, model choice, acceptance tests) |
| 4–6  | Alpha Demo (Local Docker, minimal UI, one evaluation script) |
| 7–9  | Beta Deployment (Remote deployment, monitoring, baseline evaluation) |
| 10–14 | Final Release (All constraints met, final report and recorded demo) |

---

## References

- **HETA Project Page**  
  https://vishalpramanik.github.io/hetaproject.html
