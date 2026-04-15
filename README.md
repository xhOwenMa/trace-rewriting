# Trace Rewriting

Source code of our paper:

> **[Protecting Language Models Against Unauthorized Distillation through Trace Rewriting](https://arxiv.org/abs/2602.15143)**,
> by Xinhang Ma, William Yeoh, Ning Zhang, Yevgeniy Vorobeychik

## Abstract

Knowledge distillation is a widely adopted technique for transferring capabilities from LLMs to smaller, more efficient student models. However, unauthorized use of knowledge distillation takes unfair advantage of the considerable effort and cost put into developing frontier models. We investigate methods for modifying teacher-generated reasoning traces to achieve two objectives that deter unauthorized distillation: (1) anti-distillation, or degrading the training usefulness of query responses, and (2) API watermarking, which embeds verifiable signatures in student models. We introduce several approaches for dynamically rewriting a teacher's reasoning outputs while preserving answer correctness and semantic coherence. Two of these leverage the rewriting capabilities of LLMs, while others use gradient-based techniques. Our experiments show that a simple instruction-based rewriting approach achieves a strong anti-distillation effect while maintaining or even improving teacher performance. Furthermore, we show that our rewriting approach also enables highly reliable watermark detection with essentially no false alarms.

## Repository Structure

```
trace-rewriting/
├── README.md
├── requirements.txt
├── config/
│   ├── generate.yaml          # Trace generation & rewriting config
│   ├── distill.yaml           # Student distillation config
│   └── optimize.yaml          # Prompt optimization config
├── src/
│   ├── utils.py               # Config management, dataset loaders, system prompts
│   ├── generate.py            # Generate teacher traces, rewrite them, and score accuracy
│   ├── distill.py             # Student model SFT training
│   └── evaluate.py            # Student model evaluation (raw + answer-forcing)
├── optimize/
│   ├── rewrite_candidates.py  # Stage 1: rewrite traces with each candidate instruction
│   ├── score_candidates.py    # Stage 2: score candidates via proxy student training
│   └── propose_candidates.py  # Stage 3: LLM-based instruction evolution (OPRO)
└── data/
    ├── gsm8k/
    │   ├── semantic/
    │   └── optimized/
    └── math/
        ├── semantic/
        └── optimized/
```

## Pre-generated Data

We include pre-generated reasoning trace datasets for GSM8K and MATH so you can skip the trace generation step and directly reproduce the distillation and evaluation results. Each dataset is a HuggingFace `datasets` directory with the following columns:

| Column | Description |
|---|---|
| `problem` | The original problem |
| `solution` | Ground-truth solution |
| `original_trace` | Clean teacher-generated reasoning trace |
| `rewrite_trace` | Rewritten trace (semantic or optimized) |

Two rewriting variants are provided:

- **`data/<dataset>/semantic/`** — Traces rewritten with the semantic instruction (Section 5.1.1 of the paper)
- **`data/<dataset>/optimized/`** — Traces rewritten with the OPRO-optimized instruction (Section 5.1.2 of the paper)

## Setup

Tested with Python 3.11.

See [`requirements.txt`](requirements.txt) for the core packages and the exact pinned versions used in the paper. We do not recommend installing it directly with `pip install -r requirements.txt` — several dependencies (`torch`, `vllm`, `flash-attn`) need toolchain-specific install flags (CUDA build, `--no-build-isolation`, etc.) and a one-shot `pip install` will usually fail on a fresh environment. 

## Reproducing the paper's main GSM8K result with Optimized Rewrite Instruction

The full pipeline — teacher trace generation → rewriter rewriting + answer-forcing → student LoRA distillation → student evaluation — is wrapped in a single script: [`run_ad_optimized_inst_gsm8k.sh`](run_ad_optimized_inst_gsm8k.sh).

The script auto-detects the pre-generated traces under `data/gsm8k/optimized/` and skips stages 1–2 (teacher generation + rewriter rewriting). This means a default reproduction only needs GPU memory for the student (LoRA SFT on Llama-3.2-3B + vLLM eval) and not for the larger rewriter. Delete or rename `data/gsm8k/optimized/` to force regeneration from scratch.

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...      # required
export NUM_GPUS=2                          # optional (default 2)
export SCRATCH_BASE=/tmp/trace-rewriting-cache   # optional (this is the default)

bash run_ad_optimized_inst_gsm8k.sh
```

Settings to configure before running:

| Variable | Required? | Default | Notes |
|---|---|---|---|
| `HUGGING_FACE_HUB_TOKEN` | **yes** | — | Token with access to `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (teacher) and `meta-llama/Llama-3.2-3B` (student). The script aborts early if unset. |
| `NUM_GPUS` | effectively yes | `2` | Used as `--tensor-parallel-size` for vLLM and `--num_processes` for distillation. The default rewriter `openai/gpt-oss-120b` needs **at least 2× A100 80GB (or 1× H100)**. Set to `1` only if you also swap in a smaller rewriter in `config/generate.yaml`. |
| `SCRATCH_BASE` | no | `/tmp/trace-rewriting-cache` | Parent dir for HF / vLLM / torch caches, generated traces, and distilled student weights. Combined footprint **can exceed 100 GB**; point at a larger filesystem if `/tmp` is limited. |
| `RUN`, `STUDENT_MODEL`, `VLLM_PORT`, `VLLM_EXTRA_ARGS` | no | see the script header | Optional knobs for run naming, alternate students, port conflicts, and extra vLLM flags. |

Lightweight results (eval metrics, traces dataset, sample report, effective config) are copied into `results/gsm8k/${RUN}/` when the run finishes; model weights stay under `${SCRATCH_BASE}/trace-rewriting/`.

For datasets or model combinations other than the paper's main setup, edit `config/generate.yaml` / `config/distill.yaml`; the scripts in `src/` are also usable standalone.

### Prompt Optimization (Optional)

To reproduce the OPRO-style rewrite-instruction evolution (Section 5.1.2), run the three stages under `optimize/` in a loop, driven by `config/optimize.yaml`. This step additionally requires an OpenRouter API key:

```bash
export OPENROUTER_API_KEY=<your-key>
```

See the docstrings in `optimize/rewrite_candidates.py`, `optimize/score_candidates.py`, and `optimize/propose_candidates.py` for per-stage details.

## Note on gradient-based methods

This repository only includes the two LLM-based rewriting methods from the paper (semantic and OPRO-optimized instructions), which are the ones used for the main results. If you are interested in the gradient-based rewriting variants, please contact the first author directly.

## Citation

```bibtex
@article{ma2026protecting,
  title={Protecting Language Models Against Unauthorized Distillation through Trace Rewriting},
  author={Ma, Xinhang and Yeoh, William and Zhang, Ning and Vorobeychik, Yevgeniy},
  journal={arXiv preprint arXiv:2602.15143},
  year={2026}
}
```
