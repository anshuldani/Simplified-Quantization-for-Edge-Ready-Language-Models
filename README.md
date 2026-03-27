# Salient-Quant: Mixed-Precision Quantization for Edge-Ready LLMs

**CS595-2 Course Project — Illinois Institute of Technology**

Anshul Dani (A20580060) 

Rohit Lahori (A20582911)

---

## Overview

Large language models are too heavy for edge deployment. This project implements a **salience-guided mixed-precision post-training quantization (PTQ)** method that exploits the well-known 80/20 rule of weight importance: a small fraction of weights carry most of the model's predictive power.

Our approach assigns higher bit-width to salient weights and aggressively compresses the rest, reaching a target average precision of **1.61 bits** — below the 2-bit barrier — while preserving model accuracy on standard benchmarks.

---

## The Salient Mask Algorithm

The method runs in three phases, applied post-training with no gradient updates:

**Phase 1 — Salience Scoring**
Five independent metrics rank every weight by importance:
- L1 / L2 magnitude
- Gradient sensitivity (`|w · ∂L/∂w|`)
- Hessian diagonal (Fisher information approximation)
- Activation-aware scaling (inspired by SmoothQuant)

Scores are combined via a weighted ensemble (α weights sum to 1).

**Phase 2 — Greedy Bit Allocation**
Starting from 1-bit for all weights, the allocator upgrades weights in descending salience order (1→2→4 bits) until the average bit budget (1.61b) is exhausted. Supports weight-level, channel-level, and layer-level granularity.

**Phase 3 — Mixed-Precision Quantization**
Applies 1-bit (binary sign), 2-bit (INT2), or 4-bit (INT4) quantization per weight, with block-wise scale optimization to minimize L2 reconstruction error.

---

## Baselines

| Method | Avg Bits | Type |
|---|---|---|
| FP16 | 16.0 | Uncompressed upper bound |
| Uniform INT2 | 2.0 | All weights 2-bit |
| GPTQ | 4.0 | State-of-the-art PTQ |
| BitNet | 1.58 | Uniform ternary |
| **Ours (Salient-Quant)** | **1.61** | Mixed {1, 2, 4}-bit, salience-guided |

---

## Project Structure

```
salient-quant/
├── src/
│   ├── salience/         # Salience metrics (5 metrics + ensemble)
│   ├── quantizer/        # Mixed-precision quantizer (1b / 2b / 4b kernels)
│   ├── baselines/        # FP16, Uniform INT2, GPTQ, BitNet
│   ├── eval/             # Perplexity, MMLU, latency, memory profiling
│   └── utils/            # Data loading, logging, visualization
├── experiments/
│   ├── run_experiment.py # Main entry point
│   └── profile_model.py  # Bottleneck profiling
├── configs/              # YAML experiment configs
│   ├── gpt2_quick.yaml   # Milestone A: quick sanity check
│   ├── gpt2_full.yaml    # Milestone B: full GPT-2 eval + ablations
│   └── llama_full.yaml   # Milestone C: LLaMA-3.2 full eval
├── scripts/
│   ├── run_gpt2.sh       # SLURM: GPT-2 pipeline
│   └── run_llama.sh      # SLURM: LLaMA-3.2 pipeline
├── tests/                # pytest suite (31+ unit tests)
└── results/              # Output CSVs, plots (git-ignored, kept via .gitkeep)
```

---

## Setup

```bash
conda create -n salient-quant python=3.10
conda activate salient-quant
pip install -r requirements.txt
```

Requires a CUDA-capable GPU. Tested on A100 40GB for LLaMA-3.2; GPT-2 experiments run on smaller GPUs.

---

## Quick Start

```bash
# Milestone A — core pipeline on GPT-2 small (fast sanity check)
python experiments/run_experiment.py --config configs/gpt2_quick.yaml

# Milestone B — full GPT-2-medium eval + ablations (SLURM)
bash scripts/run_gpt2.sh

# Milestone C — LLaMA-3.2 full eval (SLURM)
bash scripts/run_llama.sh
```

Results (CSVs, plots) are written to `results/`.

---

## Running Tests

```bash
pytest tests/ -v
```

Covers quantization kernels, bit allocator, salience metrics, and an end-to-end smoke test on a small synthetic MLP.

---

## Milestones

| Milestone | Status | Description |
|---|---|---|
| A | In Progress | Core implementation: salience, allocator, kernels |
| B | Pending | GPT-2 full eval, baseline comparisons, ablation study |
| C | Pending | LLaMA-3.2 full eval, all ablations, MMLU |
| Final | Pending | 12-page report, final presentation |

---

## Dependencies

Core: PyTorch ≥ 2.1, Transformers ≥ 4.40, Accelerate, Datasets
Eval: lm-eval ≥ 0.4.2
Quantization baselines: auto-gptq, bitsandbytes, optimum
Tracking: Weights & Biases

Full list in `requirements.txt`.
