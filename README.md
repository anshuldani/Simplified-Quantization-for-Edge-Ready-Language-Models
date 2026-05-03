# Salient-Quant: Mixed-Precision Quantization for Edge-Ready LLMs

**CS595-2 Course Project — Illinois Institute of Technology**  
Anshul Dani (A20580060) · Rohit Lahori (A20582911)

---

## Overview

Large language models are too heavy for edge deployment. This project implements **Salient-Quant**, a salience-guided mixed-precision post-training quantization (PTQ) method that exploits the well-known 80/20 rule of weight importance: a small fraction of weights carry most of the model's predictive power.

We assign higher bit-width to salient weights and aggressively compress the rest, reaching a target average precision of **1.61 bits** — below the 2-bit barrier — while preserving model accuracy on standard benchmarks. The entire pipeline is post-training (no gradient updates to the model).

---

## The Salient Mask Algorithm

Three phases, applied after training with no weight updates:

**Phase 1 — Salience Scoring**  
Five independent metrics rank every weight by importance, then are combined via a weighted ensemble (α-weights sum to 1):

| Metric | Formula | Insight |
|---|---|---|
| L1 magnitude | `\|w\|` | Large weights tend to matter more |
| L2 magnitude | `w²` | Penalises outliers more strongly |
| Gradient sensitivity | `\|w · ∂L/∂w\|` | Weights whose change moves the loss |
| Hessian diagonal | `E[(∂L/∂w)²]` | Curvature — sharp basin = important |
| Activation-aware | `\|w\| · max_act` | Inspired by SmoothQuant / AWQ |

**Phase 2 — Greedy Bit Allocation**  
Starting from 1-bit for all weights, the allocator upgrades weights in descending salience order (1→2→4 bits) until the average-bit budget is exhausted. Supports weight-level, channel-level, and layer-level granularity.

**Phase 3 — Mixed-Precision Quantization**  
Applies 1-bit (binary sign), 2-bit (INT2), or 4-bit (INT4) quantization per weight group, with block-wise scale optimisation to minimise L2 reconstruction error.

---

## Baselines

| Method | Avg Bits | Type |
|---|---|---|
| FP16 | 16.0 | Uncompressed upper bound |
| Uniform INT2 | 2.0 | All weights 2-bit |
| GPTQ | 4.0 | State-of-the-art PTQ |
| BitNet | 1.58 | Uniform ternary |
| **Salient-Quant (Ours)** | **1.61** | Mixed {1, 2, 4}-bit, salience-guided |

---

## Project Structure

```
salient-quant/
├── src/
│   ├── salience/            # 5 salience metrics + weighted ensemble
│   │   ├── metrics.py       # MagnitudeSalience, GradientSalience, HessianSalience,
│   │   │                    #   ActivationSalience, EnsembleSalience
│   │   └── computer.py      # SalienceComputer: orchestrates calibration passes
│   ├── quantizer/
│   │   ├── kernels.py       # 1-bit / 2-bit / 4-bit quantization kernels
│   │   ├── allocator.py     # Greedy bit allocator (weight / channel / layer)
│   │   └── salient_mask.py  # SalientMaskQuantizer: full 3-phase pipeline
│   ├── baselines/           # FP16, Uniform INT2, GPTQ, BitNet implementations
│   ├── eval/                # Perplexity (WikiText-2, PTB), MMLU, latency, memory
│   └── utils/               # Data loading (C4), logging, visualisation
├── experiments/
│   └── run_experiment.py    # Main entry point — single model, baselines, ablations
├── configs/
│   ├── gpt2_colab.yaml      # GPT-2 Small: quick Colab run (~15 min T4)
│   ├── gpt2_full.yaml       # GPT-2 Medium: full eval + ablations (~60 min T4)
│   └── llama_full.yaml      # LLaMA-3.2-1B: full eval + MMLU (~90 min T4)
├── notebooks/
│   └── salient_quant_colab.ipynb  # Complete Colab experiment runner
├── scripts/
│   ├── run_gpt2.sh          # SLURM: GPT-2 pipeline
│   └── run_llama.sh         # SLURM: LLaMA-3.2 pipeline
├── tests/
│   └── test_core.py         # pytest suite (31+ unit tests)
└── results/                 # Output JSONs + plots (git-ignored, kept via .gitkeep)
```

---

## Milestones

| Milestone | Status | Description |
|---|---|---|
| A | ✅ Complete | Core implementation: salience metrics, allocator, quantization kernels |
| B | ✅ Complete | GPT-2 full eval, baseline comparisons, 6 ablation studies |
| C | ✅ Complete | LLaMA-3.2-1B full eval, all ablations, MMLU 5-shot |
| Final | In Progress | 12-page report, final presentation |

---

## Running on Google Colab (Recommended)

The easiest way to reproduce all experiments is the included Colab notebook, which handles setup, memory management, and Drive checkpointing automatically.

**[`notebooks/salient_quant_colab.ipynb`](notebooks/salient_quant_colab.ipynb)**

### Prerequisites

1. **GPU runtime** — Runtime → Change runtime type → T4 GPU (free tier works)
2. **HuggingFace token** with LLaMA-3.2 access:
   - Request access at https://huggingface.co/meta-llama/Llama-3.2-1B
   - Generate a token at https://huggingface.co/settings/tokens
   - Paste it into Step 3 of the notebook
3. **Google Drive** (optional but recommended) — mounts automatically in Step 4 to preserve results across session resets

### Notebook Walkthrough

| Cell | What it does | Time |
|---|---|---|
| Step 0 | Verify T4 GPU is attached | — |
| Step 1 | Clone this repo | — |
| Step 2 | `pip install -r requirements_colab.txt` | ~2 min |
| Step 3 | HuggingFace login (for LLaMA-3.2) | — |
| Step 4 | Mount Google Drive | — |
| Experiment 1 | GPT-2 Small: baselines + ours | ~15 min |
| Experiment 2 | GPT-2 Medium: full eval + ablations | ~60 min |
| Experiment 3 | LLaMA-3.2-1B: ablations (all 6 types) | ~5–6 hrs |
| Summary | Print results tables + display plots | — |

> **Note on LLaMA-3.2 memory:** The full LLaMA experiment runs 21 ablation configs sequentially. Each config reloads the model and runs calibration. If the session OOMs mid-run, per-config checkpoints are saved automatically — restart the runtime and rerun; completed configs are skipped.

---

## Local / SLURM Setup

```bash
# Create environment
conda create -n salient-quant python=3.10
conda activate salient-quant
pip install -r requirements.txt   # includes torch; use requirements_colab.txt on Colab
```

Requires a CUDA-capable GPU. Tested on T4 (16 GB) for all experiments.  
LLaMA-3.2-1B needs ≥ 8 GB VRAM (loaded in fp16).

### Quick Start

```bash
# Sanity check — GPT-2 Small, all phases, ~15 min
python experiments/run_experiment.py --config configs/gpt2_colab.yaml

# Full GPT-2 Medium eval + 6 ablation studies
python experiments/run_experiment.py --config configs/gpt2_full.yaml

# LLaMA-3.2-1B ablations (memory-safe config)
python experiments/run_experiment.py --config configs/llama_full.yaml
```

### SLURM

```bash
bash scripts/run_gpt2.sh    # GPT-2 Medium
bash scripts/run_llama.sh   # LLaMA-3.2-1B
```

Results (JSON + plots) are written to `results/<run_name>/`.

---

## Running Tests

```bash
pytest tests/ -v
```

Covers: quantization kernels, bit allocator, salience metrics, ensemble combination, and an end-to-end smoke test on a synthetic MLP.

---

## Key Implementation Notes

- **No `device_map="auto"`** — accelerate's CPU-offload hooks route gradients through CPU during calibration (~9 s/batch instead of <0.5 s/batch on T4). Models are loaded with plain `from_pretrained` + `.to(device)`.
- **Salience accumulators on CPU** — gradient and Fisher accumulators for LLaMA-1B would occupy ~4 GB on VRAM each. All accumulators are kept on CPU and drained per-parameter during assembly.
- **Hessian excluded from T4 runs** — gradient + Hessian fp32 accumulators combined exceed Colab's ~12 GB RAM limit. The 4-metric ensemble (L1, L2, gradient, activation) is used for all non-salience-metric ablations.
- **fp16 salience map** — salience tensors are downcast to fp16 before the allocator runs, halving CPU RAM from 3.9 GB to 1.95 GB while preserving ranking accuracy.
- **Per-config ablation checkpoints** — each completed ablation config is saved to `ablation_<type>_partial.json`; a crashed run resumes from the last checkpoint on restart.

---

## Dependencies

| Group | Packages |
|---|---|
| Core | PyTorch ≥ 2.1, Transformers ≥ 4.40, Accelerate, Datasets |
| Evaluation | lm-eval ≥ 0.4.2 |
| Baselines | auto-gptq, bitsandbytes, optimum |
| Utils | PyYAML, tqdm, rich, psutil, matplotlib, seaborn |

Full list: `requirements.txt` (local) · `requirements_colab.txt` (Colab)
