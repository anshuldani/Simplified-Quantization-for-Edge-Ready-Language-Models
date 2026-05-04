# Salient-Quant: Mixed-Precision Quantization for Edge-Ready LLMs

**CS595-2 Final Project — Illinois Institute of Technology**
Anshul Dani (A20580060) · Rohit Lahori (A20582911)

**Project type:** Engineering / Measurement (with research-prototype elements).

📄 **[Final report (9 pages, PDF)](report/report.pdf)** · 📊 **[Ablation JSONs](results/llama_full/)** · 🧪 **[Colab reproduction notebook](notebooks/salient_quant_colab.ipynb)**

---

## Overview

Large language models are too heavy for edge deployment. This project implements **Salient-Quant**, a salience-guided mixed-precision post-training quantization (PTQ) method that exploits the heavy-tailed structure of weight importance: a small fraction of weights carry most of the predictive power.

The method assigns higher bit-width to salient weights and aggressively compresses the rest, reaching a target average precision of **1.61 bits/weight** — below the 2-bit barrier — while avoiding the catastrophic accuracy collapse that uniform sub-2-bit methods suffer. The entire pipeline is post-training (no gradient updates to model parameters).

---

## Headline results

### GPT-2 medium (345M params) — WikiText-2 perplexity

| Method | Bits/W | WT2 PPL ↓ | PTB PPL ↓ | Size (GB) | Compression |
|---|---:|---:|---:|---:|---:|
| FP16 (baseline) | 16.0 | 21.64 | 22.53 | 0.61 | 1.00× |
| Uniform INT2 | 2.0 | 1,451,721 | 1,310,861 | 0.08 | 7.97× |
| BitNet 1-bit | 1.0 | 1,556,522 | 1,448,814 | 0.04 | 15.9× |
| **SMQ (Ours)** | **1.61** | **541.05** | **547.75** | **0.06** | **9.94×** |

**SMQ is 2,683× better than Uniform INT2** on WikiText-2 *while using less memory*, and 2,874× better than BitNet at the same compression class.

### LLaMA-3.2-1B — WikiText-2 perplexity

| Method | Bits/W | WT2 PPL ↓ | PTB PPL ↓ | Size (GB) | Compression |
|---|---:|---:|---:|---:|---:|
| FP16 (baseline) | 16.0 | 10.96 | 11.32 | 2.47 | 1.00× |
| Uniform INT2 | 2.0 | 164,165 | 168,767 | 0.31 | 8.0× |
| BitNet | 1.58 | 99,786 | 104,487 | 0.16 | 10.1× |
| **SMQ (Ours)** | **1.61** | **35,831** | **31,859** | **0.24** | **10.4×** |

SMQ is 4.6× better than Uniform INT2 and 2.8× better than BitNet at the same compression. Sub-2-bit PTQ on a 1B model is more brittle than on 345M (the FP16 gap widens with scale) — closing this gap is on the future-work list.

---

## The Salient Mask Algorithm

Three phases applied after training, with no weight updates to the model:

**Phase 1 — Salience Scoring.** Five metrics rank every weight by importance, then are combined via a weighted ensemble (α weights sum to 1):

| Metric | Formula | Insight |
|---|---|---|
| L1 magnitude | `\|w\|` | Large weights tend to matter more |
| L2 magnitude | `w²` | Penalises outliers more strongly |
| Gradient sensitivity | `\|w · ∂L/∂w\|` | Weights whose change moves the loss |
| Hessian diagonal | `E[(∂L/∂w)²]` | Curvature — sharp basin = important |
| Activation-aware | `\|w\| · std(x)` | Inspired by SmoothQuant / AWQ |

Default `α = (0.15, 0.15, 0.25, 0.25, 0.20)`. The ablation study shows that on LLaMA-3.2-1B, a *magnitude-heavy* weighting empirically dominates this default — magnitude alone is the strongest single salience signal.

**Phase 2 — Greedy Bit Allocation.** Starting from 1-bit for all weights, the allocator upgrades weights in descending salience order (1 → 2 → 4 bits) until the average bit budget (1.61 b) is exhausted. A single `torch.topk` call over all parameters takes 0.45 s on 345M GPT-2 medium. Supports weight-, channel-, and layer-level granularity.

**Phase 3 — Row-Coherent Mixed-Precision Quantization.** Applies 1-bit (binary sign), 2-bit (INT2), or 4-bit (INT4) quantization per weight, with closed-form per-block OLS scale refinement. Critical detail: the row-coherence fix (processing each row independently rather than flattening across rows) reduced GPT-2 medium PPL from **12,173 to 541** — a 22× improvement from a few lines of code, and the dominant accuracy lever in the entire pipeline.

---

## Ablation study (LLaMA-3.2-1B at 1.61 b)

Six axes swept; full JSONs in [`results/llama_full/`](results/llama_full/):

| Ablation | Best config | Worst config | Finding |
|---|---|---|---|
| Salience metric (GPT-2 med) | Ensemble (5,512 PPL) | Hessian alone (243,575) | Hessian over-allocates; magnitude is the best single metric |
| Granularity (GPT-2 med) | Weight (5,681) | Layer (1,654,710) | Per-element allocation is essential — 291× better |
| Bit-budget sweep | 1.61 b (29,721 PPL) | 1.0 b (90,337) | Targets ≥ 2.0 b silently collapse to Uniform INT2 (4-bit-tier bug) |
| Calibration size | N=256 (29,877 PPL) | N=1024 (31,774) | Salience saturates at 256; more data adds noise |
| Quant scheme | Symmetric (30,792) | Asymmetric (295,858) | Symmetric beats asymmetric by 9.6× on LLaMA-1B |
| Ensemble weights | Magnitude-heavy (21,977) | Gradient-heavy (49,095) | Magnitude-heavy outperforms the gradient/Hessian default |

Three of these findings contradict the project's initial design assumptions; Section 6 of the report explains and reconciles each.

---

## Project structure

```
salient-quant/
├── src/
│   ├── salience/            # 5 salience metrics + weighted ensemble
│   │   ├── metrics.py       # Magnitude / Gradient / Hessian / Activation / Ensemble
│   │   └── computer.py      # SalienceComputer: orchestrates calibration passes
│   ├── quantizer/
│   │   ├── kernels.py       # 1-bit / 2-bit / 4-bit quantization kernels
│   │   ├── allocator.py     # Greedy bit allocator (weight / channel / layer)
│   │   └── salient_mask.py  # SalientMaskQuantizer: full 3-phase pipeline
│   ├── baselines/           # FP16, Uniform INT2, GPTQ, BitNet implementations
│   ├── eval/                # Perplexity (WT2, PTB), MMLU scaffold, latency, memory
│   └── utils/               # Data loading (C4), logging, visualisation
├── experiments/
│   └── run_experiment.py    # Main entry point — single model, baselines, ablations
├── configs/
│   ├── gpt2_quick.yaml      # Sanity check — GPT-2 small
│   ├── gpt2_colab.yaml      # GPT-2 small on Colab T4 (~15 min)
│   ├── gpt2_full.yaml       # Milestone B: GPT-2 medium full eval + ablations (~60 min T4)
│   └── llama_full.yaml      # Milestone C: LLaMA-3.2-1B full eval + 6 ablations (~90 min T4)
├── notebooks/
│   └── salient_quant_colab.ipynb   # Complete Colab experiment runner
├── scripts/                 # SLURM run scripts (run_gpt2.sh, run_llama.sh)
├── tests/                   # 31+ pytest unit tests
├── results/
│   └── llama_full/          # Ablation JSONs (committed)
└── report/
    ├── report.pdf           # Final 9-page submission report
    ├── report.tex           # LaTeX source
    └── figures/             # Ablation plot images
```

---

## Reproducing the results

### One-shot Colab T4 (recommended — no local GPU needed)

The fastest reproduction path is the included Colab notebook, which handles setup, memory management, and Drive checkpointing automatically.

**[`notebooks/salient_quant_colab.ipynb`](notebooks/salient_quant_colab.ipynb)**

**Prerequisites:**

1. **GPU runtime** — `Runtime → Change runtime type → T4 GPU` (free tier works).
2. **HuggingFace token** with LLaMA-3.2 access:
   - Request access: https://huggingface.co/meta-llama/Llama-3.2-1B
   - Generate token: https://huggingface.co/settings/tokens
   - Paste into Step 3 of the notebook.
3. **Google Drive** (recommended) — mounts in Step 4 to preserve results across session resets.

**Notebook walkthrough:**

| Cell | What it does | Time |
|---|---|---|
| Step 0 | Verify T4 GPU is attached | — |
| Step 1 | Clone this repo | — |
| Step 2 | `pip install -r requirements_colab.txt` | ~2 min |
| Step 3 | HuggingFace login (for LLaMA-3.2) | — |
| Step 4 | Mount Google Drive | — |
| Experiment 1 | GPT-2 small: baselines + ours | ~15 min |
| Experiment 2 | GPT-2 medium: full eval + ablations | ~60 min |
| Experiment 3 | LLaMA-3.2-1B: 6-axis ablations | ~90 min |
| Summary | Print results tables + display plots | — |

> **LLaMA memory note:** Each ablation config reloads the model and runs calibration. If the session OOMs mid-run, per-config checkpoints are saved automatically — restart the runtime and rerun; completed configs are skipped.

### Local / SLURM

```bash
conda create -n salient-quant python=3.10 && conda activate salient-quant
pip install -r requirements.txt   # use requirements_colab.txt on Colab

# Quick sanity check (~15 min on T4)
python experiments/run_experiment.py --config configs/gpt2_colab.yaml

# Full GPT-2 medium eval + ablations
python experiments/run_experiment.py --config configs/gpt2_full.yaml

# LLaMA-3.2-1B full eval + 6 ablations
python experiments/run_experiment.py --config configs/llama_full.yaml

# SLURM convenience scripts
bash scripts/run_gpt2.sh
bash scripts/run_llama.sh
```

Requires a CUDA-capable GPU. Tested on Colab T4 (16 GB) for all experiments. LLaMA-3.2-1B needs ≥ 8 GB VRAM (loaded in fp16). Random seeds are fixed (`seed=42`); rerunning produces bit-identical output to the JSONs in [`results/llama_full/`](results/llama_full/).

### Tests

```bash
pytest tests/ -v
```

Covers quantization kernels, bit allocator, salience metrics, ensemble combination, and an end-to-end smoke test on a synthetic MLP.

---

## Key implementation notes

The pipeline took 40+ commits of OOM debugging to fit on a 16 GB Colab T4. The critical fixes (Section 7 of the report):

- **No `device_map="auto"`** — accelerate's CPU-offload hooks route gradients through CPU during calibration (~9 s/batch instead of <0.5 s/batch). Models are loaded with plain `from_pretrained` + `.to(device)`.
- **Salience accumulators on CPU** — gradient and Fisher accumulators for LLaMA-1B would occupy ~4 GB on VRAM each. All accumulators are kept on CPU and drained per-parameter during assembly.
- **fp16 salience map** — salience tensors are downcast to fp16 before the allocator runs, halving CPU RAM from 3.9 GB to 1.95 GB while preserving ranking accuracy.
- **FP32 loss before backward** — FP16 loss underflowed without `GradScaler`, producing zero/NaN gradients on ~30% of weights. Casting the loss to FP32 fixed silent gradient collapse.
- **CPU salience statistics** — `torch.randperm(354M, device='cuda')` alone needs 2.8 GB of contiguous VRAM. Random sampling for the per-layer quantile estimator now happens on CPU.
- **Row-wise quantization** — both correctness (the 22× PPL fix) and memory: avoids 4–8 GB intermediate tensors on a 1B model.
- **Per-config ablation checkpoints** — each completed ablation config is saved to `ablation_<type>_partial.json`; a crashed run resumes from the last checkpoint on restart.

The pattern across all fixes: numerical stability and memory locality matter more than algorithmic novelty.

---

## Milestones

| Milestone | Status | Description |
|---|---|---|
| A | ✅ Complete | Core pipeline (salience, allocator, kernels) — sanity-checked on GPT-2 small |
| B | ✅ Complete | GPT-2 medium full eval + salience-metric & granularity ablations |
| C | ✅ Complete | LLaMA-3.2-1B full eval + 6-axis ablation study (MMLU deferred — see report §8) |
| Final | ✅ Submitted | 9-page report (`report/report.pdf`), submission package, slide deck |

---

## Dependencies

| Group | Packages |
|---|---|
| Core | PyTorch ≥ 2.1, Transformers ≥ 4.40, Accelerate, Datasets |
| Evaluation | lm-eval ≥ 0.4.2 (planned for MMLU; deferred for time) |
| Baselines | auto-gptq, bitsandbytes, optimum (auto-gptq disabled on Colab T4 — install was flaky) |
| Utils | PyYAML, tqdm, rich, psutil, matplotlib, seaborn |

Full list: [`requirements.txt`](requirements.txt) (local) · [`requirements_colab.txt`](requirements_colab.txt) (Colab).

---

## Authors

**Anshul Dani** — `adani2@hawk.illinoistech.edu` — A20580060
**Rohit Lahori** — `rlahori@hawk.illinoistech.edu` — A20582911

Illinois Institute of Technology, CS595-2 (Spring 2026).
