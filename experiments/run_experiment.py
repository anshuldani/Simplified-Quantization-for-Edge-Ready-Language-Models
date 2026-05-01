"""
experiments/run_experiment.py

Main experiment runner. Supports:
  - Single model quantization + eval
  - Baseline comparisons
  - Ablation sweeps
  - Results aggregation

Usage:
    python experiments/run_experiment.py --config configs/gpt2_quick.yaml
    python experiments/run_experiment.py --config configs/gpt2_full.yaml
    python experiments/run_experiment.py --config configs/llama_full.yaml
"""

import os
import sys
import json
import gc

# Reduce CUDA memory fragmentation — critical for large models on T4
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import yaml
import copy
import logging
import argparse
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.salience.metrics import SalienceConfig
from src.quantizer.allocator import AllocationConfig
from src.quantizer.salient_mask import QuantizerConfig, SalientMaskQuantizer
from src.baselines.baselines import UniformINT2Baseline, BitNetTernaryBaseline, FP16Baseline
from src.baselines.gptq_runner import GPTQRunner, prepare_gptq_calibration
from src.eval.evaluator import ModelEvaluator
from src.utils.data import get_c4_calibration_dataloader
from src.utils.logging_utils import setup_logging, ResultsTracker
from src.utils.viz import (
    plot_salience_distributions,
    plot_bit_allocation_heatmap,
    plot_ablation_comparison,
    plot_baseline_comparison,
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer from HuggingFace."""
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use fp16 on CUDA to halve weight memory (~2 GB saved for LLaMA-1B on T4).
    # Quantization maths are still precise because we operate on .float() clones
    # inside quantize_weight(); the stored param dtype stays fp16 throughout.
    #
    # CRITICAL: do NOT use device_map="auto" or low_cpu_mem_usage=True.
    # device_map="auto" installs accelerate CPU-offload hooks which cause:
    #   - "Materializing param" spam during calibration
    #   - Gradient backward running on CPU → 9 s/it instead of <0.5 s/it
    #   - Exit -9 (CPU OOM) at the first gradient batch
    # Load plainly and call .to(device) — this guarantees every param lands on
    # CUDA with no hooks attached.
    load_dtype = torch.float16 if "cuda" in device else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=load_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)

    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {n_params/1e6:.1f}M parameters")

    return model, tokenizer


def run_baseline_experiments(
    model_name: str,
    tokenizer,
    calibration_dataloader,
    eval_datasets: List[str],
    device: str,
    run_mmlu: bool,
    run_latency: bool,
    output_dir: str,
    tracker: ResultsTracker,
    max_eval_tokens: Optional[int] = None,
    run_gptq: bool = False,
):
    """Run all baseline experiments."""
    baselines = {
        "fp16": FP16Baseline(),
        "uniform_int2": UniformINT2Baseline(),
        "bitnet": BitNetTernaryBaseline(),
    }

    for baseline_name, baseline in baselines.items():
        logger.info(f"\nRunning baseline: {baseline_name}")

        # Load fresh model for each baseline
        base_model, _ = load_model_and_tokenizer(model_name, device)

        # Apply quantization
        quantized = baseline.apply(base_model, device) if baseline_name == "fp16" else baseline.apply(base_model)

        # Evaluate
        evaluator = ModelEvaluator(quantized, tokenizer, device)
        results = evaluator.evaluate_all(
            run_mmlu=run_mmlu,
            run_latency=run_latency,
            run_perplexity=True,
            datasets=eval_datasets,
            max_eval_tokens=max_eval_tokens,
        )
        results["avg_bits"] = baseline.avg_bits()

        tracker.add_result(baseline_name, results)
        with torch.no_grad():
            for param in base_model.parameters():
                param.data = param.data.new_empty(0)
        del quantized, base_model, evaluator, results
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    # GPTQ baseline (optional — requires auto-gptq)
    if run_gptq:
        logger.info("\nRunning baseline: gptq_4bit")
        try:
            gptq_samples = prepare_gptq_calibration(calibration_dataloader, n_samples=128)
            runner = GPTQRunner(model_name, bits=4, group_size=128)
            quantized = runner.run(gptq_samples)
            evaluator = ModelEvaluator(quantized, tokenizer, device)
            results = evaluator.evaluate_all(
                run_mmlu=run_mmlu,
                run_latency=run_latency,
                run_perplexity=True,
                datasets=eval_datasets,
                max_eval_tokens=max_eval_tokens,
            )
            results["avg_bits"] = GPTQRunner.avg_bits()
            tracker.add_result("gptq_4bit", results)
            del quantized
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"GPTQ baseline failed: {e}")


def run_ours(
    model_name: str,
    tokenizer,
    calibration_dataloader,
    eval_datasets: List[str],
    device: str,
    run_mmlu: bool,
    run_latency: bool,
    output_dir: str,
    tracker: ResultsTracker,
    config: QuantizerConfig,
    max_eval_tokens: Optional[int] = None,
):
    """Run our SalientMaskQuantizer."""
    logger.info("\nRunning our method: SalientMaskQuantizer")

    base_model, _ = load_model_and_tokenizer(model_name, device)

    quantizer = SalientMaskQuantizer(base_model, config, device=device)
    quantizer.quantize(calibration_dataloader)

    # Save intermediate results
    results_subdir = os.path.join(output_dir, "ours")
    quantizer.save_results(results_subdir)

    # Visualize salience
    if quantizer.salience_map:
        stats = quantizer.salience_computer.get_salience_stats(quantizer.salience_map)
        plot_salience_distributions(
            stats,
            os.path.join(output_dir, "plots", "salience_distributions.png"),
        )

    # Visualize bit allocation
    if quantizer.bit_map:
        alloc_stats = quantizer.bit_allocator.get_allocation_stats(quantizer.bit_map)
        plot_bit_allocation_heatmap(
            alloc_stats,
            os.path.join(output_dir, "plots", "bit_allocation.png"),
        )

    # Evaluate
    evaluator = ModelEvaluator(base_model, tokenizer, device)
    eval_results = evaluator.evaluate_all(
        run_mmlu=run_mmlu,
        run_latency=run_latency,
        run_perplexity=True,
        datasets=eval_datasets,
        max_eval_tokens=max_eval_tokens,
    )
    eval_results["avg_bits"] = quantizer.get_memory_footprint().get("avg_bits", 0)
    eval_results["memory_footprint"] = quantizer.get_memory_footprint()
    eval_results["timing"] = quantizer.timing

    tracker.add_result("ours", eval_results)

    del base_model, quantizer, evaluator
    gc.collect()
    torch.cuda.empty_cache()
    return eval_results


def run_ablation_study(
    model_name: str,
    tokenizer,
    calibration_dataloader,
    eval_datasets: List[str],
    device: str,
    output_dir: str,
    ablation_type: str = "salience_metric",
):
    """
    Run ablation studies.

    ablation_type options:
      - "salience_metric": Compare each of the 5 metrics individually
      - "bit_budget": Compare different target avg bits
      - "calibration_size": Compare 128 vs 512 vs 1024 samples
      - "granularity": Compare weight vs channel vs layer allocation
      - "quant_scheme": Compare symmetric vs asymmetric 2-bit
      - "ensemble_weights": Compare different alpha configurations
    """
    logger.info(f"\nRunning ablation: {ablation_type}")

    # Resume from partial checkpoint if a previous run crashed mid-ablation.
    partial_path = os.path.join(output_dir, f"ablation_{ablation_type}_partial.json")
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            ablation_results = json.load(f)
        logger.info(f"  Resuming from partial checkpoint: {list(ablation_results.keys())} already done")
    else:
        ablation_results = {}
    ablation_configs = _get_ablation_configs(ablation_type)

    for config_name, config in ablation_configs.items():
        if config_name in ablation_results:
            logger.info(f"  Skipping {config_name} (already in checkpoint)")
            continue
        logger.info(f"  Ablation config: {config_name}")

        base_model, _ = load_model_and_tokenizer(model_name, device)
        quantizer = SalientMaskQuantizer(base_model, config, device=device)
        quantizer.quantize(calibration_dataloader)

        evaluator = ModelEvaluator(base_model, tokenizer, device)
        results = evaluator.evaluate_all(
            run_mmlu=False,
            run_latency=False,
            run_perplexity=True,
            datasets=eval_datasets[:1],  # Only WikiText-2 for ablations
        )
        results["avg_bits"] = quantizer.get_memory_footprint().get("avg_bits", 0)

        ppl = results.get("ppl_wikitext2", {}).get("perplexity", float("nan"))
        ablation_results[config_name] = {
            "perplexity": ppl,
            "avg_bits": results["avg_bits"],
        }
        logger.info(f"    PPL: {ppl:.2f}, avg bits: {results['avg_bits']:.3f}")

        # Save incrementally — partial results survive a crash mid-ablation.
        with open(partial_path, "w") as f:
            json.dump(ablation_results, f, indent=2)

        # Free weights in-place before del — avoids a 2.5 GB CPU spike that
        # .cpu() / Python GC would otherwise cause on a nearly-full T4.
        with torch.no_grad():
            for param in base_model.parameters():
                param.data = param.data.new_empty(0)
        if hasattr(quantizer, "bit_map"):
            quantizer.bit_map = None
        if hasattr(quantizer, "salience_map"):
            quantizer.salience_map = None
        del base_model, quantizer, evaluator, results
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    # Save and plot
    ablation_path = os.path.join(output_dir, f"ablation_{ablation_type}.json")
    with open(ablation_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    plot_ablation_comparison(
        ablation_results,
        output_path=os.path.join(output_dir, "plots", f"ablation_{ablation_type}.png"),
        title=f"Ablation: {ablation_type.replace('_', ' ').title()}",
    )

    return ablation_results


def _get_ablation_configs(ablation_type: str) -> Dict[str, QuantizerConfig]:
    """Generate configs for each ablation type."""
    configs = {}

    # Hessian is excluded from all non-salience_metric ablations:
    # gradient + hessian fp32 accumulators together = ~10 GB CPU RAM on T4 → OOM.
    _SAFE_METRICS = ["magnitude_l1", "magnitude_l2", "gradient", "activation"]

    if ablation_type == "salience_metric":
        for metric in ["magnitude_l1", "magnitude_l2", "gradient", "hessian", "activation"]:
            sal_config = SalienceConfig(metrics=[metric])
            configs[metric] = QuantizerConfig(
                salience=sal_config,
                allocation=AllocationConfig(target_avg_bits=1.61),
            )
        # Also test ensemble (hessian excluded: too memory-intensive for T4)
        configs["combined"] = QuantizerConfig(
            salience=SalienceConfig(metrics=_SAFE_METRICS),
            allocation=AllocationConfig(target_avg_bits=1.61),
        )

    elif ablation_type == "bit_budget":
        for target_bits in [1.0, 1.3, 1.61, 2.0, 2.5, 3.0, 4.0]:
            configs[f"target_{target_bits}b"] = QuantizerConfig(
                salience=SalienceConfig(metrics=_SAFE_METRICS),
                allocation=AllocationConfig(target_avg_bits=target_bits),
            )

    elif ablation_type == "calibration_size":
        for n_samples in [128, 256, 512, 1024]:
            configs[f"n{n_samples}"] = QuantizerConfig(
                salience=SalienceConfig(
                    metrics=_SAFE_METRICS,
                    n_calibration_samples=n_samples,
                ),
                allocation=AllocationConfig(target_avg_bits=1.61),
            )

    elif ablation_type == "granularity":
        for granularity in ["weight", "channel", "layer"]:
            configs[granularity] = QuantizerConfig(
                salience=SalienceConfig(metrics=_SAFE_METRICS),
                allocation=AllocationConfig(
                    target_avg_bits=1.61,
                    granularity=granularity,
                ),
            )

    elif ablation_type == "quant_scheme":
        for scheme in ["symmetric", "asymmetric"]:
            configs[scheme] = QuantizerConfig(
                salience=SalienceConfig(metrics=_SAFE_METRICS),
                scheme_2bit=scheme,
                allocation=AllocationConfig(target_avg_bits=1.61),
            )

    elif ablation_type == "ensemble_weights":
        # Vary α weights across the 4 safe metrics (l1, l2, gradient, activation).
        # Hessian excluded — too memory-intensive for T4.
        # Tuples: (alpha_l1, alpha_l2, alpha_gradient, alpha_activation)
        weight_configs = {
            "magnitude_heavy":  (0.35, 0.35, 0.15, 0.15),
            "gradient_heavy":   (0.10, 0.10, 0.60, 0.20),
            "activation_heavy": (0.10, 0.10, 0.20, 0.60),
            "uniform":          (0.25, 0.25, 0.25, 0.25),
            "default":          (0.18, 0.18, 0.32, 0.32),
        }
        for name, (al1, al2, ag, aa) in weight_configs.items():
            sal_config = SalienceConfig(
                metrics=_SAFE_METRICS,
                alpha_magnitude_l1=al1,
                alpha_magnitude_l2=al2,
                alpha_gradient=ag,
                alpha_activation=aa,
            )
            configs[name] = QuantizerConfig(
                salience=sal_config,
                allocation=AllocationConfig(target_avg_bits=1.61),
            )

    return configs


def main():
    parser = argparse.ArgumentParser(description="SalientMask Quantization Experiments")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_device = args.device or cfg.get("device", "auto")
    if raw_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = raw_device
    model_name = cfg["model_name"]
    output_dir = cfg.get("output_dir", f"results/{model_name.replace('/', '_')}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Setup logging
    setup_logging(
        log_dir=os.path.join(output_dir, "logs"),
        experiment_name=cfg.get("experiment_name", "experiment"),
    )

    logger.info(f"Config: {json.dumps(cfg, indent=2)}")
    logger.info(f"Device: {device}")

    # Load tokenizer and calibration data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration_dataloader = get_c4_calibration_dataloader(
        tokenizer,
        n_samples=cfg.get("n_calibration_samples", 512),
        seq_len=cfg.get("calibration_seq_len", 512),
        batch_size=cfg.get("calibration_batch_size", 4),
    )

    eval_datasets = cfg.get("eval_datasets", ["wikitext2", "ptb"])
    run_mmlu = cfg.get("run_mmlu", False)
    run_latency = cfg.get("run_latency", True)

    tracker = ResultsTracker(output_dir, cfg.get("experiment_name", "experiment"))

    # Build our method config
    our_config = QuantizerConfig(
        salience=SalienceConfig(
            metrics=cfg.get("salience_metrics", ["magnitude_l1", "magnitude_l2", "gradient", "hessian", "activation"]),
            n_calibration_samples=cfg.get("n_calibration_samples", 512),
        ),
        allocation=AllocationConfig(
            target_avg_bits=cfg.get("target_avg_bits", 1.61),
            granularity=cfg.get("granularity", "weight"),
        ),
        block_size=cfg.get("block_size", 64),
        scheme_2bit=cfg.get("quant_scheme", "symmetric"),
        refine_scales=cfg.get("refine_scales", True),
    )

    # Run experiments based on config
    run_types = cfg.get("run", ["ours", "baselines"])

    max_eval_tokens = cfg.get("max_eval_tokens", None)

    if "baselines" in run_types:
        run_baseline_experiments(
            model_name, tokenizer, calibration_dataloader,
            eval_datasets, device, run_mmlu, run_latency, output_dir, tracker,
            max_eval_tokens=max_eval_tokens,
            run_gptq=cfg.get("run_gptq", False),
        )

    if "ours" in run_types:
        run_ours(
            model_name, tokenizer, calibration_dataloader,
            eval_datasets, device, run_mmlu, run_latency, output_dir, tracker,
            our_config, max_eval_tokens=max_eval_tokens,
        )

    if "ablations" in run_types:
        for ablation_type in cfg.get("ablation_types", ["salience_metric"]):
            run_ablation_study(
                model_name, tokenizer, calibration_dataloader,
                            eval_datasets, device, output_dir, ablation_type,
            )

    # Final summary
    tracker.print_summary()

    # Final comparison plot
    if len(tracker.results["models"]) > 1:
        plot_baseline_comparison(
            tracker.results["models"],
            output_path=os.path.join(output_dir, "plots", "baseline_comparison.png"),
            model_name=model_name,
        )

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
