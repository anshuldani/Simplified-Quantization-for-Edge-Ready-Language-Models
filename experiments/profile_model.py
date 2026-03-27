"""
experiments/profile_model.py

Standalone profiling script for Milestone B bottleneck analysis.
Measures:
  - VRAM usage at each pipeline stage
  - Salience computation time per metric
  - Quantization time per layer
  - Inference latency: batch=1 (latency) and batch=8 (throughput)
  - Memory bandwidth utilization estimate

Usage:
    python experiments/profile_model.py --model gpt2-medium --output results/gpt2_full/profiling
"""

import os
import sys
import json
import time
import argparse
import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.salience.metrics import SalienceConfig
from src.salience.computer import SalienceComputer
from src.quantizer.allocator import AllocationConfig, BitAllocator
from src.quantizer.kernels import quantize_weight
from src.utils.data import get_c4_calibration_dataloader
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def profile_vram_stages(model, tokenizer, calibration_loader, output_dir, device="cuda"):
    """Profile VRAM at each quantization stage."""
    torch.cuda.reset_peak_memory_stats()
    results = {}

    def vram_gb():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "peak": torch.cuda.max_memory_allocated() / 1e9,
        }

    # Stage 0: Model loaded
    results["stage_0_model_loaded"] = vram_gb()
    logger.info(f"Stage 0 (model loaded): {results['stage_0_model_loaded']}")

    # Stage 1: Salience computation
    sal_config = SalienceConfig(n_calibration_samples=64, metrics=["magnitude_l2", "gradient"])
    computer = SalienceComputer(model, sal_config, device)
    t0 = time.time()
    salience_map = computer.compute(calibration_loader)
    results["stage_1_salience_time"] = time.time() - t0
    results["stage_1_after_salience"] = vram_gb()
    logger.info(f"Stage 1 (salience): {results['stage_1_salience_time']:.1f}s, VRAM: {results['stage_1_after_salience']}")

    # Stage 2: Bit allocation
    alloc_config = AllocationConfig(target_avg_bits=1.61)
    allocator = BitAllocator(alloc_config)
    t0 = time.time()
    bit_map = allocator.allocate(salience_map)
    results["stage_2_allocation_time"] = time.time() - t0
    results["stage_2_alloc_stats"] = allocator.get_allocation_stats(bit_map)["_summary"]
    logger.info(f"Stage 2 (allocation): {results['stage_2_allocation_time']:.1f}s")

    # Stage 3: Quantization
    t0 = time.time()
    n_layers = 0
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2 and name in bit_map:
            bits_tensor = bit_map[name]
            unique_bits = bits_tensor.unique().tolist()
            bits = int(unique_bits[0]) if len(unique_bits) == 1 else 2
            deq, _ = quantize_weight(param.data, bits=bits, refine_scales=False)
            param.data.copy_(deq.to(param.data.dtype))
            n_layers += 1

    results["stage_3_quantization_time"] = time.time() - t0
    results["stage_3_n_layers"] = n_layers
    results["stage_3_after_quant"] = vram_gb()
    logger.info(f"Stage 3 (quantization): {results['stage_3_quantization_time']:.1f}s, layers={n_layers}")

    return results


def profile_inference_latency(model, tokenizer, output_dir, device="cuda"):
    """Profile inference latency across batch sizes and sequence lengths."""
    model.eval()
    results = {}

    configs = [
        (1, 128, "latency_bs1_sl128"),
        (1, 512, "latency_bs1_sl512"),
        (8, 128, "throughput_bs8_sl128"),
        (8, 256, "throughput_bs8_sl256"),
    ]

    for bs, sl, label in configs:
        input_ids = torch.randint(100, 50000, (bs, sl), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)
        torch.cuda.synchronize()

        # Time 10 runs
        times = []
        with torch.no_grad():
            for _ in range(10):
                t0 = time.perf_counter()
                out = model(input_ids)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        ms_per_tok = avg_ms / sl
        tok_per_sec = (bs * sl) / (avg_ms / 1000)

        results[label] = {
            "batch_size": bs,
            "seq_len": sl,
            "avg_ms": avg_ms,
            "ms_per_token": ms_per_tok,
            "tokens_per_sec": tok_per_sec,
        }

        logger.info(f"{label}: {avg_ms:.1f}ms, {ms_per_tok:.2f}ms/tok, {tok_per_sec:.0f}tok/s")

    return results


def profile_salience_per_metric(model, calibration_loader, device="cuda"):
    """Time each salience metric individually."""
    metrics = ["magnitude_l1", "magnitude_l2", "gradient", "hessian", "activation"]
    results = {}

    for metric in metrics:
        logger.info(f"  Timing metric: {metric}")
        sal_config = SalienceConfig(
            metrics=[metric],
            n_calibration_samples=64,
            n_fisher_samples=32,
        )
        computer = SalienceComputer(model, sal_config, device)

        t0 = time.time()
        _ = computer.compute(calibration_loader)
        elapsed = time.time() - t0

        results[metric] = {"time_seconds": elapsed}
        logger.info(f"    {metric}: {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--output", type=str, default="results/profiling")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-calib", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    setup_logging(log_dir=args.output, experiment_name="profiling")

    logger.info(f"Profiling model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto" if args.device == "cuda" else None,
    )
    model.eval()

    calibration_loader = get_c4_calibration_dataloader(
        tokenizer, n_samples=args.n_calib, seq_len=256, batch_size=4
    )

    all_results = {}

    logger.info("\n=== Stage-wise VRAM profiling ===")
    all_results["vram_stages"] = profile_vram_stages(
        model, tokenizer, calibration_loader, args.output, args.device
    )

    logger.info("\n=== Inference latency profiling ===")
    all_results["inference_latency"] = profile_inference_latency(
        model, tokenizer, args.output, args.device
    )

    logger.info("\n=== Per-metric salience timing ===")
    # Reload clean model for salience timing
    model_clean = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device).eval()
    all_results["salience_timing"] = profile_salience_per_metric(
        model_clean, calibration_loader, args.device
    )

    # Save
    output_path = os.path.join(args.output, "profiling_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nProfiling results saved to {output_path}")


if __name__ == "__main__":
    main()
