"""
baselines/baselines.py

Baseline quantization methods for comparison:
  1. FP16     - uncompressed upper bound (identity, just cast to fp16)
  2. UniformINT2 - all weights 2-bit uniform (symmetric)
  3. BitNet   - 1.58-bit ternary quantization (uniform, all weights)

GPTQ is handled separately via auto-gptq library (baselines/gptq_runner.py).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import tqdm
import logging
import copy

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# FP16 Baseline
# -----------------------------------------------------------------------

class FP16Baseline:
    """
    Cast model to FP16. Serves as uncompressed upper bound.
    """

    @staticmethod
    def apply(model: nn.Module) -> nn.Module:
        model = copy.deepcopy(model)
        model = model.half()
        logger.info("FP16 baseline: model cast to float16")
        return model

    @staticmethod
    def name() -> str:
        return "fp16"

    @staticmethod
    def avg_bits() -> float:
        return 16.0


# -----------------------------------------------------------------------
# Uniform INT2 Baseline
# -----------------------------------------------------------------------

class UniformINT2Baseline:
    """
    Applies uniform 2-bit symmetric quantization to ALL weight matrices.
    Represents the naive compression baseline our method beats.
    """

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def apply(self, model: nn.Module) -> nn.Module:
        model = copy.deepcopy(model)

        n_quantized = 0
        for name, module in tqdm(model.named_modules(), desc="Uniform INT2"):
            if isinstance(module, nn.Linear):
                weight = module.weight.data.float()
                deq_weight = self._quantize_2bit(weight)
                module.weight.data = deq_weight.to(module.weight.data.dtype)
                n_quantized += 1

        logger.info(f"UniformINT2: quantized {n_quantized} linear layers")
        return model

    def _quantize_2bit(self, weight: torch.Tensor) -> torch.Tensor:
        """Symmetric 2-bit quantization with per-block scale."""
        w_flat = weight.reshape(-1)
        n = w_flat.numel()
        n_blocks = (n + self.block_size - 1) // self.block_size
        out = torch.empty_like(w_flat)

        for i in range(n_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, n)
            block = w_flat[start:end]

            scale = block.abs().max().clamp(min=1e-8) / 1.5
            codes = torch.clamp(torch.round((block / scale + 1.5)), 0, 3)
            deq = (codes - 1.5) * scale
            out[start:end] = deq

        return out.reshape(weight.shape)

    def name(self) -> str:
        return "uniform_int2"

    def avg_bits(self) -> float:
        return 2.0


# -----------------------------------------------------------------------
# BitNet 1.58-bit Ternary Baseline
# -----------------------------------------------------------------------

class BitNetTernaryBaseline:
    """
    BitNet b1.58: Ternary quantization {-1, 0, +1} * scale
    scale = mean(|w|) — uniform across ALL weights (no salience).

    Reference: Wang et al. (2023), arXiv:2310.11453
    """

    @staticmethod
    def apply(model: nn.Module) -> nn.Module:
        model = copy.deepcopy(model)
        n_quantized = 0

        for name, module in tqdm(model.named_modules(), desc="BitNet Ternary"):
            if isinstance(module, nn.Linear):
                weight = module.weight.data.float()
                deq = BitNetTernaryBaseline._quantize_ternary(weight)
                module.weight.data = deq.to(module.weight.data.dtype)
                n_quantized += 1

        logger.info(f"BitNet ternary: quantized {n_quantized} linear layers")
        return model

    @staticmethod
    def _quantize_ternary(weight: torch.Tensor) -> torch.Tensor:
        """
        Ternary quantization: w_q = round(w / scale).clamp(-1, 1) * scale
        where scale = mean(|w|)
        """
        scale = weight.abs().mean().clamp(min=1e-8)
        ternary = weight.div(scale).round().clamp(-1, 1)
        return ternary * scale

    @staticmethod
    def name() -> str:
        return "bitnet_ternary"

    @staticmethod
    def avg_bits() -> float:
        return 1.58  # log2(3)


# -----------------------------------------------------------------------
# Baseline registry
# -----------------------------------------------------------------------

BASELINE_REGISTRY = {
    "fp16": FP16Baseline,
    "uniform_int2": UniformINT2Baseline,
    "bitnet": BitNetTernaryBaseline,
}


def get_baseline(name: str, **kwargs) -> object:
    """Get baseline by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. "
                         f"Available: {list(BASELINE_REGISTRY.keys())}")
    cls = BASELINE_REGISTRY[name]
    return cls(**kwargs) if kwargs else cls()


def apply_all_baselines(
    model: nn.Module,
    baselines: list = None,
) -> Dict[str, nn.Module]:
    """
    Apply all specified baselines and return dict of quantized models.

    Args:
        model: Original FP32/FP16 model
        baselines: List of baseline names. Defaults to all.

    Returns:
        Dict of {baseline_name: quantized_model}
    """
    if baselines is None:
        baselines = list(BASELINE_REGISTRY.keys())

    results = {}
    for name in baselines:
        logger.info(f"Applying baseline: {name}")
        baseline = get_baseline(name)
        quantized = baseline.apply(model)
        results[name] = quantized
        logger.info(f"Baseline {name} applied successfully")

    return results
