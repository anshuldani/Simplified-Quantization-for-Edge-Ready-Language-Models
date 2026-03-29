"""
quantizer/salient_mask.py

SalientMaskQuantizer: The core algorithm from the proposal.

Orchestrates:
  Phase 1 → SalienceComputer (salience/computer.py)
  Phase 2 → BitAllocator (quantizer/allocator.py)
  Phase 3 → Mixed-precision quantization (quantizer/kernels.py)

Produces a quantized model with per-weight bit assignments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import logging
import time
import json
import os

from ..salience.metrics import SalienceConfig
from ..salience.computer import SalienceComputer
from .allocator import AllocationConfig, BitAllocator
from .kernels import quantize_weight

logger = logging.getLogger(__name__)


@dataclass
class QuantizerConfig:
    # Salience config
    salience: SalienceConfig = field(default_factory=SalienceConfig)

    # Allocation config
    allocation: AllocationConfig = field(default_factory=AllocationConfig)

    # Quantization
    block_size: int = 128
    scheme_2bit: str = "symmetric"  # "symmetric" | "asymmetric"
    refine_scales: bool = True
    n_scale_refinement_iters: int = 20

    # Which layers to quantize (regex or list of substrings)
    # Default: all linear weight matrices
    target_layer_types: Tuple = (nn.Linear,)

    # Output
    save_bit_map: bool = True
    save_salience_map: bool = False  # large, optional


class SalientMaskQuantizer:
    """
    Full 3-phase quantization pipeline.

    Example:
        config = QuantizerConfig()
        quantizer = SalientMaskQuantizer(model, config)
        quantized_model = quantizer.quantize(calibration_dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: QuantizerConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device

        self.salience_computer = SalienceComputer(
            model, config.salience, device
        )
        self.bit_allocator = BitAllocator(config.allocation)

        # Results storage
        self.salience_map: Optional[Dict] = None
        self.bit_map: Optional[Dict] = None
        self.quantization_meta: Dict = {}
        self.timing: Dict = {}

    def quantize(
        self,
        calibration_dataloader: DataLoader,
        target_params: Optional[list] = None,
    ) -> nn.Module:
        """
        Full quantization pipeline. Returns quantized model (in-place modification).

        Args:
            calibration_dataloader: C4 validation dataloader (512 samples)
            target_params: Optional list of param names to quantize.
                           Defaults to all linear weight matrices.

        Returns:
            Quantized model with replaced weight tensors
        """
        logger.info("=" * 60)
        logger.info("SalientMaskQuantizer: Starting quantization")
        logger.info(f"  Target avg bits: {self.config.allocation.target_avg_bits}")
        logger.info(f"  Granularity: {self.config.allocation.granularity}")
        logger.info(f"  Metrics: {self.config.salience.metrics}")
        logger.info("=" * 60)

        # Identify target params if not provided
        if target_params is None:
            target_params = self._get_quantizable_params()
        logger.info(f"Quantizing {len(target_params)} parameter tensors")

        # ---- Phase 1: Salience computation ----
        logger.info("\n[Phase 1] Computing salience scores...")
        t0 = time.time()
        self.salience_map = self.salience_computer.compute(
            calibration_dataloader, target_params
        )
        self.timing["phase1_salience"] = time.time() - t0
        logger.info(f"Phase 1 complete in {self.timing['phase1_salience']:.1f}s")

        # Log global salience stats
        stats = self.salience_computer.get_salience_stats(self.salience_map)
        if "_global" in stats:
            g = stats["_global"]
            logger.info(f"Global salience: mean={g['mean']:.4f}, "
                        f"p80={g['p80']:.4f}, total_params={g['total_params']:,}")

        # ---- Phase 2: Bit allocation ----
        logger.info("\n[Phase 2] Greedy bit allocation...")
        t0 = time.time()
        self.bit_map = self.bit_allocator.allocate(self.salience_map)
        self.timing["phase2_allocation"] = time.time() - t0

        alloc_stats = self.bit_allocator.get_allocation_stats(self.bit_map)
        summary = alloc_stats["_summary"]
        logger.info(f"Phase 2 complete in {self.timing['phase2_allocation']:.1f}s")
        logger.info(f"Achieved avg bits: {summary['avg_bits']:.4f} "
                    f"(target: {summary['target_bits']})")
        for b_key, dist in summary["bit_distribution"].items():
            logger.info(f"  {b_key}: {dist['count']:,} ({dist['pct']:.1f}%)")

        # ---- Phase 3: Mixed-precision quantization ----
        logger.info("\n[Phase 3] Applying mixed-precision quantization...")
        t0 = time.time()
        self._apply_quantization(target_params)
        self.timing["phase3_quantization"] = time.time() - t0
        logger.info(f"Phase 3 complete in {self.timing['phase3_quantization']:.1f}s")

        total_time = sum(self.timing.values())
        logger.info(f"\nQuantization complete! Total time: {total_time:.1f}s")
        logger.info(f"Timing breakdown: {self.timing}")

        return self.model

    def _get_quantizable_params(self) -> list:
        """Get all weight matrix param names for target layer types.

        Falls back to matching any module with a 2-D weight tensor when
        target_layer_types don't match (e.g. HuggingFace Conv1D in GPT-2).
        """
        # First try the configured layer types
        params = []
        for name, module in self.model.named_modules():
            if isinstance(module, self.config.target_layer_types):
                param_name = f"{name}.weight"
                if any(n == param_name for n, _ in self.model.named_parameters()):
                    params.append(param_name)

        # Fallback: if nothing matched, collect any module with a 2-D weight
        # (covers nn.Linear and transformers Conv1D alike)
        if not params:
            logger.warning(
                "No layers matched target_layer_types=%s; "
                "falling back to any module with a 2-D weight parameter.",
                self.config.target_layer_types,
            )
            param_set = {n for n, _ in self.model.named_parameters()}
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                    continue
                param_name = f"{name}.weight"
                if param_name in param_set:
                    w = dict(module.named_parameters(recurse=False)).get("weight")
                    if w is not None and w.dim() == 2:
                        params.append(param_name)

        return params

    def _apply_quantization(self, target_params: list):
        """Apply quantization to each parameter using its assigned bit width."""
        param_dict = dict(self.model.named_parameters())
        n_quantized = 0
        total_error = 0.0

        for param_name in tqdm(target_params, desc="Quantizing weights"):
            if param_name not in self.bit_map:
                logger.warning(f"No bit map for {param_name}, skipping")
                continue

            param = param_dict.get(param_name)
            if param is None:
                continue

            original_weight = param.data.clone()
            bit_tensor = self.bit_map[param_name]

            # Check if all weights have same bit width (fast path)
            unique_bits = bit_tensor.unique().tolist()

            if len(unique_bits) == 1:
                # Uniform bits for this layer — simple path
                bits = int(unique_bits[0])
                deq_weight, meta = quantize_weight(
                    original_weight,
                    bits=bits,
                    scheme=self.config.scheme_2bit,
                    block_size=self.config.block_size,
                    refine_scales=self.config.refine_scales,
                )
                param.data.copy_(deq_weight.to(param.data.dtype))
                self.quantization_meta[param_name] = meta

            else:
                # Mixed-precision within this layer
                # Quantize each bit-group separately, combine
                deq_weight = torch.empty_like(original_weight)

                for bits in [int(b) for b in unique_bits]:
                    mask = (bit_tensor == bits)
                    if not mask.any():
                        continue

                    # Extract this bit-group's weights
                    group_weights = original_weight.clone()
                    group_weights[~mask] = 0.0

                    group_deq, meta = quantize_weight(
                        group_weights,
                        bits=bits,
                        scheme=self.config.scheme_2bit,
                        block_size=self.config.block_size,
                        refine_scales=self.config.refine_scales,
                    )
                    deq_weight[mask] = group_deq[mask].to(deq_weight.dtype)

                # Fill unquantized positions (shouldn't exist, but safety)
                param.data.copy_(deq_weight.to(param.data.dtype))
                self.quantization_meta[param_name] = {"bits": "mixed", "unique_bits": unique_bits}

            # Track reconstruction error
            error = (original_weight - param.data.float()).pow(2).mean().item()
            total_error += error
            n_quantized += 1

        avg_error = total_error / n_quantized if n_quantized > 0 else 0
        logger.info(f"Quantized {n_quantized} tensors, avg L2 reconstruction error: {avg_error:.6f}")

    def save_results(self, output_dir: str):
        """Save bit map, salience stats, and timing to disk."""
        os.makedirs(output_dir, exist_ok=True)

        if self.bit_map and self.config.save_bit_map:
            torch.save(self.bit_map, os.path.join(output_dir, "bit_map.pt"))
            logger.info(f"Saved bit map to {output_dir}/bit_map.pt")

        if self.salience_map and self.config.save_salience_map:
            torch.save(self.salience_map, os.path.join(output_dir, "salience_map.pt"))

        # Save allocation stats as JSON
        if self.bit_map:
            stats = self.bit_allocator.get_allocation_stats(self.bit_map)
            with open(os.path.join(output_dir, "allocation_stats.json"), "w") as f:
                json.dump(stats, f, indent=2)

        # Save timing
        with open(os.path.join(output_dir, "timing.json"), "w") as f:
            json.dump(self.timing, f, indent=2)

        logger.info(f"Results saved to {output_dir}")

    def get_memory_footprint(self) -> Dict:
        """Estimate memory savings from quantization."""
        if self.bit_map is None:
            return {}

        fp16_bits = 0
        quant_bits = 0

        for name, bits_tensor in self.bit_map.items():
            n = bits_tensor.numel()
            fp16_bits += n * 16
            quant_bits += bits_tensor.float().sum().item()

        return {
            "fp16_gb": fp16_bits / 8 / 1e9,
            "quantized_gb": quant_bits / 8 / 1e9,
            "compression_ratio": fp16_bits / quant_bits if quant_bits > 0 else 0,
            "avg_bits": quant_bits / (fp16_bits / 16) if fp16_bits > 0 else 0,
        }
