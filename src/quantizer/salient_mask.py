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

        # Flush CUDA cache after salience computation — gradients/activations held
        # during calibration may not be GC'd yet; this recovers that VRAM before
        # Phase 2 (allocation) and Phase 3 (quantization) run.
        if self.device == "cuda":
            torch.cuda.empty_cache()

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
        if self.device == "cuda":
            torch.cuda.empty_cache()
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
        """Get all weight matrix param names to quantize.

        Excludes:
          - nn.Embedding / nn.EmbeddingBag layers
          - Any layer whose weight is tied to an embedding (e.g. lm_head in GPT-2
            shares data_ptr with transformer.wte.weight — quantizing it would
            silently destroy the token embeddings)

        Falls back to matching any 2-D weight module when target_layer_types
        doesn't match (e.g. HuggingFace Conv1D in GPT-2).
        """
        # Collect data_ptrs of all embedding weights to detect weight tying
        embedding_ptrs: set = set()
        for _, mod in self.model.named_modules():
            if isinstance(mod, (nn.Embedding, nn.EmbeddingBag)):
                w = getattr(mod, "weight", None)
                if w is not None:
                    embedding_ptrs.add(w.data_ptr())

        def _safe(mod: nn.Module) -> bool:
            """Return True if this module's weight is safe to quantize."""
            if isinstance(mod, (nn.Embedding, nn.EmbeddingBag)):
                return False
            w = getattr(mod, "weight", None)
            if w is None:
                return False
            if w.data_ptr() in embedding_ptrs:  # weight-tied to embedding
                return False
            return True

        param_set = {n for n, _ in self.model.named_parameters()}

        # Primary: try configured layer types
        params = []
        for name, module in self.model.named_modules():
            if isinstance(module, self.config.target_layer_types) and _safe(module):
                param_name = f"{name}.weight"
                if param_name in param_set:
                    params.append(param_name)

        # Fallback: any module with a 2-D weight (covers Conv1D)
        if not params:
            logger.warning(
                "No non-tied layers matched target_layer_types=%s; "
                "falling back to 2-D weight scan (covers Conv1D etc.).",
                self.config.target_layer_types,
            )
            for name, module in self.model.named_modules():
                if not _safe(module):
                    continue
                param_name = f"{name}.weight"
                if param_name in param_set:
                    w = getattr(module, "weight")
                    if w.dim() == 2:
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

            # Work in fp32 for numerical precision — param may be fp16 if the
            # model was loaded with torch_dtype=float16.
            original_weight = param.data.float().clone()
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
                # Mixed-precision within this layer.
                # Process row-by-row so that block-wise scales are computed from
                # contiguous, spatially-coherent weight groups rather than from
                # randomly scattered positions across the full weight matrix.
                # (The old approach used boolean masking which interleaved weights
                # from unrelated rows into the same quantization block, producing
                # corrupted scale statistics.)
                deq_weight = original_weight.clone()  # default: keep original

                if original_weight.dim() == 2:
                    for row_idx in range(original_weight.shape[0]):
                        row_bits = bit_tensor[row_idx]       # [in_features]
                        row_w    = original_weight[row_idx]  # [in_features]
                        row_deq  = deq_weight[row_idx]

                        for bits in [int(b) for b in unique_bits]:
                            col_mask = (row_bits == bits)
                            if not col_mask.any():
                                continue
                            group = row_w[col_mask].unsqueeze(0)  # [1, n_selected]
                            group_deq, meta = quantize_weight(
                                group,
                                bits=bits,
                                scheme=self.config.scheme_2bit,
                                block_size=min(self.config.block_size, group.numel()),
                                refine_scales=self.config.refine_scales,
                            )
                            row_deq[col_mask] = group_deq.reshape(-1).to(row_deq.dtype)
                else:
                    # Fallback for non-2D weights: original flattened approach
                    for bits in [int(b) for b in unique_bits]:
                        mask = (bit_tensor == bits)
                        if not mask.any():
                            continue
                        group_flat = original_weight[mask]
                        group_deq, meta = quantize_weight(
                            group_flat.unsqueeze(0),
                            bits=bits,
                            scheme=self.config.scheme_2bit,
                            block_size=self.config.block_size,
                            refine_scales=self.config.refine_scales,
                        )
                        deq_weight[mask] = group_deq.reshape(-1).to(deq_weight.dtype)

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

