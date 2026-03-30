"""
quantizer/allocator.py

Greedy bit allocator under budget constraint.

Given:
  - salience_map: {param_name -> salience tensor}
  - target_avg_bits: e.g. 1.61
  - bit_choices: [1, 2, 4]

Algorithm:
  1. Start all weights at 1-bit (baseline)
  2. Sort weights by salience score (descending)
  3. Greedily upgrade 1→2→4 bits while avg_bits ≤ target
  4. Return bit_map: {param_name -> bit_tensor (same shape as param)}

Two granularity modes:
  - "weight":  per-element bit assignment (most granular, most accurate)
  - "channel": per output-channel assignment (layer_out dimension)
  - "layer":   one bit level per entire layer (least overhead)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AllocationConfig:
    target_avg_bits: float = 1.61
    bit_choices: List[int] = None
    granularity: str = "weight"   # "weight" | "channel" | "layer"
    fallback_bits: int = 2        # if budget exceeded

    def __post_init__(self):
        if self.bit_choices is None:
            self.bit_choices = [1, 2, 4]
        assert self.target_avg_bits >= min(self.bit_choices), \
            f"target_avg_bits {self.target_avg_bits} < min bit choice {min(self.bit_choices)}"


class BitAllocator:
    """
    Greedy bit allocator that respects a global average-bits budget.

    The core insight: sort ALL weights globally by salience, then upgrade
    the most salient weights first from 1-bit → 2-bit → 4-bit until
    the budget is exhausted.
    """

    def __init__(self, config: AllocationConfig):
        self.config = config

    def allocate(
        self,
        salience_map: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Main allocation entry point.

        Args:
            salience_map: {param_name -> salience tensor (same shape as param)}

        Returns:
            bit_map: {param_name -> integer tensor of same shape with bit assignments}
        """
        if self.config.granularity == "weight":
            return self._allocate_weight_wise(salience_map)
        elif self.config.granularity == "channel":
            return self._allocate_channel_wise(salience_map)
        elif self.config.granularity == "layer":
            return self._allocate_layer_wise(salience_map)
        else:
            raise ValueError(f"Unknown granularity: {self.config.granularity}")

    # ------------------------------------------------------------------
    # Weight-wise allocation (most granular)
    # ------------------------------------------------------------------

    def _allocate_weight_wise(
        self,
        salience_map: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Per-element bit allocation using vectorized torch operations.

        Replaces the previous O(n²) Python-loop approach (which built a list of
        85M tuples and sorted them in pure Python) with:
          1. torch.cat  — flatten all salience scores in one shot
          2. torch.topk — find the top-k weights to upgrade (C++ kernel, O(n log k))
          3. index assignment — apply upgrades in a single tensor op
        """
        logger.info("Weight-wise allocation: flattening salience scores...")

        names = list(salience_map.keys())
        # Move to CPU for consistent sorting (salience may be on CUDA)
        flat_parts = [salience_map[n].float().cpu().flatten() for n in names]
        sizes = [p.numel() for p in flat_parts]
        offsets = [0] + list(np.cumsum(sizes))

        all_scores = torch.cat(flat_parts)          # [total_params]
        total_params = all_scores.numel()

        min_bits = min(self.config.bit_choices)
        target_total_bits = self.config.target_avg_bits * total_params
        current_total_bits = float(min_bits * total_params)

        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Target avg bits: {self.config.target_avg_bits:.3f} "
                    f"({target_total_bits:.0f} total bits)")
        logger.info(f"Starting at {min_bits}-bit ({current_total_bits:.0f} bits)")

        bits_flat = torch.full((total_params,), min_bits, dtype=torch.uint8)

        for target_bits in sorted(set(self.config.bit_choices) - {min_bits}):
            budget = target_total_bits - current_total_bits
            if budget <= 0:
                logger.info(f"Budget exhausted before {target_bits}-bit upgrades")
                break

            gain = target_bits - min_bits
            max_upgrades = int(budget / gain)
            if max_upgrades == 0:
                continue

            upgradeable = (bits_flat == min_bits)
            n_upgradeable = int(upgradeable.sum().item())
            k = min(max_upgrades, n_upgradeable)

            logger.info(f"Upgrading up to {k:,} weights to {target_bits}-bit "
                        f"(budget: {budget:.0f} bits)")

            if k == 0:
                break

            # Mask non-upgradeable positions out with -inf, then topk
            masked = all_scores.masked_fill(~upgradeable, float("-inf"))
            _, top_idx = masked.topk(k)
            bits_flat[top_idx] = target_bits
            current_total_bits += k * gain

            logger.info(f"Upgraded {k:,} weights to {target_bits}-bit "
                        f"(avg: {current_total_bits / total_params:.4f})")

        logger.info(f"Final avg bits: {current_total_bits / total_params:.4f} "
                    f"(target: {self.config.target_avg_bits})")

        # Reshape back into per-param tensors
        bit_map: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(names):
            start, end = offsets[i], offsets[i + 1]
            bit_map[name] = bits_flat[start:end].reshape(salience_map[name].shape)

        return bit_map

    # ------------------------------------------------------------------
    # Channel-wise allocation (balanced overhead vs accuracy)
    # ------------------------------------------------------------------

    def _allocate_channel_wise(
        self,
        salience_map: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Per output-channel bit allocation.
        Groups weights by output channel, computes channel-level salience,
        then allocates bits at channel granularity.
        """
        logger.info("Channel-wise allocation...")

        # Aggregate salience per output channel
        channel_salience: Dict[str, torch.Tensor] = {}
        for name, scores in salience_map.items():
            if scores.dim() >= 2:
                # [out, in, ...] -> per-channel mean salience
                channel_sal = scores.reshape(scores.shape[0], -1).mean(dim=1)
            else:
                channel_sal = scores.unsqueeze(0)
            channel_salience[name] = channel_sal

        # Allocate channel-level bits
        channel_bit_map = self._greedy_allocate_flat(
            {name: sal for name, sal in channel_salience.items()},
            total_elements=sum(
                scores.shape[0] if scores.dim() >= 2 else 1
                for scores in salience_map.values()
            ),
        )

        # Expand channel bits back to full weight shape
        bit_map = {}
        for name, scores in salience_map.items():
            channel_bits = channel_bit_map[name]
            if scores.dim() >= 2:
                # Broadcast channel bits to full weight tensor
                expanded = channel_bits.view(-1, *([1] * (scores.dim() - 1)))
                bit_map[name] = expanded.expand_as(scores).clone()
            else:
                bit_map[name] = channel_bits.expand_as(scores).clone()

        return bit_map

    # ------------------------------------------------------------------
    # Layer-wise allocation (lowest overhead)
    # ------------------------------------------------------------------

    def _allocate_layer_wise(
        self,
        salience_map: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        One bit level per entire layer.
        Fastest but coarsest allocation.
        """
        logger.info("Layer-wise allocation...")

        # Layer salience = mean of all element saliences
        layer_salience = {
            name: torch.tensor([scores.float().mean().item()])
            for name, scores in salience_map.items()
        }

        layer_bit_map = self._greedy_allocate_flat(
            layer_salience,
            total_elements=len(salience_map),
        )

        # Expand single bit to full weight shape
        bit_map = {}
        for name, scores in salience_map.items():
            layer_bits = layer_bit_map[name].item()
            bit_map[name] = torch.full(scores.shape, layer_bits, dtype=torch.uint8)

        return bit_map

    # ------------------------------------------------------------------
    # Shared greedy allocation primitive
    # ------------------------------------------------------------------

    def _greedy_allocate_flat(
        self,
        salience_map: Dict[str, torch.Tensor],
        total_elements: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Generic greedy allocator over flat salience tensors.
        Used by channel-wise and layer-wise modes.
        """
        min_bits = min(self.config.bit_choices)
        target_total = self.config.target_avg_bits * total_elements
        current_total = float(min_bits * total_elements)

        bit_map = {
            name: torch.full(sal.shape, min_bits, dtype=torch.uint8)
            for name, sal in salience_map.items()
        }

        bit_upgrades = sorted(set(self.config.bit_choices) - {min_bits})

        for target_bits in bit_upgrades:
            budget = target_total - current_total
            if budget <= 0:
                break
            gain = target_bits - min_bits
            max_upgrades = int(budget / gain)
            if max_upgrades == 0:
                continue

            all_entries = []
            for name, sal in salience_map.items():
                flat = sal.float().flatten()
                current = bit_map[name].flatten()
                for i in range(flat.numel()):
                    if current[i].item() == min_bits:
                        all_entries.append((flat[i].item(), name, i))

            all_entries.sort(key=lambda x: x[0], reverse=True)
            for _, name, i in all_entries[:max_upgrades]:
                flat_bits = bit_map[name].flatten()
                flat_bits[i] = target_bits
                bit_map[name] = flat_bits.reshape(bit_map[name].shape)

            current_total += len(all_entries[:max_upgrades]) * gain

        return bit_map

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def get_allocation_stats(
        self,
        bit_map: Dict[str, torch.Tensor],
    ) -> Dict:
        """Compute allocation statistics for logging."""
        stats = {}
        total_params = 0
        total_bits = 0
        bit_dist = {b: 0 for b in self.config.bit_choices}

        for name, bits in bit_map.items():
            flat = bits.flatten()
            n = flat.numel()
            total_params += n
            layer_bits = flat.float().sum().item()
            total_bits += layer_bits

            layer_dist = {}
            for b in self.config.bit_choices:
                count = (flat == b).sum().item()
                layer_dist[f"{b}bit"] = count
                bit_dist[b] += count

            stats[name] = {
                "avg_bits": layer_bits / n,
                "n_params": n,
                **layer_dist,
            }

        stats["_summary"] = {
            "total_params": total_params,
            "avg_bits": total_bits / total_params if total_params > 0 else 0,
            "target_bits": self.config.target_avg_bits,
            "bit_distribution": {
                f"{b}bit": {
                    "count": bit_dist[b],
                    "pct": 100.0 * bit_dist[b] / total_params if total_params > 0 else 0
                }
                for b in self.config.bit_choices
            }
        }

        return stats
