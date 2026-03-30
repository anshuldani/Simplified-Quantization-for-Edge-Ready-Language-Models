"""
quantizer/kernels.py

Quantization kernels for 1-bit, 2-bit, and 4-bit precision.

  - 1-bit:  Sign-based binary quantization (BitNet style)
  - 2-bit:  Symmetric 4-level uniform quantization
  - 4-bit:  INT4 (16-level uniform quantization)

Also implements:
  - Asymmetric 2-bit variant (ablation)
  - Block-wise scale factor optimization (minimize L2 reconstruction error)

All kernels use fully-vectorized torch matrix ops (no Python loops over blocks).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)


def _pad_and_reshape(w_flat: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int]:
    """Pad w_flat to a multiple of block_size and reshape to [n_blocks, block_size]."""
    n = w_flat.numel()
    pad = (-n) % block_size
    if pad:
        w_padded = torch.cat([w_flat, w_flat.new_zeros(pad)])
    else:
        w_padded = w_flat
    n_blocks = w_padded.numel() // block_size
    return w_padded.reshape(n_blocks, block_size), n


# -----------------------------------------------------------------------
# 1-bit: Sign-based binary quantization
# -----------------------------------------------------------------------

def quantize_1bit(
    weight: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binary quantization: w_q ∈ {-scale, +scale}
    scale = mean(|w|) per block (BitNet-style)

    Args:
        weight: [out, in] weight tensor
        block_size: block size for scale computation

    Returns:
        (quantized_weight, scales): same shape as weight, scales [n_blocks]
    """
    w_flat = weight.reshape(-1).float()
    n = w_flat.numel()

    W, _ = _pad_and_reshape(w_flat, block_size)         # [n_blocks, block_size]
    scales = W.abs().mean(dim=1).clamp(min=1e-8)        # [n_blocks]

    # sign(w) * scale — broadcast scale over block dimension
    q = W.sign() * scales.unsqueeze(1)                  # [n_blocks, block_size]
    q_flat = q.reshape(-1)[:n]

    return q_flat.reshape(weight.shape), scales


def dequantize_1bit(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Reconstruct FP weight from 1-bit representation (for evaluation)."""
    # q_weight already stores {-scale, +scale} values
    return q_weight


# -----------------------------------------------------------------------
# 2-bit: Symmetric 4-level quantization
# -----------------------------------------------------------------------

def quantize_2bit_symmetric(
    weight: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symmetric 2-bit quantization: levels ∈ {-1.5s, -0.5s, +0.5s, +1.5s}
    Maps [-max, max] -> 4 uniform levels

    Returns:
        (q_codes, scales, zero_points)
        q_codes: int tensor with values in {0,1,2,3}
        scales: per-block float scales
        zero_points: zeros (symmetric)
    """
    w_flat = weight.reshape(-1).float()
    n = w_flat.numel()

    W, _ = _pad_and_reshape(w_flat, block_size)             # [n_blocks, block_size]
    n_blocks = W.shape[0]

    # scale such that max abs value maps to ±1.5 * scale
    scales = W.abs().max(dim=1).values.clamp(min=1e-8) / 1.5   # [n_blocks]

    # Encode: code = round(w/scale + 1.5), clamped to [0, 3]
    codes = (W / scales.unsqueeze(1) + 1.5).round().clamp(0, 3).to(torch.uint8)
    q_codes = codes.reshape(-1)[:n].reshape(weight.shape)

    zero_points = torch.zeros(n_blocks, device=weight.device)
    return q_codes, scales, zero_points


def dequantize_2bit_symmetric(
    q_codes: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Reconstruct float weight from 2-bit codes and scales.
    Levels: code 0 -> -1.5s, 1 -> -0.5s, 2 -> +0.5s, 3 -> +1.5s
    """
    flat_codes = q_codes.reshape(-1).float()
    n = flat_codes.numel()

    C, _ = _pad_and_reshape(flat_codes, block_size)     # [n_blocks, block_size]
    # Decode: level = (code - 1.5) * scale
    w = (C - 1.5) * scales.unsqueeze(1)                 # [n_blocks, block_size]
    w_flat = w.reshape(-1)[:n]

    return w_flat.reshape(q_codes.shape)


def quantize_2bit_asymmetric(
    weight: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric 2-bit: uses full [min, max] range with zero point.
    Often better for activations, provided as ablation comparison.
    """
    w_flat = weight.reshape(-1).float()
    n = w_flat.numel()

    W, _ = _pad_and_reshape(w_flat, block_size)         # [n_blocks, block_size]

    w_min = W.min(dim=1).values                         # [n_blocks]
    w_max = W.max(dim=1).values
    scales = ((w_max - w_min) / 3.0).clamp(min=1e-8)
    zero_points = (-w_min / scales).round().clamp(0, 3)

    codes = (W / scales.unsqueeze(1) + zero_points.unsqueeze(1)).round().clamp(0, 3).to(torch.uint8)
    q_codes = codes.reshape(-1)[:n].reshape(weight.shape)

    return q_codes, scales, zero_points


def dequantize_2bit_asymmetric(
    q_codes: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    flat_codes = q_codes.reshape(-1).float()
    n = flat_codes.numel()

    C, _ = _pad_and_reshape(flat_codes, block_size)     # [n_blocks, block_size]
    w = (C - zero_points.unsqueeze(1)) * scales.unsqueeze(1)
    w_flat = w.reshape(-1)[:n]

    return w_flat.reshape(q_codes.shape)


# -----------------------------------------------------------------------
# 4-bit: INT4 symmetric quantization
# -----------------------------------------------------------------------

def quantize_4bit(
    weight: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    INT4 symmetric quantization: levels ∈ {-7,...,7} (15 levels + zero)
    Scale = max(|w|) / 7 per block.

    Returns:
        (q_codes, scales): q_codes in {0..14} (offset by 7), scales per block
    """
    w_flat = weight.reshape(-1).float()
    n = w_flat.numel()

    W, _ = _pad_and_reshape(w_flat, block_size)                  # [n_blocks, block_size]
    scales = W.abs().max(dim=1).values.clamp(min=1e-8) / 7.0    # [n_blocks]

    codes = (W / scales.unsqueeze(1)).round().clamp(-7, 7).add(7).to(torch.uint8)
    q_codes = codes.reshape(-1)[:n].reshape(weight.shape)

    return q_codes, scales


def dequantize_4bit(
    q_codes: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    flat_codes = q_codes.reshape(-1).float()
    n = flat_codes.numel()

    C, _ = _pad_and_reshape(flat_codes, block_size)     # [n_blocks, block_size]
    w = (C - 7.0) * scales.unsqueeze(1)
    w_flat = w.reshape(-1)[:n]

    return w_flat.reshape(q_codes.shape)


# -----------------------------------------------------------------------
# Block-wise scale refinement (Phase 3)
# -----------------------------------------------------------------------

def refine_scale_blockwise(
    original_weight: torch.Tensor,
    q_codes: torch.Tensor,
    initial_scales: torch.Tensor,
    dequant_fn,
    n_iterations: int = 20,   # kept for API compatibility, no longer used
    lr: float = 0.01,          # kept for API compatibility, no longer used
    block_size: int = 128,
    level_offset: float = 1.5,
) -> torch.Tensor:
    """
    Closed-form optimal per-block scale that minimises L2 reconstruction error.

    For reconstruction  W ≈ (codes − offset) × s,  the least-squares solution is:

        s* = (W · L) / (L · L),   where L = codes − offset

    This replaces the previous approach of creating one Adam optimiser per block
    (O(n_blocks × n_iterations) individual gradient steps) with a single pair of
    vectorised dot-products — orders of magnitude faster.
    """
    w_flat = original_weight.reshape(-1).float()
    codes_flat = q_codes.reshape(-1).float()
    n = w_flat.numel()
    n_blocks = initial_scales.numel()

    # Pad to an exact multiple of block_size so we can reshape into a matrix
    pad = (block_size - n % block_size) % block_size
    if pad:
        w_padded = torch.cat([w_flat, w_flat.new_zeros(pad)])
        c_padded = torch.cat([codes_flat, codes_flat.new_zeros(pad)])
    else:
        w_padded = w_flat
        c_padded = codes_flat

    # [n_blocks, block_size]
    W = w_padded[: n_blocks * block_size].reshape(n_blocks, block_size)
    L = c_padded[: n_blocks * block_size].reshape(n_blocks, block_size) - level_offset

    # Optimal scale per block
    num = (W * L).sum(dim=1)
    den = (L * L).sum(dim=1).clamp(min=1e-8)
    scales = (num / den).clamp(min=1e-8)

    return scales


# -----------------------------------------------------------------------
# Unified quantizer dispatch
# -----------------------------------------------------------------------

def quantize_weight(
    weight: torch.Tensor,
    bits: int,
    scheme: str = "symmetric",
    block_size: int = 128,
    refine_scales: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """
    Dispatch to appropriate quantizer based on bit width.

    Args:
        weight: FP weight tensor
        bits: 1, 2, or 4
        scheme: "symmetric" or "asymmetric" (only affects 2-bit)
        block_size: Block size for scale computation
        refine_scales: Whether to run block-wise scale refinement

    Returns:
        (dequantized_weight, metadata)
        metadata contains: {bits, scheme, scales, zero_points, ...}
    """
    weight = weight.float()

    if bits == 1:
        q_weight, scales = quantize_1bit(weight, block_size)
        deq_weight = dequantize_1bit(q_weight, scales, block_size)
        meta = {"bits": 1, "scales": scales}

    elif bits == 2:
        if scheme == "symmetric":
            q_codes, scales, zero_points = quantize_2bit_symmetric(weight, block_size)
            if refine_scales:
                scales = refine_scale_blockwise(
                    weight, q_codes, scales,
                    dequant_fn=dequantize_2bit_symmetric,
                    block_size=block_size,
                    level_offset=1.5,   # 2-bit codes ∈ {0,1,2,3} → levels = codes − 1.5
                )
            deq_weight = dequantize_2bit_symmetric(q_codes, scales, block_size)
        else:
            q_codes, scales, zero_points = quantize_2bit_asymmetric(weight, block_size)
            deq_weight = dequantize_2bit_asymmetric(q_codes, scales, zero_points, block_size)

        meta = {"bits": 2, "scheme": scheme, "scales": scales, "zero_points": zero_points}

    elif bits == 4:
        q_codes, scales = quantize_4bit(weight, block_size)
        if refine_scales:
            scales = refine_scale_blockwise(
                weight, q_codes, scales,
                dequant_fn=dequantize_4bit,
                block_size=block_size,
                level_offset=7.0,   # 4-bit codes ∈ {0..14} → levels = codes − 7
            )
        deq_weight = dequantize_4bit(q_codes, scales, block_size)
        meta = {"bits": 4, "scales": scales}

    else:
        raise ValueError(f"Unsupported bit width: {bits}. Choose from {{1, 2, 4}}")

    return deq_weight, meta
