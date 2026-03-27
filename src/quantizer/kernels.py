"""
quantizer/kernels.py

Quantization kernels for 1-bit, 2-bit, and 4-bit precision.

  - 1-bit:  Sign-based binary quantization (BitNet style)
  - 2-bit:  Symmetric 4-level uniform quantization
  - 4-bit:  INT4 (16-level uniform quantization)

Also implements:
  - Asymmetric 2-bit variant (ablation)
  - Block-wise scale factor optimization (minimize L2 reconstruction error)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)


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
    w_flat = weight.reshape(-1)
    n = w_flat.numel()
    n_blocks = (n + block_size - 1) // block_size

    scales = []
    q_flat = torch.empty_like(w_flat)

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = w_flat[start:end]

        scale = block.abs().mean().clamp(min=1e-8)
        # Quantize: sign(w) * scale
        q_flat[start:end] = block.sign() * scale
        scales.append(scale)

    scales_tensor = torch.stack(scales)
    return q_flat.reshape(weight.shape), scales_tensor


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
    Symmetric 2-bit quantization: levels ∈ {-3, -1, +1, +3} * scale/3
    Maps [-max, max] -> 4 uniform levels

    Returns:
        (q_codes, scales, zero_points)
        q_codes: int tensor with values in {0,1,2,3}
        scales: per-block float scales
        zero_points: zeros (symmetric)
    """
    w_flat = weight.reshape(-1).float()
    n = w_flat.numel()
    n_blocks = (n + block_size - 1) // block_size

    q_codes = torch.empty(n, dtype=torch.uint8, device=weight.device)
    scales = []
    zero_points = torch.zeros(n_blocks, device=weight.device)

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = w_flat[start:end]

        scale = block.abs().max().clamp(min=1e-8) / 1.5  # max val = 3 * scale / 2
        # Map to {0,1,2,3}: 0=-max, 1=-scale/3*something, etc
        # Levels: {-1.5s, -0.5s, +0.5s, +1.5s}
        # Encode: floor((w/scale + 1.5) * 2/3 * 1.5)
        codes = torch.clamp(
            torch.round((block / scale + 1.5) / 1.0),
            0, 3
        ).to(torch.uint8)

        q_codes[start:end] = codes
        scales.append(scale)

    scales_tensor = torch.stack(scales)
    return q_codes.reshape(weight.shape), scales_tensor, zero_points


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
    n_blocks = (n + block_size - 1) // block_size

    w_flat = torch.empty(n, dtype=torch.float32, device=q_codes.device)

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        codes = flat_codes[start:end]
        scale = scales[i]
        # Decode: level = (code - 1.5) * scale
        w_flat[start:end] = (codes - 1.5) * scale

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
    n_blocks = (n + block_size - 1) // block_size

    q_codes = torch.empty(n, dtype=torch.uint8, device=weight.device)
    scales = []
    zero_points = []

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = w_flat[start:end]

        w_min = block.min()
        w_max = block.max()
        scale = (w_max - w_min) / 3.0  # 4 levels -> 3 intervals
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-w_min / scale).clamp(0, 3)

        codes = torch.clamp(torch.round(block / scale + zero_point), 0, 3).to(torch.uint8)
        q_codes[start:end] = codes
        scales.append(scale)
        zero_points.append(zero_point)

    return (
        q_codes.reshape(weight.shape),
        torch.stack(scales),
        torch.stack(zero_points),
    )


def dequantize_2bit_asymmetric(
    q_codes: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    flat_codes = q_codes.reshape(-1).float()
    n = flat_codes.numel()
    n_blocks = (n + block_size - 1) // block_size
    w_flat = torch.empty(n, dtype=torch.float32, device=q_codes.device)

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        w_flat[start:end] = (flat_codes[start:end] - zero_points[i]) * scales[i]

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
    n_blocks = (n + block_size - 1) // block_size

    q_codes = torch.empty(n, dtype=torch.uint8, device=weight.device)
    scales = []

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = w_flat[start:end]

        scale = block.abs().max().clamp(min=1e-8) / 7.0
        codes = torch.clamp(torch.round(block / scale) + 7, 0, 14).to(torch.uint8)

        q_codes[start:end] = codes
        scales.append(scale)

    return q_codes.reshape(weight.shape), torch.stack(scales)


def dequantize_4bit(
    q_codes: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    flat_codes = q_codes.reshape(-1).float()
    n = flat_codes.numel()
    n_blocks = (n + block_size - 1) // block_size
    w_flat = torch.empty(n, dtype=torch.float32, device=q_codes.device)

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        w_flat[start:end] = (flat_codes[start:end] - 7.0) * scales[i]

    return w_flat.reshape(q_codes.shape)


# -----------------------------------------------------------------------
# Block-wise scale refinement (Phase 3)
# -----------------------------------------------------------------------

def refine_scale_blockwise(
    original_weight: torch.Tensor,
    q_codes: torch.Tensor,
    initial_scales: torch.Tensor,
    dequant_fn,
    n_iterations: int = 20,
    lr: float = 0.01,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Minimize L2 reconstruction error via per-block scale optimization.

    min_scale || W_orig - dequant(q_codes, scale) ||_2^2

    Uses simple gradient descent per block.

    Args:
        original_weight: Original FP weight tensor
        q_codes: Quantized integer codes
        initial_scales: Initial scale estimates
        dequant_fn: Dequantization function (q_codes, scales) -> float tensor
        n_iterations: GD steps per block
        lr: Learning rate

    Returns:
        Refined scales tensor
    """
    scales = initial_scales.clone().float()

    w_flat = original_weight.reshape(-1).float()
    n = w_flat.numel()
    n_blocks = scales.numel()

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + n // n_blocks, n)  # approximate
        start = i * (n // n_blocks)
        end = min(start + n // n_blocks, n)

        if start >= n:
            break

        w_block = w_flat[start:end]
        codes_flat = q_codes.reshape(-1)[start:end].float()

        scale = scales[i:i+1].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([scale], lr=lr)

        for _ in range(n_iterations):
            optimizer.zero_grad()
            # Reconstruct for 2-bit symmetric (most common)
            reconstructed = (codes_flat - 1.5) * scale
            loss = (w_block - reconstructed).pow(2).mean()
            loss.backward()
            optimizer.step()
            scale.data.clamp_(min=1e-8)

        scales[i] = scale.data.item()

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
            )
        deq_weight = dequantize_4bit(q_codes, scales, block_size)
        meta = {"bits": 4, "scales": scales}

    else:
        raise ValueError(f"Unsupported bit width: {bits}. Choose from {{1, 2, 4}}")

    return deq_weight, meta
