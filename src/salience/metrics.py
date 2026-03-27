"""
salience/metrics.py

Five complementary salience metrics for weight importance scoring.
Each captures a different aspect of weight criticality.

Metrics:
  1. magnitude_l1  - L1 norm of weight tensor
  2. magnitude_l2  - L2 norm of weight tensor
  3. gradient      - gradient sensitivity (|w * dL/dw|)
  4. hessian       - Fisher information diagonal approximation
  5. activation    - activation-aware scaling (SmoothQuant-style)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SalienceConfig:
    """Configuration for salience computation."""
    metrics: List[str] = field(default_factory=lambda: [
        "magnitude_l1", "magnitude_l2", "gradient", "hessian", "activation"
    ])
    # Ensemble alpha weights (must sum to 1.0 or will be normalized)
    alpha_magnitude_l1: float = 0.15
    alpha_magnitude_l2: float = 0.15
    alpha_gradient: float = 0.25
    alpha_hessian: float = 0.25
    alpha_activation: float = 0.20

    # Calibration
    n_calibration_samples: int = 512
    calibration_seq_len: int = 512

    # Fisher approximation
    n_fisher_samples: int = 128  # subset for Hessian (expensive)

    # Granularity: "weight", "channel", "layer"
    granularity: str = "weight"


class MagnitudeSalience:
    """
    L1/L2 magnitude salience.
    Simple but effective: large weights tend to be more important.
    |w|_p for p in {1, 2}
    """

    @staticmethod
    def compute(weight: torch.Tensor, norm: str = "l2") -> torch.Tensor:
        """
        Args:
            weight: Parameter tensor of any shape
            norm: "l1" or "l2"
        Returns:
            Salience tensor of same shape as weight (element-wise absolute value)
        """
        if norm == "l1":
            return weight.abs()
        elif norm == "l2":
            return weight.pow(2)
        else:
            raise ValueError(f"Unknown norm: {norm}")


class GradientSalience:
    """
    Gradient-based salience: |w * dL/dw|
    Captures sensitivity of loss to each weight.
    Based on optimal brain damage (LeCun 1990).
    """

    def __init__(self):
        self._grad_accumulator: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def register_hooks(self, model: nn.Module):
        """Register backward hooks to capture gradients."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self._accumulate_grad(n, grad)
                )
                self._hooks.append(hook)

    def _accumulate_grad(self, name: str, grad: torch.Tensor):
        if name not in self._grad_accumulator:
            self._grad_accumulator[name] = torch.zeros_like(grad)
        self._grad_accumulator[name] += grad.abs()

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def compute(self, weight: torch.Tensor, param_name: str) -> torch.Tensor:
        """
        Compute |w * dL/dw| salience.
        Must call register_hooks() and run forward/backward passes first.
        """
        if param_name not in self._grad_accumulator:
            logger.warning(f"No gradient found for {param_name}, using magnitude fallback")
            return weight.abs()

        grad = self._grad_accumulator[param_name]
        return (weight.abs() * grad).detach()

    def reset(self):
        self._grad_accumulator.clear()


class HessianSalience:
    """
    Hessian diagonal salience via empirical Fisher information.
    F_ii ≈ E[(dL/dw_i)^2]  (diagonal Fisher approximation)
    More expensive than gradient but captures curvature.
    Based on Optimal Brain Surgeon (Hassibi & Stork 1993).
    """

    def __init__(self):
        self._fisher_accumulator: Dict[str, torch.Tensor] = {}
        self._n_samples = 0

    def accumulate(self, model: nn.Module):
        """
        Call after each backward pass to accumulate Fisher diagonal.
        Must call model.zero_grad() before each forward pass.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sq = param.grad.data.pow(2)
                if name not in self._fisher_accumulator:
                    self._fisher_accumulator[name] = torch.zeros_like(param.data)
                self._fisher_accumulator[name] += grad_sq
        self._n_samples += 1

    def compute(self, weight: torch.Tensor, param_name: str) -> torch.Tensor:
        """
        Return normalized Fisher diagonal as salience.
        Higher Fisher → weight is in a sharper basin → more important.
        """
        if param_name not in self._fisher_accumulator or self._n_samples == 0:
            logger.warning(f"No Fisher info for {param_name}, using magnitude fallback")
            return weight.abs()

        fisher = self._fisher_accumulator[param_name] / self._n_samples
        return fisher.detach()

    def reset(self):
        self._fisher_accumulator.clear()
        self._n_samples = 0


class ActivationSalience:
    """
    Activation-aware salience (inspired by AWQ/SmoothQuant).
    Weights connected to high-activation channels are more important.
    salience_ij = |w_ij| * max_activation_j
    """

    def __init__(self):
        self._activation_stats: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on linear-like layers to track input activations.

        Covers both nn.Linear and HuggingFace Conv1D (used in GPT-2), which both
        have a 2-D weight parameter but are different module types.
        """
        for name, module in model.named_modules():
            has_2d_weight = (
                isinstance(module, nn.Linear)
                or (
                    hasattr(module, "weight")
                    and isinstance(module.weight, nn.Parameter)
                    and module.weight.dim() == 2
                    and not isinstance(module, nn.Embedding)
                )
            )
            if has_2d_weight:
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._capture_activation(n, inp[0])
                )
                self._hooks.append(hook)

    def _capture_activation(self, name: str, activation: torch.Tensor):
        """Track per-channel max activation across batch/seq dims."""
        # activation: [batch, seq, in_features] or [batch, in_features]
        if activation.dim() == 3:
            act_abs = activation.abs().max(dim=0).values.max(dim=0).values  # [in_features]
        elif activation.dim() == 2:
            act_abs = activation.abs().max(dim=0).values  # [in_features]
        else:
            act_abs = activation.abs()

        if name not in self._activation_stats:
            self._activation_stats[name] = act_abs.detach()
        else:
            self._activation_stats[name] = torch.maximum(
                self._activation_stats[name], act_abs.detach()
            )

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def compute(self, weight: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Compute activation-aware salience.
        For weight matrix W [out, in], scale each input channel by its activation magnitude.
        """
        if layer_name not in self._activation_stats:
            logger.warning(f"No activation stats for {layer_name}, using magnitude fallback")
            return weight.abs()

        act_scale = self._activation_stats[layer_name]  # [in_features]

        if weight.dim() == 2 and weight.shape[1] == act_scale.shape[0]:
            # nn.Linear layout: weight is [out, in]
            salience = weight.abs() * act_scale.unsqueeze(0)
        elif weight.dim() == 2 and weight.shape[0] == act_scale.shape[0]:
            # HuggingFace Conv1D layout: weight is [in, out] (transposed)
            salience = weight.abs() * act_scale.unsqueeze(1)
        else:
            # Shape mismatch fallback
            logger.warning(f"Activation shape mismatch for {layer_name}: "
                           f"weight {weight.shape}, act {act_scale.shape}")
            salience = weight.abs()

        return salience.detach()

    def reset(self):
        self._activation_stats.clear()


class EnsembleSalience:
    """
    Weighted ensemble of all 5 salience metrics.
    Each metric normalized to [0,1] before combining.
    Final score: Σ α_i * normalize(metric_i)
    """

    def __init__(self, config: SalienceConfig):
        self.config = config
        self.alpha = self._build_alpha_dict()

    def _build_alpha_dict(self) -> Dict[str, float]:
        raw = {
            "magnitude_l1": self.config.alpha_magnitude_l1,
            "magnitude_l2": self.config.alpha_magnitude_l2,
            "gradient": self.config.alpha_gradient,
            "hessian": self.config.alpha_hessian,
            "activation": self.config.alpha_activation,
        }
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Min-max normalize to [0, 1]. Handles zero-range tensors."""
        t_min = tensor.min()
        t_max = tensor.max()
        if (t_max - t_min).abs() < 1e-8:
            return torch.zeros_like(tensor)
        return (tensor - t_min) / (t_max - t_min)

    def combine(self, scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine available metric scores with weights.
        Args:
            scores: dict of metric_name -> salience tensor (same shape)
        Returns:
            ensemble salience tensor
        """
        if not scores:
            raise ValueError("No salience scores provided")

        result = None
        total_alpha = 0.0

        for metric_name, score in scores.items():
            if metric_name not in self.alpha:
                logger.warning(f"Unknown metric {metric_name}, skipping")
                continue
            alpha = self.alpha[metric_name]
            normalized = self._normalize(score.float())

            if result is None:
                result = alpha * normalized
            else:
                result = result + alpha * normalized
            total_alpha += alpha

        if result is None:
            raise ValueError("No valid metrics found")

        # Renormalize if some metrics were missing
        if total_alpha < 0.999:
            result = result / total_alpha

        return result
