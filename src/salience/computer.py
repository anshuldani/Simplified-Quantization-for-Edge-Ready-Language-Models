"""
salience/computer.py

SalienceComputer orchestrates running the model on calibration data
and collecting all 5 salience metrics per parameter.

Returns: Dict[param_name -> salience_tensor] ready for bit allocator.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import logging

from .metrics import (
    SalienceConfig,
    MagnitudeSalience,
    GradientSalience,
    HessianSalience,
    ActivationSalience,
    EnsembleSalience,
)

logger = logging.getLogger(__name__)


class SalienceComputer:
    """
    Runs calibration data through model to compute per-weight salience scores.

    Usage:
        computer = SalienceComputer(model, config)
        salience_map = computer.compute(calibration_dataloader)
        # salience_map: {param_name: tensor of same shape as param}
    """

    def __init__(self, model: nn.Module, config: SalienceConfig, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device

        self.magnitude = MagnitudeSalience()
        self.gradient_metric = GradientSalience()
        self.hessian_metric = HessianSalience()
        self.activation_metric = ActivationSalience()
        self.ensemble = EnsembleSalience(config)

        # Map layer names (for activation hook) to param names
        self._layer_to_param: Dict[str, str] = {}

    def _build_layer_param_map(self):
        """Build mapping from layer name to weight param name for activation lookup."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                param_name = f"{name}.weight"
                self._layer_to_param[name] = param_name

    def compute(
        self,
        dataloader: DataLoader,
        target_params: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Main entry: compute salience scores for all (or specified) parameters.

        Args:
            dataloader: Calibration dataloader (C4 validation, 512 samples)
            target_params: If set, only compute for these param names.
                           Defaults to all weight matrices in linear layers.

        Returns:
            Dict mapping param_name -> salience tensor (same shape as param)
        """
        self._build_layer_param_map()
        self.model.eval()
        self.model.to(self.device)

        # Identify target parameters (weight matrices only, not biases)
        if target_params is None:
            target_params = [
                name for name, param in self.model.named_parameters()
                if "weight" in name and param.dim() >= 2
            ]
        logger.info(f"Computing salience for {len(target_params)} parameters")

        # Enable gradients for gradient/hessian metrics
        needs_grad = any(m in self.config.metrics for m in ["gradient", "hessian"])

        # ------- Phase 1: Register hooks -------
        if "gradient" in self.config.metrics:
            self.gradient_metric.register_hooks(self.model)

        if "activation" in self.config.metrics:
            self.activation_metric.register_hooks(self.model)

        # ------- Phase 2: Forward/backward passes -------
        n_samples = 0
        n_target = min(self.config.n_calibration_samples, len(dataloader.dataset)
                       if hasattr(dataloader.dataset, '__len__') else self.config.n_calibration_samples)

        logger.info(f"Running calibration forward passes ({n_target} samples)...")

        with torch.set_grad_enabled(needs_grad):
            for batch in tqdm(dataloader, desc="Calibration", total=len(dataloader)):
                if n_samples >= n_target:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                if needs_grad:
                    self.model.zero_grad()

                # Forward pass — use causal LM loss for gradient signal
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,  # standard causal LM objective
                )
                loss = outputs.loss

                if needs_grad and loss is not None:
                    loss.backward()

                    if "hessian" in self.config.metrics:
                        self.hessian_metric.accumulate(self.model)

                n_samples += input_ids.shape[0]

        logger.info(f"Calibration complete ({n_samples} samples processed)")

        # ------- Phase 3: Clean up hooks -------
        if "gradient" in self.config.metrics:
            self.gradient_metric.remove_hooks()
        if "activation" in self.config.metrics:
            self.activation_metric.remove_hooks()

        # ------- Phase 4: Assemble per-param salience -------
        salience_map: Dict[str, torch.Tensor] = {}

        for param_name in tqdm(target_params, desc="Computing salience scores"):
            param = dict(self.model.named_parameters()).get(param_name)
            if param is None:
                logger.warning(f"Parameter {param_name} not found, skipping")
                continue

            weight = param.data.detach()

            # Infer layer name from param name (strip ".weight" suffix)
            layer_name = param_name.rsplit(".weight", 1)[0]

            scores = {}

            if "magnitude_l1" in self.config.metrics:
                scores["magnitude_l1"] = self.magnitude.compute(weight, norm="l1")

            if "magnitude_l2" in self.config.metrics:
                scores["magnitude_l2"] = self.magnitude.compute(weight, norm="l2")

            if "gradient" in self.config.metrics:
                scores["gradient"] = self.gradient_metric.compute(weight, param_name)

            if "hessian" in self.config.metrics:
                scores["hessian"] = self.hessian_metric.compute(weight, param_name)

            if "activation" in self.config.metrics:
                scores["activation"] = self.activation_metric.compute(weight, layer_name)

            # Ensemble combination
            if len(scores) == 1:
                salience_map[param_name] = list(scores.values())[0]
            else:
                salience_map[param_name] = self.ensemble.combine(scores)

        # Cleanup
        self.gradient_metric.reset()
        self.hessian_metric.reset()
        self.activation_metric.reset()

        logger.info(f"Salience computation complete for {len(salience_map)} parameters")
        return salience_map

    def compute_single_metric(
        self,
        dataloader: DataLoader,
        metric: str,
        target_params: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method for ablation studies: compute only one metric.
        Temporarily overrides config metrics.
        """
        original_metrics = self.config.metrics
        self.config.metrics = [metric]
        result = self.compute(dataloader, target_params)
        self.config.metrics = original_metrics
        return result

    def get_salience_stats(self, salience_map: Dict[str, torch.Tensor]) -> Dict:
        """Compute summary statistics for logging/visualization."""
        stats = {}
        all_scores = []

        for name, scores in salience_map.items():
            flat = scores.flatten().float()
            all_scores.append(flat)
            stats[name] = {
                "mean": flat.mean().item(),
                "std": flat.std().item(),
                "min": flat.min().item(),
                "max": flat.max().item(),
                "p25": flat.quantile(0.25).item(),
                "p50": flat.quantile(0.50).item(),
                "p75": flat.quantile(0.75).item(),
                "p95": flat.quantile(0.95).item(),
                "numel": flat.numel(),
            }

        if all_scores:
            global_flat = torch.cat(all_scores)
            stats["_global"] = {
                "mean": global_flat.mean().item(),
                "std": global_flat.std().item(),
                "p80": global_flat.quantile(0.80).item(),  # 80/20 threshold
                "total_params": global_flat.numel(),
            }

        return stats
