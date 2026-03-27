"""
baselines/gptq_runner.py

GPTQ baseline via auto-gptq library.
Runs 4-bit GPTQ quantization for comparison against our method.

Usage:
    runner = GPTQRunner(model_name_or_path, n_calibration=128)
    quantized_model = runner.run(calibration_dataloader)
"""

import torch
import logging
from typing import Optional, List
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False
    logger.warning("auto-gptq not available. Install with: pip install auto-gptq")


class GPTQRunner:
    """
    Runs GPTQ 4-bit quantization for baseline comparison.

    Note: GPTQ requires the model to be loaded fresh from HF,
    not an already-loaded model, due to its quantization hooks.
    """

    def __init__(
        self,
        model_name_or_path: str,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = True,
        use_triton: bool = False,
    ):
        if not GPTQ_AVAILABLE:
            raise ImportError("auto-gptq is required for GPTQ baseline. "
                              "Install: pip install auto-gptq")

        self.model_name_or_path = model_name_or_path
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.use_triton = use_triton

    def run(
        self,
        calibration_dataset: List[dict],
        output_dir: Optional[str] = None,
    ):
        """
        Run GPTQ quantization.

        Args:
            calibration_dataset: List of {"input_ids": tensor} dicts
            output_dir: Where to save quantized model (optional)

        Returns:
            Quantized model
        """
        quantize_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
        )

        logger.info(f"Loading model for GPTQ: {self.model_name_or_path}")
        model = AutoGPTQForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantize_config=quantize_config,
        )

        logger.info(f"Running GPTQ {self.bits}-bit quantization...")
        model.quantize(calibration_dataset)

        if output_dir:
            model.save_quantized(output_dir, use_safetensors=True)
            logger.info(f"GPTQ model saved to {output_dir}")

        return model

    @staticmethod
    def name() -> str:
        return f"gptq_4bit"

    @staticmethod
    def avg_bits() -> float:
        return 4.0


def prepare_gptq_calibration(dataloader: DataLoader, n_samples: int = 128) -> List[dict]:
    """
    Convert standard dataloader to GPTQ calibration format.
    GPTQ expects List[{"input_ids": tensor}].
    """
    samples = []
    for batch in dataloader:
        if len(samples) >= n_samples:
            break
        input_ids = batch["input_ids"]
        for i in range(input_ids.shape[0]):
            samples.append({"input_ids": input_ids[i:i+1]})
            if len(samples) >= n_samples:
                break
    logger.info(f"Prepared {len(samples)} GPTQ calibration samples")
    return samples
