"""
utils/logging_utils.py

Logging setup, results tracking, and experiment utilities.
"""

import logging
import os
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import torch


def setup_logging(
    log_dir: str = "logs",
    experiment_name: str = "experiment",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.utils._http",
                  "datasets", "datasets.load", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(f"Logging to {log_file}")
    return logger


class ResultsTracker:
    """
    Track and save experiment results across baselines and ablations.
    Maintains a structured JSON results file.
    """

    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.results: Dict[str, Any] = {
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "models": {},
        }
        os.makedirs(output_dir, exist_ok=True)

    def add_result(self, model_name: str, metric_dict: Dict):
        """Add evaluation results for a model/method."""
        if model_name not in self.results["models"]:
            self.results["models"][model_name] = {}
        self.results["models"][model_name].update(metric_dict)
        self.save()

    def save(self):
        path = os.path.join(self.output_dir, f"{self.experiment_name}_results.json")
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def print_summary(self):
        """Print a comparison table to console."""
        logger = logging.getLogger(__name__)
        logger.info("\n" + "=" * 80)
        logger.info(f"RESULTS SUMMARY: {self.experiment_name}")
        logger.info("=" * 80)

        for model_name, metrics in self.results["models"].items():
            logger.info(f"\n[{model_name}]")

            # Perplexity
            for ds in ["wikitext2", "ptb"]:
                key = f"ppl_{ds}"
                if key in metrics and "perplexity" in metrics[key]:
                    ppl = metrics[key]["perplexity"]
                    logger.info(f"  PPL ({ds}): {ppl:.2f}")

            # MMLU
            if "mmlu" in metrics and "mmlu_macro_avg_pct" in metrics.get("mmlu", {}):
                logger.info(f"  MMLU: {metrics['mmlu']['mmlu_macro_avg_pct']:.2f}%")

            # Memory
            if "memory" in metrics:
                mem = metrics["memory"]
                if "param_memory_gb" in mem:
                    logger.info(f"  Model size: {mem['param_memory_gb']:.3f} GB")

            # Avg bits (if quantized)
            if "avg_bits" in metrics:
                logger.info(f"  Avg bits: {metrics['avg_bits']:.3f}")

        logger.info("=" * 80)


def get_model_size_gb(model: torch.nn.Module) -> float:
    """Calculate model parameter size in GB."""
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / 1e9


def count_parameters(model: torch.nn.Module) -> Dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
