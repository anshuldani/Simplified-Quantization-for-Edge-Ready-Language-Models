"""
eval/evaluator.py

Evaluation suite for quantized models:
  1. Perplexity on WikiText-2 and PTB
  2. MMLU (5-shot) via lm-evaluation-harness
  3. Latency / throughput profiling
  4. Memory (VRAM) profiling
"""

import contextlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
import json
import logging
import os
import math

logger = logging.getLogger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


# -----------------------------------------------------------------------
# Perplexity evaluation
# -----------------------------------------------------------------------

class PerplexityEvaluator:
    """
    Compute perplexity on WikiText-2 or Penn Treebank.
    Uses sliding window approach for long sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        stride: int = 512,
        max_length: int = 1024,
        max_eval_tokens: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.stride = stride
        self.max_length = max_length
        self.max_eval_tokens = max_eval_tokens

    def evaluate(self, dataset_name: str = "wikitext2") -> Dict:
        """
        Compute perplexity on given dataset.

        Args:
            dataset_name: "wikitext2" or "ptb"

        Returns:
            Dict with perplexity, nll, n_tokens
        """
        from datasets import load_dataset

        logger.info(f"Evaluating perplexity on {dataset_name}...")

        if dataset_name == "wikitext2":
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(data["text"])
        elif dataset_name == "ptb":
            # ptb_text_only is gated on HuggingFace; fall back to the wikitext103
            # validation set as a proxy second-dataset comparison.
            try:
                data = load_dataset("ptb_text_only", "penn_treebank", split="test")
                text = "\n\n".join(data["sentence"])
            except Exception:
                logger.warning("ptb_text_only unavailable, using wikitext-103-raw-v1 validation as PTB proxy")
                data = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
                text = "\n\n".join(data["text"])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids

        nlls = []
        n_tokens = 0
        seq_len = input_ids.size(1)
        if self.max_eval_tokens is not None:
            seq_len = min(seq_len, self.max_eval_tokens)

        self.model.eval()
        self.model.to(self.device)

        prev_end = 0
        with torch.no_grad():
            for begin_loc in tqdm(
                range(0, seq_len, self.stride),
                desc=f"Perplexity ({dataset_name})"
            ):
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - prev_end  # tokens to score this step
                input_ids_chunk = input_ids[:, begin_loc:end_loc].to(self.device)

                autocast_ctx = (
                    torch.amp.autocast("cuda")
                    if self.device == "cuda" and torch.cuda.is_available()
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    outputs = self.model(input_ids_chunk, labels=input_ids_chunk)
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood.item())
                n_tokens += trg_len
                prev_end = end_loc

                if end_loc == seq_len:
                    break

        ppl = math.exp(sum(nlls) / n_tokens)
        nll = sum(nlls) / n_tokens

        logger.info(f"{dataset_name} perplexity: {ppl:.2f} (NLL: {nll:.4f})")

        return {
            "dataset": dataset_name,
            "perplexity": ppl,
            "nll": nll,
            "n_tokens": n_tokens,
        }


# -----------------------------------------------------------------------
# MMLU evaluation
# -----------------------------------------------------------------------

class MMULEvaluator:
    """
    MMLU 5-shot evaluation via lm-evaluation-harness.
    Evaluates on all 57 MMLU subjects.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(
        self,
        tasks: Optional[List[str]] = None,
        num_fewshot: int = 5,
        batch_size: int = 4,
    ) -> Dict:
        """
        Run MMLU evaluation.

        Args:
            tasks: MMLU task names. None = all 57 subjects.
            num_fewshot: Number of few-shot examples (5 for standard MMLU)
            batch_size: Batch size for evaluation

        Returns:
            Dict with per-subject accuracies and macro-average
        """
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
        except ImportError:
            logger.error("lm-eval not installed. Run: pip install lm-eval>=0.4.2")
            return {"error": "lm-eval not available"}

        if tasks is None:
            tasks = ["mmlu"]

        logger.info(f"Running MMLU {num_fewshot}-shot evaluation...")

        lm = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=batch_size,
        )

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            log_samples=False,
        )

        # Extract MMLU aggregate
        mmlu_results = results.get("results", {})

        # Compute macro-average across subjects
        accs = []
        per_subject = {}
        for task_name, task_results in mmlu_results.items():
            if "mmlu" in task_name.lower():
                acc = task_results.get("acc,none", task_results.get("acc", 0))
                per_subject[task_name] = acc
                accs.append(acc)

        macro_avg = sum(accs) / len(accs) if accs else 0.0

        logger.info(f"MMLU macro-average: {macro_avg:.4f} ({macro_avg*100:.2f}%)")

        return {
            "mmlu_macro_avg": macro_avg,
            "mmlu_macro_avg_pct": macro_avg * 100,
            "per_subject": per_subject,
            "n_subjects": len(accs),
        }


# -----------------------------------------------------------------------
# Latency / throughput profiling
# -----------------------------------------------------------------------

class LatencyProfiler:
    """
    Measure token generation latency and throughput.
    Profiles: TTFT (time to first token), ms/token, tokens/sec
    """

    def __init__(self, model: nn.Module, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def profile(
        self,
        batch_sizes: List[int] = [1, 8],
        seq_lengths: List[int] = [128, 256, 512],
        n_generate: int = 64,
        n_warmup: int = 3,
        n_runs: int = 10,
    ) -> Dict:
        """
        Profile latency across batch sizes and sequence lengths.

        Returns:
            Dict of {f"bs{bs}_sl{sl}": {ttft_ms, ms_per_token, tokens_per_sec}}
        """
        self.model.eval()
        self.model.to(self.device)
        results = {}

        for bs in batch_sizes:
            for sl in seq_lengths:
                key = f"bs{bs}_sl{sl}"
                logger.info(f"Profiling {key}...")

                # Create dummy input
                input_ids = torch.randint(100, 50000, (bs, sl)).to(self.device)

                # Warmup
                with torch.no_grad():
                    for _ in range(n_warmup):
                        _ = self.model.generate(
                            input_ids,
                            max_new_tokens=n_generate,
                            do_sample=False,
                        )

                # Time runs
                ttft_times = []
                total_times = []

                with torch.no_grad():
                    for _ in range(n_runs):
                        if self.device == "cuda":
                            torch.cuda.synchronize()

                        # TTFT: time to generate 1 token
                        t0 = time.perf_counter()
                        _ = self.model.generate(
                            input_ids,
                            max_new_tokens=1,
                            do_sample=False,
                        )
                        if self.device == "cuda":
                            torch.cuda.synchronize()
                        ttft = (time.perf_counter() - t0) * 1000

                        # Full generation time
                        t0 = time.perf_counter()
                        _ = self.model.generate(
                            input_ids,
                            max_new_tokens=n_generate,
                            do_sample=False,
                        )
                        if self.device == "cuda":
                            torch.cuda.synchronize()
                        total = (time.perf_counter() - t0) * 1000

                        ttft_times.append(ttft)
                        total_times.append(total)

                avg_ttft = sum(ttft_times) / len(ttft_times)
                avg_total = sum(total_times) / len(total_times)
                avg_ms_per_tok = avg_total / n_generate
                tokens_per_sec = (n_generate * bs) / (avg_total / 1000)

                results[key] = {
                    "batch_size": bs,
                    "seq_length": sl,
                    "ttft_ms": avg_ttft,
                    "ms_per_token": avg_ms_per_tok,
                    "tokens_per_sec": tokens_per_sec,
                    "total_ms": avg_total,
                }

                logger.info(f"  TTFT: {avg_ttft:.1f}ms, {avg_ms_per_tok:.2f}ms/tok, "
                            f"{tokens_per_sec:.1f} tok/s")

        return results


# -----------------------------------------------------------------------
# Memory profiler
# -----------------------------------------------------------------------

class MemoryProfiler:
    """
    Profile GPU memory usage (VRAM).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def get_vram_usage(self) -> Dict:
        """Get current VRAM usage in GB."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        device_idx = torch.cuda.current_device()

        return {
            "allocated_gb": torch.cuda.memory_allocated(device_idx) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(device_idx) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(device_idx) / 1e9,
        }

    def profile_model_size(self, model: nn.Module) -> Dict:
        """Estimate model parameter memory footprint."""
        total_params = 0
        total_bytes = 0

        for name, param in model.named_parameters():
            n = param.numel()
            bytes_per_elem = param.element_size()
            total_params += n
            total_bytes += n * bytes_per_elem

        return {
            "total_params": total_params,
            "param_memory_gb": total_bytes / 1e9,
            "param_memory_mb": total_bytes / 1e6,
        }


# -----------------------------------------------------------------------
# Full evaluator
# -----------------------------------------------------------------------

class ModelEvaluator:
    """
    Runs the full evaluation suite for a model.
    Perplexity (WikiText-2, PTB) + optional MMLU + latency + memory.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_all(
        self,
        run_mmlu: bool = True,
        run_latency: bool = True,
        run_perplexity: bool = True,
        datasets: List[str] = None,
        max_eval_tokens: Optional[int] = None,
    ) -> Dict:
        """Run full evaluation suite."""
        if datasets is None:
            datasets = ["wikitext2", "ptb"]

        results = {}

        if run_perplexity:
            ppl_eval = PerplexityEvaluator(self.model, self.tokenizer, self.device,
                                           max_eval_tokens=max_eval_tokens)
            for ds in datasets:
                try:
                    ppl_result = ppl_eval.evaluate(ds)
                    results[f"ppl_{ds}"] = ppl_result
                except Exception as e:
                    logger.error(f"Perplexity eval failed for {ds}: {e}")
                    results[f"ppl_{ds}"] = {"error": str(e)}

        if run_mmlu:
            try:
                mmlu_eval = MMULEvaluator(self.model, self.tokenizer, self.device)
                results["mmlu"] = mmlu_eval.evaluate()
            except Exception as e:
                logger.error(f"MMLU eval failed: {e}")
                results["mmlu"] = {"error": str(e)}

        if run_latency:
            try:
                lat_profiler = LatencyProfiler(self.model, self.tokenizer, self.device)
                results["latency"] = lat_profiler.profile()
            except Exception as e:
                logger.error(f"Latency profiling failed: {e}")
                results["latency"] = {"error": str(e)}

        mem_profiler = MemoryProfiler(self.device)
        results["memory"] = {
            **mem_profiler.profile_model_size(self.model),
            **mem_profiler.get_vram_usage(),
        }

        return results

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {output_path}")
