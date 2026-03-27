"""
utils/viz.py

Visualization utilities:
  - Salience score distributions
  - Bit allocation heatmaps
  - Ablation comparison plots
  - Perplexity vs compression curves
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import json
import logging

logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

COLORS = {
    "ours": "#2563EB",       # Blue - our method
    "fp16": "#16A34A",       # Green - upper bound
    "uniform_int2": "#DC2626",  # Red - naive baseline
    "bitnet": "#D97706",     # Orange - BitNet
    "gptq": "#7C3AED",       # Purple - GPTQ
}

METRIC_LABELS = {
    "magnitude_l1": "Magnitude L1",
    "magnitude_l2": "Magnitude L2",
    "gradient": "Gradient",
    "hessian": "Hessian (Fisher)",
    "activation": "Activation-Aware",
    "combined": "Ensemble",
}


def plot_salience_distributions(
    salience_stats: Dict,
    output_path: str,
    title: str = "Salience Score Distributions",
):
    """
    Plot per-layer salience distributions as violin plots.
    Shows mean, std, and quantiles across all layers.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Top: Global distribution histogram ---
    ax = axes[0]
    if "_global" in salience_stats:
        g = salience_stats["_global"]
        # Generate approximate distribution from stats
        means = [v["mean"] for k, v in salience_stats.items() if k != "_global"]
        stds = [v["std"] for k, v in salience_stats.items() if k != "_global"]

        ax.hist(means, bins=50, color=COLORS["ours"], alpha=0.7, edgecolor="white")
        ax.axvline(g.get("p80", 0), color=COLORS["uniform_int2"],
                   linestyle="--", linewidth=2, label="80th percentile (20% critical)")
        ax.set_xlabel("Layer Mean Salience Score")
        ax.set_ylabel("Count (Layers)")
        ax.set_title(f"{title} — Distribution of Layer Mean Salience")
        ax.legend()

    # --- Bottom: Per-percentile breakdown ---
    ax = axes[1]
    layer_names = [k for k in salience_stats.keys() if k != "_global"]

    if layer_names:
        p25 = [salience_stats[n]["p25"] for n in layer_names]
        p50 = [salience_stats[n]["p50"] for n in layer_names]
        p75 = [salience_stats[n]["p75"] for n in layer_names]
        p95 = [salience_stats[n]["p95"] for n in layer_names]

        x = range(len(layer_names))
        ax.fill_between(x, p25, p75, alpha=0.3, color=COLORS["ours"], label="IQR (25-75%)")
        ax.plot(x, p50, color=COLORS["ours"], linewidth=1.5, label="Median")
        ax.plot(x, p95, color=COLORS["bitnet"], linewidth=1, linestyle="--", label="95th pct")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Salience Score")
        ax.set_title("Salience Score Percentiles per Layer")
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Salience distribution plot saved to {output_path}")


def plot_bit_allocation_heatmap(
    bit_map_stats: Dict,
    output_path: str,
    title: str = "Bit Allocation per Layer",
):
    """
    Heatmap showing 1-bit vs 2-bit vs 4-bit distribution per layer.
    """
    layer_names = [k for k in bit_map_stats.keys() if k != "_summary"]
    if not layer_names:
        return

    data_1b = [bit_map_stats[n].get("1bit", 0) / max(bit_map_stats[n]["n_params"], 1) * 100
                for n in layer_names]
    data_2b = [bit_map_stats[n].get("2bit", 0) / max(bit_map_stats[n]["n_params"], 1) * 100
                for n in layer_names]
    data_4b = [bit_map_stats[n].get("4bit", 0) / max(bit_map_stats[n]["n_params"], 1) * 100
                for n in layer_names]

    fig, ax = plt.subplots(figsize=(14, max(6, len(layer_names) * 0.25)))

    x = range(len(layer_names))
    bar_width = 0.6

    ax.barh(x, data_1b, bar_width, label="1-bit (low salience)", color="#94A3B8")
    ax.barh(x, data_2b, bar_width, left=data_1b, label="2-bit", color=COLORS["ours"])
    ax.barh(x, data_4b, bar_width,
            left=[a+b for a, b in zip(data_1b, data_2b)],
            label="4-bit (high salience)", color=COLORS["bitnet"])

    ax.set_yticks(list(x))
    ax.set_yticklabels([n.split(".")[-2] + "." + n.split(".")[-1]
                        for n in layer_names], fontsize=7)
    ax.set_xlabel("Weight Distribution (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Bit allocation heatmap saved to {output_path}")


def plot_ablation_comparison(
    ablation_results: Dict,
    metric: str = "ppl_wikitext2",
    output_path: str = "results/ablation.png",
    title: str = "Ablation Study: Salience Metrics",
):
    """
    Bar plot comparing different salience metrics / configurations.

    ablation_results: {
        "metric_name": {"perplexity": float, "avg_bits": float},
        ...
    }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = list(ablation_results.keys())
    ppls = [ablation_results[n].get("perplexity", 0) for n in names]
    bits = [ablation_results[n].get("avg_bits", 0) for n in names]

    # Perplexity comparison
    ax = axes[0]
    bars = ax.bar(range(len(names)), ppls,
                  color=[COLORS.get(n, COLORS["ours"]) for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([METRIC_LABELS.get(n, n) for n in names],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Perplexity (↓ better)")
    ax.set_title("Perplexity by Salience Metric")

    # Add value labels on bars
    for bar, val in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # Avg bits comparison
    ax = axes[1]
    bars = ax.bar(range(len(names)), bits,
                  color=[COLORS.get(n, COLORS["ours"]) for n in names])
    ax.axhline(1.61, color="black", linestyle="--", linewidth=1.5,
               label="Target: 1.61 bits")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([METRIC_LABELS.get(n, n) for n in names],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Avg Bits per Weight")
    ax.set_title("Average Bit-width by Method")
    ax.legend()
    ax.set_ylim(0, 5)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Ablation plot saved to {output_path}")


def plot_baseline_comparison(
    results: Dict,
    output_path: str = "results/baseline_comparison.png",
    model_name: str = "GPT-2",
):
    """
    Main results figure: compare all methods on perplexity and bits.
    Methods: FP16, Uniform INT2, BitNet, GPTQ, Ours
    """
    methods = list(results.keys())
    ppls_w2 = [results[m].get("ppl_wikitext2", {}).get("perplexity", float("nan")) for m in methods]
    ppls_ptb = [results[m].get("ppl_ptb", {}).get("perplexity", float("nan")) for m in methods]
    avg_bits_list = [results[m].get("avg_bits", 0) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = range(len(methods))
    bar_colors = [COLORS.get(m, COLORS["ours"]) for m in methods]

    # WikiText-2 perplexity
    ax = axes[0]
    bars = ax.bar(x, ppls_w2, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Perplexity (↓ better)")
    ax.set_title(f"{model_name}: WikiText-2 Perplexity")
    for bar, val in zip(bars, ppls_w2):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # PTB perplexity
    ax = axes[1]
    bars = ax.bar(x, ppls_ptb, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Perplexity (↓ better)")
    ax.set_title(f"{model_name}: PTB Perplexity")

    # Avg bits
    ax = axes[2]
    bars = ax.bar(x, avg_bits_list, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax.axhline(1.61, color="black", linestyle="--", linewidth=1.5, label="Our target: 1.61b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Average Bits per Weight")
    ax.set_title("Compression: Avg Bits")
    ax.legend()
    for bar, val in zip(bars, avg_bits_list):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(f"Baseline Comparison — {model_name}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Baseline comparison plot saved to {output_path}")
