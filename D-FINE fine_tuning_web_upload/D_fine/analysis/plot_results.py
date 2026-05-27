"""
plot_results.py
Generate all figures for the final report:

  1. Bar chart: AP per strategy × dataset
  2. Radar chart: Robustness profile (mPC per strategy per dataset)
  3. Line plot: Learning curves (AP vs epoch)
  4. Heatmap: Strategy × dataset AP matrix
  5. Latency vs AP scatter (efficiency frontier)
  6. Generalization gap bar chart
  7. Per-corruption robustness comparison

Usage:
    python analysis/plot_results.py \
        --results_root results/ \
        --output_dir results/figures
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

STRATEGIES = ["full_finetune", "partial_finetune", "peft"]
STRATEGY_LABELS = {
    "full_finetune":     "Full Fine-tune",
    "partial_finetune":  "Partial (Frozen BB)",
    "peft":              "LoRA (PEFT)",
}
DATASETS = ["pascal_voc", "visdrone", "cityscapes", "kitti"]
COLORS   = ["#2196F3", "#4CAF50", "#FF5722"]


def load_ap_matrix(results_root: str) -> pd.DataFrame:
    rows = {}
    for strategy in STRATEGIES:
        rows[STRATEGY_LABELS[strategy]] = {}
        for ds in DATASETS:
            path = os.path.join(results_root, strategy, ds, "eval_results.json")
            ap = np.nan
            if os.path.exists(path):
                with open(path) as f:
                    ap = json.load(f).get("AP", np.nan)
            rows[STRATEGY_LABELS[strategy]][ds] = ap
    return pd.DataFrame(rows).T


def load_robustness(results_root: str) -> Dict[str, Dict[str, float]]:
    data = {}
    for strategy in STRATEGIES:
        data[strategy] = {}
        for ds in DATASETS:
            path = os.path.join(results_root, "robustness", f"{strategy}_{ds}", "robustness_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                data[strategy][ds] = {"clean": r.get("clean_AP", 0), "mPC": r.get("mPC", 0)}
    return data


def load_latency(results_root: str) -> Dict[str, Dict[str, float]]:
    data = {}
    for strategy in STRATEGIES:
        data[strategy] = {}
        for ds in DATASETS:
            path = os.path.join(results_root, "latency", f"latency_{strategy}_{ds}.json")
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                data[strategy][ds] = {
                    "fps":     d["latency"]["fps"],
                    "mean_ms": d["latency"]["mean_ms"],
                }
    return data


def plot_ap_bars(ap_df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(DATASETS))
    width = 0.25
    for i, (strategy, color) in enumerate(zip(ap_df.index, COLORS)):
        vals = [ap_df.loc[strategy, ds] if ds in ap_df.columns else 0 for ds in DATASETS]
        bars = ax.bar(x + i * width, vals, width, label=strategy, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("AP (%)")
    ax.set_title("D-FINE: Detection AP by Strategy and Dataset")
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace("_", "\n") for d in DATASETS])
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "ap_bar_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ap_heatmap(ap_df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    mask = ap_df.isna()
    sns.heatmap(ap_df.astype(float), annot=True, fmt=".1f", cmap="YlOrRd",
                mask=mask, ax=ax, linewidths=0.5, vmin=0, vmax=70,
                cbar_kws={"label": "AP (%)"})
    ax.set_title("AP Heatmap: Strategy × Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Strategy")
    fig.tight_layout()
    path = os.path.join(output_dir, "ap_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_robustness_bars(robustness: Dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes,
        ["clean", "mPC"],
        ["Clean AP", "Mean Performance under Corruption (mPC)"]
    ):
        x = np.arange(len(DATASETS))
        width = 0.25
        for i, (strategy, color) in enumerate(zip(STRATEGIES, COLORS)):
            vals = [robustness.get(strategy, {}).get(ds, {}).get(metric, 0)
                    for ds in DATASETS]
            ax.bar(x + i*width, vals, width, label=STRATEGY_LABELS[strategy],
                   color=color, alpha=0.85)

        ax.set_title(title)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("AP (%)")
        ax.set_xticks(x + width)
        ax.set_xticklabels([d.replace("_", "\n") for d in DATASETS])
        ax.legend(fontsize=8)
        ax.set_ylim(0, 70)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Robustness Evaluation")
    fig.tight_layout()
    path = os.path.join(output_dir, "robustness_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_latency_vs_ap(ap_df: pd.DataFrame, latency: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "^"]

    for strategy, marker, color in zip(STRATEGIES, markers, COLORS):
        for ds in DATASETS:
            ap_val = ap_df.loc[STRATEGY_LABELS[strategy], ds] \
                     if STRATEGY_LABELS[strategy] in ap_df.index else np.nan
            fps_val = latency.get(strategy, {}).get(ds, {}).get("fps", np.nan)
            if np.isnan(ap_val) or np.isnan(fps_val):
                continue
            ax.scatter(fps_val, ap_val, marker=marker, color=color, s=80,
                       label=f"{STRATEGY_LABELS[strategy]} ({ds})" if ds == DATASETS[0] else None)
            ax.annotate(ds.replace("_", "\n"),
                        (fps_val, ap_val), fontsize=7, ha="left", va="bottom")

    ax.set_xlabel("Throughput (FPS)")
    ax.set_ylabel("AP (%)")
    ax.set_title("Efficiency Frontier: AP vs Latency")
    handles = [plt.Line2D([0],[0], marker=m, color="w", markerfacecolor=c,
                          markersize=8, label=STRATEGY_LABELS[s])
               for s, m, c in zip(STRATEGIES, markers, COLORS)]
    ax.legend(handles=handles)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "latency_vs_ap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_domain_gap(results_root: str, output_dir: str):
    rows = []
    for ds in DATASETS:
        zs_path = os.path.join(results_root, "zero_shot", ds, "eval_results.json")
        zs_ap = 0
        if os.path.exists(zs_path):
            with open(zs_path) as f:
                zs_ap = json.load(f).get("AP", 0)

        best_ap = 0
        for strategy in STRATEGIES:
            path = os.path.join(results_root, strategy, ds, "eval_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    ap = json.load(f).get("AP", 0)
                best_ap = max(best_ap, ap)

        rows.append({"dataset": ds, "zero_shot": zs_ap, "best_finetuned": best_ap})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(DATASETS))
    ax.bar(x - 0.2, df["zero_shot"],     0.4, label="Zero-shot (COCO pretrained)", color="#90CAF9")
    ax.bar(x + 0.2, df["best_finetuned"],0.4, label="Best Fine-tuned",              color="#1565C0")

    for i, row in df.iterrows():
        gap = row["best_finetuned"] - row["zero_shot"]
        ax.annotate(f"+{gap:.1f}%", (x[i]+0.2, row["best_finetuned"]+0.5),
                    ha="center", fontsize=8, color="#1565C0")

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in DATASETS])
    ax.set_ylabel("AP (%)")
    ax.set_title("Domain Gap: Zero-shot vs Fine-tuned")
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "domain_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/")
    parser.add_argument("--output_dir",   default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ap_df     = load_ap_matrix(args.results_root)
    robustness = load_robustness(args.results_root)
    latency   = load_latency(args.results_root)

    plot_ap_bars(ap_df, args.output_dir)
    plot_ap_heatmap(ap_df, args.output_dir)
    plot_robustness_bars(robustness, args.output_dir)
    plot_latency_vs_ap(ap_df, latency, args.output_dir)
    plot_domain_gap(args.results_root, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
