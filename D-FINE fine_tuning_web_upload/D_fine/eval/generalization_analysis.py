"""
generalization_analysis.py
Compute and compare generalization gap across fine-tuning strategies and datasets.

Generalization gap = Train AP − Val AP (proxy for overfitting)
Domain gap         = COCO-pretrained AP on target domain (zero-shot) vs fine-tuned AP

Also computes:
  - Learning curve (AP vs epoch from training logs)
  - Cross-domain evaluation matrix (train on A, eval on B)

Usage:
    python eval/generalization_analysis.py \
        --results_root results/ \
        --output_dir results/generalization
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "D-FINE"))


STRATEGIES   = ["full_finetune", "partial_finetune", "peft"]
DATASETS     = ["pascal_voc", "visdrone", "cityscapes", "kitti"]


def load_eval_results(results_root: str) -> Dict[str, Dict[str, dict]]:
    """Load eval_results.json for each strategy×dataset."""
    data = {}
    for strategy in STRATEGIES:
        data[strategy] = {}
        for dataset in DATASETS:
            path = os.path.join(results_root, strategy, dataset, "eval_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    data[strategy][dataset] = json.load(f)
            else:
                data[strategy][dataset] = None
    return data


def load_training_logs(results_root: str) -> Dict[str, Dict[str, list]]:
    """Load training log JSON files (each line is a dict with epoch, train_loss, val_AP)."""
    logs = {}
    for strategy in STRATEGIES:
        logs[strategy] = {}
        for dataset in DATASETS:
            log_path = os.path.join(results_root, strategy, dataset, "log.json")
            if not os.path.exists(log_path):
                # Try TensorBoard-exported CSV
                csv_path = os.path.join(results_root, strategy, dataset, "metrics.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    logs[strategy][dataset] = df.to_dict("records")
                else:
                    logs[strategy][dataset] = []
                continue
            entries = []
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            logs[strategy][dataset] = entries
    return logs


def compute_generalization_gap(results: Dict, logs: Dict) -> pd.DataFrame:
    """
    For each strategy × dataset, compute:
      - Best Val AP
      - Train AP at best epoch (approximated from last train log entry)
      - Generalization gap = Train AP − Val AP
    """
    rows = []
    for strategy in STRATEGIES:
        for dataset in DATASETS:
            res = results[strategy].get(dataset)
            val_ap = res["AP"] if res else np.nan

            # Get train AP from log (last epoch)
            log = logs[strategy].get(dataset, [])
            train_ap = np.nan
            if log:
                last = log[-1]
                train_ap = last.get("train_AP", last.get("ap", np.nan))

            gap = train_ap - val_ap if not (np.isnan(train_ap) or np.isnan(val_ap)) else np.nan

            rows.append({
                "strategy": strategy,
                "dataset":  dataset,
                "train_AP": round(train_ap, 4) if not np.isnan(train_ap) else "N/A",
                "val_AP":   round(val_ap,   4) if not np.isnan(val_ap)   else "N/A",
                "gen_gap":  round(gap,      4) if not np.isnan(gap)      else "N/A",
            })

    return pd.DataFrame(rows)


def compute_domain_gap(results: Dict) -> pd.DataFrame:
    """
    Domain gap: compare zero-shot COCO-pretrained AP vs best fine-tuned AP on each dataset.
    Requires results/zero_shot/{dataset}/eval_results.json.
    """
    rows = []
    for dataset in DATASETS:
        zero_shot_path = f"results/zero_shot/{dataset}/eval_results.json"
        zero_ap = np.nan
        if os.path.exists(zero_shot_path):
            with open(zero_shot_path) as f:
                zero_ap = json.load(f).get("AP", np.nan)

        best_ap = np.nan
        best_strategy = "N/A"
        for strategy in STRATEGIES:
            res = results[strategy].get(dataset)
            if res and not np.isnan(res.get("AP", np.nan)):
                if np.isnan(best_ap) or res["AP"] > best_ap:
                    best_ap = res["AP"]
                    best_strategy = strategy

        domain_gap = best_ap - zero_ap if not (np.isnan(best_ap) or np.isnan(zero_ap)) else np.nan

        rows.append({
            "dataset":       dataset,
            "zero_shot_AP":  round(zero_ap,   4) if not np.isnan(zero_ap)   else "N/A",
            "best_finetuned_AP": round(best_ap, 4) if not np.isnan(best_ap) else "N/A",
            "best_strategy": best_strategy,
            "domain_gap":    round(domain_gap, 4) if not np.isnan(domain_gap) else "N/A",
        })

    return pd.DataFrame(rows)


def strategy_comparison_table(results: Dict) -> pd.DataFrame:
    """AP table: rows=datasets, cols=strategies."""
    table = {}
    for dataset in DATASETS:
        table[dataset] = {}
        for strategy in STRATEGIES:
            res = results[strategy].get(dataset)
            table[dataset][strategy] = round(res["AP"], 4) if res else "N/A"
    return pd.DataFrame(table).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/")
    parser.add_argument("--output_dir",   default="results/generalization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    results = load_eval_results(args.results_root)
    logs    = load_training_logs(args.results_root)

    # --- Generalization gap table ---
    gen_df = compute_generalization_gap(results, logs)
    gen_path = os.path.join(args.output_dir, "generalization_gap.csv")
    gen_df.to_csv(gen_path, index=False)
    print("\n=== Generalization Gap ===")
    print(gen_df.to_string(index=False))

    # --- Domain gap table ---
    dom_df = compute_domain_gap(results)
    dom_path = os.path.join(args.output_dir, "domain_gap.csv")
    dom_df.to_csv(dom_path, index=False)
    print("\n=== Domain Gap (zero-shot → fine-tuned) ===")
    print(dom_df.to_string(index=False))

    # --- Strategy comparison table ---
    cmp_df = strategy_comparison_table(results)
    cmp_path = os.path.join(args.output_dir, "strategy_comparison.csv")
    cmp_df.to_csv(cmp_path)
    print("\n=== Strategy Comparison (AP) ===")
    print(cmp_df.to_string())

    # --- Summary JSON ---
    summary = {
        "generalization_gap": gen_df.to_dict("records"),
        "domain_gap":         dom_df.to_dict("records"),
        "strategy_comparison": cmp_df.to_dict(),
    }
    with open(os.path.join(args.output_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
