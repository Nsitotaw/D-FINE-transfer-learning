"""
model_scaling.py
Analyze D-FINE performance across model scales (N/S/M/L/X).
Compares AP and latency as model size grows for each fine-tuning strategy.

Usage:
    python analysis/model_scaling.py \
        --results_root results/ \
        --output_dir results/figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALES = ["n", "s", "m", "l", "x"]
SCALE_PARAMS = {"n": 4, "s": 10, "m": 19, "l": 31, "x": 62}  # approx params in M
STRATEGIES = ["full_finetune", "partial_finetune", "peft"]
STRATEGY_LABELS = {
    "full_finetune":    "Full Fine-tune",
    "partial_finetune": "Partial (Frozen BB)",
    "peft":             "LoRA (PEFT)",
}
COLORS = ["#2196F3", "#4CAF50", "#FF5722"]
DATASETS = ["pascal_voc", "visdrone", "cityscapes", "kitti"]


def load_scaling_data(results_root: str) -> dict:
    data = {}
    for strategy in STRATEGIES:
        data[strategy] = {"params": [], "ap": [], "fps": []}
        for scale in SCALES:
            # Convention: results/{strategy}/pascal_voc_scale-{scale}/eval_results.json
            # (users re-run experiments with different model sizes)
            ap_path = os.path.join(results_root, strategy,
                                   f"pascal_voc_scale_{scale}", "eval_results.json")
            lat_path = os.path.join(results_root, "latency",
                                    f"latency_{strategy}_pascal_voc_scale_{scale}.json")
            ap = np.nan
            fps = np.nan
            if os.path.exists(ap_path):
                with open(ap_path) as f:
                    ap = json.load(f).get("AP", np.nan)
            if os.path.exists(lat_path):
                with open(lat_path) as f:
                    fps = json.load(f)["latency"].get("fps", np.nan)

            data[strategy]["params"].append(SCALE_PARAMS[scale])
            data[strategy]["ap"].append(ap)
            data[strategy]["fps"].append(fps)
    return data


def plot_scaling(data: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in zip(
        axes,
        ["ap", "fps"],
        ["AP (%)", "FPS"],
        ["AP vs Model Size", "Speed vs Model Size"],
    ):
        for strategy, color in zip(STRATEGIES, COLORS):
            params = data[strategy]["params"]
            vals   = data[strategy][metric]
            valid  = [(p, v) for p, v in zip(params, vals) if not np.isnan(v)]
            if not valid:
                continue
            px, vx = zip(*valid)
            ax.plot(px, vx, "o-", color=color, label=STRATEGY_LABELS[strategy], linewidth=2)
            for p, v, s in zip(px, vx, SCALES):
                ax.annotate(f"  {s}", (p, v), fontsize=8)

        ax.set_xlabel("Params (M)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("D-FINE Model Scaling Analysis")
    fig.tight_layout()
    path = os.path.join(output_dir, "model_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/")
    parser.add_argument("--output_dir",   default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_scaling_data(args.results_root)
    plot_scaling(data, args.output_dir)


if __name__ == "__main__":
    main()
