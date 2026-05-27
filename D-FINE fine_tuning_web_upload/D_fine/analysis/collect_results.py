"""
collect_results.py
Collect all D-FINE training logs and save to Excel for analysis and plotting.

Run from D_fine/ after any/all training runs complete:
    python analysis/collect_results.py

Output: results/all_results.xlsx
Sheets:
    Summary          -- best mAP per experiment
    Training_Curves  -- all metrics per epoch (losses + COCO + detection)
    Loss_Components  -- every train_loss_* key per epoch (for loss ablation plots)
    COCO_Metrics     -- AP/AR family per epoch
    Detection_Metrics-- f1/precision/recall/iou/TPs/FPs/FNs per epoch
    AP_Breakdown     -- AP/AP50/AP75/APs/APm/APl at best epoch
    Generalization_Gap-- placeholder for zero-shot vs fine-tuned comparison
"""

import json
import os
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("results")

EXPERIMENT_MAP = {
    "full_finetune/voc_S":         "Full FT | S | VOC",
    "full_finetune/visdrone_S":    "Full FT | S | VisDrone",
    "partial_finetune/voc_S":      "Partial FT | S | VOC",
    "partial_finetune/visdrone_S": "Partial FT | S | VisDrone",
    "peft/voc_S":                  "LoRA | S | VOC",
    "peft/visdrone_S":             "LoRA-r8 | S | VisDrone",
    "peft/visdrone_S_lora_r32":    "LoRA-r32 | S | VisDrone",
    "peft/visdrone_N_lora":        "LoRA | N | VisDrone",
    "full_finetune/voc_N":         "Full FT | N | VOC",
    "full_finetune/visdrone_N":    "Full FT | N | VisDrone",
}

COCO_METRIC_NAMES = [
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
]

VAL_METRIC_NAMES = ["val_f1", "val_precision", "val_recall", "val_iou",
                    "val_TPs", "val_FPs", "val_FNs"]

TRAINABLE_PARAMS = {
    ("Full FT", "S"): 10.18,
    ("Full FT", "N"): 4.00,
    ("Partial FT", "S"): 6.10,
    ("Partial FT", "N"): 2.40,
    ("LoRA", "S"): 0.80,
    ("LoRA", "N"): 0.32,
    ("LoRA-r8", "S"): 0.80,
    ("LoRA-r32", "S"): 10.47,
}


def parse_log(log_path: Path):
    """Return (rows, all_loss_keys) where rows is a list of per-epoch dicts."""
    rows = []
    all_loss_keys = set()

    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        row = {
            "epoch": d.get("epoch", -1),
            "lr":    d.get("train_lr", None),
            "n_parameters": d.get("n_parameters", None),
        }

        # --- capture every train_loss_* key dynamically ---
        for k, v in d.items():
            if k.startswith("train_loss"):
                row[k] = v
                all_loss_keys.add(k)

        # --- COCO eval bbox (12 standard values) ---
        bbox = d.get("test_coco_eval_bbox", [])
        for i, name in enumerate(COCO_METRIC_NAMES):
            row[name] = round(bbox[i] * 100, 4) if i < len(bbox) and bbox[i] is not None else None

        # --- Validator metrics (written by det_engine.py after the engine patch) ---
        for m in VAL_METRIC_NAMES:
            key_in_log = f"test_{m}"   # det_solver prepends "test_" to every test_stats key
            row[m] = d.get(key_in_log, None)

        rows.append(row)

    return rows, sorted(all_loss_keys)


def collect_all():
    summary_rows    = []
    curve_rows      = []
    all_loss_keys_union = []

    for rel_path, exp_name in EXPERIMENT_MAP.items():
        log_path = RESULTS_ROOT / rel_path / "log.txt"
        if not log_path.exists():
            print(f"  [MISSING] {rel_path}/log.txt - skipping")
            continue

        print(f"  [OK] {exp_name}")
        epochs, loss_keys = parse_log(log_path)
        if not epochs:
            continue

        for k in loss_keys:
            if k not in all_loss_keys_union:
                all_loss_keys_union.append(k)

        parts    = [p.strip() for p in exp_name.split("|")]
        strategy = parts[0]
        model    = parts[1]
        dataset  = parts[2]

        for ep in epochs:
            curve_rows.append({
                "Experiment": exp_name,
                "Strategy":   strategy,
                "Model":      model,
                "Dataset":    dataset,
                **ep,
            })

        best = max(epochs, key=lambda x: x.get("AP") or 0)
        summary_rows.append({
            "Experiment":  exp_name,
            "Strategy":    strategy,
            "Model":       model,
            "Dataset":     dataset,
            "Best_Epoch":  best["epoch"],
            "Best_AP":     best.get("AP"),
            "AP50":        best.get("AP50"),
            "AP75":        best.get("AP75"),
            "APs":         best.get("APs"),
            "APm":         best.get("APm"),
            "APl":         best.get("APl"),
            "AR100":       best.get("AR100"),
            "Final_Loss":  round(epochs[-1]["train_loss"], 4) if epochs[-1].get("train_loss") else None,
            "Params_M":    TRAINABLE_PARAMS.get((strategy, model), round((best.get("n_parameters") or epochs[-1].get("n_parameters") or 0) / 1_000_000, 3)),
            # detection metrics at best epoch (may be None for runs before engine patch)
            "Best_F1":        best.get("val_f1"),
            "Best_Precision": best.get("val_precision"),
            "Best_Recall":    best.get("val_recall"),
        })

    # ---- DataFrames ----
    df_summary = pd.DataFrame(summary_rows)
    df_curves  = pd.DataFrame(curve_rows)

    # Loss_Components: only loss columns
    loss_cols = ["Experiment", "Strategy", "Model", "Dataset", "epoch"] + all_loss_keys_union
    df_loss = df_curves[[c for c in loss_cols if c in df_curves.columns]].copy() if not df_curves.empty else pd.DataFrame()

    # COCO_Metrics: AP/AR columns
    coco_cols = ["Experiment", "Strategy", "Model", "Dataset", "epoch"] + COCO_METRIC_NAMES
    df_coco = df_curves[[c for c in coco_cols if c in df_curves.columns]].copy() if not df_curves.empty else pd.DataFrame()

    # Detection_Metrics: f1/precision/recall etc.
    det_cols = ["Experiment", "Strategy", "Model", "Dataset", "epoch"] + VAL_METRIC_NAMES
    df_det = df_curves[[c for c in det_cols if c in df_curves.columns]].copy() if not df_curves.empty else pd.DataFrame()

    # AP_Breakdown at best epoch
    if not df_summary.empty:
        df_ap = df_summary[["Experiment", "Strategy", "Model", "Dataset",
                             "Best_AP", "AP50", "AP75", "APs", "APm", "APl"]].copy()
    else:
        df_ap = pd.DataFrame()

    # Generalization_Gap placeholder
    gap_rows = []
    if not df_summary.empty:
        for _, row in df_summary.iterrows():
            gap_rows.append({
                "Experiment":         row["Experiment"],
                "Strategy":           row["Strategy"],
                "Model":              row["Model"],
                "Dataset":            row["Dataset"],
                "Finetuned_AP":       row["Best_AP"],
                "ZeroShot_AP":        None,
                "Generalization_Gap": None,
                "Note": "Zero-shot AP to be added after running evaluate.py with COCO weights",
            })
    df_gap = pd.DataFrame(gap_rows)

    # ---- Write Excel ----
    out_path = RESULTS_ROOT / "all_results.xlsx"
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        def _write(df, name):
            if not df.empty:
                df.to_excel(writer, sheet_name=name, index=False)
                try:
                    ws = writer.sheets[name]
                    for col in ws.columns:
                        max_len = max(len(str(cell.value or "")) for cell in col) + 3
                        ws.column_dimensions[col[0].column_letter].width = min(max_len, 40)
                except Exception:
                    pass

        _write(df_summary,  "Summary")
        _write(df_curves,   "Training_Curves")
        _write(df_loss,     "Loss_Components")
        _write(df_coco,     "COCO_Metrics")
        _write(df_det,      "Detection_Metrics")
        _write(df_ap,       "AP_Breakdown")
        _write(df_gap,      "Generalization_Gap")

    print(f"\nSaved -> {out_path}")
    print(f"  Experiments collected : {len(summary_rows)}")
    print(f"  Loss keys captured    : {len(all_loss_keys_union)}")
    if not df_summary.empty:
        print("\n=== Summary ===")
        cols = [c for c in ["Experiment", "Best_Epoch", "Best_AP", "AP50", "Best_F1"] if c in df_summary.columns]
        print(df_summary[cols].to_string(index=False))


if __name__ == "__main__":
    print("Collecting results from:", RESULTS_ROOT.resolve())
    collect_all()
