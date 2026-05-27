"""
evaluate.py
Full COCO-API evaluation pipeline for D-FINE on any dataset.

Computes:
  - AP, AP50, AP75
  - APs (small), APm (medium), APl (large)
  - Per-category AP

Usage (run from D-FINE/ directory):
    python ../D_fine/eval/evaluate.py \
        --config configs/dfine/custom/dfine_s_voc_full.yml \
        --checkpoint results/full_finetune/voc_S/best.pth \
        --dataset pascal_voc \
        --output_dir ../D_fine/results/full_finetune/voc_S
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Add D-FINE to path (script may be called from D-FINE/ or D_fine/)
_here = Path(__file__).parent
for candidate in [_here.parent / "D-FINE", _here.parent, Path(".")]:
    if (candidate / "src" / "core").exists():
        sys.path.insert(0, str(candidate))
        break

DATASET_ANN_MAP = {
    "pascal_voc":  "../D_fine/data/pascal_voc/annotations/instances_val.json",
    "visdrone":    "../D_fine/data/visdrone/annotations/instances_val.json",
    "cityscapes":  "../D_fine/data/cityscapes/annotations/instances_val.json",
}

DATASET_IMG_MAP = {
    "pascal_voc":  "../D_fine/data/pascal_voc/images/val",
    "visdrone":    "../D_fine/data/visdrone/images/val",
    "cityscapes":  "../D_fine/data/cityscapes/images/val",
}


def load_model(config_path: str, checkpoint_path: str):
    from src.core import YAMLConfig
    cfg = YAMLConfig(config_path, tuning=checkpoint_path)
    model = cfg.model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def run_inference(model, dataloader, device) -> list:
    results = []
    model = model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images, targets = batch
            images = images.to(device)

            outputs = model(images)
            pred_logits = outputs["pred_logits"]   # [B, Q, C]
            pred_boxes  = outputs["pred_boxes"]    # [B, Q, 4] cxcywh normalized

            for i, (logits, boxes) in enumerate(zip(pred_logits, pred_boxes)):
                scores = torch.sigmoid(logits)
                scores_max, labels = scores.max(-1)

                img_h = targets[i]["orig_size"][0]
                img_w = targets[i]["orig_size"][1]
                cx, cy, w, h = boxes.unbind(-1)
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                bw = w * img_w
                bh = h * img_h

                img_id = int(targets[i]["image_id"])
                keep   = scores_max > 0.01

                for score, label, x, y, bw_, bh_ in zip(
                    scores_max[keep].cpu(), labels[keep].cpu(),
                    x1[keep].cpu(), y1[keep].cpu(),
                    bw[keep].cpu(), bh[keep].cpu(),
                ):
                    results.append({
                        "image_id":    img_id,
                        "category_id": int(label) + 1,
                        "bbox":        [float(x), float(y), float(bw_), float(bh_)],
                        "score":       float(score),
                    })

    return results


def evaluate_coco(ann_file: str, results: list, output_dir: str) -> dict:
    coco_gt = COCO(ann_file)

    if not results:
        print("WARNING: No detections produced.")
        return {}

    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP":    float(coco_eval.stats[0]),
        "AP50":  float(coco_eval.stats[1]),
        "AP75":  float(coco_eval.stats[2]),
        "APs":   float(coco_eval.stats[3]),
        "APm":   float(coco_eval.stats[4]),
        "APl":   float(coco_eval.stats[5]),
        "AR1":   float(coco_eval.stats[6]),
        "AR10":  float(coco_eval.stats[7]),
        "AR100": float(coco_eval.stats[8]),
    }

    cat_ids  = coco_gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    per_cat_ap = {}
    for cat_id in cat_ids:
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [cat_id]
        ev.evaluate(); ev.accumulate()
        per_cat_ap[cat_names[cat_id]] = float(ev.stats[0])

    metrics["per_category_AP"] = per_cat_ap

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved → {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True,
                        help="D-FINE YAML config (relative to working dir)")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--dataset",     required=True, choices=list(DATASET_ANN_MAP))
    parser.add_argument("--output_dir",  default="results/eval")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size",  type=int, default=4)
    args = parser.parse_args()

    device   = torch.device(args.device)
    ann_file = DATASET_ANN_MAP[args.dataset]

    print(f"Loading model from {args.checkpoint} ...")
    model, cfg = load_model(args.config, args.checkpoint)

    # Build val dataloader from config
    from src.data import build_dataloader
    val_loader = build_dataloader(cfg.val_dataloader, batch_size=args.batch_size)

    print("Running inference ...")
    t0 = time.time()
    results = run_inference(model, val_loader, device)
    print(f"Done in {time.time()-t0:.1f}s — {len(results)} detections")

    print("Running COCO evaluation ...")
    metrics = evaluate_coco(ann_file, results, args.output_dir)

    print("\n=== Evaluation Summary ===")
    for k, v in metrics.items():
        if k != "per_category_AP":
            print(f"  {k:6s}: {v:.4f}")


if __name__ == "__main__":
    main()
