"""
robustness_eval.py
Evaluate D-FINE robustness under common image corruptions.

Uses the `imagecorruptions` library (based on ImageNet-C corruptions).
Generates corrupted versions of val images on-the-fly and measures AP
across 15 corruption types × 5 severity levels.

Corruptions evaluated:
    Noise: gaussian_noise, shot_noise, impulse_noise
    Blur:  defocus_blur, glass_blur, motion_blur, zoom_blur
    Weather: snow, frost, fog, brightness
    Digital: contrast, elastic_transform, pixelate, jpeg_compression

Usage:
    python eval/robustness_eval.py \
        --config configs/dfine/full_finetune/dfine_s_pascal_voc.yml \
        --checkpoint results/full_finetune/pascal_voc/best.pth \
        --dataset pascal_voc \
        --severity 1 2 3 4 5 \
        --output_dir results/robustness/full_finetune_pascal_voc
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from imagecorruptions import corrupt, get_corruption_names
    CORRUPTIONS = get_corruption_names("all")
except ImportError:
    print("WARNING: imagecorruptions not installed. Install with: pip install imagecorruptions")
    CORRUPTIONS = []

_here = Path(__file__).parent
for _candidate in [_here.parent / "D-FINE", _here.parent, Path(".")]:
    if (_candidate / "src" / "core").exists():
        sys.path.insert(0, str(_candidate))
        break

DATASET_ANN_MAP = {
    "pascal_voc": "../D_fine/data/pascal_voc/annotations/instances_val.json",
    "visdrone":   "../D_fine/data/visdrone/annotations/instances_val.json",
    "cityscapes": "../D_fine/data/cityscapes/annotations/instances_val.json",
}

DATASET_IMG_MAP = {
    "pascal_voc": "../D_fine/data/pascal_voc/images/val",
    "visdrone":   "../D_fine/data/visdrone/images/val",
    "cityscapes": "../D_fine/data/cityscapes/images/val",
}


def load_model_and_transform(config_path: str, checkpoint_path: str, input_size: int = 640):
    from src.core import YAMLConfig
    import torchvision.transforms.functional as TF

    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    model = cfg.model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    def transform(pil_img):
        img = pil_img.convert("RGB").resize((input_size, input_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)

    return model, transform


@torch.no_grad()
def infer_single(model, tensor, img_id: int, img_w: int, img_h: int, device: str) -> List[dict]:
    tensor = tensor.to(device)
    outputs = model(tensor)
    logits = outputs["pred_logits"][0]
    boxes  = outputs["pred_boxes"][0]

    scores_max, labels = torch.sigmoid(logits).max(-1)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    bw = w * img_w
    bh = h * img_h

    keep = scores_max > 0.01
    results = []
    for s, l, x, y, bw_, bh_ in zip(
        scores_max[keep].cpu(), labels[keep].cpu(),
        x1[keep].cpu(), y1[keep].cpu(),
        bw[keep].cpu(), bh[keep].cpu()
    ):
        results.append({
            "image_id":    img_id,
            "category_id": int(l) + 1,
            "bbox":        [float(x), float(y), float(bw_), float(bh_)],
            "score":       float(s),
        })
    return results


def evaluate_corruption(
    model, transform, coco_gt: COCO, img_dir: str,
    corruption_name: str, severity: int, device: str, input_size: int,
) -> float:
    """Return AP for one (corruption, severity) pair."""
    all_results = []
    img_ids = coco_gt.getImgIds()

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            continue

        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)

        # Apply corruption
        try:
            corrupted_np = corrupt(img_np, corruption_name=corruption_name, severity=severity)
        except Exception:
            corrupted_np = img_np

        corrupted_pil = Image.fromarray(corrupted_np)
        tensor = transform(corrupted_pil)
        results = infer_single(model, tensor, img_id,
                               img_info["width"], img_info["height"], device)
        all_results.extend(results)

    if not all_results:
        return 0.0

    coco_dt = coco_gt.loadRes(all_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


def compute_mpc(clean_ap: float, corrupted_aps: Dict[str, Dict[int, float]]) -> Dict[str, float]:
    """Mean Performance under Corruption (mPC) = mean AP across all corruptions × severities."""
    all_aps = [v for sev_dict in corrupted_aps.values() for v in sev_dict.values()]
    mpc = np.mean(all_aps) if all_aps else 0.0
    rpc = mpc / clean_ap if clean_ap > 0 else 0.0

    return {
        "clean_AP": clean_ap,
        "mPC":      float(mpc),
        "rPC":      float(rpc),  # relative performance under corruption
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--dataset",     required=True, choices=list(DATASET_ANN_MAP))
    parser.add_argument("--severity",    type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--corruptions", nargs="+", default=None,
                        help="Subset of corruptions to test; default=all 15")
    parser.add_argument("--input_size",  type=int, default=640)
    parser.add_argument("--output_dir",  default="results/robustness")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not CORRUPTIONS:
        print("imagecorruptions not available. Cannot run robustness evaluation.")
        return

    corruptions = args.corruptions or CORRUPTIONS
    ann_file = DATASET_ANN_MAP[args.dataset]
    img_dir  = DATASET_IMG_MAP[args.dataset]

    print(f"Loading model...")
    model, transform = load_model_and_transform(args.config, args.checkpoint, args.input_size)
    model = model.to(args.device)
    coco_gt = COCO(ann_file)

    # --- Clean AP baseline ---
    print("Evaluating on clean images...")
    all_results = []
    for img_id in tqdm(coco_gt.getImgIds()):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        tensor = transform(Image.open(img_path))
        all_results.extend(infer_single(model, tensor, img_id,
                                        img_info["width"], img_info["height"], args.device))

    if all_results:
        coco_dt = coco_gt.loadRes(all_results)
        ce = COCOeval(coco_gt, coco_dt, "bbox")
        ce.evaluate(); ce.accumulate(); ce.summarize()
        clean_ap = float(ce.stats[0])
    else:
        clean_ap = 0.0
    print(f"Clean AP: {clean_ap:.4f}")

    # --- Corruption evaluations ---
    corrupted_aps: Dict[str, Dict[int, float]] = {}
    total_runs = len(corruptions) * len(args.severity)
    pbar = tqdm(total=total_runs, desc="Robustness eval")

    for corr in corruptions:
        corrupted_aps[corr] = {}
        for sev in args.severity:
            ap = evaluate_corruption(model, transform, coco_gt, img_dir,
                                     corr, sev, args.device, args.input_size)
            corrupted_aps[corr][sev] = ap
            pbar.set_postfix(corruption=corr, severity=sev, AP=f"{ap:.3f}")
            pbar.update(1)
    pbar.close()

    summary = compute_mpc(clean_ap, corrupted_aps)
    summary["per_corruption"] = {
        c: {"severities": sev_dict,
            "mean_AP": float(np.mean(list(sev_dict.values())))}
        for c, sev_dict in corrupted_aps.items()
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "robustness_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Robustness Summary ({args.dataset}) ===")
    print(f"  Clean AP : {summary['clean_AP']:.4f}")
    print(f"  mPC      : {summary['mPC']:.4f}")
    print(f"  rPC      : {summary['rPC']:.4f}")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
