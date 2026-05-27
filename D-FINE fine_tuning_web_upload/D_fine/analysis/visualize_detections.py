"""
visualize_detections.py
Generate Figure 6: qualitative bounding-box results for the paper.

Run from D_fine/ AFTER all training runs are done:
    python analysis/visualize_detections.py

Output: paper/fig6_detections.png
        (3 columns x 2 rows: Full FT | Partial FT | LoRA, top=VOC bottom=VisDrone)

Requirements: torch, torchvision, Pillow, matplotlib, numpy
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_ROOT = Path("results")
DATA_ROOT    = Path("data")
DFINE_ROOT   = Path("../D-FINE")
OUT_PATH     = Path("paper/fig6_detections.png")

# checkpoint paths (best_stg2 preferred, fallback to best_stg1)
CHECKPOINTS = {
    "Full FT":    RESULTS_ROOT / "full_finetune/voc_S",
    "Partial FT": RESULTS_ROOT / "partial_finetune/voc_S",
    "LoRA":       RESULTS_ROOT / "peft/voc_S",
}
CHECKPOINTS_VD = {
    "Full FT":    RESULTS_ROOT / "full_finetune/visdrone_S",
    "Partial FT": RESULTS_ROOT / "partial_finetune/visdrone_S",
    "LoRA":       RESULTS_ROOT / "peft/visdrone_S",
}

VOC_CLASSES      = ["person", "car", "bicycle", "bus", "motorbike"]
VISDRONE_CLASSES = ["pedestrian", "car", "van", "truck", "bus"]

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]

CONF_THRESH = 0.4
NUM_SAMPLES = 2   # images per dataset row
SEED = 7

# ── Add D-FINE to path ────────────────────────────────────────────────────────
for candidate in [DFINE_ROOT, Path("../D-FINE"), Path("D-FINE")]:
    if (candidate / "src" / "core").exists():
        sys.path.insert(0, str(candidate.resolve()))
        break
else:
    print("ERROR: Could not find D-FINE/src/core. Check DFINE_ROOT path.")
    sys.exit(1)

from src.core import YAMLConfig


def best_ckpt(folder: Path) -> Path:
    for name in ["best_stg2.pth", "best_stg1.pth", "last.pth"]:
        p = folder / name
        if p.exists():
            return p
    return None


def load_model(cfg_path: Path, ckpt_path: Path, device):
    cfg = YAMLConfig(str(cfg_path), resume=str(ckpt_path))
    model = cfg.model
    model.load_state_dict(cfg.state_dict["model"])
    model.to(device).eval()
    postprocessor = cfg.postprocessor
    return model, postprocessor


def infer(model, postprocessor, img_pil, device, input_size=640):
    import torchvision.transforms.functional as TF
    orig_w, orig_h = img_pil.size
    img_t = TF.to_tensor(img_pil.resize((input_size, input_size))).unsqueeze(0).to(device)
    orig_sizes = torch.tensor([[orig_h, orig_w]], device=device)
    with torch.no_grad():
        outputs = model(img_t)
        results = postprocessor(outputs, orig_sizes)
    boxes  = results[0]["boxes"].cpu().numpy()   # xyxy
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0]["labels"].cpu().numpy()
    keep   = scores >= CONF_THRESH
    return boxes[keep], scores[keep], labels[keep]


def draw_boxes(ax, img_pil, pred_boxes, pred_scores, pred_labels,
               gt_boxes, gt_labels, class_names, title=""):
    ax.imshow(img_pil)
    ax.set_title(title, fontsize=7, pad=2)
    ax.axis("off")
    # Ground truth — green outline
    for box, lbl in zip(gt_boxes, gt_labels):
        x1, y1, w, h = box
        rect = patches.Rectangle((x1, y1), w, h,
                                  linewidth=1, edgecolor="#00cc00",
                                  facecolor="none", linestyle="--")
        ax.add_patch(rect)
    # Predictions — colored
    for box, score, lbl in zip(pred_boxes, pred_scores, pred_labels):
        if int(lbl) >= len(class_names):
            continue
        x1, y1, x2, y2 = box
        color = COLORS[int(lbl) % len(COLORS)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor=color,
                                  facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 2, f"{class_names[int(lbl)]} {score:.2f}",
                fontsize=5, color="white",
                bbox=dict(boxstyle="square,pad=0.1", fc=color, ec="none", alpha=0.8))


def pick_sample_images(ann_file: Path, img_root: Path, n=2, seed=SEED):
    """Return n random (image_path, gt_boxes_xywh, gt_labels) tuples."""
    with open(ann_file) as f:
        coco = json.load(f)
    random.seed(seed)
    imgs = random.sample(coco["images"], min(n * 5, len(coco["images"])))
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)
    results = []
    for img_info in imgs:
        anns = ann_by_img.get(img_info["id"], [])
        if not anns:
            continue
        img_path = img_root / img_info["file_name"]
        if not img_path.exists():
            continue
        boxes  = [a["bbox"] for a in anns]
        labels = [a["category_id"] for a in anns]
        results.append((img_path, boxes, labels))
        if len(results) >= n:
            break
    return results


def make_figure(strategies, checkpoints_dict, cfg_template, ann_file, img_root,
                class_names, device, row_label):
    """
    Returns list of (img_pil, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    for each strategy x sample combination.
    """
    samples = pick_sample_images(ann_file, img_root, n=NUM_SAMPLES)
    row_data = []
    for strat, ckpt_folder in checkpoints_dict.items():
        ckpt = best_ckpt(Path(ckpt_folder))
        if ckpt is None:
            # placeholder: return blank images
            for img_path, gt_boxes, gt_labels in samples:
                row_data.append((Image.open(img_path).convert("RGB"),
                                 [], [], [], gt_boxes, gt_labels, strat))
            continue
        try:
            model, postprocessor = load_model(cfg_template, ckpt, device)
        except Exception as e:
            print(f"  [WARN] Could not load {strat}: {e}")
            for img_path, gt_boxes, gt_labels in samples:
                row_data.append((Image.open(img_path).convert("RGB"),
                                 [], [], [], gt_boxes, gt_labels, strat))
            continue
        for img_path, gt_boxes, gt_labels in samples:
            img = Image.open(img_path).convert("RGB")
            pred_b, pred_s, pred_l = infer(model, postprocessor, img, device)
            row_data.append((img, pred_b, pred_s, pred_l, gt_boxes, gt_labels, strat))
    return row_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path("paper").mkdir(exist_ok=True)

    voc_ann   = DATA_ROOT / "pascal_voc/annotations/instances_val.json"
    voc_imgs  = DATA_ROOT / "pascal_voc/images/val"
    vd_ann    = DATA_ROOT / "visdrone/annotations/instances_val.json"
    vd_imgs   = DATA_ROOT / "visdrone/images/val"

    cfg_voc = DFINE_ROOT / "configs/dfine/custom/dfine_s_voc_full.yml"
    cfg_vd  = DFINE_ROOT / "configs/dfine/custom/dfine_s_visdrone_full.yml"

    strategies = list(CHECKPOINTS.keys())
    n_cols = len(strategies) * NUM_SAMPLES

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.2, 5.0),
                             gridspec_kw={"hspace": 0.3, "wspace": 0.05})

    col = 0
    # ── Row 0: VOC ────────────────────────────────────────────────────────────
    samples_voc = pick_sample_images(voc_ann, voc_imgs, n=NUM_SAMPLES)
    for si, strat in enumerate(strategies):
        ckpt_folder = CHECKPOINTS[strat]
        ckpt = best_ckpt(Path(ckpt_folder))
        model = postprocessor = None
        if ckpt:
            try:
                model, postprocessor = load_model(cfg_voc, ckpt, device)
            except Exception as e:
                print(f"  [WARN] VOC {strat}: {e}")

        for img_path, gt_boxes, gt_labels in samples_voc:
            img = Image.open(img_path).convert("RGB")
            if model:
                pred_b, pred_s, pred_l = infer(model, postprocessor, img, device)
            else:
                pred_b, pred_s, pred_l = [], [], []
            title = strat if img_path == samples_voc[0][0] else ""
            ax = axes[0][col] if n_cols > 1 else axes[0]
            draw_boxes(ax, img, pred_b, pred_s, pred_l,
                       gt_boxes, gt_labels, VOC_CLASSES, title=strat)
            col += 1

    col = 0
    # ── Row 1: VisDrone ───────────────────────────────────────────────────────
    samples_vd = pick_sample_images(vd_ann, vd_imgs, n=NUM_SAMPLES)
    for si, strat in enumerate(strategies):
        ckpt_folder = CHECKPOINTS_VD[strat]
        ckpt = best_ckpt(Path(ckpt_folder))
        model = postprocessor = None
        if ckpt:
            try:
                model, postprocessor = load_model(cfg_vd, ckpt, device)
            except Exception as e:
                print(f"  [WARN] VisDrone {strat}: {e}")

        for img_path, gt_boxes, gt_labels in samples_vd:
            img = Image.open(img_path).convert("RGB")
            if model:
                pred_b, pred_s, pred_l = infer(model, postprocessor, img, device)
            else:
                pred_b, pred_s, pred_l = [], [], []
            ax = axes[1][col] if n_cols > 1 else axes[1]
            draw_boxes(ax, img, pred_b, pred_s, pred_l,
                       gt_boxes, gt_labels, VISDRONE_CLASSES, title="")
            col += 1

    # row labels
    axes[0][0].set_ylabel("VOC", fontsize=8, labelpad=4)
    axes[1][0].set_ylabel("VisDrone", fontsize=8, labelpad=4)

    # legend
    legend_elems = (
        [patches.Patch(fc=COLORS[i], label=c) for i, c in enumerate(VOC_CLASSES)] +
        [patches.Patch(fc="#00cc00", label="Ground Truth")]
    )
    fig.legend(handles=legend_elems, loc="lower center", ncol=len(legend_elems),
               fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
