"""
convert_visdrone_to_coco.py
Convert VisDrone2019-DET to COCO JSON.

Your actual structure:
  data/visdrone/
  ├── VisDrone2019-DET-train/VisDrone2019-DET-train/
  │   ├── images/       ← .jpg images
  │   └── annotations/  ← .txt per image
  └── VisDrone2019-DET-val/VisDrone2019-DET-val/
      ├── images/
      └── annotations/

VisDrone annotation format per line:
  bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion

Usage:
    cd "d:/EML/Final Project/D_fine"
    python data/convert_visdrone_to_coco.py --split train
    python data/convert_visdrone_to_coco.py --split val
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm

BASE = Path("data/visdrone")

SPLIT_MAP = {
    "train": BASE / "VisDrone2019-DET-train" / "VisDrone2019-DET-train",
    "val":   BASE / "VisDrone2019-DET-val"   / "VisDrone2019-DET-val",
}

VISDRONE_CLASSES = {
    1: "pedestrian", 2: "people",    3: "bicycle",  4: "car",
    5: "van",        6: "truck",     7: "tricycle",  8: "awning-tricycle",
    9: "bus",       10: "motor",
}
CAT_ID = {v: i for i, v in enumerate(VISDRONE_CLASSES.values())}


def convert(split: str, output_path: str):
    root    = SPLIT_MAP[split]
    img_dir = root / "images"
    ann_dir = root / "annotations"

    if not img_dir.exists():
        raise FileNotFoundError(f"Images not found: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_dir}")

    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    coco = {
        "info": {"description": f"VisDrone {split}", "date_created": datetime.now().isoformat()},
        "licenses": [], "images": [], "annotations": [],
        "categories": [{"id": v, "name": k, "supercategory": "object"} for k, v in CAT_ID.items()],
    }

    ann_id = 1
    for img_id, img_path in enumerate(tqdm(image_files, desc=f"VisDrone {split}"), start=1):
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            width, height = 1920, 1080

        coco["images"].append({
            "id": img_id, "file_name": img_path.name,
            "width": width, "height": height,
        })

        ann_file = ann_dir / img_path.with_suffix(".txt").name
        if not ann_file.exists():
            continue

        for line in ann_file.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h   = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            cat_raw       = int(parts[5])
            if cat_raw not in VISDRONE_CLASSES or w <= 0 or h <= 0:
                continue

            cat_name = VISDRONE_CLASSES[cat_raw]
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": CAT_ID[cat_name],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"Done: {len(coco['images'])} images, {len(coco['annotations'])} annotations → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = args.output or f"data/visdrone/annotations/instances_{args.split}.json"
    convert(args.split, output)
