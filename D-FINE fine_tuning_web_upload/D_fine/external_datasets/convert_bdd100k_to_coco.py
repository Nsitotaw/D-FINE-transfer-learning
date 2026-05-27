"""
Convert BDD100K detection labels to project 5-class COCO format.

Supported input formats:
  labels/det_20/det_val.json
  labels/bdd100k_labels_images_val.json

Output classes:
  0 person
  1 car
  2 bicycle
  3 bus
  4 motorbike
"""

import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
BDD_ROOT = ROOT / "data" / "bdd100k"

CLASS_MAP = {
    "person": 0,
    "pedestrian": 0,
    "car": 1,
    "bike": 2,
    "bicycle": 2,
    "bus": 3,
    "motor": 4,
    "motorcycle": 4,
}

CLASS_NAMES = ["person", "car", "bicycle", "bus", "motorbike"]


def find_label_file(split: str) -> Path:
    candidates = [
        BDD_ROOT / "labels" / "det_20" / f"det_{split}.json",
        BDD_ROOT / "labels" / f"bdd100k_labels_images_{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "BDD100K label file not found. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def image_size(path: Path):
    with Image.open(path) as img:
        return img.size


def convert(split: str):
    label_file = find_label_file(split)
    image_dir = BDD_ROOT / "images" / "100k" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"BDD100K images not found: {image_dir}")

    with open(label_file, "r") as f:
        records = json.load(f)

    coco = {
        "info": {"description": f"BDD100K {split} 5-class detection"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(CLASS_NAMES)
        ],
    }

    ann_id = 1
    img_id = 1
    for rec in tqdm(records, desc=f"BDD100K {split}"):
        img_name = rec.get("name")
        if not img_name:
            continue
        img_path = image_dir / img_name
        if not img_path.exists():
            continue

        width, height = image_size(img_path)
        anns_for_image = []
        for label in rec.get("labels", []):
            category = label.get("category")
            if category not in CLASS_MAP or "box2d" not in label:
                continue
            box = label["box2d"]
            x1, y1 = float(box["x1"]), float(box["y1"])
            x2, y2 = float(box["x2"]), float(box["y2"])
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            anns_for_image.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_MAP[category],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

        if not anns_for_image:
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height,
        })
        coco["annotations"].extend(anns_for_image)
        img_id += 1

    out_file = BDD_ROOT / "annotations" / f"instances_{split}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(coco, f)

    print(f"Saved: {out_file}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()
    convert(args.split)


if __name__ == "__main__":
    main()
