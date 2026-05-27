"""
convert_voc_to_coco.py
Convert Pascal VOC2012 to COCO JSON.

Your actual structure:
  data/pascal_voc/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/
  ├── Annotations/        ← XML files
  ├── ImageSets/Main/     ← train.txt, val.txt
  └── JPEGImages/         ← images

Usage:
    cd "d:/EML/Final Project/D_fine"
    python data/convert_voc_to_coco.py --split train
    python data/convert_voc_to_coco.py --split val
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

VOC_ROOT = Path("data/pascal_voc/VOCtrainval_11-May-2012/VOCdevkit/VOC2012")
ANN_DIR  = VOC_ROOT / "Annotations"
IMG_DIR  = VOC_ROOT / "JPEGImages"
SETS_DIR = VOC_ROOT / "ImageSets" / "Main"

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CAT_ID = {name: i for i, name in enumerate(VOC_CLASSES)}


def parse_xml(xml_path):
    tree   = ET.parse(xml_path)
    root   = tree.getroot()
    size   = root.find("size")
    width  = int(size.findtext("width"))
    height = int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CAT_ID:
            continue
        bb   = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        objects.append({"name": name, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin]})
    return width, height, objects


def convert(split: str, output_path: str):
    split_file = SETS_DIR / f"{split}.txt"
    image_ids  = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

    coco = {
        "info": {"description": f"VOC2012 {split}", "date_created": datetime.now().isoformat()},
        "licenses": [], "images": [], "annotations": [],
        "categories": [{"id": v, "name": k, "supercategory": "object"} for k, v in CAT_ID.items()],
    }

    ann_id = 1
    for img_id, name in enumerate(tqdm(image_ids, desc=f"VOC {split}"), start=1):
        xml_path = ANN_DIR / f"{name}.xml"
        if not xml_path.exists():
            continue

        width, height, objects = parse_xml(xml_path)
        coco["images"].append({
            "id": img_id, "file_name": f"{name}.jpg",
            "width": width, "height": height,
        })

        for obj in objects:
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": CAT_ID[obj["name"]],
                "bbox": obj["bbox"],
                "area": obj["bbox"][2] * obj["bbox"][3],
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
    output = args.output or f"data/pascal_voc/annotations/instances_{args.split}.json"
    convert(args.split, output)
