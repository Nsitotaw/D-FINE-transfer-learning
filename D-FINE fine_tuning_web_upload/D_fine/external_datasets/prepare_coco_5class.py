"""
Download COCO val2017 and create a 5-class COCO annotation file.

Classes:
  0 person
  1 car
  2 bicycle
  3 bus
  4 motorbike

Run:
  C:\\envs\\dfine\\python.exe D_fine\\external_datasets\\prepare_coco_5class.py
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "coco_5class"
RAW = OUT / "raw"

COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

TARGET_CLASSES = {
    "person": 0,
    "car": 1,
    "bicycle": 2,
    "bus": 3,
    "motorcycle": 4,
}

OUTPUT_NAMES = {
    "motorcycle": "motorbike",
}


class DownloadProgress(tqdm):
    def update_to(self, block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"Exists: {dst}")
        return
    print(f"Downloading {url}")
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=dst.name) as bar:
        urlretrieve(url, dst, reporthook=bar.update_to)


def extract(zip_path: Path, dst: Path, marker: Path):
    if marker.exists():
        print(f"Extracted: {marker}")
        return
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


def filter_annotations(src_ann: Path, dst_ann: Path):
    print(f"Filtering annotations: {src_ann}")
    with open(src_ann, "r") as f:
        coco = json.load(f)

    source_cats = {cat["id"]: cat["name"] for cat in coco["categories"]}
    keep_source_ids = {
        cat_id for cat_id, name in source_cats.items()
        if name in TARGET_CLASSES
    }

    anns = []
    keep_image_ids = set()
    next_ann_id = 1
    for ann in coco["annotations"]:
        name = source_cats.get(ann["category_id"])
        if name not in TARGET_CLASSES:
            continue
        new_ann = dict(ann)
        new_ann["id"] = next_ann_id
        new_ann["category_id"] = TARGET_CLASSES[name]
        anns.append(new_ann)
        keep_image_ids.add(new_ann["image_id"])
        next_ann_id += 1

    images = [img for img in coco["images"] if img["id"] in keep_image_ids]
    categories = [
        {
            "id": new_id,
            "name": OUTPUT_NAMES.get(name, name),
            "supercategory": "object",
        }
        for name, new_id in TARGET_CLASSES.items()
    ]

    filtered = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": categories,
    }

    dst_ann.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_ann, "w") as f:
        json.dump(filtered, f)

    print(f"Saved: {dst_ann}")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(anns)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    val_zip = RAW / "val2017.zip"
    ann_zip = RAW / "annotations_trainval2017.zip"
    val_dir = RAW / "val2017"
    ann_dir = RAW / "annotations"
    out_ann = OUT / "annotations" / "instances_val.json"

    if args.force and OUT.exists():
        shutil.rmtree(OUT)

    download(COCO_URLS["val2017"], val_zip)
    download(COCO_URLS["annotations"], ann_zip)
    extract(val_zip, RAW, val_dir)
    extract(ann_zip, RAW, ann_dir / "instances_val2017.json")
    filter_annotations(ann_dir / "instances_val2017.json", out_ann)


if __name__ == "__main__":
    main()
