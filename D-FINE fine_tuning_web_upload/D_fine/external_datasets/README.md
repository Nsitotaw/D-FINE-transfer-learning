# External Datasets

This folder replaces the removed Cityscapes external test setup.

Datasets prepared here:

- `coco_5class`: direct-download COCO val2017 filtered to the project classes:
  `person`, `car`, `bicycle`, `bus`, `motorbike`
- `bdd100k`: BDD100K detection data. The official source requires manual/authenticated download, so this folder includes a converter and expected placement.

## COCO 5-Class

Run from the project root:

```powershell
cd "D:\EML\Final Project"
C:\envs\dfine\python.exe D_fine\external_datasets\prepare_coco_5class.py
```

Output:

```text
D_fine/data/coco_5class/
  raw/val2017/
  raw/annotations/instances_val2017.json
  annotations/instances_val.json
```

## BDD100K

Download from the official BDD100K portal or another licensed source, then arrange files as:

```text
D_fine/data/bdd100k/
  images/100k/val/*.jpg
  labels/det_20/det_val.json
```

Older BDD labels are also supported:

```text
D_fine/data/bdd100k/
  images/100k/val/*.jpg
  labels/bdd100k_labels_images_val.json
```

Then convert:

```powershell
cd "D:\EML\Final Project"
C:\envs\dfine\python.exe D_fine\external_datasets\convert_bdd100k_to_coco.py --split val
```

Output:

```text
D_fine/data/bdd100k/annotations/instances_val.json
```
