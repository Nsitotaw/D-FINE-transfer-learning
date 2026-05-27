# Stress-Testing D-FINE for Domain Adaptation in Object Detection

This repository contains the open project files for a class study comparing D-FINE fine-tuning strategies under limited training hardware. The study evaluates COCO-pretrained D-FINE on five-class PASCAL VOC and VisDrone detection tasks using full fine-tuning, partial fine-tuning with a frozen backbone, and LoRA-based parameter-efficient fine-tuning.

## What Is Included

- `D_fine/analysis/`: scripts used to collect results, generate plots, update the paper, and visualize detections.
- `D_fine/configs/`: experiment and inference configuration files.
- `D_fine/data/*.py`: dataset conversion scripts for PASCAL VOC and VisDrone.
- `D_fine/eval/`: evaluation, robustness, latency, and generalization scripts.
- `D_fine/external_datasets/`: open dataset preparation/conversion helpers for COCO 5-class and BDD100K.
- `D_fine/finetune/`: LoRA and frozen-backbone helper code.
- `D_fine/paper/`: final report document and generated figures.
- `D_fine/results/all_results.xlsx`: summarized experiment results.
- `D_fine/results/**/log.txt`: training logs only, without checkpoints.
- `D-FINE/`: the lightweight D-FINE source/configuration files needed to show the modified training setup, including `train_lora.py` and `train_lora_r32.py`.
- `model_weights/`: compact inference-only checkpoints for the best completed runs.

## What Is Excluded

Large or non-open files are intentionally not included:

- Raw datasets and extracted images
- Model checkpoints and pretrained weights
- Full training-resume checkpoints with optimizer/scheduler/scaler state
- Generated validation sample images
- Python cache folders
- The original nested Git history from the upstream D-FINE clone

This keeps the repository compact while still showing the code, configurations, results, paper artifacts, and inference weights needed to understand the project.

## Datasets

The original experiments used COCO-format versions of:

- PASCAL VOC five-class subset: `person`, `car`, `bicycle`, `bus`, `motorbike`
- VisDrone five-class subset: `pedestrian`, `car`, `motor`, `bicycle`, `bus`

Due to training hardware limitations, only five classes were used for each dataset. Dataset conversion scripts are provided in `D_fine/data/`, but users must download the datasets separately from their official sources.

## LoRA Configuration

The main LoRA runs inject low-rank adapters into D-FINE decoder linear layers while training only LoRA parameters and detection head parameters. The main configuration uses rank 16, alpha 32, dropout 0.05, AdamW, LoRA learning rate `3e-4`, and detection-head learning rate `1e-4`. A higher-capacity VisDrone ablation uses rank 32 and alpha 64.

## Reproducing the Workflow

1. Install dependencies from `D_fine/requirements.txt` and `D-FINE/requirements.txt`.
2. Download PASCAL VOC and VisDrone from their official sources.
3. Convert datasets to COCO format using:
   - `D_fine/data/convert_voc_to_coco.py`
   - `D_fine/data/convert_visdrone_to_coco.py`
4. Place D-FINE COCO pretrained weights in `D-FINE/weights/`.
5. Run full, partial, or LoRA training using the configs in `D-FINE/configs/dfine/custom/`.
6. Use `D_fine/analysis/collect_results.py` and plotting scripts to regenerate result tables and figures.

