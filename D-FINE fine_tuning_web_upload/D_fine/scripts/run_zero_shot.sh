#!/usr/bin/env bash
# run_zero_shot.sh
# Evaluate COCO-pretrained D-FINE directly on each target dataset (no fine-tuning).
# This gives the zero-shot / domain baseline for computing domain gap.

set -e

MODEL="s"
PRETRAINED="D-FINE/weights/dfine_hgnetv2_${MODEL}_coco.pth"
DATASETS="pascal_voc visdrone cityscapes kitti"

echo "=== Zero-shot evaluation (COCO pretrained → target datasets) ==="

for ds in $DATASETS; do
    echo "  Evaluating on $ds ..."
    # Use full_finetune config for architecture (but DO NOT train, eval only)
    python eval/evaluate.py \
        --config  "configs/dfine/full_finetune/dfine_${MODEL}_${ds}.yml" \
        --checkpoint "$PRETRAINED" \
        --dataset "$ds" \
        --output_dir "results/zero_shot/${ds}"
done

echo "Zero-shot results saved to results/zero_shot/"
