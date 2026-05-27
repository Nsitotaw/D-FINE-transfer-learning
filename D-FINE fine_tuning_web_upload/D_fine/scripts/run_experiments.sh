#!/usr/bin/env bash
# run_experiments.sh
# Train D-FINE on VOC2012 + VisDrone — 3 strategies × D-FINE-S + scaling test with D-FINE-N
#
# Run from: d:/EML/Final Project/D-FINE/
#   cd "d:/EML/Final Project/D-FINE"
#   bash ../D_fine/scripts/run_experiments.sh

set -e
WEIGHTS="weights"
CONFIGS="configs/dfine/custom"
PORT=7777

run_full() {
    local config=$1 pretrained=$2 tag=$3
    echo ""
    echo "=== $tag ==="
    CUDA_VISIBLE_DEVICES=0 python train.py \
        -c "$CONFIGS/$config" \
        -t "$WEIGHTS/$pretrained" \
        --use-amp --seed 0
    PORT=$((PORT+1))
}

run_lora() {
    local config=$1 pretrained=$2 tag=$3
    echo ""
    echo "=== $tag [LoRA] ==="
    CUDA_VISIBLE_DEVICES=0 python train_lora.py \
        -c "$CONFIGS/$config" \
        -t "$WEIGHTS/$pretrained" \
        --use-amp --seed 0
    PORT=$((PORT+1))
}

# ── Strategy 1: Full fine-tuning (D-FINE-S) ───────────────────────────────────
run_full dfine_s_voc_full.yml      dfine_s_coco.pth  "Full FT | S | VOC"
run_full dfine_s_visdrone_full.yml dfine_s_coco.pth  "Full FT | S | VisDrone"

# ── Strategy 2: Partial fine-tuning / frozen backbone (D-FINE-S) ──────────────
run_full dfine_s_voc_partial.yml      dfine_s_coco.pth  "Partial FT | S | VOC"
run_full dfine_s_visdrone_partial.yml dfine_s_coco.pth  "Partial FT | S | VisDrone"

# ── Strategy 3: LoRA / PEFT (D-FINE-S) ───────────────────────────────────────
run_lora dfine_s_voc_lora.yml      dfine_s_coco.pth  "LoRA | S | VOC"
run_lora dfine_s_visdrone_lora.yml dfine_s_coco.pth  "LoRA | S | VisDrone"

# ── Model scaling stress test (D-FINE-N) ──────────────────────────────────────
run_full dfine_n_voc_full.yml      dfine_n_coco.pth  "Full FT | N | VOC"
run_full dfine_n_visdrone_full.yml dfine_n_coco.pth  "Full FT | N | VisDrone"

echo ""
echo "=== All 8 training runs complete ==="
