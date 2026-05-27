#!/usr/bin/env bash
# run_evaluation.sh
# Evaluate trained models + run inference on all 3 datasets
#
# Run from: d:/EML/Final Project/D-FINE/
#   cd "d:/EML/Final Project/D-FINE"
#   bash ../D_fine/scripts/run_evaluation.sh

set -e
WEIGHTS="weights"
RESULTS="../D_fine/results"
EVAL="../D_fine/eval"
CONFIGS="configs/dfine/custom"

find_ckpt() {
    local dir="$RESULTS/$1"
    if   [ -f "$dir/best.pth" ];   then echo "$dir/best.pth"
    elif [ -f "$dir/last.pth" ];   then echo "$dir/last.pth"
    else ls -t "$dir"/checkpoint*.pth 2>/dev/null | head -1; fi
}

# ── 1. mAP on trained datasets (VOC + VisDrone) ───────────────────────────────
echo "### mAP Evaluation — trained models ###"

for tag in full_finetune/voc_S full_finetune/visdrone_S \
           partial_finetune/voc_S partial_finetune/visdrone_S \
           peft/voc_S peft/visdrone_S \
           full_finetune/voc_N full_finetune/visdrone_N; do

    ckpt=$(find_ckpt "$tag")
    [ -z "$ckpt" ] && echo "SKIP $tag (no checkpoint)" && continue

    # Pick dataset and config based on tag
    if echo "$tag" | grep -q "visdrone"; then
        ds="visdrone"
        cfg="$CONFIGS/dfine_s_visdrone_full.yml"
        [ "$(echo $tag | grep -c '_N')" -eq 1 ] && cfg="$CONFIGS/dfine_n_visdrone_full.yml"
    else
        ds="pascal_voc"
        cfg="$CONFIGS/dfine_s_voc_full.yml"
        [ "$(echo $tag | grep -c '_N')" -eq 1 ] && cfg="$CONFIGS/dfine_n_voc_full.yml"
    fi

    echo "--- Evaluating $tag ---"
    python "$EVAL/evaluate.py" \
        --config "$cfg" \
        --checkpoint "$ckpt" \
        --dataset "$ds" \
        --output_dir "$RESULTS/$tag"
done

# ── 2. Latency: GPU vs CPU ─────────────────────────────────────────────────────
echo "### Latency Benchmark — GPU vs CPU ###"
BEST_VOC=$(find_ckpt full_finetune/voc_S)
if [ -n "$BEST_VOC" ]; then
    for device in cuda cpu; do
        python "$EVAL/latency_benchmark.py" \
            --config "$CONFIGS/dfine_s_voc_full.yml" \
            --checkpoint "$BEST_VOC" \
            --device "$device" --num_iters 100 \
            --output_dir "$RESULTS/latency" --tag "S_voc_${device}"
    done
fi

# ── 3. Robustness on best model ────────────────────────────────────────────────
echo "### Robustness Evaluation ###"
if [ -n "$BEST_VOC" ]; then
    python "$EVAL/robustness_eval.py" \
        --config "$CONFIGS/dfine_s_voc_full.yml" \
        --checkpoint "$BEST_VOC" \
        --dataset pascal_voc --severity 1 3 5 \
        --output_dir "$RESULTS/robustness/S_voc"
fi

# ── 4. Inference-only on 3 datasets (zero-shot, COCO pretrained) ──────────────
echo "### Inference-only: 3 datasets (zero-shot) ###"

# VOC zero-shot
python "$EVAL/evaluate.py" \
    --config "$CONFIGS/dfine_s_voc_full.yml" \
    --checkpoint "$WEIGHTS/dfine_s_coco.pth" \
    --dataset pascal_voc \
    --output_dir "$RESULTS/inference_only/pascal_voc"

# VisDrone zero-shot
python "$EVAL/evaluate.py" \
    --config "$CONFIGS/dfine_s_visdrone_full.yml" \
    --checkpoint "$WEIGHTS/dfine_s_coco.pth" \
    --dataset visdrone \
    --output_dir "$RESULTS/inference_only/visdrone"

# Cityscapes zero-shot (inference config uses COCO 80-class head)
python "$EVAL/evaluate.py" \
    --config "$CONFIGS/dfine_s_cityscapes_infer.yml" \
    --checkpoint "$WEIGHTS/dfine_s_coco.pth" \
    --dataset cityscapes \
    --output_dir "$RESULTS/inference_only/cityscapes"

# ── 5. Generalization gap analysis ────────────────────────────────────────────
echo "### Generalization Analysis ###"
python "$EVAL/generalization_analysis.py" \
    --results_root "$RESULTS/" \
    --output_dir "$RESULTS/generalization"

# ── 6. Plots ──────────────────────────────────────────────────────────────────
echo "### Generate Plots ###"
python "../D_fine/analysis/plot_results.py" \
    --results_root "$RESULTS/" \
    --output_dir "$RESULTS/figures"

echo ""
echo "=== All evaluations complete ==="
