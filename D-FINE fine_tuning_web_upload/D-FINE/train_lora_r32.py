"""
train_lora_r32.py
LoRA-R32 fine-tuning of D-FINE on custom datasets.

This is a copy of train_lora.py with higher LoRA adapter capacity:
rank=32 and alpha=64. It keeps the same optimizer learning rates,
dropout, seed handling, and D-FINE training loop.

Usage from the D-FINE directory:
    python train_lora_r32.py -c configs/dfine/custom/dfine_s_visdrone_lora_r32.yml \
        -t weights/dfine_s_coco.pth --use-amp --seed 0
"""

import os
import sys
import argparse
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_lora_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "D_fine", "finetune")
sys.path.insert(0, _lora_path)
from lora_wrapper import apply_lora

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS


LORA_RANK = 32
LORA_ALPHA = 64.0
LORA_DROPOUT = 0.05
LORA_LR = 3e-4
HEAD_LR = 1e-4


def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {k: v for k, v in args.__dict__.items() if k not in ["update"] and v is not None}
    )

    cfg = YAMLConfig(args.config, **update_dict)

    if args.tuning:
        cfg.yaml_cfg.setdefault("HGNetv2", {})["pretrained"] = False

    model = cfg.model
    print(f"[LoRA-R32] Base model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    model = apply_lora(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
    )

    head_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": lora_params, "lr": LORA_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    cfg._model = model
    cfg._optimizer = optimizer

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str)
    parser.add_argument("-t", "--tuning", type=str)
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--summary-dir", type=str)
    parser.add_argument("--test-only", action="store_true", default=False)
    parser.add_argument("-u", "--update", nargs="+")
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()
    main(args)
