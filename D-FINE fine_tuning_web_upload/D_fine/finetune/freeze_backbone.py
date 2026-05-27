"""
freeze_backbone.py
Utilities for partial fine-tuning of D-FINE: freeze the backbone (and optionally
the encoder), leaving only the transformer decoder + detection heads trainable.

Usage (called by the D-FINE train.py via a monkey-patch or wrapper):
    from finetune.freeze_backbone import apply_freeze_strategy, count_parameters
    apply_freeze_strategy(model, freeze_backbone=True, freeze_encoder=False)
    count_parameters(model)
"""

import re
from typing import Optional
import torch.nn as nn


# Regex patterns matching parameter name prefixes in D-FINE
_BACKBONE_PATTERN  = re.compile(r"^backbone\.")
_ENCODER_PATTERN   = re.compile(r"^encoder\.")
_DECODER_PATTERN   = re.compile(r"^decoder\.")
_HEAD_PATTERN      = re.compile(r"^(class_head|bbox_head|score_head|fdr|go_lsd)\.")


def freeze_module(module: nn.Module, name_prefix: str = ""):
    """Recursively set requires_grad=False for all parameters."""
    for param in module.parameters():
        param.requires_grad = False
    return module


def unfreeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = True
    return module


def apply_freeze_strategy(
    model: nn.Module,
    freeze_backbone: bool = True,
    freeze_encoder: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply a freeze strategy to D-FINE.

    freeze_backbone=True  → backbone weights fixed (transfers visual features as-is)
    freeze_encoder=True   → additionally freeze the transformer encoder
    Both False            → full fine-tuning

    Returns the model with modified requires_grad flags.
    """
    # First unfreeze everything so we start from a clean state
    for param in model.parameters():
        param.requires_grad = True

    frozen_params = []
    trainable_params = []

    for name, param in model.named_parameters():
        should_freeze = False

        if freeze_backbone and _BACKBONE_PATTERN.match(name):
            should_freeze = True
        if freeze_encoder and _ENCODER_PATTERN.match(name):
            should_freeze = True

        if should_freeze:
            param.requires_grad = False
            frozen_params.append(name)
        else:
            trainable_params.append(name)

    if verbose:
        print(f"[FreezeStrategy] freeze_backbone={freeze_backbone}, freeze_encoder={freeze_encoder}")
        print(f"  Frozen    : {len(frozen_params):,} tensors")
        print(f"  Trainable : {len(trainable_params):,} tensors")

    return model


def count_parameters(model: nn.Module, only_trainable: bool = True) -> dict:
    """Return parameter counts broken down by module group."""
    groups = {
        "backbone": 0,
        "encoder":  0,
        "decoder":  0,
        "head":     0,
        "other":    0,
    }
    total = 0
    trainable = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

        if _BACKBONE_PATTERN.match(name):
            groups["backbone"] += n
        elif _ENCODER_PATTERN.match(name):
            groups["encoder"] += n
        elif _DECODER_PATTERN.match(name):
            groups["decoder"] += n
        elif _HEAD_PATTERN.match(name):
            groups["head"] += n
        else:
            groups["other"] += n

    result = {
        "total_M":     total / 1e6,
        "trainable_M": trainable / 1e6,
        "frozen_M":    (total - trainable) / 1e6,
        "groups_M":    {k: v / 1e6 for k, v in groups.items()},
    }
    print(f"  Total params    : {result['total_M']:.2f}M")
    print(f"  Trainable params: {result['trainable_M']:.2f}M  "
          f"({100 * trainable / total:.1f}%)")
    print(f"  Frozen params   : {result['frozen_M']:.2f}M")
    for grp, cnt in result["groups_M"].items():
        print(f"    {grp:12s}: {cnt:.2f}M")
    return result


def freeze_bn_layers(model: nn.Module):
    """Keep BatchNorm in eval mode (prevent running-stat corruption with small batches)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
