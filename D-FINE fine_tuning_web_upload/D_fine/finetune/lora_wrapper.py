"""
lora_wrapper.py
Low-Rank Adaptation (LoRA) for D-FINE transformer decoder.

Injects LoRA into every nn.Linear inside the decoder's self-attention and
cross-attention layers. Works with any D-FINE model size (N/S/M/L/X).

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.

Usage:
    from finetune.lora_wrapper import apply_lora, merge_lora, get_lora_params
    model = apply_lora(model, rank=16, alpha=32, dropout=0.05)
    # train only lora + head params
    trainable = get_lora_params(model)
    optimizer = torch.optim.AdamW(trainable, lr=3e-4)
"""

import math
import re
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank bypass."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        original_layer: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Keep original weights frozen
        if original_layer is not None:
            self.weight = original_layer.weight
            self.bias   = original_layer.bias
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
            self.bias   = None

        # LoRA parameters (always trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_lora()

    def _init_lora(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + lora * self.scaling

    def merge(self):
        """Absorb LoRA into the base weight (for inference without overhead)."""
        with torch.no_grad():
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.lora_merged = True

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}")


def _get_target_modules(model: nn.Module, name_patterns: List[str]) -> List[str]:
    """Return fully-qualified names of Linear layers that match any pattern."""
    matched = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(re.search(p, name) for p in name_patterns):
            matched.append(name)
    return matched


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Replace module at dotted path `name` in model."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_patterns: Optional[List[str]] = None,
) -> nn.Module:
    """
    Inject LoRA into all matching Linear layers of a D-FINE model.

    Default targets: all Linear layers inside the transformer decoder
    (self-attention q/k/v/out projections and cross-attention projections).
    """
    if target_patterns is None:
        target_patterns = [
            r"decoder\.layers\.\d+\.self_attn\.(q|k|v|out)_proj",
            r"decoder\.layers\.\d+\.cross_attn\.(q|k|v|out)_proj",
            # Also target feed-forward projections in the decoder
            r"decoder\.layers\.\d+\.linear[12]",
        ]

    target_names = _get_target_modules(model, target_patterns)
    if not target_names:
        # Fallback: target all Linear layers in decoder
        target_names = [
            n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and "decoder" in n
        ]

    print(f"[LoRA] Injecting LoRA (rank={rank}, alpha={alpha}) into {len(target_names)} layers:")
    for name in target_names:
        orig: nn.Linear = dict(model.named_modules())[name]
        lora_layer = LoRALinear(
            in_features=orig.in_features,
            out_features=orig.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            original_layer=orig,
        )
        # Freeze original weight
        lora_layer.weight.requires_grad = False
        if lora_layer.bias is not None:
            lora_layer.bias.requires_grad = False

        _set_module(model, name, lora_layer)
        print(f"  ✓ {name}")

    # Freeze everything except LoRA params and detection head
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif re.search(r"(class_head|bbox_head|score_head|fdr|go_lsd)", name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*trainable/total:.2f}%)")

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base weights (zero overhead inference)."""
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            merged += 1
    print(f"[LoRA] Merged {merged} LoRA layers into base weights.")
    return model


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return only LoRA + detection head parameters for the optimizer."""
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
    return params


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA delta weights (much smaller checkpoint)."""
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_state, path)
    print(f"[LoRA] Saved {len(lora_state)} LoRA tensors → {path}")


def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA delta weights into model (base weights unchanged)."""
    lora_state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    if unexpected:
        print(f"[LoRA] WARNING: unexpected keys: {unexpected}")
    print(f"[LoRA] Loaded {len(lora_state)} LoRA tensors from {path}")
    return model
