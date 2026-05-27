# Compact Model Weights

This folder contains compact inference checkpoints for the best completed runs.
The original training checkpoints saved optimizer, scheduler, scaler, EMA, and
training-resume state, which made many files about 165 MB each. These files keep
only the EMA model weights plus minimal metadata, so they are suitable for a
compact GitHub submission.

| File | Source run | Approx size |
|---|---|---:|
| `full_S_voc_inference.pth` | Full fine-tuning, D-FINE-S, VOC | 41.46 MB |
| `full_S_visdrone_inference.pth` | Full fine-tuning, D-FINE-S, VisDrone | 41.47 MB |
| `partial_S_voc_inference.pth` | Partial fine-tuning, D-FINE-S, VOC | 41.45 MB |
| `partial_S_visdrone_inference.pth` | Partial fine-tuning, D-FINE-S, VisDrone | 41.46 MB |
| `lora_S_voc_inference.pth` | LoRA, D-FINE-S, VOC | 41.46 MB |
| `lora_r8_S_visdrone_inference.pth` | LoRA-r8, D-FINE-S, VisDrone | 41.47 MB |
| `lora_r32_S_visdrone_inference.pth` | LoRA-r32, D-FINE-S, VisDrone | 42.66 MB |
| `lora_N_visdrone_inference.pth` | LoRA, D-FINE-N, VisDrone | 15.60 MB |

# Only this two are encluded in the github repo others can be given based on request as the data are too large
| `full_N_voc_inference.pth` | Full fine-tuning, D-FINE-N, VOC | 15.29 MB |
| `full_N_visdrone_inference.pth` | Full fine-tuning, D-FINE-N, VisDrone | 15.30 MB |

Total compact weight size: about 337.6 MB.

Each checkpoint stores:

```python
{
    "model": state_dict,
    "source_checkpoint": "...",
    "source_state": "ema.module",
    "last_epoch": ...,
    "note": "Compact inference checkpoint..."
}
```
