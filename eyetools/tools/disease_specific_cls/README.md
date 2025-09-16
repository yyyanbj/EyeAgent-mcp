# Disease-Specific Classification Tools (RETFound Dinov3)

This package exposes individual disease classification models fine-tuned from
RETFound Dinov3 ViT-B/16 backbones. Each variant corresponds to a folder under
`weights/disease-specific/<variant>` that contains a `checkpoint-best.pth`.

## Variants
All `*_finetune` and `*_lp` (linear probing) folders are auto-registered via
`config.yaml`. Example variants: `AMD_finetune`, `AMD_lp`, `DR_finetune`, etc.

## Environment
Models load in the dedicated `py311-retfound` uv environment (Python 3.11 per
upstream RETFound README) which includes `torch==2.5.1`, `torchvision==0.20.1`,
`timm (0.9.x)`, `scikit-learn`, `pycm`, and related dependencies. A previous
`py312-retfound` experimental file remains for reference but config now targets
3.11 for maximum compatibility with released checkpoints.

## Usage (Programmatic)
```python
from tools.disease_specific_cls.tool_impl import load_tool
tool = load_tool("AMD_finetune")  # loads weights/disease-specific/AMD_finetune/checkpoint-best.pth
out = tool.predict({"image_path": "examples/test_images/AMD.jpg"})
print(out)
```

## Prediction Output Schema
```json
{
  "disease": "amd",
  "probability": 0.87,
  "predicted": true,
  "all_probabilities": {"amd": 0.87, "normal": 0.13},
  "inference_time": 0.024
}
```

## Class/Head Inference Logic
The tool inspects the checkpoint's classifier head weight tensor:
- Output dim == 1: binary logistic (single disease probability via sigmoid)
- Output dim == 2: softmax (disease vs normal)
- Output dim > 2: softmax multi-class (generic class names assigned if labels unavailable)

## Dinov3 Backbone
We attempt to construct the Dinov3 ViT-B/16 backbone via `torch.hub`. If this
fails (offline environment), a timm ViT-B/16 fallback is used; state dict is
loaded with `strict=False` so missing keys are tolerated.

## Threshold
Default decision threshold is `0.5`; override per variant by supplying
`params.threshold` when instantiating or editing `config.yaml`.

## Notes
- Checkpoints are expected to be saved as raw state dicts or dicts with
  `model` / `state_dict` keys.
- For reproducible performance ensure network access for official Dinov3 code.
