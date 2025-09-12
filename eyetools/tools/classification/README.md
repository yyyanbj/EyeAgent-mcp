# Classification Tool

Decoupled fundus image classification & regression tools discovered from the project root `tools/classification` directory.

## Supported Tasks
- `modality`: Multi-class (23) fundus modality categories
- `cfp_quality`: Image quality (7 classes)
- `laterality`: Left / Right (binary)
- `multidis`: Multi-label 260-class disease / finding predictions (CSRA attention)
- `cfp_age`: Age regression (renamed from legacy `fundus2age`)

## Architectures
- Classification tasks use a ResNet-101 backbone with CSRA multi-head attention (`ResNet_CSRA`, 2 heads, λ=0.1).
- Multi-label (`multidis`) and other classification tasks share identical forward producing raw logits. Tool logic applies sigmoid (multi-label) or softmax (single-label) as needed.
- Age regression uses a `res2net50d` backbone via `timm` (`CfpAgeModel`), provided through the unified environment `envs/py312`.

## Weights Layout
Place weight directories under `weights/classification/`:
```
weights/classification/
	modality23_resnet101_224_12-28-1602/
	cfp_quality7_resnet101_384_12-28-2221/
	laterality2_resnet101_224_12-30-1455/
	multidis260_resnet101_224_05-15-2006/
	cfp_age/   # renamed from fundus2age
```
Each folder may contain one or more `*.pth` checkpoint files. The loader prefers filenames beginning with `best` or `last`, otherwise the first `*.pth` file.

Missing weights are tolerated (model loads with random / ImageNet-pretrained backbone) so tests can run without distributing large files.

## Runtime
Image sizes: 384 for `cfp_quality` and `cfp_age`, else 224.
Normalization: ImageNet mean/std.
Age outputs are clamped to `[40, 70]`.

## Environment
All variants reference `environment_ref: py312` in `config.yaml`. The runtime resolves `envs/py312/pyproject.toml` and installs its dependencies (`torch`, `torchvision`, `timm`, etc.) using the uv overlay strategy. Extra per-tool dependencies can be added later via `extra_requires` in the config if needed.

## Programmatic Usage
```python
from classification.tool_impl import load_tool
tool = load_tool("modality")
res = tool.predict("/path/to/image.jpg")
```

## JSON Inference Example
Example output structures (abridged) for each task using a synthetic image:

```jsonc
// modality (multi-class)
{
	"task": "modality",
	"predictions": [
		{"label": "color_fundus", "score": 0.82},
		{"label": "oct", "score": 0.05}
	]
}

// cfp_quality (multi-class, 7 classes)
{
	"task": "cfp_quality",
	"predictions": [
		{"label": "good", "score": 0.73},
		{"label": "blur", "score": 0.12}
	]
}

// laterality (binary)
{
	"task": "laterality",
	"predictions": [
		{"label": "left", "score": 0.91}
	]
}

// multidis (multi-label 260 classes; truncated)
{
	"task": "multidis",
	"predictions": [
		{"label": "drusen", "score": 0.61},
		{"label": "exudate", "score": 0.44}
	],
	"probabilities": {
		"drusen": 0.61,
		"exudate": 0.44,
		"...": 0.01
	}
}

// cfp_age (regression)
{
	"task": "cfp_age",
	"prediction": 56.4
}
```

## Discovery
`config.yaml` declares variants; the MCP server discovers tools when `EYETOOLS_EXTRA_TOOL_PATHS` (or legacy `EYETOOLS_TOOL_PATHS`) includes the project root / `tools` directory.

## Testing
Pytest adds `tools/` to `sys.path` via `tests/conftest.py`. Synthetic images validate output shapes/keys without requiring real weights.

## Notes / Future
- Add explicit checkpoint file names in config for reproducibility (optional).
- Optional quantization / half precision path for deployment.

