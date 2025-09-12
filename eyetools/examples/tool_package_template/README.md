# Tool Package Template

This template demonstrates how to create a multi-variant tool package for the EyeTools framework.

## Layout
```
tool_package_template/
  config.yaml              # YAML multi-variant config (authoritative)
  config.py                # Dynamic config alternative (optional)
  tool_impl.py             # Core Tool implementation
  main.py                  # Local CLI for quick testing
  artifacts/               # Model weights / resources placeholder
    .gitkeep
  requirements.txt         # Extra dependencies beyond base env
  __init__.py
```

## Quick Start
1. Copy this folder and rename.
2. Edit `config.yaml` or `config.py`.
3. Implement logic in `tool_impl.py` (see DemoTool).
4. (Optional) Add weights under `artifacts/`.
5. Run local test:
```
python main.py --list
python main.py --tool demo_template:small --input sample.jpg
```

## Config Modes
- Single tool: top-level fields.
- `variants`: shared + many variant definitions.
- `tools`: independent list.

## Mandatory Fields (per tool)
- entry (module:ClassName)
- version (defaults to 0.1.0 if unset)
- python (runtime base, e.g. py310)
- model.weights (if model needed) or set model.lazy true

## Runtime Load Mode
`runtime.load_mode`: auto | inproc | subprocess

## Example IDs
Generated as: package + ':' + variant (for variants)

## License
MIT (adjust as needed).
