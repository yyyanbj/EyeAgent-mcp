"""Dynamic configuration alternative to config.yaml (v2 schema)."""

from pathlib import Path
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "package": "demo_template_dyn",
        "entry": "tool_impl:DemoTemplateTool",
        "category": "demo",
        "shared": {
            # Reference env baseline
            "environment_ref": "py312",
            "extra_requires": ["pillow"],
            "runtime": {
                "load_mode": "auto",
                "idle_timeout_s": 600,
                "device_policy": "prefer_cuda",
                "preload": False,
            },
            "warmup": {"on_first_call": True, "batch_size": 1},
            "io": {
                "input_schema": {"input": "path|bytes|text"},
                "output_schema": {"message": "str"},
            },
            "model_defaults": {
                "precision": "auto",
                "lazy": True,
                "fail_on_missing": True,
                "strict_load": False,
                "verify_num_classes": True,
            },
        },
        "variants": [],
        "metadata": {
            "tags": ["demo", "dynamic"],
            "license": "Apache-2.0",
            "authors": ["EyeAgent Team"],
        },
        "notes": ["0.1.0 dynamic template"],
    }

    for variant, size, prec, classes in [
        ("tiny", 10, "auto", ["A", "B"]),
        ("mega", 800, "fp16", ["A", "B", "C"]),
    ]:
        base["variants"].append(
            {
                "variant": variant,
                "version": "0.1.0",
                "model": {
                    "weights": f"artifacts/{variant}.pt",
                    "size_mb": size,
                    "precision": prec,
                    "lazy": True,
                },
                "params": {
                    "msg_prefix": variant.upper(),
                    "classes": classes,
                },
            }
        )
    return base

