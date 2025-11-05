from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from eyeagent.core.runtime import add_common_runtime_flags, apply_runtime_env_from_args
from eyeagent.core.settings import set_overrides

# Ensure local package is importable when running from repo root
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json_arg(val: str) -> Any:
    try:
        return json.loads(val)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")


def _read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_overridden_config(config_file: Optional[str], enable: List[str], disable: List[str]) -> Optional[str]:
    """Load the base settings, apply enable/disable overrides for agents, and write to a temp file.

    Returns the path to the temp config file or None if no overrides are needed.
    """
    if not enable and not disable and not config_file:
        return None

    # Defer import to avoid circulars at module import time
    from eyeagent.core.settings import Settings

    # If the user provided a specific config file, set override so Settings() resolves consistently
    if config_file:
        set_overrides(config_file=config_file)

    base = Settings().load()
    agents = (base.get("agents") or {}) if isinstance(base, dict) else {}

    # Normalize overrides to role keys (as used in settings["agents"])
    for key in enable:
        meta = agents.get(key) or {}
        if isinstance(meta, dict):
            meta["enabled"] = True
            agents[key] = meta
    for key in disable:
        meta = agents.get(key) or {}
        if isinstance(meta, dict):
            meta["enabled"] = False
            agents[key] = meta
    base["agents"] = agents

    # Persist to a temp file and point loader to it for the whole run
    tmp_dir = tempfile.mkdtemp(prefix="eyeagent_cfg_")
    out_path = Path(tmp_dir) / "eyeagent.yml"
    try:
        import yaml  # type: ignore
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(base, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        # Fallback to JSON if PyYAML not available
        out_path = Path(tmp_dir) / "eyeagent.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False, indent=2)
    # Ensure Settings picks this up
    set_overrides(config_file=str(out_path))
    return str(out_path)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eyeagent", description="EyeAgent CLI: config-driven multi-agent diagnosis")

    sub = p.add_subparsers(dest="cmd", required=True)

    # run subcommand
    run = sub.add_parser("run", help="Run a diagnosis with given input and configuration")
    run.add_argument("--patient", type=_load_json_arg, help="Patient JSON string")
    run.add_argument("--patient-file", type=str, help="Path to JSON file with patient object")
    run.add_argument("--images", type=_load_json_arg, help="Images JSON array")
    run.add_argument("--images-file", type=str, help="Path to JSON file containing images array")
    # unified runtime flags
    add_common_runtime_flags(run)
    # incremental / continue options
    run.add_argument("--case-id", type=str, help="Existing case ID to continue or run incrementally in the same case")
    run.add_argument("--incremental", action="store_true", help="Process only new images and merge with prior outputs (requires --case-id)")

    run.add_argument("--spec", type=str, help="Path to a YAML/JSON spec for backend=interaction")

    # topology backend related
    # (added via add_common_runtime_flags)

    run.add_argument("--enable-agent", action="append", default=[], help="Enable agent role (can be used multiple times)")
    run.add_argument("--disable-agent", action="append", default=[], help="Disable agent role (can be used multiple times)")

    # list-agents subcommand
    ls = sub.add_parser("list-agents", help="List registered/available agents from current configuration")
    ls.add_argument("--config", type=str, help="Path to eyeagent.yml to load before listing")

    # print-config subcommand
    pc = sub.add_parser("print-config", help="Print resolved configuration that EyeAgent will use")
    pc.add_argument("--config", type=str, help="Path to eyeagent.yml to load before printing")

    return p.parse_args(argv)


def _apply_overrides_and_env(ns: argparse.Namespace) -> Optional[str]:
    """Prepare overridden temp config (if needed) and apply unified env mapping.

    Returns the path to the temp config file if overrides were applied, else None.
    """
    # Apply initial mapping (may set EYEAGENT_CONFIG_FILE to ns.config)
    # We'll update ns.config to the overridden path if we produce one.
    overridden = _prepare_overridden_config(getattr(ns, "config", None), ns.enable_agent or [], ns.disable_agent or [])
    if overridden:
        ns.config = overridden
    apply_runtime_env_from_args(ns)
    return overridden


def _load_spec_file(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except Exception:
        return json.loads(text)


def cmd_run(ns: argparse.Namespace) -> int:
    # Merge and apply runtime config overrides first with unified env mapping
    _apply_overrides_and_env(ns)

    # Resolve input
    patient: Dict[str, Any] = {}
    images: List[Dict[str, Any]] = []
    if ns.patient is not None:
        patient = ns.patient
    if ns.patient_file:
        patient = _read_json_file(ns.patient_file)
    if ns.images is not None:
        images = ns.images
    if ns.images_file:
        images = _read_json_file(ns.images_file)

    # Optional spec for interaction backend
    spec: Optional[Dict[str, Any]] = None
    if ns.spec:
        spec = _load_spec_file(ns.spec)

    # Import the dispatcher and run
    from eyeagent.diagnostic_workflow import run_diagnosis_async
    report = None

    # Incremental / continue mode
    if getattr(ns, "incremental", False):
        if not ns.case_id:
            print("--incremental requires --case-id", file=sys.stderr)
            return 2
        # Load prior trace for previous images (and patient fallback)
        from eyeagent.trace.trace_logger import TraceLogger
        tl = TraceLogger()
        prior_path = os.path.join(tl.base_dir, ns.case_id, "trace.json")
        if not os.path.exists(prior_path):
            print(f"Case not found for --case-id: {ns.case_id}", file=sys.stderr)
            return 2
        with open(prior_path, "r", encoding="utf-8") as f:
            prior_doc = json.load(f)
        images_prev = prior_doc.get("images", []) if isinstance(prior_doc, dict) else []
        prev_ids = {str(i.get("image_id") or i.get("path")) for i in images_prev if isinstance(i, dict)}
        merged_images = list(images_prev)
        for it in (images or []):
            if not isinstance(it, dict):
                continue
            _id = str(it.get("image_id") or it.get("path"))
            if _id not in prev_ids:
                merged_images.append(it)
        # Patient fallback to prior if not provided this run
        if not patient:
            patient = prior_doc.get("patient", {}) if isinstance(prior_doc, dict) else {}
        # Prefer a backend that supports 'prior'
        # Prefer a backend that supports 'prior'
        set_overrides(workflow_backend="profile")
        prior = {"images": images_prev}
        try:
            report = asyncio_run_with_loop(run_diagnosis_async(patient, merged_images, case_id=ns.case_id, messages=None, spec=spec, prior=prior))
        except TypeError:
            report = asyncio_run_with_loop(run_diagnosis_async(patient, merged_images))
    else:
        try:
            report = asyncio_run_with_loop(run_diagnosis_async(patient, images, spec=spec))
        except TypeError:
            # Older builds without spec parameter
            report = asyncio_run_with_loop(run_diagnosis_async(patient, images))

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_list_agents(ns: argparse.Namespace) -> int:
    if ns.config:
        set_overrides(config_file=ns.config)
    # Register agents from settings and list
    from eyeagent.agents.registry import register_builtins, list_agents
    register_builtins()
    for k in list_agents():
        print(k)
    return 0


def cmd_print_config(ns: argparse.Namespace) -> int:
    from eyeagent.core.settings import Settings
    cfg = Settings().load()
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    return 0


def asyncio_run_with_loop(coro):
    import asyncio
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If there's an existing loop (e.g., in notebooks), fall back
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def main(argv: Optional[List[str]] = None) -> int:
    ns = _parse_args(argv)
    if ns.cmd == "run":
        return cmd_run(ns)
    if ns.cmd == "list-agents":
        return cmd_list_agents(ns)
    if ns.cmd == "print-config":
        return cmd_print_config(ns)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
