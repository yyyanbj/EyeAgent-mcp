import argparse
import json
from pathlib import Path
import importlib.util
import runpy
import yaml

# Minimal local loader for quick testing of the template without full framework.

def load_yaml(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def detect_mode(data):
    if 'variants' in data:
        return 'variants'
    if 'tools' in data:
        return 'tools'
    return 'single'

def expand_config(cfg, base_dir: Path):
    mode = detect_mode(cfg)
    tools = []
    if mode == 'single':
        tools.append(cfg)
    elif mode == 'variants':
        shared = cfg.get('shared', {})
        pkg = cfg.get('package')
        entry = cfg.get('entry')
        for v in cfg['variants']:
            merged = {**shared, **v}
            merged['package'] = pkg
            merged['entry'] = v.get('entry', entry)
            merged['variant'] = v['variant']
            merged['id'] = f"{pkg}:{v['variant']}"
            tools.append(merged)
    elif mode == 'tools':
        for t in cfg['tools']:
            tools.append(t)
    return tools

def load_py_config(py_path: Path):
    mod_globals = runpy.run_path(str(py_path))
    if 'get_config' in mod_globals:
        return mod_globals['get_config']()
    if 'CONFIGS' in mod_globals:
        return {"tools": mod_globals['CONFIGS']}
    raise RuntimeError("Python config must define get_config() or CONFIGS")

def load_all_configs(base_dir: Path):
    items = []
    yaml_path = base_dir / 'config.yaml'
    py_path = base_dir / 'config.py'
    if yaml_path.exists():
        data = load_yaml(yaml_path)
        items.extend(expand_config(data, base_dir))
    if py_path.exists():
        data = load_py_config(py_path)
        items.extend(expand_config(data, base_dir))
    return items

def dynamic_import(entry: str, base_dir: Path):
    module_name, class_name = entry.split(':', 1)
    spec = importlib.util.spec_from_file_location(module_name, base_dir / f"{module_name.replace('.', '/')}.py")
    if spec is None:
        raise ImportError(f"Cannot find module for {module_name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    cls = getattr(mod, class_name)
    return cls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--tool')
    parser.add_argument('--input')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    tools = load_all_configs(base_dir)
    if args.list:
        for t in tools:
            print(json.dumps({"id": t.get('id') or t.get('name'), "entry": t.get('entry')}))
        return

    if args.tool:
        t = next((x for x in tools if x.get('id') == args.tool or x.get('name') == args.tool), None)
        if not t:
            raise SystemExit(f"Tool {args.tool} not found")
        ToolCls = dynamic_import(t['entry'], base_dir)
        inst = ToolCls(t, t.get('params', {}))
        result = inst.predict({"inputs": {"input": args.input}})
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
