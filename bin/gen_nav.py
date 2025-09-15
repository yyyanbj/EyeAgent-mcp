#!/usr/bin/env python3
"""Generate a nav section for mkdocs.yml based on _docs_build folder.

Rules:
- Root index.md becomes Home
- First-level directories (eyeagent, eyetools, others) get capitalized group names
- Each directory: include index.md first (as Overview) then other *.md sorted alphabetically
- Writes output to stdout so CI can embed or to a temp file if --out provided
"""
from __future__ import annotations
import argparse
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "_docs_build"

TITLE_OVERRIDES = {
    'index.md': 'Overview',
}

def build_nav():
    if not BUILD.exists():
        raise SystemExit(f"Build directory {BUILD} does not exist; run prepare-docs.sh first.")
    nav = []
    # Root index
    root_index = BUILD / 'index.md'
    if root_index.exists():
        nav.append({'Home': 'index.md'})
    for sub in sorted(p for p in BUILD.iterdir() if p.is_dir()):
        group = sub.name.capitalize()
        items = []
        index_file = sub / 'index.md'
        if index_file.exists():
            items.append({'Overview': f"{sub.name}/index.md"})
        for md in sorted(sub.glob('*.md')):
            if md.name == 'index.md':
                continue
            title = TITLE_OVERRIDES.get(md.name) or md.stem.replace('_', ' ').title()
            items.append({title: f"{sub.name}/{md.name}"})
        if items:
            nav.append({group: items})
    return nav

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=pathlib.Path, help='Write YAML fragment to file')
    args = ap.parse_args()
    nav = build_nav()
    yaml_fragment = yaml.safe_dump(nav, sort_keys=False, allow_unicode=True)
    if args.out:
        args.out.write_text(yaml_fragment)
    else:
        print(yaml_fragment)

if __name__ == '__main__':
    main()
