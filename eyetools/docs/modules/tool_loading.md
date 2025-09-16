# Tool Loading & Import Path Resolution

This document explains how the `ToolManager` dynamically imports tool implementations and how import paths are stabilized to avoid intermittent `ImportError` issues (e.g. `attempted relative import with no known parent package`).

## Overview
Tools are registered with metadata containing:
- `id`: unique tool id (e.g. `classification:cfp_quality`)
- `entry`: string of the form `module_name:ClassName` (e.g. `tool_impl:ClassificationTool`)
- `root_dir`: filesystem directory where the tool implementation file(s) live

When a tool is first needed (preload or on-demand), `ToolManager._import_entry` resolves the module + class and instantiates the tool.

## Previous Fragility
Originally the manager tried to `import_module(module_name)` first and only inserted `root_dir` into `sys.path` if that failed. In some runtime contexts (different CWD, packaged invocation, parallel preload), Python could treat the module as a top-level script without its package context, breaking relative imports such as:
```python
from .class_lists import TASK_CLASS_MAP
```
This manifested as:
```
ImportError: attempted relative import with no known parent package
```
followed by a failed fallback absolute import.

## Current Strategy (Stabilized)
Before importing the module, `_import_entry` now always:
1. Inserts the tool's `root_dir` near the front of `sys.path` (index 0 or 1) if not already present.
2. Emits a DEBUG log line of the form:
   ```
   [import] inserted tool root into sys.path index=1 root=/abs/path/to/tool module=tool_impl
   ```
   or, if already present:
   ```
   [import] tool root already in sys.path root=/abs/path/to/tool module=tool_impl
   ```
3. Calls `import_module(module_name)`.
4. If the class is not found (name collision or ambiguous `tool_impl.py`), it attempts a direct file-based import as a fallback.

This deterministic insertion ensures relative imports inside the tool module reliably resolve.

## Security & Ordering Notes
- The path is inserted *after* an empty string (`''`) entry if present (Python's indicator for the current working directory) to reduce the chance of shadowing first-party libraries.
- No existing entries are removed; the operation is idempotent.
- Tools should avoid using very generic module names to minimize collision risk (`tool_impl` is acceptable but unique names are better for large ecosystems).

## Debugging Tips
Enable DEBUG logging for the `eyetools.core` logger to see import resolution messages:
```bash
export EYETOOLS_LOG_LEVEL=DEBUG
uv run eyetools-mcp serve --log-dir logs --lifecycle-mode eager --host 0.0.0.0 --port 8000 --tools-dir tools
```
(Adjust environment variable name if your logging configuration differs.)

If you still encounter an `ImportError`:
1. Confirm the tool directory contains an `__init__.py` if you intend it to be a package.
2. Check that no other earlier `sys.path` entry contains a conflicting module filename.
3. Inspect the DEBUG log lines for which path was inserted and in what order.

## Recommendations for Tool Authors
- Provide unique `module_name` values when possible (e.g. `retfound_cls_tool` instead of a generic `tool_impl`).
- Keep relative imports (`from .submodule import X`) for intra-tool code; they remain valid across relocations.
- Avoid side effects at import time that depend on external services; defer heavy initialization to `ensure_model_loaded()` or an explicit `prepare()` method.

## Future Improvements
Potential enhancements under consideration:
- Optional hashing of tool roots to construct synthetic, collision-resistant module namespaces.
- Warn (INFO) when multiple different roots provide the same `module_name`.
- Provide a CLI flag to dump final `sys.path` after tool discovery.

---
Last updated: 2025-09-16
