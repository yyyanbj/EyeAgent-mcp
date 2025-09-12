# Tool Registry (registry)

Core data structure `ToolMeta`:
- id, entry, version, runtime, model, params, tags, root_dir, extra_requires, python, category.

Operations:
- register(meta): reject duplicate id.
- get(id), list(), remove(id), clear().

Thread safety: internal RLock for mutation.

