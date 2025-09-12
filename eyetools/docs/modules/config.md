# Config Management (config_loader)

Parsing logic:
1. Read YAML or Python (`config.yaml` / `config.py`); Python must expose `get_config()` or `CONFIGS`.
2. Determine mode: single / variants / tools.
3. For variants: merge `shared` + each variant (nested merge for runtime/model/warmup/io).
4. Generate ID priority: id -> package:variant -> package -> name -> entry class.
5. Apply defaults: runtime(load_mode=auto, queue_size, idle_timeout_s ...), model(precision=device=auto, lazy=True), etc.
6. Validate: entry format, precision, load_mode.

Output: normalized list[ToolDefinition] for loader registration.

