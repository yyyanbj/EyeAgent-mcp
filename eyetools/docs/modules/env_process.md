# Environment & Process Management

## EnvManager
- Build env_key from (python_tag + extra_requires).
- Execute commands via `uv run --with deps --python=<tag>`.

## ProcessManager
- spawn: compose uv run (or system python) to start worker_entry.
- request: JSON line protocol over stdin/stdout.
- ensure_init: send INIT, track _init_done.
- cleanup_idle: terminate idle workers by last_used timestamp.

