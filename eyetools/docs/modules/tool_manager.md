# Tool Management (ToolManager)

Responsibilities:
- Dispatch prediction (inproc / subprocess)
- Instance cache + LRU eviction
- Lifecycle tracking (REGISTERED/LOADED/IDLE/UNLOADED)
- Metrics collection

Key logic:
- get_or_create_inproc: import entry, instantiate, optional prepare().
- predict: branch on runtime.load_mode; lazy model loading; update metrics.
- mark_idle / unload_idle: state transitions & memory release.

