# Testing & Quality

Current coverage:
- Discovery: test_discovery
- Classification placeholder functions: test_classification
- Metrics & errors: test_metrics_and_errors
- Role routing: test_role_router
- ToolManager inproc/subprocess: test_tool_manager_inproc / test_tool_manager_subprocess
- Added config/env/process/cli/server coverage.

Potential future additions:
- Lifecycle eviction tests (mark_idle / unload_idle)
- Worker error propagation & timeout path
- Coverage & mutation testing (pytest-cov, mutmut)

