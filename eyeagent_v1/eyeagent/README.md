EyeAgent Prototype (sync)

Features preserved:
- Inheritance-based agents with a registry and an orchestrator agent.
- Config-driven pipeline (YAML/JSON), expands *_path to inline text.
- No async.
- UUID case_id and trace logging to cases/<case_id>/ (JSON/JSONL).
- UI (Gradio) to load config, edit prompts, run, and playback traces.

Quick start
- CLI: python -m eyeagent.cli --config eyeagent/config/example.yaml
- UI:  python -m eyeagent.ui.app

Config shape
- runtime.cases_dir (optional): where traces are stored
- input: seed text
- orchestrator.pipeline: list of steps, each { agent, config }
  - sample agents: knowledge, report

Extend
- Create a new agent class inheriting Agent and decorate with @register_agent("name").
- Reference it in orchestrator.pipeline with its name.
