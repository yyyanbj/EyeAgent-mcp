from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

import gradio as gr

from ..core.config_loader import load_config
from ..core.trace_logger import TraceLogger
from ..core.registry import create_agent  # orchestrator must be imported so it's registered
from ..agents import orchestrator_agent  # noqa: F401 ensure registration


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_dir = ((cfg.get("runtime") or {}).get("cases_dir"))
    trace = TraceLogger(base_dir=base_dir)
    case_id = trace.create_case(patient=cfg.get("patient") or {}, images=cfg.get("images") or [])
    ctx: Dict[str, Any] = {
        "case_id": case_id,
        "input": (cfg.get("input") or ""),
        "config_inline": cfg,
    }
    orch_cfg = (cfg.get("orchestrator") or {})
    orch = create_agent("orchestrator", config=orch_cfg, trace=trace)
    ctx = orch.run(ctx)
    return {"case_id": case_id, "final_report": ctx.get("final_report")}


def app():
    trace = TraceLogger()  # for listing cases only

    with gr.Blocks(title="EyeAgent Prototype") as demo:
        gr.Markdown("# EyeAgent Prototype\n- Load a config (YAML/JSON)\n- Edit prompts before run\n- Playback traces by case_id")

        with gr.Tabs():
            with gr.TabItem("Run"):
                cfg_file = gr.File(label="Config file (YAML/JSON)")
                prompt_box = gr.Textbox(label="Override system prompt (optional)", lines=6)
                run_btn = gr.Button("Run")
                case_out = gr.Textbox(label="case_id")
                final_out = gr.JSON(label="final_report")

                def on_run(file, prompt_text):
                    if not file:
                        return "", {}
                    cfg = load_config(file.name)
                    # allow editing prompt via UI: orchestrator.pipeline[0].config.prompts.system
                    try:
                        pipeline = (cfg.setdefault("orchestrator", {}).setdefault("pipeline", []))
                        if pipeline:
                            step0 = pipeline[0].setdefault("config", {}).setdefault("prompts", {})
                            if prompt_text:
                                step0["system"] = prompt_text
                    except Exception:
                        pass
                    result = run_pipeline(cfg)
                    return result.get("case_id"), result.get("final_report")

                run_btn.click(on_run, inputs=[cfg_file, prompt_box], outputs=[case_out, final_out])

            with gr.TabItem("Playback"):
                base_dir_box = gr.Textbox(label="Cases dir (optional; default env or repo cases)")
                refresh_btn = gr.Button("Refresh cases")
                cases_dd = gr.Dropdown(choices=[], label="case_id")
                load_btn = gr.Button("Load trace")
                trace_view = gr.JSON(label="trace.json")

                def on_refresh(base_dir):
                    t = TraceLogger(base_dir=base_dir or None)
                    cases = t.list_cases()
                    return gr.Dropdown(choices=sorted(cases), value=(cases[0] if cases else None))

                def on_load(case_id, base_dir):
                    if not case_id:
                        return {}
                    t = TraceLogger(base_dir=base_dir or None)
                    return t.load_trace(case_id)

                refresh_btn.click(on_refresh, inputs=[base_dir_box], outputs=[cases_dd])
                load_btn.click(on_load, inputs=[cases_dd, base_dir_box], outputs=[trace_view])

    return demo


if __name__ == "__main__":
    app().launch(server_name="0.0.0.0", server_port=5787)
