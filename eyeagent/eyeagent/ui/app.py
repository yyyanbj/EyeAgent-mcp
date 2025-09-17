import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, AsyncGenerator
import html

import gradio as gr

from eyeagent.diagnostic_workflow import run_diagnosis_async
from eyeagent.tracing.trace_logger import TraceLogger
from eyeagent.config.prompts import PromptsConfig
from eyeagent.config.tools_description import ToolsDescriptionRegistry
from eyeagent.tools.tool_registry import current_server_tools


# Force light mode using the provided snippet semantics, scoped to avoid globals
JS_FORCE_LIGHT = """
function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'light') {
                url.searchParams.set('__theme', 'light');
                window.location.href = url.href;
        }
}
"""


def _find_cases_dir() -> Path:
    env_cases = os.getenv("EYEAGENT_CASES_DIR")
    env_data = os.getenv("EYEAGENT_DATA_DIR")
    if env_cases:
        return Path(env_cases)
    if env_data:
        return Path(env_data) / "cases"
    t = TraceLogger()
    return Path(t.base_dir)


def _agent_icon(agent: str, role: str) -> str:
    r = (role or "").lower()
    if "orchestrator" in r:
        return "🧭"
    if "image" in r:
        return "🔍"
    if "specialist" in r:
        return "🩺"
    if "follow" in r:
        return "📌"
    if "report" in r:
        return "📄"
    return "🤖"


def _wrap_with_avatar(agent: str, role: str, inner_html: str) -> str:
    icon = _agent_icon(agent, role)
    name = html.escape(agent)
    role_txt = html.escape(role) if role else ""
    return (
        "<div style='display:flex; gap:12px; align-items:flex-start;'>"
        "  <div style='width:64px; flex:0 0 64px; text-align:center;'>"
        "    <div style='width:64px;height:64px;border-radius:50%;background:#e0e7ff;display:flex;align-items:center;justify-content:center;font-size:28px;'>"
        f"      {icon}"
        "    </div>"
        f"    <div style='font-size:12px;color:#5f6368;margin-top:6px;line-height:1.2;word-break:break-word'>{name}{(' ('+role_txt+')') if role_txt else ''}</div>"
        "  </div>"
        f"  <div style='flex:1'>{inner_html}</div>"
        "</div>"
    )


def _resolve_media_path(p: str) -> Optional[str]:
    if not p:
        return None
    cand = Path(p)
    if cand.is_absolute() and cand.exists():
        return str(cand)
    # try relative to CWD
    cwd = Path.cwd() / p
    if cwd.exists():
        return str(cwd)
    # try under repo root (two common roots)
    root = Path.cwd().parent if (Path.cwd().name == "eyeagent") else Path.cwd()
    alt1 = root / p
    if alt1.exists():
        return str(alt1)
    # try under eyetools subdir
    alt2 = root / "eyetools" / p
    if alt2.exists():
        return str(alt2)
    return None


def _format_agent_bubble(agent: str, role: str, reasoning: Optional[str], outputs_preview: Optional[str] = None) -> str:
    title = f"{html.escape(agent)} ({html.escape(role)})" if role else html.escape(agent)
    body = html.escape(reasoning) if reasoning else ""
    details = (
        f"<details style='margin-top:6px;'><summary style='cursor:pointer;'>Agent outputs</summary>"
        f"<div style='margin-top:6px'>{outputs_preview}</div></details>"
    ) if outputs_preview else ""
    box = (
        "<div style='background:#ffffff; border:1px solid #e0e0e0; border-left:4px solid #7cb342; padding:10px;"
        " margin:6px 0; border-radius:6px;'>"
        f"<div style='font-weight:600;color:#1b5e20; margin-bottom:6px;'>{title}</div>"
        f"<div style='color:#111; white-space: pre-wrap;font-size:0.95em'>{body}</div>"
        f"{details}"
        "</div>"
    )
    return _wrap_with_avatar(agent, role, box)


def _format_tool_bubble(tool_id: str, status: str, arguments: Dict[str, Any], output_preview: Optional[str] = None, caller_agent: Optional[str] = None, caller_role: Optional[str] = None, images_html: Optional[str] = None) -> str:
    arg_text = html.escape(str(arguments)) if arguments else "{}"
    header = f"🔧 {html.escape(tool_id)}"
    sub = f"<div style='color:#455a64; font-size:0.9em;'>by {html.escape(caller_agent or '')} {f'({html.escape(caller_role or "")})' if caller_role else ''}</div>"
    status_txt = f"Status: {html.escape(status)}"
    prev = (
        f"<details style='margin-top:6px;'><summary style='cursor:pointer;font-size:0.9em'>Output summary</summary>"
        f"<div style='margin-top:6px;color:#37474f;font-size:0.9em'>{output_preview}</div></details>"
    ) if output_preview else ""
    box = (
        "<div style='background:#ffffff; border:1px solid #e0e0e0; border-left:4px solid #1e88e5; padding:10px;"
        " margin:6px 0; border-radius:6px;'>"
        f"<div style='font-weight:600;color:#0d47a1; margin-bottom:2px;font-size:0.95em'>{header}</div>"
        f"{sub}"
        f"<div style='color:#0d47a1; margin-top:4px;font-size:0.9em'>📋 Args: <code style='background:#f5f5f5;padding:2px 4px;border-radius:3px;'>{arg_text}</code></div>"
        f"<div style='color:#1565c0; margin-top:4px;font-size:0.9em'>{status_txt}</div>"
        f"{prev}"
        f"{images_html or ''}"
        "</div>"
    )
    return _wrap_with_avatar(caller_agent or "Tool", caller_role or "tool", box)


def _tool_output_images(output: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    if not isinstance(output, dict):
        return paths
    # common schema used by segmentation tools
    op = output.get("output_paths") if isinstance(output.get("output_paths"), dict) else None
    if op:
        for key in ("overlay", "merged", "colorized"):
            p = op.get(key)
            rp = _resolve_media_path(p) if isinstance(p, str) else None
            if rp:
                paths.append(rp)
    # allow generic 'image' or 'images'
    img = output.get("image")
    if isinstance(img, str):
        rp = _resolve_media_path(img)
        if rp:
            paths.append(rp)
    imgs = output.get("images")
    if isinstance(imgs, list):
        for i in imgs:
            if isinstance(i, str):
                rp = _resolve_media_path(i)
                if rp:
                    paths.append(rp)
    return paths


def _summarize_tool_output(output: Any, max_items: int = 5) -> Optional[str]:
    if not isinstance(output, dict):
        return None
    parts: List[str] = []
    task = output.get("task")
    if isinstance(task, str):
        parts.append(f"<div><strong>Task:</strong> {html.escape(task)}</div>")
    # Optional runtime summary
    if output.get("inference_time") is not None:
        parts.append(f"<div><strong>Inference time:</strong> {html.escape(str(output.get('inference_time')))}s</div>")
    if isinstance(output.get("message"), str):
        parts.append(f"<div><em>{html.escape(str(output.get('message')))}</em></div>")
    # classification style
    preds = output.get("predictions")
    probs = output.get("probabilities")
    if isinstance(preds, list) and probs and isinstance(probs, dict):
        items = []
        for p in preds:
            if p in probs:
                items.append((p, probs.get(p)))
        # sort by prob desc if numeric
        try:
            items.sort(key=lambda x: float(x[1] if x[1] is not None else 0), reverse=True)
        except Exception:
            pass
        items = items[:max_items]
        if items:
            li = "".join([f"<li>{html.escape(k)}: <code>{html.escape(str(v))}</code></li>" for k, v in items])
            parts.append(f"<div><strong>Top predictions:</strong><ul>{li}</ul></div>")
    # segmentation style counts
    counts = output.get("counts")
    if isinstance(counts, dict) and counts:
        li = "".join([f"<li>{html.escape(str(k))}: <code>{html.escape(str(v))}</code></li>" for k, v in list(counts.items())[:max_items]])
        parts.append(f"<div><strong>Counts:</strong><ul>{li}</ul></div>")
    # areas summary
    areas = output.get("areas")
    if isinstance(areas, dict) and areas:
        keys = list(areas.keys())[:max_items]
        li = "".join([f"<li>{html.escape(str(k))}: <code>{len(areas.get(k) or [])}</code> regions</li>" for k in keys])
        parts.append(f"<div><strong>Areas:</strong><ul>{li}</ul></div>")
    if not parts:
        return None
    return "".join(parts)


def _summarize_agent_outputs(outputs: Any, max_items: int = 5) -> Optional[str]:
    if not isinstance(outputs, dict):
        return None
    parts: List[str] = []
    # quality
    q = outputs.get("quality")
    if isinstance(q, dict):
        parts.append(f"<div><strong>Quality:</strong> {html.escape(str(q))}</div>")
    # diseases probabilities
    d = outputs.get("diseases")
    if isinstance(d, dict) and d:
        items = list(d.items())
        try:
            items.sort(key=lambda x: float(x[1] if x[1] is not None else 0), reverse=True)
        except Exception:
            pass
        items = items[:max_items]
        li = "".join([f"<li>{html.escape(str(k))}: <code>{html.escape(str(v))}</code></li>" for k, v in items])
        parts.append(f"<div><strong>Diseases:</strong><ul>{li}</ul></div>")
    # lesions keys only (seg tasks present)
    l = outputs.get("lesions")
    if isinstance(l, dict) and l:
        keys = list(l.keys())[:max_items]
        parts.append(f"<div><strong>Lesion tasks:</strong> {html.escape(', '.join(keys))}</div>")
    # specialist/follow_up/report fragments
    dg = outputs.get("disease_grades")
    if isinstance(dg, list) and dg:
        parts.append(f"<div><strong>Grades:</strong> {html.escape(str(dg[:max_items]))}</div>")
    mgmt = outputs.get("management")
    if mgmt:
        parts.append(f"<div><strong>Management:</strong> {html.escape(str(mgmt))}</div>")
    diag = outputs.get("diagnoses")
    if diag:
        parts.append(f"<div><strong>Diagnoses:</strong> {html.escape(str(diag))}</div>")
    if not parts:
        return None
    return "".join(parts)


def _image_to_data_uri(p: str) -> Optional[str]:
    try:
        import base64
        ext = Path(p).suffix.lower()
        mime = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png'
        with open(p, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _embed_images_html(paths: List[str], max_images: int = 4, thumb_w: int = 240) -> Optional[str]:
    if not paths:
        return None
    thumbs: List[str] = []
    for p in paths[:max_images]:
        uri = _image_to_data_uri(p)
        if uri:
            thumbs.append(
                f"<img src='{uri}' style='width:{thumb_w}px;max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;' />"
            )
    if not thumbs:
        return None
    grid = "".join([f"<div>{t}</div>" for t in thumbs])
    return (
        "<div style='margin-top:8px'>"
        "  <div style='display:grid;grid-template-columns:repeat(auto-fill, minmax(160px, 1fr));gap:8px;'>"
        f"    {grid}"
        "  </div>"
        "</div>"
    )


async def _run_and_stream(patient: Dict[str, Any], image_paths: List[str], progress: gr.Progress, chatbot: gr.Chatbot, initial_messages: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[Tuple[List[Dict[str, str]], Dict[str, Any] | str | None], None]:
    trace = TraceLogger()
    images = [{"image_id": Path(p).stem, "path": p} for p in image_paths]
    case_id = trace.create_case(patient=patient, images=images)

    # messages format: list of {role, content}
    messages: List[Dict[str, str]] = []
    # Show uploaded images as thumbnails at the top
    if image_paths:
        thumbs = []
        for p in image_paths:
            try:
                import base64
                ext = Path(p).suffix.lower()
                mime = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png'
                with open(p, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('ascii')
                uri = f"data:{mime};base64,{b64}"
                thumbs.append(f"<img src='{uri}' style='width:120px;max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;margin-right:8px;' />")
            except Exception:
                pass
        if thumbs:
            messages.append({"role": "system", "content": "<div style='margin-bottom:8px'><strong>Uploaded images:</strong><br>" + ''.join(thumbs) + "</div>"})
    if initial_messages:
        # Only keep safe roles and string content
        for m in initial_messages:
            r = m.get("role")
            c = m.get("content")
            if r in ("user", "system") and isinstance(c, str):
                messages.append({"role": r, "content": c})
    messages.append({"role": "system", "content": f"Started case {case_id}. Running diagnosis..."})
    yield messages, None

    task = asyncio.create_task(run_diagnosis_async(patient, images, trace=trace, case_id=case_id))

    trace_path = Path(trace._trace_path(case_id))
    last_count = 0
    while not task.done():
        await asyncio.sleep(0.5)
        if trace_path.exists():
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                events = doc.get("events", [])
                # stream only new events
                new_events = events[last_count:]
                last_count = len(events)
                msgs: List[Dict[str, str]] = []
                for ev in new_events:
                    et = ev.get("type")
                    if et == "tool_call":
                        tool = ev.get("tool_id")
                        status = ev.get("status")
                        args = ev.get("arguments", {})
                        output = ev.get("output")
                        imgs = _tool_output_images(output or {})
                        gallery_html = _embed_images_html(imgs) if imgs else None
                        bubble = _format_tool_bubble(tool or "tool", status or "", args, _summarize_tool_output(output), ev.get("agent"), ev.get("role"), images_html=gallery_html)
                        msgs.append({"role": "assistant", "content": bubble})
                        print(f"[tool_call] {ev.get('agent')} ({ev.get('role')}): {tool} -> {status}")
                    elif et == "agent_step":
                        agent = ev.get("agent") or "Agent"
                        role = ev.get("role") or ""
                        reasoning = ev.get("reasoning")
                        outputs_prev = _summarize_agent_outputs(ev.get("outputs"))
                        msgs.append({"role": "assistant", "content": _format_agent_bubble(agent, role, reasoning, outputs_prev)})
                        print(f"[agent_step] {agent} ({role})")
                    elif et == "error":
                        msgs.append({"role": "assistant", "content": f"[error] {ev.get('message', 'Error')}"})
                        print(f"[error] {ev.get('message')}")
                if msgs:
                    messages.extend(msgs)
                    yield messages, None
            except Exception:
                pass

    result = await task
    final = result or {}
    # Append a human-readable final report bubble in chat
    fr = final.get("final_report", {}) if isinstance(final, dict) else {}
    diag = fr.get("diagnoses")
    mgmt = fr.get("management")
    reason = fr.get("reasoning")
    narrative = fr.get("narrative") or reason
    conclusion = fr.get("conclusion")
    summary_lines = []
    if conclusion:
        summary_lines.append(f"<div style='margin-bottom:8px;line-height:1.5'><strong>Diagnostic conclusion:</strong> {html.escape(str(conclusion))}</div>")
    if narrative:
        summary_lines.append(f"<div style='margin-bottom:8px;line-height:1.5'><strong>Summary:</strong> {html.escape(str(narrative))}</div>")
    else:
        if diag:
            summary_lines.append(f"<div><strong>Diagnoses:</strong> {html.escape(str(diag))}</div>")
        if mgmt:
            summary_lines.append(f"<div><strong>Management:</strong> {html.escape(str(mgmt))}</div>")
    trace_ref = final.get("trace_log_path")
    if trace_ref:
        summary_lines.append(f"<div><small>Trace: <code>{html.escape(str(trace_ref))}</code></small></div>")
    messages.append({"role": "assistant", "content": "<div style='background:#fff;border:1px solid #e0e0e0;border-left:4px solid #6a1b9a;padding:10px;border-radius:6px;'>" + "".join(summary_lines) + "</div>"})
    messages.append({"role": "system", "content": "Diagnosis completed."})
    print("[final_report]", json.dumps(final, ensure_ascii=False)[:2000])
    yield messages, final


def _events_to_messages(events: List[Dict[str, Any]], case_id: Optional[str] = None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if case_id:
        msgs.append({"role": "system", "content": f"Loaded case {case_id}"})
    for ev in events:
        et = ev.get("type")
        if et == "tool_call":
            tool = ev.get("tool_id")
            status = ev.get("status")
            args = ev.get("arguments", {})
            output = ev.get("output")
            imgs = _tool_output_images(output or {})
            gallery_html = _embed_images_html(imgs) if imgs else None
            bubble = _format_tool_bubble(tool or 'tool', status or '', args, _summarize_tool_output(output), ev.get("agent"), ev.get("role"), images_html=gallery_html)
            msgs.append({"role": "assistant", "content": bubble})
        elif et == "agent_step":
            agent = ev.get("agent") or "Agent"
            role = ev.get("role") or ""
            reasoning = ev.get("reasoning")
            outputs_prev = _summarize_agent_outputs(ev.get("outputs"))
            msgs.append({"role": "assistant", "content": _format_agent_bubble(agent, role, reasoning, outputs_prev)})
        elif et == "error":
            msgs.append({"role": "assistant", "content": f"[error] {ev.get('message', 'Error')}"})
    return msgs


async def _continue_and_stream(case_id: str, additional_instruction: str, progress: gr.Progress) -> AsyncGenerator[Tuple[List[Dict[str, str]], Dict[str, Any] | None], None]:
    # Load existing case data
    cases_dir = _find_cases_dir()
    trace_path = cases_dir / case_id / "trace.json"
    if not trace_path.exists():
        yield ([{"role": "system", "content": f"Case {case_id} not found."}]), None
        return
    with open(trace_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    patient = doc.get("patient", {})
    images_meta = doc.get("images", [])
    image_paths = [itm.get("path") for itm in images_meta if itm.get("path")]
    # Merge additional instruction
    if additional_instruction:
        patient = {**patient, "instruction": additional_instruction}

    # Prepare streaming variables
    messages = _events_to_messages(doc.get("events", []), case_id=case_id)
    yield messages, None

    # Continue by invoking pipeline again, appending to same case
    trace = TraceLogger(base_dir=str(cases_dir))
    task = asyncio.create_task(run_diagnosis_async(patient, images_meta, trace=trace, case_id=case_id))
    last_count = len(doc.get("events", []))

    while not task.done():
        await asyncio.sleep(0.5)
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                cur = json.load(f)
            events = cur.get("events", [])
            if len(events) > last_count:
                new_msgs = _events_to_messages(events[last_count:])
                last_count = len(events)
                messages.extend(new_msgs)
                yield messages, ""
        except Exception:
            pass

    result = await task
    messages.append({"role": "system", "content": "Continuation completed."})
    yield messages, result


def build_interface() -> gr.Blocks:
    # Ensure a writable Gradio temp dir
    if not os.getenv("GRADIO_TEMP_DIR"):
        # default to <repo_root>/temp/gradio
        # use TraceLogger to discover repo root via its logic
        t = TraceLogger()
        repo_cases = Path(t.base_dir)
        repo_root = repo_cases.parent if repo_cases.name == "cases" else Path.cwd()
        temp_dir = repo_root / "temp" / "gradio"
        os.makedirs(temp_dir, exist_ok=True)
        os.environ["GRADIO_TEMP_DIR"] = str(temp_dir)

    with gr.Blocks(title="EyeAgent - Multi-Agent Diagnosis", theme=gr.themes.Soft(), js=JS_FORCE_LIGHT) as demo:
        gr.HTML("""
        <style>
        :root { color-scheme: light; }
        html, body { background: #ffffff !important; }
        body, .gradio-container {
            color: #111 !important;
            font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans",
                         "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol" !important;
        }
        .gradio-container .prose, .gradio-container p, .gradio-container span, .gradio-container div {
            color: inherit;
        }
        </style>
        """)
        gr.Markdown("# 👁️ EyeAgent - Multi-Agent Diagnosis UI")
        gr.Markdown("Upload images and provide optional instructions. You can replay previous cases.")

        with gr.Tab("Run Diagnosis"):
            with gr.Row():
                image_uploader = gr.File(label="Upload images", file_count="multiple", type="filepath")
                instruction = gr.Textbox(label="Instruction (optional)", placeholder="Describe specific goals or notes…")

            show_patient = gr.Checkbox(label="Show Patient Info", value=False)
            with gr.Accordion("Patient Info", open=False, visible=False) as patient_section:
                with gr.Row():
                    patient_id = gr.Textbox(label="Patient ID", value="UI-Patient")
                    patient_age = gr.Number(label="Age", value=60)
                    patient_gender = gr.Dropdown(choices=["M", "F", "Other"], value="M", label="Gender")

            run_btn = gr.Button("Run Diagnosis", variant="primary")
            chatbot = gr.Chatbot(label="Live Multi-Agent Trace", type="messages")
            # Simplified presets: checkbox group + auto-fill instruction
            preset_checks = gr.CheckboxGroup(label="Instruction Presets", choices=[], interactive=True)
            # Quick inline input to inject a line into the instruction/chat without using the dropdown
            quick_input = gr.Textbox(label="Quick Insert", placeholder="Type here and press Enter to insert into instruction & trace", lines=1)
            chat_state = gr.State([])  # holds current chat messages (list of {role, content})
            final_json = gr.JSON(label="Final Report")

            show_patient.change(lambda v: gr.update(visible=bool(v)), inputs=show_patient, outputs=patient_section)

            async def on_run(files: List[str], instr: str, show_p: bool, pid: str, age: float, gender: str, msgs_state: List[Dict[str, str]]):
                if show_p:
                    patient = {"patient_id": pid or "UI-Patient", "age": int(age or 0), "gender": gender or "Unknown"}
                else:
                    patient = {"patient_id": "UI-Patient"}
                if instr:
                    patient["instruction"] = instr
                file_paths = files or []
                progress = gr.Progress(track_tqdm=False)
                # Ensure user instruction is also shown as a user message on the right
                initial_msgs = list(msgs_state or [])
                if instr and not any((m.get("role") == "user" and m.get("content") == instr) for m in initial_msgs[-3:]):
                    initial_msgs.append({"role": "user", "content": instr})
                async for msgs, result in _run_and_stream(patient, file_paths, progress, chatbot, initial_messages=initial_msgs):
                    yield msgs, result, msgs

            def load_presets():
                cfg = PromptsConfig().get_ui_presets()
                presets = cfg.get("instruction_presets") or [
                    "Comprehensive ocular imaging screening and summary",
                    "Focus on diabetic retinopathy grading",
                    "Check AMD findings and follow-up suggestion"
                ]
                default_instr = cfg.get("default_instruction") or ""
                return gr.update(choices=presets), default_instr

            def on_presets_change(cur_text: str, selected: List[str], msgs: List[Dict[str, str]]):
                # Auto-fill instruction with selected presets joined by newlines
                selected = selected or []
                new_text = "\n".join(selected)
                new_msgs = list(msgs or [])
                # Append a single user message summarizing the selected presets
                if selected:
                    new_msgs.append({"role": "user", "content": new_text})
                return new_text, new_msgs

            def quick_insert(cur_text: str, quick_text: str, msgs: List[Dict[str, str]]):
                # Reuse the same logic as use_preset but with quick_text
                if not quick_text:
                    return cur_text, msgs, msgs, ""
                if not cur_text:
                    new_instr = quick_text
                else:
                    new_instr = (cur_text + ("\n" if not cur_text.endswith("\n") else "") + quick_text)[:4000]
                new_msgs = list(msgs or [])
                new_msgs.append({"role": "user", "content": quick_text})
                return new_instr, new_msgs, new_msgs, ""

            demo.load(load_presets, inputs=None, outputs=[preset_checks, instruction])
            preset_checks.change(on_presets_change, inputs=[instruction, preset_checks, chat_state], outputs=[instruction, chatbot])
            quick_input.submit(quick_insert, inputs=[instruction, quick_input, chat_state], outputs=[instruction, chatbot, chat_state, quick_input])

            run_btn.click(on_run, inputs=[image_uploader, instruction, show_patient, patient_id, patient_age, patient_gender, chat_state], outputs=[chatbot, final_json, chat_state])

        with gr.Tab("Replay Cases"):
            cases_dir = gr.Textbox(label="Cases Directory", value=str(_find_cases_dir()), interactive=True)
            refresh_btn = gr.Button("Refresh Case List")
            # Pre-populate initial case choices at build time
            try:
                _initial_cases = [d.name for d in Path(str(_find_cases_dir())).iterdir() if d.is_dir()]
            except Exception:
                _initial_cases = []
            case_list = gr.Dropdown(label="Select Case", choices=_initial_cases, interactive=True)
            load_btn = gr.Button("Load Case")
            replay_chatbot = gr.Chatbot(label="Case Trace", type="messages")
            continue_instr = gr.Textbox(label="Additional Instruction (optional)")
            continue_btn = gr.Button("Continue & Run", variant="primary")
            with gr.Accordion("Raw JSON", open=False):
                trace_json = gr.JSON(label="Trace JSON")
                final_json2 = gr.JSON(label="Final Report JSON")

            def list_cases(dir_path: str):
                p = Path(dir_path)
                if not p.exists():
                    return []
                return [d.name for d in p.iterdir() if d.is_dir()]

            def do_refresh(dir_path: str):
                return gr.update(choices=list_cases(dir_path))

            def load_case(dir_path: str, cid: str):
                if not cid:
                    return [], None, None
                trace_path = Path(dir_path) / cid / "trace.json"
                final_path = Path(dir_path) / cid / "final_report.json"
                trace = json.load(open(trace_path, "r", encoding="utf-8")) if trace_path.exists() else {"events": []}
                final = json.load(open(final_path, "r", encoding="utf-8")) if final_path.exists() else None
                msgs = _events_to_messages(trace.get("events", []), case_id=cid)
                return msgs, trace, final

            refresh_btn.click(do_refresh, inputs=[cases_dir], outputs=[case_list])
            load_btn.click(load_case, inputs=[cases_dir, case_list], outputs=[replay_chatbot, trace_json, final_json2])

            async def on_continue(dir_path: str, cid: str, instr: str):
                if not cid:
                    yield [], None
                    return
                progress = gr.Progress(track_tqdm=False)
                async for msgs, result in _continue_and_stream(cid, instr, progress):
                    yield msgs, result

            continue_btn.click(on_continue, inputs=[cases_dir, case_list, continue_instr], outputs=[replay_chatbot, final_json2])

        with gr.Tab("Settings"):
            gr.Markdown("### Prompts Configuration")
            cfg_dir = gr.Textbox(label="Config Directory (EYEAGENT_CONFIG_DIR)", value="", placeholder="Leave blank to use repo_root/config", interactive=True)
            with gr.Row():
                agent_sel = gr.Dropdown(label="Agent", choices=[
                    "OrchestratorAgent", "ImageAnalysisAgent", "SpecialistAgent", "FollowUpAgent", "ReportAgent"
                ], value="OrchestratorAgent")
                load_sp_btn = gr.Button("Load System Prompt")
                save_sp_btn = gr.Button("Save System Prompt")
            sys_prompt_box = gr.Textbox(label="System Prompt", lines=10)

            gr.Markdown("### UI Instruction Presets")
            presets_box = gr.Textbox(label="Instruction Presets (one per line)", lines=6)
            default_instr_box = gr.Textbox(label="Default Instruction", lines=2)
            with gr.Row():
                load_ui_btn = gr.Button("Load UI Presets")
                save_ui_btn = gr.Button("Save UI Presets")

            def _cfg(base: str | None) -> PromptsConfig:
                return PromptsConfig(base_dir=base) if base else PromptsConfig()

            def load_sp(base: str, agent: str):
                return _cfg(base).get_system_prompt(agent)

            def save_sp(base: str, agent: str, text: str):
                pc = _cfg(base)
                pc.set_system_prompt(agent, text or "")
                return gr.update()

            def load_ui(base: str):
                data = _cfg(base).get_ui_presets()
                presets = data.get("instruction_presets") or []
                default = data.get("default_instruction") or ""
                return "\n".join(presets), default

            def save_ui(base: str, presets_text: str, default_text: str):
                presets = [ln.strip() for ln in (presets_text or "").splitlines() if ln.strip()]
                _cfg(base).set_ui_presets({
                    "instruction_presets": presets,
                    "default_instruction": default_text or ""
                })
                return gr.update()

            load_sp_btn.click(load_sp, inputs=[cfg_dir, agent_sel], outputs=[sys_prompt_box])
            save_sp_btn.click(save_sp, inputs=[cfg_dir, agent_sel, sys_prompt_box], outputs=[])
            load_ui_btn.click(load_ui, inputs=[cfg_dir], outputs=[presets_box, default_instr_box])
            save_ui_btn.click(save_ui, inputs=[cfg_dir, presets_box, default_instr_box], outputs=[])

            gr.Markdown("### Tools Description Registry")
            with gr.Row():
                sync_btn = gr.Button("Sync Tool Descriptions", variant="primary")
                load_td_btn = gr.Button("Load Local Descriptions")
            sync_status = gr.Markdown(value="")
            local_td_json = gr.JSON(label="Local tools_descriptions.json")

            def _tdr(base: str | None) -> ToolsDescriptionRegistry:
                return ToolsDescriptionRegistry(base_dir=base) if base else ToolsDescriptionRegistry()

            def load_local_td(base: str):
                reg = _tdr(base)
                return reg.load()

            def do_sync(base: str):
                server = current_server_tools()
                reg = _tdr(base)
                before = reg.load()
                merged = reg.sync_from_server(server)
                added = len(set(merged.keys()) - set(before.keys()))
                updated = sum(1 for k in merged.keys() if k in before and merged.get(k) != before.get(k))
                status = f"Synced tool descriptions. Total: {len(merged)} | Added: {added} | Updated: {updated}"
                return status, merged

            load_td_btn.click(load_local_td, inputs=[cfg_dir], outputs=[local_td_json])
            sync_btn.click(do_sync, inputs=[cfg_dir], outputs=[sync_status, local_td_json])

        # Auto-populate recently cases list on app load
        demo.load(lambda p: gr.update(choices=[d.name for d in Path(p).iterdir() if d.is_dir()] if Path(p).exists() else []), inputs=[cases_dir], outputs=[case_list])

    return demo


def main():
    demo = build_interface()
    # determine repository root to allow serving media files generated outside CWD
    t = TraceLogger()
    repo_cases = Path(t.base_dir)
    repo_root = repo_cases.parent if repo_cases.name == "cases" else Path.cwd()
    allowed = {str(repo_root)}
    if os.getenv("GRADIO_TEMP_DIR"):
        allowed.add(os.getenv("GRADIO_TEMP_DIR"))
    allowed.add("/tmp")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("EYEAGENT_UI_PORT", "7860")),
        allowed_paths=list(allowed),
    )


if __name__ == "__main__":
    main()
