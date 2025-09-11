import yaml
import importlib
import inspect
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.applications import Starlette
from starlette.routing import Mount
import logging
import uvicorn
from fastmcp import FastMCP
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Load config file (tools.yaml)
def load_config(path="config/tools.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

tool_states = {}  # tool_name: True (enabled) or False (disabled)
registered_tools = {}  # tool_name: tool_info

# Auto-discover and register tools
def register_tools_from_config(mcp: FastMCP, config: dict):
    for tool_cfg in config.get("tools", []):
        module_name = tool_cfg["module"]
        type_name = tool_cfg.get("type", "general")  # Default to "general" if not specified
        include = tool_cfg.get("include", "all")
        exclude = set(tool_cfg.get("exclude", []))

        mod = importlib.import_module(module_name)

        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if include != "all" and name not in include:
                continue
            if name in exclude:
                continue

            # Register tool function
            mcp.tool()(func)
            tool_states[name] = True  # enabled by default
            
            # Get input schema from function signature with type constraints
            sig = inspect.signature(func)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                param_type = "string"
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
                properties[param_name] = {"type": param_type}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
            input_schema = {
                "type": "object",
                "properties": properties,
                "required": required
            }
            
            # Determine output type based on return annotation
            output_type = "string"
            if sig.return_annotation == int:
                output_type = "integer"
            elif sig.return_annotation == float:
                output_type = "number"
            elif sig.return_annotation == bool:
                output_type = "boolean"
            elif sig.return_annotation == list:
                output_type = "array"
            elif sig.return_annotation == dict:
                output_type = "object"
            
            registered_tools[name] = {
                "description": func.__doc__ or "",
                "input_schema": input_schema,
                "output_type": output_type,
                "type": type_name
            }
            logging.info(f"Registered tool: {module_name}.{name} with type: {type_name}")

# --- Main App ---
app = FastAPI()

# Custom route to list all tools and their states
@app.get("/list")
def list_tools():
    tools_list = []
    logging.info("Listing all registered tools")
    logging.info(f"Registered tools: {list(registered_tools.keys())}")
    for tool_name, tool_info in registered_tools.items():
        state = tool_states.get(tool_name, True)
        tools_list.append({
            "name": tool_name,
            "description": tool_info["description"],
            "inputs": tool_info["input_schema"],
            "outputs": {"type": tool_info["output_type"]},
            "enabled": state,
            "type": tool_info["type"]
        })
    return JSONResponse({"tools": tools_list})

# Enable a tool
@app.post("/enable/{tool_name}")
def enable_tool(tool_name: str):
    if tool_name in tool_states:
        tool_states[tool_name] = True
        return {"success": True, "tool": tool_name, "enabled": True}
    return {"success": False, "error": "Tool not found"}

# Disable a tool
@app.post("/disable/{tool_name}")
def disable_tool(tool_name: str):
    if tool_name in tool_states:
        tool_states[tool_name] = False
        return {"success": True, "tool": tool_name, "enabled": False}
    return {"success": False, "error": "Tool not found"}

# Serve a simple UI for tool management
@app.get("/ui", response_class=HTMLResponse)
async def tool_ui(request: Request):
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tool Management</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2em; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background: #f0f0f0; }
            .enabled { color: green; font-weight: bold; }
            .disabled { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>Tool Management</h2>
        <table id="tools-table">
            <thead>
                <tr><th>Name</th><th>Type</th><th>Description</th><th>Inputs</th><th>Outputs</th><th>State</th><th>Action</th></tr>
            </thead>
            <tbody></tbody>
        </table>
        <script>
        async function fetchTools() {
            const res = await fetch('/list');
            const data = await res.json();
            const tbody = document.querySelector('#tools-table tbody');
            tbody.innerHTML = '';
            data.tools.forEach(tool => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${tool.name}</td>
                    <td>${tool.type || ''}</td>
                    <td>${tool.description || ''}</td>
                    <td>${JSON.stringify(tool.inputs)}</td>
                    <td>${JSON.stringify(tool.outputs)}</td>
                    <td class="${tool.enabled ? 'enabled' : 'disabled'}">${tool.enabled ? 'Enabled' : 'Disabled'}</td>
                    <td>
                        <button onclick="toggleTool('${tool.name}', ${tool.enabled})">${tool.enabled ? 'Disable' : 'Enable'}</button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }
        async function toggleTool(name, enabled) {
            const url = enabled ? `/disable/${name}` : `/enable/${name}`;
            await fetch(url, {method: 'POST'});
            fetchTools();
        }
        fetchTools();
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html)

mcp = FastMCP()
mcp_app = mcp.http_app(transport="streamable-http")

config = load_config("config/tools.yaml")
register_tools_from_config(mcp, config)

routes = [
    *mcp_app.routes,
    *app.routes
]
app = FastAPI(
    routes=routes,
    lifespan=mcp_app.lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
