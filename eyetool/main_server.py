import yaml
import importlib
import inspect
from fastapi import FastAPI
import logging
import uvicorn
from mcp.server.fastmcp import FastMCP

# Load config file (tools.yaml)
def load_config(path="config/tools.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Auto-discover and register tools
def register_tools_from_config(mcp: FastMCP, config: dict):
    for tool_cfg in config.get("tools", []):
        module_name = tool_cfg["module"]
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

            logging.info(f"Registered tool: {module_name}.{name}")

# --- Main App ---
app = FastAPI()
mcp = FastMCP("eyetool")

config = load_config("config/tools.yaml")
register_tools_from_config(mcp, config)


# Generate the MCP ASGI app
mcp_app = mcp.streamable_http_app()

# Mount the MCP app into FastAPI
app.mount("/mcp", mcp_app)

# Custom route to list all tools
@app.get("/list")
def list_tools():
    tools_list = []
    logging.info("Listing all registered tools")
    logging.info(f"Registered tools: {list(mcp.tools.keys())}")
    for tool_name, tool in mcp.tools.items():
        tools_list.append({
            "name": tool_name,
            "description": tool.description,
            "inputs": tool.input_schema
        })
    return JSONResponse({"tools": tools_list})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
