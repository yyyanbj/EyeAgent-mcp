import json
from mcp.client import MCPClient

# Connect to your MCP server
client = MCPClient("http://localhost:8001/mcp")

# List all tools
tools = client.list_tools()
print("Available tools:")
for t in tools:
    print("-", t.name, ":", t.description)

# Example: call a tool
# Replace 'math_utils.calculate' with the name of your tool
result = client.call_tool(
    tool_name="math_utils.calculate",
    arguments={"expression": "2 + 3 * 4"}
)
print("Result of math_utils.calculate:", result)
