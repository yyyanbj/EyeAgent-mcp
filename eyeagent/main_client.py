import asyncio
import json
import httpx
from fastmcp import FastMCP, Client

async def main():
    async with Client("http://localhost:8001/mcp/") as client:
        # Basic server interaction
        await client.ping()
        
        # List available operations
        tools = await client.list_tools()
        print("Available tools:")
        for t in tools:
            print("-", t.name, ":", t.description)
        

        resources = await client.list_resources()
        prompts = await client.list_prompts()

        # Example: call a tool
        # Replace 'calculate' with the name of your tool
        result = await client.call_tool(
            name="calculate",
            arguments={"expression": "2 + 3 * 4"}
        )
        print("Result of math_utils.calculate:", result)

if __name__ == "__main__":
    asyncio.run(main())
