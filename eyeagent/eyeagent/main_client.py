import asyncio
import json
import httpx
from fastmcp import FastMCP, Client

async def main():
    async with Client("http://localhost:8000/mcp/") as client:
        # Basic server interaction
        await client.ping()
        
        # List available operations
        tools = await client.list_tools()
        print("Available tools:")
        for t in tools:
            print("-", t.name, ":", t.description)
        

        resources = await client.list_resources()
        print("Available resources:")
        for r in resources:
            print("-", r.name, ":", r.description)

        prompts = await client.list_prompts()
        print("Available prompts:")
        for p in prompts:
            print("-", p.name, ":", p.description)

if __name__ == "__main__":
    asyncio.run(main())
