from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import gradio as gr
import asyncio
import os
from dotenv import load_dotenv
from fastmcp import Client
import json
from loguru import logger

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),  # Print to stdout without extra formatting
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)

# Load environment variables
_ = load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    max_tokens=64000,
    temperature=1.0,
)

# MCP Server URL (env override, default 8000)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")

# Define tool mappings for each agent role
AGENT_TOOLS = {
    "researcher": ["calculate", "multiply", "sum_numbers", "get_weather"],
    "writer": ["generate_image", "classify_image"],
    "analyst": ["segment_image", "detect_objects", "classify_image"],
    "coordinator": []  # Coordinator doesn't use tools directly
}

# Define the state for the multi-agent system
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    available_tools: list
    mcp_client: Client

# Define agents
async def coordinator_agent(state: AgentState):
    """Coordinator agent that decides which agent to use next."""
    logger.info("🤖 [COORDINATOR] Starting coordination process...")
    messages = state["messages"]

    # Get available tools from MCP
    try:
        tools = await state["mcp_client"].list_tools()
        tool_names = [tool.name for tool in tools]
        logger.info(f"📋 [COORDINATOR] Available MCP tools: {', '.join(tool_names)}")
    except Exception as e:
        tool_names = []
        logger.error(f"❌ [COORDINATOR] Error listing tools: {e}")

    # Use LLM to decide which agent and tools to use
    logger.info(f"💭 [COORDINATOR] Analyzing user query: {messages[-1].content}")
    response = llm.invoke([
        {"role": "system", "content": f"You are a coordinator. Available tools: {', '.join(tool_names)}. Decide which agent should handle the next task: 'researcher' for information gathering, 'writer' for content creation, 'analyst' for data analysis. Respond with just the agent name."},
        *messages
    ])
    agent_choice = response.content.strip().lower()
    logger.info(f"🎯 [COORDINATOR] Selected agent: {agent_choice}")
    logger.debug(f"📝 [COORDINATOR] Coordinator response: {response.content}")
    return {"current_agent": agent_choice}

async def researcher_agent(state: AgentState):
    """Research agent for gathering information."""
    logger.info("🔍 [RESEARCHER] Starting research process...")
    messages = state["messages"]

    # Get available tools and filter by role
    try:
        all_tools = await state["mcp_client"].list_tools()
        # Filter tools based on researcher's allowed tools
        allowed_tools = AGENT_TOOLS["researcher"]
        tools = [tool for tool in all_tools if tool.name in allowed_tools]
        tool_info = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        logger.info(f"🛠️ [RESEARCHER] Available tools for researcher: {', '.join([t.name for t in tools])}")
    except Exception as e:
        tool_info = "No tools available"
        logger.error(f"❌ [RESEARCHER] Error listing tools: {e}")

    # Use LLM to decide which tool to use
    logger.info(f"💭 [RESEARCHER] Analyzing query for research: {messages[-1].content}")
    response = llm.invoke([
        {"role": "system", "content": f"You are a research agent. Available tools:\n{tool_info}\n\nChoose an appropriate tool to gather information and provide the tool name and arguments in JSON format: {{\"tool_name\": \"name\", \"arguments\": {{}}}}"},
        *messages
    ])

    logger.debug(f"📝 [RESEARCHER] Researcher LLM response: {response.content}")

    try:
        tool_call = json.loads(response.content)
        tool_name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments", {})

        logger.info(f"🔧 [RESEARCHER] Calling tool: {tool_name} with args: {arguments}")

        # Call the MCP tool
        result = await state["mcp_client"].call_tool(name=tool_name, arguments=arguments)
        logger.info(f"✅ [RESEARCHER] Tool result: {result}")

        final_response = f"Research result: {result}"
        logger.info(f"📤 [RESEARCHER] Final response: {final_response}")
        return {"messages": [AIMessage(content=final_response)]}
    except Exception as e:
        error_msg = f"Error in research: {str(e)}"
        logger.error(f"❌ [RESEARCHER] Error: {error_msg}")
        return {"messages": [AIMessage(content=error_msg)]}

async def writer_agent(state: AgentState):
    """Writer agent for content creation."""
    logger.info("✍️ [WRITER] Starting content creation process...")
    messages = state["messages"]

    # Get available tools and filter by role
    try:
        all_tools = await state["mcp_client"].list_tools()
        # Filter tools based on writer's allowed tools
        allowed_tools = AGENT_TOOLS["writer"]
        tools = [tool for tool in all_tools if tool.name in allowed_tools]
        tool_info = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        logger.info(f"🛠️ [WRITER] Available tools for writer: {', '.join([t.name for t in tools])}")
    except Exception as e:
        tool_info = "No tools available"
        logger.error(f"❌ [WRITER] Error listing tools: {e}")

    # Use LLM to decide which tool to use for content creation
    logger.info(f"💭 [WRITER] Analyzing query for content creation: {messages[-1].content}")
    response = llm.invoke([
        {"role": "system", "content": f"You are a writer agent. Available tools:\n{tool_info}\n\nChoose an appropriate tool to help create content and provide the tool name and arguments in JSON format: {{\"tool_name\": \"name\", \"arguments\": {{}}}}"},
        *messages
    ])

    logger.debug(f"📝 [WRITER] Writer LLM response: {response.content}")

    try:
        tool_call = json.loads(response.content)
        tool_name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments", {})

        logger.info(f"🔧 [WRITER] Calling tool: {tool_name} with args: {arguments}")

        # Call the MCP tool
        result = await state["mcp_client"].call_tool(name=tool_name, arguments=arguments)
        logger.info(f"✅ [WRITER] Tool result: {result}")

        final_response = f"Content created: {result}"
        logger.info(f"📤 [WRITER] Final response: {final_response}")
        return {"messages": [AIMessage(content=final_response)]}
    except Exception as e:
        error_msg = f"Error in content creation: {str(e)}"
        logger.error(f"❌ [WRITER] Error: {error_msg}")
        return {"messages": [AIMessage(content=error_msg)]}

async def analyst_agent(state: AgentState):
    """Analyst agent for data analysis."""
    logger.info("📊 [ANALYST] Starting data analysis process...")
    messages = state["messages"]

    # Get available tools and filter by role
    try:
        all_tools = await state["mcp_client"].list_tools()
        # Filter tools based on analyst's allowed tools
        allowed_tools = AGENT_TOOLS["analyst"]
        tools = [tool for tool in all_tools if tool.name in allowed_tools]
        tool_info = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        logger.info(f"🛠️ [ANALYST] Available tools for analyst: {', '.join([t.name for t in tools])}")
    except Exception as e:
        tool_info = "No tools available"
        logger.error(f"❌ [ANALYST] Error listing tools: {e}")

    # Use LLM to decide which tool to use for analysis
    logger.info(f"💭 [ANALYST] Analyzing query for data analysis: {messages[-1].content}")
    response = llm.invoke([
        {"role": "system", "content": f"You are an analyst agent. Available tools:\n{tool_info}\n\nChoose an appropriate tool to analyze data and provide the tool name and arguments in JSON format: {{\"tool_name\": \"name\", \"arguments\": {{}}}}"},
        *messages
    ])

    logger.debug(f"📝 [ANALYST] Analyst LLM response: {response.content}")

    try:
        tool_call = json.loads(response.content)
        tool_name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments", {})

        logger.info(f"🔧 [ANALYST] Calling tool: {tool_name} with args: {arguments}")

        # Call the MCP tool
        result = await state["mcp_client"].call_tool(name=tool_name, arguments=arguments)
        logger.info(f"✅ [ANALYST] Tool result: {result}")

        final_response = f"Analysis result: {result}"
        logger.info(f"📤 [ANALYST] Final response: {final_response}")
        return {"messages": [AIMessage(content=final_response)]}
    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        logger.error(f"❌ [ANALYST] Error: {error_msg}")
        return {"messages": [AIMessage(content=error_msg)]}

# Create the graph
def create_multiagent_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("coordinator", coordinator_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("analyst", analyst_agent)
    
    # Add edges
    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        lambda state: state["current_agent"],
        {
            "researcher": "researcher",
            "writer": "writer",
            "analyst": "analyst"
        }
    )
    graph.add_edge("researcher", END)
    graph.add_edge("writer", END)
    graph.add_edge("analyst", END)
    
    return graph.compile()

# Global graph instance
multiagent_graph = create_multiagent_graph()

# Function to run the multi-agent system
async def run_multiagent_async(user_input: str):
    logger.info(f"🚀 [SYSTEM] Starting multi-agent process for query: {user_input}")
    # Initialize MCP client
    try:
        client = Client(MCP_SERVER_URL)
        await client.__aenter__()

        # Get available tools
        tools = await client.list_tools()
        logger.info(f"📋 [SYSTEM] Total MCP tools available: {len(tools)}")

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "current_agent": "",
            "available_tools": tools,
            "mcp_client": client
        }

        logger.info("⚡ [SYSTEM] Executing agent workflow...")
        result = await multiagent_graph.ainvoke(initial_state)

        # Extract the final response
        final_message = result["messages"][-1]
        response = final_message.content

        logger.info(f"🎉 [SYSTEM] Process completed. Final response: {response[:100]}...")
        await client.__aexit__(None, None, None)
        return response

    except Exception as e:
        error_msg = f"System error: {str(e)}"
        logger.error(f"❌ [SYSTEM] Error: {error_msg}")
        return error_msg

# Synchronous wrapper for Gradio
def run_multiagent(user_input: str):
    return asyncio.run(run_multiagent_async(user_input))

# Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="Multi-Agent Framework with MCP") as interface:
        gr.Markdown("# 🤖 Multi-Agent Framework with MCP Integration")
        gr.Markdown("Enter your query and see how different agents collaborate using MCP tools.")
        gr.Markdown("**Check the console/terminal for detailed agent interaction logs!**")
        
        with gr.Row():
            input_text = gr.Textbox(
                label="Your Query",
                placeholder="Enter your question or task...",
                lines=3
            )
        
        output_text = gr.Textbox(
            label="Agent Response",
            lines=10,
            interactive=False
        )
        
        submit_btn = gr.Button("🚀 Submit Query")
        
        submit_btn.click(
            fn=run_multiagent,
            inputs=input_text,
            outputs=output_text
        )
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)
