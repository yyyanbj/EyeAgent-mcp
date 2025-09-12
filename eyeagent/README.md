# Multi-Agent Framework with MCP Integration

This is a multi-agent framework built using LangChain and LangGraph, integrated with MCP (Model Context Protocol) tools, and with a Gradio interface for interaction.

## ✨ New Features

### 🔍 **Structured Logging with Loguru**
- **Professional logging** using loguru library
- **Structured log messages** with timestamps, log levels, and agent identification
- **Color-coded console output** for better readability
- **Configurable log levels** (DEBUG, INFO, WARNING, ERROR)
- **Emoji-coded agent identification** for easy tracking:
  - 🤖 [COORDINATOR]: Coordination decisions
  - 🔍 [RESEARCHER]: Research activities
  - ✍️ [WRITER]: Content creation
  - 📊 [ANALYST]: Data analysis
  - 🚀 [SYSTEM]: System-level events

### 🎯 **Role-Based Tool Access**
Each agent now has **specific tools** based on their role:

- **Researcher Agent**: `calculate`, `multiply`, `sum_numbers`, `get_weather`
- **Writer Agent**: `generate_image`, `classify_image`
- **Analyst Agent**: `segment_image`, `detect_objects`, `classify_image`
- **Coordinator Agent**: No direct tools (focuses on orchestration)

## Features

- **MCP Integration**: Agents can discover and use tools from MCP servers
- **Coordinator Agent**: Analyzes user queries and routes to appropriate specialized agents
- **Researcher Agent**: Uses MCP tools for information gathering
- **Writer Agent**: Uses MCP tools for content creation
- **Analyst Agent**: Uses MCP tools for data analysis
- **Gradio UI**: Interactive web interface to communicate with the agents
- **Comprehensive Logging**: Detailed logs of agent interactions and tool usage

## Setup & CI/CD

1. Install dependencies:
   ```bash
   cd /home/bingjie/workspace/EyeAgent-mcp/eyeagent
   uv sync
   ```

2. Set up your OpenAI API key:
   - Edit the `.env` file and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_actual_api_key_here
     ```

3. Configure logging (optional):
   - The framework uses loguru for structured logging
   - Default log level is INFO
   - Logs are displayed in console with colors and timestamps
   - You can modify logging configuration in `multiagent_framework.py`

4. Ensure MCP server is running:
   - The framework expects an MCP server running at `http://localhost:8001/mcp/`
   - You can test the MCP server using the `main_client.py` script

## Running the Framework

Run the multi-agent system with the Gradio interface:

```bash
cd /home/bingjie/workspace/EyeAgent-mcp/eyeagent
uv run python run_multiagent.py
```

The Gradio interface will be available at `http://localhost:7860`.

## How it Works

### 🔄 **Agent Workflow with Loguru Logging**

1. **System Initialization**:
   ```
   2024-01-15 10:30:15.123 | INFO     | __main__:main:45 - 🚀 [SYSTEM] Starting multi-agent process for query: "Calculate 2 + 3"
   2024-01-15 10:30:15.124 | INFO     | __main__:main:46 - 📋 [SYSTEM] Total MCP tools available: 8
   ```

2. **Coordinator Analysis**:
   ```
   2024-01-15 10:30:15.125 | INFO     | __main__:coordinator_agent:78 - 🤖 [COORDINATOR] Starting coordination process...
   2024-01-15 10:30:15.126 | INFO     | __main__:coordinator_agent:80 - 📋 [COORDINATOR] Available MCP tools: calculate, multiply, sum_numbers, get_weather, ...
   2024-01-15 10:30:15.127 | INFO     | __main__:coordinator_agent:82 - 💭 [COORDINATOR] Analyzing user query: Calculate 2 + 3
   2024-01-15 10:30:15.128 | INFO     | __main__:coordinator_agent:85 - 🎯 [COORDINATOR] Selected agent: researcher
   ```

3. **Specialized Agent Execution**:
   ```
   2024-01-15 10:30:15.129 | INFO     | __main__:researcher_agent:112 - 🔍 [RESEARCHER] Starting research process...
   2024-01-15 10:30:15.130 | INFO     | __main__:researcher_agent:114 - 🛠️ [RESEARCHER] Available tools for researcher: calculate, multiply, sum_numbers, get_weather
   2024-01-15 10:30:15.131 | INFO     | __main__:researcher_agent:116 - 💭 [RESEARCHER] Analyzing query for research: Calculate 2 + 3
   2024-01-15 10:30:15.132 | INFO     | __main__:researcher_agent:120 - 📝 [RESEARCHER] Researcher LLM response: {"tool_name": "calculate", "arguments": {"expression": "2 + 3"}}
   2024-01-15 10:30:15.133 | INFO     | __main__:researcher_agent:125 - 🔧 [RESEARCHER] Calling tool: calculate with args: {'expression': '2 + 3'}
   2024-01-15 10:30:15.134 | SUCCESS  | __main__:researcher_agent:130 - ✅ [RESEARCHER] Tool result: 5
   2024-01-15 10:30:15.135 | INFO     | __main__:researcher_agent:132 - 📤 [RESEARCHER] Final response: Research result: 5
   ```

4. **System Completion**:
   ```
   2024-01-15 10:30:15.136 | SUCCESS  | __main__:main:52 - 🎉 [SYSTEM] Process completed. Final response: Research result: 5...
   ```

## Architecture

## CI/CD

PyPI 发布使用标签策略：
- `eyeagent-vX.Y.Z`
- 推送前更新 `pyproject.toml` 中版本。

文档 (GitHub Pages) 构建：
- 主分支推送自动构建 `mkdocs.eyeagent.yml` & `mkdocs.eyetools.yml`。
- 访问：https://beiyuouo.github.io/EyeAgent-mcp/eyeagent/  与 `/eyetools/`。

所需仓库 Secrets：
- `PYPI_TOKEN_EYEAGENT`
- `PYPI_TOKEN_EYETOOLS`


The system uses LangGraph to create an asynchronous state graph with the following nodes:
- **Coordinator**: Routes tasks to appropriate agents and provides tool context
- **Researcher**: Handles information gathering using role-specific MCP tools
- **Writer**: Handles content creation using role-specific MCP tools
- **Analyst**: Handles data analysis using role-specific MCP tools

Each agent is powered by GPT-4o-mini through LangChain and can dynamically discover and use MCP tools based on their assigned role.

## MCP Server Requirements

The framework expects the MCP server to provide tools that agents can use. Example tools might include:
- Math calculation tools (`calculate`, `multiply`, `sum_numbers`)
- Data processing tools (`get_weather`)
- Image processing tools (`classify_image`, `segment_image`, `detect_objects`, `generate_image`)

Make sure your MCP server is running and accessible before starting the multi-agent framework.

## Loguru Logging Benefits

The framework uses loguru for professional-grade logging with the following advantages:

- **Structured Logging**: Each log entry includes timestamp, log level, module, function, and line number
- **Color Coding**: Different log levels have distinct colors for easy identification
- **Emoji Integration**: Agent-specific emojis help track which agent is performing actions
- **Configurable Levels**: DEBUG, INFO, WARNING, ERROR, SUCCESS levels available
- **Performance**: Minimal performance impact compared to print statements
- **Thread Safety**: Safe for concurrent agent execution
- **Easy Filtering**: Can filter logs by agent type or log level

## Log Analysis

The structured logs help you understand:
- **Agent decision-making process** with precise timestamps
- **Tool selection logic** with detailed context
- **MCP tool execution results** with success/failure status
- **Error handling and debugging** with stack traces
- **Performance monitoring** with execution timing
- **Concurrent execution tracking** in multi-agent scenarios

Check the console output when running queries to see the complete agent interaction flow with professional loguru formatting!
