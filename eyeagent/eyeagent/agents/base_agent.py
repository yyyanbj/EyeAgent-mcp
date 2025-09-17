import uuid
import json
import os
import datetime
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from fastmcp import Client
from loguru import logger
from dotenv import load_dotenv

# Environment configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
CHAT_MODEL = os.getenv("AGENT_LLM_MODEL", "deepseek-chat")
CHAT_BASE_URL = os.getenv("AGENT_LLM_BASE_URL", "https://api.deepseek.com/v1")
CHAT_TEMPERATURE = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.8"))
MAX_TOKENS = int(os.getenv("AGENT_LLM_MAX_TOKENS", "4096"))

# Directory for persisting conversations (created lazily)
CONV_DIR = os.getenv("AGENT_CONV_DIR", os.path.join(os.path.dirname(__file__), "..", "ui", "conversations"))

# Ensure env vars from .env are loaded
load_dotenv()
# If using DeepSeek, allow DEEPSEEK_API_KEY to populate OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("DEEPSEEK_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY") or ""

def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=CHAT_BASE_URL,
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

class ConversationStore:
    """Simple JSON file-based conversation persistence.

    Layout:
    conversations/
        <case_uuid>.json
    File content: {
        "case_id": str,
        "role": str,
        "created_at": ISO,
        "updated_at": ISO,
        "messages": [{"type": "human"|"ai", "content": str, "ts": ISO}]
    }
    """

    def __init__(self, directory: str = CONV_DIR):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, case_id: str) -> str:
        return os.path.join(self.directory, f"{case_id}.json")

    def create(self, role: str) -> str:
        case_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        data = {
            "case_id": case_id,
            "role": role,
            "created_at": now,
            "updated_at": now,
            "messages": []
        }
        with open(self._path(case_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return case_id

    def append(self, case_id: str, message_type: str, content: str):
        path = self._path(case_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Conversation {case_id} not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["messages"].append({
            "type": message_type,
            "content": content,
            "ts": datetime.datetime.utcnow().isoformat()
        })
        data["updated_at"] = datetime.datetime.utcnow().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, case_id: str) -> Dict[str, Any]:
        with open(self._path(case_id), "r", encoding="utf-8") as f:
            return json.load(f)

class BaseAgent:
    """Abstract base agent with tool selection + MCP integration."""

    role: str = "generic"
    system_prompt: str = "You are a helpful generic agent."
    allowed_tools: List[str] = []

    def __init__(self, mcp_url: str = MCP_SERVER_URL, conversation_store: Optional[ConversationStore] = None):
        self.mcp_url = mcp_url
        self.conversation_store = conversation_store or ConversationStore()
        self.case_id: Optional[str] = None

    async def _ensure_case(self):
        if self.case_id is None:
            self.case_id = self.conversation_store.create(self.role)
            logger.info(f"New conversation created case_id={self.case_id} role={self.role}")

    async def _list_allowed_tools(self, client: Client):
        try:
            tools = await client.list_tools()
            filtered = [t for t in tools if not self.allowed_tools or t.name in self.allowed_tools]
            return filtered
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []

    def build_tool_selection_prompt(self, user_query: str, tools_desc: str) -> str:
        return (
            f"You are the {self.role} agent. {self.system_prompt}\n"
            f"User query: {user_query}\n"
            f"Available tools (only these):\n{tools_desc}\n"
            "If a tool is suitable, respond ONLY with JSON: {\"tool_name\": str, \"arguments\": object}. "
            "If no tool is needed, respond ONLY with JSON: {\"tool_name\": null, \"answer\": str}."
        )

    async def _decide_and_call_tool(self, client: Client, user_query: str, messages: List[BaseMessage]):
        tools = await self._list_allowed_tools(client)
        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in tools]) or "(No tools available)"
        prompt = self.build_tool_selection_prompt(user_query, tools_desc)
        # Initialize LLM lazily to ensure env vars are loaded
        llm = _make_llm()
        response = llm.invoke([
            {"role": "system", "content": prompt},
            *messages
        ])
        raw = response.content.strip()
        logger.debug(f"LLM tool decision raw response: {raw}")
        try:
            data = json.loads(raw)
        except Exception as e:
            return None, f"LLM returned invalid JSON: {e}; raw: {raw}"

        tool_name = data.get("tool_name")
        if tool_name is None:
            # Direct answer path
            return None, data.get("answer", "")

        tool_args = data.get("arguments", {})
        try:
            result = await client.call_tool(name=tool_name, arguments=tool_args)
            return tool_name, result
        except Exception as e:
            return tool_name, f"Tool call failed: {e}"

    async def a_run(self, user_query: str) -> Dict[str, Any]:
        await self._ensure_case()
        self.conversation_store.append(self.case_id, "human", user_query)
        messages: List[BaseMessage] = [HumanMessage(content=user_query)]

        async with Client(self.mcp_url) as client:
            tool_name, result_or_answer = await self._decide_and_call_tool(client, user_query, messages)

        if tool_name is None:
            final_answer = result_or_answer
        else:
            final_answer = f"Tool {tool_name} returned: {result_or_answer}"

        self.conversation_store.append(self.case_id, "ai", final_answer)
        return {
            "case_id": self.case_id,
            "role": self.role,
            "answer": final_answer
        }

    def run(self, user_query: str) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self.a_run(user_query))
