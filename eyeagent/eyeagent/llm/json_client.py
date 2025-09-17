import os
import json
from typing import Any, Dict, Optional, List
from loguru import logger
from dotenv import load_dotenv

# LangChain OpenAI-style chat client (works with OpenAI-compatible APIs)
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from ..metrics.metrics import add_tokens

# Ensure .env is loaded and keys propagated
load_dotenv()
if not os.getenv("OPENAI_API_KEY") and os.getenv("DEEPSEEK_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY") or ""

# Global defaults (can be overridden per-agent via env)
DEFAULT_BASE_URL = os.getenv("AGENT_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"))
DEFAULT_MODEL = os.getenv("AGENT_LLM_MODEL", os.getenv("OPENAI_MODEL", "deepseek-chat"))
DEFAULT_TEMPERATURE = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("AGENT_LLM_MAX_TOKENS", "4096"))


def _agent_model(agent_name: Optional[str]) -> str:
    if not agent_name:
        return DEFAULT_MODEL
    # Allow per-agent overrides, e.g., EYEAGENT_MODEL_ReportAgent
    key = f"EYEAGENT_MODEL_{agent_name}"
    return os.getenv(key, DEFAULT_MODEL)


def _make_llm(agent_name: Optional[str] = None) -> ChatOpenAI:
    """Create a chat model. We keep it OpenAI-compatible for flexibility.

    Note: For strict JSON mode on providers that support it, you can set
    OPENAI_RESPONSE_FORMAT='{"type":"json_object"}' in env; we still hard-nudge
    via the prompt for safety.
    """
    base_url = os.getenv("EYEAGENT_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = _agent_model(agent_name)
    temperature = float(os.getenv("EYEAGENT_LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    max_tokens = int(os.getenv("EYEAGENT_LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    logger.debug(f"[LLM] init agent={agent_name} base_url={base_url} model={model} temp={temperature} max_tokens={max_tokens}")
    return ChatOpenAI(base_url=base_url, model=model, temperature=temperature, max_tokens=max_tokens)


class JsonLLM:
    """Thin wrapper to obtain JSON-only responses from an LLM.

    We enforce JSON via instruction and attempt to parse the first JSON object.
    Optionally accepts future VLM inputs via image URLs (not consumed yet).
    """

    def __init__(self, agent_name: Optional[str] = None):
        self.agent_name = agent_name
        # Create underlying chat model (OpenAI-compatible)
        self.llm = _make_llm(agent_name)

    def _coerce_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        # Fast path
        try:
            return json.loads(text)
        except Exception:
            pass
        # Fallback: find first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                logger.debug("JSON parse fallback failed.")
        # Last resort
        raise ValueError(f"Model did not return JSON. Got: {text[:300]}...")

    def invoke_json(self, system_prompt: str, user_prompt: str, schema_hint: Optional[str] = None, images: Optional[List[str]] = None) -> Dict[str, Any]:
        """Invoke LLM and parse JSON output.

        - system_prompt: role/system content
        - user_prompt: role/user content
        - schema_hint: optional textual schema guide (fields, types, constraints)
        - images: reserved for future VLM usage (image URLs or data URIs)
        """
        instructions = [
            {"role": "system", "content": system_prompt.strip()},
        ]
        # If VLM is used in future, we can add image blocks in content here for supported providers
        prompt_parts = [
            "Respond ONLY with a single JSON object that strictly matches the required schema.",
        ]
        if schema_hint:
            prompt_parts.append("Schema (MUST match exactly):\n" + schema_hint.strip())
        prompt_parts.append("Do not include any explanatory text outside of the JSON.")
        instructions.append({"role": "user", "content": ("\n\n".join(prompt_parts) + "\n\n" + user_prompt.strip())})
        logger.debug(f"[LLM] invoke_json agent={self.agent_name} msgs={len(instructions)} schema_hint_len={(len(schema_hint) if schema_hint else 0)}")
        resp = self.llm.invoke(instructions)
        try:
            meta = getattr(resp, "response_metadata", {}) or {}
            usage = meta.get("token_usage") or {}
            if usage:
                if usage.get("prompt_tokens"):
                    add_tokens(self.agent_name or "JsonLLM", "prompt", int(usage.get("prompt_tokens", 0)))
                if usage.get("completion_tokens"):
                    add_tokens(self.agent_name or "JsonLLM", "completion", int(usage.get("completion_tokens", 0)))
                if usage.get("total_tokens"):
                    add_tokens(self.agent_name or "JsonLLM", "total", int(usage.get("total_tokens", 0)))
        except Exception:
            pass
        raw = resp.content if hasattr(resp, "content") else str(resp)
        return self._coerce_json(raw)

    def invoke_structured(self, system_prompt: str, user_prompt: str, schema_model: type[BaseModel]):
        """Prefer structured output via Pydantic schema; falls back to JSON parsing if provider unsupported.

        Returns a Pydantic model instance on success, else raises and caller may fallback.
        """
        try:
            structured = self.llm.with_structured_output(schema_model, method="json_schema")
            logger.debug(f"[LLM] invoke_structured agent={self.agent_name} model={schema_model.__name__}")
            res = structured.invoke([
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ])
            try:
                meta = getattr(res, "response_metadata", {}) or {}
                usage = meta.get("token_usage") or {}
                if usage:
                    if usage.get("prompt_tokens"):
                        add_tokens(self.agent_name or "JsonLLM", "prompt", int(usage.get("prompt_tokens", 0)))
                    if usage.get("completion_tokens"):
                        add_tokens(self.agent_name or "JsonLLM", "completion", int(usage.get("completion_tokens", 0)))
                    if usage.get("total_tokens"):
                        add_tokens(self.agent_name or "JsonLLM", "total", int(usage.get("total_tokens", 0)))
            except Exception:
                pass
            return res
        except Exception as e:
            # Let caller decide fallback to invoke_json
            raise e

    def invoke_json_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Invoke with a list of messages (role/content), expect JSON response."""
        logger.debug(f"[LLM] invoke_json_messages agent={self.agent_name} msgs={len(messages)}")
        resp = self.llm.invoke(messages)
        try:
            meta = getattr(resp, "response_metadata", {}) or {}
            usage = meta.get("token_usage") or {}
            if usage:
                if usage.get("prompt_tokens"):
                    add_tokens(self.agent_name or "JsonLLM", "prompt", int(usage.get("prompt_tokens", 0)))
                if usage.get("completion_tokens"):
                    add_tokens(self.agent_name or "JsonLLM", "completion", int(usage.get("completion_tokens", 0)))
                if usage.get("total_tokens"):
                    add_tokens(self.agent_name or "JsonLLM", "total", int(usage.get("total_tokens", 0)))
        except Exception:
            pass
        raw = resp.content if hasattr(resp, "content") else str(resp)
        return self._coerce_json(raw)

    def invoke_structured_messages(self, messages: List[Dict[str, str]], schema_model: type[BaseModel]):
        """Invoke with a list of messages and parse into a Pydantic model using structured output."""
        structured = self.llm.with_structured_output(schema_model, method="json_schema")
        logger.debug(f"[LLM] invoke_structured_messages agent={self.agent_name} model={schema_model.__name__} msgs={len(messages)}")
        res = structured.invoke(messages)
        try:
            meta = getattr(res, "response_metadata", {}) or {}
            usage = meta.get("token_usage") or {}
            if usage:
                if usage.get("prompt_tokens"):
                    add_tokens(self.agent_name or "JsonLLM", "prompt", int(usage.get("prompt_tokens", 0)))
                if usage.get("completion_tokens"):
                    add_tokens(self.agent_name or "JsonLLM", "completion", int(usage.get("completion_tokens", 0)))
                if usage.get("total_tokens"):
                    add_tokens(self.agent_name or "JsonLLM", "total", int(usage.get("total_tokens", 0)))
        except Exception:
            pass
        return res
