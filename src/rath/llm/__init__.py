"""OpenAI-compatible LLM access (synchronous chat completions)."""

from rath.llm._client import OpenAIChatClient
from rath.llm._openai_create_kwargs import to_create_kwargs
from rath.llm._openai_normalize import normalize_chat_completion
from rath.llm._settings import LLMSettings, default_dotenv_path, load_llm_settings
from rath.llm._types_request import LLMChatRequest, LLMFunctionTool, LLMMessage, LLMRole
from rath.llm._types_response import (
    LLMAssistantMessage,
    LLMChatChoice,
    LLMChatResponse,
    LLMTokenUsage,
    LLMToolCallFunction,
    LLMToolCallPart,
)

__all__ = [
    "OpenAIChatClient",
    "LLMSettings",
    "default_dotenv_path",
    "load_llm_settings",
    "to_create_kwargs",
    "normalize_chat_completion",
    "LLMChatRequest",
    "LLMMessage",
    "LLMFunctionTool",
    "LLMRole",
    "LLMChatResponse",
    "LLMChatChoice",
    "LLMAssistantMessage",
    "LLMTokenUsage",
    "LLMToolCallPart",
    "LLMToolCallFunction",
]
