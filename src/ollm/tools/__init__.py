"""Optional Python/MCP tool orchestration for oLLM.

Importing this package does not load a model or import torch, MCP, Harmony, or
Qwen-Agent. Those integrations are loaded only when requested.
"""
from .adapters import GPTOSSAdapter, QwenAdapter
from .agent import Agent, AgentResult
from .backend import GenerationConfig, InferenceBackend
from .mcp import MCPClient, MCPServerConfig
from .registry import Tool, ToolPolicy, ToolRegistry
from .types import (AgentLimitError, AssistantTurn, IncompleteGeneration, ToolCall,
                    ToolCallParseError, ToolingError, ToolResult)

__all__ = ["Agent", "AgentResult", "AgentLimitError", "AssistantTurn", "GenerationConfig",
           "GPTOSSAdapter", "InferenceBackend", "IncompleteGeneration", "MCPClient", "MCPServerConfig",
           "QwenAdapter", "Tool", "ToolCall", "ToolCallParseError", "ToolPolicy", "ToolRegistry",
           "ToolResult", "ToolingError"]
