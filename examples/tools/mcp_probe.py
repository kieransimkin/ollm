"""Exercise real MCP discovery and execution without a model or GPU."""
import argparse
import asyncio
import sys
from pathlib import Path


async def run(url):
    from ollm.tools import MCPClient, MCPServerConfig, ToolCall, ToolRegistry
    from ollm.tools.mcp import tool_alias
    registry = ToolRegistry()
    config = (MCPServerConfig(name="demo", transport="streamable-http", url=url, allow_tools=frozenset({"add"}), auto_approve=frozenset({"add"}))
              if url else MCPServerConfig(name="demo", command=sys.executable,
                  args=(str(Path(__file__).with_name("mcp_server.py").resolve()),), allow_tools=frozenset({"add"}), auto_approve=frozenset({"add"})))
    async with MCPClient(registry) as client:
        tools = await client.connect(config)
        print("Discovered:", [tool.name for tool in tools])
        result = await registry.execute(ToolCall(tool_alias("demo", "add"), {"a": 17, "b": 25}))
        print(result.to_content())
        if result.is_error:
            raise RuntimeError("MCP probe failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", help="Use this Streamable HTTP endpoint instead of local stdio")
    asyncio.run(run(p.parse_args().url))
