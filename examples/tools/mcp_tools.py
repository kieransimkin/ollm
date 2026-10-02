"""Native tool loop using a local or remote MCP server."""
import asyncio
import json
import os
import sys
from pathlib import Path
from common import backend, parser, print_result


def approve(tool, arguments):
    answer = input(f"Approve {tool.name} {json.dumps(arguments)}? [y/N] ")
    return answer.strip().lower() == "y"


async def run(args, model):
    from ollm.tools import Agent, MCPClient, MCPServerConfig, ToolPolicy, ToolRegistry
    registry = ToolRegistry(ToolPolicy(approve=approve))
    if args.url:
        token = os.environ.get(args.token_env)
        config = MCPServerConfig(name="remote", transport="streamable-http", url=args.url,
            headers={"Authorization": "Bearer " + token} if token else None,
            allow_tools=frozenset(args.tool) if args.tool else None,
            auto_approve=frozenset(args.approve_tool))
    else:
        config = MCPServerConfig(name="demo", command=sys.executable,
            args=(str(Path(__file__).with_name("mcp_server.py").resolve()),),
            allow_tools=frozenset({"add", "multiply"}), auto_approve=frozenset({"add", "multiply"}))
    async with MCPClient(registry) as client:
        tools = await client.connect(config)
        print("Available tools:", ", ".join(tool.name for tool in tools))
        result = await Agent(model, registry).run(args.prompt)
        print_result(result, args)


def main():
    p = parser(__doc__)
    p.add_argument("--url", help="Streamable HTTP endpoint; otherwise launch the bundled stdio server")
    p.add_argument("--token-env", default="OLLM_MCP_TOKEN", help="Environment variable holding a remote bearer token")
    p.add_argument("--tool", action="append", default=[], help="Expose only these remote tool names; repeat as needed")
    p.add_argument("--approve-tool", action="append", default=[], help="Explicitly auto-approve this remote tool; repeat as needed")
    args = p.parse_args()
    model = backend(args)  # Load before opening long-lived MCP sessions.
    asyncio.run(run(args, model))


if __name__ == "__main__":
    main()
