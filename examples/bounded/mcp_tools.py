"""Bounded Qwen plus the existing read-only MCP arithmetic server."""
import asyncio
import sys
from pathlib import Path
from common import parser,backend


async def run(args,model):
    from ollm.tools import Agent,MCPClient,MCPServerConfig,ToolRegistry
    registry=ToolRegistry()
    server=Path(__file__).resolve().parents[1]/'tools'/'mcp_server.py'
    config=MCPServerConfig(name='arithmetic',command=sys.executable,args=(str(server),),
        allow_tools=frozenset({'add','multiply'}),auto_approve=frozenset({'add','multiply'}))
    async with MCPClient(registry) as client:
        await client.connect(config)
        result=await Agent(model,registry,max_rounds=6,max_tool_calls=6).run(args.prompt)
        print(result.text)


def main():
    args=parser(__doc__).parse_args()
    model=backend(args)
    asyncio.run(run(args,model))


if __name__=='__main__':
    main()
