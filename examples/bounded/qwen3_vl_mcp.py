"""Qwen3-VL image input plus the existing read-only MCP arithmetic example."""
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
    content=[{'type':'image','image':str(args.image)},
             {'type':'text','text':args.prompt}]
    history=[{'role':'system','content':'Use the image and provided tools when useful.'},
             {'role':'user','content':content}]
    async with MCPClient(registry) as client:
        await client.connect(config)
        result=await Agent(model,registry,max_rounds=6,max_tool_calls=6).run(history)
        print(result.text)


def main():
    p=parser(__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.set_defaults(model_key='qwen3-vl-2b-instruct',
                   prompt='Read the two numbers in the image, add them with the arithmetic tool, and report the result.')
    args=p.parse_args()
    asyncio.run(run(args,backend(args)))


if __name__=='__main__':
    main()
