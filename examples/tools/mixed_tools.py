"""Mix a Python music-timing tool with MCP arithmetic in one registry."""
import asyncio
import sys
from pathlib import Path
from common import backend, parser, print_result


async def run(args, model):
    from ollm.tools import Agent, MCPClient, MCPServerConfig, ToolRegistry
    registry = ToolRegistry()

    @registry.tool(parameters={"type": "object", "properties": {
        "bpm": {"type": "number", "exclusiveMinimum": 0},
        "beats": {"type": "number", "minimum": 0}},
        "required": ["bpm", "beats"], "additionalProperties": False}, requires_approval=False)
    def beats_to_seconds(bpm: float, beats: float) -> float:
        """Convert a number of beats to seconds at a constant BPM."""
        return beats * 60 / bpm

    async with MCPClient(registry) as client:
        await client.connect(MCPServerConfig(name="math", command=sys.executable,
            args=(str(Path(__file__).with_name("mcp_server.py").resolve()),),
            allow_tools=frozenset({"add"}), auto_approve=frozenset({"add"})))
        print_result(await Agent(model, registry).run(args.prompt), args)


if __name__ == "__main__":
    p = parser(__doc__)
    p.set_defaults(prompt="Use beats_to_seconds for 24 beats at 120 BPM, then use math__add to add 5 seconds.")
    args = p.parse_args()
    asyncio.run(run(args, backend(args)))
