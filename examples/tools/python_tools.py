"""Native Qwen or gpt-oss calls to explicitly registered Python functions."""
from common import NUMBER_PAIR, backend, parser, print_result


def main():
    args = parser(__doc__).parse_args()
    from ollm.tools import Agent, ToolRegistry
    registry = ToolRegistry()

    @registry.tool(parameters=NUMBER_PAIR, requires_approval=False)
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @registry.tool(parameters=NUMBER_PAIR, requires_approval=False)
    async def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    result = Agent(backend(args), registry).run_sync(args.prompt)
    print_result(result, args)


if __name__ == "__main__":
    main()
