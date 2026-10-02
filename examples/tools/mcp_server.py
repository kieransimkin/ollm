"""Read-only arithmetic MCP server. stdout is reserved for the stdio protocol."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("oLLM arithmetic demo", host="127.0.0.1", port=args.port)

    @server.tool()
    def add(a: float, b: float) -> float:
        """Add two finite numbers."""
        return a + b

    @server.tool()
    def multiply(a: float, b: float) -> float:
        """Multiply two finite numbers."""
        return a * b

    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
