"""Bounded Qwen inference with the existing native Python-tool execution policy."""
from common import parser,backend,NUMBER_PAIR


def main():
    args=parser(__doc__).parse_args()
    from ollm.tools import Agent,ToolRegistry
    registry=ToolRegistry()
    @registry.tool(parameters=NUMBER_PAIR,requires_approval=False)
    def add(a:float,b:float)->float:
        """Add two numbers."""
        return a+b
    @registry.tool(parameters=NUMBER_PAIR,requires_approval=False)
    def multiply(a:float,b:float)->float:
        """Multiply two numbers."""
        return a*b
    result=Agent(backend(args),registry,max_rounds=6,max_tool_calls=6).run_sync(args.prompt)
    print(result.text)


if __name__=='__main__':
    main()
