import os
import sys
import json
import asyncio
from typing import Optional

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Use the AsyncOpenAI client
        self.openai = AsyncOpenAI(api_key="")
        self.model = "gpt-4o"  

    async def connect_to_server(self, server_script_path: str):
        """Start an MCP server (subprocess) and connect."""
        if server_script_path.endswith(".py"):
            command = "python"
        elif server_script_path.endswith(".js"):
            command = "node"
        else:
            raise ValueError("Server script must be .py or .js")

        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # Print list of tools
        tool_list = await self.session.list_tools()
        print(f"Connected to server. Tools: {[t.name for t in tool_list.tools]}")

    async def _get_openai_functions(self):
        """Convert MCP tools to OpenAI function-calling format."""
        tool_list = await self.session.list_tools()
        functions = []
        for tool in tool_list.tools:
            fn_schema = {
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
                "required": []
            }
            if tool.inputSchema and tool.inputSchema.get("type") == "object":
                fn_schema["parameters"] = tool.inputSchema
            functions.append(fn_schema)
        return functions

    async def process_user_message(self, user_message: str) -> str:
        if not self.session:
            raise RuntimeError("No MCP session found. Did you call connect_to_server?")

        functions = await self._get_openai_functions()
        # First call to GPT: provide user's message + available functions
        response = await self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_message}],
            functions=functions,
            function_call="auto"
        )
        assistant_message = response.choices[0].message

        if hasattr(assistant_message, "function_call") and assistant_message.function_call:
            fn_name = assistant_message.function_call.name
            fn_args_json = assistant_message.function_call.arguments
            try:
                fn_args = json.loads(fn_args_json) if fn_args_json else {}
            except json.JSONDecodeError:
                return "[Error] GPT produced invalid JSON arguments."

            print(f"\n[Tool request] GPT wants to call {fn_name} with {fn_args}")
            tool_result = await self.session.call_tool(fn_name, fn_args)

            # Second call to GPT: provide the tool's result
            followup = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": user_message},
                    assistant_message.to_dict(),
                    {"role": "function", "name": fn_name, "content": tool_result.content}
                ]
            )
            return followup.choices[0].message.content
        else:
            # GPT didn't request a function
            return assistant_message.content

    async def chat_loop(self):
        print("MCP + OpenAI Client. Type 'quit' or 'exit' to stop.")
        while True:
            user_inp = input("\nYou: ")
            if user_inp.strip().lower() in ("quit", "exit"):
                break
            answer = await self.process_user_message(user_inp)
            print(f"\nAssistant: {answer}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    server_path = sys.argv[1]

    client = MCPClient()
    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())