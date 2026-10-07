"""
MCP client + LLM: the "host" that ties everything together.

Flow for every question you type:
  1. The client sends your question plus the list of tool schemas to the LLM.
  2. The LLM either answers directly, or replies with "please call tool X
     with arguments Y" (a tool call). The LLM never runs code itself.
  3. The client forwards each tool call to the MCP server, which runs it.
  4. The client sends the tool results back to the LLM.
  5. Repeat steps 2-4 until the LLM gives a final text answer.

Run:
    uv run python -m mcp_con.client
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

SERVER_PATH = Path(__file__).with_name("server.py")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_STEPS = 8


def to_text(result) -> str:
    """MCP tools return a list of content blocks; join the text parts."""
    if isinstance(result, list):
        return "\n".join(b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in result)
    return str(result)


async def run_agent(llm_with_tools, tools_by_name, messages: list[BaseMessage]) -> str:
    """The agent loop: keep calling the LLM until it stops requesting tools."""
    for _ in range(MAX_STEPS):
        ai_msg: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return ai_msg.content

        for call in ai_msg.tool_calls:
            print(f"  [tool call] {call['name']}({call['args']})")
            tool = tools_by_name[call["name"]]
            # ainvoke on an MCP-backed tool sends a `tools/call` request to the
            # server over stdio and waits for the result.
            result = await tool.ainvoke(call["args"])
            text = to_text(result)
            print(f"  [tool result] {text}")
            messages.append(ToolMessage(content=text, tool_call_id=call["id"]))

    return "Stopped: too many tool steps."


async def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    # STEP 1: Describe how to reach the server. With "stdio", the client starts
    # the server as a subprocess using this command.
    client = MultiServerMCPClient(
        {
            "demo": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(SERVER_PATH)],
            }
        }
    )

    # STEP 2: Open one long-lived session. This starts the server subprocess and
    # performs the MCP handshake (`initialize`). Without a session, every tool
    # call would spawn a fresh server and in-memory state (notes) would be lost.
    async with client.session("demo") as session:
        # STEP 3: Discover tools via a `tools/list` request. Each MCP tool is
        # wrapped as a LangChain tool object bound to this session.
        tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in tools}
        print("Tools discovered from MCP server:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        # STEP 4: Give the tool schemas to the LLM. bind_tools converts them
        # into the function-calling format the Groq API expects.
        llm = ChatGroq(model=MODEL, temperature=0)
        llm_with_tools = llm.bind_tools(tools)

        messages: list[BaseMessage] = [
            SystemMessage("You are a helpful assistant. Use the available tools when they help.")
        ]

        print(f"\nChatting with {MODEL}. Type 'quit' to exit.\n")
        while True:
            question = (await asyncio.to_thread(input, "You: ")).strip()
            if question.lower() in {"quit", "exit"}:
                break
            if not question:
                continue
            messages.append(HumanMessage(question))
            # STEP 5: Run the agent loop (LLM -> tool calls -> results -> LLM).
            answer = await run_agent(llm_with_tools, tools_by_name, messages)
            print(f"AI: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
