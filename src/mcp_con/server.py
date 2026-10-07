"""
A beginner-friendly MCP server.

An MCP server is just a program that *advertises* capabilities (tools,
resources, prompts) and *executes* them when a client asks. It knows nothing
about LLMs. The client (see client.py) is what connects those tools to a model.

Run standalone for debugging with the MCP Inspector:
    uv run mcp dev src/mcp_con/server.py
"""

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP

# With the stdio transport, stdout IS the protocol channel. Anything printed to
# stdout corrupts the JSON-RPC messages, so all logging must go to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.getLogger("mcp").setLevel(logging.WARNING)
log = logging.getLogger("demo-server")

# STEP 1: Create the server. The name is what clients see during the handshake.
mcp = FastMCP("demo-server")

# In-memory storage for the notes tools. Lives as long as the server process.
NOTES: list[str] = []


# STEP 2: Register tools with @mcp.tool().
# FastMCP reads the function name, type hints and docstring and turns them into
# a JSON schema. The LLM only ever sees that schema, so the docstring is the
# instruction manual the model uses to decide *when* and *how* to call the tool.


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    log.info("add(%s, %s)", a, b)
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    log.info("multiply(%s, %s)", a, b)
    return a * b


@mcp.tool()
def get_current_time(timezone: str = "Asia/Kolkata") -> str:
    """Get the current date and time in an IANA timezone, e.g. 'Asia/Kolkata',
    'America/New_York', 'Europe/London'."""
    log.info("get_current_time(%s)", timezone)
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        return f"Unknown timezone: {timezone}"
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


@mcp.tool()
def save_note(text: str) -> str:
    """Save a short note so it can be listed later."""
    log.info("save_note(%r)", text)
    NOTES.append(text)
    return f"Saved note #{len(NOTES)}"


@mcp.tool()
def list_notes() -> list[str]:
    """Return all notes saved so far."""
    log.info("list_notes()")
    return NOTES or ["(no notes yet)"]


# Resources are read-only data a client can fetch by URI (like GET endpoints).
# They are not called by the LLM automatically; shown here for completeness.
@mcp.resource("info://about")
def about() -> str:
    """Describe this server."""
    return "Demo MCP server with math, time and notes tools."


def main() -> None:
    # STEP 3: Start serving. "stdio" means the client launches this file as a
    # subprocess and talks to it over stdin/stdout.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
