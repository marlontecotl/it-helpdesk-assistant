"""IT Helpdesk MCP Server.

A small Model Context Protocol server that exposes the IT support operations
the chatbot is allowed to perform:

  - create_support_ticket : create a new IT support ticket
  - check_ticket_status   : retrieve the status of an existing ticket
  - tickets://all         : read-only resource listing every ticket

Tickets live in an in-memory Python dictionary (seeded from
data/mcp_tool_data.json), so they reset every time the server restarts.

Run with:  python mcp_server.py
The server listens on http://127.0.0.1:3000/mcp (Streamable HTTP transport).
"""

import json
import os
from datetime import datetime
from pathlib import Path

from typing import Annotated

import uvicorn
from mcp.server import MCPServer
from pydantic import Field

HOST = os.getenv("MCP_HOST", "127.0.0.1")  # localhost only, as the MCP spec recommends
PORT = int(os.getenv("MCP_PORT", "3000"))
SEED_FILE = Path(__file__).parent / "data" / "mcp_tool_data.json"
VALID_PRIORITIES = ("Low", "Medium", "High")

mcp_server = MCPServer(
    name="IT Helpdesk MCP Server",
    version="1.0.0",
    instructions="Create IT support tickets and check the status of existing ones.",
)

# ---------------------------------------------------------------------------
# In-memory ticket store: {ticket_id: ticket_dict}
# ---------------------------------------------------------------------------
TICKETS: dict[str, dict] = {}


def _load_seed_tickets() -> None:
    """Fill the in-memory store with the sample tickets, if the file exists."""
    if not SEED_FILE.exists():
        return
    for ticket in json.loads(SEED_FILE.read_text(encoding="utf-8")):
        TICKETS[ticket["ticket_id"].upper()] = ticket


def _error(category: str, message: str, retryable: bool = False) -> str:
    """Structured error the agent can act on (instead of a generic failure).

    category  : validation | not_found  -> tells the agent how to react
    retryable : False means calling again with the same input will not help
    """
    return json.dumps({"error": {"category": category, "retryable": retryable, "message": message}})


def _next_ticket_id() -> str:
    """Return the next free id, continuing the IT-101, IT-102, ... sequence."""
    numbers = [int(tid.split("-")[1]) for tid in TICKETS if tid.split("-")[-1].isdigit()]
    return f"IT-{max(numbers, default=100) + 1}"


_load_seed_tickets()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp_server.tool(
    name="create_support_ticket",
    description=(
        "Create a new IT support ticket for an employee and return the created ticket, "
        "including its new ticket_id. Use it only when the employee explicitly asks to "
        "open, create or raise a ticket. Do not use it to look up existing tickets."
    ),
)
def create_support_ticket(
    employee: Annotated[str, Field(description="Name of the employee reporting the problem.")],
    issue: Annotated[str, Field(description="Short title of the problem, e.g. 'Laptop running slowly'.")],
    description: Annotated[str, Field(description="Extra details about the problem. Optional.")] = "",
    priority: Annotated[str, Field(description="One of: Low, Medium, High. Defaults to Medium.")] = "Medium",
) -> str:
    if not employee.strip():
        return _error("validation", "'employee' is required. Ask the employee for their name.")
    if not issue.strip():
        return _error("validation", "'issue' is required. Ask the employee what the problem is.")

    priority = priority.strip().capitalize()
    if priority not in VALID_PRIORITIES:
        return _error("validation", f"priority must be one of: {', '.join(VALID_PRIORITIES)}.")

    ticket_id = _next_ticket_id()
    ticket = {
        "ticket_id": ticket_id,
        "employee": employee.strip(),
        "issue": issue.strip(),
        "description": description.strip() or issue.strip(),
        "priority": priority,
        "status": "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    TICKETS[ticket_id] = ticket
    return json.dumps(ticket)


@mcp_server.tool(
    name="check_ticket_status",
    description=(
        "Retrieve an existing IT support ticket by its id (format 'IT-<number>', e.g. "
        "'IT-101') and return its status, priority, issue and employee. Use it when the "
        "employee asks about a ticket that already exists. Do not use it to create tickets."
    ),
)
def check_ticket_status(
    ticket_id: Annotated[str, Field(description="Ticket identifier in the form IT-<number>, e.g. 'IT-101'.")],
) -> str:
    ticket = TICKETS.get(ticket_id.strip().upper())
    if ticket is None:
        # The lookup worked and found nothing: a valid answer, not a transient failure.
        return _error(
            "not_found",
            f"No ticket with id '{ticket_id}'. Existing ids: {', '.join(sorted(TICKETS)) or 'none'}.",
        )
    return json.dumps(ticket)


# ---------------------------------------------------------------------------
# Resource (read-only data the application can fetch by URI)
# ---------------------------------------------------------------------------
@mcp_server.resource("tickets://all")
def all_tickets() -> str:
    """Every ticket currently stored, as a JSON list."""
    return json.dumps(list(TICKETS.values()))


# Starlette app that serves the MCP endpoint at /mcp over Streamable HTTP.
app = mcp_server.streamable_http_app()

if __name__ == "__main__":
    print(f"IT Helpdesk MCP Server running at http://{HOST}:{PORT}/mcp  ({len(TICKETS)} tickets loaded)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
