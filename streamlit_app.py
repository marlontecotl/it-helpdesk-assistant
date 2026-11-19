"""Streamlit chat interface for the AI-Powered IT Helpdesk Assistant.

Run with:  streamlit run streamlit_app.py
(start the MCP server first, in another terminal:  python mcp_server.py)

All the AI logic lives in app_implementation.py; this file only draws the UI.
"""

import json

import streamlit as st

from app_implementation import (
    EMBEDDING_MODEL,
    MCP_SERVER_URL,
    MODEL,
    ConfigurationError,
    HelpdeskAssistant,
    read_all_tickets,
)

st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️", layout="centered")

SAMPLE_PROMPTS = {
    "RAG": [
        "How do I fix network connectivity issues?",
        "My computer is running slow. What should I do?",
        "How do I configure my email?",
        "How can I protect my laptop from malware?",
    ],
    "Web search": [
        "Search for the latest Windows 11 Wi-Fi troubleshooting steps.",
        "Find solutions for Outlook not syncing emails.",
    ],
    "MCP": [
        "Create a ticket for my laptop running slowly.",
        "Check the status of ticket IT-101.",
    ],
    "Tool calling": ["What is the current date and time?"],
}

SOURCE_ICONS = {"RAG": "📚", "MCP": "🎫", "Client tool": "🕒", "Server tool": "🌐"}


@st.cache_resource(show_spinner="Indexing the IT Support FAQ with Voyage AI...")
def load_assistant() -> HelpdeskAssistant:
    """Built once per server process: embeds the FAQ and discovers the MCP tools."""
    return HelpdeskAssistant()


def render_trace(events: list[dict], sources: list[dict]) -> None:
    """Show which tools answered the question, so the behaviour is easy to verify."""
    if events:
        used = ", ".join(dict.fromkeys(f"{SOURCE_ICONS.get(e['source'], '🔧')} {e['tool']}" for e in events))
        with st.expander(f"How this was answered: {used}"):
            for event in events:
                status = "failed" if event["is_error"] else "ok"
                st.markdown(f"**{event['source']}** · `{event['tool']}` · {status}")
                st.caption("Input")
                st.code(json.dumps(event["input"], indent=2), language="json")
                st.caption("Output")
                st.code(_pretty(event["output"]), language="json")
    if sources:
        st.caption("Web sources: " + " · ".join(f"[{s['title']}]({s['url']})" for s in sources[:5]))


def _pretty(output: str, limit: int = 1500) -> str:
    try:
        output = json.dumps(json.loads(output), indent=2)
    except (ValueError, TypeError):
        pass
    return output if len(output) <= limit else output[:limit] + "\n..."


# ---------------------------------------------------------------------------
# Start-up
# ---------------------------------------------------------------------------
st.title("🛠️ IT Helpdesk Assistant")
st.caption("Ask an IT question, search the web for a fix, or create and track support tickets.")

try:
    assistant = load_assistant()
except ConfigurationError as error:
    st.error(f"{error}\n\nCopy `.env.example` to `.env`, fill in both API keys and restart the app.")
    st.stop()
except Exception as error:  # e.g. invalid key, no network
    st.error(f"The assistant could not start: {type(error).__name__}: {error}")
    st.stop()

st.session_state.setdefault("api_messages", [])  # full history sent to Claude (incl. tool turns)
st.session_state.setdefault("chat", [])  # what is drawn on screen
st.session_state.setdefault("pending_prompt", None)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Employee")
    employee_name = st.text_input("Your name", placeholder="Used when creating tickets")

    st.header("System status")
    st.markdown(f"📚 **Knowledge base:** {len(assistant.knowledge_base)} FAQ sections indexed")
    if assistant.mcp_tools:
        st.markdown(f"🎫 **MCP server:** connected, {len(assistant.mcp_tools)} tools")
    else:
        st.markdown("🎫 **MCP server:** not reachable")
        st.caption(f"Start it with `python mcp_server.py` ({MCP_SERVER_URL}). RAG and web search still work.")
        if st.button("Retry connection", width="stretch"):
            assistant.refresh_mcp_tools()
            st.rerun()
    st.caption(f"Model: `{MODEL}` · Embeddings: `{EMBEDDING_MODEL}`")

    st.header("Try a sample prompt")
    for category, prompts in SAMPLE_PROMPTS.items():
        with st.expander(category):
            for sample in prompts:
                if st.button(sample, key=f"sample-{sample}", width="stretch"):
                    st.session_state.pending_prompt = sample

    if assistant.mcp_tools:
        st.header("Tickets")
        try:
            tickets = read_all_tickets()  # MCP resource tickets://all
            st.dataframe(
                [{"ID": t["ticket_id"], "Issue": t["issue"], "Status": t["status"]} for t in tickets],
                hide_index=True,
                width="stretch",
            )
        except Exception:
            st.caption("Tickets are unavailable right now.")

    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.api_messages = []
        st.session_state.chat = []
        st.rerun()

# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------
for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message["role"] == "assistant":
            render_trace(message["events"], message["sources"])

typed_prompt = st.chat_input("Describe your IT problem...")
prompt = typed_prompt or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if prompt:
    st.session_state.chat.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_size = len(st.session_state.api_messages)
    st.session_state.api_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Working on it..."):
                result = assistant.run_turn(st.session_state.api_messages, employee_name)
        except Exception as error:
            # Roll back the half-finished turn so the history stays valid for the next question.
            del st.session_state.api_messages[history_size:]
            result = {"text": f"⚠️ Something went wrong: {type(error).__name__}: {error}", "events": [], "sources": []}
        st.markdown(result["text"])
        render_trace(result["events"], result["sources"])

    st.session_state.chat.append({"role": "assistant", **result})
    st.rerun()  # refresh the sidebar (a ticket may have been created)
