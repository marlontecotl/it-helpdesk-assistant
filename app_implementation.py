"""Core logic of the AI-Powered IT Helpdesk Assistant (no UI code here).

Four capabilities, all offered to Claude as tools so the model decides when
to use each one:

  1. RAG          search_it_knowledge_base  -> Voyage AI embeddings + ChromaDB
  2. Client tool  get_current_datetime      -> plain Python function
  3. Server tool  web_search                -> executed by Anthropic
  4. MCP tools    create_support_ticket, check_ticket_status
                  -> discovered at runtime from the IT Helpdesk MCP Server

`HelpdeskAssistant.run_turn()` is the agentic loop: call Claude, run whatever
tools it asks for, send the results back, and repeat until stop_reason is no
longer "tool_use".
"""

import asyncio
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (override any of these in .env)
# ---------------------------------------------------------------------------
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
EMBEDDING_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3-large")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:3000/mcp")
FAQ_FILE = Path(__file__).parent / "data" / "it_sector.txt"
CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).parent / ".chroma"))  # vector DB on disk
MAX_TOKENS = 1500
MAX_LOOP_TURNS = 10  # safety net only; the loop normally ends on stop_reason

SYSTEM_PROMPT = """You are the IT Helpdesk Assistant of the company. You help employees \
solve IT problems and manage support tickets.

How to decide what to do:
- IT how-to or troubleshooting question: call search_it_knowledge_base FIRST and answer \
from what it returns. Present the steps as a short numbered list.
- The knowledge base has no relevant answer, or the employee asks for the "latest" \
information or explicitly asks you to search the internet: use web_search.
- The employee asks to open, create or raise a ticket: call create_support_ticket. It \
needs the employee name and the issue. Use the employee name given below; if there is \
none and the employee did not say it, ask for it instead of inventing one.
- The employee asks about an existing ticket: call check_ticket_status with the id.
- The employee asks for the current date or time: call get_current_datetime. Never guess it.
- Greetings or small talk: answer directly, no tools.

Rules:
- Never invent ticket ids, ticket statuses or troubleshooting steps. If a tool returns \
an error or nothing relevant, say so plainly and suggest the next step.
- After creating or checking a ticket, show its id, status and priority.
- If the knowledge base steps do not solve the problem, offer to create a ticket.
- Say where the answer came from: the IT knowledge base, a web search, or the ticket system.
- Keep answers concise and practical. Reply in the language the employee writes in."""


class ConfigurationError(RuntimeError):
    """Raised when an API key is missing, so the UI can show a clear message."""


# ---------------------------------------------------------------------------
# 1. RAG: chunk -> embed (Voyage AI) -> store (ChromaDB) -> semantic search
# ---------------------------------------------------------------------------
def chunk_by_section(document_text: str) -> list[str]:
    """One chunk per '## ' heading: each FAQ topic stays whole."""
    chunks = re.split(r"\n(?=## )", document_text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


STOPWORDS = frozenset(
    "a an and are as at be can do does for from how i in is it my not of on or should the to "
    "what when with you your q".split()
)


def _tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


class KnowledgeBase:
    """IT Support FAQ stored in a vector database and searched by meaning.

    Semantic search (Voyage AI + ChromaDB) is the primary path. If the embedding
    service is unavailable or rate limited, search falls back to BM25 keyword
    ranking over the same chunks, so the assistant can still answer.
    """

    def __init__(self, faq_path: Path = FAQ_FILE):
        if not os.getenv("VOYAGE_API_KEY"):
            raise ConfigurationError("VOYAGE_API_KEY is missing. Add it to your .env file.")
        self._voyage = voyageai.Client()  # reads VOYAGE_API_KEY from the environment
        self._collection = chromadb.PersistentClient(path=CHROMA_DIR).get_or_create_collection(
            name=f"it_support_faq_{re.sub(r'[^a-z0-9]+', '_', EMBEDDING_MODEL.lower())}",
            metadata={"hnsw:space": "cosine"},  # compare direction (meaning), not length
        )
        self._chunks = chunk_by_section(faq_path.read_text(encoding="utf-8"))
        self._titles = [chunk.splitlines()[0].lstrip("# ").strip() for chunk in self._chunks]
        self._source = faq_path.name
        self._query_cache: dict[str, list[float]] = {}
        self._index()

    # -- Embeddings ---------------------------------------------------------
    def _embed(self, texts: list[str], input_type: str, retries: int = 0) -> list[list[float]]:
        # input_type matters: "document" for what we store, "query" for what we search with.
        for attempt in range(retries + 1):
            try:
                return self._voyage.embed(texts, model=EMBEDDING_MODEL, input_type=input_type).embeddings
            except voyageai.error.RateLimitError:
                if attempt == retries:
                    raise
                time.sleep(21)  # free-tier Voyage accounts allow about 3 requests per minute

    def _index(self) -> None:
        """Embed and store the FAQ once; later runs reuse the vectors saved on disk."""
        ids = [f"faq-{i}" for i in range(len(self._chunks))]
        stored = self._collection.get(ids=ids)
        if stored["documents"] == self._chunks:
            return  # same content already indexed: no embedding call needed
        if self._collection.count():
            self._collection.delete(ids=self._collection.get()["ids"])  # FAQ changed: re-index
        self._collection.add(
            ids=ids,
            documents=self._chunks,
            embeddings=self._embed(self._chunks, input_type="document", retries=3),  # one batched call
            metadatas=[{"title": title, "source": self._source} for title in self._titles],
        )

    def __len__(self) -> int:
        return len(self._chunks)

    # -- Search -------------------------------------------------------------
    def search(self, query: str, k: int = 2) -> list[dict]:
        """Return the k most relevant FAQ sections, each with a relevance score."""
        try:
            return self._semantic_search(query, k)
        except Exception:
            return self._keyword_search(query, k)

    def _semantic_search(self, query: str, k: int) -> list[dict]:
        if query not in self._query_cache:
            self._query_cache[query] = self._embed([query], input_type="query")[0]
        found = self._collection.query(query_embeddings=[self._query_cache[query]], n_results=min(k, len(self)))
        return [
            {
                "title": meta["title"],
                "source": meta["source"],
                "retrieval": "semantic",
                "similarity": round(1 - distance, 3),
                "content": doc,
            }
            for doc, meta, distance in zip(found["documents"][0], found["metadatas"][0], found["distances"][0])
        ]

    def _keyword_search(self, query: str, k: int, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        """BM25: rank chunks by exact term matches, weighting rare terms higher."""
        docs = [_tokenize(chunk) for chunk in self._chunks]
        avg_len = sum(len(doc) for doc in docs) / len(docs)
        doc_freq = Counter(term for doc in docs for term in set(doc))
        scores = []
        for doc in docs:
            counts = Counter(doc)
            score = 0.0
            for term in set(_tokenize(query)):
                if term not in counts:
                    continue
                idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += idf * counts[term] * (k1 + 1) / (counts[term] + k1 * (1 - b + b * len(doc) / avg_len))
            scores.append(score)
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "title": self._titles[i],
                "source": self._source,
                "retrieval": "keyword_fallback",
                "bm25_score": round(scores[i], 3),
                "content": self._chunks[i],
            }
            for i in ranked
            if scores[i] > 0
        ]


# ---------------------------------------------------------------------------
# 2. Client tool: runs in this process
# ---------------------------------------------------------------------------
def get_current_datetime(date_format: str = "%A, %d %B %Y, %H:%M:%S") -> str:
    """Current local date and time, formatted with a strftime pattern."""
    return datetime.now().astimezone().strftime(date_format or "%A, %d %B %Y, %H:%M:%S")


# ---------------------------------------------------------------------------
# Tool schemas: what Claude reads to decide which tool to call
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_TOOL = {
    "name": "search_it_knowledge_base",
    "description": (
        "Search the company's internal IT Support FAQ by meaning and return the most relevant "
        "sections. Use it first for any IT how-to or troubleshooting question (network, software "
        "installation, slow computer, backups, email, security and malware, printers and hardware, "
        "VPN, mobile devices). Do not use it for ticket operations or for the date and time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The employee's question rewritten as a short search query in English.",
            }
        },
        "required": ["query"],
    },
}

DATETIME_TOOL = {
    "name": "get_current_datetime",
    "description": (
        "Return the current local date and time. Use it whenever the employee asks what day or "
        "time it is, or when you need today's date to reason about deadlines."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "Optional Python strftime pattern, e.g. '%H:%M' for only the time.",
            }
        },
        "required": [],
    },
}

# Server tool: Anthropic runs the search, so there is no local function for it.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}

TOOL_SOURCES = {
    "search_it_knowledge_base": "RAG",
    "get_current_datetime": "Client tool",
    "web_search": "Server tool",
}


# ---------------------------------------------------------------------------
# 4. MCP client: connect, discover tools, relay calls
# ---------------------------------------------------------------------------
async def _mcp_call(fn):
    """Open a fresh MCP connection, run fn(session), then close it."""
    async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()  # MCP handshake: protocol version + capabilities
            return await fn(session)


def discover_mcp_tools() -> list[dict]:
    """Ask the MCP server what tools it offers and convert them to Claude tool schemas."""

    async def _list(session):
        return await session.list_tools()

    result = asyncio.run(_mcp_call(_list))
    return [
        {"name": tool.name, "description": tool.description or tool.name, "input_schema": tool.input_schema}
        for tool in result.tools
    ]


def call_mcp_tool(name: str, arguments: dict) -> tuple[str, bool]:
    """Run a tool on the MCP server. Returns (text_output, is_error)."""

    async def _call(session):
        return await session.call_tool(name, arguments)

    result = asyncio.run(_mcp_call(_call))
    text = "\n".join(block.text for block in result.content if block.type == "text")
    is_error = bool(getattr(result, "is_error", False)) or text.lstrip().startswith('{"error"')
    return text, is_error


def read_all_tickets() -> list[dict]:
    """Read the tickets://all resource (used by the UI sidebar)."""

    async def _read(session):
        return await session.read_resource("tickets://all")

    result = asyncio.run(_mcp_call(_read))
    return json.loads(result.contents[0].text)


# ---------------------------------------------------------------------------
# The assistant: tools + agentic loop
# ---------------------------------------------------------------------------
class HelpdeskAssistant:
    def __init__(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ConfigurationError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
        self.client = Anthropic()
        self.knowledge_base = KnowledgeBase()
        self.mcp_tools: list[dict] = []
        self.mcp_error: str | None = None
        self.refresh_mcp_tools()

    # -- MCP ---------------------------------------------------------------
    def refresh_mcp_tools(self) -> bool:
        """(Re)discover MCP tools. The chatbot keeps working without them."""
        try:
            self.mcp_tools = discover_mcp_tools()
            self.mcp_error = None
        except Exception as error:  # server not running, wrong URL, ...
            self.mcp_tools = []
            self.mcp_error = f"{type(error).__name__}: {error}"
        return bool(self.mcp_tools)

    @property
    def tools(self) -> list[dict]:
        return [KNOWLEDGE_BASE_TOOL, DATETIME_TOOL, WEB_SEARCH_TOOL, *self.mcp_tools]

    # -- Tool execution ----------------------------------------------------
    def _run_tool(self, name: str, tool_input: dict) -> tuple[str, bool]:
        """Dispatch one tool call by name. Returns (output, is_error)."""
        if name == "search_it_knowledge_base":
            return json.dumps(self.knowledge_base.search(tool_input["query"])), False
        if name == "get_current_datetime":
            return get_current_datetime(**tool_input), False
        if name in {tool["name"] for tool in self.mcp_tools}:
            return call_mcp_tool(name, tool_input)
        return f"Unknown tool: {name}", True

    def _run_tools(self, response, events: list[dict]) -> list[dict]:
        """Run every tool_use block in the response (there can be several in one turn)."""
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output, is_error = self._run_tool(block.name, block.input)
            except Exception as error:
                # Report the failure to Claude instead of crashing or hiding it.
                output, is_error = f"{type(error).__name__}: {error}", True
            events.append(
                {
                    "source": TOOL_SOURCES.get(block.name, "MCP"),
                    "tool": block.name,
                    "input": block.input,
                    "output": output,
                    "is_error": is_error,
                }
            )
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
            )
        return results

    # -- Agentic loop ------------------------------------------------------
    def run_turn(self, messages: list[dict], employee_name: str = "") -> dict:
        """Answer the last user message. Appends every new turn to `messages` in place.

        Returns {"text", "events", "sources"}:
          text    final answer
          events  every tool that ran, in order (for the UI trace)
          sources web pages cited by web_search
        """
        if not self.mcp_tools:
            self.refresh_mcp_tools()  # the server may have been started after the app

        system = SYSTEM_PROMPT
        if employee_name.strip():
            system += f"\n\nEmployee name: {employee_name.strip()}"
        if not self.mcp_tools:
            system += (
                "\n\nThe ticket system is currently unavailable, so you cannot create or check "
                "tickets. If asked, say so and suggest trying again later."
            )

        events: list[dict] = []
        sources: dict[str, str] = {}
        text_parts: list[str] = []

        for _ in range(MAX_LOOP_TURNS):
            response = self.client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=messages, tools=self.tools
            )
            messages.append({"role": "assistant", "content": response.content})

            text_parts.clear()  # only the text of the latest response can be the final answer
            for block in response.content:
                if block.type != "text":
                    text_parts.clear()  # keep only what Claude wrote after its last tool activity
                if block.type == "server_tool_use":  # web_search, already executed by Anthropic
                    events.append(
                        {
                            "source": "Server tool",
                            "tool": block.name,
                            "input": block.input,
                            "output": "Executed on Anthropic's servers; results were returned to Claude.",
                            "is_error": False,
                        }
                    )
                elif block.type == "text":
                    text_parts.append(block.text)
                    for citation in getattr(block, "citations", None) or []:
                        url = getattr(citation, "url", None)
                        if url:
                            sources[url] = getattr(citation, "title", None) or url

            if response.stop_reason == "tool_use":
                messages.append({"role": "user", "content": self._run_tools(response, events)})
            elif response.stop_reason == "pause_turn":
                continue  # a long server-tool turn was paused: send it back to resume
            else:
                break  # end_turn (or max_tokens): we are done

        return {
            "text": "".join(text_parts).strip() or "I could not produce an answer. Please try again.",
            "events": events,
            "sources": [{"title": title, "url": url} for url, title in sources.items()],
        }
