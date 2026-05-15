"""
Virtual SDR Agent — FastAPI backend
Three-stage pipeline: Signals → Contacts → Emails
"""

import asyncio
import json
import os
import logging
import re
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import hashlib
import secrets

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import anthropic
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from skills.prompts import SIGNAL_MONITORING_SKILL, LEAD_SOURCING_SKILL, EMAIL_COPYWRITE_SKILL

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sdr-agent")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY       = os.getenv("ANTHROPIC_API_KEY", "")
PRIUS_SIGNALS_BASE_URL  = os.getenv("PRIUS_SIGNALS_BASE_URL", "https://signals.priusintelli.com")
PRIUS_SIGNALS_TOKEN     = os.getenv("PRIUS_SIGNALS_TOKEN", "")
HUBSPOT_TOKEN           = os.getenv("HUBSPOT_TOKEN", "")
APOLLO_API_KEY          = os.getenv("APOLLO_API_KEY", "")
MODEL                   = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_TOKENS              = 8096

# Auth — password stored as SHA-256 hex digest to avoid plain-text comparison
APP_PASSWORD_HASH = os.getenv("APP_PASSWORD_HASH", "")   # sha256 of your chosen password
APP_PASSWORD      = os.getenv("APP_PASSWORD", "")         # OR plain-text (hashed on first use)
SECRET_KEY        = os.getenv("SECRET_KEY", secrets.token_hex(32))


def _verify_password(submitted: str) -> bool:
    """Check submitted password against configured hash or plain-text password."""
    if APP_PASSWORD_HASH:
        h = hashlib.sha256(submitted.encode()).hexdigest()
        return secrets.compare_digest(h, APP_PASSWORD_HASH)
    if APP_PASSWORD:
        return secrets.compare_digest(submitted, APP_PASSWORD)
    return False

# ---------------------------------------------------------------------------
# In-memory pipeline state
# ---------------------------------------------------------------------------
pipeline_state: dict[str, Any] = {
    "signals": None,
    "approved_signals": None,
    "contacts": None,
    "emails": None,
    "last_run": None,
    "running": False,
}

# ---------------------------------------------------------------------------
# Tool definitions (given to Claude so it knows what it can call)
# ---------------------------------------------------------------------------
TOOL_QUERY_SIGNALS = {
    "name": "query_signals",
    "description": "Search and filter Prius Signals. Returns a list of signals matching the given criteria.",
    "input_schema": {
        "type": "object",
        "properties": {
            "topicId":    {"type": "string",  "description": "Filter by topic ID"},
            "dateRange":  {"type": "string",  "enum": ["today", "week", "month", "all"]},
            "confidence": {"type": "string",  "enum": ["high", "medium", "low", "all"]},
            "relevance":  {"type": "string",  "enum": ["active", "interested", "dismissed", "all"]},
            "limit":      {"type": "integer", "minimum": 1, "maximum": 100},
            "sortBy":     {"type": "string",  "enum": ["createdAt", "confidence"]},
            "sortOrder":  {"type": "string",  "enum": ["asc", "desc"]},
        },
    },
}

TOOL_HUBSPOT_SEARCH = {
    "name": "search_crm_objects",
    "description": "Search HubSpot CRM contacts by job title, company, or other properties.",
    "input_schema": {
        "type": "object",
        "required": ["objectType"],
        "properties": {
            "objectType": {"type": "string", "description": "CRM object type, e.g. 'contacts'"},
            "filterGroups": {
                "type": "array",
                "description": "Array of filter groups (OR logic between groups, AND within a group)",
                "items": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "propertyName": {"type": "string"},
                                    "operator": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                            },
                        }
                    },
                },
            },
            "properties": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of property names to return",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    },
}

TOOL_APOLLO_PEOPLE_SEARCH = {
    "name": "apollo_mixed_people_api_search",
    "description": "Search Apollo.io for people/contacts by company name, title, or other criteria.",
    "input_schema": {
        "type": "object",
        "properties": {
            "q_organization_name": {"type": "string", "description": "Company name to search within"},
            "person_titles":       {"type": "array",  "items": {"type": "string"}, "description": "Job titles to filter by"},
            "page":                {"type": "integer"},
            "per_page":            {"type": "integer", "minimum": 1, "maximum": 25},
        },
    },
}

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def execute_query_signals(params: dict) -> dict:
    """Call Prius Signals via MCP (JSON-RPC over HTTP)."""
    mcp_url = f"{PRIUS_SIGNALS_BASE_URL}/mcp"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if PRIUS_SIGNALS_TOKEN:
        headers["Authorization"] = f"Bearer {PRIUS_SIGNALS_TOKEN}"

    arguments = {k: v for k, v in params.items() if v is not None}

    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/call",
        "params": {"name": "query_signals", "arguments": arguments},
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(mcp_url, json=payload, headers=headers)
        resp.raise_for_status()

        if "text/event-stream" in resp.headers.get("content-type", ""):
            return _extract_mcp_sse_result(resp.text)

        data = resp.json()

    if "error" in data:
        err = data["error"]
        raise ValueError(f"MCP error {err.get('code')}: {err.get('message')}")

    return _deduplicate_signals(_extract_mcp_result(data.get("result", {})))


def _deduplicate_signals(data):
    """Remove duplicate signals by id, keeping first occurrence."""
    if isinstance(data, list):
        seen = set()
        deduped = []
        for sig in data:
            sid = sig.get("id") if isinstance(sig, dict) else None
            if sid is None or sid not in seen:
                deduped.append(sig)
                if sid is not None:
                    seen.add(sid)
        return deduped
    if isinstance(data, dict) and "signals" in data:
        data["signals"] = _deduplicate_signals(data["signals"])
    return data


def _extract_mcp_result(result: dict) -> dict:
    """Parse the content blocks from an MCP tools/call result."""
    content = result.get("content", [])
    if content and content[0].get("type") == "text":
        try:
            return json.loads(content[0]["text"])
        except (json.JSONDecodeError, KeyError):
            return {"text": content[0].get("text", "")}
    return result


def _extract_mcp_sse_result(sse_body: str) -> dict:
    """Extract the JSON-RPC result from a text/event-stream response body."""
    for line in sse_body.splitlines():
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if "error" in data:
            err = data["error"]
            raise ValueError(f"MCP error {err.get('code')}: {err.get('message')}")
        if "result" in data:
            return _deduplicate_signals(_extract_mcp_result(data["result"]))
    raise ValueError("No valid MCP result found in SSE response")


async def execute_hubspot_search(params: dict) -> dict:
    """Call HubSpot CRM search API."""
    object_type = params.get("objectType", "contacts")
    url = f"https://api.hubapi.com/crm/v3/objects/{object_type}/search"
    headers = {
        "Authorization": f"Bearer {HUBSPOT_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "filterGroups": params.get("filterGroups", []),
        "properties":   params.get("properties", [
            "firstname", "lastname", "email", "jobtitle",
            "company", "mobilephone", "phone", "industry",
        ]),
        "limit": params.get("limit", 50),
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def execute_apollo_people_search(params: dict) -> dict:
    """Call Apollo.io people search API."""
    url = "https://api.apollo.io/api/v1/mixed_people/search"
    headers = {
        "x-api-key":    APOLLO_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "q_organization_name": params.get("q_organization_name", ""),
        "person_titles":       params.get("person_titles", []),
        "page":                params.get("page", 1),
        "per_page":            params.get("per_page", 10),
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def dispatch_tool(name: str, params: dict) -> str:
    """Route a Claude tool_use call to the right implementation."""
    try:
        if name == "query_signals":
            result = await execute_query_signals(params)
        elif name == "search_crm_objects":
            result = await execute_hubspot_search(params)
        elif name == "apollo_mixed_people_api_search":
            result = await execute_apollo_people_search(params)
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result)
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Agentic loop — handles tool_use / tool_result cycles
# ---------------------------------------------------------------------------

async def run_agent(
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    sse_queue: asyncio.Queue | None = None,
) -> str:
    """
    Run Claude with tool-use loop until a final text response is returned.
    Sends SSE progress events to sse_queue if provided.
    Returns the final text content.
    """
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": user_message}]
    max_turns = 10

    for turn in range(max_turns):
        response = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Collect text content and tool calls from this response
        text_parts   = []
        tool_results = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

            elif block.type == "tool_use":
                tool_name   = block.name
                tool_input  = block.input
                tool_use_id = block.id

                if sse_queue:
                    await sse_queue.put({
                        "type": "tool_call",
                        "tool": tool_name,
                        "params": tool_input,
                    })

                result_str = await dispatch_tool(tool_name, tool_input)

                if sse_queue:
                    await sse_queue.put({
                        "type": "tool_result",
                        "tool": tool_name,
                        "preview": result_str[:200],
                    })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_str,
                })

        # Append assistant message
        messages.append({"role": "assistant", "content": response.content})

        # If there were tool calls, send results and continue loop
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            continue

        # No tool calls — we have the final response
        if response.stop_reason in ("end_turn", "stop_sequence"):
            return "\n".join(text_parts).strip()

        # Unexpected stop (e.g. max_tokens)
        return "\n".join(text_parts).strip()

    return '{"error": "Max turns reached without final response"}'


# ---------------------------------------------------------------------------
# Pipeline stage runners
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Extract a JSON substring from raw text using multiple strategies."""
    cleaned = raw.strip()
    # Code fence: ```json ... ``` or ``` ... ```
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', cleaned)
    if m:
        return m.group(1).strip()
    # Find outermost JSON object or array
    for start_char in ('{', '['):
        idx = cleaned.find(start_char)
        if idx != -1:
            return cleaned[idx:]
    return cleaned


async def _parse_with_retry(
    raw: str,
    system_prompt: str,
    stage_name: str,
    sse_queue: asyncio.Queue | None = None,
    max_retries: int = 2,
) -> dict:
    """Parse JSON from raw text, asking Claude to fix it if parsing fails."""
    for attempt in range(max_retries + 1):
        try:
            return json.loads(_extract_json(raw))
        except json.JSONDecodeError:
            if attempt == max_retries:
                break
            log.warning("%s: JSON parse failed (attempt %d/%d), requesting correction", stage_name, attempt + 1, max_retries)
            if sse_queue:
                await sse_queue.put({
                    "type": "tool_result",
                    "tool": "json_retry",
                    "preview": f"Response was not valid JSON — retrying (attempt {attempt + 2})…",
                })
            raw = await run_agent(
                system_prompt=system_prompt,
                user_message=(
                    "Your previous response was not valid JSON. "
                    "Respond with ONLY the JSON object — no markdown, no code fences, no explanation. "
                    f"Previous response:\n\n{raw[:2000]}"
                ),
                tools=[],
                sse_queue=None,
            )
    log.error("%s: JSON parse failed after %d retries. Raw: %s", stage_name, max_retries, raw[:300])
    return {"error": f"Failed to parse {stage_name} response after {max_retries} retries", "raw": raw[:500]}


async def run_signals_stage(sse_queue: asyncio.Queue | None = None) -> dict:
    """Tab 1: Morning Scout — fetch and rank signals."""
    if sse_queue:
        await sse_queue.put({"type": "stage", "stage": "signals", "status": "running"})

    raw = await run_agent(
        system_prompt=SIGNAL_MONITORING_SKILL,
        user_message=(
            "Run the signal monitoring workflow now. "
            "Query Prius Signals with: sortBy=createdAt, sortOrder=desc, dateRange=month, "
            "relevance=all, limit=100. Filter out NOT_RELEVANT signals, rank the rest "
            "using all five criteria, group by topicName, and return the top 5 per topic as JSON."
        ),
        tools=[TOOL_QUERY_SIGNALS],
        sse_queue=sse_queue,
    )

    result = await _parse_with_retry(raw, SIGNAL_MONITORING_SKILL, "signals", sse_queue)

    pipeline_state["signals"] = result
    if sse_queue:
        await sse_queue.put({"type": "stage", "stage": "signals", "status": "done", "data": result})
    return result


async def run_contacts_stage(
    signals: dict | list,
    sse_queue: asyncio.Queue | None = None,
) -> dict:
    """Tab 2: Coordinator — source and rank contacts per signal."""
    if sse_queue:
        await sse_queue.put({"type": "stage", "stage": "contacts", "status": "running"})

    # Build signal list — accepts either approved flat list or full {topics:[]} structure
    signal_list = []
    if isinstance(signals, list):
        for sig in signals:
            signal_list.append({
                "id":      sig.get("id", ""),
                "title":   sig.get("title", ""),
                "company": sig.get("company", ""),
                "topic":   sig.get("topicName", ""),
                "summary": sig.get("summary", ""),
                "owner":   sig.get("owner", ""),
            })
    else:
        for topic in signals.get("topics", []):
            for sig in topic.get("signals", [])[:5]:
                signal_list.append({
                    "id":      sig.get("id", ""),
                    "title":   sig.get("title", ""),
                    "company": sig.get("company", ""),
                    "topic":   sig.get("topicName", ""),
                    "summary": sig.get("summary", ""),
                })

    if not signal_list:
        result = {"signals": [], "error": "No signals to source contacts for"}
        pipeline_state["contacts"] = result
        return result

    user_message = (
        "Source contacts for these signals. For EACH signal, search both HubSpot and Apollo "
        "for contacts at the company. Deduplicate, rank, and return the top 5 per signal.\n\n"
        "Signals:\n" + json.dumps(signal_list, indent=2)
    )

    raw = await run_agent(
        system_prompt=LEAD_SOURCING_SKILL,
        user_message=user_message,
        tools=[TOOL_HUBSPOT_SEARCH, TOOL_APOLLO_PEOPLE_SEARCH],
        sse_queue=sse_queue,
    )

    result = await _parse_with_retry(raw, LEAD_SOURCING_SKILL, "contacts", sse_queue)

    pipeline_state["contacts"] = result
    if sse_queue:
        await sse_queue.put({"type": "stage", "stage": "contacts", "status": "done", "data": result})
    return result


async def run_emails_stage(
    signals: dict,
    contacts: dict,
    sse_queue: asyncio.Queue | None = None,
    approved_contacts: list | None = None,
) -> dict:
    """Tab 3: Email Campaigns — generate 3-touch sequences."""
    if sse_queue:
        await sse_queue.put({"type": "stage", "stage": "emails", "status": "running"})

    # Build signal context map
    signal_map: dict[str, dict] = {}
    if isinstance(signals, list):
        for sig in signals:
            signal_map[str(sig.get("id", ""))] = sig
    else:
        for topic in signals.get("topics", []):
            for sig in topic.get("signals", [])[:5]:
                signal_map[str(sig.get("id", ""))] = sig

    # Build campaign inputs from approved contacts (if provided) or full contacts result
    campaign_inputs = []
    if approved_contacts:
        for entry in approved_contacts:
            sig_id  = str(entry.get("signalId", ""))
            sig_ctx = signal_map.get(sig_id, {})
            contact = entry.get("contact", {})
            campaign_inputs.append({
                "signalId":      sig_id,
                "signalTitle":   entry.get("signalTitle", ""),
                "company":       entry.get("company", ""),
                "signalSummary": sig_ctx.get("summary", ""),
                "outreachHook":  sig_ctx.get("outreachHook", ""),
                "intentScore":   sig_ctx.get("intentScore", ""),
                "contact": {
                    "firstName": contact.get("firstName", ""),
                    "lastName":  contact.get("lastName", ""),
                    "title":     contact.get("title", ""),
                    "email":     contact.get("email", ""),
                },
            })
    else:
        for sig_entry in contacts.get("signals", []):
            sig_id    = str(sig_entry.get("signalId", ""))
            sig_ctx   = signal_map.get(sig_id, {})
            for contact in sig_entry.get("contacts", [])[:5]:
                campaign_inputs.append({
                    "signalId":      sig_id,
                    "signalTitle":   sig_entry.get("signalTitle", ""),
                    "company":       sig_entry.get("company", ""),
                    "signalSummary": sig_ctx.get("summary", ""),
                    "outreachHook":  sig_ctx.get("outreachHook", ""),
                    "intentScore":   sig_ctx.get("intentScore", ""),
                    "contact": {
                        "firstName": contact.get("firstName", ""),
                        "lastName":  contact.get("lastName", ""),
                        "title":     contact.get("title", ""),
                        "email":     contact.get("email", ""),
                    },
                })

    if not campaign_inputs:
        result = {"campaigns": [], "error": "No contacts to generate emails for"}
        pipeline_state["emails"] = result
        return result

    user_message = (
        "Generate a 3-touch email sequence for each of these signal + contact pairs. "
        "Use the signal context and outreach hook to personalize each email.\n\n"
        "Campaign inputs:\n" + json.dumps(campaign_inputs, indent=2)
    )

    raw = await run_agent(
        system_prompt=EMAIL_COPYWRITE_SKILL,
        user_message=user_message,
        tools=[],   # No external tools needed — pure Claude generation
        sse_queue=sse_queue,
    )

    result = await _parse_with_retry(raw, EMAIL_COPYWRITE_SKILL, "emails", sse_queue)

    pipeline_state["emails"] = result
    if sse_queue:
        await sse_queue.put({"type": "stage", "stage": "emails", "status": "done", "data": result})
    return result


# ---------------------------------------------------------------------------
# SSE event generator
# ---------------------------------------------------------------------------

async def sse_generator(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Convert queue events to SSE format."""
    while True:
        event = await queue.get()
        if event is None:
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"
            break
        yield "data: " + json.dumps(event) + "\n\n"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
scheduler = AsyncIOScheduler()

async def scheduled_pipeline():
    log.info("Scheduled pipeline run triggered (Thursday 7am CT)")
    if pipeline_state["running"]:
        log.info("Pipeline already running — skipping scheduled run")
        return
    pipeline_state["running"] = True
    try:
        signals  = await run_signals_stage()
        if "error" not in signals:
            contacts = await run_contacts_stage(signals)
            if "error" not in contacts:
                await run_emails_stage(signals, contacts)
        pipeline_state["last_run"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        log.error("Scheduled pipeline error: %s", e)
    finally:
        pipeline_state["running"] = False


# Thursday = 3 (Mon=0), 7am US/Central = 13:00 UTC (CT is UTC-6 in CDT / UTC-5 in CST)
# Using 13:00 UTC as approximate — adjust SCHEDULE_HOUR_UTC env var if needed
SCHEDULE_HOUR_UTC = int(os.getenv("SCHEDULE_HOUR_UTC", "13"))

scheduler.add_job(
    scheduled_pipeline,
    CronTrigger(day_of_week="thu", hour=SCHEDULE_HOUR_UTC, minute=0, timezone="UTC"),
    id="morning_scout",
    replace_existing=True,
)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Virtual SDR Agent", docs_url=None, redoc_url=None)

# Session middleware must be added before CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ---------------------------------------------------------------------------
# Auth middleware — protects all routes except /login, /logout, /health
# ---------------------------------------------------------------------------
PUBLIC_PATHS = {"/login", "/logout", "/api/health", "/favicon.ico"}

@app.middleware("http")
async def require_auth(request: Request, call_next):
    path = request.url.path
    # Allow public paths and static assets
    if path in PUBLIC_PATHS or path.startswith("/static/"):
        return await call_next(request)

    if not request.session.get("authenticated"):
        if path.startswith("/api/"):
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        return RedirectResponse("/login", status_code=302)

    return await call_next(request)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=60 * 60 * 12)


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/login")
async def login_page():
    return FileResponse("frontend/login.html")


@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    if not APP_PASSWORD and not APP_PASSWORD_HASH:
        # No password configured — allow access (useful for first setup check)
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=302)

    if _verify_password(password):
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=302)

    return RedirectResponse("/login?error=1", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.on_event("startup")
async def startup():
    scheduler.start()
    log.info("Scheduler started. Thursday 7am CT pipeline scheduled.")


@app.on_event("shutdown")
async def shutdown():
    scheduler.shutdown()


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


# ---------------------------------------------------------------------------
# API routes — individual stages
# ---------------------------------------------------------------------------

@app.post("/api/signals/run")
async def api_run_signals():
    if not ANTHROPIC_API_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not configured")

    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        try:
            await run_signals_stage(queue)
        except Exception as e:
            await queue.put({"type": "error", "stage": "signals", "message": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run())

    return StreamingResponse(
        sse_generator(queue),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/contacts/run")
async def api_run_contacts(request: Request):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not configured")

    body: dict = {}
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
    except Exception:
        pass

    approved_signals = body.get("approvedSignals")
    signals_input = approved_signals or pipeline_state.get("signals")
    if not signals_input:
        raise HTTPException(400, "No signals available. Run Scout and approve signals first.")

    if approved_signals:
        pipeline_state["approved_signals"] = approved_signals

    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        try:
            await run_contacts_stage(signals_input, queue)
        except Exception as e:
            await queue.put({"type": "error", "stage": "contacts", "message": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run())

    return StreamingResponse(
        sse_generator(queue),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/emails/run")
async def api_run_emails(request: Request):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not configured")

    body: dict = {}
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
    except Exception:
        pass

    approved_contacts = body.get("approvedContacts")
    if not approved_contacts and not pipeline_state.get("contacts"):
        raise HTTPException(400, "No contacts available. Run Greeter and approve contacts first.")

    signals_src  = pipeline_state.get("approved_signals") or pipeline_state.get("signals") or {}
    contacts_src = pipeline_state.get("contacts") or {}

    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        try:
            await run_emails_stage(signals_src, contacts_src, queue, approved_contacts=approved_contacts)
        except Exception as e:
            await queue.put({"type": "error", "stage": "emails", "message": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run())

    return StreamingResponse(
        sse_generator(queue),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Full pipeline run
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/run")
async def api_run_pipeline():
    if pipeline_state["running"]:
        raise HTTPException(409, "Pipeline already running")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not configured")

    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        pipeline_state["running"] = True
        try:
            await queue.put({"type": "pipeline_start"})

            # Stage 1
            signals = await run_signals_stage(queue)
            if "error" in signals:
                await queue.put({"type": "pipeline_error", "stage": "signals", "message": signals["error"]})
                return

            # Stage 2
            contacts = await run_contacts_stage(signals, queue)
            if "error" in contacts:
                await queue.put({"type": "pipeline_error", "stage": "contacts", "message": contacts["error"]})
                return

            # Stage 3
            emails = await run_emails_stage(signals, contacts, queue)
            if "error" in emails:
                await queue.put({"type": "pipeline_error", "stage": "emails", "message": emails["error"]})
                return

            pipeline_state["last_run"] = datetime.now(timezone.utc).isoformat()
            await queue.put({"type": "pipeline_complete", "lastRun": pipeline_state["last_run"]})

        except Exception as e:
            log.error("Pipeline error: %s", e)
            await queue.put({"type": "pipeline_error", "message": str(e)})
        finally:
            pipeline_state["running"] = False
            await queue.put(None)

    asyncio.create_task(run())

    return StreamingResponse(
        sse_generator(queue),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# State & schedule endpoints
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def api_get_state():
    return {
        "signals":  pipeline_state["signals"],
        "contacts": pipeline_state["contacts"],
        "emails":   pipeline_state["emails"],
        "lastRun":  pipeline_state["last_run"],
        "running":  pipeline_state["running"],
        "nextRun":  _next_run(),
    }


@app.delete("/api/state")
async def api_clear_state():
    pipeline_state.update({"signals": None, "approved_signals": None, "contacts": None, "emails": None})
    return {"cleared": True}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "configured": {
            "anthropic": bool(ANTHROPIC_API_KEY),
            "prius_signals": bool(PRIUS_SIGNALS_TOKEN),
            "hubspot": bool(HUBSPOT_TOKEN),
            "apollo": bool(APOLLO_API_KEY),
        },
    }


def _next_run() -> str | None:
    job = scheduler.get_job("morning_scout")
    if job and job.next_run_time:
        return job.next_run_time.isoformat()
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
