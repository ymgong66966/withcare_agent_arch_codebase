"""
Local Chat Server — interactive browser UI for the WithCare agent.

Usage:
    python chat_server.py          # starts on http://localhost:8000
    python chat_server.py --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)


class _LogCapture(logging.Handler):
    """Collects log records into a list during a request."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord):
        try:
            self.records.append(self.format(record))
        except Exception:
            pass

from state_models import UnifiedState, Meta, ChatMessage, PendingQueueItem, RequestRecord
from merge_utils import apply_node_output
from graph import build_graph, check_prereq_lifecycle
from ddb_client import get_ddb_resource, USER_REQUEST_TABLE
from conversation_store import get_conversation_store

_chat_logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────

app = FastAPI(title="WithCare Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation store  {conversation_id: UnifiedState}
_conversations: Dict[str, UnifiedState] = {}

# Lazily-built graph (built once on first request)
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Helpers (same pattern as executor.py / test files) ───────────────

def _init_state(conversation_id: str, user_id: str = "") -> UnifiedState:
    uid = user_id or f"chat-user-{conversation_id[:8]}"
    return UnifiedState(
        meta=Meta(conversation_id=conversation_id, user_id=uid)
    )


def _sanitize_ddb_value(v):
    """Recursively convert Python types to DynamoDB-safe types."""
    from datetime import datetime
    from decimal import Decimal
    if isinstance(v, datetime):
        return v.isoformat()
    elif isinstance(v, float):
        return Decimal(str(v))
    elif isinstance(v, dict):
        return {k: _sanitize_ddb_value(val) for k, val in v.items() if _sanitize_ddb_value(val) is not None}
    elif isinstance(v, list):
        return [_sanitize_ddb_value(item) for item in v]
    elif v is None:
        return None
    return v


def _consume_ddb_writes(state: UnifiedState) -> UnifiedState:
    """Persist queued DDB writes, then clear the queue from state."""
    writes = getattr(state, "ddb_writes", None)
    if not writes:
        state.ddb_writes = []
        return state

    ddb = get_ddb_resource()
    if ddb is None:
        state.ddb_writes = []
        return state

    for write in writes:
        try:
            op = write.get("op", "put")
            table_name = write.get("table", USER_REQUEST_TABLE)
            if op == "put":
                table = ddb.Table(table_name)
                item = _sanitize_ddb_value(write["item"])
                table.put_item(Item=item)
            elif op == "update":
                table = ddb.Table(table_name)
                params = _sanitize_ddb_value(write["params"])
                table.update_item(**params)
            elif op == "delete":
                table = ddb.Table(table_name)
                table.delete_item(**write["params"])
        except Exception as e:
            _chat_logger.error(f"DDB write failed (non-fatal): {e}")

    state.ddb_writes = []
    return state


def _cleanup_transient(state: UnifiedState) -> UnifiedState:
    state.__dict__.pop("_mcp_result", None)
    return state


def _last_assistant_message(state: UnifiedState) -> str:
    for m in reversed(state.messages):
        if m.role == "assistant":
            return m.content
    return ""


def _build_checkpoint_data(state: UnifiedState) -> Dict[str, Any]:
    """Extract the minimal checkpoint fields from state."""
    rm = state.request_manager
    return {
        "current_agent": state.routing.current_agent,
        "conversation_stage": state.routing.conversation_stage,
        "needs_human": state.routing.needs_human,
        "turn_mode": state.routing.turn_mode,
        "active_request_id": rm.active_request_id,
        "request_ids": list(rm.requests.keys()),
        "pending_queue": [item.model_dump() for item in rm.pending_queue],
    }


def _apply_checkpoint(state: UnifiedState, checkpoint: Dict[str, Any]) -> None:
    """Apply checkpoint data to routing and request_manager metadata."""
    if checkpoint.get("current_agent"):
        state.routing.current_agent = checkpoint["current_agent"]
    if checkpoint.get("conversation_stage"):
        state.routing.conversation_stage = checkpoint["conversation_stage"]
    if checkpoint.get("needs_human"):
        state.routing.needs_human = True  # sticky: only set to True
    if checkpoint.get("turn_mode"):
        state.routing.turn_mode = checkpoint["turn_mode"]

    if checkpoint.get("active_request_id"):
        state.request_manager.active_request_id = checkpoint["active_request_id"]

    for item_dict in checkpoint.get("pending_queue", []):
        try:
            state.request_manager.pending_queue.append(
                PendingQueueItem.model_validate(item_dict)
            )
        except Exception:
            pass


def _rebuild_request_manager(
    state: UnifiedState,
    raw_items: Dict[str, Dict[str, Any]],
    checkpoint: Dict[str, Any],
) -> None:
    """Rebuild state.request_manager.requests from DDB items."""
    for rid, item in raw_items.items():
        payload = item.get("payload", {})
        if not payload:
            payload = {
                "request_id": rid,
                "name": item.get("name", ""),
                "goal": item.get("goal", ""),
                "status": item.get("status", "created"),
                "target": item.get("target", "unknown"),
                "priority": item.get("priority", "normal"),
                "request_type": item.get("request_type", ""),
                "subject_entity_id": item.get("subject_entity_id", ""),
            }

        payload["request_id"] = rid

        try:
            record = RequestRecord.model_validate(payload)
            state.request_manager.requests[rid] = record
        except Exception as e:
            _chat_logger.warning(f"Failed to rebuild request {rid}: {e}")

    active_id = checkpoint.get("active_request_id")
    if active_id and active_id in state.request_manager.requests:
        state.request_manager.active_request_id = active_id


async def _restore_state(state: UnifiedState, conv_id: str) -> None:
    """Restore full state from DDB: messages, checkpoint, and requests."""
    conv_store = get_conversation_store()

    # 1. Restore messages
    previous_msgs = await conv_store.get_messages_for_conversation(conv_id)
    if previous_msgs:
        restored = []
        for msg in previous_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                restored.append(ChatMessage(role=role, content=content))
        if restored:
            state.messages = restored
            _chat_logger.info(
                f"Restored {len(restored)} messages from DDB for conv {conv_id[:8]}"
            )

    # 2. Restore checkpoint (routing + request manager metadata)
    checkpoint = await conv_store.get_checkpoint(conv_id)
    if checkpoint:
        _apply_checkpoint(state, checkpoint)
        _chat_logger.info(
            f"Restored checkpoint for conv {conv_id[:8]}: "
            f"agent={checkpoint.get('current_agent')}, "
            f"active_req={checkpoint.get('active_request_id')}"
        )

        # 3. Reload request records from UserRequestTable
        request_ids = checkpoint.get("request_ids", [])
        if request_ids and state.meta.user_id:
            from request_store import get_request_store
            req_store = get_request_store()
            raw_items = await req_store.get_requests_by_ids(
                user_id=state.meta.user_id,
                request_ids=request_ids,
            )
            _rebuild_request_manager(state, raw_items, checkpoint)
            _chat_logger.info(
                f"Restored {len(raw_items)} requests from DDB"
            )


async def _run_turn(graph, state: UnifiedState, user_message: str):
    """Run one conversation turn.  Returns (state, debug_info)."""
    debug: Dict[str, Any] = {"nodes": [], "node_outputs": {}}

    # Create Langfuse trace for this turn
    _lf_trace = None
    try:
        from anthropic_client import langfuse as _lf
        if _lf:
            _lf_trace = _lf.trace(
                name="chat-turn",
                session_id=state.meta.conversation_id,
                user_id=state.meta.user_id,
                input=user_message,
            )
    except Exception:
        pass

    state = apply_node_output(state, {"messages": [{"role": "user", "content": user_message}]})
    state = _consume_ddb_writes(state)
    state = _cleanup_transient(state)

    # Persist user message (fire-and-forget)
    try:
        conv_store = get_conversation_store()
        await conv_store.write_message(
            user_id=state.meta.user_id,
            conversation_id=state.meta.conversation_id,
            role="user",
            content=user_message,
        )
    except Exception as e:
        _chat_logger.warning(f"Failed to persist user message: {e}")

    state_dict = state.model_dump()
    if _lf_trace:
        state_dict["_langfuse_trace"] = _lf_trace

    async for event in graph.astream(state_dict):
        node_name, node_output = next(iter(event.items()))
        debug["nodes"].append(node_name)
        # Capture routing decisions from each node for debugging
        if isinstance(node_output, dict):
            _chat_logger.info(
                f"[debug] node '{node_name}' output keys: {list(node_output.keys())}"
            )
            routing_snapshot = node_output.get("routing")
            if routing_snapshot:
                debug["node_outputs"][node_name] = {
                    k: v for k, v in routing_snapshot.items()
                    if k in ("turn_mode", "turn_reason", "llm_recommended_agent",
                             "current_agent", "conversation_stage", "pending_handoff",
                             "_catcher_next", "delegator_debug")
                }
            # Capture memory/fact debug info from info_collection
            ic_debug = node_output.get("info_collection_debug")
            if ic_debug:
                _chat_logger.info(
                    f"Captured info_collection_debug from '{node_name}': "
                    f"keys={list(ic_debug.keys())}"
                )
                debug["memory_context"] = ic_debug
        state = apply_node_output(state, node_output)
        state = _consume_ddb_writes(state)
        state = _cleanup_transient(state)
        state_dict = state.model_dump()

    # Fallback: read from state.__dict__ (setattr'd by apply_node_output)
    if "memory_context" not in debug:
        ic_fallback = getattr(state, "info_collection_debug", None)
        if ic_fallback:
            _chat_logger.info(
                f"Captured info_collection_debug from state fallback: "
                f"keys={list(ic_fallback.keys()) if isinstance(ic_fallback, dict) else type(ic_fallback)}"
            )
            debug["memory_context"] = ic_fallback

    # Post-graph: prereq lifecycle check (runs on full Pydantic state)
    prereq_patch = check_prereq_lifecycle(state)
    if prereq_patch:
        debug["nodes"].append("prereq_lifecycle")
        state = apply_node_output(state, prereq_patch)
        state = _consume_ddb_writes(state)

    # Persist assistant reply (fire-and-forget)
    try:
        assistant_reply = _last_assistant_message(state)
        if assistant_reply:
            conv_store = get_conversation_store()
            await conv_store.write_message(
                user_id=state.meta.user_id,
                conversation_id=state.meta.conversation_id,
                role="assistant",
                content=assistant_reply,
            )
    except Exception as e:
        _chat_logger.warning(f"Failed to persist assistant message: {e}")

    # Persist checkpoint (fire-and-forget)
    try:
        conv_store = get_conversation_store()
        await conv_store.write_checkpoint(
            conversation_id=state.meta.conversation_id,
            user_id=state.meta.user_id,
            checkpoint_data=_build_checkpoint_data(state),
        )
    except Exception as e:
        _chat_logger.warning(f"Failed to persist checkpoint: {e}")

    # Finalize Langfuse trace with debug metadata
    if _lf_trace:
        try:
            _lf_trace.update(
                output=_last_assistant_message(state),
                metadata={
                    "nodes_visited": debug["nodes"],
                    "current_agent": state.routing.current_agent,
                    "turn_mode": state.routing.turn_mode,
                    "turn_reason": state.routing.turn_reason,
                    "active_request_id": state.request_manager.active_request_id,
                    "needs_human": state.routing.needs_human,
                    "conversation_stage": state.routing.conversation_stage,
                },
            )
        except Exception:
            pass

    return state, debug


# ── Request / response models ────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None


class ResetRequest(BaseModel):
    conversation_id: str


class ExternalSendRequest(BaseModel):
    user_id: str
    messages: list  # [{role, text}]
    needs_human: bool = False
    conversation_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    import traceback

    conv_id = req.conversation_id or str(uuid.uuid4())

    if conv_id not in _conversations:
        state = _init_state(conv_id, user_id=req.user_id or "")

        # Restore full state from DDB (messages + checkpoint + requests)
        try:
            await _restore_state(state, conv_id)
        except Exception as e:
            _chat_logger.warning(f"Failed to restore state from DDB: {e}")

        _conversations[conv_id] = state

    state = _conversations[conv_id]
    graph = _get_graph()

    # Capture logs for this request
    log_capture = _LogCapture()
    log_capture.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_capture)

    try:
        state, debug = await _run_turn(graph, state, req.message)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[chat] ERROR in run_turn: {exc}\n{tb}")
        return {
            "reply": f"[Server error] {exc}",
            "conversation_id": conv_id,
            "debug": {"error": str(exc), "traceback": tb, "logs": log_capture.records},
        }
    finally:
        root_logger.removeHandler(log_capture)

    _conversations[conv_id] = state

    # Build debug payload
    active_req = None
    rm = state.request_manager
    if rm.active_request_id and rm.active_request_id in rm.requests:
        r = rm.requests[rm.active_request_id]
        info_state = r.info_collection_state or {}
        active_req = {
            "request_id": r.request_id,
            "name": r.name,
            "status": r.status,
            "stage_detail": r.stage_detail,
            "awaiting_user_input": r.awaiting_user_input,
            "slot_refs": r.slot_refs,
            "request_type": r.request_type,
            "subject_entity_id": r.subject_entity_id,
            "info_collection": {
                "summary": info_state.get("summary_of_collected_info", ""),
                "readiness": info_state.get("readiness_to_proceed", ""),
                "turns": info_state.get("conversation_turns_with_agent", 0),
                "key_info_status": info_state.get("key_info_status", []),
            },
        }

    return {
        "reply": _last_assistant_message(state),
        "conversation_id": conv_id,
        "debug": {
            "nodes_visited": debug["nodes"],
            "node_routing": debug.get("node_outputs", {}),
            "current_agent": state.routing.current_agent,
            "turn_mode": state.routing.turn_mode,
            "turn_reason": state.routing.turn_reason,
            "active_request": active_req,
            "total_requests": len(rm.requests),
            "pending_queue_size": len(rm.pending_queue),
            "memory_context": debug.get("memory_context"),
            "logs": log_capture.records,
        },
    }


@app.post("/external/send")
async def external_send(req: ExternalSendRequest):
    """External endpoint for lambda integration.

    Returns {content, agent_type, needs_human} so the lambda can
    trigger Slack when agent_type == "human_support".
    """
    import traceback

    conv_id = req.conversation_id or str(uuid.uuid4())

    # Extract the last user message text
    user_text = ""
    for msg in reversed(req.messages):
        if msg.get("role") == "user" and msg.get("text"):
            user_text = msg["text"]
            break
    if not user_text:
        return {"content": "", "agent_type": "error", "needs_human": False, "error": "No user message found"}

    if conv_id not in _conversations:
        state = _init_state(conv_id, user_id=req.user_id)

        # Restore full state from DDB (messages + checkpoint + requests)
        try:
            await _restore_state(state, conv_id)
        except Exception as e:
            _chat_logger.warning(f"Failed to restore state for external/send: {e}")

        _conversations[conv_id] = state

    state = _conversations[conv_id]

    # If caller indicates needs_human, set it on state before running
    if req.needs_human:
        state.routing.needs_human = True

    graph = _get_graph()

    try:
        state, debug = await _run_turn(graph, state, user_text)
    except Exception as exc:
        tb = traceback.format_exc()
        _chat_logger.error(f"[external/send] ERROR: {exc}\n{tb}")
        return {"content": f"[Server error] {exc}", "agent_type": "error", "needs_human": False}

    _conversations[conv_id] = state

    reply = _last_assistant_message(state)
    needs_human = getattr(state.routing, "needs_human", False)
    current_agent = state.routing.current_agent or ""

    # When needs_human is True, report agent_type as "human_support" so lambda triggers Slack
    agent_type = "human_support" if needs_human else current_agent

    return {
        "content": reply,
        "agent_type": agent_type,
        "needs_human": needs_human,
        "conversation_id": conv_id,
    }


class OnboardingIngestRequest(BaseModel):
    user_id: str
    user_data: Optional[Dict] = None
    care_recipients: Optional[list] = None  # [{relationship, data: {...}}]
    assessment_score: Optional[int] = None
    assessment_answers: Optional[list] = None  # [{question, answer}]
    tasks: Optional[list] = None


@app.post("/onboarding/ingest")
async def onboarding_ingest(req: OnboardingIngestRequest):
    """Segment onboarding data into structured fact keys.

    Called after onboarding completes. Parses user/recipient fields,
    mental assessment, and tasks into WithCare_UserFactTable.
    """
    from onboarding_fact_bridge import ingest_onboarding_data

    try:
        result = await ingest_onboarding_data(
            user_id=req.user_id,
            user_data=req.user_data,
            care_recipients=req.care_recipients,
            assessment_score=req.assessment_score,
            assessment_answers=req.assessment_answers,
            tasks=req.tasks,
        )
        return {"status": "ok", **result}
    except Exception as e:
        _chat_logger.error(f"Onboarding ingest failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/reset")
async def reset(req: ResetRequest):
    _conversations.pop(req.conversation_id, None)
    return {"status": "ok", "conversation_id": req.conversation_id}


@app.get("/health")
async def health():
    return {"status": "ok", "conversations": len(_conversations)}


# ── Static frontend ──────────────────────────────────────────────────

_UI_DIR = Path(__file__).parent / "chat_ui"


@app.get("/")
async def index():
    return FileResponse(_UI_DIR / "index.html")


if _UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="WithCare Chat Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting WithCare chat server on http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
