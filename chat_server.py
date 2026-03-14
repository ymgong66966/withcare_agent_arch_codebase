"""
Local Chat Server — interactive browser UI for the WithCare agent.

Usage:
    python chat_server.py          # starts on http://localhost:8000
    python chat_server.py --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import Any, Dict, Optional

from decimal import Decimal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


def _strip_decimals(obj):
    """Recursively convert Decimal values to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _strip_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_decimals(v) for v in obj]
    return obj

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
from ddb_client import get_ddb_resource, get_table, USER_REQUEST_TABLE, CHAT_MESSAGES_TABLE
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

# In-memory conversation store  {(user_id, conversation_id): UnifiedState}
# Keyed by (user_id, conv_id) to prevent cross-user state contamination.
_conversations: Dict[tuple, UnifiedState] = {}


def _conv_key(user_id: str, conv_id: str) -> tuple:
    """Build the cache key for the conversations dict.

    user_id must be non-empty — callers are responsible for generating
    a unique one if the client didn't provide it.
    """
    return (user_id, conv_id)

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


async def _write_chat_message(user_id: str, conversation_id: str, role: str, content: str) -> None:
    """Write a single message to the ChatMessages time-series table (fire-and-forget)."""
    table = get_table(CHAT_MESSAGES_TABLE)
    if table is None:
        return
    from datetime import datetime
    from id_utils import new_uuid
    now_iso = datetime.utcnow().isoformat()
    msg_id = new_uuid("msg")
    try:
        table.put_item(Item={
            "user_Id": user_id,
            "sort_key": f"{now_iso}#{msg_id}",
            "chat_Id": conversation_id,
            "role": role,
            "text": content,
            "message_Id": msg_id,
            "dateSent": now_iso,
        })
    except Exception as e:
        _chat_logger.warning(f"ChatMessages write failed (non-fatal): {e}")


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
    """Restore full state from DDB: messages, checkpoint, and requests.

    If the given conv_id has no data (e.g., new browser session after pod
    restart), falls back to the user's most recent conversation so that
    prior messages, checkpoints, and request references are not lost.
    """
    conv_store = get_conversation_store()
    restore_conv_id = conv_id

    # 1. Restore messages (scoped by user_id to prevent cross-user leaks)
    previous_msgs = await conv_store.get_messages_for_conversation(
        restore_conv_id, user_id=state.meta.user_id,
    )

    # Fallback: if no messages found for this conv_id, look up the user's
    # most recent conversation and restore from that instead.
    if not previous_msgs and state.meta.user_id:
        latest_conv_id = await conv_store.get_latest_conversation_id_for_user(
            state.meta.user_id,
        )
        if latest_conv_id and latest_conv_id != conv_id:
            _chat_logger.info(
                f"No data for conv {conv_id[:8]}, falling back to user's "
                f"latest conversation {latest_conv_id[:8]}"
            )
            restore_conv_id = latest_conv_id
            previous_msgs = await conv_store.get_messages_for_conversation(
                restore_conv_id, user_id=state.meta.user_id,
            )
    if previous_msgs:
        restored = []
        for msg in previous_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant", "human") and content:
                restored.append(ChatMessage(role=role, content=content))
        if restored:
            state.messages = restored
            _chat_logger.info(
                f"Restored {len(restored)} messages from DDB for conv {conv_id[:8]}"
            )

    # 2. Restore checkpoint (routing + request manager metadata)
    #    Verify the checkpoint belongs to the requesting user before applying.
    checkpoint = await conv_store.get_checkpoint(restore_conv_id)
    if checkpoint:
        checkpoint_owner = checkpoint.get("user_id", "")
        requesting_user = state.meta.user_id
        if checkpoint_owner and requesting_user and checkpoint_owner != requesting_user:
            _chat_logger.warning(
                f"Checkpoint owner mismatch: checkpoint has user_id={checkpoint_owner}, "
                f"but requesting user is {requesting_user}. Skipping checkpoint restore."
            )
        else:
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

    # Dual-write user message to ChatMessages (fire-and-forget)
    await _write_chat_message(
        user_id=state.meta.user_id,
        conversation_id=state.meta.conversation_id,
        role="user",
        content=user_message,
    )

    state_dict = _strip_decimals(state.model_dump())
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
        state_dict = _strip_decimals(state.model_dump())

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
    assistant_reply = _last_assistant_message(state)
    try:
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

    # Dual-write assistant reply to ChatMessages (fire-and-forget)
    if assistant_reply:
        await _write_chat_message(
            user_id=state.meta.user_id,
            conversation_id=state.meta.conversation_id,
            role="assistant",
            content=assistant_reply,
        )

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
    conversation_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    import traceback

    conv_id = req.conversation_id or str(uuid.uuid4())
    # Never allow empty user_id — generate a unique one per conversation
    # to prevent cache key collisions between anonymous users.
    uid = req.user_id or f"anon-{conv_id[:12]}"
    key = _conv_key(uid, conv_id)

    if key not in _conversations:
        state = _init_state(conv_id, user_id=uid)

        # Restore full state from DDB (messages + checkpoint + requests)
        try:
            await _restore_state(state, conv_id)
        except Exception as e:
            _chat_logger.warning(f"Failed to restore state from DDB: {e}")

        _conversations[key] = state

    state = _conversations[key]
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

    _conversations[key] = state

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

    Returns {content, agent_type} so the lambda can determine
    how to handle the response.
    """
    import traceback

    conv_id = req.conversation_id or str(uuid.uuid4())
    key = _conv_key(req.user_id, conv_id)

    # Extract the last user message text
    user_text = ""
    for msg in reversed(req.messages):
        if msg.get("role") == "user" and msg.get("text"):
            user_text = msg["text"]
            break
    if not user_text:
        return {"content": "", "agent_type": "error", "error": "No user message found"}

    if key not in _conversations:
        state = _init_state(conv_id, user_id=req.user_id)

        # Restore full state from DDB (messages + checkpoint + requests)
        try:
            await _restore_state(state, conv_id)
        except Exception as e:
            _chat_logger.warning(f"Failed to restore state for external/send: {e}")

        _conversations[key] = state

    state = _conversations[key]

    graph = _get_graph()

    try:
        state, debug = await _run_turn(graph, state, user_text)
    except Exception as exc:
        tb = traceback.format_exc()
        _chat_logger.error(f"[external/send] ERROR: {exc}\n{tb}")
        return {"content": f"[Server error] {exc}", "agent_type": "error"}

    _conversations[key] = state

    reply = _last_assistant_message(state)
    current_agent = state.routing.current_agent or ""

    return {
        "content": reply,
        "agent_type": current_agent,
        "conversation_id": conv_id,
    }


class OnboardingIngestRequest(BaseModel):
    user_id: str
    user_data: Optional[Dict] = None
    care_recipients: Optional[list] = None  # [{relationship, data: {...}}]
    assessment_score: Optional[int] = None
    assessment_answers: Optional[list] = None  # [{question, answer}]
    tasks: Optional[list] = None
    tree_qa_pairs: Optional[list] = None  # [{question, answer}] from decision tree Q&A


@app.post("/onboarding/ingest")
async def onboarding_ingest(req: OnboardingIngestRequest):
    """Segment onboarding data into structured fact keys.

    Called after onboarding completes. Parses user/recipient fields,
    mental assessment, tree Q&A, and tasks into WithCare_UserFactTable.
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
            tree_qa_pairs=req.tree_qa_pairs,
        )
        return {"status": "ok", **result}
    except Exception as e:
        _chat_logger.error(f"Onboarding ingest failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/reset")
async def reset(req: ResetRequest):
    # Remove all cache entries for this conversation_id (any user)
    keys_to_remove = [k for k in _conversations if k[1] == req.conversation_id]
    for k in keys_to_remove:
        _conversations.pop(k, None)
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
