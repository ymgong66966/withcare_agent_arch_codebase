from __future__ import annotations

from typing import Any, Dict, List, Union
from datetime import datetime
from pydantic import BaseModel

from state_models import UnifiedState, ChatMessage, RequestRecord, Slot, OpenQuestion, Artifact, PendingQueueItem, ToolFailure, ToolRun, PrereqGate, DeepSearchState, Handoff

def _utcnow() -> datetime:
    return datetime.utcnow()

def _as_model(obj: Any, model_cls: type[BaseModel]) -> BaseModel:
    if isinstance(obj, model_cls):
        return obj
    if isinstance(obj, dict):
        return model_cls.model_validate(obj)
    raise TypeError(f"Expected {model_cls.__name__} or dict, got {type(obj)}")

def _ensure_request(state: UnifiedState, request_id: str, name: str = "") -> RequestRecord:
    rm = state.request_manager
    if request_id in rm.requests:
        return rm.requests[request_id]
    rec = RequestRecord(request_id=request_id, name=name or request_id)
    rm.requests[request_id] = rec
    rm.active_request_id = rm.active_request_id or request_id
    return rec

def merge_messages(state: UnifiedState, new_messages: List[Union[ChatMessage, dict]]) -> None:
    for m in new_messages:
        state.messages.append(_as_model(m, ChatMessage))

def upsert_slot(state: UnifiedState, request_id: str, slot: Union[Slot, dict]) -> None:
    rec = _ensure_request(state, request_id)
    s = _as_model(slot, Slot)
    rec.slots = [x for x in rec.slots if x.key != s.key] + [s]
    rec.last_touched_at = _utcnow()

def upsert_open_question(state: UnifiedState, request_id: str, q: Union[OpenQuestion, dict]) -> None:
    rec = _ensure_request(state, request_id)
    oq = _as_model(q, OpenQuestion)
    rec.open_questions.append(oq)
    rec.awaiting_user_input = True
    rec.last_touched_at = _utcnow()

def append_artifact(state: UnifiedState, request_id: str, artifact: Union[Artifact, dict]) -> None:
    rec = _ensure_request(state, request_id)
    a = _as_model(artifact, Artifact)
    rec.artifacts.append(a)
    rec.last_touched_at = _utcnow()

def enqueue_pending(state: UnifiedState, item: Union[PendingQueueItem, dict]) -> None:
    state.request_manager.pending_queue.append(_as_model(item, PendingQueueItem))

def dequeue_pending(state: UnifiedState, request_id: str) -> Union[PendingQueueItem, None]:
    """Remove and return a PendingQueueItem by request_id (LIFO-friendly).
    Returns None if not found.
    
    NOTE: When wiring to a persistent store (DynamoDB / Redis), replace the
    in-memory list mutation below with a conditional delete:
        ddb.delete_item(Key={"pk": f"QUEUE#{conversation_id}", "sk": f"ITEM#{request_id}"})
    """
    pq = state.request_manager.pending_queue
    for i, item in enumerate(pq):
        if item.request_id == request_id:
            return pq.pop(i)
    return None

def append_tool_failure(state: UnifiedState, failure: Union[ToolFailure, dict]) -> None:
    state.tools.tool_failures.append(_as_model(failure, ToolFailure))

def append_tool_run(state: UnifiedState, run: Union[ToolRun, dict]) -> None:
    state.tools.tool_runs.append(_as_model(run, ToolRun))

def apply_node_output(state: UnifiedState, node_output: Union[Dict[str, Any], BaseModel, None]) -> UnifiedState:
    if node_output is None:
        out: Dict[str, Any] = {}
    else:
        out = node_output.model_dump() if isinstance(node_output, BaseModel) else dict(node_output)

    if "messages" in out:
        merge_messages(state, out.pop("messages") or [])

    if "artifacts_append" in out:
        ap = out.pop("artifacts_append") or {}
        if ap.get("request_id") and ap.get("artifact"):
            append_artifact(state, ap["request_id"], ap["artifact"])

    if "slots_upsert" in out:
        su = out.pop("slots_upsert") or {}
        rid = su.get("request_id")
        for s in (su.get("slots") or []):
            upsert_slot(state, rid, s)

    if "open_questions_upsert" in out:
        qu = out.pop("open_questions_upsert") or {}
        rid = qu.get("request_id")
        for q in (qu.get("open_questions") or []):
            upsert_open_question(state, rid, q)

    if "tools" in out:
        tools = out.pop("tools") or {}
        for tf in tools.get("tool_failures", []) or []:
            append_tool_failure(state, tf)
        for tr in tools.get("tool_runs", []) or []:
            append_tool_run(state, tr)

    if "request_manager" in out:
        rm_patch = out.pop("request_manager") or {}
        if "active_request_id" in rm_patch:
            state.request_manager.active_request_id = rm_patch["active_request_id"]
        for item in rm_patch.get("pending_queue", []) or []:
            enqueue_pending(state, item)
        for rid in rm_patch.get("pending_queue_remove", []) or []:
            dequeue_pending(state, rid)
        for rid, rp in (rm_patch.get("requests") or {}).items():
            rec = _ensure_request(state, rid, name=rp.get("name", ""))
            for k, v in rp.items():
                if hasattr(rec, k):
                    if k in {"open_questions", "slots", "artifacts"}:
                        continue
                    if k == "prereq_gate" and isinstance(v, dict):
                        v = PrereqGate.model_validate(v)
                    if k == "deep_search_state" and isinstance(v, dict):
                        v = DeepSearchState.model_validate(v)
                    setattr(rec, k, v)

    if "routing" in out:
        r = out.pop("routing") or {}
        # Clean up legacy _escalation_delivered flag if present
        r.pop("_escalation_delivered", None)
        for k, v in r.items():
            if k == "pending_handoff" and isinstance(v, dict):
                v = Handoff.model_validate(v)
            setattr(state.routing, k, v)

    # leftover => keep as extra
    for k, v in out.items():
        setattr(state, k, v)

    state.meta.last_updated_at = _utcnow()
    return state
