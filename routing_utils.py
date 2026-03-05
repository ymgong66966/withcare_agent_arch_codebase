from __future__ import annotations
from typing import Dict, Any, Optional

def last_user_text(state: Dict[str, Any]) -> str:
    for m in reversed(state.get("messages", []) or []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip().lower()
    return ""

def get_active_request(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rm = state.get("request_manager") or {}
    rid = rm.get("active_request_id")
    if not rid:
        return None
    return ((rm.get("requests") or {}).get(rid))

def user_explicit_switch(text: str) -> bool:
    switch_signals = ["算了", "不做了", "换一个", "先不", "stop", "never mind", "forget it", "另外", "新问题"]
    return any(s in text for s in switch_signals)

def infer_turn_mode(state: Dict[str, Any]) -> str:
    text = last_user_text(state)
    if user_explicit_switch(text):
        return "new_intent"

    req = get_active_request(state)
    if req:
        status = req.get("status")
        if status in ["paused", "completed", "aborted"]:
            return "new_intent"
        if req.get("awaiting_user_input") is True:
            return "continuation"
        gate = req.get("prereq_gate") or {}
        if gate.get("status") == "proposed":
            return "continuation"
        ds = req.get("deep_search_state") or {}
        if ds.get("awaiting_user_input") is True:
            return "continuation"

    return "continuation"
