"""Progress signal queue registry for streaming subagent status to callers."""

import asyncio
import time
from typing import Any, Dict

_progress_queues: Dict[str, asyncio.Queue] = {}


def emit_progress(state: Dict[str, Any], agent_name: str) -> None:
    """Fire-and-forget: push status event to the queue for this conversation."""
    try:
        conv_id = (state.get("meta") or {}).get("conversation_id", "")
        q = _progress_queues.get(conv_id)
        if q:
            q.put_nowait({"type": "status", "agent": agent_name, "timestamp": time.time()})
    except Exception:
        pass


def register_queue(conv_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _progress_queues[conv_id] = q
    return q


def unregister_queue(conv_id: str) -> None:
    _progress_queues.pop(conv_id, None)
