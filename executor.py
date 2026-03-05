from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Tuple

from state_models import UnifiedState, Meta
from merge_utils import apply_node_output
from graph import build_graph
from ddb_client import get_ddb_resource, USER_REQUEST_TABLE

_executor_logger = logging.getLogger(__name__)

def init_state(conversation_id: str, user_id: str) -> UnifiedState:
    return UnifiedState(meta=Meta(conversation_id=conversation_id, user_id=user_id))

def _consume_ddb_writes(state: UnifiedState) -> UnifiedState:
    """Persist queued DDB writes, then clear the queue from state."""
    writes = getattr(state, "ddb_writes", None)
    if not writes:
        state.ddb_writes = []
        return state

    ddb = get_ddb_resource()
    if ddb is None:
        # DDB not configured — drop writes silently (log-only mode)
        state.ddb_writes = []
        return state

    for write in writes:
        try:
            op = write.get("op", "put")
            table_name = write.get("table", USER_REQUEST_TABLE)
            if op == "put":
                table = ddb.Table(table_name)
                table.put_item(Item=write["item"])
            elif op == "update":
                table = ddb.Table(table_name)
                table.update_item(**write["params"])
            elif op == "delete":
                table = ddb.Table(table_name)
                table.delete_item(**write["params"])
        except Exception as e:
            _executor_logger.warning(f"DDB write failed (non-fatal): {e}")

    state.ddb_writes = []
    return state

def _cleanup_transient(state: UnifiedState) -> UnifiedState:
    state.__dict__.pop("_mcp_result", None)
    return state

async def run_turn_async(graph, state: UnifiedState, user_message: str) -> Tuple[UnifiedState, Dict[str, Any]]:
    debug: Dict[str, Any] = {"steps": []}

    state = apply_node_output(state, {"messages": [{"role": "user", "content": user_message}]})
    state = _consume_ddb_writes(state)
    state = _cleanup_transient(state)

    state_dict = state.model_dump()

    async for event in graph.astream(state_dict):
        node_name, node_output = next(iter(event.items()))
        debug["steps"].append({"node": node_name, "output_keys": list((node_output or {}).keys()) if isinstance(node_output, dict) else str(type(node_output))})

        state = apply_node_output(state, node_output)
        state = _consume_ddb_writes(state)
        state = _cleanup_transient(state)
        state_dict = state.model_dump()

    return state, debug

def last_assistant_message(state: UnifiedState) -> str:
    for m in reversed(state.messages):
        if m.role == "assistant":
            return m.content
    return ""

async def demo():
    g = build_graph()
    s = init_state("conv-demo-1", "user-demo-1")

    turns = [
        "我想做一件事：帮我规划一下接下来两周如何安排照护和工作。",
        "我希望预算别太高，地点在芝加哥周边。照护对象是我妈妈。",
        "先用现有的继续",
    ]

    for t in turns:
        s, _ = await run_turn_async(g, s, t)
        print("\nUSER:", t)
        print("ASSISTANT:", last_assistant_message(s))

if __name__ == "__main__":
    asyncio.run(demo())
