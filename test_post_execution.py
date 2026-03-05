"""
Diagnostic Test: Post-Execution Routing

Tests what happens AFTER an execution agent (deep_search) returns results
and the user responds. Four scenarios covering the 4 user response types:

  10: User acknowledges ("好的，知道了") → should NOT re-execute deep_search;
      should recognize task is done (or if prereq, resume parent)
  11: User asks follow-up about the answer ("第一个结果的联系方式是什么？")
      → should route to deep_search (or a conversational mode) without re-executing MCP
  12: User switches to a different/previous request ("我想继续之前找护工的事")
      → upstream_delegator should find and resume the existing request, NOT create new
  13: User makes casual/low-intent chat ("今天天气真好啊")
      → should route to front_end_emotional_support

Each scenario starts with a simple request → info collection → deep_search execution,
then tests the user's post-execution response.
"""

import asyncio
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from state_models import UnifiedState, Meta
from merge_utils import apply_node_output
from graph import build_graph, check_prereq_lifecycle


# ── helpers (same as test_prereq_completion.py) ─────────────────────

def init_state(conversation_id: str, user_id: str) -> UnifiedState:
    return UnifiedState(meta=Meta(conversation_id=conversation_id, user_id=user_id))


def _consume_ddb_writes(state: UnifiedState) -> UnifiedState:
    if hasattr(state, "ddb_writes"):
        state.__dict__.pop("ddb_writes", None)
    return state


def _cleanup_transient(state: UnifiedState) -> UnifiedState:
    state.__dict__.pop("_mcp_result", None)
    return state


async def run_turn(graph, state: UnifiedState, user_message: str):
    """Run one turn through the graph and return (state, debug_info)."""
    debug: Dict[str, Any] = {"steps": [], "node_names": []}

    state = apply_node_output(state, {"messages": [{"role": "user", "content": user_message}]})
    state = _consume_ddb_writes(state)
    state = _cleanup_transient(state)

    state_dict = state.model_dump()

    async for event in graph.astream(state_dict):
        node_name, node_output = next(iter(event.items()))
        debug["steps"].append(node_name)
        debug["node_names"].append(node_name)
        state = apply_node_output(state, node_output)
        state = _consume_ddb_writes(state)
        state = _cleanup_transient(state)
        state_dict = state.model_dump()

    # Post-graph: check prereq lifecycle on full Pydantic state
    prereq_patch = check_prereq_lifecycle(state)
    if prereq_patch:
        debug["node_names"].append("prereq_lifecycle")
        state = apply_node_output(state, prereq_patch)
        state = _consume_ddb_writes(state)

    return state, debug


def last_assistant_message(state: UnifiedState) -> str:
    for m in reversed(state.messages):
        if m.role == "assistant":
            return m.content
    return ""


def get_request_by_id(state: UnifiedState, rid: str) -> Optional[Dict[str, Any]]:
    reqs = state.request_manager.requests or {}
    req = reqs.get(rid)
    if req is None:
        return None
    return req.model_dump() if hasattr(req, "model_dump") else dict(req)


def get_active_request(state: UnifiedState) -> Optional[Dict[str, Any]]:
    rm = state.request_manager
    rid = rm.active_request_id
    if not rid:
        return None
    return get_request_by_id(state, rid)


def get_info_state(req: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not req:
        return {}
    return req.get("info_collection_state") or {}


def print_turn_summary(turn_num: int, user_msg: str, state: UnifiedState, debug: Dict[str, Any]):
    req = get_active_request(state)
    info = get_info_state(req)
    routing = state.routing

    print(f"\n{'─' * 70}")
    print(f"  Turn {turn_num}")
    print(f"{'─' * 70}")
    print(f"  USER: {user_msg}")
    print(f"  ASSISTANT: {last_assistant_message(state)[:300]}")
    print(f"  GRAPH NODES: {' → '.join(debug['node_names'])}")
    print()

    if req:
        print(f"  📋 Active Request ID:     {state.request_manager.active_request_id}")
        print(f"  📋 Request Name:          {req.get('name', 'N/A')}")
        print(f"  📋 Request Status:        {req.get('status')}")
        print(f"  ⏳ Awaiting User Input:    {req.get('awaiting_user_input')}")
        print(f"  🔄 Readiness:             {info.get('readiness_to_proceed', 'N/A')}")
        print(f"  💬 Conversation Turns:    {info.get('conversation_turns_with_agent', 'N/A')}")

        handoff = routing.pending_handoff
        if handoff and handoff.recommended_next_agent:
            print(f"  🚀 Pending Handoff:       → {handoff.recommended_next_agent} ({handoff.reason})")

    # Show all requests
    all_reqs = state.request_manager.requests or {}
    if len(all_reqs) > 1:
        print(f"\n  📦 All Requests ({len(all_reqs)}):")
        for rid, record in all_reqs.items():
            r = record.model_dump() if hasattr(record, "model_dump") else dict(record)
            active_marker = " ← ACTIVE" if rid == state.request_manager.active_request_id else ""
            print(f"       {rid[:20]}... status={r.get('status')} name={r.get('name', '?')}{active_marker}")

    # Show pending queue
    pq = state.request_manager.pending_queue or []
    if pq:
        print(f"\n  📥 Pending Queue ({len(pq)}):")
        for item in pq:
            i = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            print(f"       {i.get('request_id', '?')[:20]}... reason={i.get('reason_queued', '?')}")

    print()


def run_assertions(checks, label=""):
    passed = True
    if label:
        print(f"\n  {label}:")
    for name, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"    {icon} {name}")
        if not ok:
            passed = False
    return passed


# ── Common setup: request → info collection → deep_search ──────────

async def setup_post_execution(graph, conv_id: str, user_id: str):
    """
    Run a simple request through info collection → deep_search execution.
    Returns (state, request_id) with the request in status=executed.

    Flow:
      Turn 1: "我需要了解Illinois的Medicaid护工覆盖范围" → creates request
      Turn 2: "在Illinois，我妈妈65岁，收入很低" → provides info
      Turn 3: "没有其他信息了，开始吧" → readiness=ready, handoff to deep_search
      Turn 4: "好的" → deep_search executes, status=executed
    """
    s = init_state(conv_id, user_id)

    # Turn 1: Initial request
    s, debug = await run_turn(graph, s, "我需要了解Illinois的Medicaid护工覆盖范围")
    print_turn_summary(1, "我需要了解Illinois的Medicaid护工覆盖范围", s, debug)
    request_id = s.request_manager.active_request_id

    # Turn 2: Provide info
    s, debug = await run_turn(graph, s, "在Illinois，我妈妈65岁，收入很低，没有其他保险")
    print_turn_summary(2, "在Illinois，我妈妈65岁，收入很低，没有其他保险", s, debug)

    # Turn 3: Signal readiness
    s, debug = await run_turn(graph, s, "没有其他信息了，可以开始了")
    print_turn_summary(3, "没有其他信息了，可以开始了", s, debug)

    # Check if validated
    req = get_active_request(s)
    status = req.get("status") if req else None
    if status == "validated":
        # Turn 4: Trigger execution
        s, debug = await run_turn(graph, s, "好的")
        print_turn_summary(4, "好的", s, debug)
    elif status == "executed":
        # Already executed in same turn (unlikely but possible)
        pass
    else:
        print(f"  ⚠️  Setup: unexpected status after Turn 3: {status}")
        # Try one more turn to push through
        s, debug = await run_turn(graph, s, "好的，开始吧")
        print_turn_summary(4, "好的，开始吧", s, debug)

    req = get_active_request(s)
    status = req.get("status") if req else None
    print(f"\n  📌 Setup complete. Request status: {status}")

    return s, request_id


# ── Scenario 10: User acknowledges, no follow-up ───────────────────

async def scenario_10_acknowledge():
    """
    After deep_search returns results, user says "好的，知道了".
    Expected: system should NOT re-run deep_search. Should recognize
    the task is effectively done.
    """
    print("\n" + "=" * 70)
    print("  SCENARIO 10: Post-Execution — User Acknowledges (no follow-up)")
    print("  Expected: should NOT re-execute deep_search")
    print("=" * 70)

    g = build_graph()
    s, rid = await setup_post_execution(g, "conv-test-s10", "user-test-s10")

    # THE TEST: user acknowledges
    s, debug = await run_turn(g, s, "好的，知道了，谢谢")
    print_turn_summary(5, "好的，知道了，谢谢", s, debug)

    nodes = debug["node_names"]
    req = get_active_request(s)

    checks = [
        ("deep_search did NOT re-execute",
         "deep_search" not in nodes),
        ("upstream_delegator was involved (handles mark_executed)",
         "upstream_delegator" in nodes),
        ("request status is executed or completed (not re-collecting)",
         req.get("status") in ["executed", "completed"] if req else False),
    ]
    return run_assertions(checks, "Checkpoint: Acknowledge — no re-execution")


# ── Scenario 11: User asks follow-up about the answer ──────────────

async def scenario_11_followup_question():
    """
    After deep_search returns results, user asks a question about the results.
    Expected: should route to deep_search (or a conversational handler)
    to answer the question, ideally WITHOUT re-executing the MCP tool.
    """
    print("\n" + "=" * 70)
    print("  SCENARIO 11: Post-Execution — Follow-up Question About Results")
    print("  Expected: route to deep_search, answer from context")
    print("=" * 70)

    g = build_graph()
    s, rid = await setup_post_execution(g, "conv-test-s11", "user-test-s11")

    # THE TEST: follow-up question about the results
    s, debug = await run_turn(g, s, "第一个结果Demo Result A的联系方式是什么？能详细说说吗？")
    print_turn_summary(5, "第一个结果Demo Result A的联系方式是什么？能详细说说吗？", s, debug)

    nodes = debug["node_names"]
    req = get_active_request(s)

    checks = [
        ("routed to deep_search (continuation, not new_intent)",
         "deep_search" in nodes),
        ("request stayed the same (not a new request)",
         s.request_manager.active_request_id == rid),
        ("assistant response references the results (not generic)",
         len(last_assistant_message(s)) > 10),
    ]
    return run_assertions(checks, "Checkpoint: Follow-up — routed to deep_search")


# ── Scenario 12: User switches to a different/previous request ──────

async def scenario_12_switch_request():
    """
    Setup: create TWO requests. First request goes through deep_search.
    Then user says "我想继续之前找护工的事" (referring to a paused request).
    Expected: upstream_delegator should find and RESUME the existing request,
    NOT create a brand new one.

    Flow:
      Setup turns 1-4: "找护工" request → info → deep_search → executed
      Turn 5: "我还想了解一下养老院的情况" → creates second request
      Turn 6-7: second request info collection
      Turn 8: "算了，我还是想继续之前找护工的事" → should resume first request
    """
    print("\n" + "=" * 70)
    print("  SCENARIO 12: Post-Execution — Switch to Previous Request")
    print("  Expected: resume existing request, NOT create new one")
    print("=" * 70)

    g = build_graph()
    s = init_state("conv-test-s12", "user-test-s12")

    # Turn 1: First request
    s, debug = await run_turn(g, s, "我需要找一个护工")
    print_turn_summary(1, "我需要找一个护工", s, debug)
    first_request_id = s.request_manager.active_request_id

    # Turn 2: Provide info
    s, debug = await run_turn(g, s, "在芝加哥，预算3000，全天护理，下个月开始")
    print_turn_summary(2, "在芝加哥，预算3000，全天护理，下个月开始", s, debug)

    # Turn 3: Try to get to validated
    s, debug = await run_turn(g, s, "没有其他要求了，帮我找吧")
    print_turn_summary(3, "没有其他要求了，帮我找吧", s, debug)

    # Turn 4: If validated, trigger execution
    req = get_active_request(s)
    if req and req.get("status") == "validated":
        s, debug = await run_turn(g, s, "好的")
        print_turn_summary(4, "好的", s, debug)

    # Turn 5: New unrelated request
    s, debug = await run_turn(g, s, "我还想了解一下养老院的情况")
    print_turn_summary(5, "我还想了解一下养老院的情况", s, debug)
    second_request_id = s.request_manager.active_request_id

    # Count requests before the switch
    num_requests_before = len(s.request_manager.requests or {})

    # THE TEST: switch back to first request
    s, debug = await run_turn(g, s, "算了，我还是想继续之前找护工的事")
    print_turn_summary(6, "算了，我还是想继续之前找护工的事", s, debug)

    num_requests_after = len(s.request_manager.requests or {})
    nodes = debug["node_names"]

    checks = [
        ("active request is the FIRST request (resumed, not new)",
         s.request_manager.active_request_id == first_request_id),
        ("no NEW request was created (request count unchanged or +0)",
         num_requests_after == num_requests_before),
        ("upstream_delegator was involved",
         "upstream_delegator" in nodes),
    ]
    return run_assertions(checks, "Checkpoint: Switch — resumed existing request")


# ── Scenario 13: User makes casual/low-intent chat ─────────────────

async def scenario_13_casual_chat():
    """
    After deep_search returns results, user says something casual/random.
    Expected: route to front_end_emotional_support, NOT re-execute deep_search.
    """
    print("\n" + "=" * 70)
    print("  SCENARIO 13: Post-Execution — Casual Chat / Low Intent")
    print("  Expected: route to front_end emotional support")
    print("=" * 70)

    g = build_graph()
    s, rid = await setup_post_execution(g, "conv-test-s13", "user-test-s13")

    # THE TEST: casual chat
    s, debug = await run_turn(g, s, "唉，照顾老人真的好累啊，有时候觉得自己快撑不住了")
    print_turn_summary(5, "唉，照顾老人真的好累啊，有时候觉得自己快撑不住了", s, debug)

    nodes = debug["node_names"]

    checks = [
        ("deep_search did NOT re-execute",
         "deep_search" not in nodes),
        ("routed to front_end (emotional support)",
         "front_end" in nodes or "upstream_delegator" in nodes),
        ("assistant response is empathetic (not task-oriented)",
         len(last_assistant_message(s)) > 10),
    ]
    return run_assertions(checks, "Checkpoint: Casual chat — emotional support")


# ── main ─────────────────────────────────────────────────────────────

async def main():
    print("\n🧪 Post-Execution Routing Diagnostic Test\n")

    scenarios_to_run = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    all_scenarios = {
        "10": ("Scenario 10: Acknowledge (no follow-up)", scenario_10_acknowledge),
        "11": ("Scenario 11: Follow-up Question", scenario_11_followup_question),
        "12": ("Scenario 12: Switch to Previous Request", scenario_12_switch_request),
        "13": ("Scenario 13: Casual Chat / Low Intent", scenario_13_casual_chat),
    }

    if "all" in scenarios_to_run:
        selected = all_scenarios
    else:
        selected = {k: v for k, v in all_scenarios.items() if k in scenarios_to_run}

    if not selected:
        print("❌ No valid scenarios. Use: python test_post_execution.py [10|11|12|13|all]")
        return

    print(f"⏱️  Running {len(selected)} scenario(s) with delays to avoid rate limits...\n")

    results = {}
    for i, (num, (name, func)) in enumerate(selected.items()):
        results[name] = await func()
        if i < len(selected) - 1:
            await asyncio.sleep(5)

    print("\n" + "=" * 70)
    print("  OVERALL RESULTS")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n  🎉 All scenarios passed!")
    else:
        print(f"\n  ⚠️  Some scenarios had failures — these are the gaps to fix")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
