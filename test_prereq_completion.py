"""
Diagnostic Test: Prerequisite Completion & Parent Resume

This test verifies the GAP in the current system:
- When a prerequisite request completes (readiness="ready", status="validated"),
  the parent request should be automatically resumed.

Currently expected to FAIL on the "parent resumed" assertions,
proving the gap exists and needs implementation.

Scenarios:
  7: Prereq completes → parent should resume
  8: User abandons prereq mid-way → parent should resume
  9: Nested prereq chain (prereq of prereq) → LIFO resume
"""

import asyncio
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from state_models import UnifiedState, Meta
from merge_utils import apply_node_output
from graph import build_graph, check_prereq_lifecycle


# ── helpers ──────────────────────────────────────────────────────────

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


# ── Scenario 7: Prerequisite Completion & Parent Resume ──────────────

async def scenario_7_prereq_completion():
    """
    Scenario 7: Prerequisite completes → parent request should resume.

    Lifecycle: created → collecting → validated → executed → completed → parent resumes

    Flow:
      Turn 1: "我需要找一个护工"                          → creates caregiver request
      Turn 2: "预算不确定，还没申请Medicaid"               → prereq detected
      Turn 3: "要，先帮我处理Medicaid申请"                 → accept prereq, pause parent
      Turn 4: "在Illinois，我妈妈65岁"                     → answer prereq questions
      Turn 5: "收入很低，资产不多"                          → more prereq info
      Turn 6: "没有其他要求了，可以开始找了"                → prereq info done (validated), handoff to execution
      Turn 7: "好的"                                       → execution agent runs → executed → prereq_lifecycle resumes parent
      Turn 8: "全天护理，希望下个月开始"                    → continue parent request
    """
    print("\n" + "=" * 70)
    print("  SCENARIO 7: Prerequisite Completion & Parent Resume")
    print("  (Full lifecycle: validated → executed → completed → parent resumes)")
    print("=" * 70)

    g = build_graph()
    s = init_state("conv-test-s7", "user-test-s7")

    original_request_id = None
    prereq_request_id = None

    # Turn 1: Initial request
    s, debug = await run_turn(g, s, "我需要找一个护工")
    print_turn_summary(1, "我需要找一个护工", s, debug)
    original_request_id = s.request_manager.active_request_id

    # Turn 2: Mention Medicaid (prereq detection)
    s, debug = await run_turn(g, s, "预算的话我不太确定，因为我还没申请Medicaid，不知道能cover多少")
    print_turn_summary(2, "预算的话我不太确定，因为我还没申请Medicaid...", s, debug)

    # Turn 3: Accept prerequisite
    s, debug = await run_turn(g, s, "要，先帮我处理Medicaid申请")
    print_turn_summary(3, "要，先帮我处理Medicaid申请", s, debug)
    prereq_request_id = s.request_manager.active_request_id

    # Checkpoint: verify prereq was created and parent paused
    checks_prereq_created = [
        ("prereq request created", prereq_request_id is not None and prereq_request_id != original_request_id),
        ("active is prereq", s.request_manager.active_request_id == prereq_request_id),
        ("original paused", (get_request_by_id(s, original_request_id) or {}).get("status") == "paused" if original_request_id else False),
    ]
    p1 = run_assertions(checks_prereq_created, "Checkpoint: Prereq Created")

    # Turn 4: Answer prereq questions
    s, debug = await run_turn(g, s, "在Illinois，我妈妈65岁，没有工作收入")
    print_turn_summary(4, "在Illinois，我妈妈65岁，没有工作收入", s, debug)

    # Turn 5: More prereq info
    s, debug = await run_turn(g, s, "收入很低，资产不多，没有其他保险")
    print_turn_summary(5, "收入很低，资产不多，没有其他保险", s, debug)

    # Turn 6: Finalize prereq info → should become validated with handoff
    s, debug = await run_turn(g, s, "没有其他要求了，可以开始找了")
    print_turn_summary(6, "没有其他要求了，可以开始找了", s, debug)

    # Checkpoint: prereq info done (validated), but NOT yet executed
    prereq_req = get_request_by_id(s, prereq_request_id) if prereq_request_id else None
    checks_prereq_validated = [
        ("prereq status = validated (info done, awaiting execution)",
         prereq_req.get("status") == "validated" if prereq_req else False),
        ("active is still prereq (execution pending)",
         s.request_manager.active_request_id == prereq_request_id),
        ("parent still paused",
         (get_request_by_id(s, original_request_id) or {}).get("status") == "paused" if original_request_id else False),
    ]
    p2 = run_assertions(checks_prereq_validated, "Checkpoint: Prereq Validated (info done, execution pending)")

    # Turn 7: Trigger execution of prereq → execution agent runs → status=executed → prereq_lifecycle fires
    s, debug = await run_turn(g, s, "好的")
    print_turn_summary(7, "好的", s, debug)

    # KEY ASSERTIONS: After prereq execution + lifecycle transition
    prereq_req = get_request_by_id(s, prereq_request_id) if prereq_request_id else None
    parent_req = get_request_by_id(s, original_request_id) if original_request_id else None

    checks_after_prereq_executed = [
        ("prereq status = completed (executed + lifecycle closed it)",
         prereq_req.get("status") == "completed" if prereq_req else False),
        ("active switches back to parent",
         s.request_manager.active_request_id == original_request_id),
        ("parent status = collecting (not paused)",
         parent_req.get("status") == "collecting" if parent_req else False),
        ("pending_queue drained",
         len(s.request_manager.pending_queue) == 0),
    ]
    p3 = run_assertions(checks_after_prereq_executed, "Checkpoint: After Prereq Execution & Parent Resume")

    # Turn 8: Continue parent request
    s, debug = await run_turn(g, s, "全天护理，希望下个月开始")
    print_turn_summary(8, "全天护理，希望下个月开始", s, debug)

    checks_parent_continues = [
        ("active is original request",
         s.request_manager.active_request_id == original_request_id),
        ("parent info_state has summary",
         bool(get_info_state(get_request_by_id(s, original_request_id)).get("summary_of_collected_info", "")) if original_request_id else False),
    ]
    p4 = run_assertions(checks_parent_continues, "Checkpoint: Parent Continues")

    return p1 and p2 and p3 and p4


# ── Scenario 8: Prerequisite Abandon ─────────────────────────────────

async def scenario_8_prereq_abandon():
    """
    Scenario 8: User abandons prerequisite mid-way → parent should resume.

    Flow:
      Turn 1: "我需要找一个护工"
      Turn 2: "预算不确定，还没申请Medicaid"
      Turn 3: "要，先帮我处理Medicaid申请"
      Turn 4: "在Illinois"
      Turn 5: "算了，不申请Medicaid了，直接帮我找护工吧"  ← ABANDON
      --- parent should resume ---
      Turn 6: "预算3000左右，芝加哥"
    """
    print("\n" + "=" * 70)
    print("  SCENARIO 8: Prerequisite Abandon → Parent Resume")
    print("  (Diagnostic — expected to show current gap)")
    print("=" * 70)

    g = build_graph()
    s = init_state("conv-test-s8", "user-test-s8")

    original_request_id = None
    prereq_request_id = None

    # Turn 1-3: Same as Scenario 7
    s, debug = await run_turn(g, s, "我需要找一个护工")
    print_turn_summary(1, "我需要找一个护工", s, debug)
    original_request_id = s.request_manager.active_request_id

    s, debug = await run_turn(g, s, "预算的话我不太确定，因为我还没申请Medicaid，不知道能cover多少")
    print_turn_summary(2, "预算的话我不太确定...", s, debug)

    s, debug = await run_turn(g, s, "要，先帮我处理Medicaid申请")
    print_turn_summary(3, "要，先帮我处理Medicaid申请", s, debug)
    prereq_request_id = s.request_manager.active_request_id

    # Turn 4: Some prereq info
    s, debug = await run_turn(g, s, "在Illinois")
    print_turn_summary(4, "在Illinois", s, debug)

    # Turn 5: ABANDON prereq
    s, debug = await run_turn(g, s, "算了，不申请Medicaid了，直接帮我找护工吧")
    print_turn_summary(5, "算了，不申请Medicaid了，直接帮我找护工吧", s, debug)

    prereq_req = get_request_by_id(s, prereq_request_id) if prereq_request_id else None
    parent_req = get_request_by_id(s, original_request_id) if original_request_id else None

    checks_after_abandon = [
        ("prereq status = aborted",
         prereq_req.get("status") == "aborted" if prereq_req else False),
        ("active switches back to parent",
         s.request_manager.active_request_id == original_request_id),
        ("parent status = collecting (not paused)",
         parent_req.get("status") == "collecting" if parent_req else False),
        ("pending_queue drained",
         len(s.request_manager.pending_queue) == 0),
    ]
    p1 = run_assertions(checks_after_abandon, "Checkpoint: After Prereq Abandon (THE GAP)")

    # Turn 6: Continue parent
    s, debug = await run_turn(g, s, "预算3000左右，在芝加哥")
    print_turn_summary(6, "预算3000左右，在芝加哥", s, debug)

    checks_parent_continues = [
        ("active is original request",
         s.request_manager.active_request_id == original_request_id),
    ]
    p2 = run_assertions(checks_parent_continues, "Checkpoint: Parent Continues After Abandon")

    return p1 and p2


# ── main ─────────────────────────────────────────────────────────────

async def main():
    print("\n🧪 Prerequisite Completion & Abandon Diagnostic Test\n")

    scenarios_to_run = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    all_scenarios = {
        "7": ("Scenario 7: Prereq Completion & Parent Resume", scenario_7_prereq_completion),
        "8": ("Scenario 8: Prereq Abandon & Parent Resume", scenario_8_prereq_abandon),
    }

    if "all" in scenarios_to_run:
        selected = all_scenarios
    else:
        selected = {k: v for k, v in all_scenarios.items() if k in scenarios_to_run}

    if not selected:
        print("❌ No valid scenarios. Use: python test_prereq_completion.py [7|8|all]")
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
        print(f"\n  ⚠️  Some scenarios had failures (expected — these are the gaps to implement)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
