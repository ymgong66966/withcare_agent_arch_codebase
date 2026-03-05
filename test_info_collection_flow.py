"""
测试 Info Collection Recursive Q&A Flow

验证 info_collection_node 的完整递归一问一答 workflow:
1. 用户发起请求 → LLM 生成 collection plan → 向用户提问
2. 用户回答 → LLM 总结已收集信息 + 评估 readiness → 继续追问或 handoff
3. 用户说"先用现有的继续" → 系统 finalize 并 handoff
4. 用户提供完整信息 → readiness = "ready" → 自动 handoff
"""

import asyncio
import json
import sys
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

from state_models import UnifiedState, Meta
from merge_utils import apply_node_output
from graph import build_graph


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

    return state, debug


def last_assistant_message(state: UnifiedState) -> str:
    for m in reversed(state.messages):
        if m.role == "assistant":
            return m.content
    return ""


def get_active_request(state: UnifiedState) -> Optional[Dict[str, Any]]:
    rm = state.request_manager
    rid = rm.active_request_id
    if not rid:
        return None
    reqs = rm.requests or {}
    req = reqs.get(rid)
    if req is None:
        return None
    return req.model_dump() if hasattr(req, "model_dump") else dict(req)


def get_info_state(req: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not req:
        return {}
    return req.get("info_collection_state") or {}


def get_routing(state: UnifiedState) -> Dict[str, Any]:
    r = state.routing
    return r.model_dump() if hasattr(r, "model_dump") else dict(r)


def print_turn_summary(turn_num: int, user_msg: str, state: UnifiedState, debug: Dict[str, Any]):
    """Print a concise summary after each turn."""
    req = get_active_request(state)
    info = get_info_state(req)
    routing = get_routing(state)
    handoff = routing.get("pending_handoff") or {}

    print(f"\n{'─' * 70}")
    print(f"  Turn {turn_num}")
    print(f"{'─' * 70}")
    print(f"  USER: {user_msg}")
    print(f"  ASSISTANT: {last_assistant_message(state)[:300]}")
    print(f"  GRAPH NODES: {' → '.join(debug['node_names'])}")
    print()

    if req:
        print(f"  📋 Request Status:        {req.get('status')}")
        print(f"  ⏳ Awaiting User Input:    {req.get('awaiting_user_input')}")
        print(f"  🔄 Readiness:             {info.get('readiness_to_proceed', 'N/A')}")
        print(f"  💬 Conversation Turns:    {info.get('conversation_turns_with_agent', 'N/A')}")
        print(f"  🤖 Current Agent:         {routing.get('current_agent')}")

        handoff_agent = handoff.get("recommended_next_agent")
        if handoff_agent:
            print(f"  🚀 Pending Handoff:       → {handoff_agent} ({handoff.get('reason', '')})")

        summary = info.get("summary_of_collected_info", "")
        if summary:
            print(f"  📝 Collected Info Summary:")
            for line in summary.strip().split("\n")[:6]:
                print(f"       {line}")

        prereqs = info.get("detected_prerequisites") or []
        if prereqs:
            print(f"  ⚠️  Detected Prerequisites:")
            for p in prereqs:
                print(f"       - {p.get('type', '?')}: {p.get('reason', '')}")

        key_status = info.get("key_info_status") or []
        if key_status:
            print(f"  📊 Key Info Status:")
            for ks in key_status[:6]:
                icon = {"collected": "✅", "partial": "🟡", "missing": "❌"}.get(ks.get("status", ""), "❓")
                print(f"       {icon} {ks.get('question', '?')[:60]} → {ks.get('status')}")
    else:
        print("  (no active request)")

    print()


def print_final_summary(state: UnifiedState):
    """Print final state summary after all turns."""
    req = get_active_request(state)
    info = get_info_state(req)
    routing = get_routing(state)
    handoff = routing.get("pending_handoff") or {}

    print(f"\n{'═' * 70}")
    print(f"  FINAL STATE SUMMARY")
    print(f"{'═' * 70}")

    if req:
        print(f"  Request ID:            {state.request_manager.active_request_id}")
        print(f"  Request Name:          {req.get('name', 'N/A')}")
        print(f"  Request Goal:          {req.get('goal', 'N/A')[:100]}")
        print(f"  Status:                {req.get('status')}")
        print(f"  Awaiting User Input:   {req.get('awaiting_user_input')}")
        print(f"  Routing Hint:          {req.get('routing_hint', 'N/A')}")
        print()

        print(f"  Info Collection State:")
        print(f"    Readiness:           {info.get('readiness_to_proceed', 'N/A')}")
        print(f"    Turns with Agent:    {info.get('conversation_turns_with_agent', 'N/A')}")
        print(f"    Key Info Needed:     {len(info.get('key_info_needed', []))} items")
        print(f"    Nice to Have:        {len(info.get('nice_to_have_info', []))} items")
        print()

        summary = info.get("summary_of_collected_info", "")
        if summary:
            print(f"  Full Collected Info Summary:")
            for line in summary.strip().split("\n"):
                print(f"    {line}")
            print()

        history = req.get("stage_history") or []
        if history:
            print(f"  Stage History ({len(history)} transitions):")
            for h in history:
                print(f"    {h.get('from_stage', '?')} → {h.get('to_stage', '?')} [{h.get('agent', '?')}] : {h.get('reason', '')[:60]}")
            print()

    handoff_agent = handoff.get("recommended_next_agent")
    if handoff_agent:
        print(f"  ✅ HANDOFF: → {handoff_agent}")
    else:
        print(f"  ⏳ No handoff yet (still collecting)")

    print(f"{'═' * 70}\n")


# ── test scenarios ───────────────────────────────────────────────────

async def scenario_1_full_collection():
    """Scenario 1: 完整 3-turn 信息收集 → 自动 handoff"""
    print("\n" + "=" * 70)
    print("  SCENARIO 1: Full 3-Turn Info Collection → Auto Handoff")
    print("=" * 70)

    g = build_graph()
    s = init_state("conv-test-s1", "user-test-s1")

    turns = [
        "我需要找一个护工，会说中文的，在芝加哥附近",
        "预算大概2000-3000一个月，尽快开始，需要会做中餐",
        "没有其他要求了，可以开始找了",
    ]

    for i, t in enumerate(turns, 1):
        s, debug = await run_turn(g, s, t)
        print_turn_summary(i, t, s, debug)

    print_final_summary(s)

    # Assertions
    req = get_active_request(s)
    info = get_info_state(req)
    routing = get_routing(s)
    handoff = routing.get("pending_handoff") or {}

    passed = True
    checks = [
        ("active_request_id exists", s.request_manager.active_request_id is not None),
        ("info_collection_state exists", bool(info)),
        ("conversation_turns >= 2", (info.get("conversation_turns_with_agent") or 0) >= 2),
        ("summary non-empty", bool(info.get("summary_of_collected_info", ""))),
    ]

    print("  Assertions:")
    for name, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"    {icon} {name}")
        if not ok:
            passed = False

    return passed


async def scenario_2_early_handoff():
    """Scenario 2: 用户中途说"先用现有的继续" → 提前 handoff"""
    print("\n" + "=" * 70)
    print("  SCENARIO 2: Early Handoff via '先用现有的继续'")
    print("=" * 70)

    g = build_graph()
    s = init_state("conv-test-s2", "user-test-s2")

    turns = [
        "我需要找一个护工，会说中文的，在芝加哥附近",
        "先用现有的继续",
    ]

    for i, t in enumerate(turns, 1):
        s, debug = await run_turn(g, s, t)
        print_turn_summary(i, t, s, debug)

    print_final_summary(s)

    req = get_active_request(s)
    routing = get_routing(s)
    handoff = routing.get("pending_handoff") or {}

    passed = True
    checks = [
        ("active_request_id exists", s.request_manager.active_request_id is not None),
        ("handoff agent exists", handoff.get("recommended_next_agent") is not None),
    ]

    print("  Assertions:")
    for name, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"    {icon} {name}")
        if not ok:
            passed = False

    return passed


async def scenario_3_multi_turn_gradual():
    """Scenario 3: 多轮追问 → 信息逐步补全"""
    print("\n" + "=" * 70)
    print("  SCENARIO 3: Multi-Turn Gradual Info Collection")
    print("=" * 70)

    g = build_graph()
    s = init_state("conv-test-s3", "user-test-s3")

    turns = [
        "帮我申请Medicaid",
        "在Illinois",
        "我妈妈65岁，没有工作收入",
        "好的，信息够了就开始吧",
    ]

    summaries = []
    for i, t in enumerate(turns, 1):
        s, debug = await run_turn(g, s, t)
        print_turn_summary(i, t, s, debug)
        req = get_active_request(s)
        info = get_info_state(req)
        summaries.append(info.get("summary_of_collected_info", ""))

    print_final_summary(s)

    passed = True
    checks = [
        ("active_request_id exists", s.request_manager.active_request_id is not None),
        ("summary grows over turns", len(summaries[-1]) >= len(summaries[0])),
        ("conversation_turns >= 3", (get_info_state(get_active_request(s)).get("conversation_turns_with_agent") or 0) >= 3),
    ]

    print("  Assertions:")
    for name, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"    {icon} {name}")
        if not ok:
            passed = False

    return passed


async def scenario_4_prerequisite_detection():
    """Scenario 4: Prerequisite 检测"""
    print("\n" + "=" * 70)
    print("  SCENARIO 4: Prerequisite Detection")
    print("=" * 70)

    g = build_graph()
    s = init_state("conv-test-s4", "user-test-s4")

    turns = [
        "我需要找一个护工",
        "预算的话我不太确定，因为我还没申请Medicaid，不知道能cover多少",
    ]

    for i, t in enumerate(turns, 1):
        s, debug = await run_turn(g, s, t)
        print_turn_summary(i, t, s, debug)

    print_final_summary(s)

    req = get_active_request(s)
    info = get_info_state(req)
    prereqs = info.get("detected_prerequisites") or []

    passed = True
    checks = [
        ("active_request_id exists", s.request_manager.active_request_id is not None),
        ("detected_prerequisites non-empty", len(prereqs) > 0),
    ]

    print("  Assertions:")
    for name, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"    {icon} {name}")
        if not ok:
            passed = False

    if not prereqs:
        print("    ⚠️  Note: Prerequisite detection depends on LLM quality. May not always detect.")

    return passed


async def scenario_5_prerequisite_accept():
    """Scenario 5: User accepts prerequisite → new request created → new Q&A round"""
    print("\n" + "=" * 70)
    print("  SCENARIO 5: Prerequisite Acceptance → New Request & Q&A")
    print("=" * 70)

    g = build_graph()
    s = init_state("conv-test-s5", "user-test-s5")

    turns = [
        "我需要找一个护工",
        "预算的话我不太确定，因为我还没申请Medicaid，不知道能cover多少",
        "要，先帮我处理Medicaid申请",  # User accepts prerequisite
        "在Illinois，我妈妈65岁",  # Answer questions for the new Medicaid request
    ]

    original_request_id = None
    prereq_request_id = None

    for i, t in enumerate(turns, 1):
        s, debug = await run_turn(g, s, t)
        print_turn_summary(i, t, s, debug)

        # Track request IDs
        if i == 1:
            original_request_id = s.request_manager.active_request_id
        elif i == 3:
            # After accepting prerequisite, active_request_id should change
            prereq_request_id = s.request_manager.active_request_id

    print_final_summary(s)

    req = get_active_request(s)
    rm = s.request_manager
    all_requests = rm.requests or {}

    # Convert Pydantic models to dicts for easier access
    all_requests_dict = {}
    for rid, record in all_requests.items():
        if hasattr(record, "model_dump"):
            all_requests_dict[rid] = record.model_dump()
        else:
            all_requests_dict[rid] = dict(record)

    passed = True
    checks = [
        ("original request exists", original_request_id is not None),
        ("prerequisite request created", prereq_request_id is not None and prereq_request_id != original_request_id),
        ("active request is prerequisite", s.request_manager.active_request_id == prereq_request_id if prereq_request_id else False),
        ("original request paused", all_requests_dict.get(original_request_id, {}).get("status") == "paused" if original_request_id else False),
        ("prerequisite request collecting", req.get("status") in ["created", "collecting"] if req else False),
    ]

    print("  Assertions:")
    for name, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"    {icon} {name}")
        if not ok:
            passed = False

    # Check if new Q&A round started for prerequisite
    info = get_info_state(req)
    if info.get("key_info_needed"):
        print(f"    ✅ New Q&A round started (key_info_needed: {len(info.get('key_info_needed', []))} items)")
    else:
        print(f"    ⚠️  New Q&A round may not have started properly")

    return passed


async def scenario_6_prerequisite_reject():
    """Scenario 6: User rejects prerequisite → continue with original request"""
    print("\n" + "=" * 70)
    print("  SCENARIO 6: Prerequisite Rejection → Continue Original Request")
    print("=" * 70)

    g = build_graph()
    s = init_state("conv-test-s6", "user-test-s6")

    turns = [
        "我需要找一个护工",
        "预算的话我不太确定，因为我还没申请Medicaid，不知道能cover多少",
        "不要，先不处理Medicaid，直接帮我找护工",  # User rejects prerequisite
        "芝加哥，预算3000左右",  # Continue answering original request questions
    ]

    original_request_id = None

    for i, t in enumerate(turns, 1):
        s, debug = await run_turn(g, s, t)
        print_turn_summary(i, t, s, debug)

        if i == 1:
            original_request_id = s.request_manager.active_request_id

    print_final_summary(s)

    req = get_active_request(s)
    rm = s.request_manager
    all_requests = rm.requests or {}

    # Convert Pydantic models to dicts for easier access
    all_requests_dict = {}
    for rid, record in all_requests.items():
        if hasattr(record, "model_dump"):
            all_requests_dict[rid] = record.model_dump()
        else:
            all_requests_dict[rid] = dict(record)

    passed = True
    checks = [
        ("original request still active", s.request_manager.active_request_id == original_request_id),
        ("only one request exists", len(all_requests) == 1),
        ("original request not paused", req.get("status") != "paused" if req else False),
        ("still collecting info", req.get("status") in ["created", "collecting"] if req else False),
    ]

    print("  Assertions:")
    for name, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"    {icon} {name}")
        if not ok:
            passed = False

    return passed


# ── main ─────────────────────────────────────────────────────────────

async def main():
    print("\n🧪 Info Collection Recursive Q&A Flow Test\n")

    # Parse command-line arguments for selective scenario running
    scenarios_to_run = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    
    all_scenarios = {
        "1": ("Scenario 1: Full Collection", scenario_1_full_collection),
        "2": ("Scenario 2: Early Handoff", scenario_2_early_handoff),
        "3": ("Scenario 3: Multi-Turn", scenario_3_multi_turn_gradual),
        "4": ("Scenario 4: Prerequisite Detection", scenario_4_prerequisite_detection),
        "5": ("Scenario 5: Prerequisite Accept", scenario_5_prerequisite_accept),
        "6": ("Scenario 6: Prerequisite Reject", scenario_6_prerequisite_reject),
    }

    # Determine which scenarios to run
    if "all" in scenarios_to_run:
        selected = all_scenarios
    else:
        selected = {k: v for k, v in all_scenarios.items() if k in scenarios_to_run}

    if not selected:
        print("❌ No valid scenarios specified. Use: python test_info_collection_flow.py [1|2|3|4|5|6|all]")
        return

    print(f"⏱️  Running {len(selected)} scenario(s) with delays to avoid rate limits...\n")

    results = {}
    for i, (num, (name, func)) in enumerate(selected.items()):
        results[name] = await func()
        # Add delay between scenarios (except after the last one)
        if i < len(selected) - 1:
            await asyncio.sleep(3)

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
        print(f"\n  ⚠️  Some scenarios had failures. Check output above.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
