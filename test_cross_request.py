#!/usr/bin/env python3
"""
End-to-end test: cross-request fact reuse via the graph directly (no server needed).

Runs two conversations:
  1. Caregiver search — stores facts about mom
  2. Medicare renewal — should see those facts in memory context
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv(".env")

from state_models import UnifiedState, Meta
from merge_utils import apply_node_output
from graph import build_graph, check_prereq_lifecycle


def init_state(conv_id: str) -> UnifiedState:
    return UnifiedState(
        meta=Meta(conversation_id=conv_id, user_id="test-cross-req-user")
    )


async def run_turn(graph, state: UnifiedState, message: str) -> UnifiedState:
    state = apply_node_output(state, {"messages": [{"role": "user", "content": message}]})
    state_dict = state.model_dump()

    async for event in graph.astream(state_dict):
        node_name, node_output = next(iter(event.items()))
        state = apply_node_output(state, node_output)
        state_dict = state.model_dump()

    prereq_patch = check_prereq_lifecycle(state)
    if prereq_patch:
        state = apply_node_output(state, prereq_patch)

    return state


def get_active_request(state):
    rm = state.request_manager
    if rm.active_request_id and rm.active_request_id in rm.requests:
        return rm.requests[rm.active_request_id]
    return None


def last_reply(state):
    for m in reversed(state.messages):
        if m.role == "assistant":
            return m.content
    return ""


async def main():
    print("Building graph...")
    graph = build_graph()

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} — {detail}")
            failed += 1

    # ═══════════════════════════════════════════════════════════
    # CONVERSATION 1: Caregiver search
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONVERSATION 1: Home Caregiver Search")
    print("=" * 60)

    state1 = init_state("conv-test-001")

    # Turn 1: State intent
    print("\n--- Turn 1: State intent ---")
    state1 = await run_turn(graph, state1, "I need help finding a home caregiver for my mom")
    req1 = get_active_request(state1)

    print(f"  Reply: {last_reply(state1)[:100]}...")
    print(f"  request_type: {repr(req1.request_type if req1 else 'NO REQUEST')}")
    print(f"  subject_entity_id: {repr(req1.subject_entity_id if req1 else 'NO REQUEST')}")
    print(f"  slot_refs: {req1.slot_refs if req1 else {}}")

    check("Request created", req1 is not None)
    check("request_type is not empty", req1 and req1.request_type != "",
          f"got {repr(req1.request_type if req1 else None)}")
    check("subject_entity_id is not empty", req1 and req1.subject_entity_id != "",
          f"got {repr(req1.subject_entity_id if req1 else None)}")
    check("subject_entity_id contains 'mom'",
          req1 and "mom" in req1.subject_entity_id,
          f"got {repr(req1.subject_entity_id if req1 else None)}")

    # Turn 2: Provide info
    print("\n--- Turn 2: Provide basic info ---")
    state1 = await run_turn(graph, state1,
        "She lives in Chicago in a small apartment by herself. "
        "She likes congee and dumplings. "
        "She needs help Monday through Friday from 8am to 4pm. "
        "She uses a walker to get around.")
    req1 = get_active_request(state1)

    print(f"  Reply: {last_reply(state1)[:100]}...")
    print(f"  slot_refs: {req1.slot_refs if req1 else {}}")
    ic = req1.info_collection_state if req1 else {}
    print(f"  summary: {ic.get('summary_of_collected_info', '')[:100]}...")

    check("slot_refs has entries after Turn 2",
          req1 and len(req1.slot_refs) > 0,
          f"got {len(req1.slot_refs) if req1 else 0}")
    check("preference.food.like in slot_refs",
          req1 and "preference.food.like" in req1.slot_refs)
    check("mobility.assistive_devices in slot_refs",
          req1 and "mobility.assistive_devices" in req1.slot_refs)

    # Turn 3: Medical + insurance info
    print("\n--- Turn 3: Medical + insurance info ---")
    state1 = await run_turn(graph, state1,
        "She has diabetes and high blood pressure. "
        "Her insurance is Medicare Part A and B. "
        "Her primary doctor is Dr. Chen at Northwestern. "
        "Budget is around $25 per hour.")
    req1 = get_active_request(state1)

    print(f"  Reply: {last_reply(state1)[:100]}...")
    print(f"  slot_refs: {req1.slot_refs if req1 else {}}")

    check("slot_refs grew after Turn 3",
          req1 and len(req1.slot_refs) >= 3,
          f"got {len(req1.slot_refs) if req1 else 0}")

    # Record what facts were stored
    entity_id_used = req1.subject_entity_id if req1 else ""
    print(f"\n  Facts stored under entity_id: {repr(entity_id_used)}")
    print(f"  Total slot_refs: {len(req1.slot_refs) if req1 else 0}")

    # ═══════════════════════════════════════════════════════════
    # CONVERSATION 2: Medicare renewal (NEW conversation)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONVERSATION 2: Medicare Insurance Renewal (new conversation)")
    print("=" * 60)

    state2 = init_state("conv-test-002")

    # Turn 1: New intent
    print("\n--- Turn 1: State intent ---")
    state2 = await run_turn(graph, state2, "I need to renew my moms Medicare insurance")
    req2 = get_active_request(state2)

    print(f"  Reply: {last_reply(state2)[:150]}...")
    print(f"  request_type: {repr(req2.request_type if req2 else 'NO REQUEST')}")
    print(f"  subject_entity_id: {repr(req2.subject_entity_id if req2 else 'NO REQUEST')}")

    check("Request 2 created", req2 is not None)
    check("Request 2 request_type not empty", req2 and req2.request_type != "",
          f"got {repr(req2.request_type if req2 else None)}")
    check("Request 2 subject_entity_id not empty", req2 and req2.subject_entity_id != "",
          f"got {repr(req2.subject_entity_id if req2 else None)}")

    # Turn 2: Ask if it remembers
    print("\n--- Turn 2: Ask about memory ---")
    state2 = await run_turn(graph, state2,
        "Do you already have her address and doctor info from before?")
    req2 = get_active_request(state2)

    reply2 = last_reply(state2)
    print(f"  Reply: {reply2[:200]}...")

    # Check if the reply references known facts
    reply_lower = reply2.lower()
    has_memory = any(kw in reply_lower for kw in ["chicago", "dr. chen", "walker", "congee", "4521", "diabetes"])
    check("Reply references previously stored facts", has_memory,
          f"Reply did not mention any known facts. Full reply: {reply2[:300]}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return failed


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(1 if exit_code > 0 else 0)
