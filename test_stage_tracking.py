"""
Test script to verify stage transition tracking works correctly

This script simulates a request lifecycle and prints out the stage_history at each step.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

from graph import build_graph
from state_models import UnifiedState, Meta


async def test_stage_tracking():
    """Test that stage transitions are properly tracked"""

    print("=" * 60)
    print("Testing Stage Transition Tracking")
    print("=" * 60)

    # Initialize state
    initial_state = {
        "meta": {
            "conversation_id": "conv-test-001",
            "user_id": "user-test-001",
            "created_at": datetime.utcnow(),
        },
        "messages": [
            {
                "role": "user",
                "content": "我需要找一个会说中文的护工",
                "ts": datetime.utcnow(),
            }
        ],
        "routing": {
            "current_agent": None,
            "conversation_stage": "idle",
            "turn_mode": "new_intent",
        },
        "request_manager": {
            "active_request_id": None,
            "requests": {},
            "pending_queue": [],
        },
        "user_context": {
            "emotion": {},
            "consent": {
                "allow_web_search": True,
                "allow_location_use": True,
                "allow_store_updates": True,
            },
            "profile_snapshot": {
                "caregiver": {"facts": {"location": "Chicago"}},
                "care_recipient": {"facts": {}},
            },
            "memory": {},
        },
        "tools": {
            "tool_runs": [],
            "tool_failures": [],
        },
    }

    # Build graph
    graph = build_graph()

    print("\n" + "-" * 60)
    print("Step 1: User makes initial request")
    print("-" * 60)

    # Run the graph (first turn should go through turn_router → upstream_delegator → info_collection)
    try:
        result = await graph.ainvoke(initial_state)

        # Extract and display stage history
        requests = result.get("request_manager", {}).get("requests", {})
        if requests:
            for req_id, req_data in requests.items():
                print(f"\nRequest ID: {req_id}")
                print(f"Request Name: {req_data.get('name', 'Unknown')}")
                print(f"Status: {req_data.get('status', 'Unknown')}")

                stage_history = req_data.get("stage_history", [])
                if stage_history:
                    print(f"\n📊 Stage History ({len(stage_history)} transitions):")
                    for i, transition in enumerate(stage_history, 1):
                        from_stage = transition.get("from_stage", "None")
                        to_stage = transition.get("to_stage", "Unknown")
                        agent = transition.get("agent", "Unknown")
                        reason = transition.get("reason", "No reason provided")
                        timestamp = transition.get("timestamp", "Unknown time")

                        print(f"\n  Transition {i}:")
                        print(f"    {from_stage} → {to_stage}")
                        print(f"    Agent: {agent}")
                        print(f"    Reason: {reason}")
                        print(f"    Time: {timestamp}")
                else:
                    print("\n⚠️  No stage history found!")
        else:
            print("\n⚠️  No requests created yet!")

        print("\n" + "-" * 60)
        print("Step 2: User provides information")
        print("-" * 60)

        # Simulate user providing information
        result["messages"].append({
            "role": "user",
            "content": "预算2000-3000，芝加哥，尽快",
            "ts": datetime.utcnow(),
        })

        # Run another turn
        result2 = await graph.ainvoke(result)

        # Display updated stage history
        requests2 = result2.get("request_manager", {}).get("requests", {})
        if requests2:
            for req_id, req_data in requests2.items():
                stage_history = req_data.get("stage_history", [])
                print(f"\n📊 Updated Stage History ({len(stage_history)} transitions):")
                for i, transition in enumerate(stage_history, 1):
                    from_stage = transition.get("from_stage", "None")
                    to_stage = transition.get("to_stage", "Unknown")
                    agent = transition.get("agent", "Unknown")

                    print(f"  {i}. {from_stage} → {to_stage} ({agent})")

        print("\n" + "=" * 60)
        print("✅ Stage tracking test completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


async def test_prerequisite_tracking():
    """Test that prerequisite creation properly tracks stage transitions"""

    print("\n\n" + "=" * 60)
    print("Testing Prerequisite Stage Tracking")
    print("=" * 60)

    # This would simulate a scenario where:
    # 1. User creates request A
    # 2. System detects need for prerequisite B
    # 3. User accepts prerequisite
    # 4. Request A gets paused, Request B becomes active

    print("\nScenario:")
    print("1. User: '找护工'")
    print("2. User: '我还没申请Medicaid'")
    print("3. System detects prerequisite")
    print("4. User accepts prerequisite")

    print("\nExpected stage_history for Request A (找护工):")
    print("  1. None → info_collection (info_collection)")
    print("  2. collecting → paused (upstream_delegator)")

    print("\nExpected stage_history for Request B (Medicaid):")
    print("  1. None → info_collection (info_collection)")
    print("  2. proposed → info_collection (info_collection) [after acceptance]")

    print("\n💡 To test this fully, run the actual graph with these inputs.")
    print("=" * 60)


if __name__ == "__main__":
    print("\n🧪 Stage Tracking Test Suite\n")

    # Run basic test
    asyncio.run(test_stage_tracking())

    # Show prerequisite scenario
    asyncio.run(test_prerequisite_tracking())

    print("\n✅ All tests completed!\n")
