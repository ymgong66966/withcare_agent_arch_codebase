"""
Test 03: EventStore — write events, verify TTL, read back.

Usage:
    sky-withcare-prod
    python tests/test_03_event_store.py
"""

import asyncio
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddb_client import get_ddb_resource, USER_EVENT_TABLE
from event_store import EventStore

TEST_USER = "test-user-event-001"


async def run_tests():
    ddb = get_ddb_resource()
    if ddb is None:
        print("[SKIP] DynamoDB not available")
        sys.exit(1)

    table = ddb.Table(USER_EVENT_TABLE)
    store = EventStore(dynamodb_client=ddb)

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

    # ── Test 1: Write a dialogue_summary event ────────────────────────
    print("\n1. Write dialogue_summary event")
    evt1 = await store.add_event(
        user_id=TEST_USER,
        event_type="dialogue_summary",
        content="User asked about renewing mom's Medicaid in Illinois.",
        request_id="test-req-001",
        care_recipient_id="care_recipient:mom",
        tags=["medicaid", "renewal"],
    )
    check("event created", evt1 is not None)
    check("event has id", bool(evt1.event_id))
    check("event_type is dialogue_summary", evt1.event_type == "dialogue_summary")
    check("ttl_epoch is set", evt1.ttl_epoch > 0)
    expected_ttl = int(time.time()) + 30*86400
    check("ttl is ~30 days from now", abs(evt1.ttl_epoch - expected_ttl) < 300)

    # Verify in DDB
    resp = table.get_item(Key={"pk": evt1.dynamo_pk(), "sk": evt1.dynamo_sk()})
    check("event in DDB", "Item" in resp)
    if "Item" in resp:
        check("DDB has ttl_epoch", "ttl_epoch" in resp["Item"])
        # DDB returns Decimal, compare as int
        ddb_ttl = int(resp["Item"]["ttl_epoch"])
        check("DDB ttl_epoch matches", abs(ddb_ttl - evt1.ttl_epoch) < 5)

    # ── Test 2: Write a memory_candidate event (short TTL) ────────────
    print("\n2. Write memory_candidate event (14-day TTL)")
    evt2 = await store.add_event(
        user_id=TEST_USER,
        event_type="memory_candidate",
        content='{"fact_key": "insurance.member_id", "value": "ABC123", "needs_confirm": true}',
        request_id="test-req-001",
        tags=["candidate", "high_risk"],
    )
    check("memory_candidate created", evt2 is not None)
    expected_ttl_14 = int(time.time()) + 14*86400
    check("ttl is ~14 days", abs(evt2.ttl_epoch - expected_ttl_14) < 300)

    # ── Test 3: Write a tool_result event ─────────────────────────────
    print("\n3. Write tool_result event")
    evt3 = await store.add_event(
        user_id=TEST_USER,
        event_type="tool_result",
        content="Web search returned: Medicaid renewal deadline is Dec 31, 2026",
        request_id="test-req-001",
        structured={"source": "firecrawl", "query": "medicaid renewal deadline IL"},
        tags=["search", "medicaid"],
    )
    check("tool_result created", evt3 is not None)

    # ── Test 4: Custom TTL override ───────────────────────────────────
    print("\n4. Write event with custom TTL (7 days)")
    evt4 = await store.add_event(
        user_id=TEST_USER,
        event_type="error",
        content="MCP tool call failed: timeout",
        ttl_days=7,
    )
    check("custom TTL event created", evt4 is not None)
    expected_ttl_7 = int(time.time()) + 7*86400
    check("ttl is ~7 days", abs(evt4.ttl_epoch - expected_ttl_7) < 300)

    # ── Cleanup ───────────────────────────────────────────────────────
    print("\n5. Cleanup")
    cleanup_count = 0
    resp = table.scan(
        FilterExpression="begins_with(pk, :prefix)",
        ExpressionAttributeValues={":prefix": f"USER#{TEST_USER}"},
    )
    for item in resp.get("Items", []):
        table.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})
        cleanup_count += 1
    print(f"  Cleaned up {cleanup_count} items")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(run_tests())
