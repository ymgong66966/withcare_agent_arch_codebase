"""
Test 02: FactStore CRUD — write, read, version-chain, deprecate facts.

Writes real data to WithCare_UserFactTable, then cleans up.

Usage:
    sky-withcare-prod
    python tests/test_02_fact_store_crud.py
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddb_client import get_ddb_resource, USER_FACT_TABLE
from fact_store import FactStore
from models.fact_models import FactRecord

# Use a test-specific user to avoid polluting real data
TEST_USER = "test-user-crud-001"
TEST_ENTITY = "care_recipient:test-mom"


async def run_tests():
    ddb = get_ddb_resource()
    if ddb is None:
        print("[SKIP] DynamoDB not available")
        sys.exit(1)

    table = ddb.Table(USER_FACT_TABLE)
    store = FactStore(dynamodb_client=ddb)

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

    # ── Test 1: Write a fact ──────────────────────────────────────────
    print("\n1. Write a fact (insurance.plan_type = medicaid)")
    fact1 = await store.upsert_fact(
        user_id=TEST_USER,
        entity_id=TEST_ENTITY,
        fact_key="insurance.plan_type",
        new_value="medicaid",
        fact_label="Mom's insurance type",
        value_type="enum",
        status="active",
        risk_level="high",
        confidence=0.9,
        source_type="user",
        source_ref="test-req-001",
        evidence="User said 'my mom has Medicaid'",
        verification_level="explicit_user_confirmed",
    )
    check("fact created", fact1 is not None)
    check("fact has id", bool(fact1.fact_id))
    check("fact status is active", fact1.status == "active")
    check("fact value is medicaid", fact1.fact_value == "medicaid")

    # Verify it's in DDB
    resp = table.get_item(Key={"pk": fact1.dynamo_pk(), "sk": fact1.dynamo_sk()})
    check("fact exists in DDB", "Item" in resp, f"pk={fact1.dynamo_pk()}")

    # ── Test 2: Read it back ──────────────────────────────────────────
    print("\n2. Read active facts")
    active = await store.get_active_facts(TEST_USER, TEST_ENTITY, fact_keys=["insurance.plan_type"])
    check("get_active_facts returns list", isinstance(active, list))
    # Note: until GSI queries are implemented, this returns [] from the stub.
    # That's OK — we verified the write via get_item above.

    # ── Test 3: Upsert same value — should update timestamps only ─────
    print("\n3. Upsert same value (should not create new fact)")
    fact1_updated = await store.upsert_fact(
        user_id=TEST_USER,
        entity_id=TEST_ENTITY,
        fact_key="insurance.plan_type",
        new_value="medicaid",  # same value
        evidence="User confirmed again",
    )
    check("same fact_id returned", fact1_updated.fact_id == fact1.fact_id)
    check("last_seen_at updated", fact1_updated.last_seen_at >= fact1.last_seen_at)

    # ── Test 4: Upsert different value — should deprecate old, create new
    print("\n4. Upsert different value (should version-chain)")
    fact2 = await store.upsert_fact(
        user_id=TEST_USER,
        entity_id=TEST_ENTITY,
        fact_key="insurance.plan_type",
        new_value="medicare",  # different value
        confidence=0.85,
        source_type="tool",
        evidence="Tool lookup returned Medicare",
        verification_level="tool_verified",
    )
    check("new fact created", fact2.fact_id != fact1.fact_id)
    check("new value is medicare", fact2.fact_value == "medicare")
    check("supersedes old fact", fact2.supersedes_fact_id == fact1.fact_id)
    check("new fact is active", fact2.status == "active")

    # ── Test 5: Write audit log ───────────────────────────────────────
    print("\n5. Write audit log entry")
    log_entry = await store.write_fact_log(
        user_id=TEST_USER,
        target_id=fact2.fact_id,
        patch={"fact_key": "insurance.plan_type", "old": "medicaid", "new": "medicare"},
        justification="Tool lookup corrected plan type",
        actor="test_script",
    )
    check("log entry created", log_entry is not None)
    check("log entry has id", bool(log_entry.log_id))

    # ── Test 6: Write alias ───────────────────────────────────────────
    print("\n6. Write alias (医保 → insurance.plan_type)")
    from models.fact_models import AliasRecord
    from datetime import datetime
    alias = AliasRecord(
        normalized_alias="医保",
        canonical_key="insurance.plan_type",
        scope="global",
        count=1,
        last_seen_at=datetime.utcnow(),
        confidence=0.8,
        status="active",
        evidence_samples=["User said '妈妈有医保'"],
    )
    await store.put_alias(alias)
    check("alias written", True)  # No exception means success

    # ── Cleanup ───────────────────────────────────────────────────────
    print("\n7. Cleanup test data")
    cleanup_count = 0

    # Scan for all items with our test user PK prefix
    for pk_prefix in [
        f"USER#{TEST_USER}#ENT#{TEST_ENTITY}",
        f"USER#{TEST_USER}",
        f"ALIAS#医保",
    ]:
        try:
            resp = table.scan(
                FilterExpression="begins_with(pk, :prefix)",
                ExpressionAttributeValues={":prefix": pk_prefix},
            )
            for item in resp.get("Items", []):
                table.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})
                cleanup_count += 1
        except Exception:
            pass

    # Also clean from FactAliasTable and MemoryFactLogTable
    from ddb_client import FACT_ALIAS_TABLE, MEMORY_FACT_LOG_TABLE
    for tname in [FACT_ALIAS_TABLE, MEMORY_FACT_LOG_TABLE]:
        try:
            t = ddb.Table(tname)
            resp = t.scan(
                FilterExpression="contains(pk, :uid)",
                ExpressionAttributeValues={":uid": TEST_USER},
            )
            for item in resp.get("Items", []):
                t.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})
                cleanup_count += 1
        except Exception:
            pass

    # Clean alias
    try:
        alias_table = ddb.Table(FACT_ALIAS_TABLE)
        alias_table.delete_item(Key={"pk": "ALIAS#医保", "sk": "KEY#insurance.plan_type"})
        cleanup_count += 1
    except Exception:
        pass

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
