"""
Test 04: Memory Write Gate — risk classification + commit with real DDB.

Tests:
  - Low/medium/high risk fact classification
  - Auto-patch vs needs_confirm routing
  - commit_updates writes to FactStore + audit log
  - Confirmation question generation (EN + ZH)

Usage:
    sky-withcare-prod
    python tests/test_04_write_gate.py
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddb_client import get_ddb_resource, USER_FACT_TABLE, MEMORY_FACT_LOG_TABLE
from fact_store import FactStore
from memory_write_gate import propose_updates, commit_updates, build_confirmation_questions
from models.fact_models import ExtractedFact

TEST_USER = "test-user-gate-001"
TEST_ENTITY = "care_recipient:test-mom"


async def run_tests():
    ddb = get_ddb_resource()
    if ddb is None:
        print("[SKIP] DynamoDB not available")
        sys.exit(1)

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

    # ── Test 1: Risk classification ───────────────────────────────────
    print("\n1. Risk classification (propose_updates)")
    facts = [
        # Low risk, high confidence → auto
        ExtractedFact(
            entity_id=TEST_ENTITY, fact_key="preference.food.like",
            value=["congee"], confidence=0.8, source_type="user",
            evidence="User said mom likes congee",
        ),
        # High risk, user source, not confirmed → needs_confirm
        ExtractedFact(
            entity_id=TEST_ENTITY, fact_key="insurance.member_id",
            value="XYZ999", confidence=0.9, source_type="user",
            evidence="User mentioned card number",
        ),
        # High risk, tool source, high confidence → auto
        ExtractedFact(
            entity_id=TEST_ENTITY, fact_key="insurance.plan_type",
            value="medicaid", confidence=0.85, source_type="tool",
            evidence="Tool returned plan type",
        ),
        # High risk, user confirmed → auto
        ExtractedFact(
            entity_id=TEST_ENTITY, fact_key="contact.primary_phone",
            value="312-555-0100", confidence=0.95, source_type="user",
            evidence="User provided phone", explicit_user_confirmed=True,
        ),
        # Medium risk, low confidence → needs_confirm
        ExtractedFact(
            entity_id=TEST_ENTITY, fact_key="provider.primary_care.name",
            value="Dr. Chen", confidence=0.5, source_type="agent_inference",
            evidence="Inferred from conversation",
        ),
    ]

    proposal = propose_updates(facts)
    check("3 facts auto-patched", len(proposal.auto_patch) == 3,
          f"got {len(proposal.auto_patch)}")
    check("2 facts need confirm", len(proposal.needs_confirm) == 2,
          f"got {len(proposal.needs_confirm)}")

    auto_keys = {f.fact_key for f in proposal.auto_patch}
    check("preference.food.like is auto", "preference.food.like" in auto_keys)
    check("insurance.plan_type is auto (tool source)", "insurance.plan_type" in auto_keys)
    check("contact.primary_phone is auto (confirmed)", "contact.primary_phone" in auto_keys)

    confirm_keys = {f.fact_key for f in proposal.needs_confirm}
    check("insurance.member_id needs confirm", "insurance.member_id" in confirm_keys)
    check("provider.primary_care.name needs confirm", "provider.primary_care.name" in confirm_keys)

    # ── Test 2: Confirmation questions ────────────────────────────────
    print("\n2. Confirmation questions")
    questions_en = build_confirmation_questions(proposal.needs_confirm, language="en")
    check("2 questions generated (EN)", len(questions_en) == 2)
    if questions_en:
        check("question has fact_key", "fact_key" in questions_en[0])
        check("question has text", "?" in questions_en[0].get("question", ""))

    questions_zh = build_confirmation_questions(proposal.needs_confirm, language="zh")
    check("2 questions generated (ZH)", len(questions_zh) == 2)
    if questions_zh:
        check("ZH question has Chinese", "确认" in questions_zh[0].get("question", ""))

    # ── Test 3: Commit to DDB ─────────────────────────────────────────
    print("\n3. Commit auto_patch facts to DDB")
    committed = await commit_updates(
        user_id=TEST_USER,
        request_id="test-req-gate-001",
        facts_to_commit=proposal.auto_patch,
        justification="test_04 write gate test",
        actor="test_script",
        store=store,
    )
    check("3 facts committed", len(committed) == 3, f"got {len(committed)}")
    for rec in committed:
        check(f"  {rec.fact_key} is active", rec.status == "active")

    # Verify in DDB
    fact_table = ddb.Table(USER_FACT_TABLE)
    for rec in committed:
        resp = fact_table.get_item(Key={"pk": rec.dynamo_pk(), "sk": rec.dynamo_sk()})
        check(f"  {rec.fact_key} in DDB", "Item" in resp)

    # ── Test 4: Verify audit log was written ──────────────────────────
    print("\n4. Verify audit log entries")
    log_table = ddb.Table(MEMORY_FACT_LOG_TABLE)
    resp = log_table.scan(
        FilterExpression="begins_with(pk, :prefix)",
        ExpressionAttributeValues={":prefix": f"USER#{TEST_USER}"},
    )
    log_count = len(resp.get("Items", []))
    check("audit log entries written", log_count >= 3, f"got {log_count}")

    # ── Cleanup ───────────────────────────────────────────────────────
    print("\n5. Cleanup")
    cleanup_count = 0
    for tname in [USER_FACT_TABLE, MEMORY_FACT_LOG_TABLE]:
        t = ddb.Table(tname)
        for prefix in [f"USER#{TEST_USER}#ENT#{TEST_ENTITY}", f"USER#{TEST_USER}"]:
            resp = t.scan(
                FilterExpression="begins_with(pk, :prefix)",
                ExpressionAttributeValues={":prefix": prefix},
            )
            for item in resp.get("Items", []):
                t.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})
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
