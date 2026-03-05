"""
Test 06: Context Bundle assembly — with and without DDB data.

Tests:
  - Bundle assembly with empty stores (new user)
  - Bundle assembly with pre-populated facts
  - Safety notes generation for stale/unverified high-risk facts
  - bundle_to_prompt_block rendering

Usage:
    sky-withcare-prod
    python tests/test_06_context_bundle.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddb_client import get_ddb_resource, USER_FACT_TABLE
from fact_store import FactStore
from event_store import EventStore
from context_bundle import get_context_bundle, bundle_to_prompt_block, _compute_safety_notes
from models.fact_models import FactRecord

TEST_USER = "test-user-bundle-001"
TEST_ENTITY = "care_recipient:test-mom"


async def run_tests():
    ddb = get_ddb_resource()
    if ddb is None:
        print("[SKIP] DynamoDB not available")
        sys.exit(1)

    fact_store = FactStore(dynamodb_client=ddb)
    event_store = EventStore(dynamodb_client=ddb)

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

    # ── Test 1: Empty bundle (new user) ───────────────────────────────
    print("\n1. Empty bundle (new user, no data in stores)")
    bundle = await get_context_bundle(
        user_id=TEST_USER,
        request_dict=None,
        message="I need help with care",
        fact_store=fact_store,
        event_store=event_store,
    )
    check("bundle created", bundle is not None)
    check("user_id set", bundle.user_id == TEST_USER)
    check("no active request", bundle.active_request is None)
    check("empty profile_facts", len(bundle.profile_facts) == 0)
    check("empty safety_notes", len(bundle.safety_notes) == 0)

    prompt = bundle_to_prompt_block(bundle)
    check("prompt block is fallback text", "No memory context" in prompt)

    # ── Test 2: Bundle with active request ────────────────────────────
    print("\n2. Bundle with active request dict")
    bundle2 = await get_context_bundle(
        user_id=TEST_USER,
        request_dict={
            "request_id": "req-test-001",
            "request_type": "insurance_renewal",
            "name": "Renew Medicaid",
            "title": "Renew Mom's Medicaid",
            "status": "collecting",
            "subject_entity_id": TEST_ENTITY,
            "slot_refs": {},
            "summary_current": "Need to collect card number and deadline.",
        },
        message="I want to renew insurance",
        fact_store=fact_store,
        event_store=event_store,
    )
    check("active_request populated", bundle2.active_request is not None)
    check("request_type correct", bundle2.active_request.request_type == "insurance_renewal")
    check("title correct", bundle2.active_request.title == "Renew Mom's Medicaid")
    check("status correct", bundle2.active_request.status == "collecting")

    prompt2 = bundle_to_prompt_block(bundle2)
    check("prompt has Active Request section", "Active Request" in prompt2)
    check("prompt has insurance_renewal", "insurance_renewal" in prompt2)

    # ── Test 3: Safety notes for stale/unverified facts ───────────────
    print("\n3. Safety notes computation")
    now = datetime.utcnow()
    mock_facts = [
        FactRecord(
            fact_id="f1", user_id=TEST_USER, entity_id=TEST_ENTITY,
            fact_key="insurance.member_id", fact_value="ABC123",
            risk_level="high", confidence=0.9,
            verification_level="unverified",  # high risk + unverified → flagged
            created_at=now,
        ),
        FactRecord(
            fact_id="f2", user_id=TEST_USER, entity_id=TEST_ENTITY,
            fact_key="insurance.plan_type", fact_value="medicaid",
            risk_level="high", confidence=0.95,
            verification_level="explicit_user_confirmed",
            last_verified_at=now - timedelta(days=120),  # stale → flagged
            created_at=now - timedelta(days=120),
        ),
        FactRecord(
            fact_id="f3", user_id=TEST_USER, entity_id=TEST_ENTITY,
            fact_key="preference.food.like", fact_value=["congee"],
            risk_level="low", confidence=0.8,
            verification_level="unverified",  # low risk → NOT flagged
            created_at=now,
        ),
        FactRecord(
            fact_id="f4", user_id=TEST_USER, entity_id=TEST_ENTITY,
            fact_key="contact.primary_phone", fact_value="312-555-0100",
            risk_level="high", confidence=0.95,
            verification_level="explicit_user_confirmed",
            last_verified_at=now - timedelta(days=30),  # recent → NOT flagged
            created_at=now - timedelta(days=30),
        ),
    ]

    notes = _compute_safety_notes(mock_facts, stale_threshold_days=90)
    note_keys = {n.fact_key for n in notes}
    check("2 safety notes generated", len(notes) == 2, f"got {len(notes)}: {note_keys}")
    check("member_id flagged (unverified)", "insurance.member_id" in note_keys)
    check("plan_type flagged (stale)", "insurance.plan_type" in note_keys)
    check("food.like NOT flagged (low risk)", "preference.food.like" not in note_keys)
    check("phone NOT flagged (recent)", "contact.primary_phone" not in note_keys)

    for note in notes:
        check(f"  {note.fact_key} has rule text", len(note.rule) > 10)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(run_tests())
