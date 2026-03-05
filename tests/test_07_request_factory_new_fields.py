"""
Test 07: Verify request_factory backward compatibility and new field emission.

No DDB required — tests only in-memory patch generation.

Usage:
    python tests/test_07_request_factory_new_fields.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from request_factory import build_request_patch
from state_models import RequestRecord


def run_tests():
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

    # ── Test 1: Old-style call (no new params) ────────────────────────
    print("\n1. Old-style build_request_patch (backward compatibility)")
    patch = build_request_patch(
        conversation_id="conv-1",
        user_id="user-1",
        name="Test Request",
        goal="Test goal",
    )
    rm = patch["request_manager"]
    rid = list(rm["requests"].keys())[0]
    req = rm["requests"][rid]

    check("request_id starts with req_", rid.startswith("req_"))
    check("name is set", req["name"] == "Test Request")
    check("request_type defaults to empty", req["request_type"] == "")
    check("subject_entity_id defaults to empty", req["subject_entity_id"] == "")
    check("variant defaults to empty dict", req["variant"] == {})
    check("title defaults to name", req["title"] == "Test Request")
    check("summary_current defaults to empty", req["summary_current"] == "")
    check("slot_refs defaults to empty dict", req["slot_refs"] == {})
    check("ddb_writes present", len(patch.get("ddb_writes", [])) == 1)

    # DDB item also has defaults
    ddb_item = patch["ddb_writes"][0]["item"]
    check("DDB item has request_type", ddb_item["request_type"] == "")
    check("DDB item has subject_entity_id", ddb_item["subject_entity_id"] == "")

    # ── Test 2: New-style call (with all new params) ──────────────────
    print("\n2. New-style build_request_patch (with request_type, subject_entity_id)")
    patch2 = build_request_patch(
        conversation_id="conv-2",
        user_id="user-2",
        name="Renew Insurance",
        goal="Renew mom's Medicaid",
        request_type="insurance_renewal",
        subject_entity_id="care_recipient:mom",
        variant={"plan_type": "medicaid", "state": "IL"},
        title="Renew Mom's Medicaid (IL)",
    )
    rm2 = patch2["request_manager"]
    rid2 = list(rm2["requests"].keys())[0]
    req2 = rm2["requests"][rid2]

    check("request_type set", req2["request_type"] == "insurance_renewal")
    check("subject_entity_id set", req2["subject_entity_id"] == "care_recipient:mom")
    check("variant set", req2["variant"] == {"plan_type": "medicaid", "state": "IL"})
    check("title set", req2["title"] == "Renew Mom's Medicaid (IL)")

    ddb2 = patch2["ddb_writes"][0]["item"]
    check("DDB request_type", ddb2["request_type"] == "insurance_renewal")
    check("DDB subject_entity_id", ddb2["subject_entity_id"] == "care_recipient:mom")
    check("DDB variant", ddb2["variant"] == {"plan_type": "medicaid", "state": "IL"})

    # ── Test 3: RequestRecord model accepts new fields ────────────────
    print("\n3. RequestRecord model backward compatibility")
    # Old-style (no new fields)
    r1 = RequestRecord(request_id="r1", name="Old Style")
    check("old-style record works", r1.request_type == "")
    check("old-style has empty slot_refs", r1.slot_refs == {})

    # New-style
    r2 = RequestRecord(
        request_id="r2", name="New Style",
        request_type="find_caregiver",
        subject_entity_id="care_recipient:dad",
        variant={"language": "chinese"},
        summary_current="Looking for caregiver",
        slot_refs={"budget": "fact_123"},
    )
    check("new-style request_type", r2.request_type == "find_caregiver")
    check("new-style subject_entity_id", r2.subject_entity_id == "care_recipient:dad")
    check("new-style slot_refs", r2.slot_refs == {"budget": "fact_123"})

    # Serialization roundtrip
    d = r2.model_dump()
    r3 = RequestRecord(**d)
    check("serialization roundtrip preserves request_type", r3.request_type == "find_caregiver")
    check("serialization roundtrip preserves slot_refs", r3.slot_refs == {"budget": "fact_123"})

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
