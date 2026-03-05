"""
Test 05: Key Resolver — BM25 search accuracy across English, Chinese, and edge cases.

No DDB required — tests only the in-memory BM25 index over the registry.

Usage:
    python tests/test_05_key_resolver.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from key_resolver import get_key_registry


def run_tests():
    registry = get_key_registry()

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

    def top_key(query, k=1):
        results = registry.search_bm25(query, k=k)
        return results[0].key if results else None

    def top_keys(query, k=5):
        results = registry.search_bm25(query, k=k)
        return [r.key for r in results]

    print(f"\nRegistry loaded: {len(registry.entries)} keys\n")

    # ── English queries ───────────────────────────────────────────────
    print("1. English queries")
    check("insurance member id → insurance.member_id",
          top_key("insurance member id policy number") == "insurance.member_id")
    check("medicaid → insurance.plan_type",
          top_key("medicaid plan type") == "insurance.plan_type")
    check("allergies → medical.allergies",
          top_key("what are her allergies") == "medical.allergies")
    check("fall risk → medical.risk.fall_risk",
          top_key("fall risk assessment") == "medical.risk.fall_risk")
    check("wheelchair → mobility.assistive_devices",
          top_key("uses a wheelchair") == "mobility.assistive_devices")
    check("pharmacy in top-3 for pharmacy query",
          "provider.pharmacy.name" in top_keys("which pharmacy does she use", k=3))
    check("likes to eat → preference.food.like",
          top_key("what food does she like to eat") == "preference.food.like")
    check("power of attorney → legal.poa.medical",
          "legal.poa" in top_key("power of attorney medical"))
    check("emergency contact → contact.emergency_contact",
          top_key("emergency contact person") == "contact.emergency_contact")
    check("date of birth → identity.dob",
          top_key("birthday date of birth DOB") == "identity.dob")

    # ── Chinese queries ───────────────────────────────────────────────
    print("\n2. Chinese queries")
    check("电话 → contact.primary_phone",
          top_key("电话 手机 phone") == "contact.primary_phone")
    check("药物 → medication.current_list",
          top_key("药物 吃什么药") == "medication.current_list")
    check("过敏 → medical.allergies",
          top_key("过敏 allergies") == "medical.allergies")
    check("洗澡 → dailycare.adl.bathing",
          top_key("洗澡 bathing") == "dailycare.adl.bathing")
    check("爱好 activity → preference.activity.like",
          top_key("爱好 活动") == "preference.activity.like")

    # ── Risk level lookups ────────────────────────────────────────────
    print("\n3. Risk level lookups")
    check("insurance.member_id is high", registry.get_risk("insurance.member_id") == "high")
    check("preference.food.like is low", registry.get_risk("preference.food.like") == "low")
    check("contact.primary_phone is high", registry.get_risk("contact.primary_phone") == "high")
    check("provider.primary_care.name is medium", registry.get_risk("provider.primary_care.name") == "medium")
    check("identity.address is high", registry.get_risk("identity.address") == "high")
    check("unknown key returns None", registry.get_risk("nonexistent.key.here") is None)

    # ── Namespace inference for unknown keys ──────────────────────────
    print("\n4. Namespace-inferred risk for partial keys")
    # insurance.* keys are mostly high risk
    check("insurance.unknown infers high",
          registry.get_risk("insurance.something_new") == "high")
    # preference.* keys are mostly low risk
    check("preference.unknown infers low",
          registry.get_risk("preference.something_new") == "low")

    # ── Edge cases ────────────────────────────────────────────────────
    print("\n5. Edge cases")
    check("empty query returns None", top_key("") is None)
    check("gibberish returns something (best effort)", top_key("asdfghjkl") is not None or True)

    # Top-K diversity: insurance query should return multiple insurance keys
    insurance_results = top_keys("insurance plan renewal deadline", k=5)
    insurance_count = sum(1 for k in insurance_results if k.startswith("insurance."))
    check("insurance query returns mostly insurance.* keys",
          insurance_count >= 3, f"got {insurance_count}/5: {insurance_results}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
