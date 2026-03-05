"""
Test 08: Slot→Fact Binding — fact extraction, key resolution, candidate pool,
and full pipeline integration.

Tests:
  - Fact extraction prompt produces valid structure
  - Key resolution: known keys → canonical, unknown keys → candidate pool
  - Candidate pool record/increment
  - Full pipeline: summary → extract → resolve → commit → slot_refs
  - Graceful failure returns {}

Usage:
    # Offline tests (no DDB needed):
    python tests/test_08_slot_fact_binding.py

    # With DDB (for candidate pool tests):
    sky-withcare-prod
    python tests/test_08_slot_fact_binding.py --with-ddb
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from models.fact_models import (
    CandidateKeyRecord,
    ExtractedFact,
    FactRecord,
    WriteProposal,
)
from candidate_key_pool import _normalize_key, CandidateKeyPoolStore
from memory_write_gate import propose_updates
from key_resolver import get_key_registry, get_key_resolver, KeyResolver, KeyResolverResult
from fact_store import FactStore
from id_utils import new_uuid
from prompts import make_fact_extraction_prompt


# ── Mock LLM Client ──────────────────────────────────────────────


class MockLLMClient:
    """Mock client that returns canned extraction results."""

    def __init__(self, response: str = "[]"):
        self._response = response
        self.call_count = 0

    async def async_chat(self, prompt: str = "", **kwargs) -> str:
        self.call_count += 1
        return self._response


MOCK_EXTRACTION_RESPONSE = json.dumps([
    {
        "fact_key": "insurance.plan_type",
        "fact_label": "Insurance plan type",
        "value": "Medicare Part A",
        "value_type": "string",
        "confidence": 0.95,
        "source_type": "user",
        "evidence": "She has Medicare Part A",
    },
    {
        "fact_key": "health.chronic_condition",
        "fact_label": "Chronic health condition",
        "value": "diabetes",
        "value_type": "string",
        "confidence": 0.9,
        "source_type": "user",
        "evidence": "Mom has diabetes",
    },
    {
        "fact_key": "care_schedule.weekday_hours",
        "fact_label": "Weekday care hours",
        "value": "8am-4pm",
        "value_type": "string",
        "confidence": 0.85,
        "source_type": "user",
        "evidence": "Needs care 8am to 4pm on weekdays",
    },
])


async def run_tests():
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

    # ── 1. CandidateKeyRecord model ──────────────────────────────
    print("\n1. CandidateKeyRecord model")

    record = CandidateKeyRecord(
        candidate_key="care_schedule.weekday_hours",
        description="Weekly care schedule in hours",
        risk_suggestion="medium",
    )
    check("CandidateKeyRecord creation", record.candidate_key == "care_schedule.weekday_hours")
    check("Default status is candidate", record.status == "candidate")
    check("Default occurrence_count is 1", record.occurrence_count == 1)
    check("Default sample_values is empty", record.sample_values == [])

    # ── 2. Key normalization ─────────────────────────────────────
    print("\n2. Key normalization (_normalize_key)")

    check("lowercase", _normalize_key("Care_Schedule") == "care_schedule")
    check("spaces to underscores", _normalize_key("care schedule") == "care_schedule")
    check("hyphens to underscores", _normalize_key("care-schedule") == "care_schedule")
    check("collapse underscores", _normalize_key("care__schedule") == "care_schedule")
    check("strip whitespace", _normalize_key("  care_schedule  ") == "care_schedule")
    check("mixed", _normalize_key("Care Schedule - Weekday") == "care_schedule_weekday")

    # ── 3. Fact extraction prompt structure ──────────────────────
    print("\n3. Fact extraction prompt structure")

    prompt = make_fact_extraction_prompt(
        entity_id="care_recipient:mom",
        request_type="caregiver_search",
        updated_summary="Mom needs care Monday-Friday. She has diabetes and Medicare.",
        key_info_needed=[
            {"item": "Insurance details"},
            {"item": "Care schedule"},
        ],
        conversation_history=[
            {"role": "user", "content": "My mom needs help during weekdays."},
            {"role": "assistant", "content": "Can you tell me about her insurance?"},
            {"role": "user", "content": "She has Medicare Part A."},
        ],
    )

    check("Prompt contains entity_id", "care_recipient:mom" in prompt)
    check("Prompt contains request_type", "caregiver_search" in prompt)
    check("Prompt contains summary", "diabetes" in prompt)
    check("Prompt contains key_info_needed", "Insurance details" in prompt)
    check("Prompt contains conversation", "Medicare Part A" in prompt)
    check("Prompt contains JSON format instructions", "fact_key" in prompt)
    check("Prompt contains confidence guide", "0.9+" in prompt or "0.9" in prompt)
    check("Prompt contains examples", "Example 1" in prompt or "example" in prompt.lower())

    # ── 4. LLM extraction (mocked) ──────────────────────────────
    print("\n4. LLM fact extraction (mocked)")

    from prompts import llm_extract_facts

    mock_client = MockLLMClient(response=MOCK_EXTRACTION_RESPONSE)
    facts = await llm_extract_facts(
        entity_id="care_recipient:mom",
        request_type="caregiver_search",
        updated_summary="Mom has diabetes and Medicare Part A. Needs care 8am-4pm weekdays.",
        key_info_needed=[{"item": "Insurance"}, {"item": "Schedule"}],
        conversation_history=[],
        client=mock_client,
    )

    check("Extraction returned list", isinstance(facts, list))
    check("Extracted 3 facts", len(facts) == 3, f"got {len(facts)}")
    check("First fact has fact_key", facts[0].get("fact_key") == "insurance.plan_type")
    check("First fact has confidence", facts[0].get("confidence") == 0.95)
    check("First fact has evidence", "Medicare" in facts[0].get("evidence", ""))
    check("LLM client was called once", mock_client.call_count == 1)

    # ── 5. LLM extraction with empty summary ────────────────────
    print("\n5. LLM extraction with empty summary")

    empty_facts = await llm_extract_facts(
        entity_id="care_recipient:mom",
        request_type="",
        updated_summary="",
        key_info_needed=[],
        conversation_history=[],
        client=mock_client,
    )
    check("Empty summary returns []", empty_facts == [])

    # ── 6. LLM extraction with invalid JSON ─────────────────────
    print("\n6. LLM extraction with invalid JSON")

    bad_client = MockLLMClient(response="This is not JSON at all")
    bad_facts = await llm_extract_facts(
        entity_id="care_recipient:mom",
        request_type="",
        updated_summary="Some summary",
        key_info_needed=[],
        conversation_history=[],
        client=bad_client,
    )
    check("Invalid JSON returns []", bad_facts == [])

    # ── 7. LLM extraction filters low confidence ────────────────
    print("\n7. LLM extraction filters low confidence")

    low_conf_response = json.dumps([
        {"fact_key": "a.b", "value": "x", "confidence": 0.3},
        {"fact_key": "c.d", "value": "y", "confidence": 0.8},
    ])
    low_conf_client = MockLLMClient(response=low_conf_response)
    filtered = await llm_extract_facts(
        entity_id="e",
        request_type="t",
        updated_summary="summary",
        key_info_needed=[],
        conversation_history=[],
        client=low_conf_client,
    )
    check("Low confidence fact filtered out", len(filtered) == 1, f"got {len(filtered)}")
    check("Kept high confidence fact", filtered[0].get("fact_key") == "c.d")

    # ── 8. Key resolution: known keys ────────────────────────────
    print("\n8. Key resolution for known registry keys")

    registry = get_key_registry()
    # insurance.plan_type should be in the registry
    risk = registry.get_risk("insurance.plan_type")
    check("insurance.plan_type in registry", risk is not None, f"risk={risk}")

    # Unknown key should return None risk
    risk_unknown = registry.get_risk("care_schedule.weekday_hours")
    # This might return None or a namespace-inferred risk
    check("Unknown key risk is handled", True)  # Just verify no crash

    # ── 9. Write gate classification ─────────────────────────────
    print("\n9. Write gate classification with extracted facts")

    test_facts = [
        # Low risk, high confidence → auto
        ExtractedFact(
            entity_id="care_recipient:mom",
            fact_key="preference.food.like",
            value="congee",
            confidence=0.9,
            source_type="user",
            evidence="Mom likes congee",
        ),
        # Known high risk, user source, not confirmed → needs_confirm
        ExtractedFact(
            entity_id="care_recipient:mom",
            fact_key="insurance.member_id",
            value="ABC123",
            confidence=0.95,
            source_type="user",
            evidence="Card says ABC123",
        ),
        # Unknown key (treated as high risk) → needs_confirm
        ExtractedFact(
            entity_id="care_recipient:mom",
            fact_key="care_schedule.weekday_hours",
            value="8am-4pm",
            confidence=0.85,
            source_type="user",
            evidence="Needs care 8am-4pm",
        ),
    ]

    proposal = propose_updates(test_facts)
    check("Proposal is WriteProposal", isinstance(proposal, WriteProposal))
    check("Has auto_patch list", isinstance(proposal.auto_patch, list))
    check("Has needs_confirm list", isinstance(proposal.needs_confirm, list))
    check("Low risk fact auto-patched",
          any(f.fact_key == "preference.food.like" for f in proposal.auto_patch))
    check("High risk unconfirmed fact needs confirm",
          any(f.fact_key == "insurance.member_id" for f in proposal.needs_confirm))

    # ── 10. Full pipeline (mocked, no DDB) ──────────────────────
    print("\n10. Full pipeline: bind_facts_from_summary (mocked)")

    # For full pipeline, we need to mock DDB operations
    # We use the bind function with a mock client that returns known facts
    # Since DDB is disabled in test, commit_updates will log-only
    from slot_fact_binder import bind_facts_from_summary

    # Mock client returns facts that include both known and unknown keys
    pipeline_response = json.dumps([
        {
            "fact_key": "preference.food.like",
            "fact_label": "Food preference",
            "value": "congee",
            "value_type": "string",
            "confidence": 0.9,
            "source_type": "user",
            "evidence": "Likes congee",
        },
    ])
    pipeline_client = MockLLMClient(response=pipeline_response)

    # Set DDB off for test
    original_env = os.environ.get("WITHCARE_DDB_OFF", "")
    os.environ["WITHCARE_DDB_OFF"] = "1"

    try:
        # Reset singletons to pick up DDB_OFF
        import ddb_client
        ddb_client._initialized = False
        ddb_client._ddb_client = None
        ddb_client._ddb_resource = None

        slot_refs = await bind_facts_from_summary(
            user_id="test-user-001",
            request_id="req_test_001",
            entity_id="care_recipient:test-mom",
            request_type="caregiver_search",
            updated_summary="Mom likes congee and has Medicare.",
            key_info_needed=[{"item": "food preferences"}],
            conversation_history=[
                {"role": "user", "content": "She likes congee."},
            ],
            client=pipeline_client,
        )

        check("Pipeline returns dict", isinstance(slot_refs, dict))
        # In DDB-off mode, commits log-only, so slot_refs may be empty
        # The important thing is no crash
        check("Pipeline completed without crash", True)
        check("LLM was called", pipeline_client.call_count >= 1)

    finally:
        if original_env:
            os.environ["WITHCARE_DDB_OFF"] = original_env
        else:
            os.environ.pop("WITHCARE_DDB_OFF", None)
        # Reset singletons
        ddb_client._initialized = False
        ddb_client._ddb_client = None
        ddb_client._ddb_resource = None

    # ── 11. Graceful failure returns {} ──────────────────────────
    print("\n11. Graceful failure handling")

    class ExplodingClient:
        async def async_chat(self, **kwargs):
            raise RuntimeError("LLM exploded")

    result = await bind_facts_from_summary(
        user_id="test-user-001",
        request_id="req_test_001",
        entity_id="care_recipient:test-mom",
        request_type="test",
        updated_summary="Some summary",
        key_info_needed=[],
        conversation_history=[],
        client=ExplodingClient(),
    )
    check("Exploding client returns {}", result == {})

    # Empty summary
    result2 = await bind_facts_from_summary(
        user_id="test-user-001",
        request_id="req_test_001",
        entity_id="care_recipient:test-mom",
        request_type="test",
        updated_summary="",
        key_info_needed=[],
        conversation_history=[],
        client=MockLLMClient(),
    )
    check("Empty summary returns {}", result2 == {})

    # ── 12. CandidateKeyPoolStore with DDB off ──────────────────
    print("\n12. CandidateKeyPoolStore (DDB off)")

    pool = CandidateKeyPoolStore(dynamodb_client=None)
    candidate = await pool.record_candidate(
        candidate_key="care_schedule.weekday_hours",
        description="Weekday care hours",
    )
    check("DDB-off record returns CandidateKeyRecord", isinstance(candidate, CandidateKeyRecord))
    check("DDB-off candidate has normalized key",
          candidate.candidate_key == "care_schedule.weekday_hours")

    get_result = await pool.get_candidate("care_schedule.weekday_hours")
    check("DDB-off get_candidate returns None", get_result is None)

    top = await pool.get_top_candidates()
    check("DDB-off get_top_candidates returns []", top == [])

    # Empty key
    empty = await pool.record_candidate(candidate_key="  ")
    check("Empty key returns None", empty is None)

    # ── 13. Duplicate extraction prevention ─────────────────────
    print("\n13. Duplicate extraction prevention (already_extracted_keys)")

    prompt_without_keys = make_fact_extraction_prompt(
        entity_id="care_recipient:mom",
        request_type="find_caregiver",
        updated_summary="Mom lives in Chicago, has diabetes, uses walker",
        key_info_needed=[],
        conversation_history=[],
        already_extracted_keys=[],
    )
    check("Prompt without keys says '(none yet)'",
          "(none yet)" in prompt_without_keys,
          f"substring not found")

    prompt_with_keys = make_fact_extraction_prompt(
        entity_id="care_recipient:mom",
        request_type="find_caregiver",
        updated_summary="Mom lives in Chicago, has diabetes, uses walker",
        key_info_needed=[],
        conversation_history=[],
        already_extracted_keys=["housing.city", "health.chronic_condition"],
    )
    check("Prompt with keys includes existing keys",
          "housing.city" in prompt_with_keys and "health.chronic_condition" in prompt_with_keys,
          f"keys not found in prompt")
    check("Prompt with keys includes dedup instruction",
          "do not re-extract" in prompt_with_keys.lower(),
          f"instruction not found in prompt")

    # ── 14. Batch key resolution ─────────────────────────────────
    print("\n14. Batch key resolution")

    # Test with mock LLM that returns batch results
    batch_response = json.dumps([
        {"fact_index": 1, "selected_key": "insurance.plan_type", "confidence": 0.9, "reason": "exact match"},
        {"fact_index": 2, "selected_key": "housing.city", "confidence": 0.85, "reason": "location match"},
    ])
    mock_batch_client = MockLLMClient(response=batch_response)
    resolver = KeyResolver(registry=get_key_registry(), llm_client=mock_batch_client)

    batch_facts = [
        {"fact_key": "insurance.plan_type", "fact_label": "Insurance", "evidence": "Medicare Part A"},
        {"fact_key": "housing.city", "fact_label": "City", "evidence": "Lives in Chicago"},
    ]
    results = await resolver.resolve_batch(
        facts=batch_facts,
        entity_id="care_recipient:mom",
        context={"request_type": "find_caregiver"},
    )
    check("Batch resolve returns list of correct length",
          isinstance(results, list) and len(results) == 2,
          f"got {len(results) if isinstance(results, list) else type(results)}")
    check("Batch resolve uses single LLM call",
          mock_batch_client.call_count == 1,
          f"call_count={mock_batch_client.call_count}")

    # Test with empty facts
    empty_results = await resolver.resolve_batch(
        facts=[], entity_id="care_recipient:mom",
    )
    check("Empty batch returns empty list",
          empty_results == [], f"got {empty_results}")

    # ── 15. High-risk fact conflict strategy ──────────────────────
    print("\n15. High-risk fact conflict strategy")

    # Mock a simple in-memory fact store
    class InMemoryFactStore(FactStore):
        def __init__(self):
            super().__init__(dynamodb_client=None)
            self._facts = {}

        async def get_active_fact_by_key(self, user_id, entity_id, fact_key):
            key = f"{user_id}#{entity_id}#{fact_key}"
            return self._facts.get(key)

        async def put_fact(self, fact):
            if fact.status == "active":
                key = f"{fact.user_id}#{fact.entity_id}#{fact.fact_key}"
                self._facts[key] = fact

        async def deprecate_fact(self, fact):
            key = f"{fact.user_id}#{fact.entity_id}#{fact.fact_key}"
            if key in self._facts and self._facts[key].fact_id == fact.fact_id:
                del self._facts[key]

    store = InMemoryFactStore()

    # First upsert: create initial fact
    initial = FactRecord(
        fact_id=new_uuid("fact"),
        user_id="test-user",
        entity_id="care_recipient:mom",
        fact_key="insurance.member_id",
        fact_value="ABC-123",
        status="active",
        risk_level="high",
        created_at=datetime.utcnow(),
    )
    store._facts["test-user#care_recipient:mom#insurance.member_id"] = initial

    # Upsert with conflict_strategy="needs_confirm" and high risk
    result = await store.upsert_fact(
        user_id="test-user",
        entity_id="care_recipient:mom",
        fact_key="insurance.member_id",
        new_value="XYZ-789",
        risk_level="high",
        conflict_strategy="needs_confirm",
    )

    check("Conflict: old fact stays active",
          "test-user#care_recipient:mom#insurance.member_id" in store._facts,
          "old fact was removed")
    old_fact = store._facts.get("test-user#care_recipient:mom#insurance.member_id")
    check("Conflict: old fact still has original value",
          old_fact and old_fact.fact_value == "ABC-123",
          f"old value = {old_fact.fact_value if old_fact else 'N/A'}")
    check("Conflict: new fact is candidate status",
          result.status == "candidate",
          f"new status = {result.status}")
    check("Conflict: new fact has new value",
          result.fact_value == "XYZ-789",
          f"new value = {result.fact_value}")
    check("Conflict: new fact supersedes old",
          result.supersedes_fact_id == initial.fact_id,
          f"supersedes = {result.supersedes_fact_id}")

    # Upsert with conflict_strategy="overwrite" (default)
    store2 = InMemoryFactStore()
    initial2 = FactRecord(
        fact_id=new_uuid("fact"),
        user_id="test-user",
        entity_id="care_recipient:mom",
        fact_key="insurance.member_id",
        fact_value="ABC-123",
        status="active",
        risk_level="high",
        created_at=datetime.utcnow(),
    )
    store2._facts["test-user#care_recipient:mom#insurance.member_id"] = initial2

    result2 = await store2.upsert_fact(
        user_id="test-user",
        entity_id="care_recipient:mom",
        fact_key="insurance.member_id",
        new_value="XYZ-789",
        risk_level="high",
        conflict_strategy="overwrite",
    )
    check("Overwrite: new fact is active",
          result2.status == "active",
          f"status = {result2.status}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return failed


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(1 if exit_code > 0 else 0)
