"""
Tests for the Onboarding Fact Bridge — deterministic field mapping,
mental assessment parsing, missing field robustness, and multi-recipient handling.
"""

import sys
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-dummy-key-for-unit-tests"


# ── Sample data ──────────────────────────────────────────────────────

SAMPLE_RECIPIENT_MOM = {
    "firstName": "Yiming",
    "lastName": "Gong",
    "dateOfBirth": "1954-04-11",
    "gender": "Male",
    "address": "11650 National Boulevard, Los Angeles, California 90064",
    "veteranStatus": "Veteran",
    "pronouns": "he/him/his",
    "dependentStatus": "Not a child/dependent",
    "relationship": "dad",
    "legalName": "Yiming Gong",
    "isSelf": False,
}

SAMPLE_RECIPIENT_MINIMAL = {
    "firstName": "Jane",
    "relationship": "mom",
}

SAMPLE_RECIPIENT_SELF = {
    "firstName": "Self",
    "isSelf": True,
    "relationship": "self",
}


# ── Helper ───────────────────────────────────────────────────────────

def _make_mock_store():
    """Create a mock FactStore that records calls."""
    store = MagicMock()
    store.upsert_fact = AsyncMock(return_value=MagicMock(fact_id="test-fact-id"))
    return store


# ── Test 1: Deterministic field mapping ──────────────────────────────

class TestDeterministicMapping:
    @pytest.mark.asyncio
    async def test_full_recipient_fields(self):
        """All known fields should be mapped and written."""
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            result = await ingest_onboarding_data(
                user_id="test-user",
                care_recipients=[{
                    "relationship": "dad",
                    "data": SAMPLE_RECIPIENT_MOM,
                }],
            )

        assert result["total_facts_written"] > 0
        assert "care_recipient:dad" in result["entities"]

        # Check specific fact keys were written
        call_args_list = mock_store.upsert_fact.call_args_list
        written_keys = {call.kwargs.get("fact_key") or call[1].get("fact_key", "") for call in call_args_list}

        assert "identity.full_name" in written_keys
        assert "identity.dob" in written_keys
        assert "identity.gender" in written_keys
        assert "identity.veteran_status" in written_keys
        assert "identity.pronouns" in written_keys

    @pytest.mark.asyncio
    async def test_full_name_combines_first_last(self):
        """identity.full_name should combine firstName + lastName."""
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            await ingest_onboarding_data(
                user_id="test-user",
                care_recipients=[{
                    "relationship": "dad",
                    "data": SAMPLE_RECIPIENT_MOM,
                }],
            )

        # Find the full_name call
        for call in mock_store.upsert_fact.call_args_list:
            if call.kwargs.get("fact_key") == "identity.full_name":
                assert call.kwargs["new_value"] == "Yiming Gong"
                return
        pytest.fail("identity.full_name was not written")


# ── Test 2: Mental assessment ────────────────────────────────────────

class TestMentalAssessment:
    @pytest.mark.asyncio
    async def test_burnout_score_written(self):
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            result = await ingest_onboarding_data(
                user_id="test-user",
                assessment_score=7,
            )

        # Should write burnout_score and burnout_level
        written_keys = {call.kwargs.get("fact_key") for call in mock_store.upsert_fact.call_args_list}
        assert "caregiver.burnout_score" in written_keys
        assert "caregiver.burnout_level" in written_keys

    @pytest.mark.asyncio
    async def test_burnout_levels(self):
        from onboarding_fact_bridge import _burnout_level

        assert _burnout_level(0) == "low"
        assert _burnout_level(4) == "low"
        assert _burnout_level(5) == "moderate"
        assert _burnout_level(10) == "moderate"
        assert _burnout_level(11) == "high"
        assert _burnout_level(12) == "high"

    @pytest.mark.asyncio
    async def test_assessment_answers_stored(self):
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        answers = [
            {"question": "Trouble concentrating?", "answer": "sometimes"},
            {"question": "Sleeping less?", "answer": "yes"},
        ]
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            await ingest_onboarding_data(
                user_id="test-user",
                assessment_score=3,
                assessment_answers=answers,
            )

        written_keys = {call.kwargs.get("fact_key") for call in mock_store.upsert_fact.call_args_list}
        assert "caregiver.burnout_assessment_detail" in written_keys

    @pytest.mark.asyncio
    async def test_no_assessment_no_burnout_facts(self):
        """When assessment_score is None, no burnout facts should be written."""
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            result = await ingest_onboarding_data(
                user_id="test-user",
                assessment_score=None,
            )

        written_keys = {call.kwargs.get("fact_key") for call in mock_store.upsert_fact.call_args_list}
        assert "caregiver.burnout_score" not in written_keys
        assert "caregiver.burnout_level" not in written_keys


# ── Test 3: Missing/None fields don't crash ──────────────────────────

class TestMissingFields:
    @pytest.mark.asyncio
    async def test_minimal_recipient(self):
        """Recipient with only firstName and relationship should not crash."""
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            result = await ingest_onboarding_data(
                user_id="test-user",
                care_recipients=[{
                    "relationship": "mom",
                    "data": SAMPLE_RECIPIENT_MINIMAL,
                }],
            )

        assert result["total_facts_written"] > 0
        assert "care_recipient:mom" in result["entities"]

    @pytest.mark.asyncio
    async def test_empty_data(self):
        """Empty data dict should produce no facts."""
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            result = await ingest_onboarding_data(
                user_id="test-user",
                care_recipients=[{"relationship": "unknown", "data": {}}],
            )

        # Only onboarding.completed_at should be written
        assert result["entities"].get("care_recipient:unknown", 0) == 0

    @pytest.mark.asyncio
    async def test_none_values_skipped(self):
        """Fields with None values should be skipped silently."""
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        data = {"firstName": "Test", "dateOfBirth": None, "gender": None, "relationship": "sibling"}
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            result = await ingest_onboarding_data(
                user_id="test-user",
                care_recipients=[{"relationship": "sibling", "data": data}],
            )

        # Should not crash and should skip None fields
        written_keys = {call.kwargs.get("fact_key") for call in mock_store.upsert_fact.call_args_list}
        assert "identity.dob" not in written_keys
        assert "identity.gender" not in written_keys

    @pytest.mark.asyncio
    async def test_completely_empty_request(self):
        """No data at all should still succeed with just the timestamp."""
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            result = await ingest_onboarding_data(user_id="test-user")

        # Should write at least the completion timestamp
        assert result["total_facts_written"] >= 1


# ── Test 4: Multiple care recipients ─────────────────────────────────

class TestMultipleRecipients:
    @pytest.mark.asyncio
    async def test_two_recipients_separate_entities(self):
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            result = await ingest_onboarding_data(
                user_id="test-user",
                care_recipients=[
                    {"relationship": "mom", "data": {"firstName": "Alice", "lastName": "Smith", "gender": "Female"}},
                    {"relationship": "dad", "data": {"firstName": "Bob", "lastName": "Smith", "gender": "Male"}},
                ],
            )

        assert "care_recipient:mom" in result["entities"]
        assert "care_recipient:dad" in result["entities"]
        assert result["entities"]["care_recipient:mom"] > 0
        assert result["entities"]["care_recipient:dad"] > 0


# ── Test 5: Entity ID resolution ─────────────────────────────────────

class TestEntityIdResolution:
    def test_relationship_based(self):
        from onboarding_fact_bridge import _resolve_entity_id
        assert _resolve_entity_id("Mom") == "care_recipient:mom"
        assert _resolve_entity_id("Dad") == "care_recipient:dad"
        assert _resolve_entity_id("spouse") == "care_recipient:spouse"

    def test_firstname_fallback(self):
        from onboarding_fact_bridge import _resolve_entity_id
        assert _resolve_entity_id("", "Alice") == "care_recipient:alice"

    def test_unknown_fallback(self):
        from onboarding_fact_bridge import _resolve_entity_id
        assert _resolve_entity_id("", "") == "care_recipient:unknown"


# ── Test 6: Tasks stored ─────────────────────────────────────────────

class TestTasksStored:
    @pytest.mark.asyncio
    async def test_tasks_written(self):
        from onboarding_fact_bridge import ingest_onboarding_data

        mock_store = _make_mock_store()
        tasks = ["Explore Medicare Benefits", "Find In-Home Support"]
        with patch("onboarding_fact_bridge.get_fact_store", return_value=mock_store):
            result = await ingest_onboarding_data(
                user_id="test-user",
                tasks=tasks,
            )

        written_keys = {call.kwargs.get("fact_key") for call in mock_store.upsert_fact.call_args_list}
        assert "onboarding.recommended_tasks" in written_keys

        # Find the tasks call and verify value
        for call in mock_store.upsert_fact.call_args_list:
            if call.kwargs.get("fact_key") == "onboarding.recommended_tasks":
                assert call.kwargs["new_value"] == tasks
                return


# ── Test 7: Syntax check ─────────────────────────────────────────────

class TestSyntaxCheck:
    def test_onboarding_fact_bridge_parses(self):
        import ast
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "onboarding_fact_bridge.py")
        with open(path) as f:
            ast.parse(f.read())

    def test_chat_server_parses(self):
        import ast
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_server.py")
        with open(path) as f:
            ast.parse(f.read())
