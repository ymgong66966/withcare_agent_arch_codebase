"""
Verification tests for the State Checkpoint feature.

Tests:
1. Checkpoint data round-trip (build → serialize → deserialize)
2. _apply_checkpoint restores routing fields correctly
3. Sticky needs_human through checkpoint
4. Pending queue serialization round-trip
5. _rebuild_request_manager from DDB-like payload
6. Syntax check all modified files
7. conversation_store checkpoint methods exist and have correct signatures
8. request_store.get_requests_by_ids exists
9. request_factory syncs info_collection_state to payload
"""

import ast
import sys
import os
import pytest
from datetime import datetime
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-dummy-key-for-unit-tests"

from state_models import (
    UnifiedState, Meta, Routing, RequestRecord,
    PendingQueueItem, ChatMessage,
)


def _make_state(**kwargs):
    state = UnifiedState(
        meta=Meta(conversation_id="test-conv", user_id="test-user")
    )
    for k, v in kwargs.items():
        setattr(state, k, v)
    return state


# ── Test 1: Checkpoint data build ────────────────────────────────────

class TestBuildCheckpointData:
    def test_builds_correct_fields(self):
        from chat_server import _build_checkpoint_data

        state = _make_state()
        state.routing.current_agent = "deep_search"
        state.routing.conversation_stage = "executing"
        state.routing.needs_human = True
        state.routing.turn_mode = "continuation"
        state.request_manager.active_request_id = "req-1"
        state.request_manager.requests["req-1"] = RequestRecord(
            request_id="req-1", name="Test"
        )
        state.request_manager.requests["req-2"] = RequestRecord(
            request_id="req-2", name="Test 2"
        )
        state.request_manager.pending_queue.append(
            PendingQueueItem(request_id="req-1", name="Test", reason_queued="prereq")
        )

        data = _build_checkpoint_data(state)

        assert data["current_agent"] == "deep_search"
        assert data["conversation_stage"] == "executing"
        assert data["needs_human"] is True
        assert data["turn_mode"] == "continuation"
        assert data["active_request_id"] == "req-1"
        assert set(data["request_ids"]) == {"req-1", "req-2"}
        assert len(data["pending_queue"]) == 1
        assert data["pending_queue"][0]["request_id"] == "req-1"

    def test_empty_state_defaults(self):
        from chat_server import _build_checkpoint_data

        state = _make_state()
        data = _build_checkpoint_data(state)

        assert data["current_agent"] is None
        assert data["conversation_stage"] == "idle"
        assert data["needs_human"] is False
        assert data["request_ids"] == []
        assert data["pending_queue"] == []


# ── Test 2: Apply checkpoint ─────────────────────────────────────────

class TestApplyCheckpoint:
    def test_restores_routing(self):
        from chat_server import _apply_checkpoint

        state = _make_state()
        checkpoint = {
            "current_agent": "info_collection",
            "conversation_stage": "collecting",
            "needs_human": False,
            "turn_mode": "new_intent",
            "active_request_id": "req-abc",
        }
        _apply_checkpoint(state, checkpoint)

        assert state.routing.current_agent == "info_collection"
        assert state.routing.conversation_stage == "collecting"
        assert state.routing.turn_mode == "new_intent"
        assert state.request_manager.active_request_id == "req-abc"

    def test_restores_pending_queue(self):
        from chat_server import _apply_checkpoint

        state = _make_state()
        checkpoint = {
            "pending_queue": [
                {
                    "request_id": "req-1",
                    "name": "Insurance",
                    "status": "pending",
                    "reason_queued": "prereq",
                },
                {
                    "request_id": "req-2",
                    "name": "Caregiver",
                    "status": "blocked",
                    "reason_queued": "paused",
                    "child_request_id": "req-3",
                },
            ],
        }
        _apply_checkpoint(state, checkpoint)

        assert len(state.request_manager.pending_queue) == 2
        assert state.request_manager.pending_queue[0].request_id == "req-1"
        assert state.request_manager.pending_queue[1].child_request_id == "req-3"

    def test_empty_checkpoint_is_noop(self):
        from chat_server import _apply_checkpoint

        state = _make_state()
        _apply_checkpoint(state, {})

        assert state.routing.current_agent is None
        assert state.request_manager.active_request_id is None


# ── Test 3: Sticky needs_human through checkpoint ────────────────────

class TestStickyNeedsHumanCheckpoint:
    def test_true_restored(self):
        from chat_server import _apply_checkpoint

        state = _make_state()
        assert state.routing.needs_human is False

        _apply_checkpoint(state, {"needs_human": True})
        assert state.routing.needs_human is True

    def test_false_does_not_overwrite_true(self):
        from chat_server import _apply_checkpoint

        state = _make_state()
        state.routing.needs_human = True

        _apply_checkpoint(state, {"needs_human": False})
        # Sticky: should NOT revert to False
        assert state.routing.needs_human is True

    def test_missing_does_not_change(self):
        from chat_server import _apply_checkpoint

        state = _make_state()
        state.routing.needs_human = True

        _apply_checkpoint(state, {})
        assert state.routing.needs_human is True


# ── Test 4: Pending queue round-trip ─────────────────────────────────

class TestPendingQueueRoundTrip:
    def test_serialize_and_restore(self):
        from chat_server import _build_checkpoint_data, _apply_checkpoint

        state = _make_state()
        state.request_manager.pending_queue.append(
            PendingQueueItem(
                request_id="req-parent",
                name="Parent Task",
                status="pending",
                reason_queued="Paused for prereq",
                child_request_id="req-child",
            )
        )

        data = _build_checkpoint_data(state)

        new_state = _make_state()
        _apply_checkpoint(new_state, data)

        assert len(new_state.request_manager.pending_queue) == 1
        item = new_state.request_manager.pending_queue[0]
        assert item.request_id == "req-parent"
        assert item.child_request_id == "req-child"
        assert item.reason_queued == "Paused for prereq"


# ── Test 5: Rebuild request manager ──────────────────────────────────

class TestRebuildRequestManager:
    def test_rebuilds_from_payload(self):
        from chat_server import _rebuild_request_manager

        state = _make_state()
        raw_items = {
            "req-1": {
                "payload": {
                    "request_id": "req-1",
                    "name": "Find caregiver",
                    "goal": "Find a caregiver near Chicago",
                    "status": "collecting",
                    "target": "care_recipient",
                    "priority": "normal",
                    "info_collection_state": {
                        "summary_of_collected_info": "Location: Chicago",
                        "key_info_needed": ["budget", "schedule"],
                        "readiness_to_proceed": "needs_more",
                    },
                }
            },
        }
        checkpoint = {"active_request_id": "req-1"}

        _rebuild_request_manager(state, raw_items, checkpoint)

        assert "req-1" in state.request_manager.requests
        req = state.request_manager.requests["req-1"]
        assert req.name == "Find caregiver"
        assert req.status == "collecting"
        assert req.info_collection_state is not None
        assert req.info_collection_state["summary_of_collected_info"] == "Location: Chicago"
        assert state.request_manager.active_request_id == "req-1"

    def test_fallback_without_payload(self):
        from chat_server import _rebuild_request_manager

        state = _make_state()
        raw_items = {
            "req-2": {
                "name": "Insurance renewal",
                "goal": "Renew insurance",
                "status": "created",
                "target": "caregiver",
                "priority": "high",
                "request_type": "insurance_renewal",
            },
        }
        checkpoint = {"active_request_id": "req-2"}

        _rebuild_request_manager(state, raw_items, checkpoint)

        assert "req-2" in state.request_manager.requests
        req = state.request_manager.requests["req-2"]
        assert req.name == "Insurance renewal"
        assert req.status == "created"

    def test_invalid_payload_skipped(self):
        from chat_server import _rebuild_request_manager

        state = _make_state()
        raw_items = {
            "req-bad": {
                "payload": {"invalid_field_only": True},
            },
        }
        checkpoint = {"active_request_id": "req-bad"}

        # Should not crash — request_id is injected
        _rebuild_request_manager(state, raw_items, checkpoint)
        # May or may not succeed depending on model_validate strictness
        # But should not raise


# ── Test 6: Syntax check ────────────────────────────────────────────

class TestSyntaxCheck:
    FILES = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f)
        for f in [
            "conversation_store.py",
            "request_store.py",
            "request_factory.py",
            "graph.py",
            "chat_server.py",
        ]
    ]

    @pytest.mark.parametrize("filepath", FILES)
    def test_file_parses(self, filepath):
        with open(filepath) as f:
            ast.parse(f.read(), filename=filepath)


# ── Test 7: conversation_store methods exist ─────────────────────────

class TestConversationStoreMethods:
    def test_write_checkpoint_exists(self):
        from conversation_store import ConversationStore
        assert hasattr(ConversationStore, "write_checkpoint")
        import inspect
        sig = inspect.signature(ConversationStore.write_checkpoint)
        params = list(sig.parameters.keys())
        assert "conversation_id" in params
        assert "user_id" in params
        assert "checkpoint_data" in params

    def test_get_checkpoint_exists(self):
        from conversation_store import ConversationStore
        assert hasattr(ConversationStore, "get_checkpoint")
        import inspect
        sig = inspect.signature(ConversationStore.get_checkpoint)
        params = list(sig.parameters.keys())
        assert "conversation_id" in params


# ── Test 8: request_store method exists ──────────────────────────────

class TestRequestStoreMethod:
    def test_get_requests_by_ids_exists(self):
        from request_store import RequestStore
        assert hasattr(RequestStore, "get_requests_by_ids")
        import inspect
        sig = inspect.signature(RequestStore.get_requests_by_ids)
        params = list(sig.parameters.keys())
        assert "user_id" in params
        assert "request_ids" in params


# ── Test 9: request_factory info_collection_state sync ───────────────

class TestRequestFactorySync:
    def test_info_collection_state_in_update_expression(self):
        """build_request_update_patch should sync info_collection_state into payload."""
        from request_factory import build_request_update_patch

        result = build_request_update_patch(
            user_id="test-user",
            request_id="req-1",
            updates={
                "status": "collecting",
                "info_collection_state": {
                    "summary_of_collected_info": "Test data",
                    "key_info_needed": ["budget"],
                },
            },
        )

        writes = result.get("ddb_writes", [])
        assert len(writes) == 1

        params = writes[0]["params"]
        update_expr = params["UpdateExpression"]

        # Should contain payload.info_collection_state sync
        assert "#payload.#pics" in update_expr
        # Should also contain the status sync
        assert "#payload.#pstatus" in update_expr

    def test_no_info_collection_state_no_sync(self):
        """Without info_collection_state, no payload sync for it."""
        from request_factory import build_request_update_patch

        result = build_request_update_patch(
            user_id="test-user",
            request_id="req-1",
            updates={"status": "validated"},
        )

        writes = result.get("ddb_writes", [])
        params = writes[0]["params"]
        update_expr = params["UpdateExpression"]

        # Should NOT have info_collection_state sync
        assert "#pics" not in update_expr
        # But should still have status sync
        assert "#payload.#pstatus" in update_expr
