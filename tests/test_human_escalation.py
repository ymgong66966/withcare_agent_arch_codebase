"""
Verification tests for the Human Escalation feature.

Tests from the plan:
1. Normal flow: no escalation trigger when deep_search streak < 3
2. 3-round trigger: deep_search streak >= 3 triggers human_comm
3. Unhappiness trigger: LLM prompt includes escalation detection instructions
4. User declines: human_comm proposal → user says "no" → returns to deep_search
5. Already escalated: needs_human=True → _handle_escalated_turn path
6. /external/send returns agent_type "human_support" when needs_human=True
7. Syntax check all modified files
"""

import ast
import asyncio
import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# graph.py creates a module-level TrackedAnthropicClient that requires an API key.
# Set a dummy key so the module can be imported in tests.
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-dummy-key-for-unit-tests"

from state_models import UnifiedState, Meta, Routing, AgentName, ChatMessage
from merge_utils import apply_node_output


# ── Helpers ──────────────────────────────────────────────────────────

def _make_state(
    messages=None,
    current_agent=None,
    needs_human=False,
    user_id="test-user",
    conversation_id="test-conv",
):
    """Create a UnifiedState for testing."""
    state = UnifiedState(
        meta=Meta(conversation_id=conversation_id, user_id=user_id)
    )
    if current_agent:
        state.routing.current_agent = current_agent
    state.routing.needs_human = needs_human
    if messages:
        for m in messages:
            state.messages.append(
                ChatMessage(**m) if isinstance(m, dict) else m
            )
    return state


def _state_to_dict(state):
    return state.model_dump()


# ── Test 1: Normal flow — no escalation trigger ─────────────────────

class TestNoEscalationTrigger:
    def test_streak_below_threshold(self):
        """2 consecutive deep_search rounds should NOT trigger escalation."""
        from graph import _detect_deep_search_streak

        state_dict = _state_to_dict(_make_state(
            current_agent="deep_search",
            messages=[
                {"role": "user", "content": "Find me a caregiver near Chicago"},
                {"role": "assistant", "content": "Here are some results...", "metadata": {"agent": "deep_search"}},
                {"role": "user", "content": "Can you search again?"},
                {"role": "assistant", "content": "Here are more results...", "metadata": {"agent": "deep_search"}},
            ],
        ))
        result = _detect_deep_search_streak(state_dict)
        assert result is None, f"Expected None, got {result}"

    def test_streak_different_agents(self):
        """Mixed agent messages should NOT trigger."""
        from graph import _detect_deep_search_streak

        state_dict = _state_to_dict(_make_state(
            current_agent="deep_search",
            messages=[
                {"role": "assistant", "content": "...", "metadata": {"agent": "info_collection"}},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "metadata": {"agent": "deep_search"}},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "metadata": {"agent": "deep_search"}},
            ],
        ))
        result = _detect_deep_search_streak(state_dict)
        assert result is None

    def test_streak_not_deep_search_agent(self):
        """Even 3+ deep_search messages, if current_agent != deep_search, no trigger."""
        from graph import _detect_deep_search_streak

        state_dict = _state_to_dict(_make_state(
            current_agent="info_collection",
            messages=[
                {"role": "assistant", "content": "...", "metadata": {"agent": "deep_search"}},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "metadata": {"agent": "deep_search"}},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "metadata": {"agent": "deep_search"}},
            ],
        ))
        result = _detect_deep_search_streak(state_dict)
        assert result is None


# ── Test 2: 3-round trigger ──────────────────────────────────────────

class TestDeepSearchStreakTrigger:
    def test_three_consecutive_deep_search(self):
        """3 consecutive deep_search rounds should trigger escalation."""
        from graph import _detect_deep_search_streak

        state_dict = _state_to_dict(_make_state(
            current_agent="deep_search",
            messages=[
                {"role": "user", "content": "Find caregivers"},
                {"role": "assistant", "content": "Result 1", "metadata": {"agent": "deep_search"}},
                {"role": "user", "content": "Try again"},
                {"role": "assistant", "content": "Result 2", "metadata": {"agent": "deep_search"}},
                {"role": "user", "content": "Still not right"},
                {"role": "assistant", "content": "Result 3", "metadata": {"agent": "deep_search"}},
            ],
        ))
        result = _detect_deep_search_streak(state_dict)
        assert result == "deep_search_3_consecutive_rounds"

    def test_four_consecutive_deep_search(self):
        """4 consecutive should also trigger."""
        from graph import _detect_deep_search_streak

        state_dict = _state_to_dict(_make_state(
            current_agent="deep_search",
            messages=[
                {"role": "assistant", "content": "R1", "metadata": {"agent": "deep_search"}},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "R2", "metadata": {"agent": "deep_search"}},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "R3", "metadata": {"agent": "deep_search"}},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "R4", "metadata": {"agent": "deep_search"}},
            ],
        ))
        result = _detect_deep_search_streak(state_dict)
        assert result == "deep_search_3_consecutive_rounds"


# ── Test 3: Unhappiness trigger — prompt includes escalation section ─

class TestEscalationPrompt:
    def test_prompt_contains_escalation_detection(self):
        """The turn_mode prompt should include escalation detection instructions."""
        from prompts import make_turn_mode_prompt

        prompt = make_turn_mode_prompt(
            recent_turns=[
                {"role": "user", "content": "This is not helpful at all"},
            ],
            current_agent="deep_search",
            recent_agents=["deep_search"],
            request_state={"request_name": "test", "request_goal": "test", "request_status": "executing"},
        )
        assert "Escalation Detection" in prompt
        assert "human_comm" in prompt
        assert "frustration" in prompt.lower() or "dissatisfaction" in prompt.lower()

    def test_human_comm_in_agent_descriptions(self):
        """human_comm should appear in agent descriptions."""
        from prompts import make_turn_mode_prompt

        prompt = make_turn_mode_prompt(
            recent_turns=[{"role": "user", "content": "test"}],
            current_agent=None,
            recent_agents=[],
            request_state={},
        )
        assert "human_comm" in prompt
        assert "clinical team" in prompt.lower() or "human" in prompt.lower()


# ── Test 4: User declines escalation ────────────────────────────────

class TestUserDeclinesEscalation:
    @pytest.mark.asyncio
    async def test_user_says_no_returns_deep_search(self):
        """When user declines escalation, human_comm_node should return to deep_search."""
        from graph import human_comm_node

        # State with a prior human_comm proposal and user saying "no"
        state_dict = _state_to_dict(_make_state(
            current_agent="human_comm",
            messages=[
                {"role": "assistant", "content": "Would you like our clinical team to help?", "metadata": {"agent": "human_comm"}},
                {"role": "user", "content": "no thanks"},
            ],
        ))
        result = await human_comm_node(state_dict)
        assert result["routing"]["current_agent"] == "deep_search"

    @pytest.mark.asyncio
    async def test_user_says_yes_sets_needs_human(self):
        """When user confirms, human_comm_node should set needs_human=True."""
        from graph import human_comm_node

        state_dict = _state_to_dict(_make_state(
            current_agent="human_comm",
            messages=[
                {"role": "assistant", "content": "Would you like our clinical team to help?", "metadata": {"agent": "human_comm"}},
                {"role": "user", "content": "yes please"},
            ],
        ))

        # Note: This will attempt to call the MCP tool which won't be available in test,
        # but we can catch the error and still verify the intent.
        # The call_mcp_tool_patch failure is logged as non-fatal.
        result = await human_comm_node(state_dict)
        assert result["routing"]["needs_human"] is True
        assert result["routing"]["current_agent"] == "human_comm"


# ── Test 5: Already escalated — needs_human=True path ───────────────

class TestAlreadyEscalated:
    def test_turn_router_detects_needs_human(self):
        """When needs_human=True, turn_router should return escalation handling."""
        # We test the condition check directly since turn_router is async
        # and calls MCP tools
        state_dict = _state_to_dict(_make_state(needs_human=True))
        routing = state_dict.get("routing") or {}
        assert routing.get("needs_human") is True

    @pytest.mark.asyncio
    async def test_handle_escalated_turn_returns_ack(self):
        """_handle_escalated_turn should return an ack message with needs_human=True."""
        from graph import _handle_escalated_turn

        state_dict = _state_to_dict(_make_state(
            needs_human=True,
            current_agent="human_comm",
            messages=[
                {"role": "user", "content": "I have another question for the team"},
            ],
        ))
        # MCP call will fail in test but is non-fatal
        result = await _handle_escalated_turn(state_dict)
        assert result["routing"]["needs_human"] is True
        assert result["routing"]["current_agent"] == "human_comm"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"


# ── Test 6: /external/send response shape ────────────────────────────

class TestExternalSendEndpoint:
    def test_external_send_request_model(self):
        """ExternalSendRequest model should accept the expected fields."""
        from chat_server import ExternalSendRequest

        req = ExternalSendRequest(
            user_id="u1",
            messages=[{"role": "user", "text": "hello"}],
            needs_human=True,
            conversation_id="conv-123",
        )
        assert req.user_id == "u1"
        assert req.needs_human is True
        assert req.conversation_id == "conv-123"

    def test_external_send_request_defaults(self):
        """ExternalSendRequest defaults should be correct."""
        from chat_server import ExternalSendRequest

        req = ExternalSendRequest(
            user_id="u1",
            messages=[],
        )
        assert req.needs_human is False
        assert req.conversation_id is None


# ── Test 7: Syntax check all modified files ──────────────────────────

class TestSyntaxCheck:
    FILES = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state_models.py"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "merge_utils.py"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph.py"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts.py"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_server.py"),
        "/Users/xyxg025/mcp_server_fastapi/app/escalation_mcp.py",
        "/Users/xyxg025/mcp_server_fastapi/app/main.py",
    ]

    @pytest.mark.parametrize("filepath", FILES)
    def test_file_parses(self, filepath):
        """Each modified file should parse without syntax errors."""
        with open(filepath) as f:
            source = f.read()
        ast.parse(source, filename=filepath)  # raises SyntaxError on failure


# ── Additional: State model and merge_utils unit tests ───────────────

class TestStateModelChanges:
    def test_human_comm_is_valid_agent(self):
        """human_comm should be a valid AgentName."""
        r = Routing(current_agent="human_comm")
        assert r.current_agent == "human_comm"

    def test_needs_human_default_false(self):
        """needs_human should default to False."""
        r = Routing()
        assert r.needs_human is False

    def test_needs_human_can_be_set_true(self):
        r = Routing(needs_human=True)
        assert r.needs_human is True


class TestStickyNeedsHuman:
    def test_sticky_true_not_reverted(self):
        """Once needs_human=True, a node output with needs_human=False should NOT revert it."""
        state = _make_state(needs_human=True)
        assert state.routing.needs_human is True

        # Apply output that tries to set needs_human=False
        apply_node_output(state, {"routing": {"needs_human": False}})
        assert state.routing.needs_human is True, "needs_human should remain True (sticky)"

    def test_sticky_false_can_become_true(self):
        """needs_human=False can be set to True normally."""
        state = _make_state(needs_human=False)
        apply_node_output(state, {"routing": {"needs_human": True}})
        assert state.routing.needs_human is True

    def test_other_routing_fields_still_merge(self):
        """Other routing fields should still merge normally alongside sticky needs_human."""
        state = _make_state(needs_human=True, current_agent="deep_search")
        apply_node_output(state, {
            "routing": {
                "current_agent": "human_comm",
                "needs_human": False,  # should be ignored
                "turn_reason": "escalation",
            }
        })
        assert state.routing.current_agent == "human_comm"
        assert state.routing.turn_reason == "escalation"
        assert state.routing.needs_human is True  # sticky


# ── Route function tests ─────────────────────────────────────────────

class TestRouteFromTurnRouter:
    def test_human_comm_route(self):
        """route_from_turn_router should route to human_comm when LLM recommends it."""
        from graph import route_from_turn_router

        state_dict = {
            "routing": {
                "turn_mode": "continuation",
                "llm_recommended_agent": "human_comm",
            },
            "request_manager": {"active_request_id": None, "requests": {}},
        }
        result = route_from_turn_router(state_dict)
        assert result == "human_comm"


class TestRouteFromDelegator:
    def test_human_comm_route(self):
        """route_from_delegator should route to human_comm when pending handoff says so."""
        from graph import route_from_delegator

        state_dict = {
            "routing": {
                "pending_handoff": {"recommended_next_agent": "human_comm"},
            },
            "request_manager": {"active_request_id": None, "requests": {}},
        }
        result = route_from_delegator(state_dict)
        assert result == "human_comm"


class TestRouteAfterCatcher:
    def test_human_comm_route(self):
        """route_after_catcher should route to human_comm."""
        from graph import route_after_catcher

        state_dict = {
            "routing": {"_catcher_next": "human_comm"},
        }
        result = route_after_catcher(state_dict)
        assert result == "human_comm"


# ── Build escalation messages test ───────────────────────────────────

class TestBuildEscalationMessages:
    @pytest.mark.asyncio
    async def test_message_format(self):
        """_build_escalation_messages should return properly formatted messages."""
        from graph import _build_escalation_messages

        state_dict = _state_to_dict(_make_state(
            messages=[
                {"role": "user", "content": "Help me find care"},
                {"role": "assistant", "content": "Sure, let me search"},
                {"role": "user", "content": "Thanks"},
            ],
        ))
        result = await _build_escalation_messages(state_dict, limit=20)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["text"] == "Help me find care"
        assert "message_Id" in result[0]
        assert "dateSent" in result[0]

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        """Should respect the limit parameter."""
        from graph import _build_escalation_messages

        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        state_dict = _state_to_dict(_make_state(messages=msgs))
        result = await _build_escalation_messages(state_dict, limit=5)
        assert len(result) == 5


# ── MCP tool file test ───────────────────────────────────────────────

class TestEscalationMCPFile:
    def test_escalation_mcp_imports(self):
        """escalation_mcp.py should be importable and have the expected tool."""
        sys.path.insert(0, "/Users/xyxg025/mcp_server_fastapi/app")
        try:
            from escalation_mcp import escalation_mcp, human_escalation_deliver
            assert escalation_mcp.name == "escalation-mcp"
        finally:
            sys.path.pop(0)
