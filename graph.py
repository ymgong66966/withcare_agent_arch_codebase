from __future__ import annotations

import json
import logging
import os
from typing import Dict, Any, Literal, Optional, List, TypedDict
from datetime import datetime

_logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END

from routing_utils import infer_turn_mode, last_user_text
from id_utils import new_uuid
from request_factory import build_request_patch, build_prereq_switch_patch, build_accept_prereq_patch, build_request_update_patch
from mcp_wrappers import MCPClientManager, MCPServerConfig, call_mcp_tool_patch, call_memory_tool

from prompts import (
    make_collection_plan_prompt,
    make_info_collection_summarize_prompt,
    default_collection_plan,
    default_parse_user_answer,
    llm_collection_plan,
    llm_info_collection_summarize,
    llm_prerequisite_acceptance_response,
    make_turn_mode_prompt,
    default_turn_mode_decision,
    llm_turn_mode_decision,
    make_upstream_delegator_prompt,
    default_upstream_delegation,
    llm_upstream_delegation,
    llm_emotional_support,
    llm_reconcile_facts,
    filter_questions_with_known_facts,
)
from anthropic_client import TrackedAnthropicClient
from retrieval import fetch_similar_requests, fetch_prior_qas_for_questions
from context_bundle import get_context_bundle, bundle_to_prompt_block
from deep_search_prompts import (
    TOOL_DESCRIPTIONS,
    DEEP_SEARCH_STRATEGY_PROMPT,
    SEARCH_CONTINUATION_PROMPT,
    SEARCH_SUMMARY_PROMPT,
    DOMAIN_EXPERT_SYNTHESIS_PROMPT,
    QUICK_ANSWER_PROMPT,
    QUICK_ANSWER_DECISION_PROMPT,
)

RouteKey = Literal[
    "upstream_delegator",
    "front_end",
    "info_collection",
    "deep_search",
    "user_info",
    "domain_expert",
    "quick_answer",
    "human_comm",
    "respond",
    "escalate",
]

class GraphState(TypedDict, total=False):
    meta: Dict[str, Any]
    messages: List[Dict[str, Any]]
    routing: Dict[str, Any]
    request_manager: Dict[str, Any]
    user_context: Dict[str, Any]
    tools: Dict[str, Any]
    slots_upsert: Dict[str, Any]
    open_questions_upsert: Dict[str, Any]
    artifacts_append: Dict[str, Any]
    ddb_writes: List[Dict[str, Any]]
    info_collection_debug: Dict[str, Any]

def _detect_user_language(state: Dict[str, Any]) -> str:
    """Detect user's language from their most recent message. Returns 'zh' or 'en'."""
    text = last_user_text(state)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars > max(1, len(text) * 0.1):
        return "zh"
    return "en"

def _msg(state: Dict[str, Any], zh: str, en: str) -> str:
    """Return zh or en string based on detected user language."""
    return zh if _detect_user_language(state) == "zh" else en

def _get_current_agent(state: Dict[str, Any]) -> Optional[str]:
    return ((state.get("routing") or {}).get("current_agent"))

def _get_active_request_id(state: Dict[str, Any]) -> Optional[str]:
    return ((state.get("request_manager") or {}).get("active_request_id"))

def _get_active_request(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rm = state.get("request_manager") or {}
    rid = rm.get("active_request_id")
    return ((rm.get("requests") or {}).get(rid)) if rid else None

def _get_prereq_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    req = _get_active_request(state) or {}
    return (req.get("prereq_gate") or {})

def _detect_deep_search_streak(state: Dict[str, Any]) -> Optional[str]:
    """Count consecutive deep_search assistant messages walking backwards.
    Returns a trigger string if >= 3 consecutive rounds, else None.
    """
    messages = state.get("messages") or []
    streak = 0
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "user":
            continue
        if role == "assistant":
            agent = msg.get("agent") or (msg.get("metadata") or {}).get("agent")
            if agent == "deep_search":
                streak += 1
            else:
                break
    cur = _get_current_agent(state)
    if streak >= 3 and cur == "deep_search":
        return "deep_search_3_consecutive_rounds"
    return None


async def _human_comm_llm_reply(
    state: Dict[str, Any],
    *,
    instruction: str,
    fallback_zh: str,
    fallback_en: str,
) -> str:
    """Generate a human_comm reply via LLM, with a rigid fallback only if the call fails."""
    meta = state.get("meta") or {}
    messages = state.get("messages") or []
    conversation_context = "\n".join(
        f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')[:200]}"
        for m in messages[-10:]
    )
    lang_hint = "Respond in Chinese." if _detect_user_language(state) == "zh" else "Respond in English."

    client = TrackedAnthropicClient(
        session_id=meta.get("conversation_id", ""),
        agent_role="human_comm",
        user_id=meta.get("user_id", ""),
    )
    try:
        resp = await client.create_message(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=(
                "You are a warm and supportive care coordinator assistant. "
                "Be natural and conversational. Do NOT use bullet points or lists. "
                f"{lang_hint} {instruction}"
            ),
            messages=[{
                "role": "user",
                "content": f"Recent conversation:\n{conversation_context}",
            }],
        )
        return resp.content[0].text
    except Exception as e:
        _logger.warning(f"human_comm LLM call failed, using fallback: {e}")
        return _msg(state, fallback_zh, fallback_en)


async def _handle_escalated_turn(state: Dict[str, Any]) -> Dict[str, Any]:
    """When needs_human=True, forward the user message to the lambda and return ack."""
    meta = state.get("meta") or {}
    user_id = meta.get("user_id", "")
    conversation_id = meta.get("conversation_id", "")
    user_text = last_user_text(state)
    message_id = new_uuid("msg")

    escalation_messages = [{
        "role": "user",
        "text": user_text,
        "message_Id": message_id,
        "dateSent": datetime.utcnow().isoformat(),
    }]

    try:
        await call_mcp_tool_patch(
            mgr=MCP_MGR,
            server=ESCALATION_SERVER,
            tool_name="human_escalation_deliver",
            arguments={
                "user_id": user_id,
                "chat_id": conversation_id,
                "messages": escalation_messages,
            },
            purpose="Forward user message to human support team",
        )
    except Exception as e:
        _logger.warning(f"Escalation delivery failed (non-fatal): {e}")

    ack = await _human_comm_llm_reply(
        state,
        instruction=(
            "The user's message has just been forwarded to the clinical support team. "
            "Generate a brief acknowledgement (1-2 sentences) letting the user know their "
            "message was sent and the team will follow up. Be reassuring."
        ),
        fallback_zh="您的消息已转发给我们的临床团队，他们会尽快与您联系。",
        fallback_en="Your message has been forwarded to our clinical team. They will reach out to you shortly.",
    )

    return {
        "routing": {
            "current_agent": "human_comm",
            "needs_human": True,
        },
        "messages": [{
            "role": "assistant",
            "content": ack,
            "metadata": {"agent": "human_comm"},
        }],
    }


async def _build_escalation_messages(state: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """Build a list of recent messages formatted for the escalation lambda.

    Reads from DynamoDB (UserConversationTable) first so that messages survive
    pod restarts. Falls back to the in-memory state["messages"] if DDB is
    unavailable.
    """
    from conversation_store import get_conversation_store

    meta = state.get("meta") or {}
    conversation_id = meta.get("conversation_id", "")
    raw_messages = None

    # Try DDB first — authoritative source that survives pod restarts
    if conversation_id:
        try:
            store = get_conversation_store()
            all_msgs = await store.get_messages_for_conversation(conversation_id)
            if all_msgs:
                raw_messages = [
                    {
                        "role": m.get("role", "user"),
                        "content": m.get("content", ""),
                        "message_id": m.get("message_id"),
                        "ts": m.get("timestamp", ""),
                    }
                    for m in all_msgs
                ]
        except Exception as e:
            _logger.warning(f"DDB conversation read failed, falling back to in-memory: {e}")

    # Fallback to in-memory
    if not raw_messages:
        raw_messages = state.get("messages") or []

    result = []
    for msg in raw_messages[-limit:]:
        result.append({
            "role": msg.get("role", "user"),
            "text": msg.get("content", ""),
            "message_Id": msg.get("message_id") or new_uuid("msg"),
            "dateSent": msg.get("ts", datetime.utcnow().isoformat()),
        })
    return result


async def human_comm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Human escalation agent node — proposal and confirmation modes."""
    meta = state.get("meta") or {}
    user_id = meta.get("user_id", "")
    conversation_id = meta.get("conversation_id", "")
    user_text = last_user_text(state)
    messages = state.get("messages") or []

    # Check if there's a prior human_comm assistant message (proposal already sent)
    has_prior_proposal = any(
        m.get("role") == "assistant"
        and (m.get("agent") or (m.get("metadata") or {}).get("agent")) == "human_comm"
        for m in messages
    )

    if not has_prior_proposal:
        # ── Proposal mode: generate warm proposal via LLM ──
        client = TrackedAnthropicClient(
            session_id=conversation_id,
            agent_role="human_comm",
            user_id=user_id,
        )

        proposal_prompt = (
            "You are a warm and supportive care coordinator assistant. "
            "The automated search has not been able to fully address the user's needs "
            "after multiple attempts. Generate a brief, empathetic message (2-3 sentences) "
            "asking if the user would like a member of our clinical team to reach out to "
            "them directly to help. Be natural and conversational. Do NOT use bullet points. "
            "End with a clear yes/no question."
        )

        conversation_context = "\n".join(
            f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')[:200]}"
            for m in messages[-10:]
        )

        try:
            resp = await client.create_message(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=proposal_prompt,
                messages=[{"role": "user", "content": f"Recent conversation:\n{conversation_context}\n\nGenerate the escalation proposal message."}],
            )
            proposal_text = resp.content[0].text
        except Exception:
            proposal_text = _msg(
                state,
                "看起来我们目前的搜索还没有完全满足您的需求。您希望我们的临床团队直接联系您来帮助吗？",
                "It looks like our search hasn't fully addressed your needs yet. Would you like a member of our clinical team to reach out to you directly to help?",
            )

        return {
            "routing": {"current_agent": "human_comm"},
            "messages": [{
                "role": "assistant",
                "content": proposal_text,
                "metadata": {"agent": "human_comm"},
            }],
        }

    # ── Confirmation mode: user is responding to proposal ──
    answer = _user_yes_no(user_text)

    if answer is True:
        # User confirmed — activate escalation and deliver message history
        escalation_messages = await _build_escalation_messages(state)

        try:
            await call_mcp_tool_patch(
                mgr=MCP_MGR,
                server=ESCALATION_SERVER,
                tool_name="human_escalation_deliver",
                arguments={
                    "user_id": user_id,
                    "chat_id": conversation_id,
                    "messages": escalation_messages,
                },
                purpose="Deliver conversation to human support team after user confirmation",
            )
        except Exception as e:
            _logger.warning(f"Escalation delivery failed (non-fatal): {e}")

        confirm_text = await _human_comm_llm_reply(
            state,
            instruction=(
                "The user just confirmed they want the clinical team to help. Their conversation "
                "has been forwarded. Generate a warm confirmation (2-3 sentences) letting them know "
                "the team will follow up, and that they can continue chatting in the meantime."
            ),
            fallback_zh="好的，我已经将您的对话转给了我们的临床团队。他们会尽快与您联系。在等待期间，如果有任何其他问题，请随时告诉我。",
            fallback_en="I've forwarded your conversation to our clinical team. They will reach out to you shortly. In the meantime, feel free to let me know if there's anything else I can help with.",
        )

        return {
            "routing": {
                "current_agent": "human_comm",
                "needs_human": True,
            },
            "messages": [{
                "role": "assistant",
                "content": confirm_text,
                "metadata": {"agent": "human_comm"},
            }],
        }

    if answer is False:
        # User declined — return to deep_search
        decline_text = await _human_comm_llm_reply(
            state,
            instruction=(
                "The user was offered a handoff to the clinical team but declined. "
                "Generate a brief, friendly acknowledgement (1-2 sentences) and offer to "
                "continue helping with the search. Do not pressure them about the clinical team."
            ),
            fallback_zh="没问题！我们继续搜索。请告诉我您还想找什么，或者需要调整哪些条件。",
            fallback_en="No problem! Let's continue searching. Let me know what else you'd like to look for, or if you'd like to adjust any criteria.",
        )

        return {
            "routing": {"current_agent": "deep_search"},
            "messages": [{
                "role": "assistant",
                "content": decline_text,
                "metadata": {"agent": "human_comm"},
            }],
        }

    # Ambiguous — re-ask
    reask_text = await _human_comm_llm_reply(
        state,
        instruction=(
            "The user's response to the clinical team handoff proposal was ambiguous — "
            "it was not clearly yes or no. Generate a brief, gentle message (1-2 sentences) "
            "asking them to clarify whether they would like the clinical team to reach out."
        ),
        fallback_zh="抱歉，我没有完全理解您的意思。您希望我们的临床团队联系您吗？",
        fallback_en="Sorry, I didn't quite catch that. Would you like our clinical team to reach out to you?",
    )

    return {
        "routing": {"current_agent": "human_comm"},
        "messages": [{
            "role": "assistant",
            "content": reask_text,
            "metadata": {"agent": "human_comm"},
        }],
    }


def _mk_pending_handoff_patch(next_agent: Optional[str], reason: str = "") -> Dict[str, Any]:
    return {"routing": {"pending_handoff": {"recommended_next_agent": next_agent, "reason": reason}}}


def _append_request_ddb_sync(
    patch: Dict[str, Any],
    state: Dict[str, Any],
    request_id: str,
    updates: Dict[str, Any],
) -> None:
    """Append a DDB update write to an existing patch for request status sync."""
    user_id = (state.get("meta") or {}).get("user_id", "")
    if not user_id or not request_id:
        return
    sync_patch = build_request_update_patch(
        user_id=user_id,
        request_id=request_id,
        updates=updates,
    )
    existing_writes = patch.get("ddb_writes", [])
    patch["ddb_writes"] = existing_writes + sync_patch.get("ddb_writes", [])

async def turn_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router = Continuation vs New Intent.
    Uses LLM-based decision making via Claude to intelligently route based on conversation context.
    Falls back to heuristic-based routing if LLM call fails.
    """
    routing = state.get("routing") or {}

    # ── If already in human escalation mode, deliver message and respond ──
    if routing.get("needs_human") is True:
        return await _handle_escalated_turn(state)

    # ── Condition 1: deep_search streak check (heuristic, before LLM call) ──
    streak_trigger = _detect_deep_search_streak(state)
    if streak_trigger:
        return {
            "routing": {
                "turn_mode": "continuation",
                "turn_reason": f"Escalation trigger: {streak_trigger}",
                "llm_recommended_agent": "human_comm",
            }
        }

    # Extract metadata for client initialization
    meta = state.get("meta") or {}
    user_id = meta.get("user_id", "user-unknown")
    conversation_id = meta.get("conversation_id", "conv-unknown")

    # Initialize TrackedAnthropicClient for this decision
    client = TrackedAnthropicClient(
        session_id=conversation_id,
        agent_role="turn_router",
        user_id=user_id,
    )

    try:
        # Use LLM-based decision
        decision = await llm_turn_mode_decision(state=state, client=client)
    except Exception as e:
        # Fallback to heuristic if LLM call fails
        import logging
        logging.warning(f"LLM turn mode decision failed: {e}. Using fallback heuristic.")
        decision = default_turn_mode_decision(state=state)

    return {
        "routing": {
            "turn_mode": decision.get("turn_mode", infer_turn_mode(state)),
            "turn_reason": decision.get("reason", ""),
            "llm_recommended_agent": decision.get("recommended_agent"),  # Store LLM's agent recommendation
        }
    }

def route_from_turn_router(state: Dict[str, Any]) -> RouteKey:
    routing = state.get("routing") or {}
    mode = routing.get("turn_mode") or "continuation"

    if mode == "new_intent":
        llm_recommended = routing.get("llm_recommended_agent")

        # quick_answer and user_info bypass upstream_delegator entirely —
        # they are read-only lookups with no request lifecycle, so send
        # them straight to the node without creating a new request.
        if llm_recommended == "quick_answer":
            return "quick_answer"
        if llm_recommended == "user_info":
            return "user_info"

        # Trust the LLM's new_intent classification — it already has full
        # conversation context and active request info to distinguish a
        # genuine new request from a Q&A answer.
        return "upstream_delegator"

    # GUARD: if active request is still awaiting user input, always route to
    # info_collection so it can finalize or continue the Q&A loop.
    req = _get_active_request(state) or {}
    if req.get("awaiting_user_input") is True:
        return "info_collection"

    # CONTINUATION: prefer LLM's recommended agent, fallback to current agent
    llm_recommended = routing.get("llm_recommended_agent")

    # Map LLM agent names to route keys
    if llm_recommended:
        agent_route_map = {
            "info_collection": "info_collection",
            "deep_search": "deep_search",
            "user_info": "user_info",
            "domain_expert": "domain_expert",
            "front_end_emotional_support": "front_end",
            "quick_answer": "quick_answer",
            "human_comm": "human_comm",
        }
        if llm_recommended in agent_route_map:
            return agent_route_map[llm_recommended]

    # Fallback to current agent
    cur = _get_current_agent(state)
    if cur == "info_collection":
        return "info_collection"
    if cur == "deep_search":
        return "deep_search"
    if cur == "user_info":
        return "user_info"
    if cur == "domain_expert":
        return "domain_expert"
    if cur == "front_end_emotional_support":
        return "front_end"
    if cur == "quick_answer":
        return "quick_answer"
    if cur == "human_comm":
        return "human_comm"

    return "upstream_delegator"

async def front_end_node(state: Dict[str, Any]) -> Dict[str, Any]:
    meta = state.get("meta") or {}
    conversation_id = meta.get("conversation_id", "conv-unknown")
    user_id = meta.get("user_id", "user-unknown")
    user_text = last_user_text(state)
    known_facts = _extract_known_facts(state)
    messages = state.get("messages") or []

    client = TrackedAnthropicClient(
        session_id=conversation_id,
        agent_role="front_end_emotional_support",
        user_id=user_id,
    )

    content = await llm_emotional_support(
        user_message=user_text,
        conversation_history=messages,
        known_facts=known_facts,
        client=client,
    )

    if not content:
        content = _msg(
            state,
            "我听到你了。照顾家人真的不容易，你愿意多说说现在的感受吗？",
            "I hear you. Caregiving is really hard, and what you're feeling makes complete sense. Want to tell me more about what's going on?",
        )

    return {
        "routing": {"current_agent": "front_end_emotional_support"},
        "messages": [{
            "role": "assistant",
            "content": content,
            "metadata": {"agent": "front_end_emotional_support"}
        }],
    }

def _asked_questions_from_request(req: Dict[str, Any]) -> List[Dict[str, str]]:
    return req.get("open_questions") or []

def _extract_known_facts(state: Dict[str, Any]) -> Dict[str, Any]:
    uc = state.get("user_context") or {}
    prof = (uc.get("profile_snapshot") or {})
    return {
        "caregiver": (prof.get("caregiver") or {}).get("facts", {}),
        "care_recipient": (prof.get("care_recipient") or {}).get("facts", {}),
    }


async def _build_memory_context_block(state: Dict[str, Any]) -> str:
    """
    Build a memory context block from the Fact Store and Event Store
    for injection into LLM prompts. Returns a text block or empty string.

    This is additive — it supplements the existing known_facts, not replaces it.
    """
    meta = state.get("meta") or {}
    user_id = meta.get("user_id", "")
    if not user_id:
        return ""

    req = _get_active_request(state) or {}
    user_text = last_user_text(state)

    try:
        bundle = await get_context_bundle(
            user_id=user_id,
            request_dict=req if req else None,
            message=user_text,
            k=5,
        )
        block = bundle_to_prompt_block(bundle)
        if block and block != "(No memory context available.)":
            return f"\n\n## Memory Context (from User Memory Framework)\n{block}"
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Memory context build failed (non-fatal): {e}")

    return ""


async def _build_memory_context_with_facts(
    state: Dict[str, Any],
) -> tuple:
    """
    Like _build_memory_context_block but also returns the structured
    profile_facts dict for deterministic question-to-fact matching.

    Returns (text_block: str, profile_facts: Dict[str, Any]).
    """
    meta = state.get("meta") or {}
    user_id = meta.get("user_id", "")
    if not user_id:
        return "", {}

    req = _get_active_request(state) or {}
    user_text = last_user_text(state)

    try:
        bundle = await get_context_bundle(
            user_id=user_id,
            request_dict=req if req else None,
            message=user_text,
            k=5,
        )
        block = bundle_to_prompt_block(bundle)
        text = ""
        if block and block != "(No memory context available.)":
            text = f"\n\n## Memory Context (from User Memory Framework)\n{block}"
        return text, bundle.profile_facts or {}
    except Exception as e:
        _logger.debug(f"Memory context build with facts failed (non-fatal): {e}")
        return "", {}


def _is_user_answering_questions(state: Dict[str, Any]) -> bool:
    rid = _get_active_request_id(state)
    if not rid:
        return False
    req = _get_active_request(state) or {}
    return bool(req.get("awaiting_user_input")) and len(_asked_questions_from_request(req)) > 0

def _user_says_decline_more_questions(text: str) -> bool:
    return any(x in text for x in ["不想回答", "别问了", "先用现有的", "不用问了", "proceed", "go ahead", "先继续"])

def _user_yes_no(text: str) -> Optional[bool]:
    yes = any(x in text for x in ["要", "好", "可以", "是", "yes", "ok", "sure"])
    no = any(x in text for x in ["不要", "不", "否", "no", "not now"])
    if yes and not no:
        return True
    if no and not yes:
        return False
    return None

async def upstream_delegator(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    NEW_INTENT only:
    - Decide what type of new intent (natural_progression, prerequisite_task, new_unrelated_task, same_agent_continue)
    - Route to appropriate agent
    - Manage request lifecycle (create new request if needed, pause current if needed)

    This function now uses LLM-based decision making via Claude.
    """
    # Extract metadata for client initialization
    meta = state.get("meta") or {}
    user_id = meta.get("user_id", "user-unknown")
    conversation_id = meta.get("conversation_id", "conv-unknown")

    # Initialize TrackedAnthropicClient for this decision
    client = TrackedAnthropicClient(
        session_id=conversation_id,
        agent_role="upstream_delegator",
        user_id=user_id,
    )

    # Fetch historical similar requests for context (like info_collection does)
    user_text = last_user_text(state)

    try:
        similar_historical = await fetch_similar_requests(
            user_id=user_id, request_text=user_text, top_k=3
        )
        similar_dicts = [s.model_dump() for s in similar_historical] if similar_historical else []
    except Exception:
        similar_dicts = []

    # Get LLM decision
    try:
        decision = await llm_upstream_delegation(state=state, client=client, similar_requests=similar_dicts)
    except Exception as e:
        import logging
        logging.warning(f"LLM upstream delegation failed: {e}. Using fallback heuristic.")
        rm = state.get("request_manager") or {}
        active_id = rm.get("active_request_id")
        current_req = None
        if active_id:
            req = (rm.get("requests") or {}).get(active_id)
            if req:
                current_req = {
                    "request_id": active_id,
                    "status": req.get("status"),
                }
        decision = default_upstream_delegation(last_user_text(state), current_req)

    # Extract decision fields
    action_type = decision.get("action_type")
    recommended_agent = decision.get("recommended_agent")
    current_request_action = decision.get("current_request_action")
    new_request_info = decision.get("new_request_info")
    reason = decision.get("reason", "")

    # Get current request state
    rm = state.get("request_manager") or {}
    active_id = rm.get("active_request_id")
    req = _get_active_request(state) if active_id else None

    # Build base patch with routing info
    patch: Dict[str, Any] = {
        "routing": {
            "pending_handoff": {
                "recommended_next_agent": recommended_agent,
                "reason": reason,
            },
            "delegator_debug": {
                "decision": decision,
                "action_type": action_type,
            }
        }
    }

    # Handle different action types
    if action_type == "natural_progression":
        # Keep current request active, just update stage to reflect progression
        if active_id and req:
            # ✅ Record stage transition for natural progression
            existing_history = req.get("stage_history", [])
            current_stage = req.get("status", "unknown")
            new_transition = {
                "from_stage": current_stage,
                "to_stage": recommended_agent,
                "agent": recommended_agent,
                "timestamp": datetime.utcnow(),
                "reason": f"Natural progression: {reason}"
            }

            new_status = "executing" if req.get("status") == "collecting" else req.get("status")
            patch["request_manager"] = {
                "requests": {
                    active_id: {
                        "status": new_status,
                        "stage_detail": f"progressing_to_{recommended_agent}",
                        "last_touched_at": datetime.utcnow(),
                        "stage_history": existing_history + [new_transition],
                    }
                }
            }
            _append_request_ddb_sync(patch, state, active_id, {
                "status": new_status,
                "stage_detail": f"progressing_to_{recommended_agent}",
            })

    elif action_type == "prerequisite_task":
        # Create new prerequisite request and pause current request
        if active_id and req and new_request_info:
            new_req_id = new_uuid("req")
            prereq_name = new_request_info.get("name", "Prerequisite Task")

            prereq_type = prereq_name

            # Create new request
            new_request_patch = build_request_patch(
                conversation_id=conversation_id,
                user_id=user_id,
                name=prereq_name,
                goal=new_request_info.get("goal", ""),
                target="unknown",
                set_active=True,  # Set as active
                request_id=new_req_id,
                request_source="chat",
            )

            # Pause current request and mark prerequisite relationship
            prereq_patch = build_accept_prereq_patch(
                parent_request_id=active_id,
                prereq_request_id=new_req_id,
                prereq_type=prereq_type,
            )

            # ✅ Record initial stage transition for new prerequisite request
            initial_transition = {
                "from_stage": None,
                "to_stage": "info_collection",
                "agent": "info_collection",
                "timestamp": datetime.utcnow(),
                "reason": f"Prerequisite request created: {prereq_type}"
            }

            # ✅ Record stage transition for paused parent request
            parent_history = req.get("stage_history", [])
            pause_transition = {
                "from_stage": req.get("status", "unknown"),
                "to_stage": "paused",
                "agent": "upstream_delegator",
                "timestamp": datetime.utcnow(),
                "reason": f"Paused for prerequisite request: {prereq_type}"
            }

            # Merge patches
            patch = {
                **patch,
                "request_manager": {
                    **(new_request_patch.get("request_manager") or {}),
                    **(prereq_patch.get("request_manager") or {}),
                    "requests": {
                        **((new_request_patch.get("request_manager") or {}).get("requests") or {}),
                        **((prereq_patch.get("request_manager") or {}).get("requests") or {}),
                        new_req_id: {
                            **((new_request_patch.get("request_manager") or {}).get("requests") or {}).get(new_req_id, {}),
                            "stage_history": [initial_transition],
                        },
                        active_id: {
                            **((prereq_patch.get("request_manager") or {}).get("requests") or {}).get(active_id, {}),
                            "status": "paused",
                            "stage_detail": f"paused_for_prerequisite_{new_req_id}",
                            "stage_history": parent_history + [pause_transition],
                        }
                    },
                    "pending_queue": prereq_patch.get("request_manager", {}).get("pending_queue", []),
                    "active_request_id": new_req_id,
                },
                "ddb_writes": new_request_patch.get("ddb_writes", []),
            }
            _append_request_ddb_sync(patch, state, active_id, {
                "status": "paused",
                "stage_detail": f"paused_for_prerequisite_{new_req_id}",
            })

    elif action_type == "new_unrelated_task":
        # Create new request and pause current request
        if new_request_info:
            new_req_id = new_uuid("req")

            # Infer request_type and subject_entity_id from the request info
            from prompts import _infer_request_type, _infer_subject_entity
            from fact_store import get_fact_store as _get_fs
            _req_name = new_request_info.get("name", "New Task")
            _req_goal = new_request_info.get("goal", "")
            _inferred_type = _infer_request_type(_req_name, _req_goal)
            # Fetch known entities from DDB so the inference can match
            # entities like "uncle" that aren't in the hardcoded keyword list
            try:
                _known_eids = await _get_fs().get_user_entities(user_id)
            except Exception:
                _known_eids = []
            _inferred_entity = await _infer_subject_entity(
                user_text, client=client, known_entity_ids=_known_eids,
            )

            # Create new request
            new_request_patch = build_request_patch(
                conversation_id=conversation_id,
                user_id=user_id,
                name=_req_name,
                goal=_req_goal,
                target="unknown",
                set_active=True,  # Set as active
                request_id=new_req_id,
                request_source="chat",
                request_type=_inferred_type,
                subject_entity_id=_inferred_entity,
                title=_req_name,
            )

            # ✅ Record initial stage transition for new request
            initial_transition = {
                "from_stage": None,
                "to_stage": "info_collection",
                "agent": "info_collection",
                "timestamp": datetime.utcnow(),
                "reason": f"New unrelated task created: {new_request_info.get('name', 'New Task')}"
            }

            # Merge patches
            patch = {
                **patch,
                **new_request_patch,
            }

            # Ensure request_manager exists
            if "request_manager" not in patch:
                patch["request_manager"] = {}
            if "requests" not in patch["request_manager"]:
                patch["request_manager"]["requests"] = {}

            # Add stage_history to new request
            patch["request_manager"]["requests"][new_req_id] = {
                **patch["request_manager"]["requests"].get(new_req_id, {}),
                "stage_history": [initial_transition],
            }

            # If there's an active request, pause it
            if active_id and req and req.get("status") not in ["completed", "aborted"]:
                # ✅ Record stage transition for paused request
                parent_history = req.get("stage_history", [])
                pause_transition = {
                    "from_stage": req.get("status", "unknown"),
                    "to_stage": "paused",
                    "agent": "upstream_delegator",
                    "timestamp": datetime.utcnow(),
                    "reason": "Paused due to new unrelated task"
                }

                patch["request_manager"]["pending_queue"] = [{
                    "request_id": active_id,
                    "name": req.get("name", ""),
                    "status": "pending",
                    "reason_queued": "Paused due to new unrelated task",
                    "queued_at": datetime.utcnow(),
                }]
                patch["request_manager"]["requests"][active_id] = {
                    **(patch["request_manager"]["requests"].get(active_id) or {}),
                    "status": "paused",
                    "stage_detail": "paused_by_new_task",
                    "last_touched_at": datetime.utcnow(),
                    "stage_history": parent_history + [pause_transition],
                }
                _append_request_ddb_sync(patch, state, active_id, {
                    "status": "paused",
                    "stage_detail": "paused_by_new_task",
                })

    elif action_type == "resume_existing":
        # Resume an existing request instead of creating a new one
        resume_id = decision.get("resume_request_id")
        all_requests = (rm.get("requests") or {})
        resume_req = all_requests.get(resume_id) if resume_id else None

        if resume_id and resume_req:
            resume_status = resume_req.get("status", "unknown")
            resume_history = resume_req.get("stage_history", [])
            resume_transition = {
                "from_stage": resume_status,
                "to_stage": recommended_agent,
                "agent": "upstream_delegator",
                "timestamp": datetime.utcnow(),
                "reason": f"Resumed by user request: {reason}"
            }

            patch["request_manager"] = {
                "active_request_id": resume_id,
                "requests": {
                    resume_id: {
                        "status": "collecting" if resume_status in ("paused", "executed", "completed") else resume_status,
                        "stage_detail": f"resumed_by_user",
                        "awaiting_user_input": True,
                        "last_touched_at": datetime.utcnow(),
                        "stage_history": resume_history + [resume_transition],
                    }
                },
                "pending_queue_remove": [resume_id],
            }

            # Pause the current active request if it's still active
            if active_id and active_id != resume_id and req and req.get("status") not in ("completed", "aborted", "executed"):
                parent_history = req.get("stage_history", [])
                pause_transition = {
                    "from_stage": req.get("status", "unknown"),
                    "to_stage": "paused",
                    "agent": "upstream_delegator",
                    "timestamp": datetime.utcnow(),
                    "reason": f"Paused: user switched to existing request {resume_id}"
                }
                patch["request_manager"]["requests"][active_id] = {
                    "status": "paused",
                    "stage_detail": "paused_by_resume_switch",
                    "last_touched_at": datetime.utcnow(),
                    "stage_history": parent_history + [pause_transition],
                }
                patch["request_manager"]["pending_queue"] = [{
                    "request_id": active_id,
                    "name": req.get("name", ""),
                    "status": "pending",
                    "reason_queued": f"Paused: user switched to existing request",
                    "queued_at": datetime.utcnow(),
                }]

    elif action_type == "task_acknowledged":
        # User acknowledged results — mark as executed, send confirmation, route to END
        if active_id and req:
            req_name = req.get("name", "task")
            existing_history = req.get("stage_history", [])
            current_stage = req.get("status", "unknown")
            executed_transition = {
                "from_stage": current_stage,
                "to_stage": "executed",
                "agent": "upstream_delegator",
                "timestamp": datetime.utcnow(),
                "reason": f"Task acknowledged by user: {reason}"
            }
            patch["request_manager"] = {
                "requests": {
                    active_id: {
                        "status": "executed",
                        "stage_detail": "executed_task_acknowledged",
                        "awaiting_user_input": False,
                        "last_touched_at": datetime.utcnow(),
                        "stage_history": existing_history + [executed_transition],
                    }
                }
            }
            patch["routing"]["pending_handoff"] = {
                "recommended_next_agent": None,
                "reason": "Task acknowledged, no further routing needed",
            }
            lang = _detect_user_language(state)
            ack_content = (
                f"好的，{req_name}已经处理完了。还有什么我可以帮你的吗？"
                if lang == "zh" else
                f"Got it, {req_name} is all done. Is there anything else I can help with?"
            )
            patch["messages"] = [{
                "role": "assistant",
                "content": ack_content,
                "metadata": {"agent": "upstream_delegator"}
            }]
        return patch

    elif action_type == "same_agent_continue":
        # No request management changes needed
        pass

    # Handle "mark_executed" action — task is done, results delivered
    if current_request_action == "mark_executed" and active_id and req:
        if "request_manager" not in patch:
            patch["request_manager"] = {}
        if "requests" not in patch["request_manager"]:
            patch["request_manager"]["requests"] = {}

        existing_history = req.get("stage_history", [])
        current_stage = req.get("status", "unknown")
        executed_transition = {
            "from_stage": current_stage,
            "to_stage": "executed",
            "agent": "upstream_delegator",
            "timestamp": datetime.utcnow(),
            "reason": f"Task fulfilled — marked executed by upstream_delegator: {reason}"
        }

        patch["request_manager"]["requests"][active_id] = {
            **(patch["request_manager"]["requests"].get(active_id) or {}),
            "status": "executed",
            "stage_detail": "executed_by_delegator",
            "awaiting_user_input": False,
            "last_touched_at": datetime.utcnow(),
            "stage_history": existing_history + [executed_transition],
        }
        _append_request_ddb_sync(patch, state, active_id, {
            "status": "executed",
            "stage_detail": "executed_by_delegator",
        })

    # Handle explicit "complete" action
    if current_request_action == "complete" and active_id and req:
        if "request_manager" not in patch:
            patch["request_manager"] = {}
        if "requests" not in patch["request_manager"]:
            patch["request_manager"]["requests"] = {}

        # ✅ Record stage transition for completion
        existing_history = req.get("stage_history", [])
        current_stage = req.get("status", "unknown")
        completion_transition = {
            "from_stage": current_stage,
            "to_stage": "completed",
            "agent": "upstream_delegator",
            "timestamp": datetime.utcnow(),
            "reason": "User confirmed completion"
        }

        patch["request_manager"]["requests"][active_id] = {
            **(patch["request_manager"]["requests"].get(active_id) or {}),
            "status": "completed",
            "stage_detail": "completed_by_user_confirmation",
            "last_touched_at": datetime.utcnow(),
            "stage_history": existing_history + [completion_transition],
        }
        _append_request_ddb_sync(patch, state, active_id, {
            "status": "completed",
            "stage_detail": "completed_by_user_confirmation",
        })

    return patch

def route_from_delegator(state: Dict[str, Any]) -> RouteKey:
    pending = ((state.get("routing") or {}).get("pending_handoff") or {})
    nxt = pending.get("recommended_next_agent")

    # GUARD: for new / newly-created requests that haven't gone through
    # info_collection yet, route to info_collection first so the Q&A loop
    # can start. Skip this for agents that handle requests directly:
    # - front_end: emotional support (no info gathering needed)
    # - quick_answer: standalone factual Q&A
    # - user_info: retrieves and summarizes stored facts/memory
    req = _get_active_request(state) or {}
    info_state = req.get("info_collection_state") or {}
    if nxt and nxt not in ("front_end_emotional_support", "quick_answer", "user_info", "human_comm"):
        if not info_state.get("key_info_needed") and req.get("status") in (None, "created", "collecting"):
            return "info_collection"

    if nxt == "front_end_emotional_support":
        return "front_end"
    if nxt == "quick_answer":
        return "quick_answer"
    if nxt == "info_collection":
        return "info_collection"
    if nxt == "deep_search":
        return "deep_search"
    if nxt == "user_info":
        return "user_info"
    if nxt == "domain_expert":
        return "domain_expert"
    if nxt == "human_comm":
        return "human_comm"
    return "respond"

async def info_collection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Conversational information collection agent.

    Key changes from old version:
    - NO structured slot extraction/parsing
    - LLM generates natural language summaries (bullet points)
    - LLM assesses readiness to proceed
    - Offers user the option to proceed when info is sufficient but incomplete
    """
    meta = state.get("meta") or {}
    conversation_id = meta.get("conversation_id", "conv-unknown")
    user_id = meta.get("user_id", "user-unknown")
    user_text = last_user_text(state)
    known_facts = _extract_known_facts(state)

    # Enrich known_facts with memory context from Fact Store
    memory_block, profile_facts_dict = await _build_memory_context_with_facts(state)
    if memory_block:
        known_facts["_memory_context"] = memory_block

    # Initialize TrackedAnthropicClient
    client = TrackedAnthropicClient(
        session_id=conversation_id,
        agent_role="info_collection",
        user_id=user_id,
    )

    active_id = _get_active_request_id(state)
    gate = _get_prereq_gate(state)

    # ── Pending fact confirmations check (via MCP) ──
    # If there are high-risk facts pending confirmation, check if user responded
    try:
        req_for_confirm = _get_active_request(state) or {}
        entity_for_confirm = req_for_confirm.get("subject_entity_id") or ""
        if user_id and entity_for_confirm:
            pending = await call_memory_tool(
                mgr=MCP_MGR, server=MEMORY_SERVER,
                tool_name="memory_get_pending_confirmations",
                arguments={"user_id": user_id, "entity_id": entity_for_confirm},
                timeout_ms=3000,
            )
            if isinstance(pending, list) and pending:
                # Check if user's message confirms or denies a pending fact
                yn = _user_yes_no(user_text)
                if yn is not None and len(pending) > 0:
                    top_pending = pending[0]
                    await call_memory_tool(
                        mgr=MCP_MGR, server=MEMORY_SERVER,
                        tool_name="memory_confirm_fact",
                        arguments={
                            "user_id": user_id,
                            "event_id": top_pending.get("event_id", ""),
                            "confirmed": yn,
                        },
                        timeout_ms=3000,
                    )
    except Exception as e:
        _logger.debug(f"Pending confirmations check failed (non-fatal): {e}")

    # CASE 0: Prerequisite proposal pending (consent-gated)
    # This is when the system (not user) detected a prerequisite and needs consent
    if active_id and gate.get("status") == "proposed":
        yn = _user_yes_no(user_text)
        parent_id = gate.get("parent_request_id") or active_id
        prereq_id = gate.get("proposed_request_id")
        prereq_type = gate.get("prereq_type") or "other"

        if yn is True and prereq_id:
            accept = build_accept_prereq_patch(
                parent_request_id=parent_id,
                prereq_request_id=prereq_id,
                prereq_type=prereq_type,
            )

            # ✅ Record stage transition for accepting prerequisite
            prereq_req = (state.get("request_manager") or {}).get("requests", {}).get(prereq_id) or {}
            prereq_history = prereq_req.get("stage_history", [])
            accept_transition = {
                "from_stage": "proposed",
                "to_stage": "info_collection",
                "agent": "info_collection",
                "timestamp": datetime.utcnow(),
                "reason": f"User accepted prerequisite: {prereq_type}"
            }

            return {
                "routing": {"current_agent": "info_collection", "conversation_stage": "collecting"},
                **accept,
                "request_manager": {
                    **(accept.get("request_manager") or {}),
                    "requests": {
                        **((accept.get("request_manager") or {}).get("requests") or {}),
                        prereq_id: {
                            **((accept.get("request_manager") or {}).get("requests") or {}).get(prereq_id, {}),
                            "stage_history": prereq_history + [accept_transition],
                        }
                    }
                },
                "messages": [{
                    "role": "assistant",
                    "content": _msg(
                        state,
                        f"OK，我们先把先决条件（{prereq_type}）搞定。为了开始，我需要你先补充一点基本信息。",
                        f"OK, let's handle the prerequisite ({prereq_type}) first. To get started, I'll need a bit of basic information from you.",
                    ),
                    "metadata": {"agent": "info_collection"}
                }],
            }

        if yn is False:
            return {
                "routing": {"current_agent": "info_collection", "conversation_stage": "collecting"},
                "request_manager": {
                    "requests": {
                        active_id: {"prereq_gate": {**gate, "status": "rejected", "resolved_at": datetime.utcnow()}}
                    }
                },
                "messages": [{
                    "role": "assistant",
                    "content": _msg(
                        state,
                        "好，那我们先不处理这个先决条件。我会基于现有信息尽量往下推进。",
                        "OK, let's skip this prerequisite for now. I'll do my best to move forward with what we have.",
                    ),
                    "metadata": {"agent": "info_collection"}
                }],
            }

        return {
            "routing": {"current_agent": "info_collection", "conversation_stage": "collecting"},
            "messages": [{
                "role": "assistant",
                "content": _msg(
                    state,
                    f'确认一下：要不要先处理先决条件（{prereq_type}）再继续？回复"要/不要"。',
                    f'Just to confirm: would you like to handle the prerequisite ({prereq_type}) before continuing? Reply "yes" or "no".',
                ),
                "metadata": {"agent": "info_collection"}
            }],
        }

    # CASE 1: New request - Generate collection plan and ask initial questions
    # Also enter CASE 1 if request exists but info_collection_state hasn't been
    # initialized yet (e.g., upstream_delegator created the request this turn).
    _active_req = _get_active_request(state) or {}
    _ic_state = _active_req.get("info_collection_state") or {}
    _needs_plan = not active_id or (active_id and not _ic_state.get("key_info_needed"))
    if _needs_plan:
        # Check in-session requests for a resumable duplicate before creating new
        all_reqs = (state.get("request_manager") or {}).get("requests") or {}
        for existing_rid, existing_req in all_reqs.items():
            existing_status = existing_req.get("status", "")
            if existing_status in ("paused", "collecting", "validated", "executed"):
                existing_goal = (existing_req.get("goal") or "").lower()
                existing_name = (existing_req.get("name") or "").lower()
                user_lower = user_text.lower()
                # Simple keyword overlap check — if user text overlaps with existing request goal/name
                if existing_goal and (
                    any(word in user_lower for word in existing_goal.split() if len(word) > 2)
                    or any(word in user_lower for word in existing_name.split() if len(word) > 2)
                ):
                    # Resume this existing request instead of creating a new one
                    resume_history = existing_req.get("stage_history", [])
                    resume_transition = {
                        "from_stage": existing_status,
                        "to_stage": "info_collection",
                        "agent": "info_collection",
                        "timestamp": datetime.utcnow(),
                        "reason": f"Resumed in-session duplicate: user said '{user_text[:80]}'"
                    }
                    collected_summary = (existing_req.get("info_collection_state") or {}).get("summary_of_collected_info", "")
                    lang = _detect_user_language(state)
                    req_label = existing_req.get('name', 'task')
                    if lang == "zh":
                        resume_msg = f"我找到了之前的请求「{req_label}」，我们继续吧。"
                        if collected_summary:
                            resume_msg += f"\n之前已经收集到的信息：{collected_summary[:200]}"
                        resume_msg += "\n还有什么需要补充的吗？"
                    else:
                        resume_msg = f"I found your previous request \"{req_label}\". Let's continue."
                        if collected_summary:
                            resume_msg += f"\nInformation collected so far: {collected_summary[:200]}"
                        resume_msg += "\nAnything else to add?"

                    resume_patch = {
                        "routing": {"current_agent": "info_collection", "conversation_stage": "collecting"},
                        "request_manager": {
                            "active_request_id": existing_rid,
                            "requests": {
                                existing_rid: {
                                    "status": "collecting",
                                    "stage_detail": "resumed_in_session_duplicate",
                                    "awaiting_user_input": True,
                                    "last_touched_at": datetime.utcnow(),
                                    "stage_history": resume_history + [resume_transition],
                                }
                            },
                            "pending_queue_remove": [existing_rid],
                        },
                        "messages": [{
                            "role": "assistant",
                            "content": resume_msg,
                            "metadata": {"agent": "info_collection"}
                        }],
                    }
                    _append_request_ddb_sync(resume_patch, state, existing_rid, {
                        "status": "collecting",
                        "stage_detail": "resumed_in_session_duplicate",
                    })
                    return resume_patch

        # Use existing request ID if delegator already created one, else create new
        rid = active_id or new_uuid("req")

        # ── Resolve entity BEFORE plan generation ──
        # Query DDB for known entities, match against user's message +
        # the active request context (name/goal from delegator), and fetch
        # the right facts so the plan LLM has full context.
        resolved_entity_id = ""
        if user_id and not profile_facts_dict:
            try:
                from fact_store import get_fact_store
                _fs = get_fact_store()
                known_entities = await _fs.get_user_entities(user_id)
                if known_entities:
                    _logger.info(f"Known entities for user: {known_entities}")

                    # Build search text from: user message + active request name/goal
                    # This handles pronoun cases like "he needs X" where the
                    # delegator's request goal says "for the user's uncle".
                    _active_for_entity = _get_active_request(state) or {}
                    search_text = " ".join([
                        user_text,
                        _active_for_entity.get("name", ""),
                        _active_for_entity.get("goal", ""),
                    ]).lower()

                    for eid in known_entities:
                        label = eid.split(":")[-1] if ":" in eid else eid
                        if label.lower() in search_text:
                            _eid_bundle = await get_context_bundle(
                                user_id=user_id,
                                request_dict={"subject_entity_id": eid},
                                message=user_text,
                                k=5,
                            )
                            if _eid_bundle.profile_facts:
                                resolved_entity_id = eid
                                profile_facts_dict = _eid_bundle.profile_facts
                                # Replace known_facts memory context with
                                # this entity's facts ONLY (no cross-contamination)
                                block = bundle_to_prompt_block(_eid_bundle)
                                if block and block != "(No memory context available.)":
                                    known_facts["_memory_context"] = (
                                        f"\n\n## Memory Context (from User Memory Framework)\n{block}"
                                    )
                                _logger.info(
                                    f"Pre-plan entity resolution: '{eid}' "
                                    f"({len(profile_facts_dict)} facts)"
                                )
                                break
            except Exception as e:
                _logger.warning(f"Pre-plan entity resolution failed (non-blocking): {e}")

        # Fetch similar requests for context
        similar = await fetch_similar_requests(user_id=user_id, request_text=user_text, top_k=5)

        # Convert HistoricalRequestRecord objects to dicts for LLM
        similar_dicts = [s.model_dump() for s in similar] if similar else []

        # LLM generates collection plan (now with correct entity facts in known_facts)
        plan = await llm_collection_plan(
            user_request=user_text,
            known_facts=known_facts,
            similar_requests=similar_dicts,
            client=client,
        )

        # Override plan's entity_id if we already resolved it
        if resolved_entity_id:
            plan["subject_entity_id"] = resolved_entity_id

        # Create request (or reuse if delegator already created one)
        if active_id:
            # Delegator already created the request — just build a minimal patch
            create = {
                "request_manager": {
                    "active_request_id": active_id,
                    "requests": {
                        active_id: {
                            "request_type": plan.get("request_type", ""),
                            "subject_entity_id": plan.get("subject_entity_id", ""),
                            "routing_hint": plan.get("routing_hint", "deep_search"),
                        }
                    }
                }
            }
        else:
            create = build_request_patch(
                conversation_id=conversation_id,
                user_id=user_id,
                name=plan.get("request_name", "General Request"),
                goal=plan.get("request_goal", user_text[:200]),
                target="unknown",
                set_active=True,
                request_id=rid,
                request_source="chat",
                request_type=plan.get("request_type", ""),
                subject_entity_id=plan.get("subject_entity_id", ""),
                title=plan.get("request_name", "General Request"),
            )

        key_info_needed = plan.get("key_info_needed", [])
        nice_to_have = plan.get("nice_to_have_info", [])

        # ── Summarize known facts + generate contextual questions ──
        lang = _detect_user_language(state)
        questions_text = None

        if profile_facts_dict:
            try:
                from prompts import select_relevant_facts, llm_summarize_and_ask
                relevant_facts = select_relevant_facts(
                    profile_facts=profile_facts_dict,
                    request_type=plan.get("request_type", ""),
                    plan_questions=key_info_needed,
                )
                _logger.info(
                    f"Selected {len(relevant_facts)}/{len(profile_facts_dict)} "
                    f"relevant facts for summarize-and-ask"
                )
                if relevant_facts:
                    sa_result = await llm_summarize_and_ask(
                        user_request=user_text,
                        request_goal=plan.get("request_goal", ""),
                        profile_facts=relevant_facts,
                        plan_questions=key_info_needed,
                        nice_to_have=nice_to_have,
                        client=client,
                        lang=lang,
                    )
                    if sa_result and sa_result.get("combined_message"):
                        questions_text = sa_result["combined_message"]
                        # Use LLM's follow-up questions as key_info_needed
                        llm_questions = sa_result.get("follow_up_questions", [])
                        if llm_questions:
                            key_info_needed = llm_questions
            except Exception as e:
                _logger.warning(f"Summarize-and-ask failed (non-blocking): {e}")

        # Fallback: format questions without fact summary
        if not questions_text:
            if lang == "zh":
                questions_text = "好，我理解你的请求了。为了把事情办成，我想了解几个关键点：\n"
                questions_text += "\n".join([f"{i+1}. {q}" for i, q in enumerate(key_info_needed)])
                if nice_to_have:
                    questions_text += "\n\n（如果方便的话也可以告诉我：" + "；".join(nice_to_have) + "）"
            else:
                questions_text = "Got it, I understand your request. To help you effectively, I'd like to know a few key things:\n"
                questions_text += "\n".join([f"{i+1}. {q}" for i, q in enumerate(key_info_needed)])
                if nice_to_have:
                    questions_text += "\n\n(If convenient, you can also tell me: " + "; ".join(nice_to_have) + ")"

        # Initialize info_collection_state
        info_collection_state = {
            "summary_of_collected_info": "",
            "key_info_needed": key_info_needed,
            "nice_to_have_info": nice_to_have,
            "conversation_turns_with_agent": 1,
            "last_summary_at": datetime.utcnow(),
            "readiness_to_proceed": "needs_more",
        }

        # ✅ Record initial stage transition
        initial_transition = {
            "from_stage": None,
            "to_stage": "info_collection",
            "agent": "info_collection",
            "timestamp": datetime.utcnow(),
            "reason": "New request created from user query"
        }

        # Merge create's request_manager with additional fields (don't overwrite)
        create_rm = create.get("request_manager", {})
        create_req = (create_rm.get("requests") or {}).get(rid, {})
        merged_req = {
            **create_req,
            "awaiting_user_input": True,
            "routing_hint": plan.get("routing_hint", "deep_search"),
            "info_collection_state": info_collection_state,
            "status": "collecting",
            "stage_history": [initial_transition],
        }

        new_req_patch = {
            "routing": {"current_agent": "info_collection", "conversation_stage": "collecting"},
            **create,
            "request_manager": {
                **create_rm,
                "requests": {
                    rid: merged_req,
                }
            },
            "messages": [{
                "role": "assistant",
                "content": questions_text,
                "metadata": {"agent": "info_collection"}
            }],
            "info_collection_debug": {
                "plan": plan,
                "profile_facts_retrieved": profile_facts_dict,
                "fact_pre_filter": None,
            },
        }
        # Sync collecting status and info_collection_state to DDB
        _append_request_ddb_sync(new_req_patch, state, rid, {
            "status": "collecting",
            "stage_detail": "info_collection_started",
            "info_collection_state": info_collection_state,
        })
        return new_req_patch

    # CASE 2 & 3: User is providing information or responding
    # Unified handling with LLM summarization
    req = _get_active_request(state) or {}
    info_state = req.get("info_collection_state") or {}

    # Extract info from info_collection_state
    key_info_needed = info_state.get("key_info_needed", [])
    nice_to_have = info_state.get("nice_to_have_info", [])
    previous_summary = info_state.get("summary_of_collected_info", "")
    conversation_turns = info_state.get("conversation_turns_with_agent", 0)

    # Get conversation history
    messages = state.get("messages", []) or []

    # LLM summarizes user's response and assesses readiness
    last_asked_questions = info_state.get("last_asked_questions") or []
    summary_result = await llm_info_collection_summarize(
        request_goal=req.get("goal", ""),
        key_info_needed=key_info_needed,
        nice_to_have_info=nice_to_have,
        previous_summary=previous_summary,
        user_latest_reply=user_text,
        conversation_history=messages,
        known_facts=known_facts,
        client=client,
        last_asked_questions=last_asked_questions,
    )

    updated_summary = summary_result.get("updated_summary", "")
    readiness = summary_result.get("readiness_to_proceed", "needs_more")
    suggested_response = summary_result.get("suggested_response", "")
    detected_prerequisites = summary_result.get("detected_prerequisites", [])
    user_signals = summary_result.get("user_signals") or {}

    # ── Track disputed facts ──
    disputed_keys = list(info_state.get("disputed_fact_keys", []))
    disputed_facts_from_llm = summary_result.get("disputed_facts") or []
    new_disputes = []        # keys to deprecate in DDB
    new_dispute_only = []    # keys with no correction (need follow-up)
    for df in disputed_facts_from_llm:
        dk = df.get("fact_key", "")
        if not dk:
            continue
        has_correction = df.get("has_correction", False)
        # Always deprecate the old value in DDB
        if dk not in new_disputes:
            new_disputes.append(dk)
        # Only mark as "disputed" (blocking re-match) if no correction provided
        if not has_correction and dk not in disputed_keys:
            disputed_keys.append(dk)
            new_dispute_only.append(dk)

    # ── Deprecate old facts in DDB (both corrected and disputed) ──
    if new_disputes and user_id:
        try:
            from fact_store import get_fact_store
            _fs = get_fact_store()
            entity_id = req.get("subject_entity_id") or ""
            if entity_id:
                stale_facts = await _fs.get_active_facts(
                    user_id=user_id,
                    entity_id=entity_id,
                    fact_keys=new_disputes,
                )
                for sf in stale_facts:
                    await _fs.deprecate_fact(sf)
                    _logger.info(f"Deprecated disputed fact: {sf.fact_key}={sf.fact_value}")
        except Exception as e:
            _logger.warning(f"Failed to deprecate disputed facts (non-blocking): {e}")

    # ── Pre-filter follow-up questions against stored facts ──
    follow_ups = summary_result.get("missing_or_unclear", [])
    if follow_ups and profile_facts_dict:
        try:
            filter_result = await filter_questions_with_known_facts(
                questions=follow_ups,
                nice_to_have=[],
                profile_facts=profile_facts_dict,
                request_type=req.get("request_type", ""),
                entity_id=req.get("subject_entity_id", ""),
                client=client,
                lang=_detect_user_language(state),
                excluded_keys=disputed_keys,
            )
            if filter_result["pre_answered"]:
                _logger.info(
                    f"Pre-answered {len(filter_result['pre_answered'])} follow-up questions "
                    f"from stored facts"
                )
                summary_result["missing_or_unclear"] = filter_result["remaining_questions"]
                # Append confirmation of known facts to suggested response
                if filter_result["confirmation_text"]:
                    suggested_response = (
                        suggested_response + "\n\n" + filter_result["confirmation_text"]
                    )
                # If all follow-ups answered, upgrade readiness
                if (
                    not filter_result["remaining_questions"]
                    and readiness == "needs_more"
                ):
                    readiness = "can_proceed_but_incomplete"
        except Exception as e:
            _logger.warning(f"Fact pre-filter (follow-up) failed (non-blocking): {e}")

    # ── Fact extraction + binding (non-blocking on failure) ──
    slot_refs = {}
    try:
        from slot_fact_binder import bind_facts_from_summary
        slot_refs = await bind_facts_from_summary(
            user_id=user_id,
            request_id=active_id,
            entity_id=req.get("subject_entity_id") or "care_recipient:unknown",
            request_type=req.get("request_type") or "",
            updated_summary=updated_summary,
            key_info_needed=key_info_needed,
            conversation_history=messages,
            client=client,
            existing_slot_refs=req.get("slot_refs", {}),
            existing_profile_facts=profile_facts_dict,
            disputed_fact_keys=disputed_keys,
        )
    except Exception as e:
        _logger.warning(f"Fact binding failed (non-blocking): {e}")

    # LLM-driven override: if LLM detected user wants to proceed, trust it
    if user_signals.get("wants_to_proceed_with_existing_info") is True:
        readiness = "ready"

    # LLM-driven consent: if user agrees to handle prerequisite
    # Look for consent in current LLM signals + stored prerequisites from previous turn
    stored_prerequisites = info_state.get("detected_prerequisites") or []
    all_prerequisites = detected_prerequisites or stored_prerequisites
    consent = user_signals.get("consent_response", "none")

    if all_prerequisites and consent == "yes":
        # User agreed to handle prerequisite - create new request and pause current
        prereq = all_prerequisites[0]  # Take first prerequisite
        prereq_type_raw = prereq.get("type", "other")
        prereq_reason = prereq.get("reason", "")
        
        prereq_type = prereq_type_raw
        
        new_req_id = new_uuid("req")
        
        # Create new prerequisite request
        new_request_patch = build_request_patch(
            conversation_id=conversation_id,
            user_id=user_id,
            name=prereq_type,
            goal=prereq_reason,
            target="unknown",
            set_active=True,
            request_id=new_req_id,
            request_source="chat",
        )
        
        # Pause current request
        prereq_patch = build_accept_prereq_patch(
            parent_request_id=active_id,
            prereq_request_id=new_req_id,
            prereq_type=prereq_type,
        )
        
        # Record stage transitions
        initial_transition = {
            "from_stage": None,
            "to_stage": "info_collection",
            "agent": "info_collection",
            "timestamp": datetime.utcnow(),
            "reason": f"Prerequisite request created: {prereq_type}"
        }
        
        parent_history = req.get("stage_history", [])
        pause_transition = {
            "from_stage": req.get("status", "unknown"),
            "to_stage": "paused",
            "agent": "info_collection",
            "timestamp": datetime.utcnow(),
            "reason": f"Paused for prerequisite request: {prereq_type}"
        }
        
        # Generate natural confirmation message using LLM
        confirmation_msg = await llm_prerequisite_acceptance_response(
            prereq_type=prereq_type,
            prereq_reason=prereq_reason,
            user_message=user_text,
            conversation_history=messages,
            client=client,
        )
        
        # Return patch that creates prerequisite request and pauses parent
        return {
            "routing": {
                "current_agent": "info_collection",
                "conversation_stage": "collecting",
                "pending_handoff": {
                    "recommended_next_agent": "info_collection",
                    "reason": f"Starting prerequisite: {prereq_type}",
                },
            },
            "request_manager": {
                **(new_request_patch.get("request_manager") or {}),
                **(prereq_patch.get("request_manager") or {}),
                "requests": {
                    **((new_request_patch.get("request_manager") or {}).get("requests") or {}),
                    **((prereq_patch.get("request_manager") or {}).get("requests") or {}),
                    new_req_id: {
                        **((new_request_patch.get("request_manager") or {}).get("requests") or {}).get(new_req_id, {}),
                        **((prereq_patch.get("request_manager") or {}).get("requests") or {}).get(new_req_id, {}),
                        "stage_history": [initial_transition],
                    },
                    active_id: {
                        **((prereq_patch.get("request_manager") or {}).get("requests") or {}).get(active_id, {}),
                        "status": "paused",
                        "stage_detail": f"paused_for_prerequisite_{new_req_id}",
                        "stage_history": parent_history + [pause_transition],
                    }
                },
                "pending_queue": prereq_patch.get("request_manager", {}).get("pending_queue", []),
                "active_request_id": new_req_id,
            },
            "ddb_writes": new_request_patch.get("ddb_writes", []),
            "messages": [{
                "role": "assistant",
                "content": confirmation_msg,
                "metadata": {"agent": "info_collection"}
            }],
        }

    # Update info_collection_state
    missing_or_unclear = summary_result.get("missing_or_unclear", [])
    updated_info_state = {
        **info_state,
        "summary_of_collected_info": updated_summary,
        "conversation_turns_with_agent": conversation_turns + 1,
        "last_summary_at": datetime.utcnow(),
        "readiness_to_proceed": readiness,
        "key_info_status": summary_result.get("key_info_status", []),
        "detected_prerequisites": detected_prerequisites,
        "user_signals": user_signals,
        "last_asked_questions": missing_or_unclear,
        "disputed_fact_keys": disputed_keys,
    }

    patch: Dict[str, Any] = {
        "routing": {"current_agent": "info_collection", "conversation_stage": "collecting"},
        "request_manager": {
            "requests": {
                active_id: {
                    "info_collection_state": updated_info_state,
                    "last_touched_at": datetime.utcnow(),
                    "slot_refs": {**req.get("slot_refs", {}), **slot_refs},
                }
            }
        },
        "messages": [{
            "role": "assistant",
            "content": suggested_response,
            "metadata": {"agent": "info_collection"}
        }],
        "info_collection_debug": {
            "summary_result": summary_result,
            "readiness": readiness,
            "detected_prerequisites": detected_prerequisites,
            "profile_facts_retrieved": profile_facts_dict,
            "disputed_fact_keys": disputed_keys,
        }
    }

    # If ready to proceed, hand off to next agent
    if readiness == "ready":
        # Prefer turn_router's LLM recommendation (most recent context) over
        # the original plan's routing_hint (which may be stale if the user
        # changed what they want during info collection).
        turn_router_rec = (state.get("routing") or {}).get("llm_recommended_agent")
        existing_handoff = ((state.get("routing") or {}).get("pending_handoff") or {})
        upstream_rec = existing_handoff.get("recommended_next_agent")

        if turn_router_rec and turn_router_rec not in ("info_collection", "quick_answer", None):
            routing_hint = turn_router_rec
        elif upstream_rec and upstream_rec not in ("info_collection", None):
            routing_hint = upstream_rec
        else:
            routing_hint = req.get("routing_hint") or "deep_search"

        patch["request_manager"]["requests"][active_id]["awaiting_user_input"] = False
        patch["request_manager"]["requests"][active_id]["status"] = "validated"
        patch["routing"]["pending_handoff"] = {
            "recommended_next_agent": routing_hint,
            "reason": "Information collection complete; proceeding to execution"
        }

        # ✅ Record stage transition
        existing_history = req.get("stage_history", [])
        new_transition = {
            "from_stage": "info_collection",
            "to_stage": routing_hint,
            "agent": routing_hint,
            "timestamp": datetime.utcnow(),
            "reason": "Information collection complete; proceeding to execution"
        }
        patch["request_manager"]["requests"][active_id]["stage_history"] = existing_history + [new_transition]

        # Sync validated status and final info_collection_state to DDB
        _append_request_ddb_sync(patch, state, active_id, {
            "status": "validated",
            "stage_detail": f"info_collection_complete_routing_to_{routing_hint}",
            "info_collection_state": updated_info_state,
        })
    else:
        # Still collecting
        patch["request_manager"]["requests"][active_id]["awaiting_user_input"] = True
        # Sync updated info_collection_state to DDB mid-Q&A
        _append_request_ddb_sync(patch, state, active_id, {
            "info_collection_state": updated_info_state,
        })

    return patch


# ---------------------------------------------------------------------------
# MCP configuration — demo vs full mode
# ---------------------------------------------------------------------------
import json as _json
import logging as _logging

_deep_search_logger = _logging.getLogger("deep_search")
_deep_search_logger.setLevel(_logging.DEBUG)
_deep_search_logger.propagate = False
if not _deep_search_logger.handlers:
    _dsh = _logging.StreamHandler()
    _dsh.setFormatter(_logging.Formatter("%(name)s %(levelname)s: %(message)s"))
    _deep_search_logger.addHandler(_dsh)

DEEP_SEARCH_MODE = os.environ.get("DEEP_SEARCH_MODE", "full")  # "demo" or "full"

MCP_MGR = MCPClientManager()

_MCP_URL = os.getenv("MCP_SERVER_URL")  # single URL for all remote tools

DEMO_SERVER = MCPServerConfig(
    name="demo",
    transport="stdio",
    command="python",
    args=["-u", "servers/demo_mcp_server.py"],
    keep_alive=True,
)

SEARCH_SERVER = MCPServerConfig(
    name="search",
    transport="http" if _MCP_URL else "stdio",
    url=_MCP_URL if _MCP_URL else None,
    command="python" if not _MCP_URL else None,
    args=["-u", "servers/search_mcp_server.py"] if not _MCP_URL else None,
    keep_alive=True,
)

FOLLOW_UP_SERVER = MCPServerConfig(
    name="follow_up",
    transport="http" if _MCP_URL else "stdio",
    url=_MCP_URL if _MCP_URL else None,
    command="python" if not _MCP_URL else None,
    args=["-u", "servers/follow_up_mcp_server.py"] if not _MCP_URL else None,
    keep_alive=True,
)

MEMORY_SERVER = MCPServerConfig(
    name="memory",
    transport="http" if _MCP_URL else "stdio",
    url=_MCP_URL if _MCP_URL else None,
    command="python" if not _MCP_URL else None,
    args=["-u", "servers/memory_mcp_server.py"] if not _MCP_URL else None,
    keep_alive=True,
)

ESCALATION_SERVER = MCPServerConfig(
    name="escalation",
    transport="http" if _MCP_URL else "stdio",
    url=_MCP_URL if _MCP_URL else None,
    command="python" if not _MCP_URL else None,
    args=["-u", "servers/escalation_mcp_server.py"] if not _MCP_URL else None,
    keep_alive=True,
)

# Shared LLM client for deep_search / domain_expert full-mode calls
_search_llm = TrackedAnthropicClient(agent_role="deep_search")


async def deep_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rid = _get_active_request_id(state)
    req = _get_active_request(state) or {}

    # ── Demo mode: unchanged original behavior ──────────────────────────
    if DEEP_SEARCH_MODE == "demo":
        tool_patch = await call_mcp_tool_patch(
            mgr=MCP_MGR,
            server=DEMO_SERVER,
            tool_name="nearby_providers",
            arguments={"query": req.get("goal") or req.get("name") or "request", "location": "UNKNOWN"},
            purpose="Demo MCP tool call (replace with real deep search tooling)",
        )

        if tool_patch.get("tools", {}).get("tool_failures"):
            return {"routing": {"current_agent": "deep_search", "conversation_stage": "error_recovery"}, **tool_patch}

        data = (tool_patch.get("_mcp_result") or {}).get("data") or {}

        return {
            "routing": {"current_agent": "deep_search", "conversation_stage": "executing"},
            **tool_patch,
            "artifacts_append": {
                "request_id": rid,
                "artifact": {
                    "type": "search_results",
                    "artifact_id": new_uuid("art"),
                    "data": data,
                    "produced_by": "deep_search",
                },
            },
            "messages": [{
                "role": "assistant",
                "content": _msg(
                    state,
                    "我已经跑了一步（demo）工具调用，拿到了初步结果。接下来我可以根据你的约束再筛选/补全。",
                    "I've run an initial (demo) tool call and got preliminary results. I can refine or expand based on your constraints.",
                ),
                "metadata": {"agent": "deep_search"}
            }],
        }

    # ── Full mode: LLM-driven tool selection loop ───────────────────────
    #
    # NOTE: Inside the LangGraph execution, state keys like request_manager
    # and messages undergo shallow replacement per node output.  By the time
    # deep_search runs (after downstream_catcher), the original user messages
    # and full request record may have been replaced.  We therefore extract
    # context from ALL available sources, including assistant messages which
    # often carry a summary of what was collected.

    goal = req.get("goal") or req.get("name") or ""
    info_state = req.get("info_collection_state") or {}
    collected_info = info_state.get("summary_of_collected_info") or ""

    # Gather text from every message still in state (user AND assistant)
    msgs = state.get("messages") or []
    all_msg_texts = []
    for m in msgs:
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if content:
            all_msg_texts.append(content)

    # Build a rich context string from whatever is available
    context_parts = []
    if goal:
        context_parts.append(f"Request name: {goal}")
    if collected_info:
        context_parts.append(f"Collected info summary: {collected_info}")
    if all_msg_texts:
        context_parts.append(f"Conversation so far:\n" + "\n".join(all_msg_texts[-6:]))

    # Inject memory context so search knows stored facts (location, preferences, etc.)
    memory_block = await _build_memory_context_block(state)
    if memory_block:
        context_parts.append(memory_block)

    full_context = "\n".join(context_parts) or "No context available"

    # Build a default search query for fallback (from best available source)
    default_query = collected_info[:200] or " ".join(all_msg_texts[-3:])[:200] or goal or "caregiver services"

    _deep_search_logger.info(
        f"deep_search_node full-mode: goal={goal!r}, "
        f"collected_info_len={len(collected_info)}, "
        f"msg_count={len(all_msg_texts)}, "
        f"full_context_len={len(full_context)}, "
        f"default_query={default_query[:100]!r}"
    )

    # 1. Initial strategy call
    strategy_prompt = DEEP_SEARCH_STRATEGY_PROMPT.format(
        tool_descriptions=TOOL_DESCRIPTIONS,
        goal=goal or "(see collected info)",
        collected_info=full_context,
        location="(extract from collected info / user messages)",
    )

    all_tool_runs: List[Dict[str, Any]] = []
    all_results: List[str] = []
    max_iterations = 3

    try:
        _deep_search_logger.info(f"[STEP 1] Calling LLM for strategy (prompt length={len(strategy_prompt)})")
        llm_response = await _search_llm.async_chat(strategy_prompt, max_tokens=1500)
        _deep_search_logger.info(f"[STEP 1] LLM response ({len(llm_response)} chars): {llm_response[:300]}")
    except Exception as e:
        _deep_search_logger.error(f"LLM call failed in deep_search: {e}")
        return {"routing": {"current_agent": "deep_search", "conversation_stage": "error_recovery"}}

    for iteration in range(1, max_iterations + 1):
        _deep_search_logger.info(f"[LOOP] iteration={iteration}")
        # Parse LLM decision
        try:
            decision = _json.loads(llm_response)
            _deep_search_logger.info(f"[LOOP] Parsed JSON directly: {decision}")
        except _json.JSONDecodeError:
            # Try to extract JSON from response
            try:
                start = llm_response.index("{")
                end = llm_response.rindex("}") + 1
                decision = _json.loads(llm_response[start:end])
                _deep_search_logger.info(f"[LOOP] Extracted JSON from text: {decision}")
            except (ValueError, _json.JSONDecodeError):
                _deep_search_logger.warning(f"[LOOP] Could not parse JSON, using fallback. Response: {llm_response[:300]}")
                # Fallback: force a general_online_search with the best query we have
                decision = {"tool": "general_online_search_with_one_query", "args": {"query": default_query}}

        # If LLM says done, use its summary
        if decision.get("done"):
            _deep_search_logger.info(f"[LOOP] LLM says done. summary length={len(decision.get('summary', ''))}")
            all_results.append(decision.get("summary", ""))
            break

        # Execute the chosen tool
        tool_name = decision.get("tool", "")
        tool_args = decision.get("args", {})
        _deep_search_logger.info(f"[LOOP] Tool decision: name={tool_name!r} args={tool_args}")
        if not tool_name:
            _deep_search_logger.warning(f"[LOOP] Empty tool name, breaking. Full decision: {decision}")
            break

        _deep_search_logger.info(f"[LOOP] Calling MCP tool: {tool_name}({tool_args})")
        tool_patch = await call_mcp_tool_patch(
            mgr=MCP_MGR,
            server=SEARCH_SERVER,
            tool_name=tool_name,
            arguments=tool_args,
            purpose=f"Deep search iteration {iteration}: {tool_name}",
            timeout_s=180,
        )
        _deep_search_logger.info(f"[LOOP] MCP tool returned. Keys: {list(tool_patch.keys())}")

        # Collect tool runs
        for tr in (tool_patch.get("tools") or {}).get("tool_runs", []):
            all_tool_runs.append(tr)

        if tool_patch.get("tools", {}).get("tool_failures"):
            _deep_search_logger.warning(f"[LOOP] Tool {tool_name} FAILED: {tool_patch['tools']['tool_failures']}")
            break

        tool_result_data = (tool_patch.get("_mcp_result") or {}).get("data") or {}
        tool_result_str = _json.dumps(tool_result_data, ensure_ascii=False, default=str)[:3000]
        _deep_search_logger.info(f"[LOOP] Tool result ({len(tool_result_str)} chars): {tool_result_str[:200]}")
        all_results.append(f"[{tool_name}] {tool_result_str}")

        # If last iteration, break and summarize
        if iteration == max_iterations:
            _deep_search_logger.info(f"[LOOP] Max iterations reached, breaking to summarize")
            break

        # Ask LLM for next decision
        continuation = SEARCH_CONTINUATION_PROMPT.format(
            tool_name=tool_name,
            tool_result=tool_result_str,
            iteration=iteration,
            max_iterations=max_iterations,
            remaining=max_iterations - iteration,
        )
        try:
            llm_response = await _search_llm.async_chat(
                strategy_prompt + "\n\n" + "\n\n".join(all_results) + "\n\n" + continuation,
                max_tokens=1500,
            )
            _deep_search_logger.info(f"[LOOP] Continuation LLM response: {llm_response[:300]}")
        except Exception as e:
            _deep_search_logger.error(f"LLM continuation call failed: {e}")
            break

    _deep_search_logger.info(f"[DONE] Loop finished. all_results count={len(all_results)}, all_tool_runs count={len(all_tool_runs)}")

    # Generate final summary if we haven't got a "done" summary
    if all_results and not (isinstance(all_results[-1], str) and not all_results[-1].startswith("[")):
        summary_prompt = SEARCH_SUMMARY_PROMPT.format(
            goal=goal or "(see context)",
            location="(see collected info)",
            all_results="\n\n".join(all_results),
        )
        try:
            final_summary = await _search_llm.async_chat(summary_prompt, max_tokens=2000)
        except Exception:
            final_summary = "\n\n".join(all_results)
    else:
        final_summary = all_results[-1] if all_results else _msg(
            state,
            "未能获取搜索结果。",
            "Unable to retrieve search results.",
        )

    combined_data = {"raw_results": all_results, "summary": final_summary}

    return {
        "routing": {"current_agent": "deep_search", "conversation_stage": "executing"},
        "tools": {"tool_runs": all_tool_runs},
        "artifacts_append": {
            "request_id": rid,
            "artifact": {
                "type": "search_results",
                "artifact_id": new_uuid("art"),
                "data": combined_data,
                "produced_by": "deep_search",
            },
        },
        "messages": [{
            "role": "assistant",
            "content": final_summary,
            "metadata": {"agent": "deep_search"},
        }],
    }

async def user_info_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    User info agent — retrieves stored facts/memory for an entity and
    answers the user's question about them. Can also call follow-up
    question MCP tools for clarification.
    """
    rid = _get_active_request_id(state)
    req = _get_active_request(state) or {}
    meta = state.get("meta") or {}
    user_id = meta.get("user_id", "")
    conversation_id = meta.get("conversation_id", "")
    user_text = last_user_text(state)
    lang = _detect_user_language(state)

    client = TrackedAnthropicClient(
        session_id=conversation_id,
        agent_role="user_info",
        user_id=user_id,
    )

    # ── Resolve entity from user text (not just from the active request) ──
    entity_id = ""
    if user_text:
        try:
            from fact_store import get_fact_store as _get_fs_info
            _fs_info = _get_fs_info()
            _known_eids = await _fs_info.get_user_entities(user_id) if user_id else []
        except Exception:
            _known_eids = []
        try:
            from prompts import _infer_subject_entity
            entity_id = await _infer_subject_entity(user_text, client=client, known_entity_ids=_known_eids)
        except Exception as e:
            _logger.debug(f"user_info: entity inference failed: {e}")

    if not entity_id or entity_id == "care_recipient:unknown":
        entity_id = req.get("subject_entity_id") or ""

    profile_facts = {}
    recent_events = []

    if user_id and entity_id:
        try:
            bundle = await get_context_bundle(
                user_id=user_id,
                request_dict=req,
                message=user_text,
                k=10,
            )
            profile_facts = bundle.profile_facts or {}
            recent_events = [
                {"when": e.when, "type": e.event_type, "content": e.content}
                for e in (bundle.recent_events or [])
            ]
        except Exception as e:
            _logger.warning(f"user_info: failed to fetch context bundle: {e}")

    # If no entity resolved, try to find one from DDB
    if not profile_facts and user_id:
        try:
            from fact_store import get_fact_store
            _fs = get_fact_store()
            known_entities = await _fs.get_user_entities(user_id)
            for eid in known_entities:
                label = eid.split(":")[-1] if ":" in eid else eid
                if label.lower() in user_text.lower():
                    _eid_bundle = await get_context_bundle(
                        user_id=user_id,
                        request_dict={"subject_entity_id": eid},
                        message=user_text,
                        k=10,
                    )
                    if _eid_bundle.profile_facts:
                        profile_facts = _eid_bundle.profile_facts
                        entity_id = eid
                        break
        except Exception as e:
            _logger.warning(f"user_info: entity lookup failed: {e}")

    # ── Build facts block for LLM ──
    facts_lines = []
    for key, info in profile_facts.items():
        val = info.get("value", "")
        if isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False)
        ver = info.get("verification_level", "?")
        facts_lines.append(f"  - {key}: {val} [{ver}]")
    facts_block = "\n".join(facts_lines) if facts_lines else "(no stored facts)"

    events_block = "(no recent events)"
    if recent_events:
        events_block = "\n".join(
            f"  - [{e['when']}] ({e['type']}): {e['content']}"
            for e in recent_events[:10]
        )

    # ── Multi-round query loop (max 3 rounds) ──
    request_summaries = []
    all_query_results = {}
    _query_loop_ok = False

    if user_id:
        try:
            previous_summary = ""
            for query_round in range(3):
                # Step A: Generate query plan
                plan = await call_memory_tool(
                    mgr=MCP_MGR, server=MEMORY_SERVER,
                    tool_name="memory_query_planner",
                    arguments={
                        "user_question": user_text,
                        "user_id": user_id,
                        "known_entity_ids": [entity_id] if entity_id else [],
                        "previous_results_summary": previous_summary,
                    },
                    timeout_ms=5000,
                )

                steps = plan.get("steps", []) if isinstance(plan, dict) else []
                if not steps:
                    break

                _query_loop_ok = True

                # Step B: Execute each query step
                round_results = {}
                for step in steps:
                    result = await call_memory_tool(
                        mgr=MCP_MGR, server=MEMORY_SERVER,
                        tool_name="memory_execute_query",
                        arguments={
                            "user_id": user_id,
                            "table": step["table"],
                            "method": step["method"],
                            "params": step.get("params", {}),
                        },
                        timeout_ms=4000,
                    )
                    round_results[step.get("description", f"step_{step['step_id']}")] = result

                all_query_results.update(round_results)

                # Step C: Build summary for potential next round
                previous_summary = json.dumps({
                    k: {"count": v.get("result_count", 0), "table": v.get("table")}
                    for k, v in round_results.items()
                    if isinstance(v, dict)
                }, ensure_ascii=False)

                # If planner didn't indicate follow-up needed, stop
                if not (isinstance(plan, dict) and plan.get("needs_followup", False)):
                    break

        except Exception as e:
            _logger.debug(f"user_info: query loop failed: {e}")

        # Extract request summaries from query results
        for key, result in all_query_results.items():
            if isinstance(result, dict) and result.get("table") == "requests":
                items = result.get("results", [])
                if isinstance(items, list):
                    request_summaries.extend(items)

        # Dedup by request_id
        seen_ids = set()
        deduped = []
        for r in request_summaries:
            rid_val = r.get("request_id") if isinstance(r, dict) else None
            if rid_val and rid_val not in seen_ids:
                seen_ids.add(rid_val)
                deduped.append(r)
        request_summaries = deduped

        # Fallback: if query planner was unavailable, use legacy approach
        if not _query_loop_ok:
            try:
                recent_reqs = await call_memory_tool(
                    mgr=MCP_MGR, server=MEMORY_SERVER,
                    tool_name="memory_list_recent_requests",
                    arguments={"user_id": user_id, "limit": 10},
                    timeout_ms=3000,
                )
                if isinstance(recent_reqs, list):
                    request_summaries.extend(recent_reqs)
            except Exception as e:
                _logger.debug(f"user_info: fallback recent requests fetch failed: {e}")

    # Build request history block
    # Sort by created_at descending (safety net in case the store doesn't sort)
    request_summaries.sort(key=lambda r: r.get("created_at", ""), reverse=True)

    requests_block = "(no request history)"
    if request_summaries:
        req_lines = []
        for rs in request_summaries[:20]:
            title = rs.get("title") or rs.get("name", "Untitled")
            status = rs.get("status", "?")
            entity = rs.get("subject_entity_id", "")
            created = rs.get("created_at", "?")
            req_type = rs.get("request_type", "")
            summary = rs.get("summary_current", "")
            line = f"  - [{created}] {title} | status={status}"
            if entity:
                line += f" | for={entity}"
            if req_type:
                line += f" | type={req_type}"
            if summary:
                line += f"\n    Summary: {summary}"
            req_lines.append(line)
        requests_block = "\n".join(req_lines)

    # ── Get conversation history for context ──
    messages = state.get("messages", []) or []
    recent_msgs = []
    for msg in messages[-10:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            recent_msgs.append(f"[{role}]: {content[:300]}")
    conversation_text = "\n".join(recent_msgs) if recent_msgs else "(no history)"

    # ── LLM call ──
    lang_instruction = "Respond in Chinese (中文)." if lang == "zh" else "Respond in English."

    prompt = f"""You are a user information agent for a caregiving platform. \
Your job is to answer the user's question using stored facts and recent events \
about a care recipient or the user themselves.

## Entity: {entity_id or "(unknown)"}

## Stored Facts
{facts_block}

## Recent Events
{events_block}

## Request History
{requests_block}

## Recent Conversation
{conversation_text}

## User's Question
{user_text}

## Instructions
1. Answer the user's question based on the stored facts and events above.
2. Present the information naturally — summarize, don't just dump raw data.
3. If the stored facts don't fully answer the question, say what you know and \
what's missing. Suggest where they might find the missing information.
4. If the user asks to update or correct information, acknowledge it and note \
that the changes will be saved.
5. Be concise and helpful.
6. If the user asks about requests, tasks, or what you've helped with, use the \
Request History section above. Include titles, statuses, and who the request \
was for. If they ask about a specific entity, focus on that entity's requests.

{lang_instruction}"""

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=800,
            temperature=0.3,
        )
        answer = response.strip()
    except Exception as e:
        _logger.warning(f"user_info LLM call failed: {e}")
        if lang == "zh":
            answer = "抱歉，我无法检索到相关信息。请稍后再试。"
        else:
            answer = "Sorry, I wasn't able to retrieve the information. Please try again."

    # ── Fact correction: reconcile if user is correcting stored facts ──
    if profile_facts and user_id and entity_id:
        try:
            from prompts import llm_extract_facts as _extract

            # Lightweight extraction: treat user text as a mini-summary
            correction_facts = await _extract(
                entity_id=entity_id,
                request_type=req.get("request_type", ""),
                updated_summary=user_text,
                key_info_needed=[],
                conversation_history=messages[-10:],
                client=client,
                already_extracted_keys=[],
                existing_profile_facts={
                    k: v for k, v in profile_facts.items()
                },
                disputed_fact_keys=[],
            )

            if correction_facts:
                # Build dicts for reconciliation
                resolved_input = [
                    {"fact_key": f.get("fact_key", ""), "value": f.get("value", "")}
                    for f in correction_facts
                    if f.get("fact_key")
                ]
                profile_dict = {
                    k: v for k, v in profile_facts.items()
                }
                reconciliation = await llm_reconcile_facts(
                    existing_profile_facts=profile_dict,
                    resolved_facts=resolved_input,
                    conversation_history=messages[-10:],
                    client=client,
                )

                if reconciliation is not None:
                    from fact_store import get_fact_store
                    _store = get_fact_store()

                    # Deprecate existing facts the reconciler flagged
                    for dep in reconciliation.get("deprecate_existing", []):
                        dep_key = dep.get("fact_key") if isinstance(dep, dict) else None
                        if not dep_key:
                            continue
                        try:
                            old_facts = await _store.get_active_facts(
                                user_id=user_id,
                                entity_id=entity_id,
                                fact_keys=[dep_key],
                            )
                            for of in old_facts:
                                await _store.deprecate_fact(of)
                                _logger.info(
                                    f"user_info reconciliation deprecated "
                                    f"'{dep_key}={of.fact_value}'"
                                )
                        except Exception as e:
                            _logger.warning(
                                f"user_info: failed to deprecate {dep_key}: {e}"
                            )

                    # Write new/corrected facts
                    for wf in reconciliation.get("write_new", []):
                        wf_key = wf.get("fact_key") if isinstance(wf, dict) else None
                        wf_val = wf.get("new_value") if isinstance(wf, dict) else None
                        if not wf_key or wf_val is None:
                            continue
                        try:
                            await _store.upsert_fact(
                                user_id=user_id,
                                entity_id=entity_id,
                                fact_key=wf_key,
                                new_value=wf_val,
                                fact_label=wf_key,
                                status="active",
                                confidence=0.9,
                                source_type="user",
                                source_ref=rid or "",
                                evidence=user_text[:200],
                            )
                            _logger.info(
                                f"user_info reconciliation wrote "
                                f"'{wf_key}={wf_val}'"
                            )
                        except Exception as e:
                            _logger.warning(
                                f"user_info: failed to write {wf_key}: {e}"
                            )
        except Exception as e:
            _logger.warning(f"user_info fact correction failed (non-blocking): {e}")

    result = {
        "routing": {"current_agent": "user_info"},
        "messages": [{
            "role": "assistant",
            "content": answer,
            "metadata": {"agent": "user_info"},
        }],
        "info_collection_debug": {
            "profile_facts_retrieved": profile_facts,
            "request_query_debug": {
                "user_id": user_id,
                "entity_id": entity_id,
                "recent_reqs_count": len(request_summaries),
                "recent_reqs_raw": request_summaries,
                "requests_block_sent_to_llm": requests_block,
            },
        },
    }

    # Record stage transition
    if rid and req:
        existing_history = req.get("stage_history", [])
        userinfo_transition = {
            "from_stage": req.get("status", "validated"),
            "to_stage": "user_info_executing",
            "agent": "user_info",
            "timestamp": datetime.utcnow(),
            "reason": "User info retrieval and summarization"
        }
        result["request_manager"] = {
            "requests": {
                rid: {
                    "stage_history": existing_history + [userinfo_transition],
                }
            }
        }

    return result

async def domain_expert_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rid = _get_active_request_id(state)
    req = _get_active_request(state) or {}

    # ── Demo mode: unchanged original behavior ──────────────────────────
    if DEEP_SEARCH_MODE == "demo":
        tool_patch = await call_mcp_tool_patch(
            mgr=MCP_MGR,
            server=DEMO_SERVER,
            tool_name="medicaid_checklist",
            arguments={"state": "IL"},
            purpose="Demo guidance tool call (replace with real domain expert generation)",
        )
        if tool_patch.get("tools", {}).get("tool_failures"):
            return {"routing": {"current_agent": "domain_expert", "conversation_stage": "error_recovery"}, **tool_patch}

        data = (tool_patch.get("_mcp_result") or {}).get("data") or {}
        steps = data.get("steps") or []

        return {
            "routing": {"current_agent": "domain_expert", "conversation_stage": "executing"},
            **tool_patch,
            "messages": [{
                "role": "assistant",
                "content": _msg(
                    state,
                    "（Demo）这里是一个示例 checklist：\n- " + "\n- ".join(steps),
                    "(Demo) Here is a sample checklist:\n- " + "\n- ".join(steps),
                ),
                "metadata": {"agent": "domain_expert"}
            }],
        }

    # ── Full mode: web search + LLM synthesis ───────────────────────────
    goal = req.get("goal") or req.get("name") or "domain question"
    info_state = req.get("info_collection_state") or {}
    collected_info = info_state.get("summary_of_collected_info") or ""

    # Same shallow-replacement issue as deep_search: pull from all messages
    if not collected_info:
        msgs = state.get("messages") or []
        all_texts = []
        for m in msgs:
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            if content:
                all_texts.append(content)
        collected_info = " ".join(all_texts[-5:]) if all_texts else goal
    # Inject memory context so domain expert knows stored facts
    memory_block = await _build_memory_context_block(state)
    context = f"{goal}\n{collected_info}"
    if memory_block:
        context = f"{context}\n{memory_block}"

    # 1. Search the web for domain knowledge
    search_query = collected_info[:200] if collected_info else goal
    tool_patch = await call_mcp_tool_patch(
        mgr=MCP_MGR,
        server=SEARCH_SERVER,
        tool_name="general_online_search_with_one_query",
        arguments={"query": search_query},
        purpose="Domain expert: web search for guidance",
    )

    tool_runs = (tool_patch.get("tools") or {}).get("tool_runs", [])

    if tool_patch.get("tools", {}).get("tool_failures"):
        return {"routing": {"current_agent": "domain_expert", "conversation_stage": "error_recovery"}, **tool_patch}

    search_data = (tool_patch.get("_mcp_result") or {}).get("data") or {}
    search_results_str = _json.dumps(search_data, ensure_ascii=False, default=str)[:3000]

    # 2. Synthesize with LLM
    synthesis_prompt = DOMAIN_EXPERT_SYNTHESIS_PROMPT.format(
        context=context,
        search_results=search_results_str,
    )

    try:
        guidance = await _search_llm.async_chat(synthesis_prompt, max_tokens=2000)
    except Exception as e:
        _deep_search_logger.error(f"Domain expert LLM call failed: {e}")
        guidance = f"Search results:\n{search_results_str}"

    return {
        "routing": {"current_agent": "domain_expert", "conversation_stage": "executing"},
        "tools": {"tool_runs": tool_runs},
        "artifacts_append": {
            "request_id": rid,
            "artifact": {
                "type": "written_guidance",
                "artifact_id": new_uuid("art"),
                "data": {"guidance": guidance, "search_results": search_data},
                "produced_by": "domain_expert",
            },
        },
        "messages": [{
            "role": "assistant",
            "content": guidance,
            "metadata": {"agent": "domain_expert"},
        }],
    }

async def quick_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight agent for quick factual questions, definitions, eligibility info.
    Does NOT create a request or produce artifacts — just answers directly.
    Uses an LLM decision call to route: direct answer, web search, or follow-up questions.
    """
    meta = state.get("meta") or {}
    conversation_id = meta.get("conversation_id", "conv-unknown")
    user_id = meta.get("user_id", "user-unknown")

    # Extract the user's question
    user_text = last_user_text(state)

    # Gather recent conversation for context
    msgs = state.get("messages") or []
    recent_context = []
    for m in msgs[-6:]:
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
        if content:
            recent_context.append(f"[{role}]: {content[:200]}")
    question_context = "\n".join(recent_context) if recent_context else user_text

    # Enrich with memory context (non-blocking, fails gracefully)
    memory_block = await _build_memory_context_block(state)
    if memory_block:
        question_context = question_context + "\n" + memory_block

    tool_runs = []

    # Step 1: LLM decides action (direct_answer / web_search / follow_up)
    decision_prompt = QUICK_ANSWER_DECISION_PROMPT.format(
        question_context=question_context,
    )
    try:
        decision_raw = await _search_llm.async_chat(decision_prompt, max_tokens=200)
        decision = _json.loads(decision_raw.strip())
    except Exception as e:
        _deep_search_logger.warning(f"quick_answer decision parse failed: {e}")
        decision = {"action": "direct_answer"}

    action = decision.get("action", "direct_answer")

    # Step 2: Execute the chosen action
    if action == "follow_up":
        # Build chat_turns from recent messages for the follow-up MCP tool
        chat_turns = []
        for m in msgs[-6:]:
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
            if content:
                chat_turns.append(f"{role}: {content[:200]}")

        try:
            tool_patch = await call_mcp_tool_patch(
                mgr=MCP_MGR,
                server=FOLLOW_UP_SERVER,
                tool_name="generate_follow_up_questions",
                arguments={"chat_turns": chat_turns},
                purpose="Quick answer: generating follow-up questions",
            )
            for tr in (tool_patch.get("tools") or {}).get("tool_runs", []):
                tool_runs.append(tr)

            if not tool_patch.get("tools", {}).get("tool_failures"):
                answer = (tool_patch.get("_mcp_result") or {}).get("data") or ""
                if not answer:
                    answer = str((tool_patch.get("_mcp_result") or {}))
            else:
                _deep_search_logger.warning("quick_answer follow-up MCP tool failed, falling back to direct answer")
                answer = await _search_llm.async_chat(
                    QUICK_ANSWER_PROMPT.format(
                        search_context="(No web search performed — answer from your own knowledge.)",
                        question_context=question_context,
                    ),
                    max_tokens=1000,
                )
        except Exception as e:
            _deep_search_logger.warning(f"quick_answer follow-up failed: {e}")
            answer = await _search_llm.async_chat(
                QUICK_ANSWER_PROMPT.format(
                    search_context="(No web search performed — answer from your own knowledge.)",
                    question_context=question_context,
                ),
                max_tokens=1000,
            )

    elif action == "web_search":
        search_query = decision.get("query", user_text[:200])
        search_context = ""

        try:
            tool_patch = await call_mcp_tool_patch(
                mgr=MCP_MGR,
                server=SEARCH_SERVER,
                tool_name="general_online_search_with_one_query",
                arguments={"query": search_query},
                purpose="Quick answer: web search for factual data",
            )
            for tr in (tool_patch.get("tools") or {}).get("tool_runs", []):
                tool_runs.append(tr)

            if not tool_patch.get("tools", {}).get("tool_failures"):
                search_data = (tool_patch.get("_mcp_result") or {}).get("data") or {}
                search_str = _json.dumps(search_data, ensure_ascii=False, default=str)[:3000]
                search_context = f"## Web search results\n{search_str}"
        except Exception as e:
            _deep_search_logger.warning(f"quick_answer web search failed: {e}")

        if not search_context:
            search_context = "(No web search performed — answer from your own knowledge.)"

        try:
            answer = await _search_llm.async_chat(
                QUICK_ANSWER_PROMPT.format(
                    search_context=search_context,
                    question_context=question_context,
                ),
                max_tokens=1000,
            )
        except Exception as e:
            _deep_search_logger.error(f"quick_answer LLM call failed: {e}")
            answer = _msg(
                state,
                "抱歉，我暂时无法回答这个问题。请稍后再试，或者你可以换一种方式提问。",
                "Sorry, I'm unable to answer that question right now. Please try again later or rephrase your question.",
            )

    else:  # direct_answer (default)
        try:
            answer = await _search_llm.async_chat(
                QUICK_ANSWER_PROMPT.format(
                    search_context="(No web search performed — answer from your own knowledge.)",
                    question_context=question_context,
                ),
                max_tokens=1000,
            )
        except Exception as e:
            _deep_search_logger.error(f"quick_answer LLM call failed: {e}")
            answer = _msg(
                state,
                "抱歉，我暂时无法回答这个问题。请稍后再试，或者你可以换一种方式提问。",
                "Sorry, I'm unable to answer that question right now. Please try again later or rephrase your question.",
            )

    result: Dict[str, Any] = {
        "routing": {"current_agent": "quick_answer"},
        "messages": [{
            "role": "assistant",
            "content": answer,
            "metadata": {"agent": "quick_answer"},
        }],
    }
    if tool_runs:
        result["tools"] = {"tool_runs": tool_runs}

    return result


def check_prereq_lifecycle(state) -> Optional[Dict[str, Any]]:
    """Check if a prereq lifecycle transition is needed and return a patch if so.

    This runs OUTSIDE LangGraph, on the full Pydantic UnifiedState, because
    LangGraph's TypedDict state does shallow replacement on Dict[str, Any] keys
    and nodes inside the graph only see partial state from the last node's output.

    Called by the orchestration loop after each graph turn completes.
    Returns None if no transition needed, or a patch dict for apply_node_output.

    Detects two conditions:
      1. COMPLETION: active prereq request has status=validated → resume parent
      2. ABANDON: user_signals.wants_to_abandon_current_task → abort prereq, resume parent

    NOTE (DDB persistence hook): Every mutation here should be mirrored to
    DynamoDB in production:
      - UpdateItem on prereq request (status → completed / aborted)
      - DeleteItem on PendingQueueItem (dequeue parent)
      - UpdateItem on parent request (status → collecting, stage_history append)
      - UpdateItem on RequestManager (active_request_id → parent)
    """
    rm = state.request_manager
    active_id = rm.active_request_id
    if not active_id:
        return None

    reqs = rm.requests or {}
    req = reqs.get(active_id)
    if not req:
        return None

    parent_id = req.parent_request_id

    # Only act if the active request IS a prerequisite (has a parent)
    if not parent_id:
        return None

    info_state = req.info_collection_state or {}
    user_signals = info_state.get("user_signals") or {}

    # ── CONDITION 1: Prereq completed (status=executed — work is done) ──
    prereq_completed = req.status == "executed"

    # ── CONDITION 2: User wants to abandon current prereq ────────────
    prereq_abandoned = user_signals.get("wants_to_abandon_current_task") is True

    if not prereq_completed and not prereq_abandoned:
        return None

    # ── Resolve: resume parent request ───────────────────────────────
    parent_req = reqs.get(parent_id)
    if not parent_req:
        return None

    parent_history = parent_req.stage_history or []

    if prereq_completed:
        prereq_final_status = "completed"
        resume_reason = f"Prerequisite {active_id} completed; resuming parent request"
        prereq_stage_detail = "completed_as_prerequisite"
    else:
        prereq_final_status = "aborted"
        abandon_reason = user_signals.get("abandon_reason") or "User chose to abandon"
        resume_reason = f"Prerequisite {active_id} abandoned: {abandon_reason}"
        prereq_stage_detail = "aborted_by_user"

    resume_transition = {
        "from_stage": "paused",
        "to_stage": "collecting",
        "agent": "info_collection",
        "timestamp": datetime.utcnow(),
        "reason": resume_reason,
    }

    prereq_history = req.stage_history or []
    prereq_close_transition = {
        "from_stage": req.status,
        "to_stage": prereq_final_status,
        "agent": "prereq_lifecycle_manager",
        "timestamp": datetime.utcnow(),
        "reason": resume_reason,
    }

    # Build the "welcome back" message for the parent request
    parent_info = parent_req.info_collection_state or {}
    parent_summary = parent_info.get("summary_of_collected_info", "")

    # Detect language from the last user message on the Pydantic state
    _last_user = ""
    for m in reversed(state.messages or []):
        if m.role == "user":
            _last_user = m.content or ""
            break
    _cn = sum(1 for c in _last_user if '\u4e00' <= c <= '\u9fff')
    _lang = "zh" if _cn > max(1, len(_last_user) * 0.1) else "en"

    prereq_label = req.name or ("先决条件" if _lang == "zh" else "prerequisite")
    parent_name = parent_req.name or ("之前的任务" if _lang == "zh" else "previous task")

    if _lang == "zh":
        if prereq_completed:
            welcome_msg = f"好的，{prereq_label}已经处理完了。现在我们继续{parent_name}。"
        else:
            welcome_msg = f"好的，我们先不处理{prereq_label}了。现在继续{parent_name}。"
        if parent_summary:
            welcome_msg += f"\n\n之前已经收集到的信息：\n{parent_summary[:300]}"
        welcome_msg += "\n\n还有什么需要补充的吗？"
    else:
        if prereq_completed:
            welcome_msg = f"OK, {prereq_label} is done. Let's continue with {parent_name}."
        else:
            welcome_msg = f"OK, let's skip {prereq_label} for now. Continuing with {parent_name}."
        if parent_summary:
            welcome_msg += f"\n\nInformation collected so far:\n{parent_summary[:300]}"
        welcome_msg += "\n\nAnything else to add?"

    return {
        "routing": {
            "current_agent": "info_collection",
            "conversation_stage": "collecting",
            "pending_handoff": {
                "recommended_next_agent": None,
                "reason": "",
            },
        },
        "request_manager": {
            "active_request_id": parent_id,
            "pending_queue_remove": [parent_id],
            "requests": {
                active_id: {
                    "status": prereq_final_status,
                    "stage_detail": prereq_stage_detail,
                    "awaiting_user_input": False,
                    "last_touched_at": datetime.utcnow(),
                    "stage_history": prereq_history + [prereq_close_transition],
                },
                parent_id: {
                    "status": "collecting",
                    "stage_detail": "resumed_from_prerequisite",
                    "awaiting_user_input": True,
                    "last_touched_at": datetime.utcnow(),
                    "stage_history": parent_history + [resume_transition],
                },
            },
        },
        "messages": [{
            "role": "assistant",
            "content": welcome_msg,
            "metadata": {"agent": "prereq_lifecycle_manager"},
        }],
    }
    # Sync prereq completion and parent resume to DDB
    state_as_dict = {"meta": {"user_id": state.meta.user_id}}
    _append_request_ddb_sync(patch, state_as_dict, active_id, {
        "status": prereq_final_status,
        "stage_detail": prereq_stage_detail,
    })
    _append_request_ddb_sync(patch, state_as_dict, parent_id, {
        "status": "collecting",
        "stage_detail": "resumed_from_prerequisite",
    })
    return patch


def task_complete_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles the task_complete signal from turn_router.
    Sets the active request status to 'executed' and sends an acknowledgment.
    The post-graph check_prereq_lifecycle will handle parent resumption if needed.
    """
    rid = _get_active_request_id(state)
    req = _get_active_request(state) or {}

    if not rid:
        return {
            "messages": [{
                "role": "assistant",
                "content": _msg(state, "好的，还有什么我可以帮你的吗？", "OK, is there anything else I can help with?"),
                "metadata": {"agent": "task_complete"}
            }],
        }

    existing_history = req.get("stage_history", [])
    current_status = req.get("status", "unknown")
    complete_transition = {
        "from_stage": current_status,
        "to_stage": "executed",
        "agent": "task_complete",
        "timestamp": datetime.utcnow(),
        "reason": "User acknowledged results; task complete"
    }

    req_name = req.get("name", "task")
    lang = _detect_user_language(state)
    done_content = (
        f"好的，{req_name}已经处理完了。还有什么我可以帮你的吗？"
        if lang == "zh" else
        f"Got it, {req_name} is all done. Is there anything else I can help with?"
    )

    result_patch = {
        "request_manager": {
            "requests": {
                rid: {
                    "status": "executed",
                    "stage_detail": "user_acknowledged_results",
                    "awaiting_user_input": False,
                    "last_touched_at": datetime.utcnow(),
                    "stage_history": existing_history + [complete_transition],
                }
            }
        },
        "messages": [{
            "role": "assistant",
            "content": done_content,
            "metadata": {"agent": "task_complete"}
        }],
    }
    _append_request_ddb_sync(result_patch, state, rid, {
        "status": "executed",
        "stage_detail": "user_acknowledged_results",
    })
    return result_patch


def downstream_catcher(state: Dict[str, Any]) -> Dict[str, Any]:
    import logging
    _routing_dbg = state.get("routing") or {}
    _ph_dbg = _routing_dbg.get("pending_handoff") or {}
    logging.info(f"[downstream_catcher] routing keys={list(_routing_dbg.keys())}, pending_handoff={_ph_dbg}, _catcher_next={_routing_dbg.get('_catcher_next')}")

    failures = ((state.get("tools") or {}).get("tool_failures")) or []
    if failures:
        return {
            "routing": {
                "conversation_stage": "error_recovery",
                "pending_handoff": {"recommended_next_agent": "front_end_emotional_support", "reason": "Tool failure recovery"},
                "_catcher_next": "front_end_emotional_support",
            },
        }

    pending = ((state.get("routing") or {}).get("pending_handoff") or {})
    nxt = pending.get("recommended_next_agent")
    if nxt:
        # Consume the pending_handoff: save the decision in _catcher_next, then clear pending_handoff
        return {
            "routing": {
                "pending_handoff": {"recommended_next_agent": None, "reason": ""},
                "_catcher_next": nxt,
            }
        }

    return {
        "routing": {
            "pending_handoff": {"recommended_next_agent": None, "reason": "Done"},
            "_catcher_next": None,
        }
    }

def route_after_catcher(state: Dict[str, Any]) -> RouteKey:
    routing = state.get("routing") or {}
    nxt = routing.get("_catcher_next")
    if nxt is None:
        return "respond"
    if nxt == "front_end_emotional_support":
        return "front_end"
    if nxt == "info_collection":
        return "info_collection"
    if nxt == "deep_search":
        return "deep_search"
    if nxt == "user_info":
        return "user_info"
    if nxt == "domain_expert":
        return "domain_expert"
    if nxt == "quick_answer":
        return "quick_answer"
    if nxt == "human_comm":
        return "human_comm"
    return "respond"

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("turn_router", turn_router)
    g.add_node("upstream_delegator", upstream_delegator)
    g.add_node("front_end", front_end_node)
    g.add_node("info_collection", info_collection_node)
    g.add_node("deep_search", deep_search_node)
    g.add_node("user_info", user_info_node)
    g.add_node("domain_expert", domain_expert_node)
    g.add_node("quick_answer", quick_answer_node)
    g.add_node("human_comm", human_comm_node)
    g.add_node("downstream_catcher", downstream_catcher)

    g.set_entry_point("turn_router")

    g.add_conditional_edges(
        "turn_router",
        route_from_turn_router,
        {
            "upstream_delegator": "upstream_delegator",
            "front_end": "front_end",
            "info_collection": "info_collection",
            "deep_search": "deep_search",
            "user_info": "user_info",
            "domain_expert": "domain_expert",
            "quick_answer": "quick_answer",
            "human_comm": "human_comm",
            "respond": END,
            "escalate": END,
        },
    )

    g.add_conditional_edges(
        "upstream_delegator",
        route_from_delegator,
        {
            "front_end": "front_end",
            "info_collection": "info_collection",
            "deep_search": "deep_search",
            "user_info": "user_info",
            "domain_expert": "domain_expert",
            "quick_answer": "quick_answer",
            "human_comm": "human_comm",
            "respond": END,
            "escalate": END,
        },
    )

    for n in ["front_end", "info_collection", "deep_search", "user_info", "domain_expert", "quick_answer", "human_comm"]:
        g.add_edge(n, "downstream_catcher")

    g.add_conditional_edges(
        "downstream_catcher",
        route_after_catcher,
        {
            "front_end": "front_end",
            "info_collection": "info_collection",
            "deep_search": "deep_search",
            "user_info": "user_info",
            "domain_expert": "domain_expert",
            "quick_answer": "quick_answer",
            "human_comm": "human_comm",
            "respond": END,
            "escalate": END,
        },
    )

    return g.compile()