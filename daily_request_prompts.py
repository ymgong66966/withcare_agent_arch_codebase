"""
Daily Request Prompts — LLM prompt template and caller for request hygiene
during the daily cron job.

Analyzes each request against the full day's conversation to determine
true status and recommended action.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from anthropic_client import TrackedAnthropicClient

logger = logging.getLogger(__name__)


DAILY_REQUEST_HYGIENE_PROMPT = """\
You are analyzing requests from a caregiving assistant to determine their true
status at end of day. The system tracks requests in-memory but statuses sometimes
drift from reality.

## ALL REQUESTS CREATED/ACTIVE TODAY:
{requests_text}

## FULL DAY'S CONVERSATION:
{conversation_text}

## INSTRUCTIONS:
For EACH request, determine:

1. **true_status**: What is the real status based on the conversation?
   - "completed": User got what they needed, conversation moved on naturally
   - "still_needed": Request is legitimately still in progress or waiting
   - "abandoned": User lost interest, changed topics permanently, or explicitly declined
   - "ad_hoc_resolved": Was a quick question that got answered immediately

2. **updated_summary**: A comprehensive 1-3 sentence summary reflecting the FULL
   conversation (not just the initial request). What was asked? What was provided?
   What's the current state?

3. **left_off_at**: What was the last meaningful interaction point for this request?
   (e.g., "User received search results for in-home caregivers",
    "Agent asked about insurance details, user hasn't responded yet")

4. **recommended_action**: What should the system do?
   - "close": Mark as completed/resolved — no further action needed
   - "keep_active": Still being actively worked on
   - "keep_queued": Not active but user may return to it
   - "archive": Move to historical records

## RULES:
- Ad-hoc questions that got answered in 1-2 turns → close (ad_hoc_resolved)
- Quick Q&A where user got the info → close
- User firmly changed topics and never came back → abandoned → archive
- Epic requests with prerequisites still pending → keep_queued
- Requests where user needs to do something first (get docs, etc.) → keep_queued
- Requests still actively being worked → keep_active
- If the system shows status "created" but user already got answers → close

## OUTPUT (JSON only):
{{
  "request_analyses": [
    {{
      "request_id": "...",
      "true_status": "completed|still_needed|abandoned|ad_hoc_resolved",
      "updated_summary": "...",
      "left_off_at": "...",
      "recommended_action": "close|keep_active|keep_queued|archive",
      "reason": "Brief explanation of your judgment"
    }}
  ]
}}

Respond ONLY with the JSON object."""


def _format_requests_for_prompt(requests: List[Dict[str, Any]]) -> str:
    """Format request records into a readable text block."""
    if not requests:
        return "(no requests)"

    lines = []
    for r in requests:
        rid = r.get("request_id", "unknown")
        name = r.get("name", "Unknown")
        goal = r.get("goal", "")[:150]
        status = r.get("status", "unknown")
        stage = r.get("stage_detail", "")
        created = r.get("created_at", "")
        summary = r.get("summary_current", "")
        req_type = r.get("request_type", "")

        lines.append(
            f"- [{rid}] {name}\n"
            f"  Goal: {goal}\n"
            f"  System status: {status} | Stage: {stage}\n"
            f"  Type: {req_type} | Created: {created}\n"
            f"  Current summary: {summary or '(none)'}"
        )
    return "\n\n".join(lines)


def _format_conversation_for_prompt(messages: List[Dict[str, Any]]) -> str:
    """Format conversation messages into a readable text block."""
    if not messages:
        return "(no conversation today)"

    lines = []
    for msg in messages[-80:]:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")[:300]
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        return json.loads(text[json_start:json_end])
    return json.loads(text)


async def llm_daily_request_hygiene(
    client: TrackedAnthropicClient,
    requests: List[Dict[str, Any]],
    conversation: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call LLM to analyze request status and recommend actions.

    Returns dict with 'request_analyses' list.
    """
    prompt = DAILY_REQUEST_HYGIENE_PROMPT.format(
        requests_text=_format_requests_for_prompt(requests),
        conversation_text=_format_conversation_for_prompt(conversation),
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1,
        )
        return _extract_json(response)
    except Exception as e:
        logger.error(f"LLM daily request hygiene failed: {e}")
        return {"request_analyses": []}
