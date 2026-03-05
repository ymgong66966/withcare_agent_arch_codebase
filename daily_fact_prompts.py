"""
Daily Fact Prompts — LLM prompt templates and callers for fact validation
and candidate key review during the daily cron job.

Follows the same pattern as prompts.py: template string + async LLM caller
with JSON extraction.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from anthropic_client import TrackedAnthropicClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Sub-workflow A: Fact Cross-Validation
# ─────────────────────────────────────────────────────────────

DAILY_FACT_VALIDATION_PROMPT = """\
You are a data quality auditor for a caregiving assistant's memory system.

Your job is to review ALL active facts for a user against today's conversation
and flag problems. Be conservative — only flag things that are clearly wrong.

## ALL ACTIVE FACTS FOR THIS USER:
{active_facts_text}

## TODAY'S CONVERSATION:
{conversation_text}

## FACT KEY REGISTRY NAMESPACES:
{registry_summary}

## INSTRUCTIONS:
Review each active fact and identify any that are:

1. **CONTRADICTED**: The user said something today that directly contradicts this fact.
   Example: Fact says "insurance.carrier: Blue Cross" but user said "We switched to Aetna."

2. **STALE_TIMESTAMP**: Contains a relative time reference that is now incorrect.
   Example: "housing.move_date: next week", "appointment.next_visit: tomorrow",
   "medication.started: last month"

3. **CONVERSATION_ARTIFACT**: Describes the conversation itself, not a real fact about
   the person. These should never have been stored.
   Example: "insurance.clarification_needed: true", "preference.information_request: yes",
   "contact.question_about_phone: asked"

4. **NULL_OR_MEANINGLESS**: The value is empty, "none", "N/A", "unknown", "not sure",
   or similarly meaningless.

5. **DUPLICATE**: The same information is stored under two different keys.
   Example: "identity.full_name: John Smith" and "identity.legal_name: John Smith"

IMPORTANT RULES:
- Do NOT flag a fact just because it wasn't mentioned today.
- Do NOT flag facts that are simply old but still likely true.
- Only flag things that are CLEARLY wrong, stale, or artifacts.
- When in doubt, put it in flag_for_review instead of deprecate.

## OUTPUT (JSON only):
{{
  "deprecate": [
    {{"entity_id": "...", "fact_key": "...", "reason": "..."}}
  ],
  "flag_for_review": [
    {{"entity_id": "...", "fact_key": "...", "concern": "...", "severity": "low|medium|high"}}
  ]
}}

If nothing to flag, return: {{"deprecate": [], "flag_for_review": []}}

Respond ONLY with the JSON object."""


# ─────────────────────────────────────────────────────────────
# Sub-workflow B: Candidate Key Review
# ─────────────────────────────────────────────────────────────

CANDIDATE_KEY_REVIEW_PROMPT = """\
You are reviewing proposed fact keys that have been extracted from user conversations
but don't exist in the official registry. Your job is to classify each candidate.

## CANDIDATE KEYS TO REVIEW:
{candidates_text}

## OFFICIAL FACT KEY REGISTRY (all namespaces and keys):
{registry_summary}

## INSTRUCTIONS:
For each candidate key, determine one of three outcomes:

1. **DUPLICATE / ALIAS**: This candidate maps to an existing registry key under a
   different name. Suggest it as an alias.
   Example: "insurance.plan_name" → alias for "insurance.plan_type"

2. **NEW_KEY**: This is a genuinely new concept not covered by the registry.
   Propose a proper namespace.facet name, risk_level, and value_type.
   Example: "care_schedule.weekday_hours" → new key for tracking care hours

3. **REJECT**: This is a conversation artifact, too vague, or not a real fact type.
   Example: "preference.information_request" → conversation artifact, reject

## OUTPUT (JSON only):
{{
  "alias_mappings": [
    {{
      "candidate_key": "...",
      "canonical_key": "existing.registry.key",
      "suggested_aliases": ["alias1", "alias2"]
    }}
  ],
  "proposed_new_keys": [
    {{
      "key": "namespace.facet",
      "description": "What this fact represents",
      "risk_level": "low|medium|high",
      "value_type": "string|enum|date|number|list|object",
      "namespace": "namespace",
      "reason": "Why this should be added"
    }}
  ],
  "reject": [
    {{"candidate_key": "...", "reason": "..."}}
  ]
}}

If no candidates to review, return empty arrays for all fields.

Respond ONLY with the JSON object."""


# ─────────────────────────────────────────────────────────────
# LLM Callers
# ─────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        return json.loads(text[json_start:json_end])
    return json.loads(text)


def _format_facts_for_prompt(facts: List[Dict[str, Any]]) -> str:
    """Format active facts into a readable text block for the prompt."""
    if not facts:
        return "(no active facts)"

    lines = []
    for f in facts:
        entity = f.get("entity_id", "unknown")
        key = f.get("fact_key", "unknown")
        value = f.get("fact_value", "")
        status = f.get("status", "")
        confidence = f.get("confidence", 0)
        lines.append(
            f"- [{entity}] {key} = {value!r}  "
            f"(status={status}, confidence={confidence})"
        )
    return "\n".join(lines)


def _format_candidates_for_prompt(candidates: List[Dict[str, Any]]) -> str:
    """Format candidate keys into a readable text block."""
    if not candidates:
        return "(no candidates)"

    lines = []
    for c in candidates:
        key = c.get("candidate_key", "unknown")
        count = c.get("occurrence_count", 0)
        desc = c.get("description", "")
        samples = c.get("sample_values", [])
        entities = c.get("sample_entities", [])
        lines.append(
            f"- {key} (seen {count}x): {desc}\n"
            f"  Sample values: {samples[:3]}\n"
            f"  Sample entities: {entities[:3]}"
        )
    return "\n".join(lines)


def _format_conversation_for_prompt(messages: List[Dict[str, Any]]) -> str:
    """Format conversation messages into a readable text block."""
    if not messages:
        return "(no conversation today)"

    lines = []
    for msg in messages[-80:]:  # Last 80 messages
        role = msg.get("role", "").upper()
        content = msg.get("content", "")[:300]
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


async def llm_daily_fact_validation(
    client: TrackedAnthropicClient,
    active_facts: List[Dict[str, Any]],
    conversation: List[Dict[str, Any]],
    registry_summary: str,
) -> Dict[str, Any]:
    """
    Call LLM to cross-validate active facts against today's conversation.

    Returns dict with 'deprecate' and 'flag_for_review' lists.
    """
    prompt = DAILY_FACT_VALIDATION_PROMPT.format(
        active_facts_text=_format_facts_for_prompt(active_facts),
        conversation_text=_format_conversation_for_prompt(conversation),
        registry_summary=registry_summary,
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1,
        )
        return _extract_json(response)
    except Exception as e:
        logger.error(f"LLM daily fact validation failed: {e}")
        return {"deprecate": [], "flag_for_review": []}


async def llm_candidate_key_review(
    client: TrackedAnthropicClient,
    candidates: List[Dict[str, Any]],
    registry_summary: str,
) -> Dict[str, Any]:
    """
    Call LLM to review candidate keys and classify them.

    Returns dict with 'alias_mappings', 'proposed_new_keys', and 'reject' lists.
    """
    prompt = CANDIDATE_KEY_REVIEW_PROMPT.format(
        candidates_text=_format_candidates_for_prompt(candidates),
        registry_summary=registry_summary,
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1,
        )
        return _extract_json(response)
    except Exception as e:
        logger.error(f"LLM candidate key review failed: {e}")
        return {"alias_mappings": [], "proposed_new_keys": [], "reject": []}
