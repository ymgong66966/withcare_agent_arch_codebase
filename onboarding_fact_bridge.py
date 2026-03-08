"""
Onboarding Fact Bridge — Segment onboarding data into structured fact keys.

Receives completed onboarding data (user info + care recipient info + mental
assessment) and writes structured facts to WithCare_UserFactTable.

Two-phase approach:
  Phase 1: Deterministic field mapping for known JSON fields
  Phase 2: (future) LLM-assisted extraction for tree Q&A data
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fact_store import get_fact_store

_logger = logging.getLogger(__name__)

# ── Deterministic field mappings ─────────────────────────────────────
# Each entry: source_field → (fact_key, value_type, transform_fn_or_None)
# transform_fn receives the full recipient dict (for multi-field transforms)
# or None for direct value copy.

RECIPIENT_FIELD_MAP: Dict[str, Tuple[str, str, Any]] = {
    "firstName": (
        "identity.full_name", "string",
        lambda r: f"{r.get('firstName', '')} {r.get('lastName', '')}".strip() or None,
    ),
    "dateOfBirth": ("identity.dob", "date", None),
    "gender": ("identity.gender", "enum", lambda r: r.get("gender", "").lower() or None),
    "address": ("identity.address", "object", None),
    "veteranStatus": ("identity.veteran_status", "enum", None),
    "pronouns": ("identity.pronouns", "string", None),
    "dependentStatus": ("identity.dependent_status", "string", None),
    "relationship": ("identity.relationship_to_user", "string", lambda r: r.get("relationship", "").lower() or None),
    "legalName": ("identity.full_name", "string", None),  # fallback if firstName missing
}

# Fields to skip in firstName-based full_name (handled by the lambda above)
_SKIP_FOR_FULLNAME = {"firstName", "lastName", "legalName"}

# Risk levels per fact key (from registry)
_RISK_MAP = {
    "identity.full_name": "medium",
    "identity.dob": "high",
    "identity.gender": "medium",
    "identity.address": "high",
    "identity.veteran_status": "medium",
    "identity.pronouns": "low",
    "identity.dependent_status": "low",
    "identity.relationship_to_user": "low",
    "caregiver.burnout_score": "medium",
    "caregiver.burnout_level": "medium",
    "caregiver.burnout_assessment_detail": "medium",
    "onboarding.recommended_tasks": "low",
    "onboarding.completed_at": "low",
}


def _resolve_entity_id(relationship: str, first_name: str = "") -> str:
    """Build entity_id for a care recipient."""
    rel = (relationship or "").lower().strip()
    if rel:
        return f"care_recipient:{rel}"
    if first_name:
        return f"care_recipient:{first_name.lower().strip()}"
    return "care_recipient:unknown"


def _burnout_level(score: int) -> str:
    """Derive burnout level from assessment score (0-12)."""
    if score < 5:
        return "low"
    elif score < 11:
        return "moderate"
    return "high"


async def ingest_onboarding_data(
    *,
    user_id: str,
    user_data: Optional[Dict[str, Any]] = None,
    care_recipients: Optional[List[Dict[str, Any]]] = None,
    assessment_score: Optional[int] = None,
    assessment_answers: Optional[List[Dict[str, Any]]] = None,
    tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Segment onboarding data and write structured facts.

    Args:
        user_id: The caregiver's user ID.
        user_data: User's own profile data (filtered, excluding careRecipients).
        care_recipients: List of dicts, each with at least:
            - relationship: str
            - data: dict (the recipient's profile fields)
        assessment_score: Caregiver burnout score (0-12), None if skipped.
        assessment_answers: List of {question, answer} dicts for burnout detail.
        tasks: Accumulated onboarding task recommendations.

    Returns:
        Summary dict with counts of facts written per entity.
    """
    store = get_fact_store()
    if not store:
        _logger.warning("FactStore not available, skipping onboarding ingest")
        return {"error": "FactStore not available"}

    result = {"user_id": user_id, "entities": {}, "total_facts_written": 0}

    # ── 1. Care recipient facts ──────────────────────────────────
    for recipient in (care_recipients or []):
        relationship = recipient.get("relationship", "")
        data = recipient.get("data", {})
        if not data:
            continue

        first_name = data.get("firstName", "")
        entity_id = _resolve_entity_id(relationship, first_name)
        facts_written = 0

        # Track which fact_keys we've written to avoid duplicates
        written_keys = set()

        for source_field, (fact_key, value_type, transform) in RECIPIENT_FIELD_MAP.items():
            if fact_key in written_keys:
                continue

            try:
                if transform is not None:
                    value = transform(data)
                else:
                    value = data.get(source_field)

                if value is None or value == "":
                    continue

                risk = _RISK_MAP.get(fact_key, "medium")
                await store.upsert_fact(
                    user_id=user_id,
                    entity_id=entity_id,
                    fact_key=fact_key,
                    new_value=value,
                    fact_label=fact_key.replace(".", " ").replace("_", " ").title(),
                    value_type=value_type,
                    risk_level=risk,
                    confidence=0.9,
                    source_type="document",
                    source_ref="onboarding_v1",
                    evidence=f"From onboarding: {source_field}",
                    verification_level="explicit_user_confirmed",
                    conflict_strategy="overwrite",
                )
                written_keys.add(fact_key)
                facts_written += 1
            except Exception as e:
                _logger.warning(f"Failed to write fact {fact_key} for {entity_id}: {e}")

        # Store any extra fields not in the map as raw onboarding data
        mapped_sources = set(RECIPIENT_FIELD_MAP.keys()) | _SKIP_FOR_FULLNAME
        for field, value in data.items():
            if field in mapped_sources or value is None or value == "":
                continue
            raw_key = f"onboarding.raw_{field.lower()}"
            try:
                await store.upsert_fact(
                    user_id=user_id,
                    entity_id=entity_id,
                    fact_key=raw_key,
                    new_value=value,
                    value_type="string" if isinstance(value, str) else "object",
                    risk_level="low",
                    confidence=0.8,
                    source_type="document",
                    source_ref="onboarding_v1",
                    evidence=f"Extra onboarding field: {field}",
                    verification_level="explicit_user_confirmed",
                    conflict_strategy="overwrite",
                )
                facts_written += 1
            except Exception as e:
                _logger.warning(f"Failed to write raw field {field} for {entity_id}: {e}")

        result["entities"][entity_id] = facts_written
        result["total_facts_written"] += facts_written

    # ── 2. Caregiver burnout assessment ──────────────────────────
    if assessment_score is not None:
        caregiver_facts = 0
        entity_id = "user:self"

        try:
            await store.upsert_fact(
                user_id=user_id,
                entity_id=entity_id,
                fact_key="caregiver.burnout_score",
                new_value=assessment_score,
                value_type="number",
                risk_level="medium",
                confidence=0.95,
                source_type="user",
                source_ref="onboarding_v1",
                evidence=f"Onboarding burnout assessment: score {assessment_score}/12",
                verification_level="explicit_user_confirmed",
                conflict_strategy="overwrite",
            )
            caregiver_facts += 1
        except Exception as e:
            _logger.warning(f"Failed to write burnout_score: {e}")

        try:
            level = _burnout_level(assessment_score)
            await store.upsert_fact(
                user_id=user_id,
                entity_id=entity_id,
                fact_key="caregiver.burnout_level",
                new_value=level,
                value_type="enum",
                risk_level="medium",
                confidence=0.95,
                source_type="user",
                source_ref="onboarding_v1",
                evidence=f"Derived from burnout score {assessment_score}: {level}",
                verification_level="explicit_user_confirmed",
                conflict_strategy="overwrite",
            )
            caregiver_facts += 1
        except Exception as e:
            _logger.warning(f"Failed to write burnout_level: {e}")

        if assessment_answers:
            try:
                await store.upsert_fact(
                    user_id=user_id,
                    entity_id=entity_id,
                    fact_key="caregiver.burnout_assessment_detail",
                    new_value=assessment_answers,
                    value_type="object",
                    risk_level="medium",
                    confidence=0.95,
                    source_type="user",
                    source_ref="onboarding_v1",
                    evidence="Individual burnout screening Q&A responses",
                    verification_level="explicit_user_confirmed",
                    conflict_strategy="overwrite",
                )
                caregiver_facts += 1
            except Exception as e:
                _logger.warning(f"Failed to write burnout_assessment_detail: {e}")

        result["entities"][entity_id] = result["entities"].get(entity_id, 0) + caregiver_facts
        result["total_facts_written"] += caregiver_facts

    # ── 3. Onboarding tasks ──────────────────────────────────────
    if tasks:
        try:
            await store.upsert_fact(
                user_id=user_id,
                entity_id="user:self",
                fact_key="onboarding.recommended_tasks",
                new_value=tasks,
                value_type="list",
                risk_level="low",
                confidence=0.95,
                source_type="document",
                source_ref="onboarding_v1",
                evidence=f"Onboarding recommended {len(tasks)} tasks",
                verification_level="explicit_user_confirmed",
                conflict_strategy="overwrite",
            )
            result["entities"]["user:self"] = result["entities"].get("user:self", 0) + 1
            result["total_facts_written"] += 1
        except Exception as e:
            _logger.warning(f"Failed to write onboarding tasks: {e}")

    # ── 4. Onboarding completion timestamp ────────────────────────
    try:
        await store.upsert_fact(
            user_id=user_id,
            entity_id="user:self",
            fact_key="onboarding.completed_at",
            new_value=datetime.utcnow().isoformat(),
            value_type="date",
            risk_level="low",
            confidence=1.0,
            source_type="document",
            source_ref="onboarding_v1",
            evidence="Onboarding process completed",
            verification_level="explicit_user_confirmed",
            conflict_strategy="overwrite",
        )
        result["entities"]["user:self"] = result["entities"].get("user:self", 0) + 1
        result["total_facts_written"] += 1
    except Exception as e:
        _logger.warning(f"Failed to write onboarding timestamp: {e}")

    _logger.info(
        f"Onboarding ingest for user {user_id}: "
        f"{result['total_facts_written']} facts across {len(result['entities'])} entities"
    )
    return result
