"""
Onboarding Fact Bridge — Segment onboarding data into structured fact keys.

Receives completed onboarding data (user info + care recipient info + mental
assessment + tree Q&A) and writes structured facts to WithCare_UserFactTable.

Three-phase approach:
  Phase 1: Deterministic field mapping for known JSON fields (care recipient profile)
  Phase 2: Tree Q&A parsing — extract structured facts from decision tree answers
  Phase 3: Assessment & tasks — burnout scores, recommended tasks, completion
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fact_store import get_fact_store

_logger = logging.getLogger(__name__)

# ── Deterministic field mappings ─────────────────────────────────────
# Each entry: source_field → (fact_key, value_type, transform_fn_or_None)

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

_SKIP_FOR_FULLNAME = {"firstName", "lastName", "legalName"}

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
    "caregiver.care_duration": "low",
    "onboarding.recommended_tasks": "low",
    "onboarding.completed_at": "low",
    "onboarding.initial_request": "low",
    "insurance.medicare_eligible": "medium",
    "insurance.medicare_enrolled": "medium",
    "insurance.medicaid_eligible": "medium",
    "insurance.medicaid_enrolled": "medium",
    "benefits.veteran_accessing": "medium",
    "benefits.veteran_wants_help": "low",
    "living.residence_type": "low",
    "living.lives_alone": "low",
    "living.receives_inhome_care": "low",
    "living.needs_more_support": "low",
    "legal.has_attorney": "low",
    "legal.has_trust_or_will": "low",
    "legal.has_poa_or_directive": "low",
    "legal.wants_doc_help": "low",
    "medical.hospitalized_last_30_days": "medium",
    "medical.er_visit_last_3_months": "medium",
    "medical.needs_end_of_life_care": "medium",
    "care_plan.pending_tasks": "low",
}


# ── Tree Q&A pattern matchers ────────────────────────────────────────
# Each entry: (question_pattern_regex, fact_key, value_type, answer_transform)
# answer_transform: fn(answer_str) -> value or None to skip
# Patterns are matched case-insensitively against the AI question text.

def _yes_no(answer: str) -> Optional[str]:
    """Normalize yes/no/sometimes answers."""
    a = answer.strip().lower()
    if a in ("yes", "true"):
        return "yes"
    if a in ("no", "false"):
        return "no"
    if "not sure" in a or "don't know" in a:
        return "unsure"
    return a or None


def _yes_bool(answer: str) -> Optional[bool]:
    a = answer.strip().lower()
    if a in ("yes", "true"):
        return True
    if a in ("no", "false"):
        return False
    return None


TREE_QA_MATCHERS: List[Tuple[str, str, str, Any]] = [
    # ── Medicare ──
    (r"eligible for Medicare", "insurance.medicare_eligible", "enum", _yes_no),
    (r"enrolled in Medicare", "insurance.medicare_enrolled", "enum", _yes_no),

    # ── Medicaid ──
    (r"eligible for Medicaid", "insurance.medicaid_eligible", "enum", _yes_no),
    (r"enrolled in Medicaid", "insurance.medicaid_enrolled", "enum", _yes_no),

    # ── Veteran ──
    (r"accessing veterans benefits|accessing veteran", "benefits.veteran_accessing", "enum", _yes_no),
    (r"help exploring these benefits|help exploring.*veteran", "benefits.veteran_wants_help", "enum", _yes_no),

    # ── Living situation ──
    (r"private residence.*care facility|live in a private residence", "living.residence_type", "string",
     lambda a: "private_residence" if "private" in a.lower() or "home" in a.lower() or "apartment" in a.lower()
     else "care_facility" if "facility" in a.lower() or "assisted" in a.lower() or "nursing" in a.lower()
     else a.strip().lower() or None),
    (r"live in this home alone or with others|alone or with others", "living.lives_alone", "enum",
     lambda a: "alone" if "alone" in a.lower() else "with_others" if "other" in a.lower() else a.strip().lower() or None),
    (r"receive in-home care", "living.receives_inhome_care", "enum", _yes_no),
    (r"need more support.*care.*home|need more support", "living.needs_more_support", "enum", _yes_no),

    # ── Legal documents ──
    (r"attorney.*estate planning|have an attorney", "legal.has_attorney", "enum", _yes_no),
    (r"completed a trust or.*will|trust or a will", "legal.has_trust_or_will", "enum", _yes_no),
    (r"advanced directive.*power of attorney|directive and power", "legal.has_poa_or_directive", "enum", _yes_no),
    (r"support.*organizing.*documents|help.*organizing", "legal.wants_doc_help", "enum", _yes_no),

    # ── Hospitalization ──
    (r"hospitalized in the last 30 days|hospitalized.*30 days", "medical.hospitalized_last_30_days", "enum", _yes_no),
    (r"visited the ER.*last 3 months|ER.*3 months", "medical.er_visit_last_3_months", "enum", _yes_no),

    # ── End of life ──
    (r"end-of-life care|end of life", "medical.needs_end_of_life_care", "enum", _yes_no),

    # ── Care duration ──
    (r"how long have you been providing care", "caregiver.care_duration", "string", lambda a: a.strip() or None),

    # ── Initial request / immediate need ──
    (r"anything specific about.*care.*help|need help figuring out", "onboarding.initial_request", "string",
     lambda a: a.strip() if a.strip().lower() not in ("no", "not really", "nothing", "none", "nope") else None),
]


def _resolve_entity_id(relationship: str, first_name: str = "") -> str:
    """Build entity_id for a care recipient."""
    rel = (relationship or "").lower().strip()
    if rel:
        return f"care_recipient:{rel}"
    if first_name:
        return f"care_recipient:{first_name.lower().strip()}"
    return "care_recipient:unknown"


def _burnout_level(score: int) -> str:
    if score < 5:
        return "low"
    elif score < 11:
        return "moderate"
    return "high"


def _extract_tree_facts(
    qa_pairs: List[Dict[str, str]],
) -> Dict[str, Tuple[Any, str]]:
    """Extract structured facts from tree Q&A pairs.

    Returns: {fact_key: (value, value_type)} — only non-None values.
    """
    extracted = {}
    for pair in qa_pairs:
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        if not question or not answer:
            continue

        for pattern, fact_key, value_type, transform in TREE_QA_MATCHERS:
            if fact_key in extracted:
                continue  # first match wins
            if re.search(pattern, question, re.IGNORECASE):
                value = transform(answer)
                if value is not None:
                    extracted[fact_key] = (value, value_type)
                break

    return extracted


async def ingest_onboarding_data(
    *,
    user_id: str,
    user_data: Optional[Dict[str, Any]] = None,
    care_recipients: Optional[List[Dict[str, Any]]] = None,
    assessment_score: Optional[int] = None,
    assessment_answers: Optional[List[Dict[str, Any]]] = None,
    tasks: Optional[List[str]] = None,
    tree_qa_pairs: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Segment onboarding data and write structured facts.

    Args:
        user_id: The caregiver's user ID.
        user_data: User's own profile data.
        care_recipients: [{relationship, data: {...}}]
        assessment_score: Caregiver burnout score (0-12), None if skipped.
        assessment_answers: [{question, answer}] for burnout detail.
        tasks: Accumulated onboarding task recommendations.
        tree_qa_pairs: [{question, answer}] from decision tree conversations.

    Returns:
        Summary dict with counts of facts written per entity.
    """
    store = get_fact_store()
    if not store:
        _logger.warning("FactStore not available, skipping onboarding ingest")
        return {"error": "FactStore not available"}

    result = {"user_id": user_id, "entities": {}, "total_facts_written": 0}

    # Resolve entity_id from first care recipient
    entity_id = "care_recipient:unknown"
    if care_recipients:
        first = care_recipients[0]
        relationship = first.get("relationship", "")
        data = first.get("data", {})
        first_name = data.get("firstName", "") if data else ""
        entity_id = _resolve_entity_id(relationship, first_name)

    # ── 1. Care recipient profile facts ───────────────────────────
    for recipient in (care_recipients or []):
        relationship = recipient.get("relationship", "")
        data = recipient.get("data", {})
        if not data:
            continue

        first_name = data.get("firstName", "")
        eid = _resolve_entity_id(relationship, first_name)
        facts_written = 0
        written_keys = set()

        for source_field, (fact_key, value_type, transform) in RECIPIENT_FIELD_MAP.items():
            if fact_key in written_keys:
                continue
            try:
                value = transform(data) if transform is not None else data.get(source_field)
                if value is None or value == "":
                    continue
                risk = _RISK_MAP.get(fact_key, "medium")
                await store.upsert_fact(
                    user_id=user_id, entity_id=eid, fact_key=fact_key,
                    new_value=value,
                    fact_label=fact_key.replace(".", " ").replace("_", " ").title(),
                    value_type=value_type, risk_level=risk, confidence=0.9,
                    source_type="document", source_ref="onboarding_v1",
                    evidence=f"From onboarding: {source_field}",
                    verification_level="explicit_user_confirmed",
                    conflict_strategy="overwrite",
                )
                written_keys.add(fact_key)
                facts_written += 1
            except Exception as e:
                _logger.warning(f"Failed to write fact {fact_key} for {eid}: {e}")

        # Extra fields as raw
        mapped_sources = set(RECIPIENT_FIELD_MAP.keys()) | _SKIP_FOR_FULLNAME
        for field, value in data.items():
            if field in mapped_sources or value is None or value == "":
                continue
            raw_key = f"onboarding.raw_{field.lower()}"
            try:
                await store.upsert_fact(
                    user_id=user_id, entity_id=eid, fact_key=raw_key,
                    new_value=value,
                    value_type="string" if isinstance(value, str) else "object",
                    risk_level="low", confidence=0.8,
                    source_type="document", source_ref="onboarding_v1",
                    evidence=f"Extra onboarding field: {field}",
                    verification_level="explicit_user_confirmed",
                    conflict_strategy="overwrite",
                )
                facts_written += 1
            except Exception as e:
                _logger.warning(f"Failed to write raw field {field} for {eid}: {e}")

        result["entities"][eid] = facts_written
        result["total_facts_written"] += facts_written

    # ── 2. Tree Q&A parsed facts ──────────────────────────────────
    if tree_qa_pairs:
        tree_facts = _extract_tree_facts(tree_qa_pairs)
        _logger.info(f"Extracted {len(tree_facts)} facts from {len(tree_qa_pairs)} tree Q&A pairs")

        for fact_key, (value, value_type) in tree_facts.items():
            # Decide entity: caregiver-related facts go to user:self,
            # care-recipient facts go to the recipient entity
            if fact_key.startswith(("caregiver.", "onboarding.")):
                eid = "user:self"
            else:
                eid = entity_id  # care_recipient entity

            try:
                risk = _RISK_MAP.get(fact_key, "low")
                await store.upsert_fact(
                    user_id=user_id, entity_id=eid, fact_key=fact_key,
                    new_value=value,
                    fact_label=fact_key.replace(".", " ").replace("_", " ").title(),
                    value_type=value_type, risk_level=risk, confidence=0.9,
                    source_type="user", source_ref="onboarding_v1",
                    evidence=f"From onboarding tree Q&A",
                    verification_level="explicit_user_confirmed",
                    conflict_strategy="overwrite",
                )
                result["entities"][eid] = result["entities"].get(eid, 0) + 1
                result["total_facts_written"] += 1
            except Exception as e:
                _logger.warning(f"Failed to write tree fact {fact_key} for {eid}: {e}")

    # ── 3. Caregiver burnout assessment ───────────────────────────
    if assessment_score is not None:
        caregiver_facts = 0
        eid = "user:self"

        try:
            await store.upsert_fact(
                user_id=user_id, entity_id=eid,
                fact_key="caregiver.burnout_score", new_value=assessment_score,
                value_type="number", risk_level="medium", confidence=0.95,
                source_type="user", source_ref="onboarding_v1",
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
                user_id=user_id, entity_id=eid,
                fact_key="caregiver.burnout_level", new_value=level,
                value_type="enum", risk_level="medium", confidence=0.95,
                source_type="user", source_ref="onboarding_v1",
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
                    user_id=user_id, entity_id=eid,
                    fact_key="caregiver.burnout_assessment_detail",
                    new_value=assessment_answers, value_type="object",
                    risk_level="medium", confidence=0.95,
                    source_type="user", source_ref="onboarding_v1",
                    evidence="Individual burnout screening Q&A responses",
                    verification_level="explicit_user_confirmed",
                    conflict_strategy="overwrite",
                )
                caregiver_facts += 1
            except Exception as e:
                _logger.warning(f"Failed to write burnout_assessment_detail: {e}")

        result["entities"][eid] = result["entities"].get(eid, 0) + caregiver_facts
        result["total_facts_written"] += caregiver_facts

    # ── 4. Onboarding tasks ───────────────────────────────────────
    if tasks:
        # Store under user:self for caregiver context
        try:
            await store.upsert_fact(
                user_id=user_id, entity_id="user:self",
                fact_key="onboarding.recommended_tasks", new_value=tasks,
                value_type="list", risk_level="low", confidence=0.95,
                source_type="document", source_ref="onboarding_v1",
                evidence=f"Onboarding recommended {len(tasks)} tasks",
                verification_level="explicit_user_confirmed",
                conflict_strategy="overwrite",
            )
            result["entities"]["user:self"] = result["entities"].get("user:self", 0) + 1
            result["total_facts_written"] += 1
        except Exception as e:
            _logger.warning(f"Failed to write onboarding tasks: {e}")

        # Also store under the care recipient so tasks appear when querying
        # "what needs to be done for my dad?"
        try:
            await store.upsert_fact(
                user_id=user_id, entity_id=entity_id,
                fact_key="care_plan.pending_tasks", new_value=tasks,
                value_type="list", risk_level="low", confidence=0.95,
                source_type="document", source_ref="onboarding_v1",
                evidence=f"Onboarding identified {len(tasks)} care plan tasks",
                verification_level="explicit_user_confirmed",
                conflict_strategy="overwrite",
            )
            result["entities"][entity_id] = result["entities"].get(entity_id, 0) + 1
            result["total_facts_written"] += 1
        except Exception as e:
            _logger.warning(f"Failed to write care_plan.pending_tasks: {e}")

    # ── 5. Onboarding completion timestamp ────────────────────────
    try:
        await store.upsert_fact(
            user_id=user_id, entity_id="user:self",
            fact_key="onboarding.completed_at",
            new_value=datetime.utcnow().isoformat(),
            value_type="date", risk_level="low", confidence=1.0,
            source_type="document", source_ref="onboarding_v1",
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
