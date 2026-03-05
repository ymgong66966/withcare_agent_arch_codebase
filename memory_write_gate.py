"""
Memory Write Gate — risk-based write control for the Fact Store.

Core principle: risk_level is determined by the FactKey registry, NOT by the LLM.
LLM only provides risk_suggestion which can be overridden.

Risk rules:
  low:    Auto-write if confidence >= 0.6
  medium: Auto-write but mark unverified; re-confirm in next interaction
  high:   Requires source_type in (tool, document) with confidence >= 0.8
          OR explicit_user_confirmed=True. Otherwise → candidate only.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fact_store import FactStore, get_fact_store
from key_resolver import KeyRegistry, get_key_registry
from models.fact_models import (
    ExtractedFact,
    FactRecord,
    RiskLevel,
    WriteProposal,
)

logger = logging.getLogger(__name__)


def propose_updates(
    extracted_facts: List[ExtractedFact],
    registry: Optional[KeyRegistry] = None,
) -> WriteProposal:
    """
    Classify extracted facts into auto_patch / needs_confirm / reject.

    Risk level comes from the FactKey registry, not from the LLM's suggestion.
    This is a synchronous, deterministic function — no LLM calls.
    """
    if registry is None:
        registry = get_key_registry()

    auto_patch: List[ExtractedFact] = []
    needs_confirm: List[ExtractedFact] = []
    reject: List[ExtractedFact] = []

    for fact in extracted_facts:
        # Registry risk overrides LLM suggestion
        risk = registry.get_risk(fact.fact_key)
        if risk is None:
            # Unknown key — conservative: treat as high risk
            risk = "high"

        if risk == "low":
            if fact.confidence >= 0.6:
                auto_patch.append(fact)
            else:
                needs_confirm.append(fact)

        elif risk == "medium":
            if fact.confidence >= 0.7:
                # Auto-write but will be stored as "unverified"
                auto_patch.append(fact)
            else:
                needs_confirm.append(fact)

        else:  # high
            if fact.source_type in ("tool", "document") and fact.confidence >= 0.8:
                auto_patch.append(fact)
            elif fact.explicit_user_confirmed:
                auto_patch.append(fact)
            else:
                needs_confirm.append(fact)

    return WriteProposal(
        auto_patch=auto_patch,
        needs_confirm=needs_confirm,
        reject=reject,
    )


async def commit_updates(
    user_id: str,
    request_id: str,
    facts_to_commit: List[ExtractedFact],
    justification: str,
    actor: str = "system",
    *,
    registry: Optional[KeyRegistry] = None,
    store: Optional[FactStore] = None,
    conflict_strategy: str = "auto",
) -> List[FactRecord]:
    """
    Write approved facts to the Fact Store with proper conflict handling
    and audit logging.

    For each fact:
    1. Determine verification_level based on source + risk
    2. Upsert to UserFactTable (handles version chaining internally)
    3. Write audit log to MemoryFactLogTable

    Returns the list of FactRecords that were written.
    """
    if registry is None:
        registry = get_key_registry()
    if store is None:
        store = get_fact_store()

    committed: List[FactRecord] = []

    for fact in facts_to_commit:
        risk = registry.get_risk(fact.fact_key) or "high"

        # Determine verification level
        if fact.explicit_user_confirmed:
            verification_level = "explicit_user_confirmed"
        elif fact.source_type in ("tool", "document"):
            verification_level = "tool_verified"
        else:
            verification_level = "unverified"

        # Determine initial status based on risk and verification
        if risk == "high" and verification_level == "unverified":
            status = "candidate"
        else:
            status = "active"

        # Determine conflict strategy for this fact
        if conflict_strategy == "auto":
            fact_conflict = "needs_confirm" if risk == "high" else "overwrite"
        else:
            fact_conflict = conflict_strategy

        try:
            record = await store.upsert_fact(
                user_id=user_id,
                entity_id=fact.entity_id,
                fact_key=fact.fact_key,
                new_value=fact.value,
                fact_label=fact.fact_label,
                value_type=fact.value_type,
                status=status,
                risk_level=risk,
                confidence=fact.confidence,
                source_type=fact.source_type,
                source_ref=fact.source_ref or request_id,
                evidence=fact.evidence,
                verification_level=verification_level,
                conflict_strategy=fact_conflict,
            )

            # Write audit log
            await store.write_fact_log(
                user_id=user_id,
                target_id=record.fact_id,
                patch={
                    "fact_key": fact.fact_key,
                    "entity_id": fact.entity_id,
                    "value": fact.value,
                    "status": status,
                    "verification_level": verification_level,
                },
                justification=justification,
                actor=actor,
                source_ref=fact.source_ref or request_id,
                result="applied",
            )

            committed.append(record)

        except Exception as e:
            logger.error(
                f"Failed to commit fact {fact.fact_key} for {fact.entity_id}: {e}"
            )
            # Log the failure
            await store.write_fact_log(
                user_id=user_id,
                target_id=fact.fact_key,
                patch={"fact_key": fact.fact_key, "value": fact.value},
                justification=justification,
                actor=actor,
                source_ref=fact.source_ref or request_id,
                result="rejected",
            )

    return committed


def build_confirmation_questions(
    needs_confirm: List[ExtractedFact],
    language: str = "en",
) -> List[Dict[str, str]]:
    """
    Generate human-readable confirmation questions for facts that need
    explicit user verification before they can be committed.

    Returns a list of dicts with fact_key, question, and evidence.
    """
    questions: List[Dict[str, str]] = []

    for fact in needs_confirm:
        value_str = str(fact.value) if fact.value is not None else "unknown"

        if language == "zh":
            question = (
                f"我想确认一下：{fact.fact_label or fact.fact_key} "
                f"是 \"{value_str}\" 吗？"
            )
        else:
            label = fact.fact_label or fact.fact_key.replace(".", " ").replace("_", " ")
            question = (
                f"I'd like to confirm: is the {label} \"{value_str}\"?"
            )

        questions.append({
            "fact_key": fact.fact_key,
            "entity_id": fact.entity_id,
            "question": question,
            "evidence": fact.evidence,
            "value": value_str,
        })

    return questions
