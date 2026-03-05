"""
Context Bundle — structured context assembly for LLM calls.

Replaces raw conversation dumping with a minimal, focused "work desk."
Each turn, the system assembles a ContextBundle containing:
  1. Active request block (if any)
  2. Relevant profile facts (only fields relevant to current intent)
  3. Relevant history (recency + relevance merged)
  4. Recent events (episodic)
  5. Safety notes (stale or unverified high-risk facts)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from event_store import EventStore, get_event_store
from fact_store import FactStore, get_fact_store
from key_resolver import KeyRegistry, get_key_registry
from models.fact_models import (
    ActiveRequestBlock,
    ContextBundle,
    EventItem,
    FactRecord,
    RequestSummaryItem,
    SafetyNote,
    SlotWithFactRef,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_needed_fact_keys(
    request_type: str,
    taxonomy_path: str = "configs/request_type_taxonomy.yaml",
) -> List[str]:
    """
    Look up which FactKey namespaces are commonly needed for a given
    request_type, based on the taxonomy config.

    Returns a list of FactKey prefixes (namespaces) like ["insurance", "identity"].
    """
    try:
        import yaml
        import os

        full_path = os.path.join(os.path.dirname(__file__), taxonomy_path)
        with open(full_path, "r", encoding="utf-8") as f:
            entries = yaml.safe_load(f) or []

        for entry in entries:
            if entry.get("type") == request_type:
                # related_fact_keys has entries like "insurance.*", "identity.address"
                raw_keys = entry.get("related_fact_keys", [])
                # Extract namespace prefixes
                return [k.replace(".*", "") for k in raw_keys]

        return []

    except Exception as e:
        logger.warning(f"Failed to load taxonomy for {request_type}: {e}")
        return []


def _compute_safety_notes(
    facts: List[FactRecord],
    stale_threshold_days: int = 90,
) -> List[SafetyNote]:
    """
    Flag high-risk facts that are unverified or stale (not re-confirmed
    within threshold).
    """
    notes: List[SafetyNote] = []
    now = datetime.utcnow()

    for fact in facts:
        if fact.risk_level != "high":
            continue

        is_unverified = fact.verification_level == "unverified"
        is_stale = False
        if fact.last_verified_at:
            days_since = (now - fact.last_verified_at).days
            is_stale = days_since > stale_threshold_days
        elif fact.created_at:
            days_since = (now - fact.created_at).days
            is_stale = days_since > stale_threshold_days

        if is_unverified or is_stale:
            reason_parts = []
            if is_unverified:
                reason_parts.append("unverified")
            if is_stale:
                reason_parts.append(
                    f"not confirmed in {stale_threshold_days}+ days"
                )

            notes.append(SafetyNote(
                topic=fact.fact_key.split(".")[0],
                fact_key=fact.fact_key,
                current_value=fact.fact_value,
                confidence=fact.confidence,
                last_verified_at=(
                    fact.last_verified_at.isoformat()
                    if fact.last_verified_at else None
                ),
                rule=(
                    f"High-risk fact ({', '.join(reason_parts)}): "
                    f"confirm before using in submissions or decisions"
                ),
            ))

    return notes


def _format_active_request(
    request_dict: Dict[str, Any],
    profile_facts: List[FactRecord],
) -> ActiveRequestBlock:
    """Build the ActiveRequestBlock from a request dict and available facts."""
    # Build slot map with fact references
    slots: Dict[str, SlotWithFactRef] = {}
    slot_refs = request_dict.get("slot_refs", {})

    # Check existing slots from info_collection_state
    ic_state = request_dict.get("info_collection_state") or {}
    collected = ic_state.get("summary_of_collected_info", "")

    for key, fact_id in slot_refs.items():
        # Find matching fact
        matching = [f for f in profile_facts if f.fact_id == fact_id]
        if matching:
            fact = matching[0]
            slots[key] = SlotWithFactRef(
                status="filled",
                value=fact.fact_value,
                fact_ref=fact.fact_id,
                risk=fact.risk_level,
                verified=fact.verification_level != "unverified",
            )
        else:
            slots[key] = SlotWithFactRef(
                status="missing",
                fact_ref=fact_id,
            )

    return ActiveRequestBlock(
        request_id=request_dict.get("request_id", ""),
        request_type=request_dict.get("request_type", ""),
        title=request_dict.get("title", request_dict.get("name", "")),
        status=request_dict.get("status", ""),
        subject_entity_id=request_dict.get("subject_entity_id", ""),
        slots=slots,
        next_steps=[],  # Populated by the caller if needed
        summary_current=request_dict.get("summary_current", collected),
    )


# ─────────────────────────────────────────────────────────────
# Main assembly function
# ─────────────────────────────────────────────────────────────

async def get_context_bundle(
    user_id: str,
    request_dict: Optional[Dict[str, Any]] = None,
    message: str = "",
    k: int = 5,
    *,
    fact_store: Optional[FactStore] = None,
    event_store: Optional[EventStore] = None,
) -> ContextBundle:
    """
    Assemble a ContextBundle for the current turn.

    This is the main entry point that other modules (graph nodes, MCP server)
    call to get structured context for LLM calls.

    Args:
        user_id: The user's ID
        request_dict: The active request's dict (from RequestManager)
        message: The user's current message (for relevance-based retrieval)
        k: Max items for history and events
        fact_store: Optional FactStore instance (uses global if None)
        event_store: Optional EventStore instance (uses global if None)

    Returns:
        ContextBundle with all blocks populated (or empty if no data).
    """
    if fact_store is None:
        fact_store = get_fact_store()
    if event_store is None:
        event_store = get_event_store()

    now = datetime.utcnow()

    # 1. Determine entity and needed fact keys
    entity_id = ""
    needed_namespaces: List[str] = []
    if request_dict:
        entity_id = request_dict.get("subject_entity_id") or ""
        request_type = request_dict.get("request_type") or ""
        if request_type:
            needed_namespaces = _get_needed_fact_keys(request_type)

    # 2. Pull active facts for the entity
    profile_facts: List[FactRecord] = []
    profile_facts_dict: Dict[str, Any] = {}
    if entity_id:
        all_facts = await fact_store.get_active_facts(
            user_id=user_id,
            entity_id=entity_id,
        )

        # Include all active facts, but prioritize namespace-relevant ones first
        if needed_namespaces:
            relevant = [
                f for f in all_facts
                if any(f.fact_key.startswith(ns) for ns in needed_namespaces)
            ]
            other = [f for f in all_facts if f not in relevant]
            profile_facts = relevant + other
        else:
            profile_facts = all_facts

        # Build dict representation for the bundle
        for fact in profile_facts:
            profile_facts_dict[fact.fact_key] = {
                "value": fact.fact_value,
                "confidence": fact.confidence,
                "last_verified_at": (
                    fact.last_verified_at.isoformat()
                    if fact.last_verified_at else None
                ),
                "verification_level": fact.verification_level,
                "source_type": fact.source_type,
            }

    # 3. Build active request block
    active_request_block = None
    if request_dict:
        active_request_block = _format_active_request(request_dict, profile_facts)

    # 4. Recent events from EventStore
    raw_events = await event_store.get_recent_events(user_id=user_id, limit=k)
    recent_events = [
        EventItem(
            when=evt.timestamp.isoformat(),
            event_type=evt.event_type,
            content=evt.content[:300],
            request_id=evt.request_id,
        )
        for evt in raw_events
    ]

    # 5. Relevant history — currently from recency only (vector search in Sprint 4)
    # TODO: Add Milvus vector search and merge with recency
    relevant_history: List[RequestSummaryItem] = []

    # 6. Safety notes
    safety_notes = _compute_safety_notes(profile_facts)

    return ContextBundle(
        user_id=user_id,
        now=now,
        active_request=active_request_block,
        profile_facts=profile_facts_dict,
        relevant_history=relevant_history,
        recent_events=recent_events,
        safety_notes=safety_notes,
    )


def bundle_to_prompt_block(bundle: ContextBundle) -> str:
    """
    Render a ContextBundle as a text block suitable for inclusion in an
    LLM system/user prompt.

    This is a convenience function for graph nodes that want to inject
    memory context into their existing prompt templates.
    """
    sections: List[str] = []

    # Active request
    if bundle.active_request:
        ar = bundle.active_request
        sections.append(
            f"## Active Request\n"
            f"- Type: {ar.request_type}\n"
            f"- Title: {ar.title}\n"
            f"- Status: {ar.status}\n"
            f"- Entity: {ar.subject_entity_id}\n"
            f"- Summary: {ar.summary_current}"
        )

    # Profile facts
    if bundle.profile_facts:
        lines = []
        for key, info in bundle.profile_facts.items():
            val = info.get("value", "unknown")
            verified = info.get("verification_level", "unverified")
            lines.append(f"  - {key}: {val} [{verified}]")
        sections.append("## Known Facts\n" + "\n".join(lines))

    # Relevant history
    if bundle.relevant_history:
        lines = []
        for item in bundle.relevant_history:
            lines.append(
                f"  - [{item.when}] {item.title} ({item.status}): {item.summary}"
            )
        sections.append("## Relevant History\n" + "\n".join(lines))

    # Recent events
    if bundle.recent_events:
        lines = []
        for evt in bundle.recent_events:
            lines.append(f"  - [{evt.when}] ({evt.event_type}): {evt.content}")
        sections.append("## Recent Events\n" + "\n".join(lines))

    # Safety notes
    if bundle.safety_notes:
        lines = []
        for note in bundle.safety_notes:
            lines.append(f"  - {note.fact_key}: {note.rule}")
        sections.append("## Safety Notes\n" + "\n".join(lines))

    return "\n\n".join(sections) if sections else "(No memory context available.)"
