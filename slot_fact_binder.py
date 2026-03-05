"""
Slot→Fact Binder — extracts structured facts from info_collection summaries
and commits them to the UserFactTable via the write gate.

Pipeline:
  1. llm_extract_facts()       → List[ExtractedFact]  (LLM call)
  2. KeyResolver.resolve()     → canonical key or candidate pool
  2b. llm_reconcile_facts()   → reconcile new vs existing (deprecate/discard/write)
  3. propose_updates()         → WriteProposal (risk classification)
  4. commit_updates()          → writes to DDB + audit log
  4b. Namespace-based deprecation (safety net)
  5. EventStore.add_event()    → store needs_confirm as memory_candidate
  6. Return slot_refs          → {fact_key: fact_id}

This module is called from graph.py after llm_info_collection_summarize()
returns. It is non-blocking on failure — returns {} on any exception.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from candidate_key_pool import CandidateKeyPoolStore, get_candidate_key_pool_store
from event_store import EventStore, get_event_store
from fact_store import get_fact_store
from key_resolver import KeyResolver, get_key_resolver
from memory_write_gate import commit_updates, propose_updates
from models.fact_models import ExtractedFact
from prompts import llm_extract_facts, llm_reconcile_facts

logger = logging.getLogger(__name__)


async def bind_facts_from_summary(
    user_id: str,
    request_id: str,
    entity_id: str,
    request_type: str,
    updated_summary: str,
    key_info_needed: List[Any],
    conversation_history: List[Dict[str, Any]],
    client: Any,
    existing_slot_refs: Optional[Dict[str, str]] = None,
    existing_profile_facts: Optional[Dict[str, Any]] = None,
    disputed_fact_keys: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Extract structured facts from an info_collection summary, resolve keys,
    run through the write gate, and commit to DDB.

    Returns:
        slot_refs: mapping of fact_key → fact_id for committed facts.
        Returns {} on any failure (non-blocking).
    """
    try:
        return await _bind_facts_impl(
            user_id=user_id,
            request_id=request_id,
            entity_id=entity_id,
            request_type=request_type,
            updated_summary=updated_summary,
            key_info_needed=key_info_needed,
            conversation_history=conversation_history,
            client=client,
            existing_slot_refs=existing_slot_refs,
            existing_profile_facts=existing_profile_facts,
            disputed_fact_keys=disputed_fact_keys,
        )
    except Exception as e:
        logger.warning(f"Fact binding failed (non-blocking): {e}")
        return {}


async def _bind_facts_impl(
    user_id: str,
    request_id: str,
    entity_id: str,
    request_type: str,
    updated_summary: str,
    key_info_needed: List[Any],
    conversation_history: List[Dict[str, Any]],
    client: Any,
    existing_slot_refs: Optional[Dict[str, str]] = None,
    existing_profile_facts: Optional[Dict[str, Any]] = None,
    disputed_fact_keys: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Internal implementation — exceptions propagate to caller."""

    if not updated_summary or not updated_summary.strip():
        return {}

    # ── Step 1: LLM fact extraction ──────────────────────────
    already_extracted_keys = list((existing_slot_refs or {}).keys())
    raw_facts = await llm_extract_facts(
        entity_id=entity_id,
        request_type=request_type,
        updated_summary=updated_summary,
        key_info_needed=key_info_needed,
        conversation_history=conversation_history,
        client=client,
        already_extracted_keys=already_extracted_keys,
        existing_profile_facts=existing_profile_facts,
        disputed_fact_keys=disputed_fact_keys,
    )

    if not raw_facts:
        logger.debug("No facts extracted from summary")
        return {}

    # ── Step 2: Resolve keys (batch) ────────────────────────────
    resolver = get_key_resolver(llm_client=client)
    pool_store = get_candidate_key_pool_store()

    # Filter out facts with empty keys and build ExtractedFact list
    valid_raws = [r for r in raw_facts if r.get("fact_key")]
    extracted_facts_pre: List[ExtractedFact] = []
    for raw in valid_raws:
        extracted_facts_pre.append(ExtractedFact(
            entity_id=entity_id,
            fact_key=raw.get("fact_key", ""),
            fact_label=raw.get("fact_label", ""),
            value=raw.get("value"),
            value_type=raw.get("value_type", "string"),
            confidence=float(raw.get("confidence", 0.5)),
            source_type=raw.get("source_type", "user"),
            source_ref=request_id,
            evidence=raw.get("evidence", ""),
            # Facts from info_collection summaries are directly stated by
            # the user in conversation — treat as explicitly confirmed so
            # the write gate doesn't block high-risk keys.
            explicit_user_confirmed=True,
        ))

    # Batch resolve all facts in a single LLM call
    resolved_facts: List[ExtractedFact] = []
    try:
        batch_inputs = [
            {"fact_key": e.fact_key, "fact_label": e.fact_label, "evidence": e.evidence}
            for e in extracted_facts_pre
        ]
        batch_results = await resolver.resolve_batch(
            facts=batch_inputs,
            entity_id=entity_id,
            context={"request_type": request_type},
        )

        for extracted, result in zip(extracted_facts_pre, batch_results):
            if result.decision == "map" and result.canonical_key != "unknown":
                extracted.fact_key = result.canonical_key
            else:
                await _record_to_candidate_pool(
                    pool_store=pool_store,
                    extracted=extracted,
                    request_id=request_id,
                )
                extracted.confidence = min(extracted.confidence, 0.65)

            resolved_facts.append(extracted)

    except Exception as e:
        logger.warning(f"Batch key resolution failed: {e}, using raw keys")
        for extracted in extracted_facts_pre:
            extracted.confidence = min(extracted.confidence, 0.6)
            resolved_facts.append(extracted)

        resolved_facts.append(extracted)

    if not resolved_facts:
        return {}

    # ── Step 2b: Reconcile new facts against existing profile ──
    if existing_profile_facts:
        try:
            reconciliation_input = [
                {"fact_key": f.fact_key, "value": f.value}
                for f in resolved_facts
            ]
            reconciliation = await llm_reconcile_facts(
                existing_profile_facts=existing_profile_facts,
                resolved_facts=reconciliation_input,
                conversation_history=conversation_history,
                client=client,
            )

            if reconciliation is not None:
                # Execute deprecations on existing facts
                deprecate_keys = {
                    d["fact_key"]
                    for d in reconciliation.get("deprecate_existing", [])
                    if isinstance(d, dict) and d.get("fact_key")
                }
                if deprecate_keys:
                    store = get_fact_store()
                    for dep_key in deprecate_keys:
                        try:
                            old_facts = await store.get_active_facts(
                                user_id=user_id,
                                entity_id=entity_id,
                                fact_keys=[dep_key],
                            )
                            for of in old_facts:
                                await store.deprecate_fact(of)
                                logger.info(
                                    f"Reconciliation deprecated '{dep_key}={of.fact_value}'"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to deprecate {dep_key} during reconciliation: {e}"
                            )

                # Filter out discarded new facts
                discard_keys = {
                    d["fact_key"]
                    for d in reconciliation.get("discard_new", [])
                    if isinstance(d, dict) and d.get("fact_key")
                }
                if discard_keys:
                    before_count = len(resolved_facts)
                    resolved_facts = [
                        f for f in resolved_facts if f.fact_key not in discard_keys
                    ]
                    logger.info(
                        f"Reconciliation discarded {before_count - len(resolved_facts)} "
                        f"new facts: {discard_keys}"
                    )

                if not resolved_facts:
                    logger.info("All new facts discarded by reconciliation")
                    return {}
            else:
                logger.debug("Reconciliation returned None, skipping")

        except Exception as e:
            logger.warning(f"Fact reconciliation failed (non-blocking): {e}")

    # ── Step 3: Write gate classification ────────────────────
    proposal = propose_updates(resolved_facts)

    # ── Step 4: Commit auto_patch facts ──────────────────────
    committed = []
    if proposal.auto_patch:
        committed = await commit_updates(
            user_id=user_id,
            request_id=request_id,
            facts_to_commit=proposal.auto_patch,
            justification=f"Auto-extracted from info_collection summary (request={request_id})",
            actor="slot_fact_binder",
        )

    # ── Step 4b: Deprecate old facts superseded by new ones ──
    # When the user corrects a fact, the extraction may create a different key
    # (e.g., "preference.service_type" instead of updating "preference.massage_type").
    # Detect these by comparing committed fact namespaces against existing profile
    # facts and deprecating old facts that the new ones semantically replace.
    if committed and existing_profile_facts:
        store = get_fact_store()
        for record in committed:
            new_ns = record.fact_key.split(".")[0] if "." in record.fact_key else ""
            new_facet = record.fact_key.split(".", 1)[1] if "." in record.fact_key else ""
            for old_key, old_info in existing_profile_facts.items():
                if old_key == record.fact_key:
                    continue  # Same key — upsert_fact already handled it
                old_ns = old_key.split(".")[0] if "." in old_key else ""
                if old_ns != new_ns:
                    continue  # Different namespace — not related
                old_val = str(old_info.get("value", ""))
                new_val = str(record.fact_value)
                # Same namespace, different key, different value — likely a replacement
                # Check if this is a disputed key or if the values are contradictory
                is_disputed = old_key in (disputed_fact_keys or [])
                # Also check semantic overlap in facet names
                old_facet = old_key.split(".", 1)[1] if "." in old_key else ""
                facet_overlap = bool(set(old_facet.split("_")) & set(new_facet.split("_")))
                if is_disputed or facet_overlap:
                    try:
                        old_facts = await store.get_active_facts(
                            user_id=user_id, entity_id=entity_id, fact_keys=[old_key],
                        )
                        for of in old_facts:
                            await store.deprecate_fact(of)
                            logger.info(
                                f"Auto-deprecated '{old_key}={old_val}' "
                                f"(superseded by '{record.fact_key}={new_val}')"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to auto-deprecate {old_key}: {e}")

    # ── Step 5: Store needs_confirm as memory_candidate events ──
    if proposal.needs_confirm:
        await _store_memory_candidates(
            user_id=user_id,
            request_id=request_id,
            entity_id=entity_id,
            needs_confirm=proposal.needs_confirm,
        )

    # ── Step 6: Build slot_refs ──────────────────────────────
    slot_refs: Dict[str, str] = {}
    for record in committed:
        slot_refs[record.fact_key] = record.fact_id

    logger.info(
        f"Fact binding complete: {len(committed)} committed, "
        f"{len(proposal.needs_confirm)} need confirm, "
        f"{len(proposal.reject)} rejected"
    )

    return slot_refs


async def _record_to_candidate_pool(
    pool_store: CandidateKeyPoolStore,
    extracted: ExtractedFact,
    request_id: str,
) -> None:
    """Record an unresolved fact key in the global candidate pool."""
    try:
        value_str = str(extracted.value) if extracted.value is not None else ""
        await pool_store.record_candidate(
            candidate_key=extracted.fact_key,
            description=extracted.fact_label,
            risk_suggestion=extracted.risk_suggestion,
            value=value_str,
            entity_id=extracted.entity_id,
            request_id=request_id,
        )
    except Exception as e:
        logger.warning(f"Failed to record candidate key {extracted.fact_key}: {e}")


async def _store_memory_candidates(
    user_id: str,
    request_id: str,
    entity_id: str,
    needs_confirm: List[ExtractedFact],
) -> None:
    """Store facts that need confirmation as memory_candidate events."""
    try:
        event_store = get_event_store()
        for fact in needs_confirm:
            await event_store.add_event(
                user_id=user_id,
                event_type="memory_candidate",
                content=f"Unconfirmed fact: {fact.fact_label or fact.fact_key} = {fact.value}",
                request_id=request_id,
                care_recipient_id=entity_id if entity_id.startswith("care_recipient:") else None,
                structured={
                    "fact_key": fact.fact_key,
                    "fact_label": fact.fact_label,
                    "value": fact.value,
                    "confidence": fact.confidence,
                    "evidence": fact.evidence,
                    "source_type": fact.source_type,
                },
                tags=["needs_confirm", "slot_binding"],
            )
    except Exception as e:
        logger.warning(f"Failed to store memory candidates: {e}")
