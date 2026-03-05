"""
Memory MCP Server — unified tool interface for the User Memory Framework.

Exposes tools via FastMCP v3 (stdio transport):
  - memory_get_context_bundle: Assemble context for current turn
  - memory_get_profile_facts: Get specific active facts
  - memory_get_requests_by_status: Query requests by status (GSI1)
  - memory_get_requests_by_entity: Query requests by care recipient (GSI2)
  - memory_get_request_detail: Full detail for a single request
  - memory_list_recent_requests: Recent requests with date filter
  - memory_resolve_fact_key: Key Resolver — map text to canonical key
  - memory_propose_updates: Write Gate — classify facts by risk
  - memory_commit_updates: Write facts + audit log
  - memory_add_event: Write episodic event

Env vars: ANTHROPIC_API_KEY (for Key Resolver LLM calls)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

from fastmcp import FastMCP

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_bundle import get_context_bundle, bundle_to_prompt_block
from event_store import get_event_store
from fact_store import get_fact_store
from request_store import get_request_store
from key_resolver import get_key_registry, get_key_resolver
from memory_write_gate import (
    build_confirmation_questions,
    commit_updates,
    propose_updates,
)
from models.fact_models import ExtractedFact

memory_mcp = FastMCP(name="memory-mcp")


def _log(msg: str) -> None:
    print(f"[memory_mcp] {msg}", file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────
# Tool 1: Get Context Bundle
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_get_context_bundle",
    description=(
        "Assemble a structured context bundle for the current conversation turn. "
        "Returns active request info, relevant profile facts, recent events, and "
        "safety notes. Use this at the start of each turn to get memory context."
    ),
)
async def memory_get_context_bundle(
    user_id: str,
    message: str = "",
    request_dict: Optional[dict] = None,
    k: int = 5,
) -> dict:
    """Assemble context bundle for the current turn."""
    _log(f">>> memory_get_context_bundle | user={user_id}")
    bundle = await get_context_bundle(
        user_id=user_id,
        request_dict=request_dict,
        message=message,
        k=k,
    )
    result = bundle.model_dump(mode="json")
    result["prompt_block"] = bundle_to_prompt_block(bundle)
    _log(f"<<< memory_get_context_bundle done")
    return result


# ─────────────────────────────────────────────────────────────
# Tool 2: Get Profile Facts
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_get_profile_facts",
    description=(
        "Get active facts for a specific entity (e.g., care_recipient:mom). "
        "Optionally filter by fact_keys. Returns only active (not deprecated) facts."
    ),
)
async def memory_get_profile_facts(
    user_id: str,
    entity_id: str,
    fact_keys: Optional[list[str]] = None,
) -> list[dict]:
    """Get active facts for an entity."""
    _log(f">>> memory_get_profile_facts | user={user_id}, entity={entity_id}")
    store = get_fact_store()
    facts = await store.get_active_facts(user_id, entity_id, fact_keys)
    result = [f.model_dump(mode="json") for f in facts]
    _log(f"<<< memory_get_profile_facts returned {len(result)} facts")
    return result


# ─────────────────────────────────────────────────────────────
# Tool 3: Get Requests by Status
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_get_requests_by_status",
    description=(
        "Query requests filtered by status. Use this when the user asks about "
        "in-progress, queued, blocked, completed, or created requests. "
        "Valid statuses: created, collecting, executing, paused, completed, cancelled. "
        "Returns summaries sorted by most recently touched first. "
        "Each summary includes: request_id, title, goal, status, request_type, "
        "subject_entity_id, priority, created_at, last_touched_at, summary_current, stage_detail."
    ),
)
async def memory_get_requests_by_status(
    user_id: str,
    status: str,
    limit: int = 20,
) -> list[dict]:
    """Query requests by status via GSI1."""
    _log(f">>> memory_get_requests_by_status | user={user_id}, status={status}")
    store = get_request_store()
    results = await store.query_by_status(user_id=user_id, status=status, limit=limit)
    _log(f"<<< memory_get_requests_by_status returned {len(results)} requests")
    return results


# ─────────────────────────────────────────────────────────────
# Tool 4: Get Requests by Entity
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_get_requests_by_entity",
    description=(
        "Query requests related to a specific care recipient or entity. "
        "Use this when the user asks 'How has mom been doing?' or 'What have we "
        "done for dad?'. The entity_id should be a canonical entity identifier "
        "like 'care_recipient:mom' or 'care_recipient:dad'. "
        "Optionally filter to requests after a given ISO date (e.g. '2025-12-01T00:00:00'). "
        "Returns summaries sorted by most recently touched first."
    ),
)
async def memory_get_requests_by_entity(
    entity_id: str,
    limit: int = 20,
    after_date: Optional[str] = None,
) -> list[dict]:
    """Query requests by care recipient entity via GSI2."""
    _log(f">>> memory_get_requests_by_entity | entity={entity_id}")
    store = get_request_store()
    results = await store.query_by_entity(
        entity_id=entity_id, limit=limit, after_date=after_date,
    )
    _log(f"<<< memory_get_requests_by_entity returned {len(results)} requests")
    return results


# ─────────────────────────────────────────────────────────────
# Tool 4b: Get Request Detail
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_get_request_detail",
    description=(
        "Get full detail for a single request by its request_id. "
        "Use this when the user asks for more information about a specific request, "
        "or when you need to inspect slots, open_questions, or artifacts. "
        "Returns the complete request record including payload, slots, "
        "open_questions, artifacts, prereq_gate, and audit trail."
    ),
)
async def memory_get_request_detail(
    user_id: str,
    request_id: str,
) -> dict:
    """Direct PK/SK lookup for a single request."""
    _log(f">>> memory_get_request_detail | user={user_id}, request={request_id}")
    store = get_request_store()
    result = await store.get_request(user_id=user_id, request_id=request_id)
    if result is None:
        _log(f"<<< memory_get_request_detail: not found")
        return {"error": "not_found", "message": f"Request {request_id} not found"}
    _log(f"<<< memory_get_request_detail: found")
    return result


# ─────────────────────────────────────────────────────────────
# Tool 4c: List Recent Requests
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_list_recent_requests",
    description=(
        "List recent requests for a user, optionally filtered to a date range. "
        "Use this when the user asks 'What did you help me with last month?' or "
        "'Show me my recent requests'. Pass after_date as an ISO timestamp "
        "(e.g. '2025-12-01T00:00:00') to filter by creation date. "
        "Returns summaries sorted by most recent first."
    ),
)
async def memory_list_recent_requests(
    user_id: str,
    limit: int = 20,
    after_date: Optional[str] = None,
) -> list[dict]:
    """Query recent requests with optional date filter."""
    _log(f">>> memory_list_recent_requests | user={user_id}, after={after_date}")
    store = get_request_store()
    results = await store.query_recent(
        user_id=user_id, limit=limit, after_date=after_date,
    )
    _log(f"<<< memory_list_recent_requests returned {len(results)} requests")
    return results


# ─────────────────────────────────────────────────────────────
# Tool 5: Resolve Fact Key
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_resolve_fact_key",
    description=(
        "Map a natural-language fact description to a canonical FactKey from the "
        "registry. Returns the matched key, confidence, risk level, and suggested "
        "aliases. Use this before writing any fact to ensure correct key mapping."
    ),
)
async def memory_resolve_fact_key(
    fact_text: str,
    entity_id: str,
    request_type: str = "unknown",
) -> dict:
    """Resolve a fact description to a canonical key."""
    _log(f">>> memory_resolve_fact_key | text={fact_text[:80]}")
    resolver = get_key_resolver()
    result = await resolver.resolve(
        fact_text=fact_text,
        entity_id=entity_id,
        context={"request_type": request_type},
    )
    _log(f"<<< memory_resolve_fact_key -> {result.canonical_key} ({result.confidence:.2f})")
    return result.model_dump(mode="json")


# ─────────────────────────────────────────────────────────────
# Tool 6: Propose Updates (Write Gate — read-only, proposes only)
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_propose_updates",
    description=(
        "Run the Memory Write Gate on a list of extracted facts. "
        "Returns which facts can be auto-written, which need user confirmation, "
        "and confirmation questions for the latter. Does NOT write anything."
    ),
)
async def memory_propose_updates(
    extracted_facts: list[dict],
    existing_fact_keys: Optional[list[str]] = None,
) -> dict:
    """Classify extracted facts through the write gate."""
    _log(f">>> memory_propose_updates | {len(extracted_facts)} facts")
    facts = [ExtractedFact(**f) for f in extracted_facts]

    # Filter out facts whose keys are already stored (dedup pre-classification)
    if existing_fact_keys:
        existing_set = set(existing_fact_keys)
        facts = [f for f in facts if f.fact_key not in existing_set]
        _log(f"    filtered to {len(facts)} after dedup vs {len(existing_fact_keys)} existing keys")

    proposal = propose_updates(facts)

    # Generate confirmation questions for needs_confirm
    questions = build_confirmation_questions(proposal.needs_confirm)

    result = proposal.model_dump(mode="json")
    result["confirmation_questions"] = questions
    _log(
        f"<<< memory_propose_updates: "
        f"auto={len(proposal.auto_patch)}, "
        f"confirm={len(proposal.needs_confirm)}"
    )
    return result


# ─────────────────────────────────────────────────────────────
# Tool 7: Commit Updates (writes to Fact Store + audit log)
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_commit_updates",
    description=(
        "Write approved facts to the Fact Store. This is a WRITE operation. "
        "Only call this after propose_updates has classified facts as auto_patch, "
        "or after the user has confirmed facts from needs_confirm. "
        "Writes audit log entries for every fact committed."
    ),
    tags={"write", "destructive"},
)
async def memory_commit_updates(
    user_id: str,
    request_id: str,
    facts_to_commit: list[dict],
    justification: str,
    actor: str = "agent",
    conflict_strategy: str = "auto",
) -> dict:
    """Commit approved facts to storage."""
    _log(f">>> memory_commit_updates | user={user_id}, {len(facts_to_commit)} facts, strategy={conflict_strategy}")
    facts = [ExtractedFact(**f) for f in facts_to_commit]
    committed = await commit_updates(
        user_id=user_id,
        request_id=request_id,
        facts_to_commit=facts,
        justification=justification,
        actor=actor,
        conflict_strategy=conflict_strategy,
    )
    result = {
        "committed_count": len(committed),
        "committed_fact_ids": [f.fact_id for f in committed],
    }
    _log(f"<<< memory_commit_updates: committed {len(committed)} facts")
    return result


# ─────────────────────────────────────────────────────────────
# Tool 8: Add Event
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_add_event",
    description=(
        "Write an episodic event to the UserEventTable. "
        "Events include dialogue summaries, tool results, decisions, errors, etc. "
        "Events have automatic TTL-based cleanup."
    ),
)
async def memory_add_event(
    user_id: str,
    event_type: str,
    content: str,
    request_id: Optional[str] = None,
    care_recipient_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Write an episodic event."""
    _log(f">>> memory_add_event | user={user_id}, type={event_type}")
    store = get_event_store()
    event = await store.add_event(
        user_id=user_id,
        event_type=event_type,
        content=content,
        request_id=request_id,
        care_recipient_id=care_recipient_id,
        tags=tags,
    )
    _log(f"<<< memory_add_event: {event.event_id}")
    return {"event_id": event.event_id, "timestamp": event.timestamp.isoformat()}


# ─────────────────────────────────────────────────────────────
# Tool 9: Resolve Entity ID
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_resolve_entity_id",
    description=(
        "Infer the subject entity_id from user text using keyword matching "
        "(English + Chinese) with optional LLM fallback. Returns an entity_id "
        "like 'care_recipient:mom', 'care_recipient:dad', 'user:self', etc."
    ),
)
async def memory_resolve_entity_id(
    user_text: str,
    known_entities: Optional[list[str]] = None,
    language_hint: Optional[str] = None,
) -> dict:
    """Resolve entity_id from user text."""
    _log(f">>> memory_resolve_entity_id | text={user_text[:80]}")
    from prompts import _infer_subject_entity
    # Use keyword-only path (no LLM client in MCP server context)
    entity_id = await _infer_subject_entity(user_text, client=None)
    _log(f"<<< memory_resolve_entity_id -> {entity_id}")
    return {"entity_id": entity_id}


# ─────────────────────────────────────────────────────────────
# Tool 10: Batch Resolve Fact Keys
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_resolve_fact_keys_batch",
    description=(
        "Resolve multiple natural-language fact descriptions to canonical FactKeys "
        "in a single operation. More efficient than calling memory_resolve_fact_key "
        "multiple times. Returns a list of resolution results in the same order."
    ),
)
async def memory_resolve_fact_keys_batch(
    facts: list[dict],
    entity_id: str,
    request_type: str = "unknown",
) -> list[dict]:
    """Batch resolve fact descriptions to canonical keys."""
    _log(f">>> memory_resolve_fact_keys_batch | {len(facts)} facts, entity={entity_id}")
    from key_resolver import get_key_resolver
    resolver = get_key_resolver()
    results = await resolver.resolve_batch(
        facts=facts,
        entity_id=entity_id,
        context={"request_type": request_type},
    )
    output = [r.model_dump(mode="json") for r in results]
    _log(f"<<< memory_resolve_fact_keys_batch: {len(output)} results")
    return output


# ─────────────────────────────────────────────────────────────
# Tool 11: Get Pending Confirmations
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_get_pending_confirmations",
    description=(
        "Query memory_candidate events that need user confirmation. "
        "Returns facts that were stored as candidates (high-risk conflicts) "
        "pending user verification."
    ),
)
async def memory_get_pending_confirmations(
    user_id: str,
    entity_id: Optional[str] = None,
) -> list[dict]:
    """Get pending fact confirmations."""
    _log(f">>> memory_get_pending_confirmations | user={user_id}, entity={entity_id}")
    store = get_event_store()
    events = await store.get_recent_events(
        user_id=user_id,
        event_types=["memory_candidate"],
        limit=20,
    )

    results = []
    for evt in events:
        structured = evt.structured or {}
        # Filter by entity_id if specified
        if entity_id and structured.get("entity_id") and structured["entity_id"] != entity_id:
            continue
        results.append({
            "event_id": evt.event_id,
            "fact_key": structured.get("fact_key", ""),
            "fact_label": structured.get("fact_label", ""),
            "value": structured.get("value"),
            "confidence": structured.get("confidence", 0.0),
            "evidence": structured.get("evidence", ""),
            "timestamp": evt.timestamp.isoformat(),
        })

    _log(f"<<< memory_get_pending_confirmations: {len(results)} pending")
    return results


# ─────────────────────────────────────────────────────────────
# Tool 12: Confirm Fact
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_confirm_fact",
    description=(
        "Promote or reject a candidate fact after user confirmation. "
        "If confirmed=true, the candidate fact is promoted to active. "
        "If confirmed=false, the candidate fact is rejected. "
        "Optionally provide a corrected_value if user corrected the value."
    ),
    tags={"write"},
)
async def memory_confirm_fact(
    user_id: str,
    event_id: str,
    confirmed: bool,
    corrected_value: Optional[str] = None,
) -> dict:
    """Confirm or reject a candidate fact."""
    _log(f">>> memory_confirm_fact | user={user_id}, event={event_id}, confirmed={confirmed}")
    fact_store = get_fact_store()

    # Look up the memory_candidate event to get fact details
    event_store = get_event_store()
    events = await event_store.get_recent_events(
        user_id=user_id,
        event_types=["memory_candidate"],
        limit=50,
    )
    target_event = None
    for evt in events:
        if evt.event_id == event_id:
            target_event = evt
            break

    if not target_event:
        _log(f"<<< memory_confirm_fact: event {event_id} not found")
        return {"status": "error", "message": f"Event {event_id} not found"}

    structured = target_event.structured or {}
    fact_key = structured.get("fact_key", "")
    entity_id = structured.get("entity_id") or target_event.care_recipient_id or "care_recipient:unknown"
    value = corrected_value if corrected_value is not None else structured.get("value")

    if confirmed:
        # Promote: upsert as active with explicit confirmation
        record = await fact_store.upsert_fact(
            user_id=user_id,
            entity_id=entity_id,
            fact_key=fact_key,
            new_value=value,
            fact_label=structured.get("fact_label", ""),
            confidence=structured.get("confidence", 0.9),
            source_type=structured.get("source_type", "user"),
            evidence=structured.get("evidence", ""),
            verification_level="explicit_user_confirmed",
            conflict_strategy="overwrite",
        )
        _log(f"<<< memory_confirm_fact: promoted {fact_key} -> {record.fact_id}")
        return {"status": "confirmed", "fact_id": record.fact_id, "fact_key": fact_key}
    else:
        _log(f"<<< memory_confirm_fact: rejected {fact_key}")
        return {"status": "rejected", "fact_key": fact_key}


# ─────────────────────────────────────────────────────────────
# Tool 13: Write Back Aliases
# ─────────────────────────────────────────────────────────────

@memory_mcp.tool(
    name="memory_write_back_aliases",
    description=(
        "Write alias mappings to the FactAliasTable. "
        "Used to record observed or suggested aliases from key resolution."
    ),
    tags={"write"},
)
async def memory_write_back_aliases(
    canonical_key: str,
    aliases: list[str],
    scope: str = "global",
) -> dict:
    """Write aliases for a canonical key."""
    _log(f">>> memory_write_back_aliases | key={canonical_key}, {len(aliases)} aliases")
    from models.fact_models import AliasRecord
    store = get_fact_store()
    written = 0
    for alias_text in aliases:
        normalized = alias_text.strip().lower()
        if not normalized:
            continue
        alias = AliasRecord(
            normalized_alias=normalized,
            canonical_key=canonical_key,
            scope=scope,
            confidence=0.8,
            status="active",
        )
        await store.put_alias(alias)
        written += 1
    _log(f"<<< memory_write_back_aliases: wrote {written} aliases")
    return {"written_count": written, "canonical_key": canonical_key}


if __name__ == "__main__":
    memory_mcp.run()
