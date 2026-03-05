# Epic: User Memory Framework — DynamoDB + Vector Index + MCP Integration

## 0. Executive Summary

This epic turns the current in-memory request system into a full **three-tier user memory framework** backed by DynamoDB, Zilliz/Milvus vector index, and exposed as FastMCP v3 tools. It bridges the gap between the existing codebase stubs (`historical_store.py`, `retrieval.py`, `daily_analyzer.py`, `_consume_ddb_writes()`) and the production architecture discussed in the brainstorm.

### What exists today (codebase audit)

| Component | File(s) | Status |
|-----------|---------|--------|
| Request lifecycle (create/pause/resume/complete) | `graph.py`, `request_factory.py`, `merge_utils.py` | Functional, in-memory only |
| `ddb_writes` generation | `request_factory.py:74` | Generates DDB items, consumed & **discarded** by `_consume_ddb_writes()` |
| Request queue (prereqs) | `state_models.py:197-208`, `merge_utils.py:51-66` | In-memory list, DDB placeholder noted |
| Daily analyzer (LLM retrospective) | `daily_analyzer.py` (543 lines) | Fully implemented, no trigger or data source |
| Historical store (DDB+Milvus) | `historical_store.py` (400 lines) | Interfaces complete, ~15 TODO stubs |
| Retrieval API | `retrieval.py` (144 lines) | Returns empty lists |
| User profile / memory | `state_models.py:120-135` (`UserContext`, `MemoryBlock`, `ProfileSnapshot`) | Skeleton — `facts: Dict[str, Any]`, no structure |
| Stage history tracking | `graph.py` (multiple sites) | Functional, appends to in-memory list |

### What this epic delivers

1. **Fact Store** — entity-centric, versioned, auditable facts with controlled FactKey registry
2. **DynamoDB persistence** — 5 tables replacing in-memory state
3. **Context Bundle** — structured, minimal-context assembly for every LLM call
4. **Memory Write Gate** — risk-based write control (low/med/high)
5. **Key Resolver** — dual-channel (KeyDoc + Alias) candidate recall → LLM selection
6. **Memory MCP Server** — unified tool interface for all agents
7. **Daily job wiring** — connect existing `DailyRequestAnalyzer` to real storage
8. **Request identity refactor** — `request_type` + `subject_entity_id` + `variant` replacing freeform `name`

---

## 1. Data Model — DynamoDB Tables

### 1.1 UserFactTable (NEW — core of the memory system)

The brainstorm's key insight: **facts belong to entities (mom/dad/self), not to requests**. This enables cross-request info reuse (e.g., insurance info collected during "renew insurance" auto-fills slots in "file reimbursement").

**Primary key:**
- `PK`: `USER#<user_id>#ENT#<entity_id>` — e.g., `USER#u123#ENT#care_recipient:mom`
- `SK`: `FACT#<fact_key>#TS#<timestamp>#<fact_id>` — e.g., `FACT#insurance.member_id#TS#2026-02-26T15:30:00Z#f789`

**Fields:**

```python
class FactRecord(BaseModel):
    fact_id: str
    user_id: str
    entity_id: str              # "care_recipient:mom", "user:self", "policy:medicaid_plan_123"
    fact_key: str               # Canonical key from registry: "insurance.member_id"
    fact_label: str = ""        # Free-text label: "妈妈医保卡号" (for display, not indexing)
    fact_value: Any             # Typed value (str/int/list/dict)
    value_type: str             # "string" | "enum" | "code" | "date" | "list" | "object"

    # Provenance & trust
    status: Literal["active", "deprecated", "candidate"] = "candidate"
    risk_level: Literal["low", "medium", "high"] = "medium"
    confidence: float = 0.0
    source_type: Literal["user", "tool", "document", "agent_inference"] = "agent_inference"
    source_ref: str = ""        # request_id / tool_call_id / doc_id
    evidence: str = ""          # Short evidence snippet (<=200 chars)

    # Verification
    verification_level: Literal[
        "explicit_user_confirmed", "tool_verified", "unverified"
    ] = "unverified"
    needs_reconfirm_after_days: int = 90

    # Versioning
    supersedes_fact_id: Optional[str] = None  # Links to the fact this replaces
    schema_version: int = 1

    # Timestamps
    first_seen_at: datetime
    last_seen_at: datetime
    last_verified_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
```

**GSIs:**
- **GSI1** (by fact_key): `GSI1PK = USER#<user_id>#ENT#<entity_id>`, `GSI1SK = KEY#<fact_key>#STATUS#<status>` — fast lookup of active facts by key
- **GSI2** (by status): `GSI2PK = USER#<user_id>#STATUS#active`, `GSI2SK = KEY#<fact_key>` — list all active facts for a user

**Deduplication rule (deterministic, not LLM):**
- Uniqueness key: `(entity_id, fact_key, status="active")` — at most ONE active fact per entity+key
- When a new value arrives for an existing active key:
  - Same value → update `last_seen_at` and `evidence` only
  - Different value → apply conflict strategy (see Write Gate §4)

**"Decapitate" (version chain):**
New fact gets `status=active`; old fact gets `status=deprecated` + `supersedes_fact_id` set on the new one. Never delete — full audit trail.

### 1.2 UserRequestTable (EVOLVE existing `request_factory.py` DDB items)

The existing `request_factory.py:50-66` already generates DDB items. We evolve them with the brainstorm's identity split.

**Primary key:**
- `PK`: `USER#<user_id>`
- `SK`: `REQ#<created_at>#<request_id>`

**New/changed fields (beyond what `request_factory.py` already emits):**

```python
# Identity split (replaces freeform `name` as the sole identifier)
request_type: str           # Controlled taxonomy: "insurance_renewal", "find_caregiver", etc.
subject_entity_id: str      # "care_recipient:mom", "care_recipient:dad", "user:self"
variant: Dict[str, str]     # {"plan_type": "medicaid", "state": "IL"}
title: str                  # Display name (LLM-generated, human-readable)

# Summaries (missing from current schema)
summary_current: str = ""   # Current-state summary, updated on each status change
summary_timeline: str = ""  # Brief event timeline

# Slot-to-fact binding
slot_refs: Dict[str, str]   # {"insurance_plan_id": "fact_789"} — links slots to UserFactTable
```

**GSIs:**
- **GSI1**: `GSI1PK = USER#<user_id>#STATUS#<status>`, `GSI1SK = LAST#<last_activity_at>#REQ#<request_id>` — find active/waiting requests
- **GSI2**: `GSI2PK = RECIPIENT#<care_recipient_id>`, `GSI2SK = LAST#<last_activity_at>#REQ#<request_id>` — find all requests for a specific care recipient

**Request type taxonomy (seed, ~40 types):**

```yaml
# Top-level categories for WithCare caregiving platform
insurance:
  - insurance_renewal
  - insurance_application
  - insurance_appeal
  - insurance_prior_auth
  - insurance_eligibility_check
medical:
  - find_provider
  - schedule_appointment
  - medication_management
  - medication_refill
  - symptom_assessment
care:
  - find_caregiver
  - care_plan_review
  - daily_care_setup
  - respite_care
  - care_transition
benefits:
  - medicaid_application
  - medicare_enrollment
  - ssi_ssd_application
  - benefits_eligibility
  - benefits_renewal
legal:
  - poa_setup
  - guardianship
  - hipaa_release
  - advance_directive
financial:
  - reimbursement_filing
  - bill_dispute
  - cost_estimation
  - financial_assistance
general:
  - information_lookup
  - emotional_support
  - resource_referral
  - complaint_escalation
```

LLM classifies incoming requests into this taxonomy (selection, not generation). If none match, use `general.information_lookup` and flag for taxonomy review.

### 1.3 UserEventTable (NEW — short-term episodic memory)

**Primary key:**
- `PK`: `USER#<user_id>`
- `SK`: `EVT#<timestamp>#<event_id>`

**Fields:**

```python
class EventRecord(BaseModel):
    event_id: str
    user_id: str
    timestamp: datetime
    event_type: Literal[
        "dialogue_summary",    # End-of-turn conversation digest
        "tool_result",         # MCP tool call result summary
        "escalation",          # Handed off to human / escalated
        "decision",            # User made a key decision
        "memory_candidate",    # Extracted facts awaiting confirmation
        "error",               # Agent error/failure
    ]
    request_id: Optional[str] = None
    care_recipient_id: Optional[str] = None
    content: str                     # Summary text
    structured: Dict[str, Any] = {}  # Optional structured payload
    embedding_ref: Optional[str] = None
    tags: List[str] = []
    ttl_epoch: int                   # DynamoDB TTL (30/60/90 days)
```

### 1.4 FactAliasTable (NEW — alias learning layer)

This is how "LLM creativity stays in the alias layer, not the canonical schema."

**Primary key:**
- `PK`: `ALIAS#<normalized_alias>` — e.g., `ALIAS#医保卡号`
- `SK`: `KEY#<canonical_key>` — e.g., `KEY#insurance.member_id`

**Fields:**

```python
class AliasRecord(BaseModel):
    normalized_alias: str       # Lowercase, trimmed, punctuation-stripped
    canonical_key: str          # Points to FactKey registry entry
    scope: Literal["global", "user", "locale"] = "global"
    user_id: Optional[str] = None   # When scope="user"
    count: int = 1                  # Occurrence frequency
    last_seen_at: datetime
    confidence: float = 0.0         # System-accumulated confidence
    status: Literal["active", "pending", "rejected", "deprecated"] = "pending"
    evidence_samples: List[str] = []  # Up to 3 short source excerpts
```

**Alias lifecycle:**
- `alias_observed` (from user's actual words) → `active` or `pending` depending on risk
- `alias_suggested` (LLM's creative suggestion) → always `pending`
- Promotion rule: `pending` → `active` when count >= 5 within 7 days AND maps consistently to same canonical key
- Conflict: same alias → multiple keys → keep multi-mapping with weights `P(key|alias)`, use `scope=user` to override at user level

### 1.5 MemoryFactLogTable (NEW — audit trail)

**Primary key:**
- `PK`: `USER#<user_id>`
- `SK`: `FACTLOG#<timestamp>#<log_id>`

**Fields:**

```python
class FactLogEntry(BaseModel):
    log_id: str
    user_id: str
    timestamp: datetime
    target: Literal["profile", "request", "fact"]
    target_id: str              # fact_id / request_id
    patch: Dict[str, Any]       # JSON diff of what changed
    justification: str          # Why this change was made
    source_ref: str             # request_id / tool_call_id
    actor: str                  # agent_id + model
    result: Literal["applied", "rejected", "needs_confirm"]
```

---

## 2. FactKey Registry

### 2.1 Design principles

- **Canonical keys are path-based**: `<namespace>.<facet>[.<subfacet>]`
- **LLM never invents canonical keys** — it selects from Top-K candidates or returns `unknown`
- **LLM CAN propose aliases** — these go through the alias lifecycle
- **Risk level is deterministic** — registry defines risk per key, LLM suggestions are overridden

### 2.2 Registry storage

Store as a YAML/JSON config file initially (can move to DynamoDB later). Each entry:

```yaml
- key: "insurance.member_id"
  namespace: "insurance"
  facet: "member_id"
  description: "Insurance plan member ID / policy number on card"
  risk_level: "high"
  value_type: "string"
  cardinality: "single"
  aliases: ["plan id", "member id", "policy number", "卡号", "会员号", "medicaid id"]
  examples: ["ABC123", "Member ID: 12345678"]
  canonicalization: "uppercase, strip spaces"
```

### 2.3 Registry v0 — Caregiving scenario (~130 keys)

Full registry organized by the 12 namespaces:

**identity.*** (9 keys)
`full_name`, `dob`, `gender`, `language_primary`, `language_secondary`, `address`, `relationship_to_user`, `ssn_last4` (high), `citizenship_status`

**contact.*** (6 keys)
`primary_phone`, `secondary_phone`, `email`, `preferred_channel`, `emergency_contact`, `caregiver_support_network`

**insurance.*** (10 keys)
`plan_type`, `plan_name`, `member_id`, `group_number`, `payer_phone`, `effective_date`, `renewal_deadline`, `state`, `coverage_notes`, `prior_auth_required`

**medical.*** (9 keys)
`conditions`, `allergies`, `diagnoses_recent`, `vitals.baseline_bp`, `vitals.baseline_weight`, `risk.fall_risk`, `cognitive_status`, `symptoms_current`, `surgical_history`

**provider.*** (7 keys)
`primary_care.name`, `primary_care.phone`, `specialists`, `preferred_hospital`, `pharmacy.name`, `pharmacy.phone`, `pharmacy.address`

**appointment.*** (5 keys)
`next.date_time`, `next.location`, `next.reason`, `transportation_needed`, `followup_interval`

**medication.*** (5 keys)
`current_list`, `adherence.issues`, `refill.next_date`, `refill.pharmacy`, `side_effects_reported`

**mobility.*** (4 keys)
`assistive_devices`, `stairs_at_home`, `walking_limit`, `transfer_assistance`

**dailycare.*** (7 keys)
`adl.bathing`, `adl.dressing`, `adl.toileting`, `meal_prep`, `sleep_pattern`, `behavioral_triggers`, `home_safety.concerns`

**preference.*** (9 keys)
`food.like`, `food.dislike`, `taste.like`, `color.like`, `activity.like`, `communication.style`, `routine.morning`, `routine.evening`, `environment.temperature`

**finance.*** (5 keys)
`income.proof_available`, `medical_bills.recent`, `reimbursement.claim_status`, `reimbursement.claim_id`, `out_of_pocket.monthly_estimate`

**legal.*** (5 keys)
`poa.medical`, `poa.financial`, `hipaa_release`, `guardianship`, `documents.storage_location`

> This file ships as `configs/factkey_registry_v0.yaml`. The Key Resolver loads it at startup and builds an in-memory index.

---

## 3. Key Resolver

### 3.1 Architecture

```
fact_text + entity_context
        │
        ├── Channel A: KeyDoc search (BM25/vector over registry entries)
        │   → Top-K_A candidates
        │
        ├── Channel B: Alias search (exact + fuzzy on FactAliasTable)
        │   → Top-K_B candidates
        │
        ├── Merge + dedup → Top-12 candidates
        │
        └── LLM selection (choice from Top-12, NOT free generation)
            → canonical_key | "unknown"
            → alias_observed, alias_suggested
```

### 3.2 Implementation detail

**Phase 1 (MVP):** BM25 over registry entries + exact alias lookup. No embeddings needed yet.

```python
class KeyResolver:
    def __init__(self, registry_path: str = "configs/factkey_registry_v0.yaml"):
        self.registry: List[RegistryEntry] = load_registry(registry_path)
        self.bm25_index = build_bm25_index(self.registry)  # index over key+desc+aliases+examples

    async def resolve(
        self,
        fact_text: str,
        entity_id: str,
        context: Dict[str, str],  # request_type, language, etc.
        top_k: int = 12,
    ) -> KeyResolverResult:
        # Channel A: BM25 search over registry
        candidates_a = self.bm25_index.search(fact_text, k=top_k)

        # Channel B: Alias table lookup
        tokens = tokenize(fact_text)  # simple word/phrase extraction
        candidates_b = await alias_table_lookup(tokens, top_k=top_k)

        # Merge, dedup by canonical_key, rank by combined score
        merged = merge_dedup_rank(candidates_a, candidates_b, max_k=top_k)

        # LLM selection (only if needed — if top candidate score > 0.95, skip LLM)
        if merged[0].score > 0.95:
            selected = merged[0]
        else:
            selected = await llm_select_key(fact_text, entity_id, merged)

        return KeyResolverResult(
            canonical_key=selected.key,
            confidence=selected.confidence,
            risk_level=self.registry.get_risk(selected.key),
            alias_observed=extract_surface_phrases(fact_text, selected.key),
            alias_suggested=selected.llm_aliases,
            candidates=merged,
        )
```

**Phase 2:** Add embedding-based search over registry KeyDocs + alias embeddings for semantic matching.

### 3.3 LLM selection prompt

```
You are a fact-key mapper. Select the best canonical key from the candidates below,
or output "unknown" if none match.

DO NOT invent new keys. You may ONLY select from the candidates or return "unknown".

## Input fact
Entity: {entity_id}
Text: "{fact_text}"
Context: request_type={request_type}

## Candidates (top {top_k})
1) {key} — {description} — risk={risk} — type={value_type}
2) ...

## Output (strict JSON, nothing else):
{{
  "selected_key": "...|unknown",
  "value_candidate": "...|null",
  "confidence": 0.0-1.0,
  "reason": "one sentence",
  "alias_observed": ["surface phrases from user text"],
  "alias_suggested": ["up to 2 alternative names"]
}}
```

---

## 4. Memory Write Gate

### 4.1 Risk classification

| Risk | Examples | Auto-write rule |
|------|----------|----------------|
| **low** | preferences, language, communication style | Auto-write if confidence >= 0.6 |
| **medium** | contact names, relationships, provider names | Auto-write but mark `unverified`; confirm in next interaction |
| **high** | insurance IDs, diagnoses, medications, allergies, SSN, financial | Requires `source_type in (tool, document)` with confidence >= 0.8 **OR** explicit user confirmation. Otherwise → `candidate` only |

### 4.2 Write gate logic

```python
async def propose_updates(
    user_id: str,
    request_id: str,
    extracted_facts: List[ExtractedFact],
) -> WriteProposal:
    auto_patch = []
    needs_confirm = []

    for f in extracted_facts:
        # Risk level comes from registry, NOT from LLM
        risk = factkey_registry.get_risk(f.fact_key)

        if risk == "low" and f.confidence >= 0.6:
            auto_patch.append(f)

        elif risk == "medium" and f.confidence >= 0.7:
            f.verification_level = "unverified"
            auto_patch.append(f)

        elif risk == "high":
            if f.source_type in ("tool", "document") and f.confidence >= 0.8:
                f.verification_level = "tool_verified"
                auto_patch.append(f)
            elif f.explicit_user_confirmed:
                f.verification_level = "explicit_user_confirmed"
                auto_patch.append(f)
            else:
                needs_confirm.append(f)

        else:
            needs_confirm.append(f)

    return WriteProposal(auto_patch=auto_patch, needs_confirm=needs_confirm)
```

### 4.3 Conflict strategy (new value vs. existing active)

| Risk | Same value | Different value |
|------|-----------|-----------------|
| low | Update `last_seen_at` only | Replace (old → deprecated) |
| medium | Update `last_seen_at` only | Replace, mark new as `unverified` |
| high | Update `last_seen_at` only | Do NOT auto-replace. Store as `candidate`. Generate confirmation question |

### 4.4 Explicit confirmation detection

Only recognize as confirmed:
- User says "是的，我确认..." / "Yes, confirmed" / directly provides the value
- Tool/document output matches
- NOT: "应该是吧" / "我记得是" / "I think so" — these stay `unverified`

---

## 5. Context Bundle

### 5.1 Structure (JSON schema)

Every LLM call in `quick_answer_node`, `info_collection`, `deep_search`, `domain_expert`, and `delegator` receives a Context Bundle. This replaces dumping raw conversation history.

```python
class ContextBundle(BaseModel):
    user_id: str
    now: datetime

    # Block 1: Active request (always present if there is one)
    active_request: Optional[ActiveRequestBlock] = None

    # Block 2: Relevant profile facts (only fields relevant to current intent)
    profile_facts: Dict[str, Any] = {}

    # Block 3: Relevant history (recency + relevance merged, max 5-7 items)
    relevant_history: List[RequestSummaryItem] = []

    # Block 4: Recent events (episodic, max 5 items)
    recent_events: List[EventItem] = []

    # Block 5: Safety notes (high-risk facts that need verification)
    safety_notes: List[SafetyNote] = []


class ActiveRequestBlock(BaseModel):
    request_id: str
    request_type: str
    title: str
    status: str
    subject_entity_id: str
    slots: Dict[str, SlotWithFactRef]  # includes fact_ref for cross-request reuse
    next_steps: List[str]


class SlotWithFactRef(BaseModel):
    status: Literal["filled", "missing", "unverified"]
    value: Any = None
    fact_ref: Optional[str] = None  # Points to UserFactTable fact_id
    risk: str = "low"
    verified: bool = False


class SafetyNote(BaseModel):
    topic: str
    fact_key: str
    current_value: Any
    confidence: float
    last_verified_at: Optional[datetime]
    rule: str  # e.g., "High-risk: requires explicit confirmation before submission"
```

### 5.2 Assembly logic

```python
async def get_context_bundle(
    user_id: str,
    request_id: Optional[str],
    message: str,
    k: int = 5,
) -> ContextBundle:
    # 1. Active request from DDB
    active_req = await ddb_get_request(user_id, request_id)

    # 2. Determine which fact keys are needed (from request template/skill)
    needed_keys = get_needed_fact_keys(active_req.request_type) if active_req else []

    # 3. Pull active facts for needed keys (structured query, fast)
    profile_facts = await ddb_get_active_facts(
        user_id=user_id,
        entity_id=active_req.subject_entity_id if active_req else None,
        fact_keys=needed_keys,
    )

    # 4. Recency: last K requests from DDB
    recent_reqs = await ddb_list_recent_requests(user_id, limit=10)

    # 5. Relevance: vector search (Phase 2+)
    relevant_reqs = await milvus_search_request_summaries(user_id, message, k=k)

    # 6. Merge recency + relevance, dedup by request_id
    merged_history = merge_dedup_rank(recent_reqs, relevant_reqs)[:k]

    # 7. Recent events from DDB
    recent_events = await ddb_get_recent_events(user_id, limit=k)

    # 8. Safety notes: flag high-risk facts that are unverified or stale
    safety_notes = compute_safety_notes(profile_facts, threshold_days=90)

    return ContextBundle(
        user_id=user_id,
        now=datetime.utcnow(),
        active_request=format_active_request(active_req, profile_facts),
        profile_facts=profile_facts,
        relevant_history=merged_history,
        recent_events=recent_events,
        safety_notes=safety_notes,
    )
```

**Key principle: do NOT dump memory into the prompt. Give a structured "work desk."**

---

## 6. Memory MCP Server

### 6.1 Server file: `servers/memory_mcp_server.py`

Exposed as FastMCP v3 server with stdio transport (same pattern as `search_mcp_server.py` and `follow_up_mcp_server.py`).

### 6.2 Tools

| Tool | Purpose | Risk |
|------|---------|------|
| `memory.get_context_bundle(user_id, request_id?, message, k?)` | Assemble context bundle for current turn | read-only |
| `memory.get_profile_facts(user_id, entity_id, fact_keys?)` | Get specific active facts | read-only |
| `memory.search_requests(user_id, query, k?, filters?)` | Vector + recency search over request history | read-only |
| `memory.get_active_requests(user_id)` | List in-progress requests | read-only |
| `memory.resolve_fact_key(fact_text, entity_id, context?)` | Key Resolver — map natural text to canonical key | read-only |
| `memory.propose_updates(user_id, request_id, extracted_facts)` | Write Gate — returns auto-patch + needs-confirm | read-only (proposes only) |
| `memory.commit_updates(user_id, request_id, patch, justification)` | Actually write facts + audit log | **write, destructiveHint=true** |
| `memory.add_event(user_id, request_id?, event_type, content, tags?)` | Write episodic event | write |

### 6.3 Registration in `graph.py`

```python
MEMORY_SERVER = MCPServerConfig(
    name="memory",
    transport="stdio",
    command="python",
    args=["-u", "servers/memory_mcp_server.py"],
    keep_alive=True,
)
```

### 6.4 Agent visibility rules

| Agent | Allowed tools |
|-------|--------------|
| `delegator` | All tools |
| `info_collection` | `get_context_bundle`, `get_profile_facts`, `propose_updates`, `commit_updates`, `add_event` |
| `deep_search` | `get_context_bundle`, `search_requests`, `add_event` |
| `domain_expert` | `get_context_bundle`, `get_profile_facts`, `search_requests` |
| `quick_answer` | `get_context_bundle` |
| `front_end_emotional_support` | `get_context_bundle` (read-only, no write tools visible) |

---

## 7. Cross-Request Info Reuse (Slot → Fact Binding)

This solves the brainstorm's core insight: "renew insurance" collects `member_id` which "file reimbursement" also needs.

### 7.1 Mechanism

When `info_collection` fills a slot:

1. Extract the fact via Key Resolver → get `canonical_key`
2. Write to `UserFactTable` via Write Gate (entity-level, not request-level)
3. Store `fact_ref` in the request's `slot_refs`: `{"insurance_plan_id": "fact_789"}`

When a NEW request needs the same fact:

1. Load request template → see that `insurance.member_id` is a required slot
2. Query `UserFactTable`: `entity_id=care_recipient:mom, fact_key=insurance.member_id, status=active`
3. If found and verified → auto-fill the slot, tell LLM "已有信息可用"
4. If found but unverified → pre-fill but ask for confirmation
5. If not found → collect as usual

### 7.2 Changes to `state_models.py`

Add to `RequestRecord`:

```python
# NEW fields
request_type: str = ""
subject_entity_id: str = ""
variant: Dict[str, str] = Field(default_factory=dict)
title: str = ""
summary_current: str = ""
slot_refs: Dict[str, str] = Field(default_factory=dict)  # slot_key → fact_id
```

---

## 8. Wiring Existing Stubs

### 8.1 `_consume_ddb_writes()` — make it real

**Files:** `executor.py:13-17`, `chat_server.py:62-65`

Replace the current no-op with actual DynamoDB writes:

```python
async def _consume_ddb_writes(state: UnifiedState, ddb_client) -> UnifiedState:
    writes = getattr(state, "ddb_writes", [])
    for write in writes:
        if write["op"] == "put":
            await ddb_client.put_item(
                TableName=write["table"],
                Item=serialize_for_dynamo(write["item"]),
            )
        elif write["op"] == "update":
            await ddb_client.update_item(**write["params"])
        elif write["op"] == "delete":
            await ddb_client.delete_item(**write["params"])
    state.__dict__.pop("ddb_writes", None)
    return state
```

### 8.2 `historical_store.py` — fill in TODOs

Replace ~15 TODO stubs with actual boto3/pymilvus calls. Key changes:

- `_store_to_dynamodb()` (line 122): `await self.dynamodb.put_item(...)`
- `_store_to_milvus()` (line 158): `await self.milvus.insert(...)`
- `_generate_embedding()` (line 198): Call OpenAI/Voyage embedding API
- `search_similar_requests()` (line 271): `await self.milvus.search(...)`
- `_store_daily_summary()` (line 231): `await self.dynamodb.put_item(...)`
- `get_requests_by_date()` (line 374): GSI query
- `get_daily_summary()` (line 393): `get_item`

### 8.3 `retrieval.py` — initialize with real clients

Replace lines 22-27:

```python
_historical_store = HistoricalRequestStore(
    dynamodb_client=get_boto3_dynamodb_client(),
    milvus_client=get_milvus_client(),
    embedding_client=get_embedding_client(),
)
```

### 8.4 `daily_analyzer.py` + `example_daily_job.py` — connect to real data

The analyzer is already functional. Wire it up:

1. **Data source**: Query `UserRequestTable` (DDB) for all requests where `last_activity_at` is within the target date
2. **Trigger**: Scheduled Lambda / cron that runs `run_daily_job()` at 23:59 UTC for each active user
3. **Storage**: `HistoricalRequestStore.archive_day()` now writes to real DDB + Milvus
4. **Fact extraction**: After daily analysis, extract facts from completed requests → run through Write Gate → commit to `UserFactTable`

### 8.5 `merge_utils.py:58-60` — queue persistence

Replace in-memory list mutation with DDB conditional delete:

```python
async def dequeue_pending(state, request_id, ddb_client):
    conversation_id = state.meta.conversation_id
    await ddb_client.delete_item(
        TableName="PendingQueue",
        Key={"pk": f"QUEUE#{conversation_id}", "sk": f"ITEM#{request_id}"},
    )
    # Also update in-memory state for the current turn
    state.request_manager.pending_queue = [
        item for item in state.request_manager.pending_queue
        if item.request_id != request_id
    ]
```

---

## 9. Daily Job — Fact Extraction Pipeline

The existing `DailyRequestAnalyzer` produces `HistoricalRequestRecord` with `short_summary`, `theme`, `keywords`. Extend it to also extract entity facts.

### 9.1 New step after per-request analysis

```python
# In daily_analyzer.py, after _analyze_single_request():
extracted_facts = await self._extract_facts_from_request(
    request_dict=req_dict,
    related_conversations=related_convs,
)
# Run through Write Gate
proposal = await propose_updates(user_id, req_id, extracted_facts)
# Auto-commit safe facts
await commit_updates(user_id, req_id, proposal.auto_patch, justification="daily_job_extraction")
# Store candidates for later confirmation
for candidate in proposal.needs_confirm:
    await add_event(user_id, req_id, "memory_candidate", json.dumps(candidate.dict()))
```

### 9.2 Fact extraction prompt

```
Given this request's conversations and collected info, extract factual information
about the entities involved (care recipient, caregiver, etc.).

## Request: {name} ({request_type})
## Subject: {subject_entity_id}
## Collected info: {info_collection_summary}
## Conversations: {related_convs}

For each fact, output:
{{
  "entity_id": "care_recipient:mom",
  "fact_key": "<select from registry>",
  "value": "...",
  "confidence": 0.0-1.0,
  "source_type": "user|tool|document",
  "evidence": "short quote from conversation"
}}
```

---

## 10. Edge Cases

### 10.1 Multiple care recipients

A user may be caring for both mom and dad. Every fact and request must bind to a specific `entity_id`. The delegator must detect which entity the user is talking about (by mention or context) and pass `subject_entity_id` correctly.

**Edge case:** User says "my parent" without specifying which. → Generate a follow-up question via `follow_up_mcp` to clarify.

### 10.2 Fact staleness

High-risk facts (insurance IDs, medications) can become stale. The `needs_reconfirm_after_days` field on each fact triggers a safety note when expired. The Context Bundle includes this in `safety_notes`, and the agent should ask for re-confirmation before acting on stale high-risk facts.

### 10.3 Contradictory facts from different sources

User says "Medicaid" in conversation, but a tool result shows "Medicare". Conflict resolution:

1. Both stored: new as `candidate`, old stays `active`
2. Safety note generated: "Conflicting insurance info — user said Medicaid, tool showed Medicare"
3. Agent asks user to clarify in next interaction

### 10.4 Request type misclassification

LLM picks wrong `request_type` from taxonomy. Mitigation:

- Allow re-classification mid-request (if info_collection reveals it's actually a different type)
- `request_type` change logged in `stage_history`
- Low confidence classifications (< 0.7) trigger confirmation: "It sounds like you want to renew insurance — is that right?"

### 10.5 Alias collision across languages

"卡号" could mean `insurance.member_id` OR `identity.ssn_last4` depending on context. Resolution:

- Store multi-mapping in FactAliasTable (same alias → multiple keys)
- Key Resolver presents both candidates to LLM with context
- User-scope aliases take priority over global

### 10.6 DynamoDB hot partition

User with thousands of facts could cause hot partition. Mitigation:

- `UserFactTable` PK includes `entity_id` → distributes across entities
- `UserEventTable` has TTL → auto-cleanup prevents unbounded growth
- Use DynamoDB on-demand capacity mode

### 10.7 Empty history (new user)

New user has no facts, no history, no events. Context Bundle returns empty blocks. Agents must gracefully handle "no prior context" — the system should NOT fail or hallucinate prior interactions.

---

## 11. Sprint Plan

### Sprint 1 — Foundation (DDB + Fact Store + Write Gate)

**Goal:** Persistent request storage + fact extraction + write control

| Task | Files | Depends on |
|------|-------|-----------|
| Create `configs/factkey_registry_v0.yaml` | New file | — |
| Create `models/fact_models.py` (FactRecord, AliasRecord, FactLogEntry, EventRecord) | New file | — |
| Create DynamoDB table definitions (IaC or manual) | Infra | — |
| Implement `fact_store.py` (CRUD for UserFactTable) | New file | Table definitions |
| Implement Memory Write Gate (`memory_write_gate.py`) | New file | `fact_store.py`, registry |
| Wire `_consume_ddb_writes()` in `executor.py` and `chat_server.py` | Edit existing | Table definitions |
| Add `request_type`, `subject_entity_id`, `variant`, `title`, `summary_current`, `slot_refs` to `RequestRecord` in `state_models.py` | Edit existing | — |
| Update `request_factory.py` to emit new fields + request_type classification | Edit existing | Taxonomy config |
| Add request_type taxonomy config | New file `configs/request_type_taxonomy.yaml` | — |

### Sprint 2 — Context Bundle + Key Resolver

**Goal:** Structured context assembly + controlled fact-key mapping

| Task | Files | Depends on |
|------|-------|-----------|
| Implement `key_resolver.py` (BM25 + alias lookup + LLM selection) | New file | Registry, FactAliasTable |
| Implement `context_bundle.py` (assembly logic) | New file | `fact_store.py`, UserRequestTable |
| Implement `UserEventTable` CRUD (`event_store.py`) | New file | Table definitions |
| Integrate Context Bundle into `quick_answer_node` | Edit `graph.py` | `context_bundle.py` |
| Integrate Context Bundle into `info_collection` node | Edit `graph.py` | `context_bundle.py` |
| Implement slot → fact binding in info_collection | Edit `graph.py` | `fact_store.py`, `key_resolver.py` |

### Sprint 3 — Memory MCP Server + Daily Job Wiring

**Goal:** Unified MCP interface + daily archival pipeline

| Task | Files | Depends on |
|------|-------|-----------|
| Create `servers/memory_mcp_server.py` with all 8 tools | New file | Sprint 1 + 2 modules |
| Register `MEMORY_SERVER` in `graph.py` | Edit existing | MCP server |
| Fill in `historical_store.py` TODO stubs (DDB calls) | Edit existing | DDB client |
| Initialize `retrieval.py` with real clients | Edit existing | `historical_store.py` |
| Wire `daily_analyzer.py` to real data source (DDB query) | Edit existing | UserRequestTable |
| Add fact extraction step to daily job | Edit `daily_analyzer.py` | Write Gate, Key Resolver |
| Implement daily job trigger (cron/scheduler) | New file or Lambda | Daily analyzer |

### Sprint 4 — Vector Search + Alias Learning + Governance

**Goal:** Semantic retrieval + alias lifecycle + audit

| Task | Files | Depends on |
|------|-------|-----------|
| Set up Milvus/Zilliz collections (`request_summaries`, `episodic_events`) | Infra | — |
| Implement embedding pipeline (request summary + event text → vector) | New module | Embedding API |
| Fill in `historical_store.py` Milvus TODO stubs | Edit existing | Milvus client |
| Add relevance channel to Context Bundle assembly | Edit `context_bundle.py` | Milvus search |
| Implement alias lifecycle (pending → active promotion, conflict resolution) | Edit `key_resolver.py` | FactAliasTable |
| Implement `MemoryFactLogTable` writes in `commit_updates` | Edit `fact_store.py` | Table definitions |
| Add Snowflake CDC pipeline (DDB → Snowflake) for analytics | Infra | DDB tables |
| Memory quality metrics: hit rate, re-ask rate, correction rate | New dashboard / queries | Snowflake |

---

## 12. Summary of Files Changed / Created

### New files

| File | Purpose |
|------|---------|
| `configs/factkey_registry_v0.yaml` | Canonical FactKey registry (130 keys, 12 namespaces) |
| `configs/request_type_taxonomy.yaml` | Request type controlled vocabulary (~40 types) |
| `models/fact_models.py` | Pydantic models: FactRecord, AliasRecord, FactLogEntry, EventRecord, ContextBundle |
| `fact_store.py` | DynamoDB CRUD for UserFactTable |
| `event_store.py` | DynamoDB CRUD for UserEventTable |
| `key_resolver.py` | Key Resolver: BM25 + alias + LLM selection |
| `memory_write_gate.py` | Risk-based write control |
| `context_bundle.py` | Context Bundle assembly (recency + relevance + facts) |
| `servers/memory_mcp_server.py` | Memory MCP server (8 tools) |

### Modified files

| File | Changes |
|------|---------|
| `state_models.py` | Add `request_type`, `subject_entity_id`, `variant`, `title`, `summary_current`, `slot_refs` to `RequestRecord` |
| `request_factory.py` | Emit new request fields, integrate request_type classification |
| `graph.py` | Register `MEMORY_SERVER`, integrate Context Bundle into nodes, update info_collection for slot→fact binding |
| `historical_store.py` | Replace ~15 TODO stubs with real DDB/Milvus calls |
| `retrieval.py` | Initialize with real clients |
| `daily_analyzer.py` | Add fact extraction step after per-request analysis |
| `example_daily_job.py` | Update to use real DDB data source |
| `executor.py` | Make `_consume_ddb_writes()` actually write to DDB |
| `chat_server.py` | Same as executor.py |
| `merge_utils.py` | Replace in-memory queue ops with DDB conditional writes |
