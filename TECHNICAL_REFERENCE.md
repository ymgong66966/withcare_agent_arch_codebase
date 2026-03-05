# WithCare Agent Architecture v3 — Technical Reference

> **Last updated:** 2026-03-03
> **Status:** Active development. Fact reconciliation layer deployed. MCP memory server operational.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Graph Architecture & Node Inventory](#2-graph-architecture--node-inventory)
3. [Routing Logic](#3-routing-logic)
4. [Node Deep Dives](#4-node-deep-dives)
5. [Memory Subsystem](#5-memory-subsystem)
6. [Fact Extraction & Reconciliation Pipeline](#6-fact-extraction--reconciliation-pipeline)
7. [Key Resolution System](#7-key-resolution-system)
8. [MCP Server Architecture](#8-mcp-server-architecture)
9. [LLM Functions & Prompt Inventory](#9-llm-functions--prompt-inventory)
10. [DynamoDB Schema](#10-dynamodb-schema)
11. [Entity Inference](#11-entity-inference)
12. [Configuration & Environment](#12-configuration--environment)
13. [Development Log — Fact Reconciliation](#13-development-log--fact-reconciliation)

---

## 1. System Overview

WithCare is a conversational caregiving platform built on a **LangGraph state machine**. A single user message enters at `turn_router`, flows through specialized agent nodes, and exits through `downstream_catcher`. Each turn produces a state patch that is shallow-merged into the graph state.

### High-Level Flow

```
User message
    |
    v
turn_router ──────────────────────────────────────────┐
    |                                                  |
    ├─ new_intent ──> upstream_delegator               |
    |                     |                            |
    |                     ├─ info_collection (guard)   |
    |                     ├─ deep_search               |
    |                     ├─ domain_expert             |
    |                     ├─ user_info                 |
    |                     ├─ quick_answer              |
    |                     └─ front_end                 |
    |                                                  |
    ├─ continuation ──> [mapped agent directly]        |
    |                                                  |
    └─ quick_answer (shortcut) ──> quick_answer        |
                                                       |
    All agent nodes ──> downstream_catcher ────────────┘
                              |                     (loops back
                              v                      or exits)
                           END / next agent
```

### Key Design Principles

- **Non-blocking failures.** Every LLM call, DDB write, and MCP tool call is wrapped in try/except. Failures log warnings and return safe defaults (`{}`, `[]`, `None`). The pipeline never crashes mid-turn.
- **Additive layering.** New capabilities (like fact reconciliation) are inserted as additional steps. Existing mechanisms remain as safety nets.
- **Bilingual (EN/ZH).** Language detection on every turn via CJK Unicode character ratio. All user-facing messages have `_msg(state, zh, en)` variants.
- **Risk-gated memory.** Facts classified by risk level (from a YAML registry). High-risk facts require user confirmation before becoming active.

---

## 2. Graph Architecture & Node Inventory

**Entry point:** `turn_router`
**Framework:** LangGraph `StateGraph` with conditional edges

### Graph Construction (`build_graph()`, graph.py)

```
Nodes:
  turn_router          → Decides continuation vs new intent
  upstream_delegator   → Routes new intents, manages request lifecycle
  front_end            → Emotional support responses
  info_collection      → Conversational information gathering
  deep_search          → Multi-iteration web search with tool selection
  user_info            → Fact retrieval and presentation + corrections
  domain_expert        → Domain knowledge synthesis
  quick_answer         → Lightweight factual Q&A
  downstream_catcher   → Post-node cleanup, consumes pending_handoff

Edges:
  turn_router ──[conditional]──> upstream_delegator | front_end | info_collection |
                                 deep_search | user_info | domain_expert |
                                 quick_answer | END

  upstream_delegator ──[conditional]──> (same targets as above)

  front_end ──> downstream_catcher
  info_collection ──> downstream_catcher
  deep_search ──> downstream_catcher
  user_info ──> downstream_catcher
  domain_expert ──> downstream_catcher
  quick_answer ──> downstream_catcher

  downstream_catcher ──[conditional]──> (loops to any agent or END)
```

All execution agents route through `downstream_catcher` before the next routing decision. This ensures cleanup (clearing `pending_handoff`, checking `tool_failures`) happens after every agent execution.

### State Shape (`GraphState`)

```python
GraphState = TypedDict("GraphState", {
    "meta":                  Dict,  # user_id, conversation_id
    "messages":              List,  # [{role, content, metadata}]
    "routing":               Dict,  # turn_mode, current_agent, pending_handoff, _catcher_next
    "request_manager":       Dict,  # active_request_id, requests, pending_queue
    "user_context":          Dict,  # profile_snapshot (caregiver + care_recipient facts)
    "tools":                 Dict,  # tool_runs, tool_failures
    "slots_upsert":          Dict,  # slot bindings from fact extraction
    "open_questions_upsert": Dict,  # open question tracking
    "artifacts_append":      Dict,  # search results, written guidance
    "ddb_writes":            List,  # DynamoDB write operations
    "info_collection_debug": Dict,  # plan, profile_facts, readiness, disputed_fact_keys
}, total=False)
```

---

## 3. Routing Logic

### 3.1 Turn Router (`route_from_turn_router`)

The entry point for every user message. Uses `llm_turn_mode_decision()` to classify the turn.

| Condition | Route |
|-----------|-------|
| `new_intent` + LLM recommends `quick_answer` | `quick_answer` |
| `new_intent` + anything else | `upstream_delegator` |
| `continuation` + active request has `awaiting_user_input=True` | `info_collection` |
| `continuation` + LLM recommends specific agent | that agent |
| Fallback | `upstream_delegator` |

**Turn mode classification:**
- **continuation**: User answering questions, follow-ups about results, natural progression
- **new_intent**: Topic switch, done with current task, simple acknowledgments ("okay, got it")

### 3.2 Upstream Delegator (`route_from_delegator`)

Handles new intents. Decides what to do with the current request and where to route next.

**Action types (LLM decides):**

| Action | What happens |
|--------|-------------|
| `task_acknowledged` | Mark current request `executed`. Route to END. |
| `resume_existing` | Activate a paused request. Pause current. Route to its agent. |
| `natural_progression` | Update stage of current request. Route to recommended agent. |
| `prerequisite_task` | Create prereq request, pause parent. Route to `info_collection`. |
| `new_unrelated_task` | Create new request, pause current. Route to `info_collection`. |

**Guard rule:** New requests that haven't completed info collection are forced to `info_collection` first (unless the target is `front_end`, `quick_answer`, or `user_info`).

### 3.3 Downstream Catcher (`route_after_catcher`)

Post-node cleanup. Reads `pending_handoff.recommended_next_agent`, stores in `_catcher_next`, clears the handoff.

| Condition | Route |
|-----------|-------|
| `tool_failures` exist | `front_end` (error recovery) |
| `_catcher_next` is set | mapped agent |
| `_catcher_next` is `None` | END (`respond`) |

### 3.4 Prerequisite Lifecycle (`check_prereq_lifecycle`)

Runs **outside** LangGraph after each graph turn completes. Detects:

- **Completion:** Prereq request status=`executed` → resume parent request, build welcome-back message
- **Abandonment:** `user_signals.wants_to_abandon_current_task=True` → abort prereq, resume parent

---

## 4. Node Deep Dives

### 4.1 `info_collection_node` (graph.py:776)

The most complex node. Handles the full information-gathering lifecycle.

**CASE 0 — Prerequisite Consent Gate**
- If `prereq_gate.status == "proposed"`, detect user's yes/no response
- Accept → build accept patch, initialize prereq info collection
- Reject → mark gate rejected, continue without prereq

**CASE 1 — New Request Planning**
1. Check for resumable in-session duplicates (keyword overlap matching)
2. Pre-plan entity resolution: match known entities from DDB against user text
3. Generate collection plan via `llm_collection_plan()`:
   - Returns: `request_name`, `request_goal`, `request_type`, `subject_entity_id`, `key_info_needed`, `nice_to_have_info`, `routing_hint`
4. If profile facts exist, call `llm_summarize_and_ask()` for contextual follow-ups
5. Format questions and return with `awaiting_user_input=True`

**CASE 2/3 — User Providing Information**
1. Check for pending fact confirmations (MCP: `memory_get_pending_confirmations`)
2. If pending + user says yes/no → confirm/deny via MCP: `memory_confirm_fact`
3. Summarize response via `llm_info_collection_summarize()`:
   - Returns: `updated_summary`, `readiness_to_proceed`, `suggested_response`, `detected_prerequisites`, `user_signals`, `disputed_facts`
4. Track disputed facts: deprecate old values in fact_store
5. Pre-filter follow-up questions against stored facts (`filter_questions_with_known_facts`)
6. **Fact binding** via `bind_facts_from_summary()` (full extraction → resolution → reconciliation → write gate → commit pipeline)
7. If readiness = `"ready"`:
   - Set status = `"validated"`, `awaiting_user_input=False`
   - Hand off to execution agent via `pending_handoff`
8. If readiness = `"needs_more"` or `"can_proceed_but_incomplete"`:
   - Continue collecting, return `suggested_response`

### 4.2 `user_info_node` (graph.py:1784)

Read-only information retrieval + fact correction.

1. Fetch `context_bundle` for the entity (profile_facts + recent_events)
2. Fallback: search all known entities for keyword match in user text
3. Format facts/events as readable blocks
4. LLM generates answer to user's question
5. **Fact correction step** (added 2026-03-03):
   - Call `llm_extract_facts()` on user's message to detect corrections
   - If corrections found, call `llm_reconcile_facts()` against existing profile
   - Execute deprecations (e.g., `housing.city=Denver` when user says "she's in Florida now")
   - Write new/corrected facts via `upsert_fact()`
   - Entire block is non-blocking on failure

### 4.3 `deep_search_node` (graph.py:1555)

Multi-iteration web search with LLM-driven tool selection.

**Demo mode:** Single call to `nearby_providers` on DEMO_SERVER.

**Full mode:**
1. Build context from goal, collected info, conversation history, memory
2. LLM strategy call via `DEEP_SEARCH_STRATEGY_PROMPT` → selects first tool + args
3. Loop (up to 3 iterations):
   - Execute MCP tool on SEARCH_SERVER
   - LLM evaluates results via `SEARCH_CONTINUATION_PROMPT`
   - If `"done"` flag → break with summary
   - Otherwise → next tool call
4. Final synthesis via `SEARCH_SUMMARY_PROMPT`
5. Return artifact with `search_results` type

**Available search tools:**
| Tool | Cost | Use case |
|------|------|----------|
| `google_places_search` | Cheap | Location-based services, providers |
| `general_online_search_with_one_query` | Low | General knowledge, policy questions |
| `website_map` | Medium (Firecrawl) | Find relevant sub-pages on a domain |
| `scrape_multiple_websites_after_website_map` | High (Firecrawl) | Extract structured data from URLs |

### 4.4 `domain_expert_node` (graph.py:2045)

Domain knowledge synthesis.

**Demo mode:** Single `medicaid_checklist` tool call.

**Full mode:**
1. Build context from goal, collected info, memory
2. Web search via `general_online_search_with_one_query`
3. LLM synthesis via `DOMAIN_EXPERT_SYNTHESIS_PROMPT`
4. Return artifact with `written_guidance` type

### 4.5 `quick_answer_node` (graph.py:2147)

Lightweight Q&A. No request lifecycle (standalone).

1. Gather context from recent messages + memory block
2. LLM decision via `QUICK_ANSWER_DECISION_PROMPT`:
   - `direct_answer`: Answer from knowledge
   - `web_search`: Search then answer
   - `follow_up`: Generate clarifying questions (via `generate_follow_up_questions` MCP tool)
3. Execute action, return response

### 4.6 `front_end_node` (graph.py:194)

Emotional support for distressed caregivers.

1. Extract known_facts from `user_context.profile_snapshot`
2. Call `llm_emotional_support()` with user message, history, facts
3. Return empathetic response (2-4 sentences)

Style: Warm companion, validates and listens, normalizes without judgment. Not a therapist.

---

## 5. Memory Subsystem

### Architecture Overview

```
┌──────────────────────────────────────────────────┐
│                  Memory Layer                     │
│                                                   │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │FactStore │  │EventStore │  │CandidatePool │  │
│  │ (facts)  │  │ (events)  │  │ (new keys)   │  │
│  └────┬─────┘  └─────┬─────┘  └──────┬───────┘  │
│       │              │               │           │
│       └──────────────┼───────────────┘           │
│                      │                           │
│              ┌───────┴────────┐                  │
│              │ Context Bundle │                  │
│              │  (assembly)    │                  │
│              └───────┬────────┘                  │
│                      │                           │
│          ┌───────────┴──────────┐                │
│          │   Write Gate         │                │
│          │ (risk classification)│                │
│          └───────────┬──────────┘                │
│                      │                           │
│          ┌───────────┴──────────┐                │
│          │   Key Resolver       │                │
│          │ (BM25 + LLM select) │                │
│          └──────────────────────┘                │
└──────────────────────────────────────────────────┘
```

### FactStore (fact_store.py)

CRUD operations on `UserFactTable`.

| Method | Purpose |
|--------|---------|
| `get_active_facts(user_id, entity_id, fact_keys?)` | Query active facts for entity |
| `get_user_entities(user_id)` | Scan for all distinct entity_ids |
| `put_fact(fact: FactRecord)` | Store fact (caller manages deprecation) |
| `deprecate_fact(fact: FactRecord)` | Mark status=`deprecated`, update timestamp |
| `upsert_fact(user_id, entity_id, fact_key, new_value, ...)` | Smart upsert with conflict strategy |
| `get_all_active_facts_for_user(user_id)` | Scan all entities' facts |

**Upsert conflict strategies:**
- **Same value:** Update `last_seen_at` only (no version chain)
- **Different value + `overwrite`:** Deprecate old, create new as active
- **Different value + `needs_confirm` + high risk:** Keep old active, store new as `candidate`

### EventStore (event_store.py)

Episodic memory with DynamoDB TTL auto-expiry.

| Event Type | Default TTL | Purpose |
|------------|-------------|---------|
| `dialogue_summary` | 30 days | Conversation summaries |
| `tool_result` | 30 days | MCP tool outputs |
| `escalation` | 90 days | Human escalation records |
| `decision` | 90 days | Agent decision rationale |
| `memory_candidate` | 14 days | Facts awaiting user confirmation |
| `error` | 30 days | Error records |

### Context Bundle (context_bundle.py)

Assembles structured context for LLM prompts.

**`get_context_bundle(user_id, request_dict, message, k=5)`** returns:

```python
ContextBundle:
  user_id: str
  now: datetime
  active_request: ActiveRequestBlock | None
  profile_facts: Dict[fact_key → {value, confidence, last_verified_at,
                                    verification_level, source_type}]
  relevant_history: List[RequestSummaryItem]  # (TODO: vector search, Sprint 4)
  recent_events: List[EventItem]
  safety_notes: List[SafetyNote]  # High-risk/stale facts flagged
```

**Safety notes** flag facts that are:
- Unverified (`verification_level == "unverified"`)
- Stale (not re-confirmed in 90+ days)
- Rule: "High-risk fact — confirm before using"

### CandidateKeyPool (candidate_key_pool.py)

Global cross-user pool for discovering keys not yet in the YAML registry.

- When `KeyResolver` can't map a fact to a registry key, it records to the pool
- `occurrence_count` incremented atomically per observation
- Samples collected: values, entity_ids, request_ids (up to 5 each)
- Keys with high occurrence can be promoted to the registry

---

## 6. Fact Extraction & Reconciliation Pipeline

### Full Pipeline (slot_fact_binder.py)

```
bind_facts_from_summary()
    |
    v
Step 1: llm_extract_facts()
    → LLM extracts structured facts from info_collection summary
    → Each fact: {fact_key, fact_label, value, value_type, confidence,
                   source_type, evidence}
    → Filters: confidence >= 0.5, max 15 facts
    |
    v
Step 2: KeyResolver.resolve_batch()
    → Batch resolve all fact keys in single LLM call
    → Maps to canonical registry keys or records to candidate pool
    → Unknown keys: confidence capped at 0.65
    |
    v
Step 2b: llm_reconcile_facts()  [NEW — 2026-03-03]
    → Reconcile resolved facts against existing profile facts
    → Input: existing_profile_facts + resolved_facts + conversation context
    → Output: {deprecate_existing, discard_new, write_new}
    → Execute deprecations on DDB
    → Filter out discarded facts from resolved_facts
    → On failure: skip, continue with all resolved_facts
    |
    v
Step 3: propose_updates()
    → Risk-based classification into auto_patch / needs_confirm / reject
    |
    v
Step 4: commit_updates()
    → Write auto_patch facts to DDB + audit log
    |
    v
Step 4b: Namespace-based deprecation (safety net)
    → Same namespace + different key + different value + facet word overlap
    → OR key is in disputed_fact_keys
    → Deprecate old facts superseded by new ones
    |
    v
Step 5: Store needs_confirm as memory_candidate events
    → EventStore with event_type="memory_candidate", TTL 14 days
    |
    v
Step 6: Return slot_refs {fact_key → fact_id}
```

### Reconciliation Rules (RECONCILE_FACTS_PROMPT)

| Rule | Action | Example |
|------|--------|---------|
| **Contradictions** | DEPRECATE existing | `housing.city=Denver` + user says "she's in Miami" → deprecate |
| **Superseded concept** | DEPRECATE existing | `preference.massage_type=nuru` + new `preference.service_type=muscle recovery` |
| **Duplicates** | DEPRECATE less canonical | `housing.city=Chicago` + `housing.location=Chicago` → keep city |
| **Temporal values** | DEPRECATE/DISCARD | `move_date: last week` → meaningless after time passes |
| **Conversation artifacts** | DISCARD new | `information_request: care schedule` → describes conversation, not person |
| **Null/negative values** | DISCARD/DEPRECATE | `"none"`, `"N/A"`, `"unknown"` |
| **Confirmed facts** | DISCARD new | Same key + same value already stored |
| **No changes** | KEEP | Not mentioned ≠ contradicted |

### Where Reconciliation Runs

The reconciliation layer runs in **two locations**:

1. **`slot_fact_binder.py` Step 2b** — Inside the info_collection pipeline. Runs when `existing_profile_facts` is provided and new facts have been extracted + key-resolved.

2. **`user_info_node` in graph.py** — When user corrects facts during a `user_info` turn (e.g., "she's in Florida now, not Denver"). This was added because user corrections routed to `user_info` never passed through `bind_facts_from_summary`.

Both locations are non-blocking. On failure, the pipeline continues unchanged.

### Write Gate Classification (memory_write_gate.py)

```
            ┌─────────────────────────────────────────────────┐
            │              Risk from YAML Registry             │
            ├──────────┬──────────────────────────────────────┤
            │ LOW      │ conf >= 0.6 → auto_patch             │
            │          │ conf <  0.6 → needs_confirm          │
            ├──────────┼──────────────────────────────────────┤
            │ MEDIUM   │ conf >= 0.7 → auto_patch (unverified)│
            │          │ conf <  0.7 → needs_confirm          │
            ├──────────┼──────────────────────────────────────┤
            │ HIGH     │ tool/document + conf >= 0.8 → auto   │
            │          │ explicit_user_confirmed → auto        │
            │          │ otherwise → needs_confirm             │
            ├──────────┼──────────────────────────────────────┤
            │ Unknown  │ Treated as HIGH                       │
            └──────────┴──────────────────────────────────────┘
```

---

## 7. Key Resolution System

### Architecture: Two-Channel Recall + LLM Selection

```
Input: raw fact_key from LLM extraction (e.g., "mom's insurance plan")
    |
    ├─ Channel A: BM25 search over FactKey YAML registry
    │   → Top-K candidates by text similarity
    │
    ├─ Channel B: Alias table lookup (FactAliasTable)
    │   → Exact + prefix matching against learned aliases
    │
    v
Merge & Deduplicate (boost scores for keys in both channels)
    |
    v
High-confidence shortcut?
    → If top BM25 score > 5.0 AND ratio to 2nd > 3.0
    → Auto-return without LLM call
    |
    v
LLM Selection (multiple-choice, NOT free generation)
    → Claude picks from candidate list or returns "unknown"
    → CAN suggest new aliases, CANNOT invent canonical keys
    |
    v
KeyResolverResult:
    canonical_key, confidence, risk_level, decision,
    alias_observed, alias_suggested, reason
```

### BM25 Index (key_resolver.py)

Lightweight in-memory implementation. No external dependencies.

- **Parameters:** k1=1.5 (TF saturation), b=0.75 (length normalization)
- **Indexed fields per registry entry:** key, description, aliases, examples
- **Search:** Tokenize query, compute BM25 score per document, return top-K

### FactKey Registry (configs/factkey_registry_v0.yaml)

~130 canonical keys across 12 namespaces:

| Namespace | Keys | Risk | Examples |
|-----------|------|------|----------|
| `identity.*` | 9 | mixed | `full_name` (M), `dob` (H), `language_primary` (L) |
| `contact.*` | 6 | mostly H | `primary_phone`, `emergency_contact` |
| `insurance.*` | 10 | mostly H | `plan_type`, `member_id`, `coverage_notes` |
| `medical.*` | 10 | H | `conditions`, `allergies`, `cognitive_status` |
| `provider.*` | 7 | M | `primary_care.name`, `pharmacy.name` |
| `appointment.*` | 5 | M | `next.date_time`, `transportation_needed` |
| `medication.*` | 5 | H | `current_list`, `adherence.issues` |
| `mobility.*` | 4 | M | `assistive_devices`, `walking_limit` |
| `dailycare.*` | 7 | L-M | `adl.bathing`, `meal_prep`, `behavioral_triggers` |
| `preference.*` | 10 | L | `food.like`, `activity.like`, `routine.morning` |
| `finance.*` | 5 | H | `income.proof_available`, `out_of_pocket.monthly_estimate` |
| `legal.*` | 8 | H | `poa.medical`, `hipaa_release`, `guardianship` |

Each entry specifies: `key`, `description`, `risk_level`, `value_type`, `cardinality`, `aliases`, `examples`, `canonicalization`.

---

## 8. MCP Server Architecture

### Server Topology

```
Agent (graph.py)
    |
    ├─ MEMORY_SERVER ──> memory_mcp.py (14 tools)
    │     Transport: HTTP (if MCP_SERVER_URL set) or stdio
    │
    ├─ SEARCH_SERVER ──> online_search_mcp.py (4 tools)
    │     Transport: HTTP or stdio
    │
    ├─ FOLLOW_UP_SERVER ──> follow_up_mcp.py
    │     Transport: HTTP or stdio
    │
    └─ DEMO_SERVER ──> demo_mcp_server.py (testing only)
          Transport: stdio always
```

### Memory MCP Tools (memory_mcp.py)

| Tool | Purpose |
|------|---------|
| `memory_get_context_bundle` | Assemble full context (facts + events + profile) |
| `memory_get_profile_facts` | Get active facts for entity |
| `memory_resolve_fact_key` | Map text → canonical FactKey (BM25 + LLM) |
| `memory_resolve_fact_keys_batch` | Batch resolve multiple facts |
| `memory_propose_updates` | Run write gate classification |
| `memory_commit_updates` | Write facts to DDB + audit log |
| `memory_add_event` | Write episodic event to EventStore |
| `memory_resolve_entity_id` | Infer entity_id from user text |
| `memory_get_pending_confirmations` | Query `memory_candidate` events |
| `memory_confirm_fact` | Promote/reject candidate fact |
| `memory_write_back_aliases` | Store alias → key mappings |
| `memory_bind_facts` | Composite: extract → resolve → propose → commit |

### Search MCP Tools (online_search_mcp.py)

| Tool | Cost | Purpose |
|------|------|---------|
| `google_places_search` | Cheap | Location-based provider search (Google Places API) |
| `general_online_search_with_one_query` | Low | General web search (Firecrawl, 3 results) |
| `website_map` | Medium | Find sub-pages on a domain (Firecrawl) |
| `scrape_multiple_websites_after_website_map` | High | Extract data from up to 5 URLs (Firecrawl) |

### MCP Tool Usage by Node

| Node | Server | Tools Used |
|------|--------|------------|
| `info_collection` | MEMORY | `memory_get_pending_confirmations`, `memory_confirm_fact` |
| `deep_search` | SEARCH | `google_places_search`, `general_online_search_with_one_query`, `website_map`, `scrape_multiple_websites` |
| `deep_search` (demo) | DEMO | `nearby_providers` |
| `domain_expert` | SEARCH | `general_online_search_with_one_query` |
| `domain_expert` (demo) | DEMO | `medicaid_checklist` |
| `quick_answer` | SEARCH | `general_online_search_with_one_query` |
| `quick_answer` | FOLLOW_UP | `generate_follow_up_questions` |
| `user_info` | (none) | Direct DDB via FactStore (no MCP) |

### MCP FastAPI Server Structure (/mcp_server_fastapi/)

```
app/
├── main.py                    # FastMCP server, mounts sub-servers, /health, /ready
├── memory_mcp.py              # Memory tools sub-server
├── online_search_mcp.py       # Search tools sub-server
├── follow_up_mcp.py           # Follow-up question generation
├── google_maps_api.py         # Google Places API wrapper (+ GPT-4.1 domain extraction)
├── tools/
│   └── main_tools.py          # FileSystemProvider tools
└── memory/
    ├── ddb_client.py           # DynamoDB connection
    ├── fact_store.py           # Fact CRUD
    ├── event_store.py          # Event CRUD
    ├── key_resolver.py         # BM25 + LLM key resolution
    ├── memory_write_gate.py    # Risk classification
    ├── context_bundle.py       # Context assembly
    ├── entity_inference.py     # Entity ID extraction
    ├── id_utils.py             # UUID generation
    └── models/
        └── fact_models.py      # Pydantic models
```

---

## 9. LLM Functions & Prompt Inventory

### Prompt/Function Pairs in prompts.py

| Function | LLM Caller | Temperature | Max Tokens | Purpose |
|----------|-----------|-------------|------------|---------|
| `make_collection_plan_prompt` | `llm_collection_plan` | 0.3 | 1500 | Generate info collection plan |
| `make_info_collection_summarize_prompt` | `llm_info_collection_summarize` | 0.2 | 2000 | Evaluate collected info, decide readiness |
| `make_turn_mode_prompt` | `llm_turn_mode_decision` | 0.1 | 500 | Continuation vs new intent |
| `make_upstream_delegator_prompt` | `llm_upstream_delegation` | 0.2 | 1000 | Route new intents, manage request lifecycle |
| `make_fact_extraction_prompt` | `llm_extract_facts` | 0.2 | 2000 | Extract structured facts from summary |
| `RECONCILE_FACTS_PROMPT` | `llm_reconcile_facts` | 0.1 | 2000 | Reconcile new vs existing facts |
| `make_emotional_support_prompt` | `llm_emotional_support` | 0.7 | 400 | Empathetic caregiver support |
| — | `llm_prerequisite_acceptance_response` | 0.3 | 300 | Confirm prereq acceptance |
| — | `_infer_subject_entity` | 0.1 | 300 | Entity ID inference (LLM fallback) |

### Prompt Patterns in deep_search_prompts.py

| Constant | Used By | Purpose |
|----------|---------|---------|
| `TOOL_DESCRIPTIONS` | deep_search, domain_expert | Available tool descriptions |
| `DEEP_SEARCH_STRATEGY_PROMPT` | deep_search | Initial tool selection strategy |
| `SEARCH_CONTINUATION_PROMPT` | deep_search | Evaluate results, decide next tool |
| `SEARCH_SUMMARY_PROMPT` | deep_search | Final synthesis of search results |
| `DOMAIN_EXPERT_SYNTHESIS_PROMPT` | domain_expert | Synthesize domain knowledge |
| `QUICK_ANSWER_PROMPT` | quick_answer | Answer from knowledge or search |
| `QUICK_ANSWER_DECISION_PROMPT` | quick_answer | Decide: direct / search / follow-up |

### Fallback Heuristics

Every LLM function has a keyword-based fallback:

| LLM Function | Fallback |
|-------------|----------|
| `llm_collection_plan` | `default_collection_plan()` — keyword-inferred request type |
| `llm_info_collection_summarize` | `default_parse_user_answer()` — concatenation |
| `llm_turn_mode_decision` | `default_turn_mode_decision()` — keyword patterns |
| `llm_upstream_delegation` | `default_upstream_delegation()` — keyword heuristics |
| `llm_extract_facts` | Returns `[]` |
| `llm_reconcile_facts` | Returns `None` (skip reconciliation) |

---

## 10. DynamoDB Schema

### Table Inventory

| Table | PK | SK | Purpose |
|-------|----|----|---------|
| `WithCare_UserFactTable` | `USER#{user_id}#ENT#{entity_id}` | `FACT#{fact_key}#TS#{created_at}#{fact_id}` | Versioned facts |
| `WithCare_UserEventTable` | `USER#{user_id}` | `EVT#{timestamp}#{event_id}` | Episodic events (TTL-enabled) |
| `WithCare_FactAliasTable` | `ALIAS#{normalized_alias}` | `KEY#{canonical_key}` | Alias → canonical key mappings |
| `WithCare_MemoryFactLogTable` | `USER#{user_id}` | `FACTLOG#{timestamp}#{log_id}` | Audit trail |
| `WithCare_UserRequestTable` | — | — | Request persistence |
| `WithCare_CandidateKeyPool` | `CKEY#{normalized_key}` | `META` | Global unknown key tracking |

### FactRecord Fields

```
fact_id             : str           # UUID
user_id             : str
entity_id           : str           # "care_recipient:mom"
fact_key            : str           # "insurance.plan_type"
fact_label          : str           # "Insurance plan type"
fact_value          : Any           # The actual value
value_type          : str           # string|number|date|list|enum|object
status              : FactStatus    # active|deprecated|candidate
risk_level          : RiskLevel     # low|medium|high
confidence          : float         # 0.0-1.0
verification_level  : VerificationLevel  # explicit_user_confirmed|tool_verified|unverified
source_type         : SourceType    # user|tool|document|agent_inference
source_ref          : str           # request_id or tool_call_id
evidence            : str           # Quote from conversation (<=200 chars)
supersedes_fact_id  : str | None    # Version chain link
first_seen_at       : datetime
last_seen_at        : datetime
last_verified_at    : datetime | None
created_at          : datetime
updated_at          : datetime
```

---

## 11. Entity Inference

### Resolution Strategy (3 phases)

**Phase 0 — DDB entity matching**
- Fetch `known_entity_ids` from `FactStore.get_user_entities(user_id)`
- Match entity labels against user text (e.g., if DDB has `care_recipient:uncle-bob`, match "uncle" in text)

**Phase 1 — Keyword matching (fast, no LLM)**

English patterns:
```
"my mom" / "mother"              → care_recipient:mom
"my dad" / "father"              → care_recipient:dad
"husband" / "wife" / "spouse"    → care_recipient:spouse
"grandmother" / "grandpa"        → care_recipient:grandparent
"myself" / "my own"              → user:self
```

Chinese patterns:
```
"我妈" / "母亲" / "妈妈" / "我娘"   → care_recipient:mom
"我爸" / "父亲" / "爸爸" / "我爹"   → care_recipient:dad
"老公" / "老婆" / "配偶" / "丈夫"   → care_recipient:spouse
"奶奶" / "外婆" / "爷爷" / "外公"   → care_recipient:grandparent
```

Note: Grandparent variants checked FIRST to prevent "grandmother" matching "mother".

**Phase 2 — LLM fallback**
- Claude picks from known entities or returns `care_recipient:unknown`
- Only called when keyword matching fails and an LLM client is available

### Entity ID Format

```
<entity_type>:<relation_or_id>

Examples:
  care_recipient:mom
  care_recipient:dad
  care_recipient:spouse
  care_recipient:grandparent
  care_recipient:uncle-bob    (custom, from DDB)
  user:self
  care_recipient:unknown
```

---

## 12. Configuration & Environment

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEEP_SEARCH_MODE` | `"full"` | `"demo"` for testing, `"full"` for production |
| `MCP_SERVER_URL` | `None` | Remote MCP server URL (HTTP transport). If unset, uses stdio. |
| `AWS_REGION` | `"us-east-2"` | DynamoDB region |
| `WITHCARE_DDB_PREFIX` | `"WithCare_"` | DynamoDB table name prefix |
| `WITHCARE_DDB_OFF` | `"0"` | Set to `"1"` to disable DDB (log-only mode) |
| `ANTHROPIC_API_KEY` | — | Claude API key |

### LLM Client

**Model:** `claude-sonnet-4-20250514` (via `TrackedAnthropicClient`)

**Tracing:** Langfuse integration for all LLM calls. Traces include:
- Session ID, agent role, user ID
- Input/output token counts
- Duration in milliseconds
- Error logging

### Language Detection

```python
def _detect_user_language(state) -> str:
    text = last_user_text(state)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return "zh" if chinese_chars > max(1, len(text) * 0.1) else "en"
```

If >10% of characters are CJK → Chinese. Otherwise → English.

### User Input Detection

```python
# Yes/No detection
yes_words = ["要", "好", "可以", "是", "yes", "ok", "sure"]
no_words  = ["不要", "不", "否", "no", "not now"]

# Decline more questions
decline_words = ["不想回答", "别问了", "先用现有的", "不用问了",
                 "proceed", "go ahead", "先继续"]
```

---

## 13. Development Log — Fact Reconciliation

### Problem (2026-03-03)

The fact correction system had **5 independent mechanisms** that each caught some corrections but missed others:

1. `disputed_fact_keys` tracking in info_collection summarization
2. Manual deprecation of disputed facts in info_collection (CASE 2/3)
3. Namespace-based facet overlap deprecation in `slot_fact_binder.py` (Step 4b)
4. `upsert_fact()` conflict strategy (overwrite vs needs_confirm)
5. Write gate risk classification

**Result:** Contradictory active facts persisting (e.g., `housing.city=Denver` alongside `identity.address=Miami`), conversation artifacts stored as facts (e.g., `preference.information_request=["care schedule"]`), and relative timestamps (`housing.move_date=last week`) never cleaned up.

### Solution: Additive Reconciliation Layer

Added `llm_reconcile_facts()` as an LLM-powered reconciliation step that considers the full picture: all existing facts, all new facts, and conversation context.

### Changes Made

**1. `prompts.py` — New prompt + function**

- `RECONCILE_FACTS_PROMPT`: 8-rule prompt covering contradictions, superseded concepts, duplicates, temporal values, conversation artifacts, null values, confirmed facts, and no-change defaults. 3 worked examples. Strict JSON output.
- `llm_reconcile_facts()`: Async function, temperature=0.1, returns `{deprecate_existing, discard_new, write_new}` or `None` on failure.

**2. `slot_fact_binder.py` — Step 2b insertion**

Between key resolution (Step 2) and write gate (Step 3):
- Calls `llm_reconcile_facts()` with existing profile facts, resolved new facts, and conversation history
- Executes deprecations returned in `deprecate_existing`
- Filters out facts in `discard_new` from the resolved_facts list
- Non-blocking: on failure, all resolved_facts pass through unchanged
- Step 4b namespace-based deprecation retained as safety net

**3. `graph.py` — Fact correction in `user_info_node`**

The reconciliation in `slot_fact_binder.py` only runs through the `info_collection` pipeline. But fact corrections routed to `user_info` (e.g., "she's in Florida now, not Denver") bypassed it entirely.

Added a fact-correction block to `user_info_node`:
1. `llm_extract_facts()` on the user's message to detect corrections
2. `llm_reconcile_facts()` against existing profile
3. Execute deprecations via `FactStore.deprecate_fact()`
4. Write new/corrected facts via `FactStore.upsert_fact()`
5. Entire block wrapped in try/except (non-blocking)

### Test Results

- `test_08_slot_fact_binding.py`: 57 passed, 0 failed
- `test_05_key_resolver.py`: 26 passed, 0 failed
- `prompts.py` imports cleanly
- `graph.py` syntax validated

### Verification Checklist

- [ ] Restart server, use test-user-005
- [ ] Reset conversation, ask "what do you know about my uncle?"
- [ ] Say "he doesn't want muscle recovery anymore, he needs deep tissue massage"
- [ ] Check DDB: old `preference.massage_type` and `preference.service_type` deprecated
- [ ] Check: no `preference.information_request` or artifacts stored
- [ ] Test with mom: "she lives in Miami now, not Denver" → Denver facts deprecated
- [ ] Confirm existing mechanisms still work as safety nets (Step 4b, upsert conflicts)

---

## Appendix: Request Lifecycle States

```
created → collecting → validated → executing → executed → completed
                  ↓                     ↑
               paused ──────────────────┘
                  ↓
               aborted
```

| Status | Meaning |
|--------|---------|
| `created` | Request just created, no collection started |
| `collecting` | Actively gathering information |
| `validated` | Info collection complete, ready for execution |
| `executing` | Being handled by execution agent |
| `executed` | Agent produced results |
| `paused` | Waiting for prerequisite or user to resume |
| `completed` | Archived |
| `aborted` | Cancelled |

Every transition is recorded in `stage_history` with `{from_stage, to_stage, agent, timestamp, reason}`.
