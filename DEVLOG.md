# Development Log

## 2026-02-22 — Local Chat Interface for WithCare Agent

### What was built

A browser-based chat UI for interactive multi-turn conversations with the WithCare agent, replacing the need to write individual test scenarios in `test_post_execution.py` for behavior iteration.

**Architecture:** `Browser (vanilla JS) <-> FastAPI (port 8000) <-> WithCare agent graph`

### Files created

- **`chat_server.py`** — FastAPI backend (~140 lines)
  - `POST /chat` — sends user message through the agent graph, returns reply + debug info
  - `POST /reset` — clears conversation state
  - `GET /health` — health check
  - `GET /` — serves the frontend
  - Reuses `run_turn_async` pattern from `executor.py` / test files: `apply_node_output()`, `_consume_ddb_writes()`, `_cleanup_transient()`, `check_prereq_lifecycle()`
  - In-memory `{conversation_id: UnifiedState}` store, lazy graph init, CORS enabled
  - Captures per-node routing decisions (turn_mode, turn_reason, llm_recommended_agent, pending_handoff, delegator_debug) for the debug panel
  - Try/except around graph execution — errors returned as readable messages instead of 500s

- **`chat_ui/index.html`** — Single-file vanilla JS/HTML/CSS frontend (~320 lines)
  - Adapted visual design from `langgraph-kafka-k8s/frontend/` (purple gradient, message bubbles, fade-in animations)
  - Removed: SSE/EventSource, Kafka, React/Vite, endpoint selector, connection status, user_id management
  - Added: collapsible debug panel (open by default), reset button, loading spinner
  - Debug panel shows: nodes visited, current agent, turn mode + reason, active request status, per-node routing decisions, error info
  - Input: Enter to send, Shift+Enter for newline, auto-resize textarea

- **`requirements.txt`** — Added `fastapi>=0.115.0` and `uvicorn>=0.30.0`

### Issues encountered and fixed

1. **Missing `load_dotenv()`** — First run failed with `ValueError: ANTHROPIC_API_KEY must be set`. Added `from dotenv import load_dotenv; load_dotenv()` at the top of `chat_server.py`, matching the pattern in `test_post_execution.py`.

2. **Insufficient debug visibility** — Initial debug panel was collapsed by default and only showed node names. Expanded it to show per-node routing decisions (what `turn_router` decided, what `upstream_delegator` set in `pending_handoff`) and `turn_reason` from the LLM. Panel now open by default. This was critical for diagnosing routing behavior.

### Verified working

- Server starts cleanly on `http://localhost:8000`
- First message routed correctly: `turn_router -> upstream_delegator -> info_collection -> downstream_catcher`
- LLM routing decisions visible in debug panel (turn_mode, recommended_agent, delegator action_type)
- "Milvus client not configured" warning is expected (vector DB for similar-request lookup, returns empty in local mode)

### How to run

```bash
cd /Users/xyxg025/withcare_agent_arch_codebase_v3
pip install fastapi uvicorn
python chat_server.py
# Open http://localhost:8000
```

---

## 2026-02-22 — Add `quick_answer` Agent + Refine Routing for All Three Execution Agents

### Problem

The graph had two execution agents (`deep_search`, `domain_expert`) with overlapping responsibilities. Domain knowledge questions like "What is Medicaid?" were misrouted to `deep_search`. There was no lightweight path for simple factual questions — every user message had to go through the full request lifecycle (info_collection → execution agent), even for questions that just need a direct answer.

### What was built

Added a third execution agent, `quick_answer`, with clearly differentiated roles:

| Agent | Purpose | Key trait |
|-------|---------|-----------|
| `deep_search` | Find providers/agencies/facilities near a specific location | Google Places + web scraping; needs precise location |
| `domain_expert` | Generate long-form deliverables: guidance docs, emails, checklists | Deep writer; produces structured text artifacts |
| `quick_answer` | Answer quick factual questions, definitions, eligibility info | Fast; no request created; bypasses info_collection |

`quick_answer` is fundamentally different from the other two: it's not a "request" — it answers standalone or mid-conversation questions without going through info_collection or creating a request record.

### Files modified

- **`state_models.py`** — Added `"quick_answer"` to `AgentName` Literal type
- **`graph.py`** — 7 changes:
  - Added `"quick_answer"` to `RouteKey` Literal
  - Added `quick_answer_node` function (~50 lines): extracts user question, optionally calls `general_online_search` via MCP if the question needs factual data, calls Claude to generate a concise answer, returns message with no artifact or request creation
  - Updated `route_from_turn_router`: quick_answer bypasses `upstream_delegator` even on `new_intent` — routes directly to the node
  - Updated `route_from_delegator`: quick_answer skips the info_collection guard (like `front_end_emotional_support`)
  - Updated `route_after_catcher`: added quick_answer mapping
  - Updated `build_graph()`: wired `quick_answer` node into all three conditional edge dicts + the downstream_catcher edge loop
  - Updated `info_collection_node` handoff: respects upstream `pending_handoff.recommended_next_agent` when present, falling back to `routing_hint`
- **`prompts.py`** — 3 changes:
  - `make_turn_mode_prompt`: added `quick_answer` to agent_descriptions, refined `deep_search` and `domain_expert` descriptions for clearer differentiation
  - `make_upstream_delegator_prompt`: same agent_descriptions update
  - `make_collection_plan_prompt`: updated routing_hint guidance — deep_search now requires specific location in `key_info_needed`, added note that quick factual questions are handled by quick_answer before info_collection
- **`deep_search_prompts.py`** — Added `QUICK_ANSWER_PROMPT` constant

### Routing bug found and fixed

**Problem:** When `turn_router` classified a quick question as `new_intent` (e.g., asking "What is Medicaid?" mid-conversation), `route_from_turn_router` unconditionally sent `new_intent` to `upstream_delegator`. The upstream_delegator would then create a new request before routing to quick_answer — defeating the purpose of a lightweight path.

**Fix:** Added an early intercept in `route_from_turn_router`: if `mode == "new_intent"` and `llm_recommended_agent == "quick_answer"`, return `"quick_answer"` directly, bypassing `upstream_delegator` entirely. This ensures quick_answer never triggers request creation regardless of which path it's routed from.

### Language adaptation

Also fixed hardcoded Chinese strings throughout the codebase. The agent was always responding in Chinese regardless of user language.

**Changes:**
- Added `_detect_user_language(state)` helper in `graph.py` that checks Chinese character ratio in the last user message
- Added `_msg(state, zh, en)` convenience function for bilingual hardcoded messages
- Updated all ~15 hardcoded Chinese assistant messages in `graph.py` to be bilingual
- Updated all LLM prompt instructions in `deep_search_prompts.py` from "Respond in Chinese" to "Match the language the user is writing in"
- Added language-matching instructions to key prompts in `prompts.py` (collection plan questions, suggested responses)

### Flow examples

**"What is Medicaid?" (quick factual question):**
`turn_router` → recommends `quick_answer` → `route_from_turn_router` returns `"quick_answer"` → `quick_answer_node` answers directly → `downstream_catcher` → END. No info_collection, no request created.

**"Help me find a caregiver in Chicago Oak Park" (location-based search):**
`turn_router` → `new_intent` → `upstream_delegator` → recommends `deep_search` → `info_collection` (asks for specific neighborhood) → `deep_search` uses Google Places.

**"Help me write a letter to the care facility" (deliverable):**
`turn_router` → `new_intent` → `upstream_delegator` → recommends `domain_expert` → `info_collection` → `domain_expert` generates letter.

**"Does Medicaid have age limits?" (mid-conversation question):**
`turn_router` → recommends `quick_answer` → routes directly even during active request → answers without disturbing request state.
