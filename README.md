# WithCare Agent Architecture Scaffold (Generalized Info Collection)

This codebase is a **general** scaffold for your agentic architecture:

- Unified `UnifiedState` (Pydantic v2) for all agents + delegators + request queue
- LangGraph routing graph
- Nodes return **small dict patches** only
- A single merge reducer: `merge_utils.apply_node_output()`
- **Generalized Info Collection**:
  - For a new request: generate a “collection plan” (to-collect slots + prerequisites)
  - For user replies: parse answers into structured slots + ask remaining questions
  - Optional hooks for semantic retrieval of prior Q/A and similar requests

---

## Troubleshooting / recent fixes

### Symptoms

- `PydanticSerializationUnexpectedValue(...)` warnings during `python executor.py`.
- The assistant repeatedly asks the same “关键问题” every turn, and a new request id is created on each turn.

### Fixes applied

- **Normalize nested Pydantic fields during merge** (`merge_utils.py`)
  - When node patches provide dicts for nested models, convert them via `model_validate(...)` before assignment:
    - `routing.pending_handoff` -> `Handoff`
    - `request_manager.requests[*].prereq_gate` -> `PrereqGate`
    - `request_manager.requests[*].deep_search_state` -> `DeepSearchState`
  - This removes serializer warnings and keeps state types stable.

- **Prevent request patches from overwriting list fields managed by helper merges** (`merge_utils.py`)
  - `open_questions`, `slots`, `artifacts` are merged via `open_questions_upsert` / `slots_upsert` / `artifacts_append`.
  - When merging `request_manager.requests[*]` patches, we skip direct assignment of those list fields to avoid wiping merged data.
  - Without this, `build_request_patch()`'s default empty lists could overwrite `open_questions_upsert`, causing `open_questions_len=0` and breaking the “answering questions” flow.

- **Provide an explicit LangGraph state schema** (`graph.py`)
  - `StateGraph(GraphState)` (TypedDict) is used instead of `StateGraph(dict)`.
  - This ensures top-level keys like `request_manager` are preserved across nodes/turns, preventing the graph from behaving like every turn is a brand new request.

### Quick verification

- Run `python executor.py` and confirm:
  - Warnings are gone.
  - Turn 2 no longer re-creates a new request, and the assistant does not repeat the initial question list.

> The previous v2 demo used “home care nearby” as a concrete example.  
> **v3 removes domain-specific assumptions** and keeps everything generic.

---

## Run

```bash
unzip withcare_agent_arch_codebase_v3.zip -d withcare_agent_arch
cd withcare_agent_arch

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python executor.py
```

---

## What changed vs v2

### 1) Info Collection is now general-purpose
`graph.py: info_collection_node` now supports:

- **New request**: build request record + ask a plan-driven question list
- **Answering existing questions**: parse reply into slots + ask missing questions
- **Prerequisite “插队”**:
  - The plan can propose prerequisites
  - User confirms yes/no
  - If yes: create/prioritize prereq request + push old request into queue

### 2) Prompt interfaces are separated into `prompts.py`
You asked for “it should be prompts, not hardcoded logic.”  
So v3 introduces:

- `make_collection_plan_prompt(...)`
- `make_parse_user_answer_prompt(...)`

For now, the graph runs with deterministic fallbacks:

- `default_collection_plan(...)`
- `default_parse_user_answer(...)`

You’ll replace those with your actual LLM calls.

### 3) Retrieval hooks live in `retrieval.py`
Two async stubs you can later back with Milvus/Zilliz/DynamoDB/Snowflake:

- `fetch_similar_requests(user_id, request_text, top_k)`
- `fetch_prior_qas_for_questions(user_id, questions, top_k)`

---

## File-by-file (what each script does)

### `state_models.py`
Defines the unified state schema:
- `messages`, `routing`, `request_manager`, `tools`, `user_context`, etc.

### `merge_utils.py`
Applies a node’s output patch into the canonical `UnifiedState`.
Understands helper keys:
- `open_questions_upsert`
- `slots_upsert`
- `artifacts_append`

### `prompts.py`
Holds prompt templates + deterministic fallback planners/parsers.
Replace fallback functions with your LLM call.

### `retrieval.py`
Semantic/history retrieval interfaces (stubs).

### `request_factory.py`
Request lifecycle primitives as patches:
- create request (also produces a `ddb_writes` record shape)
- propose prereq “insert”
- accept prereq (queue parent, switch active request)

### `mcp_wrappers.py` + `servers/demo_mcp_server.py`
Demo MCP tooling via fastmcp:
- async wrapper returns standardized `tools.tool_runs` / `tools.tool_failures` patches
- server returns placeholder tool outputs (no domain assumptions)

### `graph.py`
The LangGraph routing structure:
- `turn_router` -> `upstream_delegator`/current agent -> `downstream_catcher`
- `info_collection_node` is the core generalized collection flow
- `deep_search_node` / `domain_expert_node` are demo placeholders that call MCP tools

### `executor.py`
Runs a few demo turns, streams graph execution, merges patches.

---

## Next steps you’ll likely implement

1) Replace `default_collection_plan` with an LLM call that outputs:
   - `request_name`, `request_goal`
   - `to_collect` list
   - `prerequisites` list
   - `routing_hint` (deep_search/domain_expert/user_info/...)

2) Replace `default_parse_user_answer` with an LLM call that outputs:
   - normalized slots (key/value)
   - still_missing questions

3) Wire real persistence:
   - consume `ddb_writes` to actually write to DynamoDB

4) Wire real search tools for `deep_search`:
   - Google Maps / Firecrawl / web search
