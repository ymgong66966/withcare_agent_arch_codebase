Plan: Daily Cron Job System — Fact Validation + Request Hygiene                                                                                                                    │
│                                                                                                                                                                                    │
│ Context                                                                                                                                                                            │
│                                                                                                                                                                                    │
│ The WithCare agent accumulates facts and requests throughout daily conversations, but has no end-of-day reconciliation to ensure data quality. Currently:                          │
│                                                                                                                                                                                    │
│ - Facts can become stale, duplicated across keys, or contain conversation artifacts that slipped past real-time filters                                                            │
│ - Candidate keys (proposed but not in registry) accumulate in CandidateKeyPool with no promotion review                                                                            │
│ - Requests are written to DDB on creation (build_request_patch) but status/field updates happen only in-memory — paused, completed, and queued statuses are never synced back to   │
│ DDB                                                                                                                                                                                │
│ - Conversations are in-memory only (_conversations dict in chat_server.py) and lost on pod restart — the cron job needs access to daily messages                                   │
│                                                                                                                                                                                    │
│ Existing code provides a foundation: daily_analyzer.py (request retrospective analysis), historical_store.py (archival to DDB/Milvus), and historical_models.py (data models).     │
│ These handle request archival but not fact validation or request hygiene.                                                                                                          │
│                                                                                                                                                                                    │
│ Scope — 3 Workstreams                                                                                                                                                              │
│                                                                                                                                                                                    │
│ WS-1: Conversation Persistence (prerequisite for everything)                                                                                                                       │
│                                                                                                                                                                                    │
│ WS-2: Daily Fact Cross-Validation & Candidate Key Review                                                                                                                           │
│                                                                                                                                                                                    │
│ WS-3: Daily Request Hygiene & Status Sync                                                                                                                                          │
│                                                                                                                                                                                    │
│ ---                                                                                                                                                                                │
│ WS-1: Conversation Persistence                                                                                                                                                     │
│                                                                                                                                                                                    │
│ Problem                                                                                                                                                                            │
│                                                                                                                                                                                    │
│ Raw messages live only in _conversations dict (in-memory). Pod restart = total loss. The cron job needs the day's conversation to do fact validation and request analysis.         │
│                                                                                                                                                                                    │
│ Design                                                                                                                                                                             │
│                                                                                                                                                                                    │
│ New DynamoDB table: WithCare_UserConversationTable                                                                                                                                 │
│                                                                                                                                                                                    │
│ PK: CONV#{conversation_id}                                                                                                                                                         │
│ SK: MSG#{timestamp_iso}#{message_id}                                                                                                                                               │
│                                                                                                                                                                                    │
│ Fields:                                                                                                                                                                            │
│   conversation_id, user_id, message_id                                                                                                                                             │
│   role: "user" | "assistant"                                                                                                                                                       │
│   content: str                                                                                                                                                                     │
│   timestamp: datetime                                                                                                                                                              │
│   metadata: {agent, request_id, ...}  (optional)                                                                                                                                   │
│                                                                                                                                                                                    │
│ GSI1 (query by user+date):                                                                                                                                                         │
│   gsi1pk: USER#{user_id}#DATE#{YYYY-MM-DD}                                                                                                                                         │
│   gsi1sk: MSG#{timestamp_iso}#{message_id}                                                                                                                                         │
│                                                                                                                                                                                    │
│ File Changes                                                                                                                                                                       │
│                                                                                                                                                                                    │
│ 1. scripts/create_dynamodb_tables.py — Add table creation                                                                                                                          │
│                                                                                                                                                                                    │
│ Add WithCare_UserConversationTable with PK/SK schema + GSI1. Same pattern as existing table creation code.                                                                         │
│                                                                                                                                                                                    │
│ 2. ddb_client.py — Add table name constant                                                                                                                                         │
│                                                                                                                                                                                    │
│ USER_CONVERSATION_TABLE = f"{TABLE_PREFIX}UserConversationTable"                                                                                                                   │
│                                                                                                                                                                                    │
│ 3. conversation_store.py — New file (~80 lines)                                                                                                                                    │
│                                                                                                                                                                                    │
│ class ConversationStore:                                                                                                                                                           │
│     async def write_message(user_id, conversation_id, role, content, metadata=None)                                                                                                │
│     async def get_messages_for_conversation(conversation_id, limit=500)                                                                                                            │
│     async def get_messages_for_user_date(user_id, date_str) -> List[Dict]                                                                                                          │
│                                                                                                                                                                                    │
│ Follows same patterns as fact_store.py and event_store.py (lazy DDB init, non-blocking on failure, _serialize_item for type safety).                                               │
│                                                                                                                                                                                    │
│ 4. chat_server.py — Write messages on each turn                                                                                                                                    │
│                                                                                                                                                                                    │
│ In _run_turn(), after the user message is applied and after each assistant message, call conversation_store.write_message(). Non-blocking (fire-and-forget with try/except). ~10   │
│ lines added.                                                                                                                                                                       │
│                                                                                                                                                                                    │
│ ---                                                                                                                                                                                │
│ WS-2: Daily Fact Cross-Validation & Candidate Key Review                                                                                                                           │
│                                                                                                                                                                                    │
│ Problem                                                                                                                                                                            │
│                                                                                                                                                                                    │
│ Real-time extraction catches most things, but over a day's conversation:                                                                                                           │
│ - Facts that were disputed/corrected may not have been fully cleaned up                                                                                                            │
│ - Conversation artifacts may have slipped through (insurance.clarification_needed, preference.information_request)                                                                 │
│ - Relative timestamps persist (housing.move_date: last week)                                                                                                                       │
│ - Candidate keys may duplicate existing registry keys under different names                                                                                                        │
│                                                                                                                                                                                    │
│ Design                                                                                                                                                                             │
│                                                                                                                                                                                    │
│ Two LLM-powered sub-workflows, orchestrated by a new daily_fact_job.py.                                                                                                            │
│                                                                                                                                                                                    │
│ Sub-workflow A: Fact Cross-Validation                                                                                                                                              │
│                                                                                                                                                                                    │
│ Input:                                                                                                                                                                             │
│   - Day's conversation (from UserConversationTable via GSI1)                                                                                                                       │
│   - All active facts for user (from UserFactTable, all entities)                                                                                                                   │
│   - Full FactKey registry structure (namespace list + key list)                                                                                                                    │
│                                                                                                                                                                                    │
│ LLM Prompt (DAILY_FACT_VALIDATION_PROMPT):                                                                                                                                         │
│   "Given ALL active facts and today's conversation, identify facts that are:                                                                                                       │
│    1. CONTRADICTED by what the user said today                                                                                                                                     │
│    2. STALE relative timestamps ('last week', 'next month', 'tomorrow')                                                                                                            │
│    3. CONVERSATION ARTIFACTS (describe the conversation, not the person)                                                                                                           │
│    4. NULL/MEANINGLESS values ('none', 'N/A', 'unknown', empty)                                                                                                                    │
│    5. DUPLICATES across different keys (same info stored twice)                                                                                                                    │
│                                                                                                                                                                                    │
│    DO NOT flag facts just because they weren't mentioned today.                                                                                                                    │
│    Only flag things that are clearly wrong, stale, or artifacts.                                                                                                                   │
│                                                                                                                                                                                    │
│    Output JSON:                                                                                                                                                                    │
│    {                                                                                                                                                                               │
│      deprecate: [{entity_id, fact_key, reason}],                                                                                                                                   │
│      flag_for_review: [{entity_id, fact_key, concern, severity}]                                                                                                                   │
│    }"                                                                                                                                                                              │
│                                                                                                                                                                                    │
│ Execute:                                                                                                                                                                           │
│   - deprecate: call FactStore.deprecate_fact() for each                                                                                                                            │
│   - flag_for_review: write to MemoryFactLogTable with result="needs_review"                                                                                                        │
│                                                                                                                                                                                    │
│ Sub-workflow B: Candidate Key Review                                                                                                                                               │
│                                                                                                                                                                                    │
│ Input:                                                                                                                                                                             │
│   - Top candidate keys from CandidateKeyPool (occurrence_count >= 3)                                                                                                               │
│   - Full FactKey registry (all namespaces, keys, aliases)                                                                                                                          │
│   - Sample values and entities from candidate records                                                                                                                              │
│                                                                                                                                                                                    │
│ LLM Prompt (CANDIDATE_KEY_REVIEW_PROMPT):                                                                                                                                          │
│   "Review these proposed fact keys that users have been extracting but                                                                                                             │
│    don't exist in the official registry. For each candidate:                                                                                                                       │
│                                                                                                                                                                                    │
│    1. DUPLICATE: Maps to existing registry key → suggest alias                                                                                                                     │
│    2. NEW_KEY: Genuinely new concept → propose namespace.facet, risk_level, value_type                                                                                             │
│    3. REJECT: Conversation artifact or not a real fact type → mark rejected                                                                                                        │
│                                                                                                                                                                                    │
│    Output JSON:                                                                                                                                                                    │
│    {                                                                                                                                                                               │
│      alias_mappings: [{candidate_key, canonical_key, suggested_aliases}],                                                                                                          │
│      proposed_new_keys: [{key, description, risk_level, value_type, namespace, reason}],                                                                                           │
│      reject: [{candidate_key, reason}]                                                                                                                                             │
│    }"                                                                                                                                                                              │
│                                                                                                                                                                                    │
│ Execute:                                                                                                                                                                           │
│   - alias_mappings: write to FactAliasTable via FactStore.put_alias()                                                                                                              │
│   - proposed_new_keys: write to a review queue (event with type="key_proposal")                                                                                                    │
│   - reject: update CandidateKeyPoolStore status → "rejected"                                                                                                                       │
│                                                                                                                                                                                    │
│ File Changes                                                                                                                                                                       │
│                                                                                                                                                                                    │
│ 1. daily_fact_prompts.py — New file (~200 lines)                                                                                                                                   │
│                                                                                                                                                                                    │
│ - DAILY_FACT_VALIDATION_PROMPT template                                                                                                                                            │
│ - CANDIDATE_KEY_REVIEW_PROMPT template                                                                                                                                             │
│ - llm_daily_fact_validation(client, active_facts, conversation, registry_summary) → Dict                                                                                           │
│ - llm_candidate_key_review(client, candidates, registry_summary) → Dict                                                                                                            │
│                                                                                                                                                                                    │
│ Follows same pattern as prompts.py (template + async LLM caller, JSON extraction, non-blocking).                                                                                   │
│                                                                                                                                                                                    │
│ 2. daily_fact_job.py — New file (~250 lines)                                                                                                                                       │
│                                                                                                                                                                                    │
│ class DailyFactJob:                                                                                                                                                                │
│     async def run(self, user_id: str, date: str) -> DailyFactReport:                                                                                                               │
│         """Main entry point for daily fact validation."""                                                                                                                          │
│         # 1. Fetch day's conversation from ConversationStore                                                                                                                       │
│         # 2. Fetch all active facts from FactStore                                                                                                                                 │
│         # 3. Run fact cross-validation (Sub-workflow A)                                                                                                                            │
│         # 4. Execute deprecations + log flags                                                                                                                                      │
│         # 5. Fetch top candidates from CandidateKeyPool                                                                                                                            │
│         # 6. Run candidate key review (Sub-workflow B)                                                                                                                             │
│         # 7. Execute alias writes + rejections                                                                                                                                     │
│         # 8. Return report                                                                                                                                                         │
│                                                                                                                                                                                    │
│     async def _run_fact_validation(self, ...) -> Dict                                                                                                                              │
│     async def _execute_fact_actions(self, ...) -> int                                                                                                                              │
│     async def _run_candidate_review(self, ...) -> Dict                                                                                                                             │
│     async def _execute_candidate_actions(self, ...) -> int                                                                                                                         │
│                                                                                                                                                                                    │
│ 3. daily_fact_models.py — New file (~50 lines)                                                                                                                                     │
│                                                                                                                                                                                    │
│ class DailyFactReport(BaseModel):                                                                                                                                                  │
│     user_id: str                                                                                                                                                                   │
│     date: str                                                                                                                                                                      │
│     facts_deprecated: int                                                                                                                                                          │
│     facts_flagged: int                                                                                                                                                             │
│     candidates_aliased: int                                                                                                                                                        │
│     candidates_proposed: int                                                                                                                                                       │
│     candidates_rejected: int                                                                                                                                                       │
│                                                                                                                                                                                    │
│ ---                                                                                                                                                                                │
│ WS-3: Daily Request Hygiene & Status Sync                                                                                                                                          │
│                                                                                                                                                                                    │
│ Problem                                                                                                                                                                            │
│                                                                                                                                                                                    │
│ 1. Status drift: Requests are written to DDB on creation, but in-memory status changes (paused, completed, etc.) are never synced back. DDB shows status: created for completed    │
│ requests.                                                                                                                                                                          │
│ 2. Stale queued requests: Quick Q&A or abandoned topics stay in pending_queue forever.                                                                                             │
│ 3. Missing summaries: Request fields like summary_current aren't updated after info collection completes.                                                                          │
│                                                                                                                                                                                    │
│ Design: Two parts                                                                                                                                                                  │
│                                                                                                                                                                                    │
│ Part A: Real-Time Request DDB Sync (fixes the root cause)                                                                                                                          │
│                                                                                                                                                                                    │
│ Currently build_request_patch() generates ddb_writes for creation. But graph.py modifies request fields in-memory without generating DDB writes. We add a helper that generates    │
│ update writes, called from key state transitions.                                                                                                                                  │
│                                                                                                                                                                                    │
│ Part B: Daily Request Retrospective (builds on existing daily_analyzer.py)                                                                                                         │
│                                                                                                                                                                                    │
│ Extend the existing DailyRequestAnalyzer with a hygiene pass that:                                                                                                                 │
│ 1. Loads all requests created today from DDB (via GSI1)                                                                                                                            │
│ 2. Loads the day's conversation from ConversationStore                                                                                                                             │
│ 3. LLM analyzes each request against the full conversation to determine:                                                                                                           │
│   - Should pending/queued requests be closed? (ad-hoc questions, user moved on)                                                                                                    │
│   - Should summaries be updated with final state?                                                                                                                                  │
│   - Are any paused requests actually abandoned?                                                                                                                                    │
│ 4. Writes updated status/fields back to DDB                                                                                                                                        │
│                                                                                                                                                                                    │
│ File Changes                                                                                                                                                                       │
│                                                                                                                                                                                    │
│ Part A: Request DDB sync                                                                                                                                                           │
│                                                                                                                                                                                    │
│ 1. request_factory.py — Add build_request_update_patch()                                                                                                                           │
│                                                                                                                                                                                    │
│ def build_request_update_patch(                                                                                                                                                    │
│     *,                                                                                                                                                                             │
│     user_id: str,                                                                                                                                                                  │
│     request_id: str,                                                                                                                                                               │
│     updates: Dict[str, Any],  # {status, stage_detail, summary_current, ...}                                                                                                       │
│     table_name: str = USER_REQUEST_TABLE,                                                                                                                                          │
│ ) -> Dict[str, Any]:                                                                                                                                                               │
│     """Generate DDB UpdateItem write for request field changes."""                                                                                                                 │
│     # Returns {"ddb_writes": [{op: "update", table, params: {Key, UpdateExpression, ...}}]}                                                                                        │
│                                                                                                                                                                                    │
│ 2. graph.py — Call at key transitions (~5 locations)                                                                                                                               │
│                                                                                                                                                                                    │
│ Add ddb_writes entries when:                                                                                                                                                       │
│ - info_collection_node: status → collecting, validated                                                                                                                             │
│ - upstream_delegator: status → paused, executing                                                                                                                                   │
│ - downstream_catcher / completion: status → executed, completed                                                                                                                    │
│                                                                                                                                                                                    │
│ Each is a 3-4 line addition using build_request_update_patch(). Non-breaking — _consume_ddb_writes already processes them.                                                         │
│                                                                                                                                                                                    │
│ Part B: Request retrospective                                                                                                                                                      │
│                                                                                                                                                                                    │
│ 3. daily_request_prompts.py — New file (~150 lines)                                                                                                                                │
│                                                                                                                                                                                    │
│ - DAILY_REQUEST_HYGIENE_PROMPT — LLM prompt for analyzing request status                                                                                                           │
│ "Given all requests created today and the full day's conversation:                                                                                                                 │
│  For each request, determine:                                                                                                                                                      │
│  1. TRUE STATUS: completed | still_needed | abandoned | ad_hoc_resolved                                                                                                            │
│  2. UPDATED SUMMARY: comprehensive summary reflecting the full conversation                                                                                                        │
│  3. LEFT_OFF_AT: what was the last meaningful interaction point                                                                                                                    │
│  4. RECOMMENDED ACTION: close | keep_queued | keep_active | archive                                                                                                                │
│                                                                                                                                                                                    │
│  Rules:                                                                                                                                                                            │
│  - Ad-hoc questions that got answered → close (ad_hoc_resolved)                                                                                                                    │
│  - Quick Q&A where user got the info → close                                                                                                                                       │
│  - User firmly changed topics and never came back → abandoned                                                                                                                      │
│  - Epic requests with prerequisites → keep_queued                                                                                                                                  │
│  - Requests where user needs to do something first → keep_queued                                                                                                                   │
│  - Requests still actively being worked → keep_active"                                                                                                                             │
│ - llm_daily_request_hygiene(client, requests, conversation) → Dict                                                                                                                 │
│                                                                                                                                                                                    │
│ 4. daily_analyzer.py — Extend with hygiene pass                                                                                                                                    │
│                                                                                                                                                                                    │
│ Add method run_request_hygiene() to DailyRequestAnalyzer:                                                                                                                          │
│ - Calls llm_daily_request_hygiene                                                                                                                                                  │
│ - For each request with recommended action:                                                                                                                                        │
│   - close / ad_hoc_resolved → update DDB status to completed/aborted                                                                                                               │
│   - keep_queued → update summary, left_off_at fields only                                                                                                                          │
│   - archive → mark completed + update summary                                                                                                                                      │
│ - Uses build_request_update_patch() to generate DDB writes                                                                                                                         │
│ - Logs all changes to MemoryFactLogTable (target="request")                                                                                                                        │
│                                                                                                                                                                                    │
│ ---                                                                                                                                                                                │
│ Orchestration: run_daily_cron.py                                                                                                                                                   │
│                                                                                                                                                                                    │
│ New top-level entry point that runs all workstreams.                                                                                                                               │
│                                                                                                                                                                                    │
│ """                                                                                                                                                                                │
│ Daily cron job — runs at midnight Pacific Time.                                                                                                                                    │
│                                                                                                                                                                                    │
│ Deployment options:                                                                                                                                                                │
│   A) AWS Lambda + EventBridge rule (cron(0 8 * * ? *) = midnight PT in UTC)                                                                                                        │
│   B) Kubernetes CronJob with TZ=America/Los_Angeles                                                                                                                                │
│   C) Traditional cron: 0 0 * * * TZ=America/Los_Angeles python run_daily_cron.py                                                                                                   │
│ """                                                                                                                                                                                │
│                                                                                                                                                                                    │
│ async def run_daily_cron(target_date: Optional[str] = None):                                                                                                                       │
│     date = target_date or (datetime.now(ZoneInfo("America/Los_Angeles")) - timedelta(hours=1)).strftime("%Y-%m-%d")                                                                │
│                                                                                                                                                                                    │
│     # 1. Get all users who had conversations today                                                                                                                                 │
│     users = await conversation_store.get_active_users_for_date(date)                                                                                                               │
│                                                                                                                                                                                    │
│     for user_id in users:                                                                                                                                                          │
│         # 2. Fact cross-validation + candidate key review                                                                                                                          │
│         fact_report = await DailyFactJob(client).run(user_id, date)                                                                                                                │
│                                                                                                                                                                                    │
│         # 3. Request hygiene + status sync                                                                                                                                         │
│         request_report = await DailyRequestAnalyzer(client).run_request_hygiene(user_id, date)                                                                                     │
│                                                                                                                                                                                    │
│         # 4. Request archival (existing daily_analyzer flow)                                                                                                                       │
│         daily_summary = await DailyRequestAnalyzer(client).analyze_daily_requests(...)                                                                                             │
│         await HistoricalRequestStore(...).archive_day(...)                                                                                                                         │
│                                                                                                                                                                                    │
│         # 5. Log daily report                                                                                                                                                      │
│         logger.info(f"Daily cron for {user_id}: {fact_report}, {request_report}")                                                                                                  │
│                                                                                                                                                                                    │
│ ---                                                                                                                                                                                │
│ File Summary                                                                                                                                                                       │
│                                                                                                                                                                                    │
│ ┌───────────────────────────────────┬────────┬──────────────┬────────────────────────────────────────────┐                                                                         │
│ │               File                │ Status │ Lines (est.) │                  Purpose                   │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ scripts/create_dynamodb_tables.py │ Modify │ +30          │ Add UserConversationTable                  │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ ddb_client.py                     │ Modify │ +2           │ Add table constant                         │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ conversation_store.py             │ New    │ ~100         │ Message persistence CRUD                   │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ chat_server.py                    │ Modify │ +15          │ Write messages per turn                    │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ daily_fact_prompts.py             │ New    │ ~200         │ Fact validation + candidate review prompts │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ daily_fact_job.py                 │ New    │ ~250         │ Fact cross-validation orchestrator         │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ daily_fact_models.py              │ New    │ ~50          │ Report models                              │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ request_factory.py                │ Modify │ +30          │ build_request_update_patch()               │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ graph.py                          │ Modify │ +25          │ DDB writes at key transitions              │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ daily_request_prompts.py          │ New    │ ~150         │ Request hygiene prompt                     │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ daily_analyzer.py                 │ Modify │ +80          │ Add run_request_hygiene() method           │                                                                         │
│ ├───────────────────────────────────┼────────┼──────────────┼────────────────────────────────────────────┤                                                                         │
│ │ run_daily_cron.py                 │ New    │ ~120         │ Top-level orchestrator                     │                                                                         │
│ └───────────────────────────────────┴────────┴──────────────┴────────────────────────────────────────────┘                                                                         │
│                                                                                                                                                                                    │
│ Implementation Order                                                                                                                                                               │
│                                                                                                                                                                                    │
│ 1. WS-1 first (conversation persistence) — prerequisite for WS-2 and WS-3                                                                                                          │
│ 2. WS-3 Part A (request DDB sync) — small, independent, fixes real-time drift                                                                                                      │
│ 3. WS-2 (fact validation + candidate review) — the main daily job logic                                                                                                            │
│ 4. WS-3 Part B (request hygiene) — extends existing daily_analyzer                                                                                                                 │
│ 5. Orchestrator (run_daily_cron.py) — ties everything together                                                                                                                     │
│                                                                                                                                                                                    │
│ Verification                                                                                                                                                                       │
│                                                                                                                                                                                    │
│ 1. WS-1: Send messages via chat_server → query UserConversationTable → verify messages persisted with correct GSI1 keys                                                            │
│ 2. WS-2: Run DailyFactJob.run() on test-user-005 → verify stale housing.move_date: last week gets deprecated, artifacts get flagged, candidate keys get reviewed                   │
│ 3. WS-3A: Interact with chat → check DDB UserRequestTable → verify status changes are reflected (not stuck at created)                                                             │
│ 4. WS-3B: Run run_request_hygiene() on test data with queued ad-hoc questions → verify they get closed                                                                             │
│ 5. Full cron: Run run_daily_cron.py with yesterday's date → verify end-to-end: conversation loaded, facts validated, requests hygiene'd, archival completed     