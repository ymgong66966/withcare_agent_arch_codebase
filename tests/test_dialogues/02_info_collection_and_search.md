# Test 02: Full Request Lifecycle — Info Collection → Deep Search

**User ID:** `test-alice` (continue from Test 01)
**Start:** Continuing from the caregiver search request created in Test 01, Turn 4
**Purpose:** Complete the info collection Q&A loop, hand off to deep_search, and get results.

---

## Turn 1 — Resume the caregiver request

**YOU:**
```
Let's continue with finding a caregiver for my mom. She lives in the South Loop area of Chicago.
```

**EXPECT:**
- **Reply:** The agent should recognize this continues the existing caregiver request. It should acknowledge the location and ask remaining questions (budget, schedule, requirements, etc.).
- **Debug panel:**
  - `current_agent`: `info_collection`
  - `turn_mode`: `continuation`
  - `active_request`:
    - `status`: `collecting`
    - `info_collection.summary`: Should now include "South Loop, Chicago" or similar
    - `info_collection.readiness`: `needs_more`

**What this tests:** Continuation detection — the turn_router classifies this as continuation (not new_intent) and routes back to info_collection for the same request.

---

## Turn 2 — Provide more info (multiple answers at once)

**YOU:**
```
My budget is around $3000 per month. I need someone Monday through Friday, about 6 hours a day. She needs help with bathing and meals, and it would be great if they speak Mandarin since my mom is more comfortable in Chinese.
```

**EXPECT:**
- **Reply:** The agent should update the collected info summary with all provided details. It may either:
  - (a) Ask 1-2 remaining clarifying questions (e.g., "Any medical conditions the caregiver should be aware of?" or "When would you like to start?"), OR
  - (b) Indicate it has enough info and is ready to proceed
- **Debug panel:**
  - `current_agent`: `info_collection`
  - `active_request`:
    - `info_collection.summary`: Should contain budget ($3000), schedule (Mon-Fri, 6hrs), needs (bathing, meals), language (Mandarin)
    - `info_collection.readiness`: Either `can_proceed_but_incomplete` or `ready`
    - `info_collection.turns`: Should be incrementing (2 or 3)

**What this tests:**
1. Multi-answer parsing — agent correctly extracts multiple pieces of info from one message
2. Fact binding — `slot_fact_binder` should write structured facts to DDB (budget, schedule, language preference)
3. Readiness assessment — LLM evaluates whether enough info has been collected

---

## Turn 3 — Signal readiness to proceed

If the agent asked follow-up questions, answer briefly. If it said it's ready, skip this turn.

**YOU:**
```
She has mild dementia but is mostly independent. Let's go ahead and search with what we have.
```

**EXPECT:**
- **Reply:** The agent should confirm it has enough info and transition to search. The reply may say something like "Let me search for caregivers matching your criteria..." followed by actual search results (provider names, addresses, ratings) OR a summary of what was found.
- **Debug panel:**
  - `nodes_visited`: Should include `info_collection` → `downstream_catcher` → `deep_search` (handoff)
  - `active_request`:
    - `status`: Changes from `collecting` → `validated` → possibly `executing`
    - `info_collection.readiness`: `ready`
  - After deep_search runs:
    - `current_agent`: `deep_search`
    - Look for tool_runs in the debug logs showing google_places_search or web search calls

**What this tests:**
1. Info collection → deep_search handoff via `pending_handoff`
2. Deep search tool execution (Google Places, web search)
3. Request status progression: collecting → validated → executing
4. The "proceed with what we have" signal is detected by the LLM

---

## Turn 4 — Follow-up on search results

**YOU:**
```
Can you tell me more about the first result? What services do they offer?
```

**EXPECT:**
- **Reply:** The agent should provide more details about the first search result. It may do another web search or scrape the provider's website for details.
- **Debug panel:**
  - `current_agent`: `deep_search`
  - `turn_mode`: `continuation` (follow-up on same topic)
  - `nodes_visited`: Should go directly to deep_search (no upstream_delegator)

**What this tests:** Continuation within deep_search — follow-up questions about results stay in the same agent.

---

## Turn 5 — Acknowledge results (request completion)

**YOU:**
```
Great, that's really helpful. I'll look into those options. Thank you!
```

**EXPECT:**
- **Reply:** A friendly closing message. The agent may ask "Is there anything else I can help with?" or simply acknowledge.
- **Debug panel:**
  - `turn_mode`: `new_intent` (the LLM should detect "task acknowledged" — user is done)
  - `active_request`:
    - `status`: Should change to `executed` (task acknowledged by user)
  - The request lifecycle is now complete

**What this tests:**
1. Task acknowledgement detection — the upstream_delegator recognizes the user is done
2. Request status transitions to `executed`
3. Stage history should show the full lifecycle: created → collecting → validated → executing → executed

---

## Summary — Debug Panel Checks

After this full lifecycle, verify in the debug panel:
- `total_requests`: At least 1
- `active_request.status`: `executed`
- `nodes_visited` across all turns should have included: `turn_router`, `upstream_delegator`, `info_collection`, `downstream_catcher`, `deep_search`
- The info_collection summary should contain all the details the user provided (location, budget, schedule, language, condition)

## Fact Store Verification

If you have DDB access, check `WithCare_UserFactTable` for `USER#test-alice`:
- There should be facts written for `care_recipient:mom`:
  - Location/address info
  - Budget
  - Language preference (Mandarin)
  - Health condition (mild dementia)
- These were written by `slot_fact_binder` during info_collection turns 2-3
