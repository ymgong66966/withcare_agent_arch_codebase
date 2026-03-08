# Test 04: Cross-Session Request Resume

**User ID:** `test-alice` (continue from Test 03)
**Purpose:** Test that a new request matching an old one resumes from where the old one left off, using LLM-based intelligent matching.

---

## Context

From Tests 01-03, Alice has:
- A completed caregiver search request (status: executed) with collected info: South Loop Chicago, $3000/month, Mon-Fri 6hrs, Mandarin, mild dementia
- Facts stored in `UserFactTable` for `care_recipient:mom`

Now we test: if Alice asks for the same thing again, does the system detect the match and resume?

---

## Turn 1 — Similar request (same target, same intent)

**YOU:**
```
I need to find a new caregiver for my mom. The last one didn't work out.
```

**EXPECT:**
- **Reply:** One of two behaviors:
  - **(Best case — resume):** "I found your previous request 'Find In-Home Caregiver'. Let me pick up where we left off. Previously collected: South Loop area, $3000/month budget, Mandarin-speaking, Mon-Fri 6hrs. Has anything changed, or should I search with the same criteria?"
  - **(Acceptable — facts pre-filled):** Creates a new request but the questions skip what's already known. Agent says something like "I remember from before that you're looking in the South Loop area, budget around $3000..."
- **Debug panel:**
  - `current_agent`: `info_collection`
  - If resume worked: `active_request.stage_detail` = `resumed_cross_session`
  - If new request: `active_request.info_collection.summary` should contain pre-filled facts
  - Look at `nodes_visited` — should include `info_collection`

**What this tests:**
1. `request_store.query_recent()` fetches the old caregiver request from DDB
2. The collection plan prompt's matching rules identify: same target (mom), same intent (find_caregiver), status is executed
3. LLM returns `resume_from_request_id` pointing to the old request
4. The resume logic carries over `info_collection_state` from the matched request

---

## Turn 2 — Confirm or adjust criteria

**YOU:**
```
Actually, this time I need someone who can also drive her to appointments. Budget can go up to $4000. Everything else is the same.
```

**EXPECT:**
- **Reply:** Agent updates the criteria — notes the new requirement (driving) and updated budget ($4000). Should confirm the adjusted criteria and either proceed to search or ask if there's anything else.
- **Debug panel:**
  - `info_collection.summary`: Should now include driving requirement and $4000 budget alongside the previous South Loop, Mandarin, Mon-Fri details
  - `info_collection.readiness`: Likely `ready` or `can_proceed_but_incomplete`

**What this tests:** Even when resuming, the agent correctly handles modifications to previously collected info.

---

## Turn 3 — Different target, same type (should NOT match)

First, let the current request complete or say "let's do that search" to move it to deep_search. Then:

**YOU:**
```
I also need to find a caregiver for my dad. He lives in Naperville.
```

**EXPECT:**
- **Reply:** This should be treated as a NEW request, not a resume. The agent should ask fresh questions about dad's care needs — budget, schedule, requirements, etc.
- **Debug panel:**
  - `current_agent`: `info_collection`
  - `turn_mode`: `new_intent`
  - `active_request`:
    - `subject_entity_id`: `care_recipient:dad` (NOT mom)
    - `status`: `collecting`
    - `name`: Something like "Find Caregiver for Dad"
  - Should NOT resume from the mom caregiver request

**What this tests:** The matching rule "same target entity must match" — a request about dad does NOT match a request about mom, even if the type (find_caregiver) is the same.

---

## Turn 4 — Different intent, same target (should NOT match)

**YOU:**
```
Actually, forget the dad thing. I need to look into Medicaid for my mom instead.
```

**EXPECT:**
- **Reply:** New request created for Medicaid application, NOT resuming the caregiver search. Agent should ask about income, assets, state of residence, current insurance, etc.
- **Debug panel:**
  - `active_request`:
    - `request_type`: `medicaid_application` or similar
    - `subject_entity_id`: `care_recipient:mom`
    - `status`: `collecting`
  - Should NOT resume the caregiver request (different intent: medicaid ≠ find_caregiver)

**What this tests:** The matching rule "same intent/type must match" — Medicaid application doesn't match find_caregiver even though both are about mom.

---

## Summary — Matching Rules Verification

| Scenario | Target | Intent | Old Request | Should Match? | Verified? |
|---|---|---|---|---|---|
| Turn 1: Same caregiver for mom | mom | find_caregiver | Find Caregiver (executed) | YES | |
| Turn 3: Caregiver for dad | dad | find_caregiver | Find Caregiver for mom | NO (different target) | |
| Turn 4: Medicaid for mom | mom | medicaid_application | Find Caregiver for mom | NO (different intent) | |
