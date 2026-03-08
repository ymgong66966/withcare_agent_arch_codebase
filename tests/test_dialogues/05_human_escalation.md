# Test 05: Human Escalation — Detection, Proposal, Confirm/Decline

**User ID:** `test-bob`
**Start:** Fresh (no prior conversation — use a new user ID)
**Purpose:** Test the human escalation pathway: deep_search streak detection, human_comm proposal, user confirmation/decline.

---

## Phase A: Build up to escalation trigger (3 consecutive deep_search rounds)

### Turn 1 — Start a search request

**YOU:**
```
I need to find an adult daycare center near Lincoln Park in Chicago for my dad.
```

**EXPECT:**
- Agent enters info_collection, asks about budget, schedule, dad's needs, etc.
- `current_agent`: `info_collection`

---

### Turn 2 — Provide info and proceed

**YOU:**
```
Budget is flexible, he needs it Monday through Friday, he has early-stage Alzheimer's. Let's search now.
```

**EXPECT:**
- Agent collects info and hands off to deep_search. Search results returned.
- `current_agent`: `deep_search` (first deep_search round)
- Reply: Search results with daycare centers

---

### Turn 3 — Express dissatisfaction (second deep_search round)

**YOU:**
```
These don't look right. Can you search again? I need places that specifically have memory care programs.
```

**EXPECT:**
- Deep search runs again with refined criteria.
- `current_agent`: `deep_search` (second consecutive round)
- Reply: Updated results

---

### Turn 4 — Still not satisfied (third deep_search round — triggers escalation!)

**YOU:**
```
Still not what I'm looking for. None of these seem to have the right kind of support for Alzheimer's patients.
```

**EXPECT:**
- **This is the critical turn.** The `_detect_deep_search_streak()` heuristic should fire (3 consecutive deep_search assistant messages).
- **Reply:** The `human_comm` agent should generate a warm proposal asking if Bob would like a clinical team member to reach out. Something like: "I can see we haven't been able to find exactly what you need. Would you like me to connect you with a member of our clinical team who can help with this search directly?"
- **Debug panel:**
  - `current_agent`: `human_comm` (NOT deep_search)
  - `turn_reason`: Should contain "Escalation trigger: deep_search_3_consecutive_rounds"
  - `llm_recommended_agent`: `human_comm`
  - `nodes_visited`: Should include `turn_router` → `human_comm`

**What this tests:**
1. Deep_search streak detection (3+ consecutive rounds)
2. Turn_router intercept before the LLM call
3. Human_comm proposal generation via LLM (not a rigid string)

---

## Phase B: Decline the escalation

### Turn 5 — Say no to human support

**YOU:**
```
No thanks, let me try a different approach. Can you search for memory care facilities instead of adult daycare?
```

**EXPECT:**
- **Reply:** Friendly acknowledgement, returns to deep_search. Something like "No problem! Let's try a different search approach..."
- **Debug panel:**
  - `current_agent`: Changes back to `deep_search` (human_comm_node detected "no" and returned to deep_search)
  - `needs_human`: Still `false` (escalation was declined)
  - The next turn should route normally

**What this tests:** User decline flow — `_user_yes_no()` detects "no" and `human_comm_node` returns control to deep_search.

---

## Phase C: Trigger escalation again and accept

### Turn 6-8 — Build another 3-round streak

Continue with deep_search queries that express dissatisfaction:

**Turn 6 YOU:**
```
These memory care facilities are too expensive. Can you find more affordable options?
```

**Turn 7 YOU:**
```
Still too far away. I need something within 15 minutes of Lincoln Park.
```

**Turn 8 YOU:**
```
None of these work either. I'm frustrated.
```

**EXPECT on Turn 8:**
- Human_comm proposal appears again (3 consecutive deep_search rounds since the decline)
- `current_agent`: `human_comm`

---

### Turn 9 — Accept the escalation

**YOU:**
```
Yes, please have someone reach out to me.
```

**EXPECT:**
- **Reply:** Warm confirmation that the conversation has been forwarded to the clinical team. LLM-generated (not rigid). Something like "I've forwarded your conversation to our clinical team. They will reach out to you shortly."
- **Debug panel:**
  - `current_agent`: `human_comm`
  - `needs_human`: **`true`** (this is the critical flag)
  - The MCP tool `human_escalation_deliver` should have been called (check logs for "Escalation delivery")

**What this tests:**
1. `_user_yes_no()` detects "yes"
2. `human_comm_node` calls the escalation MCP tool with last 10 messages
3. `needs_human` is set to `true` (sticky)

---

## Phase D: Verify sticky escalation

### Turn 10 — Send a message while in escalation mode

**YOU:**
```
Also, can you let them know that my dad is a veteran? He might qualify for VA benefits too.
```

**EXPECT:**
- **Reply:** Acknowledgement that the message was forwarded to the clinical team. NOT a search or info collection response.
- **Debug panel:**
  - `current_agent`: `human_comm`
  - `needs_human`: `true` (still sticky)
  - The turn_router should have hit the `needs_human=True` intercept at the top and called `_handle_escalated_turn()` — forwarding to the lambda

**What this tests:** Once `needs_human=True`, ALL subsequent messages go through `_handle_escalated_turn` — they're forwarded to the clinical team, not processed by the regular agent flow.

---

## Phase E: Alternative trigger — user explicitly asks for human

**To test this, reset and start a new conversation as test-bob.**

### Turn 1 (new conversation)

**YOU:**
```
I need help finding memory care for my dad near Lincoln Park, Chicago.
```

### Turn 2

**YOU:**
```
Budget is around $5000/month, he needs full-time care. Can you search?
```

### Turn 3 — Explicitly request human help (LLM-based trigger, not streak)

**YOU:**
```
This is not helpful at all. Can I talk to a real person? I want to speak with someone from your clinical team.
```

**EXPECT:**
- **Reply:** Human_comm proposal (same as the streak-triggered one). The LLM turn_mode decision should detect the explicit request for human help via the "Escalation Detection" prompt section.
- **Debug panel:**
  - `current_agent`: `human_comm`
  - `llm_recommended_agent`: `human_comm`
  - `turn_reason`: Should mention user frustration or explicit human request

**What this tests:** The LLM-based escalation detection (Condition 2 in the prompt), separate from the heuristic streak detection.

---

## Summary — Escalation Flow Verification

| Step | What Happens | Key Debug Field |
|---|---|---|
| 3 deep_search rounds | Streak detected → human_comm proposal | `turn_reason: "Escalation trigger: deep_search_3_consecutive_rounds"` |
| User says "no" | Returns to deep_search | `current_agent: deep_search`, `needs_human: false` |
| User says "yes" | MCP tool called, needs_human=true | `needs_human: true` |
| Subsequent messages | Forwarded to clinical team | `_handle_escalated_turn` path, ack message |
| Explicit "talk to a person" | LLM detects → human_comm | `llm_recommended_agent: human_comm` |
