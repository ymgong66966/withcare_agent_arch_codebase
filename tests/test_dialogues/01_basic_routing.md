# Test 01: Basic Routing — Quick Answer, Emotional Support, Agent Selection

**User ID:** `test-alice`
**Start:** Fresh (no prior conversation)
**Purpose:** Verify the turn_router correctly routes to different agents based on user intent.

---

## Turn 1 — Quick Answer (factual question, no task creation)

**YOU:**
```
What is Medicaid and who is eligible for it?
```

**EXPECT:**
- **Reply:** A direct factual answer explaining Medicaid eligibility (income-based, federal/state program, covers healthcare for low-income individuals). Should be 1-3 paragraphs.
- **Debug panel:**
  - `current_agent`: `quick_answer`
  - `turn_mode`: `new_intent`
  - `nodes_visited`: Should include `turn_router` and likely NOT include `upstream_delegator` (quick_answer bypasses it)
  - `active_request`: Should be `null` or empty — quick_answer doesn't create a request

**What this tests:** The turn_router recognizes a factual question and routes directly to `quick_answer` without creating a request or entering info_collection.

---

## Turn 2 — Another Quick Answer (follow-up factual)

**YOU:**
```
What's the difference between Medicare and Medicaid?
```

**EXPECT:**
- **Reply:** Clear comparison of Medicare (age/disability-based, federal) vs Medicaid (income-based, federal+state).
- **Debug panel:**
  - `current_agent`: `quick_answer`
  - `turn_mode`: `new_intent` (new topic, different question)
  - `active_request`: Still `null` — no task created

**What this tests:** Sequential quick_answer turns work without accumulating state.

---

## Turn 3 — Emotional Support (distress detection)

**YOU:**
```
I'm feeling really overwhelmed. Taking care of my mom is exhausting and I don't know how much longer I can keep doing this.
```

**EXPECT:**
- **Reply:** Warm, empathetic response. Should acknowledge the difficulty of caregiving, validate feelings, NOT give a checklist or action items. Tone should be conversational and supportive.
- **Debug panel:**
  - `current_agent`: `front_end_emotional_support`
  - `turn_mode`: `new_intent`
  - `nodes_visited`: Should include `turn_router` → `upstream_delegator` → `front_end`
  - The reply should NOT ask "what is your budget?" or try to solve a problem

**What this tests:** Emotional distress keywords trigger routing to `front_end_emotional_support`, not info_collection or quick_answer.

---

## Turn 4 — Transition from Emotional to Task (natural flow)

**YOU:**
```
Thanks. Actually, I think what would help most is finding someone who can help take care of her a few days a week. Can you help me find a caregiver?
```

**EXPECT:**
- **Reply:** The agent should transition from emotional support to task mode. It should start asking clarifying questions: Where are you located? What's your budget? How many days/hours? Any specific requirements (language, experience)?
- **Debug panel:**
  - `current_agent`: `info_collection`
  - `turn_mode`: `new_intent`
  - `nodes_visited`: Should include `upstream_delegator` → `info_collection`
  - `active_request`:
    - `status`: `collecting`
    - `name`: Something like "Find In-Home Caregiver"
    - `request_type`: `find_caregiver`
    - `subject_entity_id`: `care_recipient:mom`
    - `awaiting_user_input`: `true`
    - `info_collection.readiness`: `needs_more`
    - `info_collection.key_info_status`: Should list the questions being asked

**What this tests:**
1. Turn_router correctly classifies this as `new_intent` (not continuation of emotional support)
2. Upstream_delegator creates a new request and routes to info_collection
3. Info_collection generates a collection plan with relevant questions
4. Entity detection: "my mom" → `care_recipient:mom`

---

## Turn 5 — Quick Answer mid-conversation (agent switch)

**YOU:**
```
Actually, quick question — what does ADL mean in caregiving?
```

**EXPECT:**
- **Reply:** Defines ADL (Activities of Daily Living) — bathing, dressing, eating, toileting, transferring, continence. Brief factual answer.
- **Debug panel:**
  - `current_agent`: `quick_answer`
  - `turn_mode`: `new_intent` (topic switch from info_collection to factual question)
  - `active_request`: The caregiver request should still exist but the current turn didn't interact with it

**What this tests:** The agent can switch to quick_answer mid-flow without losing the active request. The caregiver request is still there, paused.

---

## Summary — What Success Looks Like

After these 5 turns, the state should show:
- 1 active request ("Find In-Home Caregiver", status: collecting)
- 3 different agents used: quick_answer, front_end_emotional_support, info_collection
- No crashes, no agent confusion
- Debug panel correctly reflects which agent handled each turn
