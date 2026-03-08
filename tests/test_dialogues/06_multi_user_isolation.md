# Test 06: Multi-User Isolation

**User IDs:** `test-alice` and `test-carol`
**Start:** Fresh (or continuing from earlier tests for alice)
**Purpose:** Verify that two users typing in the same chat UI with different user IDs have fully isolated state — no cross-contamination of messages, facts, requests, or routing.

---

## Setup

You'll alternate between two "users" by changing the User ID input field. Open the chat UI in **one tab** — you'll switch the user ID between turns.

---

## Phase A: Alice creates a request

### User ID: `test-alice`

**Turn A1:**
```
I need to find a physical therapist for my mom near Hyde Park, Chicago.
```

**EXPECT:**
- Info collection starts. Agent asks about mom's condition, budget, schedule, etc.
- `current_agent`: `info_collection`
- `subject_entity_id`: `care_recipient:mom`
- Note the `conversation_id` (shown in header) — call it `conv-alice`

**Turn A2:**
```
She needs post-hip-replacement therapy, twice a week. Budget around $200 per session.
```

**EXPECT:**
- Info updated. Agent may proceed to search or ask one more question.
- `info_collection.summary`: Should mention hip replacement, twice/week, $200/session, Hyde Park

---

## Phase B: Switch to Carol — different world

### User ID: Change to `test-carol`

The chat UI will start a new conversation (since the user ID changed and the old conversation_id is tied to alice).

**Turn B1:**
```
What insurance options are available for my husband?
```

**EXPECT:**
- **Completely fresh state.** No reference to Alice's mom, Hyde Park, physical therapy, or any of Alice's data.
- `current_agent`: Either `quick_answer` (if treated as factual) or `info_collection` (if treated as a task)
- `subject_entity_id`: Should be `care_recipient:husband` or similar — NOT `care_recipient:mom`
- `conversation_id`: Different from `conv-alice`
- `total_requests`: 0 or 1 (Carol's own, not Alice's)

**What this tests:** The `(user_id, conversation_id)` cache key isolates Carol's state from Alice's.

**Turn B2:**
```
He's 68, retired, and we live in Dallas, Texas.
```

**EXPECT:**
- Agent collects info about Carol's husband. No mention of Chicago, Hyde Park, or anything from Alice's context.
- Facts written to DDB under `USER#test-carol`, NOT `USER#test-alice`

---

## Phase C: Switch back to Alice — her state intact

### User ID: Change back to `test-alice`

**Turn C1:**
```
Let's continue with the physical therapy search for mom.
```

**EXPECT:**
- **Alice's full context is restored.** The agent should reference:
  - Mom's hip replacement therapy
  - Hyde Park location
  - $200/session budget
  - Twice a week schedule
- No mention of Carol's husband, Dallas, or insurance.
- `current_agent`: `info_collection` or `deep_search` (depending on where Alice left off)
- `subject_entity_id`: `care_recipient:mom`

**What this tests:**
1. Switching back to Alice's user ID restores her isolated state
2. Carol's data did not contaminate Alice's state
3. The `(user_id, conversation_id)` tuple correctly separates them

---

## Phase D: Verify DDB isolation

If you have AWS CLI access, run these checks:

### Alice's facts
```bash
aws dynamodb query \
  --table-name WithCare_UserFactTable \
  --key-condition-expression "pk = :pk" \
  --expression-attribute-values '{":pk": {"S": "USER#test-alice#ENT#care_recipient:mom"}}' \
  --region us-east-2
```
Should contain: hip replacement, Hyde Park, $200/session

### Carol's facts
```bash
aws dynamodb query \
  --table-name WithCare_UserFactTable \
  --key-condition-expression "pk = :pk" \
  --expression-attribute-values '{":pk": {"S": "USER#test-carol#ENT#care_recipient:husband"}}' \
  --region us-east-2
```
Should contain: age 68, retired, Dallas TX

### Cross-check: Alice should NOT have Carol's data
```bash
aws dynamodb query \
  --table-name WithCare_UserFactTable \
  --key-condition-expression "begins_with(pk, :prefix)" \
  --expression-attribute-values '{":prefix": {"S": "USER#test-alice"}}' \
  --region us-east-2
```
Should only show `care_recipient:mom` entities, NOT `care_recipient:husband`.

---

## Phase E: Same conversation_id, different user (attack simulation)

This tests the isolation fix we implemented.

### User ID: `test-carol`

1. Note Alice's `conversation_id` from Phase A (the one shown in the header when Alice was chatting)
2. Manually craft a request to `/chat` with Carol's user_id but Alice's conversation_id:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What was the last thing discussed?",
    "conversation_id": "PASTE_ALICE_CONV_ID_HERE",
    "user_id": "test-carol"
  }'
```

**EXPECT:**
- Carol should get a **fresh state**, NOT Alice's conversation history.
- The `(test-carol, alice_conv_id)` cache key is different from `(test-alice, alice_conv_id)`.
- Reply should be a generic greeting or "I don't have any context about your previous conversations."
- Carol should NOT see Alice's mom, Hyde Park, physical therapy, or any of Alice's data.

**What this tests:** The core isolation fix — even if Carol knows Alice's conversation_id, the `(user_id, conversation_id)` composite key prevents access.

**Note:** Carol may still see Alice's messages from DDB `UserConversationTable` (since that table queries by conversation_id only, not user_id). This is a known remaining gap, but the in-memory state and fact/request stores are properly isolated.

---

## Summary — Isolation Verification Checklist

| Check | Expected | Verified? |
|---|---|---|
| Carol's chat has no Alice data | No mention of mom, Hyde Park, PT | |
| Alice's chat has no Carol data | No mention of husband, Dallas, insurance | |
| DDB facts scoped by user_id | Alice facts under `USER#test-alice`, Carol under `USER#test-carol` | |
| DDB requests scoped by user_id | Each user's requests isolated | |
| Same conv_id + different user_id → fresh state | Carol can't access Alice's in-memory state | |
| _search_llm (shared) doesn't leak | No cross-user content in responses | |
