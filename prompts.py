from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
import json
import logging

logger = logging.getLogger(__name__)

# NOTE:
# This file is intentionally model-agnostic.
# In production, you can connect these prompt templates to your LLM call (OpenAI / Anthropic / internal).
# For now, these functions return deterministic placeholders so the graph can run end-to-end.

def make_collection_plan_prompt(
    *,
    user_request: str,
    known_facts: Dict[str, Any],
    similar_requests: List[Dict[str, Any]] | None = None,
    recent_requests: List[Dict[str, Any]] | None = None,
) -> str:
    """
    Generate a prompt for Claude to create an information collection plan.
    This is conversational - NOT structured slot extraction.
    """
    similar_context = ""
    if similar_requests:
        similar_context = "\n\n## Similar Past Requests (from vector search):\n\n"
        for i, req in enumerate(similar_requests[:3], 1):
            completion = req.get('completion_status', 'unknown')
            similar_context += f"{i}. **{req.get('name', 'Unknown')}** ({completion})\n"
            similar_context += f"   - Goal: {req.get('goal', 'N/A')[:150]}\n"
            similar_context += f"   - Outcome: {req.get('short_summary', 'N/A')[:200]}\n"
            similar_context += f"   - Theme: {req.get('theme', 'N/A')}\n"
            if req.get('user_satisfaction'):
                similar_context += f"   - User was: {req.get('user_satisfaction')}\n"
            similar_context += "\n"

    recent_context = ""
    if recent_requests:
        recent_context = "\n\n## Recent Requests (last 10 from this user's history):\n\n"
        for i, req in enumerate(recent_requests[:10], 1):
            rid = req.get('request_id', '?')
            name = req.get('name') or req.get('title') or 'Unknown'
            goal = req.get('goal', '')[:200]
            status = req.get('status', 'unknown')
            entity = req.get('subject_entity_id', '')
            req_type = req.get('request_type', '')
            summary = req.get('summary_current', '')[:300]
            # Extract collected info from payload if available
            payload = req.get('payload') or {}
            ic_state = payload.get('info_collection_state') or {}
            collected_info = ic_state.get('summary_of_collected_info', '')[:400]
            recent_context += f"{i}. [request_id={rid}] **{name}** (status: {status})\n"
            recent_context += f"   - Goal: {goal}\n"
            if entity:
                recent_context += f"   - Target: {entity}\n"
            if req_type:
                recent_context += f"   - Type: {req_type}\n"
            if summary:
                recent_context += f"   - Summary: {summary}\n"
            if collected_info:
                recent_context += f"   - Collected info so far: {collected_info}\n"
            recent_context += "\n"

    return f"""You are an information collection planner for a caregiver assistant.

Your job is to create a plan for gathering information from the user to help them accomplish their request.
If a recent request matches the user's current intent, you should resume from where that request left off instead of starting fresh.

## User's Request:
{user_request}

## Known Facts About User:
Caregiver: {json.dumps(known_facts.get('caregiver', {}), indent=2, ensure_ascii=False)}
Care Recipient: {json.dumps(known_facts.get('care_recipient', {}), indent=2, ensure_ascii=False)}
{known_facts.get('_memory_context', '')}
{similar_context}
{recent_context}

## Request Matching Rules

Before creating a new plan, check the Recent Requests above for a match. A request is a MATCH if ALL of these conditions are met:

1. **Same target entity**: The request is about the same person (e.g., both about "care_recipient:mom"). A request about mom is NOT a match for a request about dad.
2. **Same intent/type**: The core goal is substantially the same (e.g., both "find caregiver", or both "apply for medicaid"). A request to "find a caregiver" is NOT a match for "apply for insurance".
3. **Not fully completed**: The previous request has status "collecting", "paused", "validated", or "executing" — meaning there's useful work to resume. A "completed" or "aborted" request should NOT be resumed unless the user explicitly wants to redo it.

A request is NOT a match if:
- The target person is different (different care recipient or user vs care_recipient)
- The core intent is different even if the topic area overlaps (e.g., "find a caregiver" vs "find a nursing home" — both are care-related but different goals)
- The old request was completed and the user seems to want a fresh start

### Matching Examples:

**MATCH — resume:**
- New: "I need to find a caregiver for mom" → Recent: [request_id=req-123] "Find In-Home Caregiver" (status: paused, target: care_recipient:mom, collected info: "Budget: $3000/month, location: Chicago South Loop, needs Chinese-speaking")
  → Resume from req-123 because same target (mom), same intent (find caregiver), has useful collected info

**MATCH — resume:**
- New: "Can we continue looking into Medicaid for dad?" → Recent: [request_id=req-456] "Medicaid Application" (status: collecting, target: care_recipient:dad, collected info: "Income under threshold, needs proof of residency")
  → Resume from req-456 because same target (dad), same intent (medicaid), explicitly continuing

**NOT a match:**
- New: "Find a caregiver for mom" → Recent: [request_id=req-789] "Find Nursing Home" (target: care_recipient:mom)
  → Different intent (caregiver ≠ nursing home), create new request

**NOT a match:**
- New: "Find a caregiver for dad" → Recent: [request_id=req-123] "Find Caregiver" (target: care_recipient:mom)
  → Different target (dad ≠ mom), create new request

**NOT a match:**
- New: "I need to find a caregiver" → Recent: [request_id=req-101] "Find In-Home Caregiver" (status: completed)
  → Previous request was completed; user likely wants a fresh search, create new request

## Your Task:

Analyze the user's request and create a collection plan. You need to determine:

1. **request_name**: A short, clear name for this request (3-6 words)
   - Example: "Find In-Home Caregiver", "Apply for Medicaid", "Schedule Doctor Appointment"

2. **request_goal**: A detailed 1-2 sentence description of what the user wants to accomplish

3. **key_info_needed**: A list of 3-6 key pieces of information you MUST collect before proceeding to execution
   - Write these as natural conversational questions
   - **LANGUAGE**: Write questions in the same language the user is using. If the user wrote in English, ask in English. If in Chinese, ask in Chinese.
   - Focus on ESSENTIAL information only (constraints, preferences, timeline, location, etc.)
   - DO NOT ask for information already in Known Facts
   - Each should be a complete question that feels natural to ask

4. **nice_to_have_info**: A list of 1-3 additional pieces of information that would be helpful but not critical
   - These are optional clarifications
   - User can skip these if they want to proceed

5. **potential_prerequisites**: List any prerequisites you detect (tasks that must be done first)
   - Example: If user wants to hire caregiver but doesn't have Medicaid, that's a prerequisite
   - Format: {{"type": "prerequisite_name", "reason": "why it's needed"}}
   - Leave empty array if none detected

6. **routing_hint**: After info is collected, which agent should handle execution?
   - Choose from: ["deep_search", "domain_expert", "user_info", "front_end_emotional_support"]
   - "deep_search": The user wants to FIND specific local providers, agencies, or facilities. Requires location-based search via Google Places.
     IMPORTANT: If routing to deep_search, your key_info_needed MUST include a question asking for a SPECIFIC location — street, neighborhood, or district level, not just city. Example: "Which area of Chicago are you looking in? For example, a specific street or neighborhood?"
   - "domain_expert": The user needs a LONG-FORM written deliverable — guidance document, email draft, formal letter, detailed checklist, or step-by-step plan.
   - "user_info": The user wants to review their own historical data.
   - "front_end_emotional_support": Primarily emotional support.

   NOTE: Do NOT use routing_hint for quick factual questions (what is X?, eligibility rules, etc.) — those are handled before info_collection by quick_answer and never reach this point.

7. **request_type**: Select from this controlled list:
   insurance_renewal, insurance_application, insurance_appeal, insurance_prior_auth,
   insurance_eligibility_check, find_provider, schedule_appointment, medication_management,
   medication_refill, symptom_assessment, find_caregiver, care_plan_review, daily_care_setup,
   respite_care, care_transition, medicaid_application, medicare_enrollment,
   ssi_ssd_application, benefits_eligibility, benefits_renewal, poa_setup, guardianship,
   hipaa_release, advance_directive, reimbursement_filing, bill_dispute, cost_estimation,
   financial_assistance, information_lookup, emotional_support, resource_referral,
   complaint_escalation
   Use "information_lookup" if none of the above match.

8. **subject_entity_id**: Who is the request about?
   - "care_recipient:mom", "care_recipient:dad", "care_recipient:spouse", "care_recipient:grandparent", "user:self"
   - Use the most specific identifier you can infer from the user's message.
   - Default to "care_recipient:unknown" if unclear.

9. **resume_from_request_id**: If you found a matching recent request (per the matching rules above), set this to the request_id of that request. Set to null if no match.
   - When resuming, still fill in all other fields (request_name, key_info_needed, etc.) but ADAPT them:
     - In key_info_needed, ONLY ask for information NOT already in the matched request's "Collected info so far"
     - Acknowledge the previous progress in the questions (e.g., "Last time we discussed budget and location. Is there anything else you'd like to add or change?")

## Output Format (JSON):

{{
  "request_name": "string",
  "request_goal": "string",
  "request_type": "string (from list above)",
  "subject_entity_id": "string",
  "resume_from_request_id": "string or null",
  "key_info_needed": [
    "Question 1 that must be answered?",
    "Question 2 that must be answered?",
    "Question 3 that must be answered?"
  ],
  "nice_to_have_info": [
    "Optional question 1?",
    "Optional question 2?"
  ],
  "potential_prerequisites": [
    {{"type": "Prerequisite Name", "reason": "Why it's needed"}}
  ],
  "routing_hint": "agent_name"
}}

## Example:

User Request: "I need to find a caregiver for my mom in Chicago"

Output:
{{
  "request_name": "Find In-Home Caregiver",
  "request_goal": "Find and hire an in-home caregiver for user's mother in Chicago area",
  "request_type": "find_caregiver",
  "subject_entity_id": "care_recipient:mom",
  "key_info_needed": [
    "What is your approximate budget? (hourly or monthly)",
    "When do you need to start?",
    "How many days per week? How many hours per day?",
    "Any special requirements? (e.g., bilingual, nursing experience, can cook, etc.)"
  ],
  "nice_to_have_info": [
    "Do you have insurance coverage?",
    "Have you used a caregiver before? How was the experience?"
  ],
  "potential_prerequisites": [],
  "routing_hint": "deep_search"
}}

Respond ONLY with the JSON object, no other text."""

def make_memory_need_decision_prompt(
    *,
    user_message: str,
    request_goal: str,
    request_entity: str,
    previous_summary: str,
    known_facts_preview: Dict[str, Any],
) -> str:
    """Quick prompt for the LLM to decide if stored facts should be loaded."""
    facts_preview = json.dumps(known_facts_preview, indent=2, ensure_ascii=False) if known_facts_preview else "(none loaded yet)"

    return f"""You are deciding whether to look up stored user profile facts from the memory system.

## Current request
- Goal: {request_goal}
- About: {request_entity}
- Collected so far: {previous_summary[:300] if previous_summary else "(nothing yet)"}

## User's latest message
{user_message}

## Facts currently available
{facts_preview}

## Decision

Should we load stored profile facts for "{request_entity}" from the memory system?

Answer YES if ANY of these apply:
- The user mentions a person (mom, dad, etc.) and we have no facts about them yet
- The user says "just search" or "go ahead" but we're missing key details (location, budget, condition) that might be stored
- The user references information they provided before ("like last time", "same as before", "you already know")
- The request goal requires details we don't have in the current summary

Answer NO if:
- We already have sufficient facts loaded for the entity
- The user is providing new information (not referencing stored data)
- The question is about something unrelated to the entity's profile

## Examples

User: "for my mom, just start searching" → YES (need mom's location, condition etc.)
User: "my budget is $3000 per month" → NO (user is providing new info, no lookup needed)
User: "same requirements as last time" → YES (need to look up what "last time" was)
User: "I prefer someone who speaks Mandarin" → NO (new info being provided)
User: "yes, 20 hours per week works" → NO (answering a question, no lookup needed)

Respond with ONLY a JSON object:
{{"needs_memory_lookup": true/false, "reason": "brief explanation"}}"""


async def llm_memory_need_decision(
    *,
    user_message: str,
    request_goal: str,
    request_entity: str,
    previous_summary: str,
    known_facts_preview: Dict[str, Any],
    client: Any,
) -> bool:
    """Ask the LLM whether stored facts should be loaded for this turn.

    Returns True if memory lookup is needed, False otherwise.
    """
    prompt = make_memory_need_decision_prompt(
        user_message=user_message,
        request_goal=request_goal,
        request_entity=request_entity,
        previous_summary=previous_summary,
        known_facts_preview=known_facts_preview,
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=100,
            temperature=0.0,
        )
        result = json.loads(response.strip().strip("`").strip())
        needs = result.get("needs_memory_lookup", False)
        reason = result.get("reason", "")
        if needs:
            logger.info(f"LLM decided memory lookup needed: {reason}")
        return bool(needs)
    except Exception as e:
        logger.warning(f"Memory need decision failed, defaulting to True: {e}")
        return True  # fail-open: load facts if we can't decide


def make_info_collection_summarize_prompt(
    *,
    request_goal: str,
    key_info_needed: List[str],
    nice_to_have_info: List[str],
    previous_summary: str,
    user_latest_reply: str,
    conversation_history: List[Dict[str, Any]],
    known_facts: Dict[str, Any],
    last_asked_questions: Optional[List[str]] = None,
) -> str:
    """
    Generate a prompt for Claude to summarize collected information and decide next steps.
    This is conversational - produces natural language summary, NOT structured slots.
    """

    conversation_text = "\n".join([
        f"[{msg.get('role', '').upper()}]: {msg.get('content', '')[:300]}"
        for msg in conversation_history[-10:]  # Last 10 turns
    ])

    previous_summary_text = previous_summary or "None (this is the first conversation turn)"

    return f"""You are an information collection agent for a caregiver assistant.

You've been gathering information from the user to help them accomplish their request.

## Request Goal:
{request_goal}

## Information We Need to Collect:

**Key Information (must collect):**
{chr(10).join([f"- {q}" for q in key_info_needed])}

**Nice-to-Have Information (optional):**
{chr(10).join([f"- {q}" for q in nice_to_have_info]) if nice_to_have_info else "None"}

## Previous Summary of Collected Info:
{previous_summary_text}

## Recent Conversation (last 10 turns):
{conversation_text}

## User's Latest Reply:
{user_latest_reply}

## Known Facts:
{json.dumps(known_facts, indent=2, ensure_ascii=False)}

## Questions Asked Last Turn (may still be unanswered):
{chr(10).join([f"- {q}" for q in last_asked_questions]) if last_asked_questions else "None (first turn or no questions asked last turn)"}

## Your Task:

Analyze the user's latest reply in the context of the conversation and update the information summary.

You need to produce:

1. **updated_summary**: A natural language bullet-point summary of ALL information collected so far
   - Include information from previous_summary AND user's latest reply
   - Use clear, concise bullet points in the user's language
   - Group related information together
   - Example format:
     ```
     - Budget: $2,000–3,000/month
     - Timeline: ASAP, hoping next week
     - Location: North Chicago
     - Special requirements: bilingual, nursing experience
     ```

2. **key_info_status**: For each key_info_needed item, mark as "collected", "partial", or "missing"
   - Format: [{{"question": "...", "status": "collected|partial|missing"}}]

3. **readiness_to_proceed**: Your assessment of whether we have enough info
   - "ready": We have all critical information, can proceed confidently
   - "can_proceed_but_incomplete": We have core info but missing some details, user can choose to proceed or provide more
   - "needs_more": We're missing critical information, should not proceed yet

4. **missing_or_unclear**: List of questions that are still unanswered or need clarification
   - Write as natural follow-up questions
   - Only include if readiness is NOT "ready"
   - Max 3-4 questions
   - **IMPORTANT: Triage questions from "Questions Asked Last Turn":**
     a. Check which of those questions the user's latest reply actually answered (fully or partially)
     b. Questions NOT addressed at all → carry forward IF still relevant to the request goal
     c. New information from the user may trigger NEW important questions → add those
     d. Merge carried-forward + new questions, then prioritize: keep total ≤ 3-4
     e. Drop any carried-forward question that is no longer relevant (e.g., user's new info made it moot)

5. **detected_prerequisites**: Detect if user has UNINTENTIONALLY revealed a prerequisite task that should be done first
   - IMPORTANT: Only include prerequisites that user REVEALED/MENTIONED but did NOT explicitly request
   - Example: User says "I haven't applied for Medicaid yet, so I'm not sure about the budget" → They revealed Medicaid is missing
   - Example: User says "I need to apply for Medicaid first" → This is explicit request, NOT unintentional revelation (will be handled by turn_router/delegator)
   - Format: [{{"type": "Prerequisite Name", "reason": "Why it's important for current request", "evidence": "What user said that revealed this"}}]
   - Leave empty array if no prerequisites detected

6. **suggested_response**: A natural, conversational message to send to the user
   - **LANGUAGE**: Always match the language the user is writing in. If the user writes in English, respond in English. If in Chinese, respond in Chinese.
   - Acknowledge what they've shared
   - **CRITICAL: If detected_prerequisites is not empty, you MUST propose it in this response:**
     - Example: "I noticed you haven't applied for Medicaid yet. The Medicaid application is important for determining your caregiver budget. Would you like me to help with that first?"
     - Example (English user): "I noticed you haven't applied for Medicaid yet. The Medicaid application is important for determining your caregiver budget. Would you like me to help with that first?"
     - Be gentle and offer it as a helpful suggestion, not a demand
   - If readiness = "ready": Confirm we have enough and will proceed
   - If readiness = "can_proceed_but_incomplete": Summarize what we have, mention what's missing, offer option to proceed or clarify
   - If readiness = "needs_more": Gently ask for the missing critical information
   - Be warm, concise, and helpful
   - If readiness is "can_proceed_but_incomplete", offer the option to proceed with existing info

## Special Cases:

**If Known Facts contain stored facts from previous conversations:**
- In your suggested_response, briefly acknowledge what you already know from previous sessions
- Example: "I remember from before that he's in the South Loop area, prefers a female therapist, and has a budget of $300/hour. Is that still correct?"
- This is especially important in early turns (conversation_turns_with_agent ≤ 2) when the user hasn't been shown the stored facts yet
- Do NOT re-ask questions whose answers are already in Known Facts — instead confirm them and ask about what's MISSING

**If user provides partial or unclear information:**
- Still summarize what you understood
- Mark as "partial" in key_info_status
- Ask clarifying follow-up

**If user seems frustrated or overwhelmed:**
- Keep suggested_response empathetic and concise
- Offer the option to proceed with partial information

**If user contradicts a previously stored/confirmed fact:**
- The assistant may have presented stored facts for confirmation (e.g., "Based on what I know: Location: Chicago, IL")
- There are TWO different cases — distinguish carefully:
  - **Case A: User provides the corrected value** (e.g., "actually she has Blue Shield not Blue Cross", "she moved to Denver"):
    - Include the corrected value in updated_summary so fact extraction can update DDB
    - Do NOT add this to "disputed_facts" — it's already corrected, not disputed
  - **Case B: User says it's wrong WITHOUT providing the new value** (e.g., "the address is wrong", "that insurance info is outdated"):
    - Add the fact key to "disputed_facts" with has_correction=false
    - Add a follow-up question in missing_or_unclear asking for the correct value
- IMPORTANT: "disputed_facts" should ONLY contain facts the user flagged as wrong but did NOT provide a replacement for

## Output Format (JSON):

{{
  "updated_summary": "Bullet-point summary of all collected information",
  "key_info_status": [
    {{"question": "key question 1", "status": "collected|partial|missing"}},
    {{"question": "key question 2", "status": "collected|partial|missing"}}
  ],
  "readiness_to_proceed": "ready" | "can_proceed_but_incomplete" | "needs_more",
  "missing_or_unclear": [
    "Follow-up question 1?",
    "Follow-up question 2?"
  ],
  "detected_prerequisites": [
    {{"type": "Prerequisite Name", "reason": "Why it's important", "evidence": "What user said"}}
  ],
  "disputed_facts": [
    {{"fact_key": "namespace.key", "reason": "User said this is wrong/outdated", "has_correction": false}}
  ],
  "user_signals": {{
    "wants_to_proceed_with_existing_info": false,
    "consent_response": "none",
    "wants_to_abandon_current_task": false,
    "abandon_reason": null
  }},
  "suggested_response": "Natural conversational message to user (MUST include prerequisite proposal if detected_prerequisites is not empty)"
}}

### user_signals field rules:

- **wants_to_proceed_with_existing_info** (bool): True if the user is signaling they want to move forward with whatever info has been collected so far, even if incomplete. Examples: "just proceed with what we have", "that's enough", "we can start now", "that's about it", "that's all", "go ahead", "proceed". If true, you MUST also set readiness_to_proceed to "ready" (override your assessment — respect user's explicit wish).
- **consent_response** ("yes" | "no" | "none"): The user's answer to a yes/no question posed by the assistant (e.g., "Would you like to handle Medicaid first?"). "yes" = user agrees, "no" = user declines, "none" = no yes/no question was pending or user didn't address it. Detect nuanced expressions: "sure, let's do that" = yes, "not right now" = no, "I think so" = yes.
- **wants_to_abandon_current_task** (bool): True if the user wants to STOP the current task entirely and switch to something else. Examples: "never mind", "I don't want to do this anymore", "let's do something else", "forget it". This is different from wants_to_proceed — abandon means CANCEL, proceed means FINISH with current info.
- **abandon_reason** (string | null): If wants_to_abandon_current_task is true, briefly describe what the user wants instead. null otherwise.

## Examples:

**Example 1: User unintentionally reveals prerequisite**
User latest reply: "I'm not sure about the budget, because I haven't applied for Medicaid yet and don't know how much it would cover"

Output:
{{
  "updated_summary": "- Budget: uncertain, awaiting Medicaid application result\\n- Location: Chicago",
  "key_info_status": [
    {{"question": "What is your approximate budget?", "status": "partial"}},
    {{"question": "When do you need to start?", "status": "missing"}}
  ],
  "readiness_to_proceed": "needs_more",
  "missing_or_unclear": ["When do you need to start?"],
  "detected_prerequisites": [
    {{
      "type": "Medicaid Application",
      "reason": "Need to know Medicaid coverage to determine budget for caregiver",
      "evidence": "User said 'I haven't applied for Medicaid yet and don't know how much it would cover'"
    }}
  ],
  "user_signals": {{
    "wants_to_proceed_with_existing_info": false,
    "consent_response": "none",
    "wants_to_abandon_current_task": false,
    "abandon_reason": null
  }},
  "suggested_response": "I noticed you haven't applied for Medicaid yet. The Medicaid application is important for determining your caregiver budget, as it affects your out-of-pocket costs. Would you like me to help with the Medicaid application first?"
}}

**Example 2: User provides info normally, no prerequisite**
User latest reply: "Budget is about 2000-3000 a month, would like to start as soon as possible"

Output:
{{
  "updated_summary": "- Budget: $2,000–3,000/month\\n- Timeline: ASAP",
  "key_info_status": [
    {{"question": "What is your approximate budget?", "status": "collected"}},
    {{"question": "When do you need to start?", "status": "collected"}}
  ],
  "readiness_to_proceed": "can_proceed_but_incomplete",
  "missing_or_unclear": ["Any special requirements?"],
  "detected_prerequisites": [],
  "user_signals": {{
    "wants_to_proceed_with_existing_info": false,
    "consent_response": "none",
    "wants_to_abandon_current_task": false,
    "abandon_reason": null
  }},
  "suggested_response": "Got it. Budget $2,000–3,000/month, hoping to start ASAP. Do you have any special requirements? (e.g., bilingual, nursing experience, etc.) Of course, if you don't have all the details right now, I can start searching for caregivers based on what we have."
}}

**Example 3: User says enough, wants to proceed**
User latest reply: "No other requirements, you can start searching"

Output:
{{
  "updated_summary": "- (previous summary preserved)",
  "key_info_status": [],
  "readiness_to_proceed": "ready",
  "missing_or_unclear": [],
  "detected_prerequisites": [],
  "user_signals": {{
    "wants_to_proceed_with_existing_info": true,
    "consent_response": "none",
    "wants_to_abandon_current_task": false,
    "abandon_reason": null
  }},
  "suggested_response": "Great, I'll start working on this with the information we have."
}}

**Example 4: User agrees to handle prerequisite**
User latest reply: "Yes, help me with the Medicaid application first"

Output:
{{
  "updated_summary": "- (previous summary preserved)",
  "key_info_status": [],
  "readiness_to_proceed": "needs_more",
  "missing_or_unclear": [],
  "detected_prerequisites": [],
  "user_signals": {{
    "wants_to_proceed_with_existing_info": false,
    "consent_response": "yes",
    "wants_to_abandon_current_task": false,
    "abandon_reason": null
  }},
  "suggested_response": "Okay, let's handle the Medicaid application first."
}}

**Example 5: User abandons current task**
User latest reply: "Forget it, I don't want to apply for Medicaid. Just help me find a caregiver."

Output:
{{
  "updated_summary": "- (previous summary preserved)",
  "key_info_status": [],
  "readiness_to_proceed": "needs_more",
  "missing_or_unclear": [],
  "detected_prerequisites": [],
  "user_signals": {{
    "wants_to_proceed_with_existing_info": false,
    "consent_response": "none",
    "wants_to_abandon_current_task": true,
    "abandon_reason": "User wants to skip Medicaid application and go back to finding a caregiver"
  }},
  "suggested_response": "Okay, let's skip the Medicaid application and go ahead with finding a caregiver."
}}

Respond ONLY with the JSON object, no other text."""

def _infer_request_type(request_name: str, user_request: str) -> str:
    """Best-effort keyword inference for request_type when LLM omits it."""
    text = f"{request_name} {user_request}".lower()
    mapping = [
        ("caregiver", "find_caregiver"),
        ("home care", "find_caregiver"),
        ("medicaid", "medicaid_application"),
        ("medicare", "medicare_enrollment"),
        ("insurance renewal", "insurance_renewal"),
        ("insurance appeal", "insurance_appeal"),
        ("insurance", "insurance_eligibility_check"),
        ("provider", "find_provider"),
        ("doctor", "schedule_appointment"),
        ("appointment", "schedule_appointment"),
        ("medication", "medication_management"),
        ("refill", "medication_refill"),
        ("power of attorney", "poa_setup"),
        ("advance directive", "advance_directive"),
        ("respite", "respite_care"),
        ("care plan", "care_plan_review"),
        ("ssi", "ssi_ssd_application"),
        ("ssdi", "ssi_ssd_application"),
        ("bill", "bill_dispute"),
        ("reimbursement", "reimbursement_filing"),
    ]
    for keyword, rtype in mapping:
        if keyword in text:
            return rtype
    return "information_lookup"


async def _infer_subject_entity(
    user_request: str,
    client: Any = None,
    known_entity_ids: Optional[List[str]] = None,
) -> str:
    """
    Best-effort inference for subject_entity_id when LLM omits it.

    Uses known entities from DDB (if provided) to match against user text,
    then falls back to keyword matching and LLM inference.
    """
    text = user_request.lower()

    # ── Phase 0: Match against known DDB entities ──
    # This catches cases like "uncle", "aunt", etc. that aren't in
    # the hardcoded keyword list but exist in the user's fact store.
    if known_entity_ids:
        for eid in known_entity_ids:
            label = eid.split(":")[-1] if ":" in eid else eid
            if label.lower() != "unknown" and label.lower() in text:
                logger.info(f"Entity inferred from DDB: '{eid}' (matched '{label}' in text)")
                return eid

    # ── Phase 1: Keyword matching ──
    if "grandma" in text or "grandmother" in text or "grandpa" in text or "grandfather" in text:
        return "care_recipient:grandparent"
    if "my mom" in text or "mother" in text:
        return "care_recipient:mom"
    if "my dad" in text or "father" in text:
        return "care_recipient:dad"
    if "my spouse" in text or "husband" in text or "wife" in text:
        return "care_recipient:spouse"
    if "myself" in text or "my own" in text or " me " in text:
        return "user:self"

    # Chinese keywords
    if any(kw in text for kw in ["我妈", "母亲", "妈妈", "我娘", "老母亲", "我老妈"]):
        return "care_recipient:mom"
    if any(kw in text for kw in ["我爸", "父亲", "爸爸", "我爹", "老父亲", "我老爸"]):
        return "care_recipient:dad"
    if any(kw in text for kw in ["老公", "老婆", "配偶", "丈夫", "妻子", "爱人", "先生", "太太"]):
        return "care_recipient:spouse"
    if any(kw in text for kw in ["奶奶", "外婆", "爷爷", "外公", "姥姥", "姥爷", "祖母", "祖父"]):
        return "care_recipient:grandparent"

    # ── Phase 2: LLM fallback with known entities ──
    if client:
        try:
            # Build entity options from known DDB entities + defaults
            default_entities = [
                "care_recipient:mom", "care_recipient:dad",
                "care_recipient:spouse", "care_recipient:grandparent",
                "user:self",
            ]
            all_options = list(dict.fromkeys(
                (known_entity_ids or []) + default_entities + ["care_recipient:unknown"]
            ))
            options_block = "\n".join(f"- {e}" for e in all_options)

            llm_prompt = (
                "You are a subject-entity extractor for a caregiver assistant.\n"
                "Given the user's request, determine WHO the request is about.\n\n"
                f"User request: \"{user_request}\"\n\n"
                "Respond with EXACTLY ONE of these identifiers:\n"
                f"{options_block}\n\n"
                "Output ONLY the identifier, nothing else."
            )
            response = await client.async_chat(prompt=llm_prompt, max_tokens=30, temperature=0.0)
            result = response.strip().lower()
            if result in set(all_options):
                return result
        except Exception as e:
            logger.warning(f"LLM entity inference fallback failed: {e}")

    return "care_recipient:unknown"


async def llm_collection_plan(
    *,
    user_request: str,
    known_facts: Dict[str, Any],
    similar_requests: List[Dict[str, Any]] | None = None,
    recent_requests: List[Dict[str, Any]] | None = None,
    client: Any,  # TrackedAnthropicClient instance
) -> Dict[str, Any]:
    """
    LLM-based collection plan generation using Claude.

    Returns:
        {
            "request_name": str,
            "request_goal": str,
            "key_info_needed": List[str],
            "nice_to_have_info": List[str],
            "potential_prerequisites": List[Dict[str, str]],
            "routing_hint": str,
            "resume_from_request_id": str | None
        }
    """
    prompt = make_collection_plan_prompt(
        user_request=user_request,
        known_facts=known_facts,
        similar_requests=similar_requests,
        recent_requests=recent_requests,
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.2,
        )

        # Parse JSON response
        response_text = response.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            plan = json.loads(json_str)
        else:
            plan = json.loads(response_text)

        # Validate
        required = ["request_name", "request_goal", "key_info_needed", "routing_hint"]
        for field in required:
            if field not in plan:
                raise ValueError(f"Missing required field: {field}")

        # Ensure request_type and subject_entity_id have defaults
        if not plan.get("request_type"):
            plan["request_type"] = _infer_request_type(plan.get("request_name", ""), user_request)
        if not plan.get("subject_entity_id"):
            plan["subject_entity_id"] = await _infer_subject_entity(user_request, client=client)

        return plan

    except Exception as e:
        logger.warning(f"LLM collection plan failed: {e}. Falling back to default.")
        return await default_collection_plan(user_request, client=client)


async def default_collection_plan(user_request: str, client: Any = None) -> Dict[str, Any]:
    """Fallback heuristic-based collection plan."""
    return {
        "request_name": "General Request",
        "request_goal": user_request[:200],
        "request_type": _infer_request_type("General Request", user_request),
        "subject_entity_id": await _infer_subject_entity(user_request, client=client),
        "key_info_needed": [
            "What are the most important constraints or preferences? (budget, timeline, location, insurance, etc.)",
            "Is this for the caregiver or the care recipient?",
            "When would you like to start or complete this?",
        ],
        "nice_to_have_info": [],
        "potential_prerequisites": [],
        "routing_hint": "deep_search",
    }

async def llm_prerequisite_acceptance_response(
    *,
    prereq_type: str,
    prereq_reason: str,
    user_message: str,
    conversation_history: List[Dict[str, Any]],
    client: Any,
) -> str:
    """
    Generate a natural response when user accepts to handle a prerequisite task.
    Adapts to the language used in the conversation (English or Chinese).
    
    Args:
        prereq_type: Type of prerequisite (e.g., "medicaid", "medicare")
        prereq_reason: Reason why this prerequisite is needed
        user_message: User's acceptance message
        conversation_history: Recent conversation for language context
        client: Anthropic client
    
    Returns:
        Natural language response confirming prerequisite acceptance
    """
    # Detect language from recent conversation
    recent_text = " ".join([m.get("content", "") for m in conversation_history[-3:]])
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in recent_text)
    
    prompt = f"""Generate a brief, natural confirmation message when a user agrees to handle a prerequisite task first.

Context:
- Prerequisite type: {prereq_type}
- Reason: {prereq_reason}
- User said: {user_message}
- Language: {"Chinese" if is_chinese else "English"}

Generate a friendly 1-2 sentence confirmation that:
1. Acknowledges their decision to handle the prerequisite first
2. Briefly mentions how this will help with their main goal

Keep it conversational and natural. Output ONLY the confirmation message, no explanations."""

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )
        return response.strip()
    except Exception as e:
        logger.warning(f"LLM prerequisite acceptance response failed: {e}. Falling back to default.")
        # Fallback based on detected language
        return f"Okay, let's handle {prereq_type} first. This will help us better understand your options going forward."


async def llm_info_collection_summarize(
    *,
    request_goal: str,
    key_info_needed: List[str],
    nice_to_have_info: List[str],
    previous_summary: str,
    user_latest_reply: str,
    conversation_history: List[Dict[str, Any]],
    known_facts: Dict[str, Any],
    client: Any,  # TrackedAnthropicClient instance
    last_asked_questions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    LLM-based information summarization and readiness assessment.

    Returns:
        {
            "updated_summary": str,
            "key_info_status": List[Dict],
            "readiness_to_proceed": "ready" | "can_proceed_but_incomplete" | "needs_more",
            "missing_or_unclear": List[str],
            "detected_prerequisites": List[Dict],  # NEW: Unintentionally revealed prerequisites
            "suggested_response": str
        }
    """
    prompt = make_info_collection_summarize_prompt(
        request_goal=request_goal,
        key_info_needed=key_info_needed,
        nice_to_have_info=nice_to_have_info,
        previous_summary=previous_summary,
        user_latest_reply=user_latest_reply,
        conversation_history=conversation_history,
        known_facts=known_facts,
        last_asked_questions=last_asked_questions,
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=1200,
            temperature=0.2,
        )

        # Parse JSON response
        response_text = response.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)

        # Validate
        required = ["updated_summary", "readiness_to_proceed", "suggested_response"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        return result

    except Exception as e:
        logger.warning(f"LLM info collection summarize failed: {e}. Falling back to default.")
        return default_parse_user_answer(user_latest_reply, previous_summary)


def default_parse_user_answer(user_reply: str, previous_summary: str = "") -> Dict[str, Any]:
    """Fallback heuristic-based answer parsing."""
    # Simple concatenation
    updated_summary = previous_summary
    if previous_summary:
        updated_summary += f"\n- User added: {user_reply[:200]}"
    else:
        updated_summary = f"- User said: {user_reply[:200]}"

    return {
        "updated_summary": updated_summary,
        "key_info_status": [],
        "readiness_to_proceed": "needs_more",
        "missing_or_unclear": ["Could you provide more details?"],
        "suggested_response": "Got it. Could you tell me more?",
    }

def make_turn_mode_prompt(
    *,
    recent_turns: List[Dict[str, Any]],
    current_agent: Optional[str],
    recent_agents: List[str],
    request_state: Dict[str, Any],
) -> str:
    """
    Generate a prompt for Claude to decide turn mode and routing.

    Args:
        recent_turns: Last 20 turns of conversation
        current_agent: Current active agent (if any)
        recent_agents: List of recent agents that interacted with user
        request_state: Current request state (awaiting_input, prereq_gate, etc.)
    """
    agent_descriptions = {
        "info_collection": "Gathers required information from user through structured questions. Asks about constraints, timeline, preferences, and other details needed to complete a request.",
        "deep_search": "Finds specific local providers, agencies, facilities, or services near a location. Has access to Google Places and web search. Use ONLY when the user wants to FIND something in a specific area.",
        "user_info": "Provides summaries and historical information about the caregiver or care recipient. Handles requests like 'summarize mom's health last month' or 'show medication history'.",
        "domain_expert": "Generates long-form deliverables: detailed guidance documents, email drafts, formal letters, checklists, step-by-step plans. Use when the user needs a WRITTEN ARTIFACT produced — not just a quick answer.",
        "front_end_emotional_support": "Provides emotional support and empathy. Handles distressed users expressing anxiety, burnout, overwhelm, or needing to vent.",
        "quick_answer": "Answers quick factual questions, definitions, policy lookups, and clarifications. Use for 'what is X?', 'how does Y work?', eligibility questions, or any standalone question that needs a direct answer without creating a task. Can optionally search the web but often answers from knowledge alone.",
        "human_comm": "Proposes and manages handoff to a human clinical team. Used when automated assistance has been insufficient after multiple attempts, or when the user explicitly requests human help.",
    }

    # Format recent conversation
    conversation_context = "\n".join([
        f"[{turn.get('role', 'unknown').upper()}]{(' (' + turn.get('agent', '') + ')') if turn.get('agent') else ''}: {turn.get('content', '')[:200]}"
        for turn in recent_turns[-20:]  # Last 20 turns max
    ])

    # Format recent agent activity
    recent_agent_info = ""
    if recent_agents:
        recent_agent_info = f"\nRecent agents (most recent first): {', '.join(recent_agents[:5])}"

    # Format active request info
    req_name = request_state.get('request_name', 'None')
    req_goal = request_state.get('request_goal', 'None')
    req_status = request_state.get('request_status', 'None')

    return f"""You are a conversation router for a caregiver assistant system. Your job is to determine:
1. Whether the user's latest message is a CONTINUATION or a NEW_INTENT
2. Which agent should handle this turn

## Agent Capabilities:
{chr(10).join([f"- {name}: {desc}" for name, desc in agent_descriptions.items()])}

## Current State:
- Current agent: {current_agent or "None"}
{recent_agent_info}
- Request awaiting user input: {request_state.get('awaiting_user_input', False)}
- Prerequisite consent pending: {request_state.get('prereq_gate_status') == 'proposed'}

## Active Request:
- Name: {req_name}
- Goal: {req_goal}
- Status: {req_status}
  (Status meanings: "collecting" = gathering info, "validated" = info complete and handed off to execution agent, "executed" = task done, "paused" = waiting for another task, "completed" = fully closed)

## Recent Conversation (last up to 20 turns):
{conversation_context}

## Your Task:
Analyze the user's LATEST message in context of the recent conversation. Determine:

1. **turn_mode**: Choose ONE:
   - "continuation": User is responding to the current conversation flow, answering questions, asking follow-up questions about results, or making requests contextually related to what the current agent is handling
   - "new_intent": User is switching topics, asking for something unrelated, requesting to go back to a previous task, requesting something the current agent cannot handle, OR simply acknowledging/accepting results with no follow-up (e.g. "okay, got it", "thanks", "got it"). When the user is done with the current task, that IS a new intent — the upstream_delegator will decide whether to mark the task as executed.

2. **recommended_agent**: Choose the BEST agent from: {list(agent_descriptions.keys())}
   - For "continuation": Can be the current agent OR a different agent if the conversation naturally progressed to a new phase
   - For "new_intent": Must be the agent best suited for the new topic. If the user is just acknowledging results (e.g. "okay"), you can recommend any agent — the upstream_delegator will handle lifecycle decisions.

3. **reason**: Brief explanation (1-2 sentences) of your decision

## IMPORTANT: When to use CONTINUATION vs NEW_INTENT

After an execution agent (deep_search, domain_expert, user_info) has delivered results:
- User says "okay, got it" / "thanks" / "okay" with NO question → **new_intent** (user is done with this task; upstream_delegator will mark it executed)
- User asks a follow-up question ABOUT the results (e.g. "What's the contact info for the first result?") → **continuation** (same agent)
- User says "I want to continue with the caregiver search from earlier" or raises a different topic → **new_intent**
- User expresses emotions like "Taking care of elderly parents is exhausting" → **new_intent** (needs emotional support)

During info collection (status = "collecting"):
- User answers a question or says "just proceed with what we have" → **continuation**
- User switches to a completely different topic → **new_intent**

## Examples of NEW_INTENT:
- Assistant (deep_search) returned search results, user says "Okay, got it, thanks" → NEW_INTENT (user acknowledged results, done with task)
- Assistant (domain_expert) provided a checklist, user says "I understand" → NEW_INTENT (user understood, done)
- Assistant (info_collection) is asking for task details, but user suddenly asks "can you summarize my mom's health last month?" → NEW_INTENT (unrelated topic)
- User says "actually, forget that, I need help with something else" → NEW_INTENT (explicit topic switch)
- User says "I want to continue with the caregiver search from earlier" → NEW_INTENT (wants to switch to a previous request)
- User says "Taking care of elderly parents is exhausting" → NEW_INTENT (emotional support needed)

## Examples of CONTINUATION:
- Assistant (info_collection) asks "what's your budget?", user responds "around $2000/month" → CONTINUATION (answering question)
- Assistant (info_collection) finishes gathering info, conversation naturally moves to deep_search to execute the task → CONTINUATION (natural progression, but agent changes to deep_search)
- Assistant (domain_expert) provides Medicaid checklist, user asks follow-up "what documents do I need for step 3?" → CONTINUATION (related follow-up)
- Assistant (deep_search) returned results, user asks "What's the contact info for the first result?" → CONTINUATION (follow-up about results)

## Escalation Detection
If the current agent is deep_search and ANY of these apply, set recommended_agent to "human_comm":
- The user expresses frustration, dissatisfaction, or says the results are wrong/unhelpful
- The user explicitly asks to talk to a person, human, or clinical team
- The user says they want to give up on the current search approach
Do NOT recommend human_comm for minor clarifications or simple follow-up questions.

## Output Format (JSON):
{{
    "turn_mode": "continuation" | "new_intent",
    "recommended_agent": "agent_name",
    "reason": "brief explanation"
}}

Respond ONLY with the JSON object, no other text."""

async def llm_turn_mode_decision(
    *,
    state: Dict[str, Any],
    client: Any,  # TrackedAnthropicClient instance
) -> Dict[str, Any]:
    """
    LLM-based turn mode decision using Claude.

    Returns:
        {
            "turn_mode": "continuation" | "new_intent",
            "recommended_agent": str,
            "reason": str
        }
    """

    # Extract messages and enrich with agent info
    messages = state.get("messages", []) or []
    routing_history = state.get("routing", {})
    current_agent = routing_history.get("current_agent")

    # Build recent turns with agent annotations
    recent_turns = []
    recent_agents = []

    for msg in messages[-20:]:  # Last 20 turns max
        turn = {
            "role": msg.get("role"),
            "content": msg.get("content", ""),
        }

        # Try to infer which agent spoke (for assistant messages)
        if msg.get("role") == "assistant":
            # Agent might be annotated in metadata
            agent = msg.get("agent") or msg.get("metadata", {}).get("agent")
            if agent:
                turn["agent"] = agent
                if agent not in recent_agents:
                    recent_agents.append(agent)

        recent_turns.append(turn)

    # If no agent annotations in messages, use current_agent as fallback
    if not recent_agents and current_agent:
        recent_agents = [current_agent]

    # Extract request state
    rm = state.get("request_manager") or {}
    rid = rm.get("active_request_id")
    req = ((rm.get("requests") or {}).get(rid)) if rid else None

    request_state = {}
    if req:
        request_state["awaiting_user_input"] = req.get("awaiting_user_input", False)
        gate = req.get("prereq_gate") or {}
        request_state["prereq_gate_status"] = gate.get("status")
        request_state["request_name"] = req.get("name", "")
        request_state["request_goal"] = req.get("goal", "")
        request_state["request_status"] = req.get("status", "")

    # Build prompt
    prompt = make_turn_mode_prompt(
        recent_turns=recent_turns,
        current_agent=current_agent,
        recent_agents=recent_agents,
        request_state=request_state,
    )

    try:
        # Call Claude
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=500,
            temperature=0.2,
        )

        # Parse JSON response
        # Try to extract JSON from response (handle cases where Claude adds extra text)
        response_text = response.strip()

        # Find JSON object in response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            decision = json.loads(json_str)
        else:
            # Fallback: try to parse entire response
            decision = json.loads(response_text)

        # Validate response structure
        if "turn_mode" not in decision or "recommended_agent" not in decision:
            raise ValueError("LLM response missing required fields")

        return {
            "turn_mode": decision.get("turn_mode", "continuation"),
            "recommended_agent": decision.get("recommended_agent"),
            "reason": decision.get("reason", "LLM decision"),
        }

    except Exception as e:
        # Fallback to simple heuristic if LLM call fails
        logger.warning(f"LLM turn mode decision failed: {e}. Falling back to heuristic.")
        return default_turn_mode_decision(state=state)


def default_turn_mode_decision(*, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback heuristic-based turn mode decision (used when LLM call fails).
    """
    text = ""
    for m in reversed(state.get("messages", []) or []):
        if m.get("role") == "user":
            text = (m.get("content") or "").strip().lower()
            break

    switch_signals = ["算了", "不做了", "换一个", "先不", "stop", "never mind", "forget it", "另外", "新问题"]
    if any(s in text for s in switch_signals):
        return {"turn_mode": "new_intent", "reason": "User explicit switch", "recommended_agent": None}

    rm = state.get("request_manager") or {}
    rid = rm.get("active_request_id")
    req = ((rm.get("requests") or {}).get(rid)) if rid else None

    current_agent = ((state.get("routing") or {}).get("current_agent"))

    if req:
        if req.get("awaiting_user_input") is True:
            return {
                "turn_mode": "continuation",
                "reason": "Active request awaiting user input",
                "recommended_agent": current_agent or "info_collection"
            }
        gate = req.get("prereq_gate") or {}
        if gate.get("status") == "proposed":
            return {
                "turn_mode": "continuation",
                "reason": "Prerequisite consent pending",
                "recommended_agent": current_agent or "info_collection"
            }
        ds = req.get("deep_search_state") or {}
        if ds.get("awaiting_user_input") is True:
            return {
                "turn_mode": "continuation",
                "reason": "Deep search awaiting user input",
                "recommended_agent": "deep_search"
            }

    return {
        "turn_mode": "continuation",
        "reason": "Default",
        "recommended_agent": current_agent or "info_collection"
    }

def make_upstream_delegator_prompt(
    *,
    user_message: str,
    turn_router_decision: Dict[str, Any],
    current_request_state: Optional[Dict[str, Any]],
    all_requests: List[Dict[str, Any]],
    similar_historical_requests: List[Dict[str, Any]] = None,
    recent_conversation: List[Dict[str, Any]],
    known_facts: Dict[str, Any],
) -> str:
    """
    Generate a prompt for Claude to decide how to handle a NEW_INTENT.

    Args:
        user_message: Latest user message
        turn_router_decision: Decision from turn_router (turn_mode, recommended_agent, reason)
        current_request_state: Current active request state (if any)
        all_requests: Summary of ALL in-session requests (for resume decisions)
        similar_historical_requests: Similar requests from past sessions (from vector search)
        recent_conversation: Last 10-15 turns
        known_facts: Known facts about caregiver and care_recipient
    """
    agent_descriptions = {
        "info_collection": "Gathers required information from user through conversational questions. Asks about constraints, timeline, preferences, and other details needed to complete a request.",
        "deep_search": "Finds specific local providers, agencies, facilities, or services near a location. Has access to Google Places and web search. Use ONLY when the user wants to FIND something in a specific area.",
        "user_info": "Provides summaries and historical information about the caregiver or care recipient. Handles requests like 'summarize mom's health last month' or 'show medication history'.",
        "domain_expert": "Generates long-form deliverables: detailed guidance documents, email drafts, formal letters, checklists, step-by-step plans. Use when the user needs a WRITTEN ARTIFACT produced — not just a quick answer.",
        "front_end_emotional_support": "Provides emotional support and empathy. Handles distressed users expressing anxiety, burnout, overwhelm, or needing to vent.",
        "quick_answer": "Answers quick factual questions, definitions, policy lookups, and clarifications. Use for 'what is X?', 'how does Y work?', eligibility questions, or any standalone question that needs a direct answer without creating a task. Can optionally search the web but often answers from knowledge alone.",
    }

    # Format recent conversation
    conversation_context = "\n".join([
        f"[{turn.get('role', 'unknown').upper()}]{(' (' + turn.get('agent', '') + ')') if turn.get('agent') else ''}: {turn.get('content', '')[:200]}"
        for turn in recent_conversation[-15:]  # Last 15 turns
    ])

    # Format current request info
    current_request_info = "None (no active request)"
    if current_request_state:
        collected_info = (current_request_state.get('info_collection_state') or {}).get('summary_of_collected_info', 'No summary yet')
        current_request_info = f"""
Active Request:
- ID: {current_request_state.get('request_id', 'unknown')}
- Name: {current_request_state.get('name', 'unknown')}
- Goal: {current_request_state.get('goal', 'unknown')}
- Status: {current_request_state.get('status', 'unknown')}
- Current Agent: {current_request_state.get('current_agent', 'unknown')}
- Stage: {current_request_state.get('stage_detail', 'unknown')}
- Awaiting User Input: {current_request_state.get('awaiting_user_input', False)}
- Collected Information Summary:
{collected_info}
"""

    # Format ALL in-session requests
    if all_requests:
        lines = []
        for r in all_requests:
            active_marker = " ← ACTIVE" if r.get("is_active") else ""
            parent = f", parent={r.get('parent_request_id')}" if r.get("parent_request_id") else ""
            info_preview = f", collected_info=\"{r.get('collected_info', '')[:100]}\"" if r.get("collected_info") else ""
            lines.append(f"- [{r.get('request_id', '?')}] \"{r.get('name', '?')}\" (status={r.get('status', '?')}{parent}{info_preview}){active_marker}")
        all_requests_info = "\n".join(lines)
    else:
        all_requests_info = "No requests in session."

    # Format similar historical requests (from past sessions)
    if similar_historical_requests:
        hist_lines = []
        for h in similar_historical_requests[:3]:
            hist_lines.append(f"- \"{h.get('request_name', '?')}\" (status={h.get('completion_status', '?')}, theme={h.get('theme', '?')})")
        historical_info = "\n".join(hist_lines)
    else:
        historical_info = "None found."

    return f"""You are an upstream delegator for a caregiver assistant system. You are called when the turn_router detected a NEW_INTENT from the user.

Your job is to decide:
1. What TYPE of new intent is this?
2. Which agent should handle it?
3. What should happen to the current active request (if any)?

## Agent Capabilities:
{chr(10).join([f"- {name}: {desc}" for name, desc in agent_descriptions.items()])}

## Turn Router Decision:
The turn_router already determined this is a NEW_INTENT:
- Turn Mode: {turn_router_decision.get('turn_mode', 'unknown')}
- Recommended Agent: {turn_router_decision.get('recommended_agent', 'unknown')}
- Reason: {turn_router_decision.get('reason', 'unknown')}

## Current Request State:
{current_request_info}

## All In-Session Requests:
{all_requests_info}

## Similar Historical Requests (from past sessions):
{historical_info}

## Recent Conversation (last 15 turns):
{conversation_context}

## Known Facts:
{json.dumps(known_facts, indent=2, ensure_ascii=False)}

## Your Task:

Analyze the user's latest message and determine:

### 1. **action_type** - Choose ONE:

⚠️ **IMPORTANT: Check for "task_acknowledged" FIRST before considering other action types!**
If the current request status is "validated" or "executed" (meaning results have already been delivered), and the user's message is a simple acknowledgment like "okay", "got it", "thanks", "got it", "okay", "I understand" — this is almost certainly "task_acknowledged". Do NOT create a new request for acknowledgments!

- **"task_acknowledged"**: User is acknowledging/accepting the results of the current task. The task is DONE. No new task is being requested. The user is simply saying "thanks", "got it", "okay", etc.
  - Example: deep_search returned results, user says "Okay, got it, thanks" → task is done
  - Example: domain_expert provided a checklist, user says "I understand" → task is done
  - Example: user says "okay" after results were delivered → task is done
  - Current request: MARK AS EXECUTED (task fulfilled)
  - New request: NO
  - recommended_agent: not needed (will be ignored)
  - CRITICAL: If the current request status is "validated" or "executed" and the user message contains NO new question or topic, this is ALWAYS "task_acknowledged". Never create a new request for simple acknowledgments.

- **"resume_existing"**: User wants to go back to a PREVIOUS request that already exists in the session. Look at "All In-Session Requests" above — if the user's intent matches a paused, executed, or collecting request, resume it instead of creating a new one.
  - Example: User says "I want to continue with the caregiver search" and there's a paused "Caregiver Search" request → resume it
  - Example: User says "can we go back to the Medicaid thing?" and there's an executed Medicaid request → resume it
  - Current request: PAUSE (if still active and not completed)
  - New request: NO — reuse the existing request by its request_id

- **"natural_progression"**: User is saying "proceed to next step" or "that's all the info I have" while in the middle of a request. This is the NATURAL next phase of the current request.
  - Example: User was answering info_collection questions, now says "Okay, go ahead and proceed" or "I've given you all the info"
  - Current request: KEEP ACTIVE, just change the stage/agent
  - New request: NO

- **"prerequisite_task"**: User is requesting to handle a prerequisite task before continuing the current request. The new task is RELATED and must be completed BEFORE the current request can proceed.
  - Example: User was finding caregivers, now says "Help me apply for Medicaid first" or "I need to get this done first"
  - Current request: PAUSE (mark as blocked by prerequisite)
  - New request: YES (create new request, mark as prerequisite of current)

- **"new_unrelated_task"**: User is switching to a completely DIFFERENT, UNRELATED task that does NOT match any existing request.
  - Example: User was finding caregivers, now says "forget it, I want to check my mom's health records from last month" or "forget that, help me with something else"
  - Current request: PAUSE or MARK_EXECUTED (mark_executed if results were already delivered, pause if still in progress)
  - New request: YES (create new request, no prerequisite relationship)
  - IMPORTANT: Before choosing this, check "All In-Session Requests" — if a matching request already exists, use "resume_existing" instead!

- **"same_agent_continue"**: User is still continuing with the current conversation flow. This should be RARE since turn_router already said NEW_INTENT.
  - Current request: KEEP ACTIVE, no change
  - New request: NO

### 2. **recommended_agent** - Choose from: {list(agent_descriptions.keys())}

Consider:
- For "resume_existing": Which agent should handle the RESUMED request? If the request was in "collecting" status, use info_collection. If it was "validated" or "executed", consider deep_search, domain_expert, or the agent that last worked on it.
- For "natural_progression": Which agent should handle the NEXT phase of the current request? (Could be any of the 5 agents)
- For "prerequisite_task" or "new_unrelated_task": Which agent should handle the NEW request? (Usually info_collection if needs info gathering, but could be user_info, domain_expert, deep_search, or front_end)

### 3. **resume_request_id** (ONLY if action_type is "resume_existing"):
- The request_id of the existing request to resume. Must be one of the IDs from "All In-Session Requests".

### 4. **new_request_info** (ONLY if action_type is "prerequisite_task" or "new_unrelated_task"):
- **name**: Short name for the new request (e.g., "Medicaid Application", "Health Summary Request")
- **goal**: Detailed description of what user wants (1-2 sentences)
- **is_prerequisite_of_current**: true if prerequisite_task, false if new_unrelated_task

### 5. **current_request_action** - Choose ONE:
- **"continue"**: Keep as active, just update stage (for natural_progression or same_agent_continue)
- **"pause"**: Pause and add to pending queue (for prerequisite_task, new_unrelated_task, or resume_existing when current is still active)
- **"mark_executed"**: Mark the current request as EXECUTED/DONE. Use this when the current request has been fulfilled (status is "validated" or later, and results have been delivered) AND the user is moving on. This is the correct action when the user acknowledges results ("okay, got it", "thanks") or switches to a new topic after results were delivered.
- **"complete"**: Mark as fully completed and closed (RARE - only if user explicitly says "we're done with that" and no further action is expected)

### 6. **reason**: Brief explanation (2-3 sentences) of your decision

## Examples:

**Example 1: Resume Existing Request**
- All requests: [{{"request_id": "req-abc", "name": "Caregiver Search", "status": "paused", "collected_info": "Location: Chicago, Budget: $3000..."}}]
- User: "I want to continue with the caregiver search"
- Output:
{{
  "action_type": "resume_existing",
  "recommended_agent": "info_collection",
  "resume_request_id": "req-abc",
  "current_request_action": "pause",
  "reason": "User wants to go back to the paused Caregiver Search request. Resuming existing request instead of creating a new one to preserve collected information."
}}

**Example 2: Natural Progression**
- Current: info_collection agent asking questions about finding caregivers
- User: "Okay, I've given you all the info, go ahead"
- Output:
{{
  "action_type": "natural_progression",
  "recommended_agent": "deep_search",
  "current_request_action": "continue",
  "reason": "User has provided info and wants to proceed. Natural next step is to execute search for caregivers."
}}

**Example 3: Prerequisite Task**
- Current: info_collection for finding caregivers
- User: "Wait, I just realized I haven't applied for Medicaid yet. Help me with that first."
- Output:
{{
  "action_type": "prerequisite_task",
  "recommended_agent": "info_collection",
  "new_request_info": {{
    "name": "Medicaid Application",
    "goal": "Help user apply for Medicaid in their state",
    "is_prerequisite_of_current": true
  }},
  "current_request_action": "pause",
  "reason": "User identified a prerequisite task (Medicaid application) that must be completed before finding caregivers. Need to pause current request and create new prerequisite request."
}}

**Example 4: New Unrelated Task**
- Current: info_collection for finding caregivers
- User: "Forget it, I don't want to look for a caregiver now. Can you check my mom's health records from last month?"
- Output:
{{
  "action_type": "new_unrelated_task",
  "recommended_agent": "user_info",
  "new_request_info": {{
    "name": "Health Record Summary",
    "goal": "Retrieve and summarize care recipient's health records from last month",
    "is_prerequisite_of_current": false
  }},
  "current_request_action": "pause",
  "reason": "User switched to a completely different topic (health records) unrelated to caregiver search. Need to pause current request and create new request."
}}

**Example 5: Emotional Support Interruption**
- Current: deep_search executing caregiver search
- User: "I really can't take it anymore. Watching my mom's condition get worse every day, I don't know what to do..."
- Output:
{{
  "action_type": "new_unrelated_task",
  "recommended_agent": "front_end_emotional_support",
  "new_request_info": {{
    "name": "Emotional Support Session",
    "goal": "Provide emotional support to distressed caregiver",
    "is_prerequisite_of_current": false
  }},
  "current_request_action": "mark_executed",
  "reason": "User is expressing severe emotional distress and needs immediate emotional support. The previous caregiver search task was already validated/executed, so marking it as executed before switching to emotional support."
}}

**Example 6: User Acknowledges Results (Task Done)**
- Current request: "Caregiver Search" (status=validated), deep_search just returned results
- User: "Okay, got it, thanks"
- Output:
{{
  "action_type": "task_acknowledged",
  "recommended_agent": "info_collection",
  "current_request_action": "mark_executed",
  "reason": "User acknowledged the search results with no follow-up questions. The caregiver search task is complete. Marking as executed."
}}

**Example 7: Switch Topic After Results Delivered**
- Current request: "Caregiver Search" (status=validated), deep_search returned results
- User: "I'd also like to learn about nursing home options"
- Output:
{{
  "action_type": "new_unrelated_task",
  "recommended_agent": "info_collection",
  "new_request_info": {{
    "name": "Nursing Home Information",
    "goal": "Help user learn about nursing home options",
    "is_prerequisite_of_current": false
  }},
  "current_request_action": "mark_executed",
  "reason": "User is switching to a new topic (nursing homes) after caregiver search results were delivered. The caregiver search is done — marking as executed, not just paused."
}}

## IMPORTANT: When to use "mark_executed" vs "pause"
- **mark_executed**: The current request's work is DONE — results have been delivered (status is "validated" or "executed") and the user is satisfied or moving on. The task was fulfilled.
- **pause**: The current request is NOT done — it's still in progress (status is "collecting" or "created") and the user is temporarily switching to something else. The task should be resumed later.

## Output Format (JSON):
Respond ONLY with a valid JSON object matching this structure:

{{
  "action_type": "task_acknowledged" | "resume_existing" | "natural_progression" | "prerequisite_task" | "new_unrelated_task" | "same_agent_continue",
  "recommended_agent": "agent_name",
  "resume_request_id": "string",  // ONLY if action_type is resume_existing
  "new_request_info": {{  // ONLY if action_type is prerequisite_task or new_unrelated_task
    "name": "string",
    "goal": "string",
    "is_prerequisite_of_current": true | false
  }},
  "current_request_action": "continue" | "pause" | "mark_executed" | "complete",
  "reason": "string"
}}

USER'S LATEST MESSAGE: {user_message}

Respond ONLY with the JSON object, no other text."""

async def llm_upstream_delegation(
    *,
    state: Dict[str, Any],
    client: Any,  # TrackedAnthropicClient instance
    similar_requests: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LLM-based upstream delegation using Claude.

    Returns:
        {
            "action_type": "resume_existing" | "natural_progression" | "prerequisite_task" | "new_unrelated_task" | "same_agent_continue",
            "recommended_agent": str,
            "resume_request_id": str,  // only if resume_existing
            "new_request_info": {  // optional, only if creating new request
                "name": str,
                "goal": str,
                "is_prerequisite_of_current": bool
            },
            "current_request_action": "continue" | "pause" | "complete",
            "reason": str
        }
    """
    from routing_utils import last_user_text

    user_message = last_user_text(state)

    # Get turn router decision
    routing = state.get("routing") or {}
    turn_router_decision = {
        "turn_mode": routing.get("turn_mode"),
        "recommended_agent": routing.get("llm_recommended_agent"),
        "reason": routing.get("turn_reason"),
    }

    # Get current request state and ALL in-session requests
    current_request_state = None
    rm = state.get("request_manager") or {}
    active_id = rm.get("active_request_id")
    all_requests_raw = rm.get("requests") or {}
    pending_queue = rm.get("pending_queue") or []

    if active_id:
        req = all_requests_raw.get(active_id)
        if req:
            current_request_state = {
                "request_id": active_id,
                "name": req.get("name", ""),
                "goal": req.get("goal", ""),
                "status": req.get("status", ""),
                "current_agent": ((state.get("routing") or {}).get("current_agent")),
                "stage_detail": req.get("stage_detail", ""),
                "awaiting_user_input": req.get("awaiting_user_input", False),
                "info_collection_state": req.get("info_collection_state") or {},
            }

    # Build summary of ALL in-session requests (for resume_existing decisions)
    all_requests_summary = []
    for rid, r in all_requests_raw.items():
        summary = {
            "request_id": rid,
            "name": r.get("name", ""),
            "goal": r.get("goal", ""),
            "status": r.get("status", ""),
            "is_active": rid == active_id,
            "parent_request_id": r.get("parent_request_id"),
            "collected_info": (r.get("info_collection_state") or {}).get("summary_of_collected_info", "")[:200],
        }
        all_requests_summary.append(summary)

    # Get recent conversation
    messages = state.get("messages", []) or []
    recent_conversation = []
    for msg in messages[-15:]:
        turn = {
            "role": msg.get("role"),
            "content": msg.get("content", ""),
        }
        if msg.get("role") == "assistant":
            agent = msg.get("agent") or msg.get("metadata", {}).get("agent")
            if agent:
                turn["agent"] = agent
        recent_conversation.append(turn)

    # Get known facts
    uc = state.get("user_context") or {}
    prof = (uc.get("profile_snapshot") or {})
    known_facts = {
        "caregiver": (prof.get("caregiver") or {}).get("facts", {}),
        "care_recipient": (prof.get("care_recipient") or {}).get("facts", {}),
    }

    # Build prompt
    prompt = make_upstream_delegator_prompt(
        user_message=user_message,
        turn_router_decision=turn_router_decision,
        current_request_state=current_request_state,
        all_requests=all_requests_summary,
        similar_historical_requests=similar_requests or [],
        recent_conversation=recent_conversation,
        known_facts=known_facts,
    )

    try:
        # Call Claude
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=800,
            temperature=0.2,
        )

        # Parse JSON response
        response_text = response.strip()

        # Find JSON object in response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            decision = json.loads(json_str)
        else:
            decision = json.loads(response_text)

        # Validate response structure
        required_fields = ["action_type", "recommended_agent", "current_request_action", "reason"]
        for field in required_fields:
            if field not in decision:
                raise ValueError(f"LLM response missing required field: {field}")

        return decision

    except Exception as e:
        # Fallback to simple heuristic if LLM call fails
        logger.warning(f"LLM upstream delegation failed: {e}. Falling back to heuristic.")
        return default_upstream_delegation(user_message, current_request_state)


def default_upstream_delegation(user_message: str, current_request_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fallback heuristic-based upstream delegation (used when LLM call fails).
    """
    text = (user_message or "").strip().lower()

    # Check for emotional distress
    if any(k in text for k in ["难受", "崩溃", "焦虑", "撑不住", "想哭", "panic", "anxious"]):
        return {
            "action_type": "new_unrelated_task",
            "recommended_agent": "front_end_emotional_support",
            "new_request_info": {
                "name": "Emotional Support Session",
                "goal": "Provide emotional support to distressed caregiver",
                "is_prerequisite_of_current": False
            },
            "current_request_action": "pause",
            "reason": "User appears distressed; front-end emotional support",
        }

    # Check for history/summary request
    if any(k in text for k in ["总结", "回顾", "过去", "history", "summary"]):
        return {
            "action_type": "new_unrelated_task",
            "recommended_agent": "user_info",
            "new_request_info": {
                "name": "Information Summary Request",
                "goal": "Provide historical summary or information review",
                "is_prerequisite_of_current": False
            },
            "current_request_action": "pause",
            "reason": "User asked for history/summary"
        }

    # Check for guidance/template request
    if any(k in text for k in ["规定", "规则", "流程", "怎么写", "邮件", "模板", "guideline"]):
        return {
            "action_type": "new_unrelated_task",
            "recommended_agent": "domain_expert",
            "new_request_info": {
                "name": "Guidance/Template Request",
                "goal": "Provide guidance, templates, or procedural knowledge",
                "is_prerequisite_of_current": False
            },
            "current_request_action": "pause",
            "reason": "User asked for written guidance/process"
        }

    # Check for "proceed" signals (natural progression)
    if any(k in text for k in ["proceed", "继续", "下一步", "go ahead", "that's all", "enough info"]):
        if current_request_state and current_request_state.get("status") == "collecting":
            return {
                "action_type": "natural_progression",
                "recommended_agent": "deep_search",
                "current_request_action": "continue",
                "reason": "User wants to proceed to next phase after info collection"
            }

    # Default: new task requiring info collection
    return {
        "action_type": "new_unrelated_task",
        "recommended_agent": "info_collection",
        "new_request_info": {
            "name": "General Request",
            "goal": user_message[:200],
            "is_prerequisite_of_current": False
        },
        "current_request_action": "pause" if current_request_state else "continue",
        "reason": "Default to info collection for new task requests"
    }


def make_emotional_support_prompt(
    *,
    user_message: str,
    conversation_history: List[Dict[str, Any]],
    known_facts: Dict[str, Any],
) -> str:
    """
    Generate a prompt for Claude to provide empathetic emotional support.
    """
    conversation_text = "\n".join([
        f"[{msg.get('role', '').upper()}]{(' (' + (msg.get('metadata', {}).get('agent', '') or '') + ')') if msg.get('metadata', {}).get('agent') else ''}: {msg.get('content', '')[:300]}"
        for msg in conversation_history[-15:]
    ])

    return f"""You are an empathetic emotional support companion for family caregivers. Caregiving is one of the hardest things a person can do — it's physically exhausting, emotionally draining, and often invisible to others.

## Your Role:
You are NOT a therapist. You are a warm, understanding companion who:
- Truly listens and validates feelings without judgment
- Acknowledges the real difficulty of what caregivers go through
- Gently reminds them that their feelings are normal and they're not alone
- Offers practical micro-suggestions ONLY when appropriate (not as the first response)
- Knows when to just be present rather than problem-solve

## Known Facts About the User:
Caregiver: {json.dumps(known_facts.get('caregiver', {}), indent=2, ensure_ascii=False)}
Care Recipient: {json.dumps(known_facts.get('care_recipient', {}), indent=2, ensure_ascii=False)}

## Recent Conversation:
{conversation_text}

## User's Latest Message:
{user_message}

## Guidelines:

1. **Match their language** — respond in the same language the user is writing in (English or Chinese).

2. **Lead with empathy, not solutions** — First acknowledge their feelings. Don't jump to advice. If someone says "I'm overwhelmed," don't immediately say "here's what you can do." Instead: "That sounds really heavy. Caregiving asks so much of you, and it makes sense that you'd feel this way."

3. **Be specific to what they shared** — Don't give generic "I understand" responses. Reference the actual situation they described. If they mentioned not having personal time, speak to that specifically.

4. **Normalize their experience** — Many caregivers feel guilty for being tired, frustrated, or wanting their own life back. Gently validate that these feelings are completely normal and don't make them a bad person.

5. **Keep it conversational and natural** — 2-4 sentences is usually enough. Don't write paragraphs. Don't use bullet points. This is a conversation, not a pamphlet.

6. **Gently open the door** — End with something that invites them to share more if they want, or pivot back to practical tasks when they're ready. Not a forced question — just an opening.

7. **Never minimize or silver-line** — Don't say "at least..." or "look on the bright side." Don't compare their situation to others. Their pain is valid as-is.

8. **If they've been venting across multiple turns** — You can gently offer one small, actionable suggestion (e.g., "Would it help if we looked into respite care options so you could get even a few hours to yourself?") but ONLY after sufficient validation.

Output ONLY the response message. No JSON, no explanations, no metadata."""


# ─────────────────────────────────────────────────────────────
# Fact Extraction (Slot→Fact Binding)
# ─────────────────────────────────────────────────────────────


def _format_existing_facts_for_extraction(
    already_extracted_keys: List[str],
    existing_profile_facts: Optional[Dict[str, Any]],
) -> str:
    """Format existing facts for the extraction prompt."""
    if existing_profile_facts:
        lines = []
        for key, info in existing_profile_facts.items():
            val = info.get("value", "?")
            if isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False)
            lines.append(f"  - {key}: {val}")
        if lines:
            return (
                "These facts are already stored in the user profile. "
                "Do NOT re-extract unless the value has CHANGED:\n"
                + "\n".join(lines)
            )
    if already_extracted_keys:
        return (
            "These fact keys are already stored — do NOT re-extract unless "
            "the value has CHANGED: " + ", ".join(already_extracted_keys)
        )
    return "(none yet)"


def _format_disputed_facts_for_extraction(
    disputed_fact_keys: Optional[List[str]],
    existing_profile_facts: Optional[Dict[str, Any]],
) -> str:
    """Format disputed facts for the extraction prompt."""
    if not disputed_fact_keys:
        return "(none)"
    lines = ["The user has indicated these stored facts are WRONG or outdated:"]
    for dk in disputed_fact_keys:
        old_val = ""
        if existing_profile_facts and dk in existing_profile_facts:
            v = existing_profile_facts[dk].get("value", "?")
            old_val = f" (old value: {v})"
        lines.append(f"  - {dk}{old_val}")
    lines.append(
        "If the user provided a corrected value, extract it using the SAME "
        "fact_key so the old value gets replaced."
    )
    return "\n".join(lines)


def make_fact_extraction_prompt(
    *,
    entity_id: str,
    request_type: str,
    updated_summary: str,
    key_info_needed: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    already_extracted_keys: List[str] = [],
    existing_profile_facts: Optional[Dict[str, Any]] = None,
    disputed_fact_keys: Optional[List[str]] = None,
) -> str:
    """
    Generate a prompt for the LLM to extract structured facts from an
    info_collection summary.

    The LLM proposes fact_key in namespace.facet format — creative, not
    constrained to the registry. The Key Resolver validates afterwards.
    """
    # Format key_info_needed for context
    info_needed_text = ""
    if key_info_needed:
        info_items = []
        for item in key_info_needed[:15]:
            if isinstance(item, dict):
                info_items.append(f"- {item.get('item', item.get('label', str(item)))}")
            else:
                info_items.append(f"- {item}")
        info_needed_text = "\n".join(info_items)

    # Last few user messages for context
    recent_msgs = []
    for msg in (conversation_history or [])[-8:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            recent_msgs.append(f"[{role}]: {content[:300]}")
    recent_conversation = "\n".join(recent_msgs) if recent_msgs else "(no recent messages)"

    return f"""You are a fact extraction engine for a caregiving platform.

Your job: extract structured facts from the conversation summary below.
Each fact should capture ONE specific piece of information that was collected.

## Context
Entity: {entity_id}
Request type: {request_type}

## Information Being Collected
{info_needed_text or "(general information gathering)"}

## Current Summary of Collected Information
{updated_summary}

## Recent Conversation
{recent_conversation}

## Output Format
Return a JSON array of extracted facts. Each fact has these fields:

```json
[
  {{
    "fact_key": "namespace.facet",
    "fact_label": "Human-readable label (2-8 words)",
    "value": "the extracted value",
    "value_type": "string|number|date|list|enum|object",
    "confidence": 0.0-1.0,
    "source_type": "user|agent_inference",
    "evidence": "Exact quote or close paraphrase from the conversation"
  }}
]
```

## Key Naming Rules
- Use `namespace.facet` format (e.g., "insurance.plan_type", "care_schedule.weekday_hours")
- Namespace = broad category (insurance, care_schedule, medication, contact, preference, mobility, cognition, provider, legal, financial, housing)
- Facet = specific attribute within that category
- Use snake_case, be specific but concise

## Confidence Scoring
- 0.9+ : User explicitly stated the fact ("Her insurance is Blue Cross")
- 0.7-0.9 : Clearly implied ("She goes to Dr. Chen every month" → provider relationship)
- 0.5-0.7 : Reasonably inferred from context
- <0.5 : Uncertain, speculative — do NOT include these

## Source Type
- "user" : User directly stated or confirmed the information
- "agent_inference" : Agent inferred from indirect statements

## Already Stored Facts (from DDB)
{_format_existing_facts_for_extraction(already_extracted_keys, existing_profile_facts)}

## Disputed/Corrected Facts
{_format_disputed_facts_for_extraction(disputed_fact_keys, existing_profile_facts)}

## Rules
1. Only extract facts with confidence >= 0.5
2. Maximum 15 facts per extraction
3. Each fact must have evidence (quote or close paraphrase)
4. Do NOT extract opinions, emotions, or conversation metadata
5. Do NOT extract facts about the caregiver unless they are relevant to care
6. Prefer specific values over vague descriptions
7. Skip any fact whose key is in "Already Stored Facts" UNLESS the value differs from what was previously stored
8. When the user CORRECTS an existing fact (e.g., "he wants X instead of Y"), extract the correction using the SAME fact_key as the old fact. Check "Already Stored Facts" for the key to target. Do NOT create a new key — update the existing one.
9. Do NOT extract the user's QUESTIONS or information requests as facts. If the user asks "tell me about his care schedule", that is a question, NOT a fact about the entity. Only extract concrete attributes.

## Example 1: Caregiver Search

Summary: "Mom needs care Monday-Friday, 8am-4pm. She has diabetes and uses a walker. Currently on Medicare Part A and B. Lives alone in a 2-bedroom apartment in Chicago."

Output:
```json
[
  {{"fact_key": "care_schedule.weekday_hours", "fact_label": "Weekday care schedule", "value": "Monday-Friday, 8am-4pm", "value_type": "string", "confidence": 0.95, "source_type": "user", "evidence": "Mom needs care Monday-Friday, 8am-4pm"}},
  {{"fact_key": "health.chronic_condition", "fact_label": "Chronic health condition", "value": "diabetes", "value_type": "string", "confidence": 0.95, "source_type": "user", "evidence": "She has diabetes"}},
  {{"fact_key": "mobility.assistive_device", "fact_label": "Assistive device used", "value": "walker", "value_type": "string", "confidence": 0.95, "source_type": "user", "evidence": "uses a walker"}},
  {{"fact_key": "insurance.plan_type", "fact_label": "Insurance plan type", "value": "Medicare Part A and B", "value_type": "string", "confidence": 0.95, "source_type": "user", "evidence": "Currently on Medicare Part A and B"}},
  {{"fact_key": "housing.living_situation", "fact_label": "Living arrangement", "value": "lives alone", "value_type": "string", "confidence": 0.95, "source_type": "user", "evidence": "Lives alone"}},
  {{"fact_key": "housing.type", "fact_label": "Housing type", "value": "2-bedroom apartment", "value_type": "string", "confidence": 0.9, "source_type": "user", "evidence": "2-bedroom apartment in Chicago"}},
  {{"fact_key": "housing.city", "fact_label": "City of residence", "value": "Chicago", "value_type": "string", "confidence": 0.95, "source_type": "user", "evidence": "apartment in Chicago"}}
]
```

## Example 2: Insurance Renewal

Summary: "Current plan expires March 31. Member ID is XYZ-789. Wants to keep same PCP Dr. Williams. Concerned about prescription coverage for metformin and lisinopril."

Output:
```json
[
  {{"fact_key": "insurance.expiration_date", "fact_label": "Insurance expiration date", "value": "March 31", "value_type": "date", "confidence": 0.95, "source_type": "user", "evidence": "Current plan expires March 31"}},
  {{"fact_key": "insurance.member_id", "fact_label": "Insurance member ID", "value": "XYZ-789", "value_type": "string", "confidence": 0.95, "source_type": "user", "evidence": "Member ID is XYZ-789"}},
  {{"fact_key": "provider.primary_care.name", "fact_label": "Primary care physician", "value": "Dr. Williams", "value_type": "string", "confidence": 0.9, "source_type": "user", "evidence": "Wants to keep same PCP Dr. Williams"}},
  {{"fact_key": "medication.current", "fact_label": "Current medications", "value": ["metformin", "lisinopril"], "value_type": "list", "confidence": 0.9, "source_type": "user", "evidence": "prescription coverage for metformin and lisinopril"}}
]
```

Now extract facts from the provided summary. Return ONLY a JSON array, no other text."""


async def llm_extract_facts(
    *,
    entity_id: str,
    request_type: str,
    updated_summary: str,
    key_info_needed: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    client: Any,
    already_extracted_keys: List[str] = [],
    existing_profile_facts: Optional[Dict[str, Any]] = None,
    disputed_fact_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Call the LLM to extract structured facts from an info_collection summary.

    Returns a list of dicts, each representing an extracted fact.
    Returns [] on any failure.
    """
    if not updated_summary or not updated_summary.strip():
        return []

    prompt = make_fact_extraction_prompt(
        entity_id=entity_id,
        request_type=request_type,
        updated_summary=updated_summary,
        key_info_needed=key_info_needed,
        conversation_history=conversation_history,
        already_extracted_keys=already_extracted_keys,
        existing_profile_facts=existing_profile_facts,
        disputed_fact_keys=disputed_fact_keys,
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2,
        )

        response_text = response.strip() if isinstance(response, str) else str(response).strip()

        # Extract JSON array from response
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            facts = json.loads(json_str)
            if isinstance(facts, list):
                # Cap at 15 facts, filter out low-confidence
                facts = [f for f in facts if isinstance(f, dict) and f.get("confidence", 0) >= 0.5]
                return facts[:15]

        logger.warning("LLM fact extraction returned non-list JSON")
        return []

    except Exception as e:
        logger.warning(f"LLM fact extraction failed: {e}")
        return []


RECONCILE_FACTS_PROMPT = """You are a fact reconciliation engine for a caregiving platform.

You are given two sets of facts about a care recipient:
1. EXISTING facts already stored in the database
2. NEWLY EXTRACTED facts from the latest conversation turn

Your job: decide what to do with each fact to keep the database clean and accurate.

## Existing Facts (currently in database)
{existing_facts_block}

## Newly Extracted Facts (from this conversation turn)
{new_facts_block}

## Recent Conversation (for context)
{conversation_block}

## Actions You Can Take

For each existing fact, decide: KEEP or DEPRECATE
For each new fact, decide: WRITE or DISCARD

## Rules

1. **CONTRADICTIONS**: If an existing fact contradicts a new fact or the user's latest
   statement, DEPRECATE the existing fact. The newer information is more accurate.
   - Example: existing "housing.city: Denver" + new "identity.address: Miami" + user says
     "she lives in Miami" → DEPRECATE housing.city

2. **SUPERSEDED BY SAME CONCEPT**: If a new fact covers the same concept as an existing
   fact but under a different key, DEPRECATE the old one.
   - Example: existing "preference.massage_type: nuru" + new "preference.service_type:
     muscle recovery" where user said "he wants muscle recovery instead of nuru"
     → DEPRECATE preference.massage_type

3. **DUPLICATES**: If two facts (existing or new) say the same thing under different keys,
   keep the one with the more canonical/specific key and DEPRECATE/DISCARD the other.
   - Example: "housing.city: Chicago" and "housing.location: Chicago" → KEEP housing.city,
     DEPRECATE housing.location

4. **TEMPORAL VALUES**: DEPRECATE existing facts or DISCARD new facts whose values are
   relative timestamps that will become meaningless.
   - Bad: "move_date: last week", "start_date: next month", "appointment: tomorrow"
   - Good: "move_date: 2026-02-25", "age: 78" (absolute values are fine)

5. **CONVERSATION ARTIFACTS**: DISCARD new facts that are conversation metadata, not
   actual attributes of the entity.
   - Bad: "information_request: care schedule", "clarification_needed: what issues",
     "insurance.status: incorrect on file"
   - These describe the CONVERSATION, not the PERSON

6. **NULL/NEGATIVE VALUES**: DISCARD new facts or DEPRECATE existing facts with values
   like "none", "N/A", "unknown", "not specified", or empty strings.

7. **CONFIRMED FACTS**: If a new fact has the same key AND same value as an existing fact,
   DISCARD the new one (it's already stored). The existing one is KEPT.

8. **NO CHANGES NEEDED**: If existing facts don't conflict with anything and aren't covered
   by rules 1-7, KEEP them. Don't deprecate facts just because they weren't mentioned.

## Output Format (strict JSON)

{{
  "deprecate_existing": [
    {{"fact_key": "housing.city", "reason": "Contradicts confirmed Miami location"}},
    {{"fact_key": "housing.move_date", "reason": "Relative timestamp, not permanent"}}
  ],
  "discard_new": [
    {{"fact_key": "preference.information_request", "reason": "Conversation artifact"}},
    {{"fact_key": "insurance.plan_type", "reason": "Already stored with same value"}}
  ],
  "write_new": [
    {{"fact_key": "identity.address", "new_value": "Miami", "reason": "User confirmed"}}
  ]
}}

IMPORTANT:
- Only include facts that need action. Omit facts that should just be kept as-is.
- "deprecate_existing" only lists EXISTING fact keys to remove.
- "discard_new" only lists NEWLY EXTRACTED facts to NOT write.
- "write_new" lists new facts that SHOULD be written (after filtering out discards).
- Be conservative with deprecation — only deprecate when there's clear evidence of
  contradiction, staleness, or duplication. Don't deprecate a fact just because it
  wasn't mentioned in the current conversation.

## Example 1: Location correction

Existing: housing.city=Denver, housing.location=Denver, identity.address=Miami, housing.state=Florida
New: housing.state=Florida
Conversation: "she lives in Miami, Florida"

Output:
{{
  "deprecate_existing": [
    {{"fact_key": "housing.city", "reason": "Contradicts confirmed Florida/Miami location"}},
    {{"fact_key": "housing.location", "reason": "Contradicts confirmed Florida/Miami location"}}
  ],
  "discard_new": [
    {{"fact_key": "housing.state", "reason": "Already stored with same value"}}
  ],
  "write_new": []
}}

## Example 2: Service preference change

Existing: preference.massage_type=nuru, preference.therapist_gender=female
New: preference.service_type=muscle recovery therapy, preference.therapist_gender=male
Conversation: "he wants muscle recovery therapy now, and prefers a male therapist"

Output:
{{
  "deprecate_existing": [
    {{"fact_key": "preference.massage_type", "reason": "User changed preference to muscle recovery"}},
    {{"fact_key": "preference.therapist_gender", "reason": "User changed preference to male"}}
  ],
  "discard_new": [],
  "write_new": [
    {{"fact_key": "preference.service_type", "new_value": "muscle recovery therapy", "reason": "New preference"}},
    {{"fact_key": "preference.therapist_gender", "new_value": "male", "reason": "Updated preference"}}
  ]
}}

## Example 3: Conversation artifacts

Existing: care_schedule.care_type=medical assistance
New: preference.information_request=["care schedule","health status"], insurance.clarification_needed="what issues"
Conversation: "tell me about his care schedule and health status"

Output:
{{
  "deprecate_existing": [],
  "discard_new": [
    {{"fact_key": "preference.information_request", "reason": "User's question, not entity attribute"}},
    {{"fact_key": "insurance.clarification_needed", "reason": "Conversation metadata, not entity attribute"}}
  ],
  "write_new": []
}}
"""


async def llm_reconcile_facts(
    *,
    existing_profile_facts: Dict[str, Any],
    resolved_facts: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    client: Any,
) -> Optional[Dict[str, Any]]:
    """
    Call the LLM to reconcile newly extracted facts against existing profile facts.

    Returns a dict with keys: deprecate_existing, discard_new, write_new.
    Returns None on any failure (caller should skip reconciliation).
    """
    # Build existing facts block
    existing_lines = []
    for key, info in existing_profile_facts.items():
        val = info.get("value", info) if isinstance(info, dict) else info
        existing_lines.append(f"  {key} = {val}")
    existing_facts_block = "\n".join(existing_lines) if existing_lines else "(none)"

    # Build new facts block
    new_lines = []
    for fact in resolved_facts:
        key = fact.get("fact_key", "")
        val = fact.get("value", "")
        new_lines.append(f"  {key} = {val}")
    new_facts_block = "\n".join(new_lines) if new_lines else "(none)"

    # Build conversation block (last 10 turns)
    recent = conversation_history[-10:] if conversation_history else []
    conv_lines = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            conv_lines.append(f"  {role}: {content[:300]}")
    conversation_block = "\n".join(conv_lines) if conv_lines else "(no conversation context)"

    prompt = RECONCILE_FACTS_PROMPT.format(
        existing_facts_block=existing_facts_block,
        new_facts_block=new_facts_block,
        conversation_block=conversation_block,
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1,
        )

        response_text = response.strip() if isinstance(response, str) else str(response).strip()

        # Extract JSON object from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
            if isinstance(result, dict):
                # Validate expected keys exist (default to empty lists)
                result.setdefault("deprecate_existing", [])
                result.setdefault("discard_new", [])
                result.setdefault("write_new", [])
                return result

        logger.warning("LLM reconciliation returned non-dict JSON")
        return None

    except Exception as e:
        logger.warning(f"LLM fact reconciliation failed: {e}")
        return None


async def llm_emotional_support(
    *,
    user_message: str,
    conversation_history: List[Dict[str, Any]],
    known_facts: Dict[str, Any],
    client: Any,
) -> str:
    """
    LLM-based emotional support response.

    Returns:
        Natural language empathetic response string.
    """
    prompt = make_emotional_support_prompt(
        user_message=user_message,
        conversation_history=conversation_history,
        known_facts=known_facts,
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=400,
            temperature=0.7,
        )
        return response.strip()
    except Exception as e:
        logger.warning(f"LLM emotional support failed: {e}. Falling back to default.")
        return None


# ─────────────────────────────────────────────────────────────
# Known-fact summarization + contextual question generation
# ─────────────────────────────────────────────────────────────


def select_relevant_facts(
    profile_facts: Dict[str, Any],
    request_type: str,
    plan_questions: List[str],
) -> Dict[str, Any]:
    """
    Select the subset of stored facts relevant to the current request.

    Uses two signals:
      1. Namespace matching from request_type taxonomy
      2. Keyword overlap between fact keys/values and plan questions
    """
    from context_bundle import _get_needed_fact_keys

    if not profile_facts:
        return {}

    relevant: Dict[str, Any] = {}

    # Signal 1: namespace match from taxonomy
    needed_ns = _get_needed_fact_keys(request_type) if request_type else []
    for key, info in profile_facts.items():
        ns = key.split(".")[0] if "." in key else key
        if any(key.startswith(n) for n in needed_ns) or ns in needed_ns:
            relevant[key] = info

    # Signal 2: keyword overlap with plan questions
    q_tokens = set()
    for q in plan_questions:
        for w in q.lower().replace("?", " ").replace(",", " ").split():
            if len(w) > 2:
                q_tokens.add(w)

    for key, info in profile_facts.items():
        if key in relevant:
            continue
        tokens = set()
        for part in key.replace(".", " ").replace("_", " ").split():
            if len(part) > 2:
                tokens.add(part.lower())
        val = str(info.get("value", ""))
        for w in val.lower().replace(",", " ").split():
            if len(w) > 2:
                tokens.add(w)
        if len(q_tokens & tokens) >= 1:
            relevant[key] = info

    return relevant


SUMMARIZE_AND_ASK_PROMPT = """\
You are a caregiver assistant starting a new conversation with a returning user. \
You already have some information about their situation from previous conversations.

## User's Current Request
{user_request}

## Request Goal
{request_goal}

## Stored Facts (from previous conversations)
{facts_block}

## Planned Questions (from collection plan)
{questions_block}

## Your Task
Write a single, natural conversational message that:

1. **Summarizes what you already know** in a warm, concise way. Group related facts \
naturally (don't list raw keys). For example: "I remember that your mom is 78, lives \
in Chicago, needs help with bathing and meals, and has Medicaid coverage."

2. **Asks the user to confirm** this is still correct.

3. **Asks follow-up questions** that build on the known context. These should be:
   - Informed by what you already know (don't re-ask known info)
   - Relevant to the current request goal
   - Natural and conversational
   - 2-4 questions max

{lang_instruction}

## Output (strict JSON):
{{
  "summary_text": "Natural summary of known facts (1-3 sentences)",
  "follow_up_questions": ["Question 1?", "Question 2?"],
  "combined_message": "The full message to send to the user (summary + confirmation + questions)"
}}"""


async def llm_summarize_and_ask(
    *,
    user_request: str,
    request_goal: str,
    profile_facts: Dict[str, Any],
    plan_questions: List[str],
    nice_to_have: List[str],
    client: Any,
    lang: str = "en",
) -> Optional[Dict[str, Any]]:
    """
    Single LLM call: summarize known facts + generate contextual follow-up questions.

    Returns dict with summary_text, follow_up_questions, combined_message.
    Returns None on failure.
    """
    # Format facts
    f_lines = []
    for key, info in profile_facts.items():
        val = info.get("value", "")
        if isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False)
        f_lines.append(f"  - {key}: {val}")
    facts_block = "\n".join(f_lines) if f_lines else "(no stored facts)"

    # Format planned questions
    q_lines = [f"  {i+1}. {q}" for i, q in enumerate(plan_questions)]
    if nice_to_have:
        q_lines.append("  (Nice to have:)")
        q_lines.extend([f"  - {q}" for q in nice_to_have])
    questions_block = "\n".join(q_lines) if q_lines else "(none)"

    lang_instruction = (
        "Write everything in Chinese (中文)." if lang == "zh"
        else "Write everything in English."
    )

    prompt = SUMMARIZE_AND_ASK_PROMPT.format(
        user_request=user_request,
        request_goal=request_goal,
        facts_block=facts_block,
        questions_block=questions_block,
        lang_instruction=lang_instruction,
    )

    try:
        response = await client.async_chat(
            prompt=prompt,
            max_tokens=800,
            temperature=0.3,
        )

        response_text = response.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            result = json.loads(response_text[json_start:json_end])
        else:
            result = json.loads(response_text)

        if "combined_message" not in result:
            logger.warning("LLM summarize_and_ask missing combined_message")
            return None

        logger.info(
            f"Summarize-and-ask: {len(profile_facts)} facts → "
            f"{len(result.get('follow_up_questions', []))} follow-up questions"
        )
        return result

    except Exception as e:
        logger.warning(f"LLM summarize_and_ask failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Fact-based question pre-filtering
# ─────────────────────────────────────────────────────────────

def rule_match_facts_to_questions(
    questions: List[str],
    profile_facts: Dict[str, Any],
    excluded_keys: Optional[List[str]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Rule-based first pass: for each question, find stored facts that
    might answer it using BM25 search over the FactKey registry.

    Only returns facts that actually exist in profile_facts (i.e., the
    user has a stored value for that key).

    Args:
        questions: List of question strings from collection plan
        profile_facts: Dict of {fact_key: {value, confidence, ...}} from DDB
        excluded_keys: Fact keys to skip (e.g., disputed facts)

    Returns:
        Dict mapping question index → list of candidate facts with values
    """
    from key_resolver import get_key_registry

    if not profile_facts or not questions:
        return {}

    excluded = set(excluded_keys or [])
    registry = get_key_registry()
    matches: Dict[int, List[Dict[str, Any]]] = {}

    # Build a keyword index from actual stored fact keys and values
    # so we can match even when stored keys differ from registry keys
    fact_keywords: Dict[str, List[str]] = {}  # fact_key → list of searchable tokens
    for key, info in profile_facts.items():
        if key in excluded:
            continue
        tokens = set()
        # Tokenize the key: "finance.out_of_pocket.monthly_estimate" → {finance, out, pocket, monthly, estimate}
        for part in key.replace(".", " ").replace("_", " ").split():
            if len(part) > 2:
                tokens.add(part.lower())
        # Tokenize the value
        val = info.get("value", "")
        val_str = str(val) if not isinstance(val, str) else val
        for word in val_str.replace(",", " ").replace(".", " ").split():
            if len(word) > 2:
                tokens.add(word.lower())
        fact_keywords[key] = list(tokens)

    for idx, question in enumerate(questions):
        hits = []
        seen_keys = set()

        # Pass 1: BM25 registry search → intersect with stored facts
        candidates = registry.search_bm25(question, k=8)
        for candidate in candidates:
            key = candidate.key
            if key in excluded or key in seen_keys:
                continue
            if key in profile_facts:
                seen_keys.add(key)
                fact_info = profile_facts[key]
                hits.append({
                    "fact_key": key,
                    "value": fact_info.get("value", ""),
                    "confidence": fact_info.get("confidence", 0),
                    "verification_level": fact_info.get("verification_level", "unverified"),
                    "bm25_score": candidate.score,
                })

        # Pass 2: Direct keyword overlap with stored fact keys/values
        q_tokens = set()
        for word in question.lower().replace("?", " ").replace(",", " ").split():
            if len(word) > 2:
                q_tokens.add(word)

        for key, tokens in fact_keywords.items():
            if key in seen_keys:
                continue
            overlap = q_tokens & set(tokens)
            if len(overlap) >= 2 or (len(overlap) == 1 and len(tokens) <= 3):
                seen_keys.add(key)
                fact_info = profile_facts[key]
                hits.append({
                    "fact_key": key,
                    "value": fact_info.get("value", ""),
                    "confidence": fact_info.get("confidence", 0),
                    "verification_level": fact_info.get("verification_level", "unverified"),
                    "bm25_score": 1.0,  # synthetic score for keyword match
                })

        if hits:
            matches[idx] = hits

    return matches


FACT_MATCH_CONFIRM_PROMPT = """\
You are a fact-matching assistant for a caregiver support platform. \
Given a list of questions and candidate facts from a user's stored profile, \
determine which questions are already answered by the stored facts.

## Questions to check:
{questions_block}

## Stored facts from user profile:
{facts_block}

## Rules:
- "fully_answered": The stored fact completely answers the question. No need to ask.
- "partially_answered": The fact has SOME relevant info but not everything the question asks. \
Rewrite the question to ask only for the MISSING part.
- "not_answered": The fact doesn't meaningfully answer this question.

Be CONSERVATIVE: only mark "fully_answered" when the fact clearly and completely addresses \
what the question asks. When in doubt, mark "not_answered".

Do NOT count facts with verification_level "unverified" and risk_level "high" as answers — \
these need user confirmation regardless.

{lang_instruction}

## Output (strict JSON, no other text):
{{
  "matches": [
    {{
      "question_index": 0,
      "status": "fully_answered|partially_answered|not_answered",
      "fact_keys_used": ["fact.key"],
      "answer_summary": "Brief summary of the known answer (or null)",
      "rewritten_question": "Narrowed question if partially_answered (or null)"
    }}
  ]
}}"""


async def llm_confirm_fact_matches(
    questions: List[str],
    candidate_matches: Dict[int, List[Dict[str, Any]]],
    profile_facts: Dict[str, Any],
    client: Any,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    LLM pass to confirm which candidate fact matches actually answer the questions.

    Returns list of match results per question that had candidates.
    """
    # Build questions block (only questions with candidates)
    q_lines = []
    for idx in sorted(candidate_matches.keys()):
        q_lines.append(f"  [{idx}] {questions[idx]}")
    questions_block = "\n".join(q_lines)

    # Build facts block
    f_lines = []
    seen_keys = set()
    for idx in sorted(candidate_matches.keys()):
        for fact in candidate_matches[idx]:
            key = fact["fact_key"]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            val = fact["value"]
            conf = fact["confidence"]
            ver = fact["verification_level"]
            f_lines.append(f"  - {key}: {val} (confidence={conf}, verification={ver})")
    facts_block = "\n".join(f_lines)

    lang_instruction = ""
    if lang == "zh":
        lang_instruction = "Write answer_summary and rewritten_question in Chinese (中文)."
    else:
        lang_instruction = "Write answer_summary and rewritten_question in English."

    prompt = FACT_MATCH_CONFIRM_PROMPT.format(
        questions_block=questions_block,
        facts_block=facts_block,
        lang_instruction=lang_instruction,
    )

    logger.info(f"Fact match LLM input — questions:\n{questions_block}")
    logger.info(f"Fact match LLM input — candidate facts:\n{facts_block}")

    response = await client.async_chat(
        prompt=prompt,
        max_tokens=800,
        temperature=0.0,
    )

    response_text = response.strip()
    logger.info(f"Fact match LLM response: {response_text[:500]}")
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1

    if json_start >= 0 and json_end > json_start:
        result = json.loads(response_text[json_start:json_end])
    else:
        result = json.loads(response_text)

    return result.get("matches", [])


async def filter_questions_with_known_facts(
    *,
    questions: List[str],
    nice_to_have: List[str],
    profile_facts: Dict[str, Any],
    request_type: str,
    entity_id: str,
    client: Any,
    lang: str = "en",
    excluded_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Orchestrator: rule-based candidate matching → LLM confirmation → result.

    Returns dict with:
        pre_answered: list of {question, fact_keys_used, answer_summary}
        remaining_questions: questions not fully answered
        remaining_nice_to_have: filtered nice-to-have
        confirmation_text: formatted "Here's what I know" block (or empty)
    """
    empty_result = {
        "pre_answered": [],
        "remaining_questions": questions,
        "remaining_nice_to_have": nice_to_have,
        "confirmation_text": "",
    }

    if not profile_facts or not questions:
        return empty_result

    # Step 1: Rule-based candidate matching
    candidates = rule_match_facts_to_questions(
        questions=questions,
        profile_facts=profile_facts,
        excluded_keys=excluded_keys,
    )

    if not any(candidates.values()):
        logger.debug("No fact candidates found for any question — skipping LLM confirmation")
        return empty_result

    logger.info(
        f"Fact pre-filter: found candidates for {len(candidates)}/{len(questions)} questions, "
        f"calling LLM to confirm"
    )

    # Step 2: LLM confirmation
    llm_matches = await llm_confirm_fact_matches(
        questions=questions,
        candidate_matches=candidates,
        profile_facts=profile_facts,
        client=client,
        lang=lang,
    )

    # Step 3: Build result
    pre_answered = []
    remaining = []
    answered_indices = set()

    for match in llm_matches:
        idx = match.get("question_index")
        if idx is None or idx >= len(questions):
            continue
        status = match.get("status", "not_answered")

        if status == "fully_answered":
            pre_answered.append({
                "question": questions[idx],
                "fact_keys_used": match.get("fact_keys_used", []),
                "answer_summary": match.get("answer_summary", ""),
            })
            answered_indices.add(idx)
        elif status == "partially_answered" and match.get("rewritten_question"):
            remaining.append(match["rewritten_question"])
            answered_indices.add(idx)

    # Add questions that weren't in candidates (or were not_answered)
    for idx, q in enumerate(questions):
        if idx not in answered_indices:
            remaining.append(q)

    # Also filter nice_to_have with the same rule-based pass (no extra LLM call)
    remaining_nice = []
    if nice_to_have:
        nth_candidates = rule_match_facts_to_questions(
            questions=nice_to_have,
            profile_facts=profile_facts,
            excluded_keys=excluded_keys,
        )
        for idx, q in enumerate(nice_to_have):
            # Only drop if there's a high-confidence match in profile_facts
            hits = nth_candidates.get(idx, [])
            has_strong_match = any(
                h.get("bm25_score", 0) > 5.0 and h.get("confidence", 0) >= 0.8
                for h in hits
            )
            if not has_strong_match:
                remaining_nice.append(q)
    else:
        remaining_nice = nice_to_have

    # Build confirmation text
    confirmation_text = ""
    if pre_answered:
        if lang == "zh":
            lines = ["根据我已有的信息："]
            for pa in pre_answered:
                lines.append(f"  - {pa['answer_summary']}")
            lines.append("\n以上信息还正确吗？如果有变化请告诉我。")
            confirmation_text = "\n".join(lines)
        else:
            lines = ["Based on what I already know:"]
            for pa in pre_answered:
                lines.append(f"  - {pa['answer_summary']}")
            lines.append("\nIs this still correct? Let me know if anything has changed.")
            confirmation_text = "\n".join(lines)

    logger.info(
        f"Fact pre-filter result: {len(pre_answered)} pre-answered, "
        f"{len(remaining)} remaining questions"
    )

    return {
        "pre_answered": pre_answered,
        "remaining_questions": remaining,
        "remaining_nice_to_have": remaining_nice,
        "confirmation_text": confirmation_text,
    }
