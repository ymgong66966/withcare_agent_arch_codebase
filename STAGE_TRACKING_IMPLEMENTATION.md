# Stage Transition Tracking - Implementation Summary

## Overview

We've successfully implemented Step 2 of the integration guide: recording stage transitions throughout the request lifecycle. All requests now maintain a `stage_history` field that tracks their progression through different stages.

## Implementation Details

### 1. **info_collection_node** - New Request Creation (graph.py:497-528)

When a new request is created:
```python
initial_transition = {
    "from_stage": None,  # None indicates initial creation
    "to_stage": "info_collection",
    "agent": "info_collection",
    "timestamp": datetime.utcnow(),
    "reason": "New request created from user query"
}
```

**File location**: graph.py:508-514

### 2. **info_collection_node** - Completion & Handoff (graph.py:609-621)

When information collection is complete and ready to proceed:
```python
new_transition = {
    "from_stage": "info_collection",
    "to_stage": routing_hint,  # e.g., "deep_search", "domain_expert"
    "agent": routing_hint,
    "timestamp": datetime.utcnow(),
    "reason": "Information collection complete; proceeding to execution"
}
```

**File location**: graph.py:609-621

### 3. **info_collection_node** - Prerequisite Acceptance (graph.py:423-449)

When user accepts a system-detected prerequisite:
```python
accept_transition = {
    "from_stage": "proposed",
    "to_stage": "info_collection",
    "agent": "info_collection",
    "timestamp": datetime.utcnow(),
    "reason": f"User accepted prerequisite: {prereq_type}"
}
```

**File location**: graph.py:428-437

### 4. **upstream_delegator** - Natural Progression (graph.py:252-277)

When user naturally progresses from one task to another:
```python
new_transition = {
    "from_stage": current_stage,
    "to_stage": recommended_agent,
    "agent": recommended_agent,
    "timestamp": datetime.utcnow(),
    "reason": f"Natural progression: {reason}"
}
```

**File location**: graph.py:258-267

### 5. **upstream_delegator** - Prerequisite Task Creation (graph.py:279-330)

When creating a new prerequisite request:
```python
# For the new prerequisite request
initial_transition = {
    "from_stage": None,
    "to_stage": "info_collection",
    "agent": "info_collection",
    "timestamp": datetime.utcnow(),
    "reason": f"Prerequisite request created: {prereq_type}"
}

# For the parent request being paused
pause_transition = {
    "from_stage": req.get("status", "unknown"),
    "to_stage": "paused",
    "agent": "upstream_delegator",
    "timestamp": datetime.utcnow(),
    "reason": f"Paused for prerequisite request: {prereq_type}"
}
```

**File location**: graph.py:296-311

### 6. **upstream_delegator** - New Unrelated Task (graph.py:332-392)

When creating a new unrelated task:
```python
# For the new request
initial_transition = {
    "from_stage": None,
    "to_stage": "info_collection",
    "agent": "info_collection",
    "timestamp": datetime.utcnow(),
    "reason": f"New unrelated task created: {new_request_info.get('name', 'New Task')}"
}

# For the paused previous request
pause_transition = {
    "from_stage": req.get("status", "unknown"),
    "to_stage": "paused",
    "agent": "upstream_delegator",
    "timestamp": datetime.utcnow(),
    "reason": "Paused due to new unrelated task"
}
```

**File location**: graph.py:348-379

### 7. **upstream_delegator** - Request Completion (graph.py:394-420)

When user explicitly completes a request:
```python
completion_transition = {
    "from_stage": current_stage,
    "to_stage": "completed",
    "agent": "upstream_delegator",
    "timestamp": datetime.utcnow(),
    "reason": "User confirmed completion"
}
```

**File location**: graph.py:404-412

### 8. **deep_search_node** - Search Execution (graph.py:666-694)

When deep search starts executing:
```python
search_transition = {
    "from_stage": "validated",
    "to_stage": "deep_search_executing",
    "agent": "deep_search",
    "timestamp": datetime.utcnow(),
    "reason": "Started deep search execution"
}
```

**File location**: graph.py:680-688

### 9. **domain_expert_node** - Expert Guidance (graph.py:696-732)

When domain expert starts generating guidance:
```python
expert_transition = {
    "from_stage": "validated",
    "to_stage": "domain_expert_executing",
    "agent": "domain_expert",
    "timestamp": datetime.utcnow(),
    "reason": "Started domain expert guidance generation"
}
```

**File location**: graph.py:719-727

### 10. **user_info_node** - User Info Summarization (graph.py:734-757)

When user info agent starts:
```python
userinfo_transition = {
    "from_stage": "validated",
    "to_stage": "user_info_executing",
    "agent": "user_info",
    "timestamp": datetime.utcnow(),
    "reason": "Started user info summarization"
}
```

**File location**: graph.py:746-754

## Data Structure

Each stage transition is stored as a dictionary with the following structure:

```python
{
    "from_stage": Optional[str],  # None for initial creation
    "to_stage": str,              # Target stage
    "agent": str,                 # Agent responsible for this stage
    "timestamp": datetime,        # When the transition occurred
    "reason": str                 # Why the transition happened
}
```

## Storage Location

All stage transitions are stored in:
```python
state["request_manager"]["requests"][request_id]["stage_history"]
```

This is a list that grows chronologically as the request progresses through different stages.

## Example Stage History

Here's what a typical request's stage_history might look like:

```python
[
    {
        "from_stage": None,
        "to_stage": "info_collection",
        "agent": "info_collection",
        "timestamp": datetime(2026, 1, 28, 10, 0, 0),
        "reason": "New request created from user query"
    },
    {
        "from_stage": "info_collection",
        "to_stage": "deep_search",
        "agent": "deep_search",
        "timestamp": datetime(2026, 1, 28, 10, 5, 0),
        "reason": "Information collection complete; proceeding to execution"
    },
    {
        "from_stage": "validated",
        "to_stage": "deep_search_executing",
        "agent": "deep_search",
        "timestamp": datetime(2026, 1, 28, 10, 6, 0),
        "reason": "Started deep search execution"
    },
    {
        "from_stage": "executing",
        "to_stage": "completed",
        "agent": "upstream_delegator",
        "timestamp": datetime(2026, 1, 28, 10, 15, 0),
        "reason": "User confirmed completion"
    }
]
```

## Next Steps

With stage tracking now implemented, the next integration steps are:

1. ✅ **Step 1**: Updated state_models.py to add stage_history field
2. ✅ **Step 2**: Updated graph.py nodes to record stage transitions
3. **Step 3**: Run daily job to analyze and archive requests (see `example_daily_job.py`)
4. **Step 4**: Configure storage backends (DynamoDB, Milvus)
5. **Step 5**: Update info_collection to use historical context from similar requests

## Usage in Daily Analysis

The stage_history will be used by the DailyRequestAnalyzer to:
- Understand the complete lifecycle of each request
- Determine completion status retrospectively
- Identify patterns in user behavior
- Generate accurate summaries for vector search

### Extract Related Conversations

**NEW**: Implemented `_extract_related_conversations()` in `daily_analyzer.py` (line 407-507)

This method extracts conversation chunks relevant to a specific request from the full day's conversations.

**How it works:**

1. **Time Range Filtering**: Uses request's `created_at` and `last_touched_at` with a 5-minute buffer on each side
2. **Agent Matching**: Extracts all agents involved from `stage_history`
3. **Context Expansion**: For each matching assistant message:
   - Includes the preceding user message (trigger)
   - Includes the following user message (response)
4. **Gap Handling**: Maintains message continuity by including user messages between related agent messages

**Example:**

Given a request with stage_history showing `info_collection` → `deep_search` agents:

```python
# Input: Full day conversations (12 messages)
# Request active: 10:00 - 10:30
# Agents: info_collection, deep_search

# Output: Related conversations (6 messages)
[
    {"role": "user", "content": "我需要找护工", "timestamp": "10:00"},
    {"role": "assistant", "content": "好的，请告诉我...", "agent": "info_collection", "timestamp": "10:01"},
    {"role": "user", "content": "预算2000-3000", "timestamp": "10:05"},
    {"role": "assistant", "content": "收到，我现在搜索...", "agent": "deep_search", "timestamp": "10:06"},
    # ... (filtered out unrelated messages at 11:00-12:00)
]
```

**Benefits:**

- Provides focused context for LLM analysis
- Reduces noise from unrelated conversations
- Handles requests that are paused/resumed correctly
- Captures complete conversation flow for each request

**Testing:**

Run `python test_extract_conversations.py` to verify the extraction logic with multiple requests and interleaved conversations.

See `HISTORICAL_REQUESTS_README.md` and `INTEGRATION_GUIDE.md` for more details.
