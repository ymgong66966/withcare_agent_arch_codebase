from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

AgentName = Literal[
    "front_end_emotional_support",
    "info_collection",
    "deep_search",
    "user_info",
    "domain_expert",
    "quick_answer",
    "human_comm",
]
DelegatorName = Literal["delegator_upstream", "delegator_downstream"]
AnyNodeName = Union[AgentName, DelegatorName]

IntentLabel = Literal[
    "chat",
    "support",
    "collect_info",
    "search",
    "expert_guidance",
    "update_profile",
    "unknown",
]
TurnMode = Literal["continuation", "new_intent"]
ConversationStage = Literal[
    "idle",
    "chat",
    "collecting",
    "executing",
    "reviewing",
    "completed",
    "error_recovery",
]
AgentRunStatus = Literal["success", "needs_more_info", "partial", "failed"]
Urgency = Literal["low", "medium", "high"]

TargetEntity = Literal["care_recipient", "caregiver", "both", "unknown"]
Priority = Literal["low", "normal", "high"]
RequestStatus = Literal[
    "created",
    "collecting",
    "validated",
    "ready_for_handoff",
    "executing",
    "paused",
    "blocked",
    "completed",
    "aborted",
]
SlotSource = Literal["user", "profile", "history", "tool", "inferred"]
InconsistencyResolution = Literal["ask_user", "update_profile", "ignore_for_now"]

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "developer", "user", "assistant", "human", "tool"]
    content: str
    message_id: Optional[str] = None
    ts: datetime = Field(default_factory=datetime.utcnow)

class IntentHint(BaseModel):
    label: IntentLabel = "unknown"
    confidence: float = 0.0

class Handoff(BaseModel):
    recommended_next_agent: Optional[AgentName] = None
    reason: str = ""
    urgency: Urgency = "low"
    blocking_gaps: List[str] = Field(default_factory=list)

class ToolFailure(BaseModel):
    ts: datetime = Field(default_factory=datetime.utcnow)
    tool: str
    error: str
    recoverable: bool = True

class ToolRun(BaseModel):
    run_id: str
    tool: str
    purpose: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["ok", "failed"] = "ok"
    error: Optional[str] = None

PrereqGateStatus = Literal["none", "suspected", "proposed", "accepted", "rejected"]

class PrereqGate(BaseModel):
    status: PrereqGateStatus = "none"
    prereq_type: Optional[str] = None
    reason: str = ""
    parent_request_id: Optional[str] = None
    proposed_request_id: Optional[str] = None
    proposed_by: Optional[AgentName] = None
    proposed_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    proposal_message_id: Optional[str] = None

DeepSearchStage = Literal["idle", "clarify", "tooling", "synthesizing", "done", "failed"]

class DeepSearchState(BaseModel):
    stage: DeepSearchStage = "idle"
    required_fields: List[str] = Field(default_factory=list)
    last_tool_failures: List[ToolFailure] = Field(default_factory=list)
    awaiting_user_input: bool = False
    last_followup_question: Optional[str] = None

class Emotion(BaseModel):
    label: Optional[str] = None
    confidence: float = 0.0

class Consent(BaseModel):
    allow_web_search: bool = True
    allow_location_use: bool = True
    allow_store_updates: bool = True

class ProfileSnapshot(BaseModel):
    facts: Dict[str, Any] = Field(default_factory=dict)
    as_of: Optional[datetime] = None

class MemoryBlock(BaseModel):
    daily_summaries_index_ref: Optional[str] = None
    recent_summary: Optional[str] = None
    salient_facts: List[str] = Field(default_factory=list)

class UserContext(BaseModel):
    emotion: Emotion = Field(default_factory=Emotion)
    consent: Consent = Field(default_factory=Consent)
    profile_snapshot: Dict[Literal["caregiver", "care_recipient"], ProfileSnapshot] = Field(
        default_factory=lambda: {"caregiver": ProfileSnapshot(), "care_recipient": ProfileSnapshot()}
    )
    memory: MemoryBlock = Field(default_factory=MemoryBlock)

class Slot(BaseModel):
    key: str
    value: Any = None
    source: SlotSource = "user"
    confidence: float = 0.0
    needs_user_validation: bool = False
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)

class OpenQuestion(BaseModel):
    question: str
    slot_key: str
    why: str = ""
    priority: Literal["high", "med", "low"] = "med"

ArtifactType = Literal["search_results","written_guidance","profile_snapshot","info_collection_summary","error_report"]

class Artifact(BaseModel):
    type: ArtifactType
    artifact_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    produced_by: Optional[AnyNodeName] = None

class RequestRecord(BaseModel):
    request_id: str
    name: str
    goal: str = ""
    target: TargetEntity = "unknown"
    priority: Priority = "normal"
    routing_hint: Optional[str] = None

    # ✅ NEW: Structured request identity (replaces freeform name as sole identifier)
    request_type: str = ""  # Controlled taxonomy: "insurance_renewal", "find_caregiver", etc.
    subject_entity_id: str = ""  # "care_recipient:mom", "user:self"
    variant: Dict[str, str] = Field(default_factory=dict)  # {"plan_type": "medicaid", "state": "IL"}
    title: str = ""  # Display name (LLM-generated, human-readable)

    # ✅ NEW: Request summary (for context bundle and history search)
    summary_current: str = ""  # Updated on each status change

    # ✅ NEW: Slot-to-fact binding (cross-request info reuse)
    slot_refs: Dict[str, str] = Field(default_factory=dict)  # slot_key → fact_id

    # ⚠️ DEPRECATED: 旧版structured collection（新版用info_collection_state）
    collection_plan: Optional[Dict[str, Any]] = None
    slots: List[Slot] = Field(default_factory=list)
    open_questions: List[OpenQuestion] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_touched_at: datetime = Field(default_factory=datetime.utcnow)

    # Status management
    status: RequestStatus = "created"
    stage_detail: Optional[str] = None
    awaiting_user_input: bool = False

    # Prereq relationship: if this request is a prerequisite, who is the parent?
    parent_request_id: Optional[str] = None

    # ✅ NEW: Stage history tracking (for retrospective analysis)
    stage_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Format: [{"from_stage": "...", "to_stage": "...", "agent": "...", "timestamp": "...", "reason": "..."}]

    # Agent-specific states
    info_collection_state: Optional[Dict[str, Any]] = None  # ✅ NEW: Conversational info collection
    prereq_gate: PrereqGate = Field(default_factory=PrereqGate)
    deep_search_state: DeepSearchState = Field(default_factory=DeepSearchState)

    # Artifacts
    artifacts: List[Artifact] = Field(default_factory=list)

class PendingQueueItem(BaseModel):
    request_id: str
    name: str = ""
    status: Literal["pending", "in_progress", "blocked", "done"] = "pending"
    reason_queued: str = ""
    queued_at: datetime = Field(default_factory=datetime.utcnow)
    child_request_id: Optional[str] = None  # The prereq request that caused this parent to be queued

class RequestManager(BaseModel):
    active_request_id: Optional[str] = None
    requests: Dict[str, RequestRecord] = Field(default_factory=dict)
    pending_queue: List[PendingQueueItem] = Field(default_factory=list)

class Routing(BaseModel):
    current_agent: Optional[AgentName] = None
    conversation_stage: ConversationStage = "idle"
    turn_mode: TurnMode = "continuation"
    turn_reason: str = ""
    llm_recommended_agent: Optional[str] = None
    delegator_debug: Optional[Dict[str, Any]] = None
    pending_handoff: Handoff = Field(default_factory=Handoff)
    # DEPRECATED: needs_human is no longer used for routing decisions.
    # Kept for backward compatibility with existing DDB checkpoints.
    # Smart routing now detects human support messages via role="human" in recent messages.
    needs_human: bool = False

class ToolState(BaseModel):
    tool_runs: List[ToolRun] = Field(default_factory=list)
    tool_failures: List[ToolFailure] = Field(default_factory=list)

class Meta(BaseModel):
    conversation_id: str
    user_id: str
    timezone: str = "America/Chicago"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)

class UnifiedState(BaseModel):
    model_config = ConfigDict(extra="allow")
    meta: Meta
    messages: List[ChatMessage] = Field(default_factory=list)
    routing: Routing = Field(default_factory=Routing)
    user_context: UserContext = Field(default_factory=UserContext)
    request_manager: RequestManager = Field(default_factory=RequestManager)
    tools: ToolState = Field(default_factory=ToolState)
    ddb_writes: List[Dict[str, Any]] = Field(default_factory=list)
