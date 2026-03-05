"""
Pydantic models for the User Memory Framework.

Covers: FactRecord, AliasRecord, FactLogEntry, EventRecord,
        ContextBundle (and sub-blocks), WriteProposal, KeyResolverResult.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Fact Store
# ─────────────────────────────────────────────────────────────

FactStatus = Literal["active", "deprecated", "candidate"]
RiskLevel = Literal["low", "medium", "high"]
SourceType = Literal["user", "tool", "document", "agent_inference"]
VerificationLevel = Literal[
    "explicit_user_confirmed", "tool_verified", "unverified"
]


class FactRecord(BaseModel):
    """A single versioned fact in the UserFactTable."""

    fact_id: str
    user_id: str
    entity_id: str  # e.g. "care_recipient:mom", "user:self"
    fact_key: str  # Canonical key from registry: "insurance.member_id"
    fact_label: str = ""  # Free-text label for display

    # Value
    fact_value: Any = None
    value_type: str = "string"  # string | enum | code | date | list | object | number

    # Provenance & trust
    status: FactStatus = "candidate"
    risk_level: RiskLevel = "medium"
    confidence: float = 0.0
    source_type: SourceType = "agent_inference"
    source_ref: str = ""  # request_id / tool_call_id / doc_id
    evidence: str = ""  # Short evidence snippet (<=200 chars)

    # Verification
    verification_level: VerificationLevel = "unverified"
    needs_reconfirm_after_days: int = 90

    # Versioning
    supersedes_fact_id: Optional[str] = None
    schema_version: int = 1

    # Timestamps
    first_seen_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen_at: datetime = Field(default_factory=datetime.utcnow)
    last_verified_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def dynamo_pk(self) -> str:
        return f"USER#{self.user_id}#ENT#{self.entity_id}"

    def dynamo_sk(self) -> str:
        ts = self.created_at.isoformat()
        return f"FACT#{self.fact_key}#TS#{ts}#{self.fact_id}"


# ─────────────────────────────────────────────────────────────
# Alias Store
# ─────────────────────────────────────────────────────────────

AliasStatus = Literal["active", "pending", "rejected", "deprecated"]
AliasScope = Literal["global", "user", "locale"]


class AliasRecord(BaseModel):
    """Maps a natural-language alias to a canonical FactKey."""

    normalized_alias: str  # Lowercase, trimmed, punctuation-stripped
    canonical_key: str  # Points to FactKey registry entry
    scope: AliasScope = "global"
    user_id: Optional[str] = None  # When scope="user"
    count: int = 1
    last_seen_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 0.0
    status: AliasStatus = "pending"
    evidence_samples: List[str] = Field(default_factory=list)  # Up to 3

    def dynamo_pk(self) -> str:
        return f"ALIAS#{self.normalized_alias}"

    def dynamo_sk(self) -> str:
        return f"KEY#{self.canonical_key}"


# ─────────────────────────────────────────────────────────────
# Fact Audit Log
# ─────────────────────────────────────────────────────────────

FactLogResult = Literal["applied", "rejected", "needs_confirm"]


class FactLogEntry(BaseModel):
    """Audit trail entry for every fact write/update."""

    log_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    target: Literal["profile", "request", "fact"] = "fact"
    target_id: str  # fact_id / request_id
    patch: Dict[str, Any] = Field(default_factory=dict)
    justification: str = ""
    source_ref: str = ""
    actor: str = ""  # agent_id + model
    result: FactLogResult = "applied"

    def dynamo_pk(self) -> str:
        return f"USER#{self.user_id}"

    def dynamo_sk(self) -> str:
        ts = self.timestamp.isoformat()
        return f"FACTLOG#{ts}#{self.log_id}"


# ─────────────────────────────────────────────────────────────
# Event Store (Episodic Memory)
# ─────────────────────────────────────────────────────────────

EventType = Literal[
    "dialogue_summary",
    "tool_result",
    "escalation",
    "decision",
    "memory_candidate",
    "error",
]


class EventRecord(BaseModel):
    """Short-term episodic event in the UserEventTable."""

    event_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: EventType = "dialogue_summary"
    request_id: Optional[str] = None
    care_recipient_id: Optional[str] = None
    content: str = ""
    structured: Dict[str, Any] = Field(default_factory=dict)
    embedding_ref: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    ttl_epoch: int = 0  # DynamoDB TTL (unix epoch)

    def dynamo_pk(self) -> str:
        return f"USER#{self.user_id}"

    def dynamo_sk(self) -> str:
        ts = self.timestamp.isoformat()
        return f"EVT#{ts}#{self.event_id}"


# ─────────────────────────────────────────────────────────────
# Extracted Fact (intermediate, before write gate)
# ─────────────────────────────────────────────────────────────

class ExtractedFact(BaseModel):
    """
    A candidate fact extracted by LLM, before Write Gate processes it.
    This is the input to propose_updates().
    """

    entity_id: str
    fact_key: str
    fact_label: str = ""
    value: Any = None
    value_type: str = "string"
    confidence: float = 0.0
    source_type: SourceType = "agent_inference"
    source_ref: str = ""
    evidence: str = ""
    explicit_user_confirmed: bool = False
    # LLM may suggest risk, but registry overrides it
    risk_suggestion: RiskLevel = "medium"


# ─────────────────────────────────────────────────────────────
# Write Gate
# ─────────────────────────────────────────────────────────────

class WriteProposal(BaseModel):
    """Output of the Memory Write Gate (propose_updates)."""

    auto_patch: List[ExtractedFact] = Field(default_factory=list)
    needs_confirm: List[ExtractedFact] = Field(default_factory=list)
    reject: List[ExtractedFact] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Key Resolver
# ─────────────────────────────────────────────────────────────

class KeyCandidate(BaseModel):
    """A single candidate from the Key Resolver search."""

    key: str
    description: str = ""
    risk_level: RiskLevel = "medium"
    value_type: str = "string"
    score: float = 0.0


class KeyResolverResult(BaseModel):
    """Output of the Key Resolver."""

    canonical_key: str  # Selected key or "unknown"
    confidence: float = 0.0
    risk_level: RiskLevel = "medium"
    value_candidate: Any = None
    decision: Literal["map", "propose_new", "need_human_confirm", "unknown"] = "map"
    candidates: List[KeyCandidate] = Field(default_factory=list)
    alias_observed: List[str] = Field(default_factory=list)
    alias_suggested: List[str] = Field(default_factory=list)
    reason: str = ""


# ─────────────────────────────────────────────────────────────
# Candidate Key Pool (global, cross-user)
# ─────────────────────────────────────────────────────────────


class CandidateKeyRecord(BaseModel):
    """A proposed fact key from the global CandidateKeyPool (cross-user)."""

    candidate_key: str  # Normalized: "care_schedule.weekday_hours"
    description: str = ""  # LLM-generated: "Weekly care schedule in hours per day"
    risk_suggestion: RiskLevel = "medium"
    occurrence_count: int = 1
    sample_values: List[str] = Field(default_factory=list)  # Up to 5
    sample_entities: List[str] = Field(default_factory=list)  # Up to 5
    source_requests: List[str] = Field(default_factory=list)  # Up to 5
    first_seen_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["candidate", "promoted", "rejected"] = "candidate"


# ─────────────────────────────────────────────────────────────
# Context Bundle (assembled per-turn for LLM calls)
# ─────────────────────────────────────────────────────────────

class SlotWithFactRef(BaseModel):
    """A request slot that may reference a fact in the UserFactTable."""

    status: Literal["filled", "missing", "unverified"] = "missing"
    value: Any = None
    fact_ref: Optional[str] = None  # Points to fact_id in UserFactTable
    risk: RiskLevel = "low"
    verified: bool = False


class ActiveRequestBlock(BaseModel):
    """Active request summary for the Context Bundle."""

    request_id: str
    request_type: str = ""
    title: str = ""
    status: str = ""
    subject_entity_id: str = ""
    slots: Dict[str, SlotWithFactRef] = Field(default_factory=dict)
    next_steps: List[str] = Field(default_factory=list)
    summary_current: str = ""


class RequestSummaryItem(BaseModel):
    """Compact request summary for relevant_history in the Context Bundle."""

    request_id: str
    request_type: str = ""
    title: str = ""
    when: str = ""  # Date string
    status: str = ""
    summary: str = ""
    subject_entity_id: str = ""


class EventItem(BaseModel):
    """Compact event summary for recent_events in the Context Bundle."""

    when: str = ""
    event_type: str = ""
    content: str = ""
    request_id: Optional[str] = None


class SafetyNote(BaseModel):
    """Flag for high-risk facts that need verification."""

    topic: str
    fact_key: str
    current_value: Any = None
    confidence: float = 0.0
    last_verified_at: Optional[str] = None
    rule: str = ""


class ContextBundle(BaseModel):
    """
    Structured context assembled per-turn for LLM calls.

    Replaces raw conversation dumping with a minimal, focused "work desk"
    containing only what the LLM needs for the current interaction.
    """

    user_id: str
    now: datetime = Field(default_factory=datetime.utcnow)

    # Block 1: Active request
    active_request: Optional[ActiveRequestBlock] = None

    # Block 2: Relevant profile facts (only fields relevant to current intent)
    profile_facts: Dict[str, Any] = Field(default_factory=dict)

    # Block 3: Relevant history (recency + relevance merged)
    relevant_history: List[RequestSummaryItem] = Field(default_factory=list)

    # Block 4: Recent events (episodic)
    recent_events: List[EventItem] = Field(default_factory=list)

    # Block 5: Safety notes
    safety_notes: List[SafetyNote] = Field(default_factory=list)
