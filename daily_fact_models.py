"""
Daily Fact Job Models — Pydantic models for fact validation and candidate review reports.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DeprecatedFact(BaseModel):
    """A fact that was deprecated during daily validation."""
    entity_id: str
    fact_key: str
    reason: str


class FlaggedFact(BaseModel):
    """A fact flagged for human review during daily validation."""
    entity_id: str
    fact_key: str
    concern: str
    severity: str = "medium"  # low | medium | high


class AliasMapping(BaseModel):
    """A candidate key mapped to an existing canonical key."""
    candidate_key: str
    canonical_key: str
    suggested_aliases: List[str] = Field(default_factory=list)


class ProposedNewKey(BaseModel):
    """A candidate key proposed as a genuinely new fact key."""
    key: str
    description: str = ""
    risk_level: str = "medium"
    value_type: str = "string"
    namespace: str = ""
    reason: str = ""


class RejectedCandidate(BaseModel):
    """A candidate key rejected during review."""
    candidate_key: str
    reason: str


class DailyFactReport(BaseModel):
    """Summary report from a daily fact validation + candidate review run."""
    user_id: str
    date: str
    run_at: datetime = Field(default_factory=datetime.utcnow)

    # Fact validation results
    facts_deprecated: int = 0
    facts_flagged: int = 0
    deprecated_details: List[DeprecatedFact] = Field(default_factory=list)
    flagged_details: List[FlaggedFact] = Field(default_factory=list)

    # Candidate key review results
    candidates_aliased: int = 0
    candidates_proposed: int = 0
    candidates_rejected: int = 0
    alias_details: List[AliasMapping] = Field(default_factory=list)
    proposed_details: List[ProposedNewKey] = Field(default_factory=list)
    rejected_details: List[RejectedCandidate] = Field(default_factory=list)
