"""
Historical Request Tracking Models

这个模块定义了historical request tracking的核心数据模型：
- Retrospective analysis of requests
- Daily summaries
- Vector search support
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class StageTransition(BaseModel):
    """Request在不同stage之间的转换记录"""
    from_stage: Optional[str] = None  # None表示初始创建
    to_stage: str
    agent: str  # 哪个agent负责这个stage
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reason: str = ""  # 为什么转到这个stage

    def __str__(self) -> str:
        if self.from_stage:
            return f"{self.from_stage} → {self.to_stage} ({self.agent})"
        return f"Created at {self.to_stage} ({self.agent})"


CompletionStatus = Literal["completed", "abandoned", "pending", "uncertain"]
UserSatisfaction = Literal["satisfied", "unsatisfied", "neutral", "unknown"]


class HistoricalRequestRecord(BaseModel):
    """
    Historical request - 已经经过retrospective分析的request

    这个是存储到vector DB和DynamoDB的格式，用于：
    1. Semantic search (通过short_summary, theme, keywords)
    2. 检索similar past requests
    3. 理解user的历史需求模式
    """
    # Core identification
    request_id: str
    user_id: str
    conversation_id: str

    # Basic info
    name: str  # Short name
    goal: str  # Detailed goal
    created_at: datetime
    analyzed_at: datetime  # 什么时候做的retrospective分析

    # Retrospective判断（由LLM分析得出）
    completion_status: CompletionStatus
    final_stage: str  # 最后停留在哪个stage
    user_satisfaction: UserSatisfaction

    # Summary for search and context
    short_summary: str  # 1-2句话总结这个request发生了什么
    theme: str  # Request的主题/类型（如"medicaid_application", "legal_advice"）
    keywords: List[str] = Field(default_factory=list)  # 关键词

    # Stage progression history
    stage_history: List[StageTransition] = Field(default_factory=list)

    # Complete snapshot
    final_state_snapshot: Dict[str, Any] = Field(default_factory=dict)  # RequestRecord的完整快照

    # Metadata
    completion_confidence: float = 0.0  # LLM对completion判断的置信度 (0.0-1.0)
    analysis_reason: str = ""  # LLM的分析理由

    # Vector search metadata
    embedding_version: str = "text-embedding-3-small"  # 使用的embedding模型

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req-123",
                "user_id": "user-456",
                "conversation_id": "conv-789",
                "name": "Find In-Home Caregiver",
                "goal": "Find a Chinese-speaking caregiver in Chicago",
                "created_at": "2026-01-28T10:00:00Z",
                "analyzed_at": "2026-01-29T00:00:00Z",
                "completion_status": "completed",
                "final_stage": "deep_search",
                "user_satisfaction": "satisfied",
                "short_summary": "User found a suitable caregiver through our search tool after collecting budget and requirements.",
                "theme": "in_home_care",
                "keywords": ["caregiver", "chinese-speaking", "chicago", "in-home"],
                "stage_history": [
                    {"to_stage": "info_collection", "agent": "info_collection"},
                    {"from_stage": "info_collection", "to_stage": "deep_search", "agent": "deep_search"}
                ],
                "completion_confidence": 0.85,
            }
        }


class RequestSummaryForDay(BaseModel):
    """单个request在某天的总结（daily job产出）"""
    request_id: str
    historical_record: HistoricalRequestRecord

    # 在当天的活动
    turns_in_day: int  # 这个request在当天占了多少轮对话
    stages_visited_today: List[str]  # 当天经过了哪些stages

    # LLM insights
    key_events_today: List[str] = Field(default_factory=list)  # 当天的关键事件
    progress_assessment: str = ""  # LLM评估的进展情况


class DailyRequestSummary(BaseModel):
    """
    某一天的所有requests总结

    由DailyRequestAnalyzer在每天结束时生成
    """
    user_id: str
    date: str  # "2026-01-28"
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    # Request summaries
    request_summaries: List[RequestSummaryForDay] = Field(default_factory=list)

    # Overall day summary
    overall_summary: str = ""  # LLM生成的当天总体总结
    total_requests_active: int = 0
    total_requests_completed_today: int = 0
    total_requests_pending: int = 0

    # Themes
    main_themes_today: List[str] = Field(default_factory=list)

    # Metadata
    total_conversation_turns: int = 0


class RequestCache(BaseModel):
    """
    Request缓存 - 当天所有活跃的requests

    这个是in-memory的working state，每天结束时会被分析并归档
    """
    user_id: str
    date: str  # "2026-01-28"

    # 当天的所有requests（不管是否完成）
    daily_requests: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # request_id -> RequestRecord dict

    # Quick access
    active_request_id: Optional[str] = None
    pending_queue: List[str] = Field(default_factory=list)  # request_ids

    # Tracking
    created_today: List[str] = Field(default_factory=list)  # request_ids
    completed_today: List[str] = Field(default_factory=list)  # request_ids (tentative)

    # Daily summary (populated at end of day)
    daily_summary: Optional[DailyRequestSummary] = None


class VectorSearchResult(BaseModel):
    """Vector search返回的单个结果"""
    historical_record: HistoricalRequestRecord
    similarity_score: float  # Cosine similarity or other metric
    rank: int  # 1-based ranking


class SearchQuery(BaseModel):
    """Vector search query"""
    user_id: str
    query_text: str

    # Filters (optional)
    completion_status_filter: Optional[List[CompletionStatus]] = None
    theme_filter: Optional[List[str]] = None
    date_range: Optional[tuple[datetime, datetime]] = None

    # Search params
    top_k: int = 5
    min_similarity: float = 0.5
