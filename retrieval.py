from __future__ import annotations

from typing import Any, Dict, List

from historical_models import HistoricalRequestRecord, VectorSearchResult
from historical_store import HistoricalRequestStore
from ddb_client import get_ddb_resource

# These are *interfaces* that you can later back with:
# - Milvus / Zilliz semantic search
# - DynamoDB request history
# - Snowflake user profile snapshots
# - Any vector DB with (user_id, request_text) embeddings

# Global store instance (initialized lazily)
_historical_store: HistoricalRequestStore = None


def get_historical_store() -> HistoricalRequestStore:
    """Get or create the global HistoricalRequestStore instance"""
    global _historical_store
    if _historical_store is None:
        _historical_store = HistoricalRequestStore(
            dynamodb_client=get_ddb_resource(),
            # TODO: Initialize when Milvus/Zilliz is provisioned (Sprint 4)
            milvus_client=None,
            embedding_client=None,
        )
    return _historical_store


async def fetch_similar_requests(
    *,
    user_id: str,
    request_text: str,
    top_k: int = 5,
    filters: Dict[str, Any] = None,
) -> List[HistoricalRequestRecord]:
    """
    ✅ 返回统一的HistoricalRequestRecord格式

    Performs vector search in Milvus/Zilliz to find similar past requests.

    Args:
        user_id: User ID
        request_text: Query text (new request description)
        top_k: Number of similar requests to return
        filters: Optional filters (e.g., {"completion_status": ["completed"]})

    Returns:
        List[HistoricalRequestRecord] - Similar past requests
    """
    store = get_historical_store()

    # Vector search
    search_results: List[VectorSearchResult] = await store.search_similar_requests(
        user_id=user_id,
        query_text=request_text,
        top_k=top_k,
        filters=filters,
    )

    # Extract HistoricalRequestRecords
    return [result.historical_record for result in search_results]


async def fetch_prior_qas_for_questions(
    *,
    user_id: str,
    questions: List[str],
    top_k: int = 5
) -> List[Dict[str, str]]:
    """
    Fetch prior Q&A pairs from user history

    This is a more specific search - for each question, find similar questions
    that were asked before and their answers.

    Args:
        user_id: User ID
        questions: List of questions to search for
        top_k: Number of prior Q&As to return per question

    Returns:
        List of Q&A dicts: [{"question": "...", "answer": "...", "source": "..."}]

    TODO: Implement semantic search for questions
    Example sources:
    - daily_summary:2026-01-01
    - request:req-123
    - profile_update:2026-01-15
    """
    # TODO: implement per-question semantic retrieval from user history/profile
    return []


async def get_requests_by_theme(
    user_id: str,
    theme: str,
    top_k: int = 10,
) -> List[HistoricalRequestRecord]:
    """
    Get all historical requests for a specific theme

    Args:
        user_id: User ID
        theme: Theme to filter by (e.g., "medicaid_application", "in_home_care")
        top_k: Maximum number of results

    Returns:
        List[HistoricalRequestRecord] filtered by theme
    """
    return await fetch_similar_requests(
        user_id=user_id,
        request_text=theme,  # Use theme as query
        top_k=top_k,
        filters={"theme": [theme]},
    )


async def get_recent_requests(
    user_id: str,
    days: int = 7,
    top_k: int = 10,
) -> List[HistoricalRequestRecord]:
    """
    Get recent requests from the last N days

    Args:
        user_id: User ID
        days: Number of days to look back
        top_k: Maximum number of results

    Returns:
        List[HistoricalRequestRecord] from recent days
    """
    from datetime import datetime, timedelta

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # TODO: Query by date range
    # For now, just return empty
    return []
