"""
示例：Daily Job - 每天结束时分析和归档requests

这个脚本展示如何使用DailyRequestAnalyzer和HistoricalRequestStore
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List

from anthropic_client import TrackedAnthropicClient
from daily_analyzer import DailyRequestAnalyzer
from historical_store import HistoricalRequestStore
from historical_models import RequestCache


async def run_daily_job(
    user_id: str,
    date: str,
    conversations: List[Dict[str, Any]],
    active_requests: Dict[str, Dict[str, Any]],
):
    """
    运行daily job

    参数：
        user_id: User ID
        date: Date string "YYYY-MM-DD"
        conversations: 当天的所有对话messages
        active_requests: 当天活跃的所有requests
    """
    print(f"\n{'='*60}")
    print(f"Running Daily Job for {user_id} on {date}")
    print(f"{'='*60}\n")

    # 1. Initialize analyzer
    client = TrackedAnthropicClient(
        session_id=f"daily-job-{date}",
        agent_role="daily_analyzer",
        user_id=user_id,
    )
    analyzer = DailyRequestAnalyzer(client=client)

    # 2. Analyze the day
    print("Step 1: Analyzing requests...")
    daily_summary = await analyzer.analyze_daily_requests(
        user_id=user_id,
        date=date,
        conversations=conversations,
        active_requests=active_requests,
        conversation_id=f"conv-{date}",
    )

    print(f"\n📊 Analysis Results:")
    print(f"  Total requests active: {daily_summary.total_requests_active}")
    print(f"  Completed today: {daily_summary.total_requests_completed_today}")
    print(f"  Still pending: {daily_summary.total_requests_pending}")
    print(f"  Main themes: {', '.join(daily_summary.main_themes_today)}")
    print(f"\n  Overall summary:\n  {daily_summary.overall_summary}\n")

    # 3. Show individual request analysis
    print("Step 2: Individual Request Analysis:")
    for i, req_summary in enumerate(daily_summary.request_summaries, 1):
        hr = req_summary.historical_record
        print(f"\n  Request {i}: {hr.name}")
        print(f"    Status: {hr.completion_status}")
        print(f"    Confidence: {hr.completion_confidence:.2f}")
        print(f"    Theme: {hr.theme}")
        print(f"    Summary: {hr.short_summary}")
        print(f"    Keywords: {', '.join(hr.keywords)}")
        print(f"    User satisfaction: {hr.user_satisfaction}")

    # 4. Archive to storage
    print("\nStep 3: Archiving to storage...")
    store = HistoricalRequestStore(
        # TODO: Initialize with real clients
        dynamodb_client=None,
        milvus_client=None,
        embedding_client=None,
    )

    request_cache = RequestCache(
        user_id=user_id,
        date=date,
        daily_requests=active_requests,
        daily_summary=daily_summary,
    )

    await store.archive_day(
        user_id=user_id,
        date=date,
        request_cache=request_cache,
        daily_analysis=daily_summary,
    )

    print("✅ Daily job completed!")
    print(f"\n{'='*60}\n")

    return daily_summary


async def example_usage():
    """示例使用"""

    # Mock data
    user_id = "user-123"
    date = "2026-01-28"

    # 模拟对话
    conversations = [
        {
            "role": "user",
            "content": "我需要找一个会说中文的护工",
            "timestamp": datetime(2026, 1, 28, 10, 0, 0),
        },
        {
            "role": "assistant",
            "content": "好的，我需要了解一些信息：预算、时间、地点...",
            "timestamp": datetime(2026, 1, 28, 10, 1, 0),
            "agent": "info_collection",
        },
        {
            "role": "user",
            "content": "预算2000-3000，芝加哥，尽快",
            "timestamp": datetime(2026, 1, 28, 10, 5, 0),
        },
        {
            "role": "assistant",
            "content": "收到，我现在帮你搜索...",
            "timestamp": datetime(2026, 1, 28, 10, 6, 0),
            "agent": "deep_search",
        },
        {
            "role": "user",
            "content": "等等，我发现我还没申请Medicaid",
            "timestamp": datetime(2026, 1, 28, 11, 0, 0),
        },
        {
            "role": "assistant",
            "content": "好的，我们先处理Medicaid申请...",
            "timestamp": datetime(2026, 1, 28, 11, 1, 0),
            "agent": "info_collection",
        },
    ]

    # 模拟requests
    active_requests = {
        "req-001": {
            "request_id": "req-001",
            "name": "Find In-Home Caregiver",
            "goal": "Find a Chinese-speaking caregiver in Chicago with budget $2000-3000/month",
            "status": "paused",
            "created_at": datetime(2026, 1, 28, 10, 0, 0),
            "last_touched_at": datetime(2026, 1, 28, 11, 0, 0),
            "info_collection_state": {
                "summary_of_collected_info": "- 预算: $2000-3000/月\n- 地点: 芝加哥\n- 时间: 尽快\n- 要求: 会说中文",
                "readiness_to_proceed": "ready",
            },
        },
        "req-002": {
            "request_id": "req-002",
            "name": "Medicaid Application",
            "goal": "Help user apply for Medicaid",
            "status": "collecting",
            "created_at": datetime(2026, 1, 28, 11, 1, 0),
            "last_touched_at": datetime(2026, 1, 28, 11, 1, 0),
            "info_collection_state": {
                "summary_of_collected_info": "",
                "readiness_to_proceed": "needs_more",
            },
        },
    }

    # Run daily job
    try:
        summary = await run_daily_job(
            user_id=user_id,
            date=date,
            conversations=conversations,
            active_requests=active_requests,
        )

        # Show what would be stored
        print("\n📦 What would be stored to Milvus for vector search:")
        for req_summary in summary.request_summaries:
            hr = req_summary.historical_record
            print(f"\n  Request ID: {hr.request_id}")
            print(f"  Embedding text: '{hr.name}. {hr.goal}. {hr.short_summary}'")
            print(f"  Metadata: theme={hr.theme}, status={hr.completion_status}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Daily Job Example - Historical Request Tracking")
    print("="*60)

    asyncio.run(example_usage())
