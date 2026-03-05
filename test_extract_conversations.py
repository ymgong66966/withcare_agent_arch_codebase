"""
测试 _extract_related_conversations 功能

这个测试验证能否正确从全天对话中提取与特定request相关的对话片段
"""

from datetime import datetime, timedelta
from daily_analyzer import DailyRequestAnalyzer
from anthropic_client import TrackedAnthropicClient
from dotenv import load_dotenv
load_dotenv()

def test_extract_related_conversations():
    """测试提取相关对话的功能"""

    print("=" * 70)
    print("Testing: Extract Related Conversations")
    print("=" * 70)

    # 创建analyzer
    client = TrackedAnthropicClient(
        session_id="test-session",
        agent_role="test",
        user_id="test-user",
    )
    analyzer = DailyRequestAnalyzer(client=client)

    # 模拟一天的对话（包含两个不同的requests）
    base_time = datetime(2026, 1, 28, 10, 0, 0)

    conversations = [
        # Request 1: 找护工 (10:00 - 10:10)
        {
            "role": "user",
            "content": "我需要找一个会说中文的护工",
            "timestamp": base_time,
        },
        {
            "role": "assistant",
            "content": "好的，我需要了解一些信息：预算、时间、地点...",
            "timestamp": base_time + timedelta(minutes=1),
            "agent": "info_collection",
        },
        {
            "role": "user",
            "content": "预算2000-3000，芝加哥，尽快",
            "timestamp": base_time + timedelta(minutes=5),
        },
        {
            "role": "assistant",
            "content": "收到，我现在帮你搜索...",
            "timestamp": base_time + timedelta(minutes=6),
            "agent": "deep_search",
        },
        # 用户发现prerequisite (11:00)
        {
            "role": "user",
            "content": "等等，我发现我还没申请Medicaid",
            "timestamp": base_time + timedelta(hours=1),
        },
        # Request 2: Medicaid申请 (11:01 - 11:30)
        {
            "role": "assistant",
            "content": "好的，我们先处理Medicaid申请...",
            "timestamp": base_time + timedelta(hours=1, minutes=1),
            "agent": "info_collection",
        },
        {
            "role": "user",
            "content": "我住在伊利诺伊州，收入大概3000一个月",
            "timestamp": base_time + timedelta(hours=1, minutes=5),
        },
        {
            "role": "assistant",
            "content": "好的，让我帮你查一下伊利诺伊州的Medicaid申请流程",
            "timestamp": base_time + timedelta(hours=1, minutes=6),
            "agent": "domain_expert",
        },
        # 无关对话 (12:00)
        {
            "role": "user",
            "content": "今天天气真好",
            "timestamp": base_time + timedelta(hours=2),
        },
        {
            "role": "assistant",
            "content": "是的，天气不错。还需要我帮忙处理什么吗？",
            "timestamp": base_time + timedelta(hours=2, minutes=1),
            "agent": "front_end_emotional_support",
        },
        # 继续Request 1 (12:30)
        {
            "role": "user",
            "content": "Medicaid申请好了，现在回去找护工吧",
            "timestamp": base_time + timedelta(hours=2, minutes=30),
        },
        {
            "role": "assistant",
            "content": "好的，继续帮你搜索护工...",
            "timestamp": base_time + timedelta(hours=2, minutes=31),
            "agent": "deep_search",
        },
    ]

    # 模拟Request 1: 找护工
    request1 = {
        "request_id": "req-001",
        "name": "Find Chinese-speaking Caregiver",
        "goal": "Find a caregiver who speaks Chinese in Chicago with budget $2000-3000",
        "created_at": base_time,
        "last_touched_at": base_time + timedelta(hours=2, minutes=31),
        "stage_history": [
            {
                "from_stage": None,
                "to_stage": "info_collection",
                "agent": "info_collection",
                "timestamp": base_time,
                "reason": "New request created"
            },
            {
                "from_stage": "info_collection",
                "to_stage": "deep_search",
                "agent": "deep_search",
                "timestamp": base_time + timedelta(minutes=6),
                "reason": "Info collection complete"
            },
            {
                "from_stage": "executing",
                "to_stage": "paused",
                "agent": "upstream_delegator",
                "timestamp": base_time + timedelta(hours=1),
                "reason": "Paused for prerequisite"
            },
            {
                "from_stage": "paused",
                "to_stage": "deep_search",
                "agent": "deep_search",
                "timestamp": base_time + timedelta(hours=2, minutes=31),
                "reason": "Resumed after prerequisite"
            },
        ],
    }

    # 模拟Request 2: Medicaid申请
    request2 = {
        "request_id": "req-002",
        "name": "Medicaid Application",
        "goal": "Help user apply for Medicaid in Illinois",
        "created_at": base_time + timedelta(hours=1, minutes=1),
        "last_touched_at": base_time + timedelta(hours=1, minutes=30),
        "stage_history": [
            {
                "from_stage": None,
                "to_stage": "info_collection",
                "agent": "info_collection",
                "timestamp": base_time + timedelta(hours=1, minutes=1),
                "reason": "Prerequisite request created"
            },
            {
                "from_stage": "info_collection",
                "to_stage": "domain_expert",
                "agent": "domain_expert",
                "timestamp": base_time + timedelta(hours=1, minutes=6),
                "reason": "Info collection complete"
            },
        ],
    }

    # 测试提取Request 1的相关对话
    print("\n" + "-" * 70)
    print("Test 1: Extract conversations for Request 1 (找护工)")
    print("-" * 70)

    related1 = analyzer._extract_related_conversations(conversations, request1)

    print(f"\n✓ Found {len(related1)} related messages (out of {len(conversations)} total)")
    print("\nExtracted conversations:")
    for i, msg in enumerate(related1, 1):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")[:50] + "..."
        agent = msg.get("agent", "N/A")
        timestamp = msg.get("timestamp", "Unknown time")
        print(f"  {i}. [{timestamp.strftime('%H:%M')}] {role} ({agent}): {content}")

    # 验证：应该包含的关键消息
    expected_contents = [
        "我需要找一个会说中文的护工",
        "好的，我需要了解一些信息",
        "预算2000-3000",
        "收到，我现在帮你搜索",
        "等等，我发现我还没申请Medicaid",  # 触发pause的消息
        "Medicaid申请好了",
        "好的，继续帮你搜索护工",
    ]

    found = []
    for content_snippet in expected_contents:
        for msg in related1:
            if content_snippet in msg.get("content", ""):
                found.append(content_snippet)
                break

    print(f"\n✓ Found {len(found)}/{len(expected_contents)} expected messages:")
    for snippet in found:
        print(f"  ✓ '{snippet[:40]}...'")

    missing = [s for s in expected_contents if s not in found]
    if missing:
        print("\n⚠️  Missing expected messages:")
        for snippet in missing:
            print(f"  ✗ '{snippet[:40]}...'")

    # 测试提取Request 2的相关对话
    print("\n" + "-" * 70)
    print("Test 2: Extract conversations for Request 2 (Medicaid申请)")
    print("-" * 70)

    related2 = analyzer._extract_related_conversations(conversations, request2)

    print(f"\n✓ Found {len(related2)} related messages (out of {len(conversations)} total)")
    print("\nExtracted conversations:")
    for i, msg in enumerate(related2, 1):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")[:50] + "..."
        agent = msg.get("agent", "N/A")
        timestamp = msg.get("timestamp", "Unknown time")
        print(f"  {i}. [{timestamp.strftime('%H:%M')}] {role} ({agent}): {content}")

    # 验证：应该包含Medicaid相关的消息
    expected_contents_2 = [
        "好的，我们先处理Medicaid申请",
        "我住在伊利诺伊州",
        "让我帮你查一下伊利诺伊州的Medicaid",
    ]

    found2 = []
    for content_snippet in expected_contents_2:
        for msg in related2:
            if content_snippet in msg.get("content", ""):
                found2.append(content_snippet)
                break

    print(f"\n✓ Found {len(found2)}/{len(expected_contents_2)} expected messages:")
    for snippet in found2:
        print(f"  ✓ '{snippet[:40]}...'")

    # 验证：不应该包含无关的消息
    print("\n" + "-" * 70)
    print("Verification: Check for false positives")
    print("-" * 70)

    irrelevant_content = "今天天气真好"
    has_irrelevant_in_req1 = any(irrelevant_content in msg.get("content", "") for msg in related1)
    has_irrelevant_in_req2 = any(irrelevant_content in msg.get("content", "") for msg in related2)

    if not has_irrelevant_in_req1 and not has_irrelevant_in_req2:
        print(f"✓ Correctly excluded irrelevant message: '{irrelevant_content}'")
    else:
        print(f"⚠️  Incorrectly included irrelevant message: '{irrelevant_content}'")

    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_extract_related_conversations()
