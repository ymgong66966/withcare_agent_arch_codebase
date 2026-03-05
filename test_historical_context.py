"""
测试 Historical Context Integration

验证info_collection能正确使用similar requests作为context
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List

from prompts import llm_collection_plan, make_collection_plan_prompt
from anthropic_client import TrackedAnthropicClient
from historical_models import HistoricalRequestRecord
from dotenv import load_dotenv

load_dotenv()


async def test_collection_plan_with_historical_context():
    """测试带historical context的collection plan"""

    print("=" * 70)
    print("Testing: Collection Plan with Historical Context")
    print("=" * 70)

    # 创建client
    client = TrackedAnthropicClient(
        session_id="test-session",
        agent_role="info_collection",
        user_id="test-user",
    )

    # 模拟similar requests（从历史中检索到的）
    similar_requests = [
        {
            "request_id": "req-001",
            "name": "Find Chinese-speaking Caregiver",
            "goal": "Find a caregiver who speaks Chinese in Chicago with budget $2000-3000",
            "completion_status": "completed",
            "short_summary": "User successfully found a caregiver through our search. Very satisfied with the match.",
            "theme": "in_home_care",
            "user_satisfaction": "satisfied",
            "keywords": ["caregiver", "chinese", "chicago", "in-home"],
            "created_at": datetime(2026, 1, 15),
        },
        {
            "request_id": "req-002",
            "name": "Medicaid Application Illinois",
            "goal": "Help user apply for Medicaid in Illinois",
            "completion_status": "pending",
            "short_summary": "Started Medicaid application process, still waiting for eligibility determination",
            "theme": "medicaid_application",
            "user_satisfaction": "neutral",
            "keywords": ["medicaid", "illinois", "application"],
            "created_at": datetime(2026, 1, 20),
        },
    ]

    # 用户的新请求（类似之前的请求）
    user_request = "我需要找一个移民律师，会说中文的，在芝加哥附近"

    # Known facts
    known_facts = {
        "caregiver": {
            "location": "Chicago, IL",
            "language": "English, Chinese",
        },
        "care_recipient": {
            "age": 78,
            "condition": "mild dementia",
        }
    }

    print("\n" + "-" * 70)
    print("Step 1: Test prompt formatting")
    print("-" * 70)

    # 测试prompt格式化
    prompt = make_collection_plan_prompt(
        user_request=user_request,
        known_facts=known_facts,
        similar_requests=similar_requests,
    )

    print("\n📝 Generated Prompt (excerpt):\n")
    # 打印similar requests部分
    if "Similar Past Requests" in prompt:
        start = prompt.index("Similar Past Requests")
        end = prompt.index("## Your Task:", start) if "## Your Task:" in prompt[start:] else len(prompt)
        print(prompt[start:end])
    else:
        print("⚠️  No similar requests section found in prompt")

    print("\n" + "-" * 70)
    print("Step 2: Test LLM collection plan generation")
    print("-" * 70)

    try:
        # 调用LLM生成collection plan
        plan = await llm_collection_plan(
            user_request=user_request,
            known_facts=known_facts,
            similar_requests=similar_requests,
            client=client,
        )

        print("\n✅ Collection plan generated successfully!")
        print(f"\n📋 Plan Details:")
        print(f"  Request Name: {plan.get('request_name')}")
        print(f"  Request Goal: {plan.get('request_goal')}")
        print(f"  Routing Hint: {plan.get('routing_hint')}")

        print(f"\n❓ Key Info Needed ({len(plan.get('key_info_needed', []))}):")
        for i, question in enumerate(plan.get('key_info_needed', []), 1):
            print(f"  {i}. {question}")

        print(f"\n💡 Nice to Have ({len(plan.get('nice_to_have_info', []))}):")
        for i, question in enumerate(plan.get('nice_to_have_info', []), 1):
            print(f"  {i}. {question}")

        prereqs = plan.get('potential_prerequisites', [])
        if prereqs:
            print(f"\n⚠️  Potential Prerequisites ({len(prereqs)}):")
            for prereq in prereqs:
                print(f"  - {prereq.get('type')}: {prereq.get('reason')}")
        else:
            print("\n✓ No prerequisites detected")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "-" * 70)
    print("Step 3: Test without historical context (comparison)")
    print("-" * 70)

    try:
        # 不使用historical context
        plan_no_context = await llm_collection_plan(
            user_request=user_request,
            known_facts=known_facts,
            similar_requests=None,  # 没有历史
            client=client,
        )

        print("\n✅ Plan without context generated successfully!")
        print(f"  Request Name: {plan_no_context.get('request_name')}")
        print(f"  Key Info Needed: {len(plan_no_context.get('key_info_needed', []))} questions")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70)


async def test_empty_history():
    """测试没有历史时的行为"""

    print("\n\n" + "=" * 70)
    print("Testing: Collection Plan with Empty History")
    print("=" * 70)

    client = TrackedAnthropicClient(
        session_id="test-session-2",
        agent_role="info_collection",
        user_id="new-user",
    )

    user_request = "我需要帮助申请Medicare"
    known_facts = {"caregiver": {}, "care_recipient": {"age": 82}}

    # 空的similar requests
    similar_requests = []

    print("\nUser Request: " + user_request)
    print("Similar Requests: [] (empty)")

    try:
        plan = await llm_collection_plan(
            user_request=user_request,
            known_facts=known_facts,
            similar_requests=similar_requests,
            client=client,
        )

        print("\n✅ Plan generated successfully even without history!")
        print(f"  Request Name: {plan.get('request_name')}")
        print(f"  Routing Hint: {plan.get('routing_hint')}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n🧪 Historical Context Integration Test\n")

    # Test 1: With historical context
    asyncio.run(test_collection_plan_with_historical_context())

    # Test 2: Without historical context (new user)
    asyncio.run(test_empty_history())

    print("\n✅ All tests completed!\n")
