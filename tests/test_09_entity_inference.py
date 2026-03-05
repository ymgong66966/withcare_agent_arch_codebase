"""
Test 09: Entity ID Inference — Chinese patterns, English patterns, LLM fallback.

Tests:
  - Chinese keyword matching (我妈, 父亲, 老公, 奶奶, etc.)
  - English keyword matching (my mom, father, spouse, etc.)
  - LLM fallback for ambiguous text
  - Edge cases (mixed language, no match)

Usage:
    python tests/test_09_entity_inference.py
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import _infer_subject_entity


class MockLLMClient:
    """Mock client for testing LLM fallback path."""

    def __init__(self, response: str = "care_recipient:unknown"):
        self._response = response
        self.call_count = 0

    async def async_chat(self, prompt: str = "", **kwargs) -> str:
        self.call_count += 1
        return self._response


async def run_tests():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} — {detail}")
            failed += 1

    # ── 1. English keyword matching ──────────────────────────────
    print("\n1. English keyword matching")

    result = await _infer_subject_entity("I need to find a caregiver for my mom in Chicago")
    check("'my mom' → care_recipient:mom", result == "care_recipient:mom", f"got {result}")

    result = await _infer_subject_entity("My father needs help with medication")
    check("'father' → care_recipient:dad", result == "care_recipient:dad", f"got {result}")

    result = await _infer_subject_entity("I'm looking for care for my spouse")
    check("'my spouse' → care_recipient:spouse", result == "care_recipient:spouse", f"got {result}")

    result = await _infer_subject_entity("My grandmother needs a walker")
    check("'grandmother' → care_recipient:grandparent", result == "care_recipient:grandparent", f"got {result}")

    result = await _infer_subject_entity("I need help for myself with anxiety")
    check("'myself' → user:self", result == "user:self", f"got {result}")

    # ── 2. Chinese keyword matching ──────────────────────────────
    print("\n2. Chinese keyword matching")

    result = await _infer_subject_entity("帮我妈找护工")
    check("'我妈' → care_recipient:mom", result == "care_recipient:mom", f"got {result}")

    result = await _infer_subject_entity("母亲需要看医生")
    check("'母亲' → care_recipient:mom", result == "care_recipient:mom", f"got {result}")

    result = await _infer_subject_entity("我爸最近身体不太好")
    check("'我爸' → care_recipient:dad", result == "care_recipient:dad", f"got {result}")

    result = await _infer_subject_entity("父亲的药快吃完了")
    check("'父亲' → care_recipient:dad", result == "care_recipient:dad", f"got {result}")

    result = await _infer_subject_entity("老公需要做康复训练")
    check("'老公' → care_recipient:spouse", result == "care_recipient:spouse", f"got {result}")

    result = await _infer_subject_entity("老婆的保险要到期了")
    check("'老婆' → care_recipient:spouse", result == "care_recipient:spouse", f"got {result}")

    result = await _infer_subject_entity("奶奶最近总是忘事")
    check("'奶奶' → care_recipient:grandparent", result == "care_recipient:grandparent", f"got {result}")

    result = await _infer_subject_entity("外婆需要找家政")
    check("'外婆' → care_recipient:grandparent", result == "care_recipient:grandparent", f"got {result}")

    result = await _infer_subject_entity("爷爷住院了")
    check("'爷爷' → care_recipient:grandparent", result == "care_recipient:grandparent", f"got {result}")

    result = await _infer_subject_entity("外公的血压一直偏高")
    check("'外公' → care_recipient:grandparent", result == "care_recipient:grandparent", f"got {result}")

    result = await _infer_subject_entity("配偶需要定期检查")
    check("'配偶' → care_recipient:spouse", result == "care_recipient:spouse", f"got {result}")

    # ── 3. LLM fallback (mock) ───────────────────────────────────
    print("\n3. LLM fallback")

    mock_client = MockLLMClient(response="care_recipient:mom")
    result = await _infer_subject_entity("my neighbor's elderly parent needs help", client=mock_client)
    check(
        "ambiguous text triggers LLM fallback",
        mock_client.call_count == 1,
        f"call_count={mock_client.call_count}",
    )
    check(
        "LLM fallback returns valid result",
        result == "care_recipient:mom",
        f"got {result}",
    )

    # ── 4. No match, no LLM → unknown ───────────────────────────
    print("\n4. No match, no LLM client")

    result = await _infer_subject_entity("someone needs care services")
    check("no match → care_recipient:unknown", result == "care_recipient:unknown", f"got {result}")

    # ── 5. LLM client that fails ─────────────────────────────────
    print("\n5. LLM client failure graceful fallback")

    class FailingClient:
        async def async_chat(self, **kwargs):
            raise RuntimeError("LLM unavailable")

    result = await _infer_subject_entity("some ambiguous request", client=FailingClient())
    check(
        "failing LLM client → care_recipient:unknown",
        result == "care_recipient:unknown",
        f"got {result}",
    )

    # ── 6. Edge cases ────────────────────────────────────────────
    print("\n6. Edge cases")

    result = await _infer_subject_entity("妈妈和爸爸都需要照顾")
    check("both parents → first match (mom)", result == "care_recipient:mom", f"got {result}")

    result = await _infer_subject_entity("")
    check("empty string → unknown", result == "care_recipient:unknown", f"got {result}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Entity Inference Tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
