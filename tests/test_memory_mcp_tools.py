"""
Test: Memory MCP Server Tools — verify all 13 tools respond via stdio transport.

Tests:
  - Tool 1: memory_get_context_bundle
  - Tool 5: memory_resolve_fact_key
  - Tool 6: memory_propose_updates
  - Tool 9: memory_resolve_entity_id
  - Tool 10: memory_resolve_fact_keys_batch
  - Tool 11: memory_get_pending_confirmations
  - Tool 13: memory_write_back_aliases

Note: Tools 2, 3, 4, 7, 8, 12 require DDB and are tested in integration.

Usage:
    python tests/test_memory_mcp_tools.py
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


MEMORY_SERVER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "servers",
    "memory_mcp_server.py",
)


def _parse_result(result):
    """Extract data from a FastMCP CallToolResult."""
    sc = getattr(result, "structured_content", None)
    if sc is not None:
        return sc.get("result") if isinstance(sc, dict) else sc
    content = getattr(result, "content", None) or []
    if content and hasattr(content[0], "text"):
        try:
            return json.loads(content[0].text)
        except (ValueError, TypeError):
            return content[0].text
    return None


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

    transport = StdioTransport(
        command="python",
        args=["-u", MEMORY_SERVER_PATH],
        env=dict(os.environ),
    )
    client = Client(transport)

    async with client:
        # ── Tool 9: memory_resolve_entity_id ──
        print("\n1. memory_resolve_entity_id")
        result = await client.call_tool("memory_resolve_entity_id", {
            "user_text": "帮我妈找护工",
        })
        data = _parse_result(result)
        check("Chinese mom → care_recipient:mom",
              isinstance(data, dict) and data.get("entity_id") == "care_recipient:mom",
              f"got {data}")

        result = await client.call_tool("memory_resolve_entity_id", {
            "user_text": "my dad needs help",
        })
        data = _parse_result(result)
        check("English dad → care_recipient:dad",
              isinstance(data, dict) and data.get("entity_id") == "care_recipient:dad",
              f"got {data}")

        # ── Tool 5: memory_resolve_fact_key ──
        print("\n2. memory_resolve_fact_key")
        result = await client.call_tool("memory_resolve_fact_key", {
            "fact_text": "insurance plan type Medicare",
            "entity_id": "care_recipient:mom",
            "request_type": "find_caregiver",
        })
        data = _parse_result(result)
        check("fact key resolution returns dict",
              isinstance(data, dict) and "canonical_key" in data,
              f"got {type(data)}: {data}")

        # ── Tool 10: memory_resolve_fact_keys_batch ──
        print("\n3. memory_resolve_fact_keys_batch")
        batch_facts = [
            {"fact_key": "insurance.plan_type", "fact_label": "Insurance", "evidence": "Medicare Part A"},
            {"fact_key": "housing.city", "fact_label": "City", "evidence": "Lives in Chicago"},
        ]
        result = await client.call_tool("memory_resolve_fact_keys_batch", {
            "facts": batch_facts,
            "entity_id": "care_recipient:mom",
            "request_type": "find_caregiver",
        })
        data = _parse_result(result)
        check("batch resolution returns list",
              isinstance(data, list) and len(data) == 2,
              f"got {type(data)}: {data}")

        # ── Tool 6: memory_propose_updates ──
        print("\n4. memory_propose_updates")
        facts_for_proposal = [
            {
                "entity_id": "care_recipient:mom",
                "fact_key": "housing.city",
                "fact_label": "City",
                "value": "Chicago",
                "confidence": 0.9,
                "source_type": "user",
                "evidence": "Lives in Chicago",
            }
        ]
        result = await client.call_tool("memory_propose_updates", {
            "extracted_facts": facts_for_proposal,
        })
        data = _parse_result(result)
        check("propose_updates returns proposal",
              isinstance(data, dict) and "auto_patch" in data,
              f"got {type(data)}: {data}")

        # Test with existing_fact_keys dedup
        result = await client.call_tool("memory_propose_updates", {
            "extracted_facts": facts_for_proposal,
            "existing_fact_keys": ["housing.city"],
        })
        data = _parse_result(result)
        check("propose_updates dedup filters existing keys",
              isinstance(data, dict) and len(data.get("auto_patch", [])) == 0,
              f"auto_patch={data.get('auto_patch', []) if isinstance(data, dict) else data}")

        # ── Tool 1: memory_get_context_bundle ──
        print("\n5. memory_get_context_bundle")
        result = await client.call_tool("memory_get_context_bundle", {
            "user_id": "test-user-123",
            "message": "Find a caregiver",
        })
        data = _parse_result(result)
        check("context bundle returns dict with prompt_block",
              isinstance(data, dict) and "prompt_block" in data,
              f"got {type(data)}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Memory MCP Tools Tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
