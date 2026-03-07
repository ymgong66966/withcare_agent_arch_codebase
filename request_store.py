"""
Request Store — read-only DynamoDB access layer for UserRequestTable queries.

Agent-side version: imports from the agent's own ddb_client.py.
Used by the stdio MCP server (servers/memory_mcp_server.py).

Provides 4 access patterns:
  - query_by_status:  GSI1 (USER#{id}#STATUS#{status})
  - query_by_entity:  GSI2 (RECIPIENT#{entity_id})
  - get_request:      Main table PK/SK lookup
  - query_recent:     Main table query + date filter
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Key, Attr

from ddb_client import USER_REQUEST_TABLE, get_ddb_resource

logger = logging.getLogger(__name__)

# GSI index names (must match table definition)
GSI1_NAME = "GSI1"
GSI2_NAME = "GSI2"

# Fields returned in summary view
_SUMMARY_FIELDS = {
    "request_id", "title", "goal", "status", "request_type",
    "subject_entity_id", "priority", "created_at", "last_touched_at",
    "summary_current", "stage_detail", "name", "target",
}

# Additional fields for detail view
_DETAIL_EXTRA_FIELDS = {
    "slots", "open_questions", "artifacts", "payload",
    "source", "conversation_id", "variant",
    "prereq_gate", "deep_search_state", "status_flags", "audit",
}


def _format_request_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    """Strip DDB keys, return summary-level fields."""
    result = {}
    for key in _SUMMARY_FIELDS:
        if key in item:
            val = item[key]
            if isinstance(val, Decimal):
                val = float(val) if val % 1 else int(val)
            result[key] = val
    return result


def _format_request_detail(item: Dict[str, Any]) -> Dict[str, Any]:
    """Full payload including slots, open_questions, artifacts."""
    result = _format_request_summary(item)
    for key in _DETAIL_EXTRA_FIELDS:
        if key in item:
            result[key] = item[key]
    return result


class RequestStore:
    """Read-only query operations for the UserRequestTable."""

    def __init__(self, dynamodb_client: Optional[Any] = None):
        self.dynamodb = dynamodb_client

    def _table(self):
        return self.dynamodb.Table(USER_REQUEST_TABLE)

    async def query_by_status(
        self,
        user_id: str,
        status: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Query requests by status via GSI1."""
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, returning empty")
            return []

        try:
            table = self._table()
            gsi1pk = f"USER#{user_id}#STATUS#{status}"

            response = table.query(
                IndexName=GSI1_NAME,
                KeyConditionExpression=Key("gsi1pk").eq(gsi1pk),
                ScanIndexForward=False,
                Limit=limit,
            )

            return [_format_request_summary(item) for item in response.get("Items", [])]

        except Exception as e:
            logger.error(f"query_by_status failed: {e}")
            return []

    async def query_by_entity(
        self,
        entity_id: str,
        limit: int = 20,
        after_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query requests by care recipient entity via GSI2."""
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, returning empty")
            return []

        try:
            table = self._table()
            gsi2pk = f"RECIPIENT#{entity_id}"

            key_cond = Key("gsi2pk").eq(gsi2pk)
            if after_date:
                key_cond = key_cond & Key("gsi2sk").gte(f"LAST#{after_date}")

            response = table.query(
                IndexName=GSI2_NAME,
                KeyConditionExpression=key_cond,
                ScanIndexForward=False,
                Limit=limit,
            )

            return [_format_request_summary(item) for item in response.get("Items", [])]

        except Exception as e:
            logger.error(f"query_by_entity failed: {e}")
            return []

    async def get_request(
        self,
        user_id: str,
        request_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Direct PK/SK lookup for a single request with full detail."""
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, returning None")
            return None

        try:
            table = self._table()
            response = table.get_item(
                Key={
                    "pk": f"USER#{user_id}",
                    "sk": f"REQ#{request_id}",
                }
            )

            item = response.get("Item")
            if not item:
                return None
            return _format_request_detail(item)

        except Exception as e:
            logger.error(f"get_request failed: {e}")
            return None

    async def query_recent(
        self,
        user_id: str,
        limit: int = 20,
        after_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query recent requests, optionally filtered by creation date.

        Fetches all REQ# items for the user, sorts by created_at descending
        in Python, then applies the limit. This avoids DDB's Limit being
        applied before FilterExpression and ensures true chronological ordering
        (the main table SK is UUID-based, not time-based).
        """
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, returning empty")
            return []

        try:
            table = self._table()
            pk = f"USER#{user_id}"

            query_kwargs: Dict[str, Any] = {
                "KeyConditionExpression": Key("pk").eq(pk) & Key("sk").begins_with("REQ#"),
            }

            if after_date:
                query_kwargs["FilterExpression"] = Attr("created_at").gte(after_date)

            response = table.query(**query_kwargs)
            items = response.get("Items", [])

            # Paginate if DDB returned a partial result set
            while "LastEvaluatedKey" in response:
                query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
                response = table.query(**query_kwargs)
                items.extend(response.get("Items", []))

            # Sort by created_at descending (true chronological recency)
            items.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            return [_format_request_summary(item) for item in items[:limit]]

        except Exception as e:
            logger.error(f"query_recent failed: {e}")
            return []

    async def get_requests_by_ids(
        self,
        user_id: str,
        request_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple requests by ID and return a dict keyed by request_id.

        Returns raw DDB items including the ``payload`` field so callers can
        reconstruct full RequestRecord objects.
        """
        if not self.dynamodb or not request_ids:
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        try:
            table = self._table()
            for rid in request_ids:
                response = table.get_item(
                    Key={
                        "pk": f"USER#{user_id}",
                        "sk": f"REQ#{rid}",
                    }
                )
                item = response.get("Item")
                if item:
                    result[rid] = item
        except Exception as e:
            logger.error(f"get_requests_by_ids failed: {e}")

        return result


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

_request_store: Optional[RequestStore] = None


def get_request_store() -> RequestStore:
    """Get or create the global RequestStore instance."""
    global _request_store
    if _request_store is None:
        _request_store = RequestStore(dynamodb_client=get_ddb_resource())
    return _request_store
