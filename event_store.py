"""
Event Store — DynamoDB CRUD for UserEventTable (short-term episodic memory).

Events represent dialogue summaries, tool results, decisions, errors, and
memory candidates. They have TTL for automatic cleanup.

The dynamodb_client is a boto3.resource('dynamodb') instance (high-level API).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Key, Attr

from id_utils import new_uuid
from models.fact_models import EventRecord, EventType
from ddb_client import USER_EVENT_TABLE, get_ddb_resource

logger = logging.getLogger(__name__)

# Default TTL durations by event type
DEFAULT_TTL_DAYS: Dict[str, int] = {
    "dialogue_summary": 30,
    "tool_result": 30,
    "escalation": 90,
    "decision": 90,
    "memory_candidate": 14,  # Short-lived: confirm or discard
    "error": 30,
}


def _serialize_value(v: Any) -> Any:
    """Convert a single Python value to a DynamoDB-safe type."""
    if isinstance(v, datetime):
        return v.isoformat()
    elif isinstance(v, float):
        return Decimal(str(v))
    elif isinstance(v, dict):
        return _serialize_item(v)
    elif isinstance(v, list):
        return [_serialize_value(item) for item in v]
    elif v is None:
        return None
    else:
        return v


def _serialize_item(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Python types to DynamoDB-safe types (handles datetime, float, etc.)."""
    cleaned = {}
    for k, v in data.items():
        converted = _serialize_value(v)
        if converted is None:
            continue
        cleaned[k] = converted
    return cleaned


def _deserialize_event(item: Dict[str, Any]) -> EventRecord:
    """Convert a DynamoDB item dict back to an EventRecord."""
    data = {k: v for k, v in item.items() if k not in ("pk", "sk", "entity")}
    return EventRecord(**data)


class EventStore:
    """CRUD operations for the UserEventTable."""

    def __init__(self, dynamodb_client: Optional[Any] = None):
        self.dynamodb = dynamodb_client

    def _table(self):
        return self.dynamodb.Table(USER_EVENT_TABLE)

    async def add_event(
        self,
        user_id: str,
        event_type: EventType,
        content: str,
        *,
        request_id: Optional[str] = None,
        care_recipient_id: Optional[str] = None,
        structured: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        ttl_days: Optional[int] = None,
    ) -> EventRecord:
        """
        Write an episodic event to the UserEventTable.

        TTL is automatically calculated based on event_type unless overridden.
        """
        import time as _time
        now = datetime.utcnow()
        days = ttl_days if ttl_days is not None else DEFAULT_TTL_DAYS.get(event_type, 30)
        ttl_epoch = int(_time.time()) + days * 86400

        event = EventRecord(
            event_id=new_uuid("evt"),
            user_id=user_id,
            timestamp=now,
            event_type=event_type,
            request_id=request_id,
            care_recipient_id=care_recipient_id,
            content=content,
            structured=structured or {},
            tags=tags or [],
            ttl_epoch=ttl_epoch,
        )

        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, skipping event write")
            return event

        try:
            raw = event.model_dump(mode="json")
            item = _serialize_item({
                "pk": event.dynamo_pk(),
                "sk": event.dynamo_sk(),
                "entity": "event",
                **raw,
            })

            table = self._table()
            table.put_item(Item=item)
            logger.debug(f"Stored event: {event.event_id} ({event.event_type})")

        except Exception as e:
            logger.error(f"Failed to store event: {e}")

        return event

    async def get_recent_events(
        self,
        user_id: str,
        limit: int = 10,
        event_types: Optional[List[str]] = None,
    ) -> List[EventRecord]:
        """
        Get recent events for a user, ordered by timestamp descending.

        Optionally filter by event_type.
        """
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, returning empty events")
            return []

        try:
            table = self._table()
            pk = f"USER#{user_id}"

            query_kwargs = {
                "KeyConditionExpression": Key("pk").eq(pk) & Key("sk").begins_with("EVT#"),
                "ScanIndexForward": False,  # Most recent first
                "Limit": limit,
            }

            if event_types:
                query_kwargs["FilterExpression"] = Attr("event_type").is_in(event_types)

            response = table.query(**query_kwargs)

            events = []
            for item in response.get("Items", []):
                try:
                    events.append(_deserialize_event(item))
                except Exception as e:
                    logger.warning(f"Failed to deserialize event: {e}")

            return events

        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []

    async def get_events_for_request(
        self,
        user_id: str,
        request_id: str,
        limit: int = 20,
    ) -> List[EventRecord]:
        """Get all events associated with a specific request."""
        if not self.dynamodb:
            return []

        try:
            table = self._table()
            pk = f"USER#{user_id}"

            response = table.query(
                KeyConditionExpression=Key("pk").eq(pk) & Key("sk").begins_with("EVT#"),
                FilterExpression=Attr("request_id").eq(request_id),
                ScanIndexForward=False,
                Limit=limit,
            )

            events = []
            for item in response.get("Items", []):
                try:
                    events.append(_deserialize_event(item))
                except Exception as e:
                    logger.warning(f"Failed to deserialize event: {e}")

            return events

        except Exception as e:
            logger.error(f"Failed to get events for request: {e}")
            return []

    async def get_memory_candidates(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[EventRecord]:
        """Get pending memory_candidate events (facts awaiting confirmation)."""
        return await self.get_recent_events(
            user_id=user_id,
            limit=limit,
            event_types=["memory_candidate"],
        )


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

_event_store: Optional[EventStore] = None


def get_event_store() -> EventStore:
    """Get or create the global EventStore instance."""
    global _event_store
    if _event_store is None:
        _event_store = EventStore(dynamodb_client=get_ddb_resource())
    return _event_store
