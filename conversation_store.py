"""
Conversation Store — DynamoDB CRUD for UserConversationTable.

Persists raw conversation messages so they survive pod restarts and are
available for the daily cron job (fact validation, request hygiene).

DDB schema:
  PK: CONV#{conversation_id}
  SK: MSG#{timestamp_iso}#{message_id}

GSI1 (query by user + date):
  gsi1pk: USER#{user_id}#DATE#{YYYY-MM-DD}
  gsi1sk: MSG#{timestamp_iso}#{message_id}
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Key

from id_utils import new_uuid
from ddb_client import USER_CONVERSATION_TABLE, get_ddb_resource

logger = logging.getLogger(__name__)


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
    """Convert Python types to DynamoDB-safe types."""
    cleaned = {}
    for k, v in data.items():
        converted = _serialize_value(v)
        if converted is None:
            continue
        cleaned[k] = converted
    return cleaned


class ConversationStore:
    """CRUD operations for the UserConversationTable."""

    def __init__(self, dynamodb_client: Optional[Any] = None):
        self.dynamodb = dynamodb_client

    def _table(self):
        return self.dynamodb.Table(USER_CONVERSATION_TABLE)

    async def write_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist a single message to DynamoDB.

        Non-blocking on failure — logs a warning and returns.
        """
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, skipping message write")
            return

        try:
            now = datetime.utcnow()
            message_id = new_uuid("msg")
            ts_iso = now.isoformat()
            date_str = now.strftime("%Y-%m-%d")

            item = _serialize_item({
                "pk": f"CONV#{conversation_id}",
                "sk": f"MSG#{ts_iso}#{message_id}",
                "entity": "message",
                "gsi1pk": f"USER#{user_id}#DATE#{date_str}",
                "gsi1sk": f"MSG#{ts_iso}#{message_id}",
                "conversation_id": conversation_id,
                "user_id": user_id,
                "message_id": message_id,
                "role": role,
                "content": content,
                "timestamp": ts_iso,
                "metadata": metadata or {},
            })

            table = self._table()
            table.put_item(Item=item)
            logger.debug(f"Stored message: {message_id} ({role})")

        except Exception as e:
            logger.warning(f"Failed to store message (non-fatal): {e}")

    async def get_messages_for_conversation(
        self,
        conversation_id: str,
        limit: int = 500,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all messages for a conversation, ordered by timestamp.

        If user_id is provided, only returns messages belonging to that user.
        Messages without a user_id field are included (backward compat).
        """
        if not self.dynamodb:
            return []

        try:
            table = self._table()
            response = table.query(
                KeyConditionExpression=(
                    Key("pk").eq(f"CONV#{conversation_id}")
                    & Key("sk").begins_with("MSG#")
                ),
                ScanIndexForward=True,
                Limit=limit,
            )

            messages = []
            for item in response.get("Items", []):
                # Verify ownership if user_id is provided
                if user_id:
                    item_owner = item.get("user_id", "")
                    if item_owner and item_owner != user_id:
                        continue
                messages.append({
                    k: v for k, v in item.items()
                    if k not in ("pk", "sk", "entity", "gsi1pk", "gsi1sk")
                })
            return messages

        except Exception as e:
            logger.error(f"Failed to get messages for conversation: {e}")
            return []

    async def write_checkpoint(
        self,
        conversation_id: str,
        user_id: str,
        checkpoint_data: Dict[str, Any],
    ) -> None:
        """Write or overwrite the checkpoint row for a conversation.

        Uses the same table and PK partition as messages. SK is
        CHECKPOINT#latest so it never collides with MSG# rows and is
        excluded by get_messages_for_conversation (which filters
        SK begins_with "MSG#").
        """
        if not self.dynamodb:
            return

        try:
            now = datetime.utcnow().isoformat()
            item = _serialize_item({
                "pk": f"CONV#{conversation_id}",
                "sk": "CHECKPOINT#latest",
                "entity": "checkpoint",
                "conversation_id": conversation_id,
                "user_id": user_id,
                "updated_at": now,
                **checkpoint_data,
            })
            table = self._table()
            table.put_item(Item=item)
            logger.debug(f"Wrote checkpoint for conv {conversation_id[:8]}")
        except Exception as e:
            logger.warning(f"Failed to write checkpoint (non-fatal): {e}")

    async def get_checkpoint(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read the checkpoint row for a conversation. Returns None if not found."""
        if not self.dynamodb:
            return None

        try:
            table = self._table()
            response = table.get_item(
                Key={
                    "pk": f"CONV#{conversation_id}",
                    "sk": "CHECKPOINT#latest",
                }
            )
            item = response.get("Item")
            if not item:
                return None
            return {
                k: v for k, v in item.items()
                if k not in ("pk", "sk", "entity", "gsi1pk", "gsi1sk")
            }
        except Exception as e:
            logger.error(f"Failed to get checkpoint: {e}")
            return None

    async def get_messages_for_user_date(
        self,
        user_id: str,
        date_str: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all messages for a user on a specific date via GSI1.

        Args:
            user_id: The user ID.
            date_str: Date in "YYYY-MM-DD" format.

        Returns:
            List of message dicts, ordered by timestamp.
        """
        if not self.dynamodb:
            return []

        try:
            table = self._table()
            response = table.query(
                IndexName="GSI1",
                KeyConditionExpression=(
                    Key("gsi1pk").eq(f"USER#{user_id}#DATE#{date_str}")
                    & Key("gsi1sk").begins_with("MSG#")
                ),
                ScanIndexForward=True,
            )

            messages = []
            for item in response.get("Items", []):
                messages.append({
                    k: v for k, v in item.items()
                    if k not in ("pk", "sk", "entity", "gsi1pk", "gsi1sk")
                })
            return messages

        except Exception as e:
            logger.error(f"Failed to get messages for user date: {e}")
            return []

    async def get_active_users_for_date(
        self,
        date_str: str,
    ) -> List[str]:
        """
        Get all user IDs who had conversations on a specific date.

        Scans GSI1 for all gsi1pk values matching USER#*#DATE#{date_str}
        and extracts unique user IDs.
        """
        if not self.dynamodb:
            return []

        try:
            from boto3.dynamodb.conditions import Attr

            table = self._table()
            response = table.scan(
                IndexName="GSI1",
                FilterExpression=Attr("gsi1pk").contains(f"#DATE#{date_str}"),
                ProjectionExpression="gsi1pk",
            )

            users = set()
            for item in response.get("Items", []):
                gsi1pk = item.get("gsi1pk", "")
                # gsi1pk format: USER#{user_id}#DATE#{YYYY-MM-DD}
                if gsi1pk.startswith("USER#") and f"#DATE#{date_str}" in gsi1pk:
                    user_id = gsi1pk.split("#DATE#")[0].replace("USER#", "", 1)
                    if user_id:
                        users.add(user_id)

            return sorted(users)

        except Exception as e:
            logger.error(f"Failed to get active users for date: {e}")
            return []


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

_conversation_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get or create the global ConversationStore instance."""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore(dynamodb_client=get_ddb_resource())
    return _conversation_store
