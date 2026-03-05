"""
Fact Store — DynamoDB CRUD for UserFactTable, FactAliasTable, and MemoryFactLogTable.

Follows the same "graceful None client" pattern as historical_store.py:
when dynamodb_client is None, operations log warnings and return empty results.

The dynamodb_client is a boto3.resource('dynamodb') instance (high-level API).
All DDB calls are synchronous (boto3 resource API) but wrapped in async methods
for consistency with the rest of the codebase.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from boto3.dynamodb.conditions import Key, Attr

from id_utils import new_uuid
from models.fact_models import (
    AliasRecord,
    AliasStatus,
    FactLogEntry,
    FactRecord,
    FactStatus,
    RiskLevel,
)

logger = logging.getLogger(__name__)

from ddb_client import (
    USER_FACT_TABLE,
    FACT_ALIAS_TABLE,
    MEMORY_FACT_LOG_TABLE,
    get_ddb_resource,
)


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


def _deserialize_fact(item: Dict[str, Any]) -> FactRecord:
    """Convert a DynamoDB item dict back to a FactRecord."""
    # Remove DDB key fields that aren't part of the model
    data = {k: v for k, v in item.items() if k not in ("pk", "sk", "entity")}
    return FactRecord(**data)


class FactStore:
    """
    CRUD operations for the three fact-related DynamoDB tables.

    Handles:
    - UserFactTable: entity-centric facts with versioning
    - FactAliasTable: alias → canonical key mappings
    - MemoryFactLogTable: audit trail
    """

    def __init__(self, dynamodb_client: Optional[Any] = None):
        self.dynamodb = dynamodb_client

    def _fact_table(self):
        return self.dynamodb.Table(USER_FACT_TABLE)

    def _alias_table(self):
        return self.dynamodb.Table(FACT_ALIAS_TABLE)

    def _log_table(self):
        return self.dynamodb.Table(MEMORY_FACT_LOG_TABLE)

    # ─────────────────────────────────────────────────────────
    # UserFactTable
    # ─────────────────────────────────────────────────────────

    async def get_active_facts(
        self,
        user_id: str,
        entity_id: str,
        fact_keys: Optional[List[str]] = None,
    ) -> List[FactRecord]:
        """
        Get all active facts for an entity, optionally filtered by fact_keys.

        Uses a query on the main table (PK) with a filter on status=active.
        """
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, returning empty facts")
            return []

        try:
            table = self._fact_table()
            pk = f"USER#{user_id}#ENT#{entity_id}"

            response = table.query(
                KeyConditionExpression=Key("pk").eq(pk),
                FilterExpression=Attr("status").eq("active"),
            )

            facts = []
            for item in response.get("Items", []):
                try:
                    fact = _deserialize_fact(item)
                    if fact_keys and fact.fact_key not in fact_keys:
                        continue
                    facts.append(fact)
                except Exception as e:
                    logger.warning(f"Failed to deserialize fact item: {e}")

            return facts

        except Exception as e:
            logger.error(f"Failed to get active facts: {e}")
            return []

    async def get_user_entities(self, user_id: str) -> List[str]:
        """
        Get all distinct entity_ids that have active facts for a user.

        Scans the fact table for PKs matching USER#{user_id}#ENT#*
        and extracts unique entity_ids.
        """
        if not self.dynamodb:
            return []

        try:
            table = self._fact_table()
            response = table.scan(
                FilterExpression=(
                    Attr("pk").begins_with(f"USER#{user_id}#ENT#")
                    & Attr("status").eq("active")
                ),
                ProjectionExpression="pk",
            )

            entities = set()
            for item in response.get("Items", []):
                pk = item.get("pk", "")
                # pk format: USER#{user_id}#ENT#{entity_id}
                parts = pk.split("#ENT#", 1)
                if len(parts) == 2:
                    entities.add(parts[1])

            return sorted(entities)

        except Exception as e:
            logger.error(f"Failed to get user entities: {e}")
            return []

    async def get_active_fact_by_key(
        self,
        user_id: str,
        entity_id: str,
        fact_key: str,
    ) -> Optional[FactRecord]:
        """Get the single active fact for a specific entity + fact_key."""
        facts = await self.get_active_facts(user_id, entity_id, fact_keys=[fact_key])
        return facts[0] if facts else None

    async def put_fact(self, fact: FactRecord) -> None:
        """
        Store a fact to DynamoDB.

        Caller is responsible for setting status correctly and deprecating
        any prior active fact for the same (entity_id, fact_key).
        """
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, skipping fact write")
            return

        try:
            raw = fact.model_dump(mode="json")
            item = _serialize_item({
                "pk": fact.dynamo_pk(),
                "sk": fact.dynamo_sk(),
                "entity": "fact",
                **raw,
            })

            table = self._fact_table()
            table.put_item(Item=item)
            logger.debug(f"Stored fact: {item['pk']}/{item['sk']}")

        except Exception as e:
            logger.error(f"Failed to put fact: {e}")
            raise

    async def deprecate_fact(self, fact: FactRecord) -> None:
        """Mark an existing fact as deprecated."""
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, skipping deprecation")
            return

        try:
            table = self._fact_table()
            table.update_item(
                Key={"pk": fact.dynamo_pk(), "sk": fact.dynamo_sk()},
                UpdateExpression="SET #s = :s, updated_at = :t",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":s": "deprecated",
                    ":t": datetime.utcnow().isoformat(),
                },
            )
            logger.debug(f"Deprecated fact: {fact.fact_id}")

        except Exception as e:
            logger.error(f"Failed to deprecate fact: {e}")
            raise

    async def upsert_fact(
        self,
        user_id: str,
        entity_id: str,
        fact_key: str,
        new_value: Any,
        *,
        fact_label: str = "",
        value_type: str = "string",
        status: FactStatus = "active",
        risk_level: RiskLevel = "medium",
        confidence: float = 0.0,
        source_type: str = "agent_inference",
        source_ref: str = "",
        evidence: str = "",
        verification_level: str = "unverified",
        supersedes_fact_id: Optional[str] = None,
        conflict_strategy: str = "overwrite",
    ) -> FactRecord:
        """
        Create-or-update a fact with proper version chaining.

        If an active fact exists for the same (entity_id, fact_key):
        - Same value → update last_seen_at only (returns existing)
        - Different value:
            - conflict_strategy="overwrite" (default): deprecate old, create new
            - conflict_strategy="needs_confirm" AND risk_level is "high":
              keep old fact active, store new fact as status="candidate"
              (the caller — typically the confirmation flow — will promote later)
        """
        existing = await self.get_active_fact_by_key(user_id, entity_id, fact_key)
        now = datetime.utcnow()

        if existing and existing.fact_value == new_value:
            # Same value — just touch timestamps
            existing.last_seen_at = now
            existing.updated_at = now
            if evidence and evidence not in (existing.evidence or ""):
                existing.evidence = evidence
            await self.put_fact(existing)
            return existing

        # Different value or no existing — decide how to handle
        if existing and conflict_strategy == "needs_confirm" and risk_level == "high":
            # High-risk conflict: keep old active, store new as candidate
            new_fact = FactRecord(
                fact_id=new_uuid("fact"),
                user_id=user_id,
                entity_id=entity_id,
                fact_key=fact_key,
                fact_label=fact_label,
                fact_value=new_value,
                value_type=value_type,
                status="candidate",
                risk_level=risk_level,
                confidence=confidence,
                source_type=source_type,
                source_ref=source_ref,
                evidence=evidence,
                verification_level=verification_level,
                supersedes_fact_id=existing.fact_id,
                first_seen_at=now,
                last_seen_at=now,
                created_at=now,
                updated_at=now,
            )
            await self.put_fact(new_fact)
            return new_fact

        # Default: deprecate old if exists, create new as active
        if existing:
            await self.deprecate_fact(existing)
            supersedes_fact_id = existing.fact_id

        new_fact = FactRecord(
            fact_id=new_uuid("fact"),
            user_id=user_id,
            entity_id=entity_id,
            fact_key=fact_key,
            fact_label=fact_label,
            fact_value=new_value,
            value_type=value_type,
            status=status,
            risk_level=risk_level,
            confidence=confidence,
            source_type=source_type,
            source_ref=source_ref,
            evidence=evidence,
            verification_level=verification_level,
            supersedes_fact_id=supersedes_fact_id,
            first_seen_at=now,
            last_seen_at=now,
            created_at=now,
            updated_at=now,
        )

        await self.put_fact(new_fact)
        return new_fact

    async def get_all_active_facts_for_user(
        self,
        user_id: str,
    ) -> List[FactRecord]:
        """Get all active facts across all entities for a user.

        Note: This scans the table with a filter, which is less efficient
        than a GSI query. For production with many users, use GSI2.
        """
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, returning empty")
            return []

        try:
            table = self._fact_table()
            response = table.scan(
                FilterExpression=Attr("user_id").eq(user_id) & Attr("status").eq("active"),
            )

            facts = []
            for item in response.get("Items", []):
                try:
                    facts.append(_deserialize_fact(item))
                except Exception as e:
                    logger.warning(f"Failed to deserialize fact: {e}")

            return facts

        except Exception as e:
            logger.error(f"Failed to get all active facts: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # FactAliasTable
    # ─────────────────────────────────────────────────────────

    async def get_aliases_for_text(
        self,
        text_tokens: List[str],
    ) -> List[AliasRecord]:
        """
        Look up aliases matching any of the given tokens/phrases.

        Returns all matching alias records (active or pending).
        """
        if not self.dynamodb:
            return []

        results: List[AliasRecord] = []
        try:
            table = self._alias_table()
            for token in text_tokens:
                normalized = token.strip().lower()
                if not normalized:
                    continue

                response = table.query(
                    KeyConditionExpression=Key("pk").eq(f"ALIAS#{normalized}"),
                )
                for item in response.get("Items", []):
                    try:
                        data = {k: v for k, v in item.items() if k not in ("pk", "sk", "entity")}
                        results.append(AliasRecord(**data))
                    except Exception as e:
                        logger.warning(f"Failed to deserialize alias: {e}")

            return results

        except Exception as e:
            logger.error(f"Failed to get aliases: {e}")
            return []

    async def put_alias(self, alias: AliasRecord) -> None:
        """Store or update an alias mapping."""
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, skipping alias write")
            return

        try:
            raw = alias.model_dump(mode="json")
            item = _serialize_item({
                "pk": alias.dynamo_pk(),
                "sk": alias.dynamo_sk(),
                "entity": "alias",
                **raw,
            })

            table = self._alias_table()
            table.put_item(Item=item)
            logger.debug(
                f"Stored alias: {alias.normalized_alias} -> {alias.canonical_key}"
            )

        except Exception as e:
            logger.error(f"Failed to put alias: {e}")
            raise

    async def increment_alias_count(
        self,
        normalized_alias: str,
        canonical_key: str,
    ) -> None:
        """Increment the count for an existing alias and update last_seen_at."""
        if not self.dynamodb:
            return

        try:
            table = self._alias_table()
            table.update_item(
                Key={
                    "pk": f"ALIAS#{normalized_alias}",
                    "sk": f"KEY#{canonical_key}",
                },
                UpdateExpression="SET #c = #c + :inc, last_seen_at = :now",
                ExpressionAttributeNames={"#c": "count"},
                ExpressionAttributeValues={
                    ":inc": 1,
                    ":now": datetime.utcnow().isoformat(),
                },
            )
            logger.debug(f"Incremented alias count: {normalized_alias}")

        except Exception as e:
            logger.error(f"Failed to increment alias: {e}")

    # ─────────────────────────────────────────────────────────
    # MemoryFactLogTable
    # ─────────────────────────────────────────────────────────

    async def write_fact_log(
        self,
        user_id: str,
        target_id: str,
        patch: Dict[str, Any],
        justification: str,
        actor: str,
        *,
        target: str = "fact",
        source_ref: str = "",
        result: str = "applied",
    ) -> FactLogEntry:
        """Write an audit log entry for a fact write/update."""
        entry = FactLogEntry(
            log_id=new_uuid("flog"),
            user_id=user_id,
            target=target,
            target_id=target_id,
            patch=patch,
            justification=justification,
            source_ref=source_ref,
            actor=actor,
            result=result,
        )

        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, skipping fact log write")
            return entry

        try:
            raw = entry.model_dump(mode="json")
            item = _serialize_item({
                "pk": entry.dynamo_pk(),
                "sk": entry.dynamo_sk(),
                "entity": "fact_log",
                **raw,
            })

            table = self._log_table()
            table.put_item(Item=item)
            logger.debug(f"Wrote fact log: {entry.log_id}")

        except Exception as e:
            logger.error(f"Failed to write fact log: {e}")

        return entry


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

_fact_store: Optional[FactStore] = None


def get_fact_store() -> FactStore:
    """Get or create the global FactStore instance."""
    global _fact_store
    if _fact_store is None:
        _fact_store = FactStore(dynamodb_client=get_ddb_resource())
    return _fact_store
