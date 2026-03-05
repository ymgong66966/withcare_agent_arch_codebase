"""
Candidate Key Pool — DynamoDB CRUD for the global CandidateKeyPool table.

When the Key Resolver encounters a fact key that doesn't match any canonical
key in the registry, the proposed key lands here. Frequently-observed candidate
keys can later be promoted to canonical via a cron job or admin review.

The pool is global (cross-user): the same proposed key from different users
increments the same counter, enabling organic discovery of missing keys.

DDB schema:
  PK = CKEY#<normalized_key>
  SK = META
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ddb_client import CANDIDATE_KEY_POOL_TABLE, get_ddb_resource
from models.fact_models import CandidateKeyRecord, RiskLevel

logger = logging.getLogger(__name__)

MAX_SAMPLE_SIZE = 5


def _normalize_key(raw: str) -> str:
    """Normalize a proposed key: lowercase, strip, replace spaces/hyphens with underscores."""
    key = raw.strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    # Collapse multiple underscores
    key = re.sub(r"_+", "_", key)
    # Strip leading/trailing underscores
    key = key.strip("_")
    return key


def _serialize_value(v: Any) -> Any:
    """Convert Python types to DynamoDB-safe types."""
    if isinstance(v, datetime):
        return v.isoformat()
    elif isinstance(v, float):
        return Decimal(str(v))
    elif v is None:
        return None
    return v


def _deserialize_record(item: Dict[str, Any]) -> CandidateKeyRecord:
    """Convert a DynamoDB item to a CandidateKeyRecord."""
    data = {k: v for k, v in item.items() if k not in ("pk", "sk", "entity")}
    # Convert Decimal back to int for occurrence_count
    if "occurrence_count" in data and isinstance(data["occurrence_count"], Decimal):
        data["occurrence_count"] = int(data["occurrence_count"])
    return CandidateKeyRecord(**data)


class CandidateKeyPoolStore:
    """CRUD operations for the global CandidateKeyPool table."""

    def __init__(self, dynamodb_client: Optional[Any] = None):
        self.dynamodb = dynamodb_client

    def _table(self):
        return self.dynamodb.Table(CANDIDATE_KEY_POOL_TABLE)

    async def record_candidate(
        self,
        candidate_key: str,
        description: str = "",
        risk_suggestion: RiskLevel = "medium",
        value: str = "",
        entity_id: str = "",
        request_id: str = "",
    ) -> Optional[CandidateKeyRecord]:
        """
        Record or increment a candidate key in the global pool.

        Uses DDB UpdateItem with atomic increment for occurrence_count
        and if_not_exists for first_seen_at.
        """
        normalized = _normalize_key(candidate_key)
        if not normalized:
            logger.warning("Empty candidate key after normalization, skipping")
            return None

        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, skipping candidate write")
            return CandidateKeyRecord(
                candidate_key=normalized,
                description=description,
                risk_suggestion=risk_suggestion,
            )

        now_iso = datetime.utcnow().isoformat()
        pk = f"CKEY#{normalized}"
        sk = "META"

        try:
            table = self._table()
            # Atomic upsert: increment count, set first_seen if new, always update last_seen
            update_expr = (
                "SET description = if_not_exists(description, :desc), "
                "risk_suggestion = if_not_exists(risk_suggestion, :risk), "
                "first_seen_at = if_not_exists(first_seen_at, :now), "
                "last_seen_at = :now, "
                "#st = if_not_exists(#st, :candidate), "
                "candidate_key = :ckey "
                "ADD occurrence_count :one"
            )
            expr_values = {
                ":desc": description,
                ":risk": risk_suggestion,
                ":now": now_iso,
                ":candidate": "candidate",
                ":ckey": normalized,
                ":one": 1,
            }
            expr_names = {"#st": "status"}

            table.update_item(
                Key={"pk": pk, "sk": sk},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values,
                ExpressionAttributeNames=expr_names,
            )

            # Append to sample lists (separate update to keep expressions simpler)
            self._append_samples(table, pk, sk, value, entity_id, request_id)

            logger.debug(f"Recorded candidate key: {normalized}")
            return await self.get_candidate(normalized)

        except Exception as e:
            logger.error(f"Failed to record candidate key {normalized}: {e}")
            return None

    def _append_samples(
        self,
        table: Any,
        pk: str,
        sk: str,
        value: str,
        entity_id: str,
        request_id: str,
    ) -> None:
        """Append value/entity/request to sample lists, read-then-write with cap."""
        try:
            resp = table.get_item(Key={"pk": pk, "sk": sk})
            item = resp.get("Item", {})

            sample_values = list(item.get("sample_values", []))
            sample_entities = list(item.get("sample_entities", []))
            source_requests = list(item.get("source_requests", []))

            if value and value not in sample_values:
                sample_values.append(value)
            if entity_id and entity_id not in sample_entities:
                sample_entities.append(entity_id)
            if request_id and request_id not in source_requests:
                source_requests.append(request_id)

            # Cap at MAX_SAMPLE_SIZE
            sample_values = sample_values[:MAX_SAMPLE_SIZE]
            sample_entities = sample_entities[:MAX_SAMPLE_SIZE]
            source_requests = source_requests[:MAX_SAMPLE_SIZE]

            table.update_item(
                Key={"pk": pk, "sk": sk},
                UpdateExpression=(
                    "SET sample_values = :sv, "
                    "sample_entities = :se, "
                    "source_requests = :sr"
                ),
                ExpressionAttributeValues={
                    ":sv": sample_values,
                    ":se": sample_entities,
                    ":sr": source_requests,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to append samples for {pk}: {e}")

    async def get_candidate(self, candidate_key: str) -> Optional[CandidateKeyRecord]:
        """Point read for a single candidate key."""
        normalized = _normalize_key(candidate_key)
        if not self.dynamodb:
            return None

        try:
            table = self._table()
            resp = table.get_item(Key={"pk": f"CKEY#{normalized}", "sk": "META"})
            item = resp.get("Item")
            if not item:
                return None
            return _deserialize_record(item)
        except Exception as e:
            logger.error(f"Failed to get candidate {normalized}: {e}")
            return None

    async def get_top_candidates(
        self,
        min_count: int = 3,
        limit: int = 50,
    ) -> List[CandidateKeyRecord]:
        """
        Scan for candidate keys with occurrence_count >= min_count.

        This is a scan (not ideal for large tables) — intended for periodic
        promotion review, not hot-path usage.
        """
        if not self.dynamodb:
            return []

        try:
            from boto3.dynamodb.conditions import Attr

            table = self._table()
            resp = table.scan(
                FilterExpression=(
                    Attr("occurrence_count").gte(min_count)
                    & Attr("status").eq("candidate")
                ),
                Limit=limit,
            )
            items = resp.get("Items", [])
            records = [_deserialize_record(item) for item in items]
            # Sort by occurrence_count descending
            records.sort(key=lambda r: r.occurrence_count, reverse=True)
            return records[:limit]
        except Exception as e:
            logger.error(f"Failed to scan top candidates: {e}")
            return []


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

_pool_store: Optional[CandidateKeyPoolStore] = None


def get_candidate_key_pool_store() -> CandidateKeyPoolStore:
    """Get or create the global CandidateKeyPoolStore instance."""
    global _pool_store
    if _pool_store is None:
        _pool_store = CandidateKeyPoolStore(dynamodb_client=get_ddb_resource())
    return _pool_store
