"""
Shared DynamoDB client for the WithCare User Memory Framework.

Provides a single boto3 DynamoDB resource (high-level) and client (low-level)
that all store modules use. Lazy-initialized on first access.

Configuration via environment variables:
  AWS_REGION           — defaults to us-east-2
  WITHCARE_DDB_PREFIX  — table name prefix, defaults to "WithCare_"
  WITHCARE_DDB_OFF     — set to "1" to disable DDB (returns None clients,
                         stores fall back to logging-only mode)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
TABLE_PREFIX = os.environ.get("WITHCARE_DDB_PREFIX", "WithCare_")
DDB_DISABLED = os.environ.get("WITHCARE_DDB_OFF", "0") == "1"

# Table name constants (used by all store modules)
USER_FACT_TABLE = f"{TABLE_PREFIX}UserFactTable"
USER_EVENT_TABLE = f"{TABLE_PREFIX}UserEventTable"
FACT_ALIAS_TABLE = f"{TABLE_PREFIX}FactAliasTable"
MEMORY_FACT_LOG_TABLE = f"{TABLE_PREFIX}MemoryFactLogTable"
USER_REQUEST_TABLE = f"{TABLE_PREFIX}UserRequestTable"
CANDIDATE_KEY_POOL_TABLE = f"{TABLE_PREFIX}CandidateKeyPool"
USER_CONVERSATION_TABLE = f"{TABLE_PREFIX}UserConversationTable"
CHAT_MESSAGES_TABLE = os.environ.get("CHAT_MESSAGES_TABLE", "ChatMessages")

# Lazy singletons
_ddb_client: Optional[Any] = None
_ddb_resource: Optional[Any] = None
_initialized = False


def _init() -> None:
    """Initialize boto3 DynamoDB client and resource."""
    global _ddb_client, _ddb_resource, _initialized

    if _initialized:
        return

    if DDB_DISABLED:
        logger.info("DynamoDB disabled (WITHCARE_DDB_OFF=1), stores will log-only")
        _ddb_client = None
        _ddb_resource = None
        _initialized = True
        return

    try:
        import boto3

        _ddb_client = boto3.client("dynamodb", region_name=AWS_REGION)
        _ddb_resource = boto3.resource("dynamodb", region_name=AWS_REGION)
        _initialized = True
        logger.info(f"DynamoDB client initialized (region={AWS_REGION})")
    except Exception as e:
        logger.warning(
            f"Failed to initialize DynamoDB client: {e}. "
            f"Stores will run in log-only mode."
        )
        _ddb_client = None
        _ddb_resource = None
        _initialized = True


def get_ddb_client() -> Optional[Any]:
    """Get the low-level boto3 DynamoDB client (or None if disabled/failed)."""
    _init()
    return _ddb_client


def get_ddb_resource() -> Optional[Any]:
    """Get the high-level boto3 DynamoDB resource (or None if disabled/failed)."""
    _init()
    return _ddb_resource


def get_table(table_name: str) -> Optional[Any]:
    """Get a DynamoDB Table resource object (or None if disabled)."""
    resource = get_ddb_resource()
    if resource is None:
        return None
    return resource.Table(table_name)
