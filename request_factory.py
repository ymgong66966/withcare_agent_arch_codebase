from __future__ import annotations

from typing import Any, Dict, Optional, Literal
from datetime import datetime, timezone

from ddb_client import USER_REQUEST_TABLE
from id_utils import new_uuid

TargetEntity = Literal["care_recipient", "caregiver", "both", "unknown"]
Priority = Literal["low", "normal", "high"]

def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def build_request_patch(
    *,
    conversation_id: str,
    user_id: str,
    name: str,
    goal: str = "",
    target: TargetEntity = "unknown",
    priority: Priority = "normal",
    set_active: bool = True,
    request_id: Optional[str] = None,
    request_source: str = "chat",
    table_name: str = USER_REQUEST_TABLE,
    # NEW: structured request identity fields
    request_type: str = "",
    subject_entity_id: str = "",
    variant: Optional[Dict[str, str]] = None,
    title: str = "",
) -> Dict[str, Any]:
    rid = request_id or new_uuid("req")
    now = datetime.utcnow()

    request_record = {
        "request_id": rid,
        "name": name,
        "goal": goal,
        "target": target,
        "priority": priority,
        "created_at": now,
        "last_touched_at": now,
        "status": "created",
        "stage_detail": None,
        "awaiting_user_input": False,
        # NEW: structured identity
        "request_type": request_type,
        "subject_entity_id": subject_entity_id,
        "variant": variant or {},
        "title": title or name,
        "summary_current": "",
        "slot_refs": {},
        # Existing fields
        "slots": [],
        "open_questions": [],
        "artifacts": [],
        "prereq_gate": {"status": "none"},
        "deep_search_state": {"stage": "idle", "required_fields": [], "last_tool_failures": [], "awaiting_user_input": False},
        "status_flags": {"user_declined_more_questions": False, "profile_updated": False, "needs_human_escalation": False},
        "audit": {"message_ids_related": [], "tool_runs": []},
    }

    ddb_item = {
        "pk": f"USER#{user_id}",
        "sk": f"REQ#{rid}",
        "entity": "request",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "request_id": rid,
        "name": name,
        "goal": goal,
        "target": target,
        "priority": priority,
        "status": "created",
        "created_at": utc_iso(),
        "last_touched_at": utc_iso(),
        "source": request_source,
        # NEW: structured identity in DDB item
        "request_type": request_type,
        "subject_entity_id": subject_entity_id,
        "variant": variant or {},
        "title": title or name,
        "payload": request_record,
        # GSI keys for query access patterns
        "gsi1pk": f"USER#{user_id}#STATUS#created",
        "gsi1sk": f"LAST#{utc_iso()}#REQ#{rid}",
        **({"gsi2pk": f"RECIPIENT#{subject_entity_id}",
            "gsi2sk": f"LAST#{utc_iso()}#REQ#{rid}"}
           if subject_entity_id else {}),
    }

    rm_patch: Dict[str, Any] = {"requests": {rid: request_record}}
    if set_active:
        rm_patch["active_request_id"] = rid

    return {
        "request_manager": rm_patch,
        "ddb_writes": [{"op": "put", "table": table_name, "item": ddb_item}],
    }

def build_request_update_patch(
    *,
    user_id: str,
    request_id: str,
    updates: Dict[str, Any],
    table_name: str = USER_REQUEST_TABLE,
) -> Dict[str, Any]:
    """
    Generate a DDB UpdateItem write for request field changes.

    Used to sync in-memory status transitions back to DynamoDB so the
    persisted record stays in sync with the graph's in-memory state.

    Args:
        user_id: The user who owns the request.
        request_id: The request ID to update.
        updates: Dict of field names to new values (e.g. {"status": "paused",
                 "stage_detail": "paused_for_prerequisite"}).
        table_name: DDB table name (defaults to USER_REQUEST_TABLE).

    Returns:
        Dict with a "ddb_writes" list containing one update operation.
    """
    if not updates:
        return {"ddb_writes": []}

    # Build UpdateExpression parts
    set_parts = []
    expr_values = {}
    expr_names = {}

    for i, (field, value) in enumerate(updates.items()):
        placeholder_name = f"#f{i}"
        placeholder_value = f":v{i}"
        expr_names[placeholder_name] = field
        expr_values[placeholder_value] = _sanitize_update_value(value)
        set_parts.append(f"{placeholder_name} = {placeholder_value}")

    # Sync GSI1 key and payload.status when status changes
    if "status" in updates:
        set_parts.append("#gsi1pk = :gsi1pk_val")
        expr_names["#gsi1pk"] = "gsi1pk"
        expr_values[":gsi1pk_val"] = f"USER#{user_id}#STATUS#{updates['status']}"

        set_parts.append("#payload.#pstatus = :pstatus_val")
        expr_names["#payload"] = "payload"
        expr_names["#pstatus"] = "status"
        expr_values[":pstatus_val"] = updates["status"]

    # Always update gsi1sk so GSI1 sort reflects latest touch
    set_parts.append("#gsi1sk = :gsi1sk_val")
    expr_names["#gsi1sk"] = "gsi1sk"
    expr_values[":gsi1sk_val"] = f"LAST#{utc_iso()}#REQ#{request_id}"

    # Always update last_touched_at
    set_parts.append("#lt = :lt")
    expr_names["#lt"] = "last_touched_at"
    expr_values[":lt"] = utc_iso()

    update_expression = "SET " + ", ".join(set_parts)

    return {
        "ddb_writes": [{
            "op": "update",
            "table": table_name,
            "params": {
                "Key": {
                    "pk": f"USER#{user_id}",
                    "sk": f"REQ#{request_id}",
                },
                "UpdateExpression": update_expression,
                "ExpressionAttributeNames": expr_names,
                "ExpressionAttributeValues": expr_values,
            },
        }],
    }


def _sanitize_update_value(v: Any) -> Any:
    """Convert Python types to DynamoDB-safe types for update expressions."""
    from decimal import Decimal as _Decimal
    if isinstance(v, datetime):
        return v.isoformat()
    elif isinstance(v, float):
        return _Decimal(str(v))
    elif isinstance(v, dict):
        return {k: _sanitize_update_value(val) for k, val in v.items() if val is not None}
    elif isinstance(v, list):
        return [_sanitize_update_value(item) for item in v]
    return v


def build_prereq_switch_patch(
    *,
    parent_request_id: str,
    prereq_request_id: str,
    prereq_type: str,
    proposed_by: str,
) -> Dict[str, Any]:
    now = datetime.utcnow()
    return {
        "request_manager": {
            "requests": {
                parent_request_id: {
                    "prereq_gate": {
                        "status": "proposed",
                        "prereq_type": prereq_type,
                        "reason": f"Prerequisite missing: {prereq_type}",
                        "parent_request_id": parent_request_id,
                        "proposed_request_id": prereq_request_id,
                        "proposed_by": proposed_by,
                        "proposed_at": now,
                    }
                }
            }
        }
    }

def build_accept_prereq_patch(
    *,
    parent_request_id: str,
    prereq_request_id: str,
    prereq_type: str,
) -> Dict[str, Any]:
    """Accept a prerequisite: pause parent, activate prereq, enqueue parent.

    NOTE (DDB persistence hook): When wiring to DynamoDB, each mutation here
    should produce a corresponding ddb_writes entry:
      - UpdateItem on parent request (status→paused, prereq_gate→accepted)
      - PutItem for the PendingQueueItem
        Key: pk=CONV#{conversation_id}, sk=QUEUE#{parent_request_id}
      - UpdateItem on prereq request (parent_request_id set)
    """
    now = datetime.utcnow()
    return {
        "request_manager": {
            "active_request_id": prereq_request_id,
            "pending_queue": [{
                "request_id": parent_request_id,
                "name": "",
                "status": "pending",
                "reason_queued": f"Paused for prerequisite: {prereq_type}",
                "queued_at": now,
                "child_request_id": prereq_request_id,
            }],
            "requests": {
                parent_request_id: {
                    "prereq_gate": {
                        "status": "accepted",
                        "resolved_at": now,
                        "prereq_type": prereq_type,
                        "parent_request_id": parent_request_id,
                        "proposed_request_id": prereq_request_id,
                    }
                },
                prereq_request_id: {
                    "parent_request_id": parent_request_id,
                },
            }
        }
    }
