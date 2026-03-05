"""
Backfill GSI keys (gsi1pk, gsi1sk, gsi2pk, gsi2sk) on existing
WithCare_UserRequestTable items that were written before the GSI
key fix in request_factory.py.

Usage:
    python scripts/backfill_request_gsi_keys.py [--dry-run]

Scans all items with entity="request", computes missing GSI keys
from existing fields, and writes them back via update_item().
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddb_client import get_ddb_resource, USER_REQUEST_TABLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def backfill(dry_run: bool = False) -> None:
    ddb = get_ddb_resource()
    if ddb is None:
        logger.error("DynamoDB resource not available. Check AWS config.")
        sys.exit(1)

    table = ddb.Table(USER_REQUEST_TABLE)
    logger.info(f"Scanning {USER_REQUEST_TABLE} for request items missing GSI keys...")

    scanned = 0
    updated = 0
    skipped = 0

    scan_kwargs = {
        "FilterExpression": "entity = :ent",
        "ExpressionAttributeValues": {":ent": "request"},
    }

    while True:
        response = table.scan(**scan_kwargs)
        items = response.get("Items", [])

        for item in items:
            scanned += 1
            pk = item.get("pk", "")
            sk = item.get("sk", "")
            user_id = item.get("user_id", "")
            request_id = item.get("request_id", "")
            status = item.get("status", "created")
            last_touched = item.get("last_touched_at", item.get("created_at", ""))
            subject_entity_id = item.get("subject_entity_id", "")

            if not user_id or not request_id:
                logger.warning(f"Skipping item with missing user_id/request_id: pk={pk}, sk={sk}")
                skipped += 1
                continue

            # Check if GSI keys already present
            if item.get("gsi1pk"):
                logger.debug(f"Already has gsi1pk: {request_id}")
                skipped += 1
                continue

            # Compute GSI keys
            gsi1pk = f"USER#{user_id}#STATUS#{status}"
            gsi1sk = f"LAST#{last_touched}#REQ#{request_id}"

            update_expr_parts = [
                "#gsi1pk = :gsi1pk",
                "#gsi1sk = :gsi1sk",
            ]
            expr_names = {
                "#gsi1pk": "gsi1pk",
                "#gsi1sk": "gsi1sk",
            }
            expr_values = {
                ":gsi1pk": gsi1pk,
                ":gsi1sk": gsi1sk,
            }

            if subject_entity_id:
                gsi2pk = f"RECIPIENT#{subject_entity_id}"
                gsi2sk = f"LAST#{last_touched}#REQ#{request_id}"
                update_expr_parts.append("#gsi2pk = :gsi2pk")
                update_expr_parts.append("#gsi2sk = :gsi2sk")
                expr_names["#gsi2pk"] = "gsi2pk"
                expr_names["#gsi2sk"] = "gsi2sk"
                expr_values[":gsi2pk"] = gsi2pk
                expr_values[":gsi2sk"] = gsi2sk

            update_expression = "SET " + ", ".join(update_expr_parts)

            if dry_run:
                logger.info(f"[DRY RUN] Would update {request_id}: gsi1pk={gsi1pk}")
            else:
                table.update_item(
                    Key={"pk": pk, "sk": sk},
                    UpdateExpression=update_expression,
                    ExpressionAttributeNames=expr_names,
                    ExpressionAttributeValues=expr_values,
                )
                logger.info(f"Updated {request_id}: gsi1pk={gsi1pk}")

            updated += 1

        # Pagination
        if "LastEvaluatedKey" in response:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        else:
            break

    action = "Would update" if dry_run else "Updated"
    logger.info(f"Done. Scanned={scanned}, {action}={updated}, Skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill GSI keys on UserRequestTable")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)
