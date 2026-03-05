#!/usr/bin/env python3
"""
Create all DynamoDB tables for the WithCare User Memory Framework.

Tables:
  1. WithCare_UserFactTable      — entity-centric facts with versioning
  2. WithCare_UserEventTable     — short-term episodic memory (TTL-enabled)
  3. WithCare_FactAliasTable     — alias → canonical key mappings
  4. WithCare_MemoryFactLogTable — audit trail for fact writes
  5. WithCare_UserRequestTable   — persistent request records (evolves existing)

Usage:
  # Make sure your AWS profile is active:
  sky-withcare-prod

  # Create all tables:
  python scripts/create_dynamodb_tables.py

  # Create a specific table:
  python scripts/create_dynamodb_tables.py --table UserFactTable

  # Dry run (just print what would be created):
  python scripts/create_dynamodb_tables.py --dry-run

  # Delete tables (careful!):
  python scripts/create_dynamodb_tables.py --delete
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-2"
TABLE_PREFIX = "WithCare_"

# ─────────────────────────────────────────────────────────────
# Table definitions
# ─────────────────────────────────────────────────────────────

TABLES = {
    "UserFactTable": {
        "KeySchema": [
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            # GSI1: lookup active facts by entity + fact_key
            {"AttributeName": "gsi1pk", "AttributeType": "S"},
            {"AttributeName": "gsi1sk", "AttributeType": "S"},
            # GSI2: list all active facts for a user
            {"AttributeName": "gsi2pk", "AttributeType": "S"},
            {"AttributeName": "gsi2sk", "AttributeType": "S"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "gsi1pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi1sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "GSI2",
                "KeySchema": [
                    {"AttributeName": "gsi2pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi2sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "_description": (
            "PK=USER#<user_id>#ENT#<entity_id>, "
            "SK=FACT#<fact_key>#TS#<ts>#<fact_id>. "
            "GSI1PK=USER#<user_id>#ENT#<entity_id>, GSI1SK=KEY#<fact_key>#STATUS#<status>. "
            "GSI2PK=USER#<user_id>#STATUS#active, GSI2SK=KEY#<fact_key>."
        ),
    },
    "UserEventTable": {
        "KeySchema": [
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "_description": (
            "PK=USER#<user_id>, SK=EVT#<timestamp>#<event_id>. "
            "TTL attribute: ttl_epoch."
        ),
        "_ttl_attribute": "ttl_epoch",
    },
    "FactAliasTable": {
        "KeySchema": [
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "_description": (
            "PK=ALIAS#<normalized_alias>, SK=KEY#<canonical_key>."
        ),
    },
    "MemoryFactLogTable": {
        "KeySchema": [
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "_description": (
            "PK=USER#<user_id>, SK=FACTLOG#<timestamp>#<log_id>."
        ),
    },
    "CandidateKeyPool": {
        "KeySchema": [
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "_description": "PK=CKEY#<normalized_key>, SK=META. Global pool across all users.",
    },
    "UserConversationTable": {
        "KeySchema": [
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            # GSI1: query by user + date
            {"AttributeName": "gsi1pk", "AttributeType": "S"},
            {"AttributeName": "gsi1sk", "AttributeType": "S"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "gsi1pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi1sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "_description": (
            "PK=CONV#<conversation_id>, SK=MSG#<timestamp_iso>#<message_id>. "
            "GSI1PK=USER#<user_id>#DATE#<YYYY-MM-DD>, GSI1SK=MSG#<timestamp_iso>#<message_id>."
        ),
    },
    "UserRequestTable": {
        "KeySchema": [
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            # GSI1: find requests by status
            {"AttributeName": "gsi1pk", "AttributeType": "S"},
            {"AttributeName": "gsi1sk", "AttributeType": "S"},
            # GSI2: find requests by care recipient
            {"AttributeName": "gsi2pk", "AttributeType": "S"},
            {"AttributeName": "gsi2sk", "AttributeType": "S"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "gsi1pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi1sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "GSI2",
                "KeySchema": [
                    {"AttributeName": "gsi2pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi2sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "_description": (
            "PK=USER#<user_id>, SK=REQ#<created_at>#<request_id>. "
            "GSI1PK=USER#<user_id>#STATUS#<status>, GSI1SK=LAST#<last_activity_at>#REQ#<request_id>. "
            "GSI2PK=RECIPIENT#<care_recipient_id>, GSI2SK=LAST#<last_activity_at>#REQ#<request_id>."
        ),
    },
}


def full_table_name(short_name: str) -> str:
    return f"{TABLE_PREFIX}{short_name}"


def create_table(client, short_name: str, spec: dict, dry_run: bool = False) -> bool:
    """Create a single DynamoDB table. Returns True if created, False if exists."""
    table_name = full_table_name(short_name)

    # Check if already exists
    try:
        client.describe_table(TableName=table_name)
        print(f"  [SKIP] {table_name} already exists")
        return False
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    if dry_run:
        print(f"  [DRY RUN] Would create {table_name}")
        print(f"    {spec.get('_description', '')}")
        return False

    # Build create params
    params = {
        "TableName": table_name,
        "KeySchema": spec["KeySchema"],
        "AttributeDefinitions": spec["AttributeDefinitions"],
        "BillingMode": spec["BillingMode"],
    }
    if "GlobalSecondaryIndexes" in spec:
        params["GlobalSecondaryIndexes"] = spec["GlobalSecondaryIndexes"]

    print(f"  [CREATE] {table_name} ...")
    client.create_table(**params)

    # Wait for table to become active
    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=table_name, WaiterConfig={"Delay": 2, "MaxAttempts": 30})
    print(f"  [OK] {table_name} is ACTIVE")

    # Enable TTL if specified
    ttl_attr = spec.get("_ttl_attribute")
    if ttl_attr:
        client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={
                "Enabled": True,
                "AttributeName": ttl_attr,
            },
        )
        print(f"  [OK] TTL enabled on attribute '{ttl_attr}'")

    return True


def delete_table(client, short_name: str, dry_run: bool = False) -> bool:
    """Delete a single DynamoDB table."""
    table_name = full_table_name(short_name)

    try:
        client.describe_table(TableName=table_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            print(f"  [SKIP] {table_name} does not exist")
            return False
        raise

    if dry_run:
        print(f"  [DRY RUN] Would delete {table_name}")
        return False

    print(f"  [DELETE] {table_name} ...")
    client.delete_table(TableName=table_name)
    waiter = client.get_waiter("table_not_exists")
    waiter.wait(TableName=table_name, WaiterConfig={"Delay": 2, "MaxAttempts": 30})
    print(f"  [OK] {table_name} deleted")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create WithCare DynamoDB tables")
    parser.add_argument("--table", help="Create only this table (short name)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--delete", action="store_true", help="Delete tables instead of creating")
    parser.add_argument("--region", default=REGION, help=f"AWS region (default: {REGION})")
    args = parser.parse_args()

    client = boto3.client("dynamodb", region_name=args.region)

    # Verify identity
    sts = boto3.client("sts", region_name=args.region)
    try:
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
        print(f"Region:      {args.region}")
        print(f"Identity:    {identity['Arn']}")
    except Exception as e:
        print(f"ERROR: Cannot verify AWS identity: {e}")
        print("Make sure your AWS profile is active (e.g., sky-withcare-prod)")
        sys.exit(1)

    # Select tables
    if args.table:
        if args.table not in TABLES:
            print(f"ERROR: Unknown table '{args.table}'")
            print(f"Available: {', '.join(TABLES.keys())}")
            sys.exit(1)
        tables_to_process = {args.table: TABLES[args.table]}
    else:
        tables_to_process = TABLES

    action = "delete" if args.delete else "create"
    print(f"\n{'='*60}")
    print(f"{'DRY RUN: ' if args.dry_run else ''}{action.upper()} {len(tables_to_process)} table(s)")
    print(f"{'='*60}\n")

    if args.delete and not args.dry_run:
        confirm = input(
            f"Are you sure you want to DELETE {len(tables_to_process)} table(s)? "
            f"This cannot be undone. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    count = 0
    for short_name, spec in tables_to_process.items():
        if args.delete:
            if delete_table(client, short_name, args.dry_run):
                count += 1
        else:
            if create_table(client, short_name, spec, args.dry_run):
                count += 1

    print(f"\nDone. {count} table(s) {action}d.")

    if not args.dry_run and not args.delete and count > 0:
        print("\nTable summary:")
        for short_name in tables_to_process:
            table_name = full_table_name(short_name)
            try:
                resp = client.describe_table(TableName=table_name)
                t = resp["Table"]
                status = t["TableStatus"]
                gsi_count = len(t.get("GlobalSecondaryIndexes", []))
                ttl_spec = spec.get("_ttl_attribute", "none")
                print(f"  {table_name}: {status}, GSIs={gsi_count}, TTL={ttl_spec}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
