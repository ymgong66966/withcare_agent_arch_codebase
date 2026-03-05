"""
Test 01: Verify DynamoDB tables exist and are ACTIVE.

Run after: python scripts/create_dynamodb_tables.py

Usage:
    sky-withcare-prod
    python tests/test_01_ddb_tables_exist.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddb_client import (
    get_ddb_client,
    USER_FACT_TABLE,
    USER_EVENT_TABLE,
    FACT_ALIAS_TABLE,
    MEMORY_FACT_LOG_TABLE,
    USER_REQUEST_TABLE,
    AWS_REGION,
)

EXPECTED_TABLES = [
    USER_FACT_TABLE,
    USER_EVENT_TABLE,
    FACT_ALIAS_TABLE,
    MEMORY_FACT_LOG_TABLE,
    USER_REQUEST_TABLE,
]


def test_tables_exist():
    import boto3
    client = boto3.client("dynamodb", region_name=AWS_REGION)

    print(f"Region: {AWS_REGION}")
    print(f"Checking {len(EXPECTED_TABLES)} tables...\n")

    all_ok = True
    for table_name in EXPECTED_TABLES:
        try:
            resp = client.describe_table(TableName=table_name)
            t = resp["Table"]
            status = t["TableStatus"]
            item_count = t.get("ItemCount", 0)
            gsi_count = len(t.get("GlobalSecondaryIndexes", []))
            gsi_names = [g["IndexName"] for g in t.get("GlobalSecondaryIndexes", [])]

            if status == "ACTIVE":
                print(f"  [OK] {table_name}")
                print(f"       Status={status}, Items={item_count}, GSIs={gsi_count} {gsi_names}")
            else:
                print(f"  [WARN] {table_name} status={status} (expected ACTIVE)")
                all_ok = False

        except client.exceptions.ResourceNotFoundException:
            print(f"  [FAIL] {table_name} — NOT FOUND")
            all_ok = False
        except Exception as e:
            print(f"  [FAIL] {table_name} — {e}")
            all_ok = False

    # Check TTL on UserEventTable
    print()
    try:
        ttl_resp = client.describe_time_to_live(TableName=USER_EVENT_TABLE)
        ttl_status = ttl_resp["TimeToLiveDescription"]["TimeToLiveStatus"]
        ttl_attr = ttl_resp["TimeToLiveDescription"].get("AttributeName", "N/A")
        if ttl_status in ("ENABLED", "ENABLING"):
            print(f"  [OK] TTL on {USER_EVENT_TABLE}: {ttl_status} (attribute={ttl_attr})")
        else:
            print(f"  [WARN] TTL on {USER_EVENT_TABLE}: {ttl_status} (expected ENABLED)")
    except Exception as e:
        print(f"  [WARN] TTL check failed: {e}")

    print()
    if all_ok:
        print("ALL TABLES OK")
    else:
        print("SOME TABLES MISSING OR NOT ACTIVE — run: python scripts/create_dynamodb_tables.py")
        sys.exit(1)


if __name__ == "__main__":
    test_tables_exist()
