#!/usr/bin/env python3
"""
Run all memory framework tests in order.

Usage:
    # Run everything (tests 01-04 need DDB tables to exist):
    sky-withcare-prod
    python tests/run_all.py

    # Run only offline tests (no DDB needed):
    python tests/run_all.py --offline
"""

import argparse
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tests ordered by dependency
ALL_TESTS = [
    ("test_05_key_resolver.py",            False, "Key Resolver BM25 accuracy"),
    ("test_07_request_factory_new_fields.py", False, "Request factory backward compat"),
    ("test_01_ddb_tables_exist.py",        True,  "DDB tables exist"),
    ("test_02_fact_store_crud.py",         True,  "Fact Store CRUD"),
    ("test_03_event_store.py",             True,  "Event Store write/read"),
    ("test_04_write_gate.py",              True,  "Write Gate classification + commit"),
    ("test_06_context_bundle.py",          True,  "Context Bundle assembly"),
    ("test_08_slot_fact_binding.py",       False, "Slot→Fact binding pipeline"),
    ("test_09_entity_inference.py",        False, "Entity ID inference (Chinese + LLM)"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Only run tests that don't need DDB")
    args = parser.parse_args()

    test_dir = os.path.dirname(os.path.abspath(__file__))
    total = 0
    passed_tests = 0
    failed_tests = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"WithCare Memory Framework Test Suite")
    print(f"{'='*60}")
    if args.offline:
        print("Mode: OFFLINE (skipping DDB tests)\n")
    else:
        print("Mode: FULL (requires DDB tables + AWS credentials)\n")

    for filename, needs_ddb, description in ALL_TESTS:
        if args.offline and needs_ddb:
            print(f"  [SKIP] {description} (needs DDB)")
            skipped += 1
            continue

        total += 1
        filepath = os.path.join(test_dir, filename)
        print(f"\n{'─'*60}")
        print(f"Running: {description} ({filename})")
        print(f"{'─'*60}")

        result = subprocess.run(
            [sys.executable, filepath],
            cwd=ROOT,
            timeout=60,
        )

        if result.returncode == 0:
            passed_tests += 1
        else:
            failed_tests += 1
            print(f"  *** FAILED (exit code {result.returncode}) ***")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed_tests}/{total} passed, {failed_tests} failed, {skipped} skipped")
    print(f"{'='*60}")

    sys.exit(1 if failed_tests > 0 else 0)


if __name__ == "__main__":
    main()
