#!/usr/bin/env python3
"""
Daily cron job — runs all end-of-day workstreams for the WithCare agent.

Workstreams:
  1. Fact cross-validation + candidate key review (DailyFactJob)
  2. Request hygiene + status sync (DailyRequestAnalyzer.run_request_hygiene)
  3. Request archival to historical store (existing daily_analyzer flow)

Deployment options:
  A) AWS Lambda + EventBridge rule: cron(0 8 * * ? *) = midnight PT in UTC
  B) Kubernetes CronJob with TZ=America/Los_Angeles
  C) Traditional cron: 0 0 * * * TZ=America/Los_Angeles python run_daily_cron.py

Usage:
  python run_daily_cron.py                   # Process yesterday (default)
  python run_daily_cron.py --date 2026-03-01 # Process specific date
  python run_daily_cron.py --user user-123   # Process single user
  python run_daily_cron.py --dry-run         # Log what would happen, don't execute
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

from anthropic_client import TrackedAnthropicClient
from conversation_store import get_conversation_store
from daily_analyzer import DailyRequestAnalyzer
from daily_fact_job import DailyFactJob
from historical_store import HistoricalRequestStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("daily_cron")


async def run_daily_cron(
    target_date: Optional[str] = None,
    target_user: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Main entry point for the daily cron job.

    Args:
        target_date: Date to process in "YYYY-MM-DD" format.
                     Defaults to yesterday (Pacific Time).
        target_user: If set, only process this user ID.
        dry_run: If True, log actions without executing writes.
    """
    # Determine target date
    if target_date:
        date = target_date
    else:
        # Default: yesterday Pacific Time
        pacific = ZoneInfo("America/Los_Angeles")
        now_pt = datetime.now(pacific)
        yesterday = now_pt - timedelta(days=1)
        date = yesterday.strftime("%Y-%m-%d")

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting daily cron for date={date}")

    # 1. Get all users who had conversations today
    conv_store = get_conversation_store()

    if target_user:
        users = [target_user]
        logger.info(f"Processing single user: {target_user}")
    else:
        users = await conv_store.get_active_users_for_date(date)
        logger.info(f"Found {len(users)} active user(s) for {date}")

    if not users:
        logger.info("No active users found, exiting")
        return

    for user_id in users:
        logger.info(f"{'='*60}")
        logger.info(f"Processing user: {user_id}")
        logger.info(f"{'='*60}")

        try:
            await _process_user(user_id, date, dry_run)
        except Exception as e:
            logger.error(f"Failed to process user {user_id}: {e}", exc_info=True)
            # Continue with next user — don't let one failure block others

    logger.info(f"Daily cron completed for {date}")


async def _process_user(user_id: str, date: str, dry_run: bool) -> None:
    """Run all workstreams for a single user."""
    client = TrackedAnthropicClient(
        session_id=f"daily-cron-{date}-{user_id[:8]}",
        agent_role="daily_cron",
        user_id=user_id,
    )

    # ── Workstream 1: Fact cross-validation + candidate key review ──

    logger.info(f"[{user_id}] Running fact validation...")
    if not dry_run:
        try:
            fact_job = DailyFactJob(client=client)
            fact_report = await fact_job.run(user_id=user_id, date=date)
            logger.info(
                f"[{user_id}] Fact validation complete: "
                f"deprecated={fact_report.facts_deprecated}, "
                f"flagged={fact_report.facts_flagged}, "
                f"aliases={fact_report.candidates_aliased}, "
                f"proposals={fact_report.candidates_proposed}, "
                f"rejected={fact_report.candidates_rejected}"
            )
        except Exception as e:
            logger.error(f"[{user_id}] Fact validation failed: {e}", exc_info=True)
    else:
        logger.info(f"[{user_id}] [DRY RUN] Would run fact validation")

    # ── Workstream 2: Request hygiene + status sync ──────────────

    logger.info(f"[{user_id}] Running request hygiene...")
    analyzer = DailyRequestAnalyzer(client=client)

    if not dry_run:
        try:
            hygiene_result = await analyzer.run_request_hygiene(
                user_id=user_id, date=date
            )
            logger.info(
                f"[{user_id}] Request hygiene complete: "
                f"actions_taken={hygiene_result.get('actions_taken', 0)}"
            )
        except Exception as e:
            logger.error(f"[{user_id}] Request hygiene failed: {e}", exc_info=True)
    else:
        logger.info(f"[{user_id}] [DRY RUN] Would run request hygiene")

    # ── Workstream 3: Request archival (existing flow) ───────────

    logger.info(f"[{user_id}] Running request archival...")
    if not dry_run:
        try:
            # Load conversation and requests for archival
            conv_store = get_conversation_store()
            conversations = await conv_store.get_messages_for_user_date(user_id, date)

            # Load requests from DDB
            active_requests = await analyzer._load_user_requests(user_id)
            requests_dict = {
                r.get("request_id", f"unknown-{i}"): r
                for i, r in enumerate(active_requests)
            }

            if conversations and requests_dict:
                daily_summary = await analyzer.analyze_daily_requests(
                    user_id=user_id,
                    date=date,
                    conversations=conversations,
                    active_requests=requests_dict,
                )

                # Archive to historical store
                try:
                    from historical_models import RequestCache
                    request_cache = RequestCache(
                        user_id=user_id,
                        date=date,
                        daily_requests=requests_dict,
                        daily_summary=daily_summary,
                    )
                    historical_store = HistoricalRequestStore()
                    await historical_store.archive_day(
                        user_id=user_id,
                        date=date,
                        request_cache=request_cache,
                        daily_analysis=daily_summary,
                    )
                    logger.info(
                        f"[{user_id}] Archival complete: "
                        f"{daily_summary.total_requests_completed_today} completed, "
                        f"{daily_summary.total_requests_pending} pending"
                    )
                except Exception as e:
                    logger.error(f"[{user_id}] Archival to historical store failed: {e}")
            else:
                logger.info(
                    f"[{user_id}] Skipping archival: "
                    f"conversations={len(conversations)}, requests={len(requests_dict)}"
                )
        except Exception as e:
            logger.error(f"[{user_id}] Request archival failed: {e}", exc_info=True)
    else:
        logger.info(f"[{user_id}] [DRY RUN] Would run request archival")


def main():
    parser = argparse.ArgumentParser(description="WithCare Daily Cron Job")
    parser.add_argument(
        "--date",
        help="Date to process (YYYY-MM-DD). Defaults to yesterday PT.",
    )
    parser.add_argument(
        "--user",
        help="Process only this user ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would happen without executing writes",
    )
    args = parser.parse_args()

    asyncio.run(
        run_daily_cron(
            target_date=args.date,
            target_user=args.user,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
