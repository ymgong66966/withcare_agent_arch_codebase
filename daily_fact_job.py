"""
Daily Fact Job — orchestrates fact cross-validation and candidate key review.

Two sub-workflows:
  A) Fact Cross-Validation: check active facts against day's conversation
  B) Candidate Key Review: classify top candidate keys from CandidateKeyPool

Called by run_daily_cron.py for each active user.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from anthropic_client import TrackedAnthropicClient
from conversation_store import get_conversation_store
from fact_store import FactStore, get_fact_store
from candidate_key_pool import CandidateKeyPoolStore, get_candidate_key_pool_store
from event_store import get_event_store
from models.fact_models import AliasRecord
from daily_fact_models import (
    AliasMapping,
    DailyFactReport,
    DeprecatedFact,
    FlaggedFact,
    ProposedNewKey,
    RejectedCandidate,
)
from daily_fact_prompts import (
    llm_daily_fact_validation,
    llm_candidate_key_review,
)

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "configs", "factkey_registry_v0.yaml"
)


def _load_registry_summary() -> str:
    """Load the FactKey registry and produce a compact summary for prompts."""
    try:
        with open(REGISTRY_PATH, "r") as f:
            registry = yaml.safe_load(f) or []

        namespaces: Dict[str, List[str]] = {}
        for entry in registry:
            key = entry.get("key", "")
            if "." in key:
                ns = key.split(".")[0]
                namespaces.setdefault(ns, []).append(key)

        lines = []
        for ns, keys in sorted(namespaces.items()):
            lines.append(f"  {ns}: {', '.join(keys)}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to load registry: {e}")
        return "(registry unavailable)"


class DailyFactJob:
    """
    Orchestrates daily fact validation and candidate key review for a user.

    Usage:
        job = DailyFactJob(client)
        report = await job.run(user_id="user-123", date="2026-03-02")
    """

    def __init__(self, client: Optional[TrackedAnthropicClient] = None):
        self.client = client
        self.fact_store: FactStore = get_fact_store()
        self.candidate_pool: CandidateKeyPoolStore = get_candidate_key_pool_store()
        self.conversation_store = get_conversation_store()
        self.event_store = get_event_store()
        self._registry_summary: Optional[str] = None

    def _get_registry_summary(self) -> str:
        if self._registry_summary is None:
            self._registry_summary = _load_registry_summary()
        return self._registry_summary

    def _ensure_client(self, user_id: str, date: str) -> TrackedAnthropicClient:
        if self.client is None:
            self.client = TrackedAnthropicClient(
                session_id=f"daily-fact-{date}",
                agent_role="daily_fact_job",
                user_id=user_id,
            )
        return self.client

    async def run(self, user_id: str, date: str) -> DailyFactReport:
        """
        Main entry point for daily fact validation.

        Steps:
          1. Fetch day's conversation from ConversationStore
          2. Fetch all active facts from FactStore
          3. Run fact cross-validation (Sub-workflow A)
          4. Execute deprecations + log flags
          5. Fetch top candidates from CandidateKeyPool
          6. Run candidate key review (Sub-workflow B)
          7. Execute alias writes + rejections
          8. Return report
        """
        client = self._ensure_client(user_id, date)
        registry_summary = self._get_registry_summary()

        report = DailyFactReport(user_id=user_id, date=date)

        # ── Sub-workflow A: Fact Cross-Validation ──────────────────

        # 1. Fetch day's conversation
        conversation = await self.conversation_store.get_messages_for_user_date(
            user_id=user_id, date_str=date
        )

        # 2. Fetch all active facts
        active_facts = await self.fact_store.get_all_active_facts_for_user(user_id)
        active_facts_dicts = [
            f.model_dump(mode="json") for f in active_facts
        ]

        if active_facts_dicts and conversation:
            # 3. Run LLM fact validation
            validation_result = await self._run_fact_validation(
                client, active_facts_dicts, conversation, registry_summary
            )

            # 4. Execute actions
            deprecated, flagged = await self._execute_fact_actions(
                user_id, validation_result, active_facts
            )
            report.facts_deprecated = len(deprecated)
            report.facts_flagged = len(flagged)
            report.deprecated_details = deprecated
            report.flagged_details = flagged
        else:
            logger.info(
                f"Skipping fact validation for {user_id} on {date}: "
                f"facts={len(active_facts_dicts)}, messages={len(conversation)}"
            )

        # ── Sub-workflow B: Candidate Key Review ───────────────────

        # 5. Fetch top candidates
        candidates = await self.candidate_pool.get_top_candidates(min_count=3)
        candidate_dicts = [c.model_dump(mode="json") for c in candidates]

        if candidate_dicts:
            # 6. Run LLM candidate review
            review_result = await self._run_candidate_review(
                client, candidate_dicts, registry_summary
            )

            # 7. Execute actions
            aliased, proposed, rejected = await self._execute_candidate_actions(
                user_id, review_result
            )
            report.candidates_aliased = len(aliased)
            report.candidates_proposed = len(proposed)
            report.candidates_rejected = len(rejected)
            report.alias_details = aliased
            report.proposed_details = proposed
            report.rejected_details = rejected
        else:
            logger.info("No candidate keys with count >= 3 to review")

        return report

    async def _run_fact_validation(
        self,
        client: TrackedAnthropicClient,
        active_facts: List[Dict[str, Any]],
        conversation: List[Dict[str, Any]],
        registry_summary: str,
    ) -> Dict[str, Any]:
        """Run LLM-powered fact cross-validation."""
        try:
            return await llm_daily_fact_validation(
                client=client,
                active_facts=active_facts,
                conversation=conversation,
                registry_summary=registry_summary,
            )
        except Exception as e:
            logger.error(f"Fact validation LLM call failed: {e}")
            return {"deprecate": [], "flag_for_review": []}

    async def _execute_fact_actions(
        self,
        user_id: str,
        validation_result: Dict[str, Any],
        active_facts: list,
    ) -> tuple:
        """
        Execute deprecations and log flagged facts.

        Returns (deprecated_list, flagged_list).
        """
        deprecated: List[DeprecatedFact] = []
        flagged: List[FlaggedFact] = []

        # Build lookup: (entity_id, fact_key) -> FactRecord
        fact_lookup = {}
        for f in active_facts:
            fact_lookup[(f.entity_id, f.fact_key)] = f

        # Process deprecations
        for item in validation_result.get("deprecate", []):
            entity_id = item.get("entity_id", "")
            fact_key = item.get("fact_key", "")
            reason = item.get("reason", "")

            fact = fact_lookup.get((entity_id, fact_key))
            if fact:
                try:
                    await self.fact_store.deprecate_fact(fact)
                    deprecated.append(DeprecatedFact(
                        entity_id=entity_id,
                        fact_key=fact_key,
                        reason=reason,
                    ))
                    # Audit log
                    await self.fact_store.write_fact_log(
                        user_id=user_id,
                        target_id=fact.fact_id,
                        patch={"status": "deprecated"},
                        justification=f"Daily validation: {reason}",
                        actor="daily_fact_job",
                        result="applied",
                    )
                    logger.info(f"Deprecated fact: {entity_id}/{fact_key} — {reason}")
                except Exception as e:
                    logger.error(f"Failed to deprecate {entity_id}/{fact_key}: {e}")
            else:
                logger.warning(
                    f"Deprecation target not found: {entity_id}/{fact_key}"
                )

        # Process flags
        for item in validation_result.get("flag_for_review", []):
            entity_id = item.get("entity_id", "")
            fact_key = item.get("fact_key", "")
            concern = item.get("concern", "")
            severity = item.get("severity", "medium")

            flagged.append(FlaggedFact(
                entity_id=entity_id,
                fact_key=fact_key,
                concern=concern,
                severity=severity,
            ))

            # Write to audit log with result="needs_review"
            fact = fact_lookup.get((entity_id, fact_key))
            target_id = fact.fact_id if fact else f"{entity_id}/{fact_key}"
            try:
                await self.fact_store.write_fact_log(
                    user_id=user_id,
                    target_id=target_id,
                    patch={"flagged": True, "concern": concern, "severity": severity},
                    justification=f"Daily validation flag: {concern}",
                    actor="daily_fact_job",
                    result="needs_confirm",
                )
            except Exception as e:
                logger.error(f"Failed to write flag log: {e}")

        return deprecated, flagged

    async def _run_candidate_review(
        self,
        client: TrackedAnthropicClient,
        candidates: List[Dict[str, Any]],
        registry_summary: str,
    ) -> Dict[str, Any]:
        """Run LLM-powered candidate key review."""
        try:
            return await llm_candidate_key_review(
                client=client,
                candidates=candidates,
                registry_summary=registry_summary,
            )
        except Exception as e:
            logger.error(f"Candidate review LLM call failed: {e}")
            return {"alias_mappings": [], "proposed_new_keys": [], "reject": []}

    async def _execute_candidate_actions(
        self,
        user_id: str,
        review_result: Dict[str, Any],
    ) -> tuple:
        """
        Execute alias writes, key proposals, and rejections.

        Returns (aliased_list, proposed_list, rejected_list).
        """
        aliased: List[AliasMapping] = []
        proposed: List[ProposedNewKey] = []
        rejected: List[RejectedCandidate] = []

        # Process alias mappings
        for item in review_result.get("alias_mappings", []):
            candidate_key = item.get("candidate_key", "")
            canonical_key = item.get("canonical_key", "")
            suggested = item.get("suggested_aliases", [])

            if not candidate_key or not canonical_key:
                continue

            # Write alias to FactAliasTable
            all_aliases = [candidate_key] + suggested
            for alias_text in all_aliases:
                try:
                    alias_record = AliasRecord(
                        normalized_alias=alias_text.strip().lower(),
                        canonical_key=canonical_key,
                        scope="global",
                        confidence=0.8,
                        status="active",
                    )
                    await self.fact_store.put_alias(alias_record)
                except Exception as e:
                    logger.error(f"Failed to write alias {alias_text}: {e}")

            aliased.append(AliasMapping(
                candidate_key=candidate_key,
                canonical_key=canonical_key,
                suggested_aliases=suggested,
            ))

            # Update candidate status
            try:
                await self._update_candidate_status(candidate_key, "promoted")
            except Exception as e:
                logger.warning(f"Failed to update candidate status: {e}")

        # Process proposed new keys — write as events for review
        for item in review_result.get("proposed_new_keys", []):
            key = item.get("key", "")
            if not key:
                continue

            proposed.append(ProposedNewKey(
                key=key,
                description=item.get("description", ""),
                risk_level=item.get("risk_level", "medium"),
                value_type=item.get("value_type", "string"),
                namespace=item.get("namespace", ""),
                reason=item.get("reason", ""),
            ))

            # Log as event for human review
            try:
                await self.event_store.add_event(
                    user_id=user_id,
                    event_type="decision",
                    content=f"Key proposal: {key} — {item.get('description', '')}",
                    structured={
                        "type": "key_proposal",
                        **item,
                    },
                    tags=["daily_cron", "key_proposal"],
                    ttl_days=90,
                )
            except Exception as e:
                logger.error(f"Failed to log key proposal: {e}")

        # Process rejections
        for item in review_result.get("reject", []):
            candidate_key = item.get("candidate_key", "")
            reason = item.get("reason", "")

            if not candidate_key:
                continue

            rejected.append(RejectedCandidate(
                candidate_key=candidate_key,
                reason=reason,
            ))

            try:
                await self._update_candidate_status(candidate_key, "rejected")
            except Exception as e:
                logger.warning(f"Failed to reject candidate {candidate_key}: {e}")

        return aliased, proposed, rejected

    async def _update_candidate_status(
        self, candidate_key: str, new_status: str
    ) -> None:
        """Update a candidate key's status in the pool table."""
        if not self.candidate_pool.dynamodb:
            return

        try:
            from boto3.dynamodb.conditions import Key
            import re

            normalized = candidate_key.strip().lower()
            normalized = re.sub(r"[\s\-]+", "_", normalized)
            normalized = re.sub(r"_+", "_", normalized).strip("_")

            table = self.candidate_pool._table()
            table.update_item(
                Key={"pk": f"CKEY#{normalized}", "sk": "META"},
                UpdateExpression="SET #s = :s",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":s": new_status},
            )
        except Exception as e:
            logger.warning(f"Failed to update candidate {candidate_key} status: {e}")
