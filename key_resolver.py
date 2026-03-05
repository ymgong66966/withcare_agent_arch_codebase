"""
Key Resolver — maps natural-language fact descriptions to canonical FactKeys.

Architecture:
  1. Channel A: BM25 search over FactKey registry (key + desc + aliases + examples)
  2. Channel B: Alias table lookup (exact + prefix matching)
  3. Merge + dedup → Top-K candidates
  4. LLM selection (multiple-choice, NOT free generation)

The LLM can propose aliases (alias_observed from user text, alias_suggested
from its own creativity), but NEVER invents canonical keys.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import yaml

from models.fact_models import (
    AliasRecord,
    KeyCandidate,
    KeyResolverResult,
    RiskLevel,
)

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "configs", "factkey_registry_v0.yaml"
)

# LLM selection prompt — the LLM picks from Top-K, never invents keys.
KEY_RESOLVER_PROMPT = """\
You are a fact-key mapper for a caregiving platform. Select the best \
canonical key from the candidates below, or output "unknown" if none match.

DO NOT invent new keys. You may ONLY select from the candidates or return "unknown".

## Input fact
Entity: {entity_id}
Text: "{fact_text}"
Context: request_type={request_type}

## Candidates (top {top_k})
{candidates_block}

## Output (strict JSON, nothing else):
{{"selected_key": "...|unknown", "value_candidate": "...|null", \
"confidence": 0.0, "reason": "one sentence", \
"alias_observed": ["surface phrases from user text"], \
"alias_suggested": ["up to 2 alternative names"]}}"""


# ─────────────────────────────────────────────────────────────
# Registry Entry
# ─────────────────────────────────────────────────────────────

class RegistryEntry:
    """A single entry from the FactKey registry YAML."""

    __slots__ = (
        "key", "description", "risk_level", "value_type",
        "cardinality", "aliases", "examples", "canonicalization",
        "namespace", "facet",
    )

    def __init__(self, raw: Dict[str, Any]):
        self.key: str = raw["key"]
        self.description: str = raw.get("description", "")
        self.risk_level: RiskLevel = raw.get("risk_level", "medium")
        self.value_type: str = raw.get("value_type", "string")
        self.cardinality: str = raw.get("cardinality", "single")
        self.aliases: List[str] = raw.get("aliases", [])
        self.examples: List[str] = [str(e) for e in raw.get("examples", [])]
        self.canonicalization: str = raw.get("canonicalization", "")

        parts = self.key.split(".", 1)
        self.namespace: str = parts[0]
        self.facet: str = parts[1] if len(parts) > 1 else ""

    def search_text(self) -> str:
        """Concatenate all searchable text for BM25 indexing."""
        parts = [
            self.key,
            self.description,
            " ".join(self.aliases),
            " ".join(self.examples),
        ]
        return " ".join(parts).lower()


# ─────────────────────────────────────────────────────────────
# BM25 Index (lightweight, no external deps)
# ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    # Split on non-alphanumeric (but keep CJK characters)
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text)
    return tokens


class BM25Index:
    """
    Minimal BM25 implementation for searching the FactKey registry.

    No external dependencies — this is intentionally lightweight.
    For production, consider replacing with a proper search library.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.entries: List[RegistryEntry] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: Counter = Counter()  # term → number of docs containing term
        self.avg_dl: float = 0.0
        self.n_docs: int = 0

    def build(self, entries: List[RegistryEntry]) -> None:
        """Build the index from registry entries."""
        self.entries = entries
        self.doc_tokens = []
        self.doc_freqs = Counter()

        for entry in entries:
            tokens = _tokenize(entry.search_text())
            self.doc_tokens.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] += 1

        self.n_docs = len(entries)
        total_tokens = sum(len(dt) for dt in self.doc_tokens)
        self.avg_dl = total_tokens / max(self.n_docs, 1)

    def search(self, query: str, k: int = 12) -> List[Tuple[RegistryEntry, float]]:
        """Return top-k registry entries ranked by BM25 score."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = []
        for i, doc_tokens in enumerate(self.doc_tokens):
            score = self._score_doc(query_tokens, doc_tokens)
            scores.append(score)

        # Get top-k indices by score
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed_scores[:k]:
            if score > 0:
                results.append((self.entries[idx], score))
        return results

    def _score_doc(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Compute BM25 score for a single document."""
        dl = len(doc_tokens)
        tf_map: Counter = Counter(doc_tokens)
        score = 0.0

        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            if tf == 0:
                continue

            df = self.doc_freqs.get(qt, 0)
            # IDF with smoothing
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            # BM25 TF normalization
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
            score += idf * numerator / denominator

        return score


# ─────────────────────────────────────────────────────────────
# KeyRegistry — loads YAML, builds index, provides lookups
# ─────────────────────────────────────────────────────────────

class KeyRegistry:
    """
    Loads the FactKey registry from YAML and provides lookup/search methods.

    This is the "system knowledge" layer — it never changes during a session
    unless explicitly reloaded.
    """

    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.entries: List[RegistryEntry] = []
        self._by_key: Dict[str, RegistryEntry] = {}
        self._by_namespace: Dict[str, List[RegistryEntry]] = defaultdict(list)
        self._bm25 = BM25Index()
        self._loaded = False
        self._registry_path = registry_path

    def load(self) -> None:
        """Load registry from YAML file and build the BM25 index."""
        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                raw_entries = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Registry not found at {self._registry_path}, using empty")
            raw_entries = []

        self.entries = [RegistryEntry(r) for r in (raw_entries or [])]
        self._by_key = {e.key: e for e in self.entries}
        self._by_namespace = defaultdict(list)
        for e in self.entries:
            self._by_namespace[e.namespace].append(e)

        self._bm25.build(self.entries)
        self._loaded = True
        logger.info(f"Loaded {len(self.entries)} FactKey entries from registry")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get_entry(self, key: str) -> Optional[RegistryEntry]:
        """Look up a registry entry by canonical key."""
        self._ensure_loaded()
        return self._by_key.get(key)

    def get_risk(self, key: str) -> Optional[RiskLevel]:
        """Get the risk level for a canonical key. Returns None if key unknown."""
        entry = self.get_entry(key)
        if entry:
            return entry.risk_level
        # Fallback: infer from namespace
        namespace = key.split(".")[0] if "." in key else key
        ns_entries = self._by_namespace.get(namespace, [])
        if ns_entries:
            # Use the most common risk level in the namespace
            risks = [e.risk_level for e in ns_entries]
            return max(set(risks), key=risks.count)
        return None

    def search_bm25(self, query: str, k: int = 12) -> List[KeyCandidate]:
        """Search registry via BM25, returning KeyCandidate list."""
        self._ensure_loaded()
        results = self._bm25.search(query, k=k)
        return [
            KeyCandidate(
                key=entry.key,
                description=entry.description,
                risk_level=entry.risk_level,
                value_type=entry.value_type,
                score=score,
            )
            for entry, score in results
        ]

    def all_keys(self) -> List[str]:
        """Return all canonical keys."""
        self._ensure_loaded()
        return list(self._by_key.keys())

    def namespace_keys(self, namespace: str) -> List[str]:
        """Return all keys in a namespace."""
        self._ensure_loaded()
        return [e.key for e in self._by_namespace.get(namespace, [])]


# ─────────────────────────────────────────────────────────────
# KeyResolver — the main interface
# ─────────────────────────────────────────────────────────────

class KeyResolver:
    """
    Resolves natural-language fact descriptions to canonical FactKeys.

    Two-channel recall (BM25 + alias) → merge → LLM selection.
    """

    def __init__(
        self,
        registry: Optional[KeyRegistry] = None,
        llm_client: Optional[Any] = None,
    ):
        self.registry = registry or get_key_registry()
        self.llm_client = llm_client  # TrackedAnthropicClient instance

    async def resolve(
        self,
        fact_text: str,
        entity_id: str,
        context: Optional[Dict[str, str]] = None,
        top_k: int = 12,
    ) -> KeyResolverResult:
        """
        Resolve a fact description to a canonical FactKey.

        Args:
            fact_text: Natural language description of the fact
            entity_id: Entity this fact belongs to
            context: Optional context (request_type, language, etc.)
            top_k: Number of candidates to consider

        Returns:
            KeyResolverResult with selected key, confidence, aliases, etc.
        """
        context = context or {}

        # Channel A: BM25 search over registry
        candidates_a = self.registry.search_bm25(fact_text, k=top_k)

        # Channel B: Alias table lookup (currently returns empty until DDB is wired)
        # In production, this would query FactAliasTable
        candidates_b: List[KeyCandidate] = []

        # Merge and dedup
        merged = self._merge_candidates(candidates_a, candidates_b, max_k=top_k)

        if not merged:
            return KeyResolverResult(
                canonical_key="unknown",
                confidence=0.0,
                decision="unknown",
                reason="No matching candidates found in registry",
            )

        # If top candidate has very high score, skip LLM call
        if merged[0].score > 5.0 and len(merged) > 1:
            score_ratio = merged[0].score / max(merged[1].score, 0.01)
            if score_ratio > 3.0:
                entry = self.registry.get_entry(merged[0].key)
                return KeyResolverResult(
                    canonical_key=merged[0].key,
                    confidence=min(0.95, merged[0].score / 10.0),
                    risk_level=entry.risk_level if entry else "medium",
                    decision="map",
                    candidates=merged,
                    reason=f"High-confidence BM25 match (score={merged[0].score:.2f})",
                )

        # LLM selection from top-K candidates
        if self.llm_client:
            return await self._llm_select(
                fact_text=fact_text,
                entity_id=entity_id,
                context=context,
                candidates=merged,
            )

        # No LLM client — return best BM25 match with lower confidence
        entry = self.registry.get_entry(merged[0].key)
        return KeyResolverResult(
            canonical_key=merged[0].key,
            confidence=min(0.7, merged[0].score / 10.0),
            risk_level=entry.risk_level if entry else "medium",
            decision="map",
            candidates=merged,
            reason="BM25 match (no LLM available for disambiguation)",
        )

    def _merge_candidates(
        self,
        candidates_a: List[KeyCandidate],
        candidates_b: List[KeyCandidate],
        max_k: int = 12,
    ) -> List[KeyCandidate]:
        """Merge two candidate lists, dedup by key, rank by combined score."""
        by_key: Dict[str, KeyCandidate] = {}

        for c in candidates_a:
            if c.key not in by_key or c.score > by_key[c.key].score:
                by_key[c.key] = c

        for c in candidates_b:
            if c.key in by_key:
                # Boost score if found in both channels
                by_key[c.key].score = max(by_key[c.key].score, c.score) * 1.2
            else:
                by_key[c.key] = c

        ranked = sorted(by_key.values(), key=lambda x: x.score, reverse=True)
        return ranked[:max_k]

    async def _llm_select(
        self,
        fact_text: str,
        entity_id: str,
        context: Dict[str, str],
        candidates: List[KeyCandidate],
    ) -> KeyResolverResult:
        """Use LLM to select the best key from candidates."""
        # Build candidates block
        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}) {c.key} — {c.description} — "
                f"risk={c.risk_level} — type={c.value_type}"
            )
        candidates_block = "\n".join(lines)

        prompt = KEY_RESOLVER_PROMPT.format(
            entity_id=entity_id,
            fact_text=fact_text[:500],
            request_type=context.get("request_type", "unknown"),
            top_k=len(candidates),
            candidates_block=candidates_block,
        )

        try:
            raw = await self.llm_client.async_chat(prompt, max_tokens=300)
            result = json.loads(raw.strip())
        except Exception as e:
            logger.warning(f"Key resolver LLM call failed: {e}")
            # Fallback to top BM25 candidate
            entry = self.registry.get_entry(candidates[0].key)
            return KeyResolverResult(
                canonical_key=candidates[0].key,
                confidence=min(0.6, candidates[0].score / 10.0),
                risk_level=entry.risk_level if entry else "medium",
                decision="map",
                candidates=candidates,
                reason=f"LLM fallback to BM25 top match: {e}",
            )

        selected_key = result.get("selected_key", "unknown")

        # Validate that selected key exists in registry or is "unknown"
        entry = self.registry.get_entry(selected_key) if selected_key != "unknown" else None

        if selected_key != "unknown" and entry is None:
            # LLM hallucinated a key — fall back to top BM25
            logger.warning(
                f"Key resolver: LLM selected non-existent key '{selected_key}', "
                f"falling back to '{candidates[0].key}'"
            )
            entry = self.registry.get_entry(candidates[0].key)
            selected_key = candidates[0].key

        return KeyResolverResult(
            canonical_key=selected_key,
            confidence=result.get("confidence", 0.0),
            risk_level=entry.risk_level if entry else "medium",
            value_candidate=result.get("value_candidate"),
            decision="map" if selected_key != "unknown" else "unknown",
            candidates=candidates,
            alias_observed=result.get("alias_observed", []),
            alias_suggested=result.get("alias_suggested", []),
            reason=result.get("reason", ""),
        )

    async def resolve_batch(
        self,
        facts: List[Dict[str, str]],
        entity_id: str,
        context: Optional[Dict[str, str]] = None,
        top_k: int = 12,
    ) -> List[KeyResolverResult]:
        """
        Resolve multiple facts to canonical FactKeys in a single LLM call.

        Args:
            facts: List of dicts with at minimum 'fact_key', 'fact_label', 'evidence'
            entity_id: Entity these facts belong to
            context: Optional context (request_type, etc.)
            top_k: Number of BM25 candidates per fact

        Returns:
            List of KeyResolverResult, one per input fact (same order).
        """
        context = context or {}
        if not facts:
            return []

        # If no LLM client or only 1 fact, fall back to per-fact resolve
        if not self.llm_client or len(facts) <= 1:
            results = []
            for f in facts:
                fact_text = f"{f.get('fact_key', '')} {f.get('fact_label', '')} {f.get('evidence', '')}"
                results.append(await self.resolve(fact_text, entity_id, context, top_k))
            return results

        # ── BM25 search per fact (in-memory, fast) ──
        per_fact_candidates: List[List[KeyCandidate]] = []
        all_candidates_by_key: Dict[str, KeyCandidate] = {}

        for f in facts:
            fact_text = f"{f.get('fact_key', '')} {f.get('fact_label', '')} {f.get('evidence', '')}"
            candidates = self.registry.search_bm25(fact_text, k=top_k)
            per_fact_candidates.append(candidates)
            for c in candidates:
                if c.key not in all_candidates_by_key or c.score > all_candidates_by_key[c.key].score:
                    all_candidates_by_key[c.key] = c

        if not all_candidates_by_key:
            return [
                KeyResolverResult(
                    canonical_key="unknown", confidence=0.0,
                    decision="unknown", reason="No candidates found",
                )
                for _ in facts
            ]

        # ── Build combined candidates block ──
        sorted_candidates = sorted(all_candidates_by_key.values(), key=lambda x: x.score, reverse=True)[:top_k * 2]
        cand_lines = []
        for i, c in enumerate(sorted_candidates, 1):
            cand_lines.append(f"{i}) {c.key} — {c.description} — risk={c.risk_level} — type={c.value_type}")
        candidates_block = "\n".join(cand_lines)

        # ── Build batch facts block ──
        fact_lines = []
        for i, f in enumerate(facts, 1):
            fact_lines.append(
                f"Fact {i}: key=\"{f.get('fact_key', '')}\", "
                f"label=\"{f.get('fact_label', '')}\", "
                f"evidence=\"{f.get('evidence', '')[:200]}\""
            )
        facts_block = "\n".join(fact_lines)

        request_type = context.get("request_type", "unknown")

        batch_prompt = f"""\
You are a fact-key mapper for a caregiving platform. For EACH fact below, \
select the best canonical key from the candidates, or output "unknown" if none match.

DO NOT invent new keys. You may ONLY select from the candidates or return "unknown".

## Entity: {entity_id}
## Context: request_type={request_type}

## Candidates (combined pool)
{candidates_block}

## Facts to resolve
{facts_block}

## Output (strict JSON array, one entry per fact, same order):
[
  {{"fact_index": 1, "selected_key": "...|unknown", "confidence": 0.0, "reason": "one sentence"}},
  ...
]
Output ONLY the JSON array, nothing else."""

        try:
            raw = await self.llm_client.async_chat(batch_prompt, max_tokens=800)
            raw_text = raw.strip()
            json_start = raw_text.find("[")
            json_end = raw_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                batch_results = json.loads(raw_text[json_start:json_end])
            else:
                batch_results = json.loads(raw_text)
        except Exception as e:
            logger.warning(f"Batch key resolution LLM call failed: {e}, falling back to per-fact")
            results = []
            for f in facts:
                fact_text = f"{f.get('fact_key', '')} {f.get('fact_label', '')} {f.get('evidence', '')}"
                results.append(await self.resolve(fact_text, entity_id, context, top_k))
            return results

        # ── Parse batch results into KeyResolverResult list ──
        # Build index from batch_results for quick lookup
        result_map: Dict[int, Dict] = {}
        for item in batch_results:
            if isinstance(item, dict):
                idx = item.get("fact_index", 0) - 1  # 1-indexed → 0-indexed
                result_map[idx] = item

        results: List[KeyResolverResult] = []
        for i, f in enumerate(facts):
            item = result_map.get(i, {})
            selected_key = item.get("selected_key", "unknown")

            # Validate key exists in registry
            entry = self.registry.get_entry(selected_key) if selected_key != "unknown" else None
            if selected_key != "unknown" and entry is None:
                # LLM hallucinated — use top BM25 candidate for this fact
                if per_fact_candidates[i]:
                    selected_key = per_fact_candidates[i][0].key
                    entry = self.registry.get_entry(selected_key)
                else:
                    selected_key = "unknown"

            results.append(KeyResolverResult(
                canonical_key=selected_key,
                confidence=item.get("confidence", 0.0),
                risk_level=entry.risk_level if entry else "medium",
                decision="map" if selected_key != "unknown" else "unknown",
                candidates=per_fact_candidates[i] if i < len(per_fact_candidates) else [],
                reason=item.get("reason", "batch resolution"),
            ))

        return results


# ─────────────────────────────────────────────────────────────
# Module-level singletons
# ─────────────────────────────────────────────────────────────

_key_registry: Optional[KeyRegistry] = None
_key_resolver: Optional[KeyResolver] = None


def get_key_registry() -> KeyRegistry:
    """Get or create the global KeyRegistry instance."""
    global _key_registry
    if _key_registry is None:
        _key_registry = KeyRegistry()
        _key_registry.load()
    return _key_registry


def get_key_resolver(llm_client: Optional[Any] = None) -> KeyResolver:
    """Get or create the global KeyResolver instance."""
    global _key_resolver
    if _key_resolver is None:
        _key_resolver = KeyResolver(
            registry=get_key_registry(),
            llm_client=llm_client,
        )
    return _key_resolver
