"""
Daily Request Analyzer

这个模块实现每天结束时的retrospective analysis：
- 分析当天的所有requests
- 判断哪些completed、哪些pending
- 生成historical records
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from anthropic_client import TrackedAnthropicClient
from historical_models import (
    HistoricalRequestRecord,
    RequestSummaryForDay,
    DailyRequestSummary,
    StageTransition,
    CompletionStatus,
    UserSatisfaction,
)
from daily_request_prompts import llm_daily_request_hygiene
from request_factory import build_request_update_patch
from conversation_store import get_conversation_store
from fact_store import get_fact_store
from ddb_client import get_ddb_resource, USER_REQUEST_TABLE

logger = logging.getLogger(__name__)


class DailyRequestAnalyzer:
    """
    每天结束时分析当天的requests

    核心功能：
    1. 对每个request做retrospective分析（completed? satisfied?）
    2. 生成HistoricalRequestRecord（用于vector search）
    3. 生成DailyRequestSummary
    """

    def __init__(self, client: Optional[TrackedAnthropicClient] = None):
        self.client = client

    async def analyze_daily_requests(
        self,
        user_id: str,
        date: str,  # "2026-01-28"
        conversations: List[Dict[str, Any]],  # ChatMessages
        active_requests: Dict[str, Dict[str, Any]],  # request_id -> RequestRecord dict
        conversation_id: str = "unknown",
    ) -> DailyRequestSummary:
        """
        分析当天的所有requests和conversations

        Args:
            user_id: User ID
            date: Date string "YYYY-MM-DD"
            conversations: All messages from the day
            active_requests: All requests that were active during the day
            conversation_id: Conversation ID

        Returns:
            DailyRequestSummary with retrospective analysis
        """
        if not self.client:
            # Initialize client if not provided
            self.client = TrackedAnthropicClient(
                session_id=f"daily-analysis-{date}",
                agent_role="daily_analyzer",
                user_id=user_id,
            )

        # 1. 先做overall day analysis
        overall_analysis = await self._analyze_overall_day(
            conversations=conversations,
            requests=active_requests,
        )

        # 2. 对每个request做individual analysis
        request_summaries = []
        for req_id, req_dict in active_requests.items():
            summary = await self._analyze_single_request(
                user_id=user_id,
                conversation_id=conversation_id,
                date=date,
                request_id=req_id,
                request_dict=req_dict,
                conversations=conversations,
                overall_insights=overall_analysis,
            )
            request_summaries.append(summary)

        # 3. 统计
        total_completed = sum(
            1 for s in request_summaries
            if s.historical_record.completion_status == "completed"
        )
        total_pending = sum(
            1 for s in request_summaries
            if s.historical_record.completion_status == "pending"
        )

        return DailyRequestSummary(
            user_id=user_id,
            date=date,
            analyzed_at=datetime.utcnow(),
            request_summaries=request_summaries,
            overall_summary=overall_analysis.get("day_summary", ""),
            total_requests_active=len(active_requests),
            total_requests_completed_today=total_completed,
            total_requests_pending=total_pending,
            main_themes_today=overall_analysis.get("main_themes", []),
            total_conversation_turns=len(conversations),
        )

    async def _analyze_overall_day(
        self,
        conversations: List[Dict[str, Any]],
        requests: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        LLM分析整天的对话和requests

        Returns overview insights that help with individual request analysis
        """
        # Format conversations
        conv_text = "\n".join([
            f"[{msg.get('role', '').upper()}]: {msg.get('content', '')[:200]}"
            for msg in conversations[-50:]  # Last 50 turns
        ])

        # Format requests
        requests_text = "\n".join([
            f"- {req.get('name', 'Unknown')} (ID: {rid}): {req.get('goal', '')[:100]}"
            for rid, req in requests.items()
        ])

        prompt = f"""You are analyzing a day's worth of conversations and requests for a caregiver assistant system.

## Requests active today:
{requests_text}

## Conversations today (last 50 turns):
{conv_text}

## Your task:
Provide a high-level overview of today's activity:

1. **day_summary**: 2-3 sentence summary of what happened today
2. **main_themes**: List of main topics/themes (e.g., ["medicaid_application", "in_home_care", "legal_advice"])
3. **user_emotional_state**: Overall emotional state of the user (e.g., "anxious", "calm", "frustrated")
4. **key_transitions**: Any major topic switches or request transitions you noticed

## Output Format (JSON):
{{
  "day_summary": "User mainly focused on...",
  "main_themes": ["theme1", "theme2"],
  "user_emotional_state": "emotional_state",
  "key_transitions": [
    {{"from": "request_id_or_topic", "to": "request_id_or_topic", "reason": "why"}}
  ]
}}

Respond ONLY with the JSON object, no other text."""

        try:
            response = await self.client.async_chat(
                prompt=prompt,
                max_tokens=800,
                temperature=0.2,
            )

            # Parse JSON
            response_text = response.strip()
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(response_text)

        except Exception as e:
            logger.warning(f"Overall day analysis failed: {e}. Using fallback.")
            return {
                "day_summary": "Analysis unavailable",
                "main_themes": [],
                "user_emotional_state": "unknown",
                "key_transitions": [],
            }

    async def _analyze_single_request(
        self,
        user_id: str,
        conversation_id: str,
        date: str,
        request_id: str,
        request_dict: Dict[str, Any],
        conversations: List[Dict[str, Any]],
        overall_insights: Dict[str, Any],
    ) -> RequestSummaryForDay:
        """
        对单个request做retrospective分析

        LLM回答：
        1. 这个request完成了吗？
        2. 最后的stage是什么？
        3. User对结果满意吗？
        4. Short summary是什么？
        5. Theme是什么？
        """
        # Extract stage history from request
        stage_history = self._extract_stage_history(request_dict)

        # Extract related conversations
        related_convs = self._extract_related_conversations(
            conversations, request_dict
        )

        # Build prompt
        prompt = self._build_single_request_analysis_prompt(
            request_dict=request_dict,
            stage_history=stage_history,
            related_conversations=related_convs,
            overall_insights=overall_insights,
        )

        try:
            response = await self.client.async_chat(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.2,
            )

            # Parse JSON
            response_text = response.strip()
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = json.loads(response_text)

        except Exception as e:
            logger.warning(f"Single request analysis failed for {request_id}: {e}. Using fallback.")
            result = self._fallback_analysis(request_dict)

        # Build HistoricalRequestRecord
        historical_record = HistoricalRequestRecord(
            request_id=request_id,
            user_id=user_id,
            conversation_id=conversation_id,
            name=request_dict.get("name", "Unknown Request"),
            goal=request_dict.get("goal", ""),
            created_at=request_dict.get("created_at", datetime.utcnow()),
            analyzed_at=datetime.utcnow(),
            completion_status=result.get("completion_status", "uncertain"),
            final_stage=result.get("final_stage", "unknown"),
            user_satisfaction=result.get("user_satisfaction", "unknown"),
            short_summary=result.get("short_summary", "No summary available"),
            theme=result.get("theme", "general"),
            keywords=result.get("keywords", []),
            stage_history=stage_history,
            final_state_snapshot=request_dict,
            completion_confidence=result.get("confidence", 0.5),
            analysis_reason=result.get("reason", ""),
        )

        # Build RequestSummaryForDay
        return RequestSummaryForDay(
            request_id=request_id,
            historical_record=historical_record,
            turns_in_day=len(related_convs),
            stages_visited_today=result.get("stages_visited_today", []),
            key_events_today=result.get("key_events_today", []),
            progress_assessment=result.get("progress_assessment", ""),
        )

    def _build_single_request_analysis_prompt(
        self,
        request_dict: Dict[str, Any],
        stage_history: List[StageTransition],
        related_conversations: List[Dict[str, Any]],
        overall_insights: Dict[str, Any],
    ) -> str:
        """构建单个request分析的prompt"""

        conv_text = "\n".join([
            f"[{msg.get('role', '').upper()}]: {msg.get('content', '')[:200]}"
            for msg in related_conversations[-20:]  # Last 20 related turns
        ])

        stage_text = "\n".join([
            f"- {s.timestamp.strftime('%H:%M')}: {str(s)}"
            for s in stage_history
        ])

        return f"""Analyze this request retrospectively to determine its completion status and outcome.

## Request Details:
- Name: {request_dict.get('name', 'Unknown')}
- Goal: {request_dict.get('goal', 'Unknown')}
- Status: {request_dict.get('status', 'unknown')}
- Created: {request_dict.get('created_at', 'Unknown')}

## Stage History:
{stage_text if stage_text else "No stage transitions recorded"}

## Related Conversations:
{conv_text if conv_text else "No related conversations found"}

## Overall Day Context:
{overall_insights.get('day_summary', 'N/A')}

## Your Task:
Analyze this request and determine:

1. **completion_status**: Is this request completed, abandoned, pending, or uncertain?
   - "completed": User got what they needed, conversation moved on naturally
   - "abandoned": User explicitly stopped or showed dissatisfaction
   - "pending": Request is still in progress, waiting for user or next step
   - "uncertain": Can't determine from available information

2. **final_stage**: What was the last stage this request was in?
   (e.g., "info_collection", "deep_search", "domain_expert")

3. **user_satisfaction**: How satisfied was the user with the outcome?
   - "satisfied": User expressed satisfaction or proceeded happily
   - "unsatisfied": User expressed dissatisfaction or frustration
   - "neutral": No clear signal
   - "unknown": Can't determine

4. **short_summary**: 1-2 sentences summarizing what happened with this request

5. **theme**: Categorize this request (e.g., "medicaid_application", "in_home_care", "legal_advice", "emotional_support")

6. **keywords**: List 3-5 keywords for search (e.g., ["medicaid", "application", "illinois"])

7. **confidence**: 0.0-1.0, how confident are you in the completion_status?

8. **stages_visited_today**: List of stages this request went through today

9. **key_events_today**: List of key events/milestones for this request today

10. **progress_assessment**: Brief assessment of progress made today

## Output Format (JSON):
{{
  "completion_status": "completed|abandoned|pending|uncertain",
  "final_stage": "stage_name",
  "user_satisfaction": "satisfied|unsatisfied|neutral|unknown",
  "short_summary": "User requested X, we did Y, outcome was Z",
  "theme": "request_category",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "confidence": 0.85,
  "reason": "explanation of your judgment",
  "stages_visited_today": ["stage1", "stage2"],
  "key_events_today": ["event1", "event2"],
  "progress_assessment": "Made good progress on..."
}}

Respond ONLY with the JSON object, no other text."""

    def _extract_stage_history(
        self, request_dict: Dict[str, Any]
    ) -> List[StageTransition]:
        """从request dict提取stage history"""
        # Check if stage_history exists in the request
        if "stage_history" in request_dict:
            return [
                StageTransition(**s) if isinstance(s, dict) else s
                for s in request_dict["stage_history"]
            ]

        # Otherwise infer from status and other fields
        transitions = []

        # Initial creation
        if "created_at" in request_dict:
            transitions.append(
                StageTransition(
                    from_stage=None,
                    to_stage="created",
                    agent="system",
                    timestamp=request_dict["created_at"],
                    reason="Request created",
                )
            )

        # Current status
        current_status = request_dict.get("status", "unknown")
        if current_status != "created":
            transitions.append(
                StageTransition(
                    from_stage="created",
                    to_stage=current_status,
                    agent="unknown",
                    timestamp=request_dict.get("last_touched_at", datetime.utcnow()),
                    reason=f"Status: {current_status}",
                )
            )

        return transitions

    def _extract_related_conversations(
        self,
        conversations: List[Dict[str, Any]],
        request_dict: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        提取与这个request相关的对话

        策略：
        1. 使用request的时间范围（created_at到last_touched_at）
        2. 从stage_history中提取涉及的agents
        3. 找出这些agents的消息以及前后的user消息
        4. 返回连续的对话片段

        Args:
            conversations: 全天的对话列表
            request_dict: Request的完整信息

        Returns:
            与这个request相关的对话列表
        """
        if not conversations:
            return []

        # 1. 提取时间范围
        created_at = self._to_datetime(request_dict.get("created_at"))
        last_touched_at = self._to_datetime(request_dict.get("last_touched_at"))

        if not created_at:
            # Fallback: return all conversations
            return conversations

        # 如果没有last_touched_at，使用当前时间
        if not last_touched_at:
            last_touched_at = datetime.utcnow()

        # 2. 从stage_history中提取涉及的agents
        stage_history = request_dict.get("stage_history", [])
        involved_agents = set()

        for stage in stage_history:
            agent = stage.get("agent")
            if agent:
                involved_agents.add(agent)

        # 如果没有stage_history，尝试从其他地方推断
        if not involved_agents:
            # 从info_collection_state推断
            if request_dict.get("info_collection_state"):
                involved_agents.add("info_collection")
            # 从deep_search_state推断
            if request_dict.get("deep_search_state"):
                involved_agents.add("deep_search")

        # 3. 找出相关的消息
        related_indices = set()

        for i, msg in enumerate(conversations):
            msg_time = self._to_datetime(msg.get("timestamp") or msg.get("ts"))

            # 跳过没有时间戳的消息
            if not msg_time:
                continue

            # 检查是否在时间范围内
            # 稍微放宽时间范围（前后各5分钟），以捕获上下文
            time_buffer = timedelta(minutes=5)
            if not (created_at - time_buffer <= msg_time <= last_touched_at + time_buffer):
                continue

            # 如果是assistant消息，检查agent是否匹配
            if msg.get("role") == "assistant":
                msg_agent = msg.get("agent") or msg.get("metadata", {}).get("agent")
                if msg_agent in involved_agents:
                    related_indices.add(i)
                    # 包含前一条用户消息（触发），但也要检查时间范围
                    if i > 0:
                        prev_msg = conversations[i - 1]
                        prev_time = self._to_datetime(prev_msg.get("timestamp") or prev_msg.get("ts"))
                        if prev_time and (created_at - time_buffer <= prev_time <= last_touched_at + time_buffer):
                            related_indices.add(i - 1)
                    # 包含后一条用户消息（回应），但也要检查时间范围
                    if i + 1 < len(conversations):
                        next_msg = conversations[i + 1]
                        next_time = self._to_datetime(next_msg.get("timestamp") or next_msg.get("ts"))
                        if next_time and (created_at - time_buffer <= next_time <= last_touched_at + time_buffer):
                            related_indices.add(i + 1)

            # 如果是user消息且夹在两个相关消息之间，也包含
            elif msg.get("role") == "user":
                if i > 0 and (i - 1) in related_indices:
                    related_indices.add(i)
                if i + 1 < len(conversations) and (i + 1) in related_indices:
                    related_indices.add(i)

        # 4. 提取相关消息并保持顺序
        if not related_indices:
            # Fallback: 至少返回时间范围内的所有消息
            related = []
            for msg in conversations:
                msg_time = self._to_datetime(msg.get("timestamp") or msg.get("ts"))
                if msg_time and created_at <= msg_time <= last_touched_at:
                    related.append(msg)
            return related if related else conversations[-10:]  # 最后10条作为fallback

        # 按索引顺序提取
        sorted_indices = sorted(related_indices)
        related_conversations = [conversations[i] for i in sorted_indices]

        return related_conversations

    @staticmethod
    def _to_datetime(val: Any) -> Optional[datetime]:
        """Convert a value to datetime, handling ISO strings from DDB."""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val)
            except (ValueError, TypeError):
                return None
        return None

    def _fallback_analysis(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        status = request_dict.get("status", "unknown")

        # Simple heuristic
        if status in ["completed", "aborted"]:
            completion_status = "completed" if status == "completed" else "abandoned"
        elif status in ["paused", "blocked"]:
            completion_status = "pending"
        else:
            completion_status = "uncertain"

        return {
            "completion_status": completion_status,
            "final_stage": status,
            "user_satisfaction": "unknown",
            "short_summary": f"Request '{request_dict.get('name', 'Unknown')}' - status: {status}",
            "theme": "general",
            "keywords": [],
            "confidence": 0.3,
            "reason": "Fallback analysis due to LLM failure",
            "stages_visited_today": [status],
            "key_events_today": [],
            "progress_assessment": "Unknown progress",
        }

    # ─────────────────────────────────────────────────────────
    # Request Hygiene (WS-3B)
    # ─────────────────────────────────────────────────────────

    async def run_request_hygiene(
        self,
        user_id: str,
        date: str,
    ) -> Dict[str, Any]:
        """
        End-of-day request hygiene pass.

        1. Load all requests for the user from DDB
        2. Load day's conversation from ConversationStore
        3. LLM analyzes each request against conversation
        4. Update DDB with corrected statuses and summaries

        Returns a summary dict with counts of actions taken.
        """
        if not self.client:
            self.client = TrackedAnthropicClient(
                session_id=f"request-hygiene-{date}",
                agent_role="request_hygiene",
                user_id=user_id,
            )

        # 1. Load requests from DDB
        requests = await self._load_user_requests(user_id)
        if not requests:
            logger.info(f"No requests found for {user_id}, skipping hygiene")
            return {"actions_taken": 0}

        # 2. Load day's conversation
        conv_store = get_conversation_store()
        conversation = await conv_store.get_messages_for_user_date(user_id, date)

        if not conversation:
            logger.info(f"No conversation found for {user_id} on {date}")
            return {"actions_taken": 0}

        # 3. LLM analysis
        try:
            hygiene_result = await llm_daily_request_hygiene(
                client=self.client,
                requests=requests,
                conversation=conversation,
            )
        except Exception as e:
            logger.error(f"Request hygiene LLM call failed: {e}")
            return {"actions_taken": 0, "error": str(e)}

        # 4. Execute recommended actions
        actions_taken = 0
        action_details = []

        for analysis in hygiene_result.get("request_analyses", []):
            request_id = analysis.get("request_id", "")
            if not request_id:
                continue

            true_status = analysis.get("true_status", "")
            action = analysis.get("recommended_action", "")
            updated_summary = analysis.get("updated_summary", "")
            left_off_at = analysis.get("left_off_at", "")

            updates: Dict[str, Any] = {}

            if action == "close":
                # Map true_status to DDB status
                if true_status == "ad_hoc_resolved":
                    updates["status"] = "completed"
                    updates["stage_detail"] = "ad_hoc_resolved_by_daily_hygiene"
                elif true_status == "completed":
                    updates["status"] = "completed"
                    updates["stage_detail"] = "completed_by_daily_hygiene"
                elif true_status == "abandoned":
                    updates["status"] = "aborted"
                    updates["stage_detail"] = "abandoned_detected_by_daily_hygiene"

            elif action == "archive":
                updates["status"] = "completed"
                updates["stage_detail"] = "archived_by_daily_hygiene"

            # Always update summary and left_off_at if provided
            if updated_summary:
                updates["summary_current"] = updated_summary
            if left_off_at:
                updates["left_off_at"] = left_off_at

            if updates:
                try:
                    self._execute_ddb_update(user_id, request_id, updates)
                    actions_taken += 1
                    action_details.append({
                        "request_id": request_id,
                        "action": action,
                        "true_status": true_status,
                        "updates": list(updates.keys()),
                    })

                    # Audit log
                    fact_store = get_fact_store()
                    await fact_store.write_fact_log(
                        user_id=user_id,
                        target_id=request_id,
                        patch=updates,
                        justification=f"Daily hygiene: {analysis.get('reason', '')}",
                        actor="daily_request_hygiene",
                        target="request",
                        result="applied",
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to update request {request_id}: {e}"
                    )

        logger.info(
            f"Request hygiene for {user_id} on {date}: "
            f"{actions_taken} actions taken"
        )

        return {
            "actions_taken": actions_taken,
            "details": action_details,
        }

    async def _load_user_requests(
        self, user_id: str
    ) -> List[Dict[str, Any]]:
        """Load all requests for a user from DDB UserRequestTable."""
        ddb = get_ddb_resource()
        if not ddb:
            return []

        try:
            from boto3.dynamodb.conditions import Key

            table = ddb.Table(USER_REQUEST_TABLE)
            response = table.query(
                KeyConditionExpression=(
                    Key("pk").eq(f"USER#{user_id}")
                    & Key("sk").begins_with("REQ#")
                ),
            )

            requests = []
            for item in response.get("Items", []):
                # Extract the payload or use the item directly
                payload = item.get("payload", item)
                if isinstance(payload, dict):
                    requests.append(payload)
            return requests

        except Exception as e:
            logger.error(f"Failed to load requests for {user_id}: {e}")
            return []

    def _execute_ddb_update(
        self,
        user_id: str,
        request_id: str,
        updates: Dict[str, Any],
    ) -> None:
        """Execute a DDB update for a request."""
        ddb = get_ddb_resource()
        if not ddb:
            logger.warning("DDB not available, skipping request update")
            return

        patch = build_request_update_patch(
            user_id=user_id,
            request_id=request_id,
            updates=updates,
        )

        for write in patch.get("ddb_writes", []):
            if write.get("op") == "update":
                table = ddb.Table(write.get("table", USER_REQUEST_TABLE))
                table.update_item(**write["params"])
