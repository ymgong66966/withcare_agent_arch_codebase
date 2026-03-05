"""
Historical Request Store

这个模块实现：
1. Historical requests的持久化存储
2. Vector search (Milvus/Zilliz)
3. 检索接口
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from historical_models import (
    HistoricalRequestRecord,
    DailyRequestSummary,
    RequestCache,
    VectorSearchResult,
    SearchQuery,
)

logger = logging.getLogger(__name__)


class HistoricalRequestStore:
    """
    Historical requests的存储和检索

    存储层级：
    1. DynamoDB: HistoricalRequestRecord的完整记录
    2. Milvus/Zilliz: Embeddings for vector search
    3. (可选) S3: 归档的daily summaries
    """

    def __init__(
        self,
        dynamodb_client: Optional[Any] = None,
        milvus_client: Optional[Any] = None,
        embedding_client: Optional[Any] = None,
    ):
        self.dynamodb = dynamodb_client
        self.milvus = milvus_client
        self.embedding_client = embedding_client

    async def archive_day(
        self,
        user_id: str,
        date: str,
        request_cache: RequestCache,
        daily_analysis: DailyRequestSummary,
    ) -> None:
        """
        每天结束时归档当天的requests

        Workflow:
        1. 存储每个HistoricalRequestRecord到DynamoDB
        2. 生成embeddings并存储到Milvus
        3. 存储DailyRequestSummary
        """
        logger.info(f"Archiving day {date} for user {user_id}")

        # 1. Store each historical request
        for summary in daily_analysis.request_summaries:
            historical_record = summary.historical_record

            # Store to DynamoDB
            await self._store_to_dynamodb(
                user_id=user_id,
                date=date,
                historical_record=historical_record,
            )

            # Generate embedding and store to Milvus
            await self._store_to_milvus(
                user_id=user_id,
                historical_record=historical_record,
            )

        # 2. Store daily summary
        await self._store_daily_summary(
            user_id=user_id,
            date=date,
            daily_summary=daily_analysis,
        )

        logger.info(
            f"Archived {len(daily_analysis.request_summaries)} requests for {date}"
        )

    async def _store_to_dynamodb(
        self,
        user_id: str,
        date: str,
        historical_record: HistoricalRequestRecord,
    ) -> None:
        """存储HistoricalRequestRecord到DynamoDB"""
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, skipping storage")
            return

        try:
            table_name = "HistoricalRequests"  # TODO: Make configurable

            item = {
                "pk": f"USER#{user_id}",
                "sk": f"REQ#{historical_record.request_id}",
                "gsi1pk": f"USER#{user_id}#DATE#{date}",  # GSI for date queries
                "gsi1sk": historical_record.created_at.isoformat(),
                "entity": "historical_request",
                "user_id": user_id,
                "date": date,
                "request_id": historical_record.request_id,
                "data": historical_record.model_dump(),
                "theme": historical_record.theme,
                "completion_status": historical_record.completion_status,
                "analyzed_at": historical_record.analyzed_at.isoformat(),
            }

            # TODO: Actual DynamoDB put_item call
            # await self.dynamodb.put_item(TableName=table_name, Item=item)
            logger.debug(f"Would store to DynamoDB: {item['pk']}/{item['sk']}")

        except Exception as e:
            logger.error(f"Failed to store to DynamoDB: {e}")
            raise

    async def _store_to_milvus(
        self,
        user_id: str,
        historical_record: HistoricalRequestRecord,
    ) -> None:
        """生成embedding并存储到Milvus"""
        if not self.milvus:
            logger.warning("Milvus client not configured, skipping vector storage")
            return

        try:
            # 1. Generate embedding
            embedding = await self._generate_embedding(historical_record)

            # 2. Prepare metadata
            metadata = {
                "request_id": historical_record.request_id,
                "user_id": user_id,
                "name": historical_record.name,
                "theme": historical_record.theme,
                "completion_status": historical_record.completion_status,
                "created_at": historical_record.created_at.isoformat(),
                "keywords": ",".join(historical_record.keywords),
            }

            # 3. Insert to Milvus
            collection_name = f"user_requests_{user_id}"

            # TODO: Actual Milvus insert
            # await self.milvus.insert(
            #     collection_name=collection_name,
            #     data=[{
            #         "id": historical_record.request_id,
            #         "embedding": embedding,
            #         "metadata": json.dumps(metadata)
            #     }]
            # )

            logger.debug(
                f"Would store to Milvus: collection={collection_name}, "
                f"id={historical_record.request_id}"
            )

        except Exception as e:
            logger.error(f"Failed to store to Milvus: {e}")
            raise

    async def _generate_embedding(
        self, historical_record: HistoricalRequestRecord
    ) -> List[float]:
        """
        为historical request生成embedding

        组合以下内容：
        - name
        - goal
        - short_summary
        - keywords
        """
        # Combine text for embedding
        text_to_embed = (
            f"{historical_record.name}. "
            f"{historical_record.goal}. "
            f"{historical_record.short_summary}. "
            f"Keywords: {', '.join(historical_record.keywords)}"
        )

        if self.embedding_client:
            # TODO: Actual embedding generation
            # embedding = await self.embedding_client.generate_embedding(text_to_embed)
            # return embedding
            pass

        # Fallback: return dummy embedding
        logger.warning("Embedding client not configured, returning dummy embedding")
        return [0.0] * 1536  # text-embedding-3-small dimension

    async def _store_daily_summary(
        self,
        user_id: str,
        date: str,
        daily_summary: DailyRequestSummary,
    ) -> None:
        """存储DailyRequestSummary"""
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured, skipping summary storage")
            return

        try:
            table_name = "DailySummaries"  # TODO: Make configurable

            item = {
                "pk": f"USER#{user_id}",
                "sk": f"SUMMARY#{date}",
                "entity": "daily_summary",
                "user_id": user_id,
                "date": date,
                "data": daily_summary.model_dump(),
                "analyzed_at": daily_summary.analyzed_at.isoformat(),
            }

            # TODO: Actual DynamoDB put_item call
            logger.debug(f"Would store daily summary: {item['pk']}/{item['sk']}")

        except Exception as e:
            logger.error(f"Failed to store daily summary: {e}")
            raise

    async def search_similar_requests(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """
        Vector search for similar requests

        Args:
            user_id: User ID
            query_text: Search query text
            top_k: Number of results to return
            filters: Optional filters (theme, completion_status, etc.)

        Returns:
            List of VectorSearchResult sorted by similarity
        """
        if not self.milvus:
            logger.warning("Milvus client not configured, returning empty results")
            return []

        try:
            # 1. Generate query embedding
            query_embedding = await self._generate_query_embedding(query_text)

            # 2. Search in Milvus
            collection_name = f"user_requests_{user_id}"

            # Build filter expression
            filter_expr = self._build_filter_expression(filters)

            # TODO: Actual Milvus search
            # results = await self.milvus.search(
            #     collection_name=collection_name,
            #     query_vectors=[query_embedding],
            #     limit=top_k,
            #     filter=filter_expr,
            #     output_fields=["request_id", "metadata"]
            # )

            # Placeholder results
            logger.info(
                f"Would search Milvus: collection={collection_name}, "
                f"query='{query_text[:50]}...', top_k={top_k}"
            )

            # 3. Load full HistoricalRequestRecords
            search_results = []

            # TODO: Load from actual search results
            # for hit in results[0]:
            #     historical_record = await self._load_historical_request(
            #         user_id=user_id,
            #         request_id=hit.id
            #     )
            #     search_results.append(VectorSearchResult(
            #         historical_record=historical_record,
            #         similarity_score=hit.score,
            #         rank=hit.rank
            #     ))

            return search_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _generate_query_embedding(self, query_text: str) -> List[float]:
        """生成query的embedding"""
        if self.embedding_client:
            # TODO: Actual embedding generation
            pass

        # Fallback
        logger.warning("Embedding client not configured, returning dummy embedding")
        return [0.0] * 1536

    def _build_filter_expression(
        self, filters: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """构建Milvus filter expression"""
        if not filters:
            return None

        # TODO: Build proper Milvus filter expression
        # Example: "theme in ['medicaid_application', 'legal_advice'] and completion_status == 'completed'"
        return None

    async def _load_historical_request(
        self,
        user_id: str,
        request_id: str,
    ) -> HistoricalRequestRecord:
        """从DynamoDB加载完整的HistoricalRequestRecord"""
        if not self.dynamodb:
            raise ValueError("DynamoDB client not configured")

        try:
            table_name = "HistoricalRequests"

            # TODO: Actual DynamoDB get_item call
            # response = await self.dynamodb.get_item(
            #     TableName=table_name,
            #     Key={
            #         "pk": f"USER#{user_id}",
            #         "sk": f"REQ#{request_id}"
            #     }
            # )
            #
            # if "Item" not in response:
            #     raise ValueError(f"Request {request_id} not found")
            #
            # data = response["Item"]["data"]
            # return HistoricalRequestRecord(**data)

            # Placeholder
            logger.warning(f"Would load from DynamoDB: USER#{user_id}/REQ#{request_id}")
            raise ValueError("DynamoDB not configured - cannot load historical request")

        except Exception as e:
            logger.error(f"Failed to load historical request: {e}")
            raise

    async def get_requests_by_date(
        self,
        user_id: str,
        date: str,
    ) -> List[HistoricalRequestRecord]:
        """获取某一天的所有requests"""
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured")
            return []

        try:
            # TODO: Query by GSI (gsi1pk = USER#user_id#DATE#date)
            logger.info(f"Would query DynamoDB for user {user_id} on {date}")
            return []

        except Exception as e:
            logger.error(f"Failed to get requests by date: {e}")
            return []

    async def get_daily_summary(
        self,
        user_id: str,
        date: str,
    ) -> Optional[DailyRequestSummary]:
        """获取某一天的DailyRequestSummary"""
        if not self.dynamodb:
            logger.warning("DynamoDB client not configured")
            return None

        try:
            # TODO: Get from DynamoDB
            logger.info(f"Would load daily summary for {user_id} on {date}")
            return None

        except Exception as e:
            logger.error(f"Failed to get daily summary: {e}")
            return None
