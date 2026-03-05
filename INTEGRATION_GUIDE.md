# Historical Request Tracking - Integration Guide

## 如何将这个架构集成到现有系统

---

## Step 1: 更新RequestRecord添加stage_history

### 修改 `state_models.py`

```python
from historical_models import StageTransition

class RequestRecord(BaseModel):
    # ... existing fields ...

    # ✅ 添加这个字段
    stage_history: List[StageTransition] = Field(default_factory=list)
```

---

## Step 2: 在节点中记录stage transitions

### 修改 `graph.py` - 每次stage变化时记录

```python
# Example: info_collection_node → deep_search

async def info_collection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing logic ...

    # When ready to proceed
    if readiness == "ready":
        routing_hint = req.get("routing_hint", "deep_search")

        # ✅ 记录stage transition
        from historical_models import StageTransition

        new_transition = StageTransition(
            from_stage="info_collection",
            to_stage=routing_hint,
            agent=routing_hint,
            timestamp=datetime.utcnow(),
            reason="Information collection complete; proceeding to execution"
        )

        # 添加到request的stage_history
        existing_history = req.get("stage_history", [])
        existing_history.append(new_transition.model_dump())

        patch["request_manager"]["requests"][active_id]["stage_history"] = existing_history

        # ... rest of logic ...
```

### 类似的，在其他关键节点也记录

```python
# upstream_delegator - 创建新request时
if action_type == "prerequisite_task":
    # Create new request
    new_request_patch = build_request_patch(...)

    # ✅ 初始化stage_history
    initial_transition = StageTransition(
        from_stage=None,
        to_stage="info_collection",
        agent="info_collection",
        timestamp=datetime.utcnow(),
        reason=f"Prerequisite request created: {prereq_type}"
    )

    patch["request_manager"]["requests"][new_req_id]["stage_history"] = [
        initial_transition.model_dump()
    ]
```

---

## Step 3: 每天运行Daily Job

### 选项A: Cron Job (推荐用于生产环境)

```bash
# /etc/cron.d/daily_request_analysis
0 1 * * * python /path/to/run_daily_job.py >> /var/log/daily_job.log 2>&1
```

### 选项B: AWS Lambda (Serverless)

```python
# lambda_handler.py
import asyncio
from daily_analyzer import DailyRequestAnalyzer
from historical_store import HistoricalRequestStore
import boto3

def lambda_handler(event, context):
    """
    Triggered daily by EventBridge (CloudWatch Events)
    """
    asyncio.run(process_all_users())
    return {"statusCode": 200}

async def process_all_users():
    dynamodb = boto3.client('dynamodb')
    # ... get all users who were active yesterday
    # ... run analysis for each user
```

### 选项C: 集成到现有的executor

```python
# executor.py - 在对话结束时或定期运行

from daily_analyzer import DailyRequestAnalyzer

async def maybe_run_daily_analysis(state, date):
    """
    检查是否需要运行daily analysis

    触发条件：
    - 新的一天开始
    - 或user说"再见"
    - 或conversation idle超过N小时
    """
    if should_run_analysis(state, date):
        analyzer = DailyRequestAnalyzer()

        # 获取昨天的数据
        yesterday = get_yesterday(date)
        conversations = get_conversations_for_date(state, yesterday)
        requests = get_requests_for_date(state, yesterday)

        # 分析
        summary = await analyzer.analyze_daily_requests(
            user_id=state.meta.user_id,
            date=yesterday,
            conversations=conversations,
            active_requests=requests,
        )

        # 归档
        store = HistoricalRequestStore(...)
        await store.archive_day(...)
```

---

## Step 4: 配置存储Backends

### DynamoDB Tables

```yaml
# CloudFormation template
HistoricalRequestsTable:
  Type: AWS::DynamoDB::Table
  Properties:
    TableName: HistoricalRequests
    BillingMode: PAY_PER_REQUEST
    AttributeDefinitions:
      - AttributeName: pk
        AttributeType: S
      - AttributeName: sk
        AttributeType: S
      - AttributeName: gsi1pk
        AttributeType: S
      - AttributeName: gsi1sk
        AttributeType: S
    KeySchema:
      - AttributeName: pk
        KeyType: HASH
      - AttributeName: sk
        KeyType: RANGE
    GlobalSecondaryIndexes:
      - IndexName: DateIndex
        KeySchema:
          - AttributeName: gsi1pk
            KeyType: HASH
          - AttributeName: gsi1sk
            KeyType: RANGE
        Projection:
          ProjectionType: ALL

DailySummariesTable:
  Type: AWS::DynamoDB::Table
  Properties:
    TableName: DailySummaries
    BillingMode: PAY_PER_REQUEST
    AttributeDefinitions:
      - AttributeName: pk
        AttributeType: S
      - AttributeName: sk
        AttributeType: S
    KeySchema:
      - AttributeName: pk
        KeyType: HASH
      - AttributeName: sk
        KeyType: RANGE
```

### Milvus/Zilliz Collection

```python
# setup_milvus.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

def create_user_request_collection(user_id: str):
    """为user创建Milvus collection"""

    # Connect
    connections.connect("default", host="localhost", port="19530")

    # Define schema
    fields = [
        FieldSchema(name="request_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="theme", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="completion_status", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=5000),
    ]

    schema = CollectionSchema(fields, description=f"Requests for user {user_id}")

    # Create collection
    collection_name = f"user_requests_{user_id}"
    collection = Collection(name=collection_name, schema=schema)

    # Create index
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    return collection
```

### 初始化Clients

```python
# historical_store.py - 更新初始化

import boto3
from pymilvus import connections, Collection
import openai

# DynamoDB
dynamodb_client = boto3.client(
    'dynamodb',
    region_name='us-east-1',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
)

# Milvus
connections.connect(
    "default",
    host=os.environ.get('MILVUS_HOST', 'localhost'),
    port=os.environ.get('MILVUS_PORT', '19530'),
)

# OpenAI (for embeddings)
openai.api_key = os.environ['OPENAI_API_KEY']

# 创建store
store = HistoricalRequestStore(
    dynamodb_client=dynamodb_client,
    milvus_client=connections,  # 或封装的MilvusClient
    embedding_client=openai,
)
```

---

## Step 5: 更新info_collection使用historical context

### 修改 `graph.py` - info_collection_node

```python
async def info_collection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing setup ...

    if not active_id:  # New request
        rid = new_uuid("req")

        # ✅ Fetch similar historical requests
        from retrieval import fetch_similar_requests

        similar = await fetch_similar_requests(
            user_id=user_id,
            request_text=user_text,
            top_k=5,
        )
        # similar: List[HistoricalRequestRecord]

        # ✅ 传递给LLM作为context
        plan = await llm_collection_plan(
            user_request=user_text,
            known_facts=known_facts,
            similar_requests=[s.model_dump() for s in similar],  # 转成dict
            client=client,
        )

        # ... rest of logic ...
```

### 修改 `prompts.py` - make_collection_plan_prompt

```python
def make_collection_plan_prompt(
    *,
    user_request: str,
    known_facts: Dict[str, Any],
    similar_requests: List[Dict[str, Any]] | None = None,
) -> str:
    # ✅ 格式化similar requests为更有用的context
    similar_context = ""
    if similar_requests:
        similar_context = "\n\nSimilar past requests:\n"
        for req in similar_requests[:3]:
            similar_context += f"""
- {req['name']} ({req['completion_status']})
  Goal: {req['goal']}
  Outcome: {req['short_summary']}
  Theme: {req['theme']}
"""

    return f"""You are an information collection planner...

## User's Request:
{user_request}

## Known Facts:
{json.dumps(known_facts, indent=2, ensure_ascii=False)}
{similar_context}

## Your Task:
Based on the user's request and similar past requests, create a collection plan...
"""
```

---

## Step 6: 监控和Debugging

### 添加Logging

```python
# 在daily_analyzer.py和historical_store.py中
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_job.log'),
        logging.StreamHandler()
    ]
)
```

### CloudWatch Metrics (如果在AWS上)

```python
import boto3
cloudwatch = boto3.client('cloudwatch')

def report_metrics(summary: DailyRequestSummary):
    cloudwatch.put_metric_data(
        Namespace='RequestTracking',
        MetricData=[
            {
                'MetricName': 'RequestsCompleted',
                'Value': summary.total_requests_completed_today,
                'Unit': 'Count',
            },
            {
                'MetricName': 'RequestsPending',
                'Value': summary.total_requests_pending,
                'Unit': 'Count',
            },
        ]
    )
```

---

## Step 7: 测试

### Unit Tests

```python
# test_daily_analyzer.py
import pytest
from daily_analyzer import DailyRequestAnalyzer

@pytest.mark.asyncio
async def test_analyze_single_request():
    analyzer = DailyRequestAnalyzer()

    request_dict = {
        "request_id": "req-test",
        "name": "Test Request",
        "goal": "Test goal",
        "status": "completed",
        "created_at": datetime.utcnow(),
    }

    summary = await analyzer._analyze_single_request(
        user_id="user-test",
        conversation_id="conv-test",
        date="2026-01-28",
        request_id="req-test",
        request_dict=request_dict,
        conversations=[],
        overall_insights={},
    )

    assert summary.historical_record.request_id == "req-test"
    assert summary.historical_record.completion_status in ["completed", "pending", "abandoned", "uncertain"]
```

### Integration Test

```bash
# Run the example
python example_daily_job.py

# Should see output:
# ✅ Analysis completed
# ✅ Would store to DynamoDB
# ✅ Would store to Milvus
```

---

## Troubleshooting

### Issue: LLM analysis fails

**Solution**: 检查fallback logic，确保至少有heuristic-based判断

```python
# In daily_analyzer.py
def _fallback_analysis(self, request_dict):
    # 基于status的简单判断
    status = request_dict.get("status")
    if status == "completed":
        return {"completion_status": "completed", ...}
```

### Issue: Milvus connection失败

**Solution**: 确保Milvus服务运行，或使用Zilliz Cloud

```bash
# Local Milvus (Docker)
docker-compose up -d

# 或使用Zilliz Cloud
connections.connect(
    "default",
    uri="https://xxx.zillizcloud.com",
    token="your_token"
)
```

### Issue: Embedding generation太慢

**Solution**: 使用batch processing

```python
async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
    """Batch generate embeddings"""
    response = await openai.Embedding.acreate(
        model="text-embedding-3-small",
        input=texts
    )
    return [item['embedding'] for item in response['data']]
```

---

## Performance Considerations

### 1. Batch Processing
- 每天一次性处理所有users，而不是分散处理
- Batch embedding generation

### 2. Caching
- Cache frequently accessed historical requests in Redis
- TTL: 1 hour for hot data

### 3. Async Operations
- 并行处理多个users
- 并行进行DynamoDB write和Milvus insert

```python
async def process_users_in_parallel(users):
    tasks = [process_user(user_id) for user_id in users]
    await asyncio.gather(*tasks)
```

---

## Next Steps

1. ✅ 完成基础架构实现
2. 🔨 集成到现有graph.py
3. 🔨 配置DynamoDB和Milvus
4. 🔨 部署daily job
5. 🔨 测试end-to-end
6. 📊 添加monitoring
7. 🚀 优化performance

---

有问题可以参考：
- `HISTORICAL_REQUESTS_README.md` - 架构详解
- `example_daily_job.py` - 使用示例
- `historical_models.py` - 数据模型定义
