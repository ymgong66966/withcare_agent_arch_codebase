# Historical Request Tracking Architecture

## 概述

这个模块实现了**retrospective request analysis和vector search**，解决以下问题：

1. ✅ **Unified Request Format**: 所有requests（current和historical）使用统一格式
2. ✅ **Retrospective Completion判断**: 每天用LLM分析哪些requests完成了
3. ✅ **Vector Search**: 基于Milvus/Zilliz的semantic search
4. ✅ **Cache策略**: 所有historical requests都被archive，不管完成与否
5. ✅ **Daily Summaries**: 自动生成每天的request总结

---

## 核心组件

### 1. 数据模型 (`historical_models.py`)

#### **HistoricalRequestRecord**
存储到vector DB和DynamoDB的格式，包含：
- 基本信息：name, goal, created_at
- **Retrospective分析** (LLM判断):
  - `completion_status`: completed | abandoned | pending | uncertain
  - `user_satisfaction`: satisfied | unsatisfied | neutral | unknown
  - `completion_confidence`: 0.0-1.0
- **Search优化**:
  - `short_summary`: 1-2句总结
  - `theme`: 分类（如"medicaid_application", "in_home_care"）
  - `keywords`: 关键词列表
- 完整上下文：`stage_history`, `final_state_snapshot`

#### **DailyRequestSummary**
每天的request总结，包含：
- 所有requests的分析
- Overall day summary
- 主题统计

#### **RequestCache**
In-memory的当天working state

---

### 2. Daily Analyzer (`daily_analyzer.py`)

**DailyRequestAnalyzer** - 每天结束时运行

#### 工作流程：

```python
# 1. 分析整天的对话
overall_analysis = await analyzer._analyze_overall_day(
    conversations=messages,
    requests=active_requests,
)

# 2. 对每个request做retrospective分析
for request in active_requests:
    historical_record = await analyzer._analyze_single_request(
        request=request,
        conversations=conversations,
        overall_insights=overall_analysis,
    )
```

#### LLM分析的内容：

```
对于每个request：
1. 完成了吗？ → completion_status
2. 最后的stage？ → final_stage
3. User满意吗？ → user_satisfaction
4. 发生了什么？ → short_summary
5. 主题是什么？ → theme
6. 关键词？ → keywords
7. 置信度？ → confidence
```

---

### 3. Historical Store (`historical_store.py`)

**HistoricalRequestStore** - 存储和检索

#### 存储层级：

```
1. DynamoDB
   - HistoricalRequestRecord (完整记录)
   - DailyRequestSummary
   - Key structure:
     pk: USER#user_id
     sk: REQ#request_id
     GSI: USER#user_id#DATE#2026-01-28

2. Milvus/Zilliz
   - Embeddings for vector search
   - Collection: user_requests_{user_id}
   - Metadata: theme, completion_status, keywords

3. S3 (可选)
   - 归档的daily summaries
```

#### 核心方法：

```python
# 归档一天的requests
await store.archive_day(
    user_id="user-123",
    date="2026-01-28",
    request_cache=cache,
    daily_analysis=summary,
)

# Vector search
results = await store.search_similar_requests(
    user_id="user-123",
    query_text="找护工",
    top_k=5,
    filters={"completion_status": ["completed"]},
)
```

---

### 4. Retrieval接口 (`retrieval.py`)

#### ✅ 统一返回格式

```python
# OLD (返回Dict[str, Any])
similar = await fetch_similar_requests(
    user_id="user-123",
    request_text="找护工",
    top_k=5,
)  # Returns: List[Dict[str, Any]]

# NEW (返回HistoricalRequestRecord)
similar = await fetch_similar_requests(
    user_id="user-123",
    request_text="找护工",
    top_k=5,
)  # Returns: List[HistoricalRequestRecord]
```

#### 新增方法

```python
# 按主题搜索
records = await get_requests_by_theme(
    user_id="user-123",
    theme="medicaid_application",
    top_k=10,
)

# 最近N天的requests
records = await get_recent_requests(
    user_id="user-123",
    days=7,
    top_k=10,
)
```

---

## 使用场景

### Scenario 1: User无意暴露prerequisite

```
Day 1, 10:00 - User创建request: "找护工"
Day 1, 10:05 - Info collection中
Day 1, 10:10 - User说: "我还没申请Medicaid，不知道预算"
              → Info collection agent检测到prerequisite
              → 提醒user
Day 1, 11:00 - User说: "好，先帮我申请Medicaid"
              → Upstream delegator创建新request
              → "找护工" request被pause

Daily Job (Day 1结束时):
  LLM分析:
  - Request 1 (找护工):
    * completion_status: "pending" (等待prerequisite)
    * final_stage: "info_collection"
    * short_summary: "User wanted to find caregiver but discovered need for Medicaid first"
    * theme: "in_home_care"

  - Request 2 (Medicaid):
    * completion_status: "pending" (刚开始)
    * final_stage: "info_collection"
    * theme: "medicaid_application"

  存储到Milvus → 下次可以search
```

### Scenario 2: User完成request并自然转移

```
Day 1, 10:00 - Request: "Legal advice"
Day 1, 10:30 - Info collection完成
Day 1, 11:00 - Domain expert给出建议
Day 1, 11:30 - User说: "好的，谢谢。现在帮我找in-home care"
              → 没有说对legal advice不满意
              → 自然转移话题
              → Upstream delegator创建新request

Daily Job (Day 1结束时):
  LLM分析:
  - Request 1 (Legal advice):
    * completion_status: "completed" ✅
    * user_satisfaction: "neutral" (没明确反馈但自然转移)
    * confidence: 0.7
    * short_summary: "User received legal advice from domain expert, then moved to next topic"
    * theme: "legal_advice"

  存储为completed → 可以在future被search到
```

### Scenario 3: 搜索similar requests

```python
# Day 5, User新request: "我需要申请Medicaid"
similar_requests = await fetch_similar_requests(
    user_id="user-123",
    request_text="我需要申请Medicaid",
    top_k=3,
)

# Returns:
[
  HistoricalRequestRecord(
    name="Medicaid Application",
    theme="medicaid_application",
    completion_status="completed",
    short_summary="User successfully applied for Medicaid with our guidance",
    created_at=datetime(2026, 1, 28, ...),
    # ... 完整上下文
  ),
  # ... more similar requests
]

# Info collection agent可以利用这些context:
# "我看到你之前在Day 1也咨询过Medicaid申请，当时是..."
```

---

## Daily Job部署

### Cron Job配置

```bash
# 每天凌晨1点运行
0 1 * * * python /path/to/run_daily_job.py
```

### Daily Job脚本

```python
# run_daily_job.py
import asyncio
from daily_analyzer import DailyRequestAnalyzer
from historical_store import HistoricalRequestStore

async def main():
    # 1. 获取今天的数据
    today = datetime.now().strftime("%Y-%m-%d")
    users = get_all_active_users(today)

    for user_id in users:
        # 获取user今天的对话和requests
        conversations = get_conversations(user_id, today)
        requests = get_active_requests(user_id, today)

        # 2. 分析
        analyzer = DailyRequestAnalyzer()
        summary = await analyzer.analyze_daily_requests(
            user_id=user_id,
            date=today,
            conversations=conversations,
            active_requests=requests,
        )

        # 3. 归档
        store = HistoricalRequestStore(
            dynamodb_client=init_dynamodb(),
            milvus_client=init_milvus(),
            embedding_client=init_openai(),
        )
        await store.archive_day(
            user_id=user_id,
            date=today,
            request_cache=...,
            daily_analysis=summary,
        )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Integration Points

### 1. Info Collection Node

```python
# graph.py - info_collection_node

# 获取similar requests作为context
similar = await fetch_similar_requests(
    user_id=user_id,
    request_text=user_text,
    top_k=5,
)
# similar现在是List[HistoricalRequestRecord]

plan = await llm_collection_plan(
    user_request=user_text,
    known_facts=known_facts,
    similar_requests=similar,  # ✅ 统一格式
    client=client,
)
```

### 2. Request Manager

当前requests应该track stage transitions:

```python
# 添加到RequestRecord
class RequestRecord(BaseModel):
    # ... existing fields
    stage_history: List[StageTransition] = Field(default_factory=list)

# 每次stage变化时记录
request.stage_history.append(
    StageTransition(
        from_stage="info_collection",
        to_stage="deep_search",
        agent="deep_search",
        timestamp=datetime.utcnow(),
        reason="Info collected, proceeding to search",
    )
)
```

---

## TODO / 下一步

### Phase 1 (基础设施) ✅
- [x] 定义数据模型
- [x] 实现DailyRequestAnalyzer
- [x] 实现HistoricalRequestStore
- [x] 更新retrieval接口

### Phase 2 (集成真实存储)
- [ ] 集成DynamoDB client
- [ ] 集成Milvus/Zilliz client
- [ ] 集成OpenAI embedding API
- [ ] 测试end-to-end workflow

### Phase 3 (优化)
- [ ] 添加cache layer (Redis)
- [ ] 优化embedding generation (batch)
- [ ] 实现incremental updates
- [ ] 添加monitoring和alerting

### Phase 4 (高级功能)
- [ ] Multi-modal embeddings (text + metadata)
- [ ] Hybrid search (vector + keyword)
- [ ] Request clustering and pattern detection
- [ ] Proactive suggestions based on history

---

## 测试

运行示例：

```bash
python example_daily_job.py
```

这会模拟一个daily job的完整流程。

---

## 架构优势

1. ✅ **Retrospective判断**: 不依赖real-time status，更准确
2. ✅ **统一格式**: Current和historical requests用同一个model
3. ✅ **Context-rich search**: Vector search + metadata filtering
4. ✅ **可扩展**: 支持多种存储backend
5. ✅ **LLM-driven**: 利用LLM的理解能力做判断
6. ✅ **完整历史**: 所有requests都被archive，支持复杂分析
