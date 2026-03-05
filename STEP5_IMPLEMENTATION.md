# Step 5 Implementation: Historical Context Integration

## 概述

实现了info_collection使用historical context的功能。当用户创建新request时，系统会：
1. 搜索相似的历史requests
2. 将这些历史作为context传递给LLM
3. LLM基于历史生成更好的collection plan

---

## 实现详情

### 1. 更新 prompts.py - 改进similar requests格式化

**文件**: `prompts.py` Line 24-36

**Before** (简单格式):
```python
similar_context = "\n\nSimilar past requests:\n" + "\n".join([
    f"- {req.get('name', 'Unknown')}: {req.get('goal', '')[:100]}"
    for req in similar_requests[:3]
])
```

**After** (详细格式):
```python
similar_context = "\n\n## Similar Past Requests:\n\n"
similar_context += "The user has worked on similar requests before. Use these as context:\n\n"
for i, req in enumerate(similar_requests[:3], 1):
    completion = req.get('completion_status', 'unknown')
    similar_context += f"{i}. **{req.get('name', 'Unknown')}** ({completion})\n"
    similar_context += f"   - Goal: {req.get('goal', 'N/A')[:150]}\n"
    similar_context += f"   - Outcome: {req.get('short_summary', 'N/A')[:200]}\n"
    similar_context += f"   - Theme: {req.get('theme', 'N/A')}\n"
    if req.get('user_satisfaction'):
        similar_context += f"   - User was: {req.get('user_satisfaction')}\n"
    similar_context += "\n"
```

**改进**:
- ✅ 包含completion_status（completed/pending/abandoned）
- ✅ 包含short_summary（LLM生成的结果总结）
- ✅ 包含theme（分类）
- ✅ 包含user_satisfaction（用户满意度）
- ✅ 更清晰的格式化，易于LLM理解

---

### 2. 更新 graph.py - 转换数据类型

**文件**: `graph.py` Line 569-579

**Before**:
```python
# Fetch similar requests for context
similar = await fetch_similar_requests(user_id=user_id, request_text=user_text, top_k=5)

# LLM generates collection plan
plan = await llm_collection_plan(
    user_request=user_text,
    known_facts=known_facts,
    similar_requests=similar,  # List[HistoricalRequestRecord]
    client=client,
)
```

**After**:
```python
# Fetch similar requests for context
similar = await fetch_similar_requests(user_id=user_id, request_text=user_text, top_k=5)

# Convert HistoricalRequestRecord objects to dicts for LLM
similar_dicts = [s.model_dump() for s in similar] if similar else []

# LLM generates collection plan
plan = await llm_collection_plan(
    user_request=user_text,
    known_facts=known_facts,
    similar_requests=similar_dicts,  # List[Dict[str, Any]]
    client=client,
)
```

**改进**:
- ✅ 将Pydantic对象转换为dict（LLM prompt需要dict格式）
- ✅ 处理空列表的情况

---

### 3. 更新 prompts.py - 让similar_requests可选

**文件**: `prompts.py` Line 300-306

**Before**:
```python
async def llm_collection_plan(
    *,
    user_request: str,
    known_facts: Dict[str, Any],
    similar_requests: List[Dict[str, Any]],  # Required
    client: Any,
) -> Dict[str, Any]:
```

**After**:
```python
async def llm_collection_plan(
    *,
    user_request: str,
    known_facts: Dict[str, Any],
    similar_requests: List[Dict[str, Any]] | None = None,  # Optional
    client: Any,
) -> Dict[str, Any]:
```

**改进**:
- ✅ similar_requests现在是可选的
- ✅ 新用户没有历史时也能正常工作

---

## 数据流

### 完整流程

```
User: "我需要找一个护工"
    ↓
info_collection_node (graph.py:566-579)
    ↓
fetch_similar_requests(user_id, request_text, top_k=5)
    ↓ returns List[HistoricalRequestRecord]
    ↓
Convert to List[Dict]
    ↓
llm_collection_plan(user_request, known_facts, similar_dicts, client)
    ↓
make_collection_plan_prompt(..., similar_requests=similar_dicts)
    ↓ formats similar requests with detailed info
    ↓
LLM receives prompt with historical context
    ↓
LLM generates better collection plan based on:
    - Current request
    - Known facts
    - Similar past requests (what worked, what didn't)
    ↓
Returns collection plan with:
    - request_name
    - request_goal
    - key_info_needed (informed by history)
    - nice_to_have_info
    - potential_prerequisites (learned from past)
    - routing_hint
```

---

## 示例场景

### Scenario 1: 有相关历史

**User**: "我需要找一个护工，会说中文的"

**Historical Context** (从vector search获取):
```python
[
    {
        "name": "Find Chinese-speaking Caregiver",
        "completion_status": "completed",
        "short_summary": "User successfully found caregiver. Budget was $2500/month.",
        "theme": "in_home_care",
        "user_satisfaction": "satisfied",
    }
]
```

**LLM Prompt** (包含历史):
```
## Similar Past Requests:

The user has worked on similar requests before. Use these as context:

1. **Find Chinese-speaking Caregiver** (completed)
   - Goal: Find a caregiver who speaks Chinese in Chicago...
   - Outcome: User successfully found caregiver. Budget was $2500/month.
   - Theme: in_home_care
   - User was: satisfied
```

**Result**:
- LLM知道之前的预算是$2500/month
- LLM知道用户在Chicago
- LLM可能直接问预算范围（而不是从零开始）

---

### Scenario 2: 无历史（新用户）

**User**: "我需要申请Medicare"

**Historical Context**: `[]` (empty)

**LLM Prompt** (无历史):
```
## User's Request:
我需要申请Medicare

## Known Facts About User:
...

## Your Task:
Analyze the user's request and create a collection plan...
```

**Result**:
- 正常工作，不会因为没有历史而失败
- LLM基于一般知识生成collection plan

---

## 测试

### 运行测试

```bash
python test_historical_context.py
```

### 测试内容

1. **Test 1**: 带历史context生成collection plan
   - 验证prompt格式化正确
   - 验证LLM能基于历史生成plan
   - 验证historical context被正确使用

2. **Test 2**: 无历史context生成collection plan
   - 验证空列表不会导致错误
   - 验证新用户也能正常使用

---

## 与现有系统的关系

### 依赖关系

```
info_collection_node (graph.py)
    ↓ depends on
retrieval.py (fetch_similar_requests)
    ↓ depends on
historical_store.py (search_similar_requests)
    ↓ depends on
- DynamoDB (存储historical records)
- Milvus/Zilliz (vector search)
```

**当前状态**:
- ✅ Code结构已就位
- ⏳ Historical store使用stub implementation
- ⏳ 等待真实数据后，vector search自动生效

### Fallback机制

如果没有历史数据：
1. `fetch_similar_requests()` 返回空列表 `[]`
2. `similar_dicts` = `[]`
3. `make_collection_plan_prompt()` 不添加similar context部分
4. LLM收到的prompt没有历史部分
5. 正常生成collection plan（基于一般知识）

---

## 好处

### 1. 更快的信息收集

用户之前做过类似任务 → LLM知道需要什么信息 → 减少来回询问

### 2. 个性化体验

基于用户的历史偏好和模式 → 更贴合用户需求

### 3. 学习用户模式

- 用户通常关心什么？
- 什么问题最重要？
- 哪些prerequisites容易被忽略？

### 4. 避免重复错误

如果上次某个request失败了 → LLM可以提前询问关键信息 → 避免同样的问题

---

## 未来改进

### 1. 更智能的历史匹配

当前：基本的vector search
未来：
- 考虑时间衰减（最近的历史权重更高）
- 考虑completion status（优先参考成功的cases）
- 考虑user satisfaction（参考满意度高的cases）

### 2. 历史对比

在prompt中明确对比：
```
Previous similar request:
- You asked for: X, Y, Z
- This time, you might also need: A, B
```

### 3. 自适应问题顺序

基于历史，调整问题的顺序：
- 上次用户最关心的问题先问
- 上次遗漏导致失败的信息先问

### 4. Prerequisite预测

如果历史中发现用户经常需要某个prerequisite：
```
"Based on your past requests, you might also need to [prerequisite].
Would you like me to help with that first?"
```

---

## 相关文件

- **实现**: `prompts.py` (Line 24-36, 304), `graph.py` (Line 569-579)
- **测试**: `test_historical_context.py`
- **Integration guide**: `INTEGRATION_GUIDE.md` (Step 5)
- **Architecture**: `HISTORICAL_REQUESTS_README.md`
- **Retrieval**: `retrieval.py`

---

## 总结

✅ **Step 5 完成！**

- ✅ info_collection现在能使用historical context
- ✅ Similar requests被格式化为详细的context
- ✅ LLM能基于历史生成更好的collection plan
- ✅ 支持新用户（无历史）的场景
- ✅ 代码结构就位，等待真实数据

**下一步**:
- Step 3: 运行daily job测试完整流程
- Step 4: 配置存储backends（可选）

---

## Info Collection Recursive Q&A Flow 测试计划

### 目标

验证 `info_collection_node` 的完整递归一问一答 workflow：
1. 用户发起请求 → LLM 生成 collection plan → 向用户提问
2. 用户回答 → LLM 总结已收集信息 + 评估 readiness → 继续追问或 handoff
3. 用户说"先用现有的继续" → 系统 finalize 并 handoff 到下一个 agent
4. 用户提供完整信息 → readiness = "ready" → 自动 handoff

### 测试场景

#### Scenario 1: 完整 3-turn 信息收集 → 自动 handoff

```
Turn 1: 用户 → "我需要找一个护工，会说中文的，在芝加哥附近"
  期望: info_collection 创建 request, 生成 collection plan, 向用户提问
  验证:
    - active_request_id 存在
    - status = "collecting"
    - awaiting_user_input = True
    - info_collection_state.key_info_needed 非空
    - info_collection_state.readiness_to_proceed = "needs_more"
    - assistant message 包含问题

Turn 2: 用户 → "预算大概2000-3000一个月，尽快开始，需要会做中餐"
  期望: LLM 总结已收集信息, 评估 readiness
  验证:
    - info_collection_state.summary_of_collected_info 包含预算/时间信息
    - info_collection_state.conversation_turns_with_agent >= 2
    - readiness 可能是 "can_proceed_but_incomplete" 或 "needs_more"
    - assistant message 确认收到 + 可能追问

Turn 3: 用户 → "没有其他要求了，可以开始找了"
  期望: LLM 判断 readiness = "ready", handoff 到 deep_search
  验证:
    - status = "validated"
    - awaiting_user_input = False
    - routing.pending_handoff.recommended_next_agent 存在
    - assistant message 确认开始执行
```

#### Scenario 2: 用户中途说"先用现有的继续" → 提前 handoff

```
Turn 1: 同 Scenario 1 Turn 1

Turn 2: 用户 → "先用现有的继续"
  期望: 系统 finalize, handoff 到 routing_hint 指定的 agent
  验证:
    - status = "validated"
    - awaiting_user_input = False
    - routing.pending_handoff.recommended_next_agent 存在
    - assistant message 确认用现有信息继续
```

#### Scenario 3: 多轮追问 → 信息逐步补全

```
Turn 1: 用户 → "帮我申请Medicaid"
Turn 2: 用户 → "在Illinois"
Turn 3: 用户 → "我妈妈65岁，没有工作收入"
Turn 4: 用户 → "好的，信息够了就开始吧"
  验证:
    - 每轮 summary 逐步增长
    - conversation_turns_with_agent 递增
    - readiness 从 needs_more → can_proceed_but_incomplete → ready
```

#### Scenario 4: Prerequisite 检测

```
Turn 1: 用户 → "我需要找一个护工"
Turn 2: 用户 → "预算的话我不太确定，因为我还没申请Medicaid，不知道能cover多少"
  验证:
    - detected_prerequisites 非空 (LLM 检测到 Medicaid)
    - assistant message 提到 Medicaid 作为先决条件
```

#### Scenario 5: Prerequisite 同意 → 创建新 request → 新一轮 Q&A

```
Turn 1: 用户 → "我需要找一个护工"
Turn 2: 用户 → "预算的话我不太确定，因为我还没申请Medicaid，不知道能cover多少"
Turn 3: 用户 → "要，先帮我处理Medicaid申请"
  验证:
    - 新的 prerequisite request 被创建
    - active_request_id 切换到 prerequisite request
    - 原 request status = "paused"
    - prerequisite request 进入 info_collection
Turn 4: 用户 → "在Illinois，我妈妈65岁"
  验证:
    - prerequisite request 的 info_collection_state 存在
    - 新一轮 Q&A 开始（key_info_needed 非空）
    - summary 包含新回答的信息
```

#### Scenario 6: Prerequisite 拒绝 → 继续原 request

```
Turn 1: 用户 → "我需要找一个护工"
Turn 2: 用户 → "预算的话我不太确定，因为我还没申请Medicaid，不知道能cover多少"
Turn 3: 用户 → "不要，先不处理Medicaid，直接帮我找护工"
  验证:
    - active_request_id 保持不变（仍是原 request）
    - 只有一个 request 存在
    - 原 request status != "paused"
Turn 4: 用户 → "芝加哥，预算3000左右"
  验证:
    - 继续收集原 request 的信息
    - summary 包含新信息
```

### 运行测试

```bash
python test_info_collection_flow.py
```

### 测试输出格式

每个 turn 打印:
- 用户输入
- Assistant 回复
- 关键 state 变化 (readiness, summary, awaiting_user_input, pending_handoff)
- Graph 经过的 nodes

最终打印:
- 完整的 info_collection_state
- stage_history
- 是否成功 handoff

---

## ✅ 已完成功能 (2026-02-20)

### 1. Info Collection 递归 Q&A 流程

**实现位置**: `graph.py` - `info_collection_node` (lines 660-900)

**核心功能**:
- ✅ **新 request 初始化**: 生成 collection plan，提出初始问题
- ✅ **LLM 总结和评估**: 使用 `llm_info_collection_summarize` 总结用户回复，评估 readiness
- ✅ **多轮对话**: 支持多轮 Q&A，动态调整问题
- ✅ **Early handoff**: 用户说"先用现有的继续"时，强制 finalize 并 handoff
- ✅ **Prerequisite 检测**: LLM 自动检测并存储 prerequisite 信息
- ✅ **Prerequisite 接受**: 用户同意处理 prerequisite 时，创建新 request 并暂停原 request
- ✅ **Prerequisite 拒绝**: 用户拒绝 prerequisite 时，继续原 request
- ✅ **多语言支持**: LLM 生成的回复自动适配中文/英文

**关键实现细节**:
1. **确定性检查**:
   - `_user_says_decline_more_questions()`: 检测"先用现有的继续"等短语
   - `_user_yes_no()`: 检测"要/不要"回复
   - Prerequisite type 映射: 将 LLM 返回的类型标准化为 `PrereqType` 枚举

2. **LLM 调用**:
   - `llm_info_collection_summarize`: 总结信息、评估 readiness、检测 prerequisites
   - `llm_prerequisite_acceptance_response`: 生成自然的 prerequisite 接受确认消息（支持中英文）

3. **State 管理**:
   - `info_collection_state`: 存储 summary, readiness, conversation_turns, detected_prerequisites
   - `stage_history`: 记录 request 的状态转换历史

### 2. Prerequisite 处理完整流程

**实现位置**: `graph.py` - `info_collection_node` (lines 697-808)

**流程**:
1. **检测**: LLM 在总结用户回复时检测到 prerequisite（如 Medicaid 申请）
2. **询问**: Assistant 询问用户是否要先处理 prerequisite
3. **接受路径**:
   - 用户说"要" → 检测到 `_user_yes_no(user_text) is True`
   - 创建新的 prerequisite request
   - 暂停原 request (`status="paused"`)
   - 将原 request 加入 `pending_queue`
   - 切换 `active_request_id` 到新 request
   - 新 request 进入 `info_collection` 开始新一轮 Q&A
4. **拒绝路径**:
   - 用户说"不要" → 继续原 request
   - 不创建新 request

**已验证场景**:
- ✅ Scenario 5: Prerequisite Accept - 所有断言通过
- ✅ Scenario 6: Prerequisite Reject - 所有断言通过

### 3. 测试覆盖

**测试文件**: `test_info_collection_flow.py`

**已通过的场景**:
1. ✅ **Scenario 1**: Full Collection - 完整信息收集流程
2. ✅ **Scenario 2**: Early Handoff - 用户提前说"先用现有的继续"
3. ✅ **Scenario 3**: Multi-Turn - 多轮渐进式信息收集
4. ✅ **Scenario 4**: Prerequisite Detection - LLM 检测 prerequisite
5. ✅ **Scenario 5**: Prerequisite Accept - 用户同意处理 prerequisite
6. ✅ **Scenario 6**: Prerequisite Reject - 用户拒绝处理 prerequisite

**测试功能**:
- 选择性运行: `python test_info_collection_flow.py 5 6`
- Rate limit 保护: 场景间自动延迟 3 秒
- 详细输出: 每轮显示 state 变化、graph nodes、assertions

---

## 🚧 待实现功能和下一步开发

### 1. Prerequisite Completion 流程 (优先级: 高)

**当前缺失**:
- Prerequisite request 完成后，如何恢复 parent request？
- 如何检测 prerequisite request 已完成？
- Parent request 从 `paused` 恢复到 `collecting` 的逻辑

**需要实现**:

#### 1.1 Prerequisite Request 完成检测

**位置**: `route_from_turn_router` 或 `upstream_delegator`

**逻辑**:
```python
# 检测当前 active request 是否是 prerequisite request
if active_request.get("status") in ["validated", "completed"]:
    # 检查 pending_queue 中是否有 parent request
    parent_id = find_parent_request_from_pending_queue()
    if parent_id:
        # Prerequisite 完成，恢复 parent request
        trigger_parent_request_resume(parent_id)
```

**关键字段**:
- `pending_queue`: 存储被暂停的 parent request ID
- `prereq_gate.parent_request_id`: 记录 parent request 的引用

#### 1.2 Parent Request 恢复逻辑

**位置**: 新建 helper function 或在 `upstream_delegator` 中实现

**需要做的事**:
1. 从 `pending_queue` 中移除 parent request ID
2. 设置 `active_request_id` 为 parent request ID
3. 更新 parent request:
   - `status`: `"paused"` → `"collecting"`
   - `stage_detail`: 清除 `"paused_for_prerequisite_*"`
   - `stage_history`: 添加恢复记录
4. 更新 prerequisite request:
   - `status`: `"validated"` → `"completed"`
   - 记录完成时间

**示例 patch**:
```python
{
    "request_manager": {
        "active_request_id": parent_request_id,
        "pending_queue": [id for id in pending_queue if id != parent_request_id],
        "requests": {
            parent_request_id: {
                "status": "collecting",
                "stage_detail": "resumed_from_prerequisite",
                "stage_history": parent_history + [{
                    "from_stage": "paused",
                    "to_stage": "collecting",
                    "agent": "info_collection",
                    "timestamp": datetime.utcnow(),
                    "reason": f"Resumed after prerequisite {prereq_id} completed"
                }]
            },
            prereq_id: {
                "status": "completed",
                "completed_at": datetime.utcnow(),
            }
        }
    },
    "messages": [{
        "role": "assistant",
        "content": "好的，prerequisite 已处理完成。现在我们继续之前的任务。",
    }]
}
```

#### 1.3 测试场景

**新增 Scenario 7**: Prerequisite Completion & Parent Resume

```python
async def scenario_7_prerequisite_completion():
    """Scenario 7: Prerequisite 完成后恢复 parent request"""
    turns = [
        "我需要找一个护工",
        "预算的话我不太确定，因为我还没申请Medicaid，不知道能cover多少",
        "要，先帮我处理Medicaid申请",  # Accept prerequisite
        "在Illinois，我妈妈65岁",  # Answer prerequisite questions
        "收入很低，资产不多",  # More prerequisite info
        "好的，信息够了就开始吧",  # Finalize prerequisite
        # 此时应该自动恢复 parent request (caregiver search)
        "全天护理，希望下个月开始",  # Continue parent request
    ]
    
    # 验证:
    # - Turn 6 后: prerequisite request status = "completed"
    # - Turn 6 后: active_request_id 切换回 parent request
    # - Turn 6 后: parent request status = "collecting" (不再是 "paused")
    # - Turn 7: 继续收集 parent request 的信息
```

### 2. 优化和增强 (优先级: 中)

#### 2.2 Prerequisite Chain 支持

**场景**: Prerequisite 本身可能需要另一个 prerequisite

**示例**:
- User: "我需要找护工"
- System: 检测到需要 Medicaid
- User: "要，先申请 Medicaid"
- System: 检测到 Medicaid 需要 POA (Power of Attorney)
- User: "要，先办 POA"

**实现考虑**:
- `pending_queue` 需要支持栈结构（LIFO）
- 每个 prerequisite request 需要记录自己的 parent
- 完成时按顺序恢复

#### 2.3 Prerequisite 超时/放弃处理

**场景**: 用户在处理 prerequisite 过程中改变主意

**示例**:
- User 正在回答 Medicaid 问题
- User: "算了，不申请 Medicaid 了，直接找护工吧"

**需要实现**:
- 检测用户放弃 prerequisite 的信号
- 将 prerequisite request 标记为 `"aborted"`
- 恢复 parent request
- 清理 `pending_queue`

### 3. 代码质量改进 (优先级: 低)

#### 3.1 重构 info_collection_node

**当前问题**: 函数过长（~300 lines），逻辑复杂

**改进方向**:
- 拆分为多个 helper functions:
  - `handle_new_request()`
  - `handle_user_response()`
  - `handle_prerequisite_acceptance()`
  - `handle_prerequisite_rejection()`

#### 3.2 增强错误处理

**当前**: LLM 调用失败时使用 fallback

**改进**:
- 添加重试逻辑（exponential backoff）
- 更详细的错误日志
- 用户友好的错误消息

#### 3.3 性能优化

**考虑**:
- 缓存 collection plan（相似 request 可复用）
- 批量 LLM 调用（减少 API 请求次数）
- 异步并行处理（如同时生成 plan 和查询 similar requests）

---

## 📝 技术债务和已知问题

### 1. Rate Limit 处理

**当前方案**: 测试脚本中添加 3 秒延迟

**更好的方案**:
- 在 `TrackedAnthropicClient` 中实现自动 rate limit 检测和重试
- 使用 token bucket 算法控制请求速率
- 支持配置不同的 rate limit 策略

### 2. Pydantic 模型访问

**问题**: `RequestRecord` 是 Pydantic 模型，不能直接用 `.get()`

**当前解决**: 在测试代码中转换为 dict

**更好的方案**:
- 统一使用 Pydantic 模型的属性访问
- 或在 `apply_node_output` 中统一处理模型转换

### 3. 硬编码的语言检测

**当前**: 简单的 Unicode 范围检测中文

**改进**:
- 使用 `langdetect` 库
- 支持更多语言（西班牙语、韩语等）
- 从 user profile 中读取首选语言

---

## 🎯 下一步行动计划

### Phase 1: Prerequisite Completion (1-2 天)

1. ✅ 实现 prerequisite request 完成检测逻辑
2. ✅ 实现 parent request 恢复逻辑
3. ✅ 添加 Scenario 7 测试
4. ✅ 验证完整的 prerequisite 生命周期

### Phase 2: Post-Execution Routing Refinement (✅ 已完成 2026-02-21)

1. ✅ 修复 request lifecycle — `executed` 状态由 `turn_router` 决定，不再由执行 agent 设置
2. ✅ 新增 `task_complete` turn_mode 和 `task_complete_node`
3. ✅ `upstream_delegator` 支持 `resume_existing` action_type
4. ✅ `upstream_delegator` 获得 MCP 工具访问和全量 request 可见性
5. ✅ `info_collection` 支持 in-session 重复检测
6. ✅ 修复 `downstream_catcher` 无限循环 bug
7. ✅ 4 个 post-execution 场景全部通过

### Phase 3: 增强和优化

1. 添加 prerequisite 放弃处理逻辑
2. 重构 `info_collection_node`
3. 改进错误处理和日志

### Phase 4: 高级功能 (可选)

1. Prerequisite chain 支持
2. Rate limit 自动处理
3. 性能优化
4. 多语言增强

---

## ✅ Post-Execution Routing Refinement (2026-02-21)

### 背景与问题

在 Phase 1 完成后，系统在 **execution agent 交付结果之后** 的用户交互中存在严重的路由缺陷：

| 问题 | 描述 | 影响 |
|------|------|------|
| **Flaw 1: 盲目路由** | `turn_router` 不知道 request 的 lifecycle 状态（name/goal/status），无法区分用户是在确认结果、追问、还是切换话题 | 用户说"好的"后仍被路由到 `deep_search` 重新执行 |
| **Flaw 2: 过早标记 executed** | `deep_search_node` 和 `domain_expert_node` 在 MCP 调用成功后立即设置 `status=executed` | Lifecycle 状态由执行 agent 决定，而非由用户意图决定 |
| **Flaw 3: 无法恢复请求** | `upstream_delegator` 只能看到 active request，无法看到 paused/executed 的请求 | 用户说"继续之前找护工的事"时，系统创建新 request 而非恢复旧的 |
| **Flaw 4: downstream_catcher 循环** | `pending_handoff` 在 state 中不被清除，导致 `downstream_catcher → agent → downstream_catcher` 无限循环 | Graph 执行 recursion limit 错误 |

### 架构设计原则

```
核心原则: turn_router 是 lifecycle 决策者，execution agents 只负责交付结果

Before (错误):
  deep_search_node → MCP 成功 → 直接设 status=executed → turn_router 看不到

After (正确):
  deep_search_node → MCP 成功 → 交付结果（不改 status）
  → 用户回复 → turn_router 判断意图 → task_complete / continuation / new_intent
  → task_complete_node 设 status=executed
```

### 完整 Graph 路由流程图

```
                         ┌──────────────┐
                         │  User Input  │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │ turn_router  │  LLM 判断: continuation / new_intent / task_complete
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────────┐
              │                 │                      │
     ┌────────▼────────┐ ┌─────▼──────┐  ┌────────────▼────────────┐
     │ upstream_       │ │ agent node │  │   task_complete_node    │
     │ delegator       │ │ (continue) │  │ sets status=executed    │
     │ (new_intent)    │ └─────┬──────┘  │ sends acknowledgment    │
     └────────┬────────┘       │         └────────────┬────────────┘
              │                │                      │
              │         ┌──────▼───────┐              │
              │         │ downstream_  │              │
              │         │ catcher      │           ┌──▼──┐
              │         └──────┬───────┘           │ END │
              │                │                   └─────┘
     ┌────────▼────────┐      │
     │ route to agent  │   ┌──▼──┐
     │ or resume req   │   │ END │
     └─────────────────┘   └─────┘
```

### 实现的 9 个修复

#### R1: 移除 execution agents 的 `status=executed` 设置

**文件**: `graph.py` — `deep_search_node`, `domain_expert_node`

**变更**: 删除了 `deep_search_node` 和 `domain_expert_node` 中设置 `status=executed` 和 `stage_history` transition 的代码块。执行 agent 只负责交付结果（artifacts + messages），不再修改 request lifecycle。

```python
# REMOVED from deep_search_node and domain_expert_node:
# result["request_manager"] = {
#     "requests": { rid: { "status": "executed", ... } }
# }
```

#### R2: `turn_router` 新增 `task_complete` turn_mode

**文件**: `prompts.py` — `make_turn_mode_prompt`, `llm_turn_mode_decision`

**变更**:
1. Prompt 新增 request context（name, goal, status）让 LLM 了解当前任务状态
2. 新增第三个 `turn_mode`: `"task_complete"` — 用户确认/接受结果，无后续问题
3. 新增详细的判断指南和示例

```
turn_mode 三选一:
  "continuation"   — 用户在追问或回答问题
  "new_intent"     — 用户切换话题或回到之前的任务
  "task_complete"  — 用户确认结果，任务完成（"好的，知道了"、"谢谢"）
```

**判断规则**:

| 用户说 | request status | turn_mode |
|--------|---------------|-----------|
| "好的，知道了" | validated/executed | `task_complete` |
| "第一个结果的联系方式？" | validated/executed | `continuation` |
| "继续之前找护工的事" | any | `new_intent` |
| "照顾老人好累" | any | `new_intent` |
| "预算3000" | collecting | `continuation` |

#### R3: 新增 `task_complete_node`

**文件**: `graph.py` — 新函数 `task_complete_node`

**功能**:
- 设置 active request `status=executed`
- 记录 `stage_history` transition
- 发送确认消息（"好的，{request_name}已经处理完了。还有什么我可以帮你的吗？"）
- 连接到 `END`，post-graph `check_prereq_lifecycle` 处理 parent 恢复

**Graph 注册**:
```python
g.add_node("task_complete", task_complete_node)
g.add_edge("task_complete", END)
# route_from_turn_router 新增: if mode == "task_complete": return "task_complete"
```

#### R4: `upstream_delegator` 全量 request 可见性

**文件**: `prompts.py` — `llm_upstream_delegation`, `make_upstream_delegator_prompt`

**变更**: LLM prompt 现在包含 **所有 in-session requests** 的摘要（不仅是 active request）：

```
## All In-Session Requests:
- [req-abc] "Caregiver Search" (status=paused, collected_info="Location: Chicago, Budget: $3000...")
- [req-def] "Nursing Home Info" (status=created) ← ACTIVE
```

这让 LLM 能看到 paused/executed/completed 的请求，从而决定是否恢复已有请求。

#### R5: `upstream_delegator` 新增 `resume_existing` action_type

**文件**: `prompts.py` — prompt 新增 action_type; `graph.py` — handler 实现

**新增 action_type**: `"resume_existing"` — 用户想回到之前的请求

**Prompt 示例**:
```
action_type: "resume_existing"
resume_request_id: "req-abc"
recommended_agent: "info_collection"
current_request_action: "pause"
```

**Graph handler 逻辑**:
1. 设置 `active_request_id` 为 `resume_request_id`
2. 更新 resumed request: `status → collecting`, `awaiting_user_input → True`
3. 从 `pending_queue` 移除 resumed request
4. Pause 当前 active request（如果还在进行中）

#### R6: `upstream_delegator` MCP 工具访问

**文件**: `graph.py` — `upstream_delegator`

**变更**: 在 LLM 决策前调用 `fetch_similar_requests` 获取历史 request context：

```python
similar_historical = await fetch_similar_requests(
    user_id=user_id, request_text=user_text, top_k=3
)
```

Prompt 新增 `## Similar Historical Requests (from past sessions)` 部分，让 LLM 参考历史做出更好的路由决策。

#### R7: `info_collection` in-session 重复检测

**文件**: `graph.py` — `info_collection_node` CASE 1

**变更**: 在创建新 request 前，扫描 `request_manager.requests` 中所有 paused/collecting/validated/executed 的请求，用关键词匹配检测是否已有类似请求：

```python
# Before creating new request:
for existing_rid, existing_req in all_reqs.items():
    if keyword_overlap(user_text, existing_req.goal, existing_req.name):
        # Resume existing request instead of creating duplicate
        return resume_patch(existing_rid)
```

**效果**: 当 `upstream_delegator` 创建新任务并路由到 `info_collection` 时，如果已有匹配的 paused request，`info_collection` 会恢复它而非创建重复。

#### R8: 修复 `downstream_catcher` 无限循环

**文件**: `graph.py` — `downstream_catcher`, `route_after_catcher`

**Root Cause**: `downstream_catcher` 读取 `pending_handoff` 后返回 `{}`（不修改 state），导致 handoff 永远留在 state 中。`route_after_catcher` 每次都看到 handoff，路由回 agent，agent 完成后又到 catcher，形成无限循环。

**Fix**: 引入 `_catcher_next` 字段：
1. `downstream_catcher` 读取 `pending_handoff`，将 next agent 存入 `_catcher_next`，同时 **清除** `pending_handoff`
2. `route_after_catcher` 从 `_catcher_next` 读取路由决策

```python
# downstream_catcher:
return {
    "routing": {
        "pending_handoff": {"recommended_next_agent": None, "reason": ""},  # CLEAR
        "_catcher_next": nxt,  # SAVE decision
    }
}

# route_after_catcher:
nxt = routing.get("_catcher_next")  # READ from saved decision
```

### 测试结果

**测试文件**: `test_post_execution.py`

**运行**: `python test_post_execution.py [10|11|12|13|all]`

| Scenario | 用户行为 | 期望路由 | 结果 |
|----------|---------|---------|------|
| **10** | 确认结果: "好的，知道了" | `turn_router → task_complete` (不重新执行 deep_search) | ✅ PASSED |
| **11** | 追问结果: "第一个结果的联系方式？" | `turn_router → deep_search` (continuation, 同一 request) | ✅ PASSED |
| **12** | 切换请求: "继续之前找护工的事" | `turn_router → upstream_delegator → resume_existing` (恢复 paused request) | ✅ PASSED |
| **13** | 情绪表达: "照顾老人好累" | `turn_router → upstream_delegator → front_end` (情绪支持) | ✅ PASSED |

### Scenario 12 详细流程

这是最复杂的场景，验证了 request 切换和恢复的完整链路：

```
Turn 1: "我需要找一个护工"
  → info_collection 创建 "Caregiver Search" request (req-A)
  → status=created, awaiting_user_input=True

Turn 2: "在芝加哥，预算3000，全天护理，下个月开始"
  → info_collection 收集信息, readiness=can_proceed_but_incomplete

Turn 3: "没有其他要求了，帮我找吧"
  → info_collection finalize, status=validated, handoff → deep_search

Turn 4: "好的"
  → deep_search 执行 MCP 工具, 交付结果

Turn 5: "我还想了解一下养老院的情况"
  → turn_router: new_intent
  → upstream_delegator: new_unrelated_task
    - Pause req-A (status=paused)
    - Create req-B "Nursing Home Information"
  → info_collection 开始收集 req-B 信息

Turn 6: "算了，我还是想继续之前找护工的事"
  → turn_router: new_intent
  → upstream_delegator: resume_existing (resume_request_id=req-A)
    - Pause req-B
    - Resume req-A (status=collecting → validated)
  → info_collection 检测到 in-session duplicate, 恢复 req-A
  → 结果: req-A 重新激活, req-B paused, 无新 request 创建
```

### Request Lifecycle 状态机

```
                    ┌──────────┐
                    │ created  │  info_collection 创建
                    └────┬─────┘
                         │
                    ┌────▼─────┐
              ┌─────│collecting│◄────────────────────────┐
              │     └────┬─────┘                         │
              │          │                               │
         (pause)    (readiness=ready)              (resume)
              │          │                               │
         ┌────▼───┐ ┌───▼──────┐                  ┌─────┴────┐
         │ paused │ │validated │                  │ executed │
         └────────┘ └───┬──────┘                  └─────┬────┘
                        │                               │
                   (agent delivers)              (task_complete
                        │                        or resume)
                   ┌────▼─────┐                        │
                   │(no status│                  ┌─────▼─────┐
                   │ change)  │                  │ completed │
                   └──────────┘                  └───────────┘
                        │
                   (user says "好的")
                        │
                   ┌────▼──────┐
                   │task_complete│
                   │node sets   │
                   │executed    │
                   └────┬──────┘
                        │
                   ┌────▼──────┐
                   │check_prereq│  (post-graph)
                   │_lifecycle  │  恢复 parent if prereq
                   └───────────┘
```

### 修改的文件清单

| 文件 | 修改内容 |
|------|---------|
| `graph.py` | R1: 移除 `deep_search_node`/`domain_expert_node` 的 `status=executed`; R3: 新增 `task_complete_node`; R5: `upstream_delegator` 新增 `resume_existing` handler + MCP 调用; R7: `info_collection_node` in-session 重复检测; R8: 修复 `downstream_catcher` 循环; Graph wiring 更新 |
| `prompts.py` | R2: `make_turn_mode_prompt` 新增 request context + `task_complete` turn_mode; R4: `make_upstream_delegator_prompt` 新增 `all_requests` + `similar_historical_requests` + `resume_existing` action_type; `llm_upstream_delegation` 新增 `similar_requests` 参数 |
| `test_post_execution.py` | 4 个 post-execution 场景测试 (Scenarios 10-13) |

---

## 📚 相关文件

- **核心实现**: `graph.py` (info_collection_node, turn_router, upstream_delegator, task_complete_node, downstream_catcher)
- **Prompts**: `prompts.py` (make_turn_mode_prompt, make_upstream_delegator_prompt, llm_turn_mode_decision, llm_upstream_delegation)
- **测试**: `test_info_collection_flow.py` (Scenarios 1-6), `test_post_execution.py` (Scenarios 10-13), `test_prereq_completion.py` (Scenario 7), `test_stage_tracking.py` (Scenario 8)
- **State 模型**: `state_models.py` (RequestRecord, PrereqGate, etc.)
- **工具函数**: `routing_utils.py` (_user_yes_no, _user_says_decline_more_questions)
- **MCP**: `mcp_wrappers.py` (MCPClientManager, call_mcp_tool_patch), `servers/demo_mcp_server.py`
- **Retrieval**: `retrieval.py` (fetch_similar_requests, fetch_prior_qas_for_questions)
