# Conversation Extraction Feature

## 概述

实现了从全天对话中提取与特定request相关的对话片段的功能。这个功能是historical request tracking系统的关键组件，用于为retrospective分析提供精确的上下文。

---

## 问题背景

在daily analysis中，我们需要分析每个request的完成状态和用户满意度。但是：

1. **全天对话混杂**：一天内可能有多个requests，对话会交织在一起
2. **时间不连续**：一个request可能被pause然后resume，导致相关对话分散
3. **需要上下文**：要理解agent的回复，需要知道之前的用户输入
4. **Agent识别**：需要知道每个时间点是哪个agent在说话

---

## 解决方案

### 核心方法：`_extract_related_conversations()`

**位置**: `daily_analyzer.py` line 407-507

**输入**:
- `conversations: List[Dict[str, Any]]` - 全天的所有对话
- `request_dict: Dict[str, Any]` - Request的完整信息，包括：
  - `created_at`: Request创建时间
  - `last_touched_at`: 最后更新时间
  - `stage_history`: 包含所有涉及的agents和时间戳

**输出**:
- `List[Dict[str, Any]]` - 与这个request相关的对话列表（按时间顺序）

---

## 实现逻辑

### 1. 时间范围过滤

```python
# 使用request的活跃时间范围
created_at = request_dict.get("created_at")
last_touched_at = request_dict.get("last_touched_at")

# 添加5分钟buffer，捕获上下文
time_buffer = timedelta(minutes=5)
if created_at - time_buffer <= msg_time <= last_touched_at + time_buffer:
    # 在时间范围内
```

### 2. Agent匹配

```python
# 从stage_history中提取所有涉及的agents
stage_history = request_dict.get("stage_history", [])
involved_agents = set()

for stage in stage_history:
    agent = stage.get("agent")
    if agent:
        involved_agents.add(agent)

# 匹配assistant消息
if msg.get("role") == "assistant":
    msg_agent = msg.get("agent") or msg.get("metadata", {}).get("agent")
    if msg_agent in involved_agents:
        # 这是相关消息
```

### 3. 上下文扩展

```python
# 对于每个匹配的assistant消息
if msg_agent in involved_agents:
    related_indices.add(i)
    # 包含前一条用户消息（触发）
    if i > 0:
        related_indices.add(i - 1)
    # 包含后一条用户消息（回应）
    if i + 1 < len(conversations):
        related_indices.add(i + 1)
```

### 4. 连续性保持

```python
# 如果user消息夹在两个相关消息之间，也包含
elif msg.get("role") == "user":
    if i > 0 and (i - 1) in related_indices:
        related_indices.add(i)
    if i + 1 < len(conversations) and (i + 1) in related_indices:
        related_indices.add(i)
```

---

## 示例场景

### Scenario 1: 单个连续的request

**输入 - 全天对话**:
```
10:00 [USER]: 我需要找护工
10:01 [ASSISTANT - info_collection]: 好的，请告诉我预算
10:05 [USER]: 预算2000-3000
10:06 [ASSISTANT - deep_search]: 收到，我现在搜索
10:10 [ASSISTANT - deep_search]: 找到3个候选人
11:00 [USER]: 今天天气真好  # 无关对话
11:01 [ASSISTANT - front_end]: 是的，天气不错
```

**Request信息**:
- created_at: 10:00
- last_touched_at: 10:10
- stage_history: [info_collection, deep_search]

**输出 - 提取的对话**:
```
10:00 [USER]: 我需要找护工
10:01 [ASSISTANT - info_collection]: 好的，请告诉我预算
10:05 [USER]: 预算2000-3000
10:06 [ASSISTANT - deep_search]: 收到，我现在搜索
10:10 [ASSISTANT - deep_search]: 找到3个候选人
# 11:00-11:01的无关对话被过滤掉
```

---

### Scenario 2: Request被pause然后resume

**输入 - 全天对话**:
```
10:00 [USER]: 我需要找护工
10:01 [ASSISTANT - info_collection]: 好的，请告诉我预算
10:05 [USER]: 预算2000-3000
10:06 [ASSISTANT - deep_search]: 收到，我现在搜索
11:00 [USER]: 等等，我还没申请Medicaid  # 触发pause
11:01 [ASSISTANT - info_collection]: 好的，我们先处理Medicaid
... [Medicaid相关对话] ...
12:30 [USER]: Medicaid好了，继续找护工
12:31 [ASSISTANT - deep_search]: 好的，继续搜索护工
```

**Request 1 信息 (找护工)**:
- created_at: 10:00
- last_touched_at: 12:31
- stage_history:
  - 10:00 info_collection
  - 10:06 deep_search
  - 11:00 paused (upstream_delegator)
  - 12:31 deep_search (resumed)

**输出 - 提取的对话**:
```
10:00 [USER]: 我需要找护工
10:01 [ASSISTANT - info_collection]: 好的，请告诉我预算
10:05 [USER]: 预算2000-3000
10:06 [ASSISTANT - deep_search]: 收到，我现在搜索
11:00 [USER]: 等等，我还没申请Medicaid  # 包含，因为触发了pause
12:30 [USER]: Medicaid好了，继续找护工  # 包含，因为resume
12:31 [ASSISTANT - deep_search]: 好的，继续搜索护工
# Medicaid相关对话被过滤（属于另一个request）
```

---

### Scenario 3: 多个交织的requests

**输入 - 全天对话**:
```
10:00 [USER]: 我需要找护工
10:01 [ASSISTANT - info_collection]: 好的，请告诉我预算
11:00 [USER]: 等等，先帮我查Medicaid
11:01 [ASSISTANT - info_collection]: 好的，我们先处理Medicaid
11:05 [USER]: 我住伊利诺伊州
11:06 [ASSISTANT - domain_expert]: 这是伊利诺伊州的Medicaid流程
12:00 [USER]: 谢谢，现在回到找护工
12:01 [ASSISTANT - deep_search]: 好的，继续搜索护工
```

**提取逻辑**:
- Request 1 (找护工): 会提取 10:00-10:01 和 12:00-12:01
- Request 2 (Medicaid): 会提取 11:00-11:06
- 每个request的提取结果互不干扰

---

## Conversation格式

### 输入格式

```python
{
    "role": "user" | "assistant" | "system",
    "content": str,
    "ts": datetime,  # ChatMessage.ts字段
    "metadata": {  # ✅ 所有assistant消息都有此字段
        "agent": str,  # Agent标识
    }
}
```

**重要**: 所有agent nodes现在都会在assistant消息中添加`metadata.agent`字段。参见`AGENT_METADATA_IMPLEMENTATION.md`。

### 关键字段

1. **ts** (ChatMessage.ts): 消息的时间戳
   - 用于时间范围过滤
   - 必须是datetime对象
   - Note: Extraction代码同时支持`timestamp`字段作为fallback

2. **metadata.agent**: Agent标识（仅assistant消息）
   - ✅ 所有agent nodes都添加了此字段
   - 从`message.metadata.agent`读取
   - 用于匹配stage_history中的agents
   - Extraction代码也支持直接的`message.agent`字段作为fallback

3. **role**: 消息角色
   - "user": 用户消息
   - "assistant": 助手消息（✅ 包含metadata.agent字段）

### Agent标识映射

| Agent Node | metadata.agent值 |
|-----------|-----------------|
| front_end_node | "front_end_emotional_support" |
| info_collection_node | "info_collection" |
| deep_search_node | "deep_search" |
| user_info_node | "user_info" |
| domain_expert_node | "domain_expert" |

---

## Fallback逻辑

如果提取失败或找不到匹配的消息，使用以下fallback策略：

### 1. 时间范围fallback
```python
# 如果没有agent匹配，至少返回时间范围内的消息
for msg in conversations:
    msg_time = msg.get("timestamp") or msg.get("ts")
    if msg_time and created_at <= msg_time <= last_touched_at:
        related.append(msg)
```

### 2. 最后N条fallback
```python
# 如果完全找不到，返回最后10条消息
return conversations[-10:]
```

---

## 测试

运行测试脚本验证功能：

```bash
python test_extract_conversations.py
```

### 测试场景

1. **Test 1: 单个request的完整生命周期**
   - 验证能提取从创建到完成的所有相关对话
   - 验证能包含正确的user和assistant消息

2. **Test 2: 多个交织的requests**
   - 验证能正确区分不同requests的对话
   - 验证不会混淆不同requests的消息

3. **Test 3: Paused/resumed request**
   - 验证能处理被暂停然后恢复的request
   - 验证能包含pause和resume时的关键消息

4. **Test 4: 过滤无关消息**
   - 验证能正确过滤掉无关的对话
   - 验证不会包含其他agents的消息

---

## 集成到Daily Analysis

这个功能已经集成到`DailyRequestAnalyzer._analyze_single_request()`中：

```python
# Extract related conversations
related_convs = self._extract_related_conversations(
    conversations, request_dict
)

# Build prompt with only related conversations
prompt = self._build_single_request_analysis_prompt(
    request_dict=request_dict,
    stage_history=stage_history,
    related_conversations=related_convs,  # ✅ 只传入相关对话
    overall_insights=overall_insights,
)
```

**好处**:
- LLM只需要分析相关的对话，不会被无关信息干扰
- 减少prompt长度，提高LLM分析速度和准确度
- 提供完整的上下文，帮助LLM准确判断completion status

---

## 性能考虑

### 时间复杂度
- O(n * m) 其中：
  - n = conversations总数
  - m = stage_history长度（通常很小）

### 优化建议

1. **Agent set预处理**: 提前构建involved_agents集合（O(m)）
2. **时间范围预过滤**: 先按时间过滤，再按agent匹配（减少比较次数）
3. **索引集合**: 使用set存储related_indices（O(1)查找）

### 内存占用
- 主要内存：conversations列表（全天对话）
- 额外内存：related_indices set（通常很小）
- 输出：related_conversations列表（通常是全天对话的子集）

---

## 未来改进

### 1. Semantic Matching
当前是基于时间+agent的硬匹配。可以增加：
- 关键词匹配（从request goal提取关键词）
- Embedding相似度匹配（语义相关的对话）

### 2. Gap Threshold配置
当前使用固定的5分钟buffer。可以：
- 根据request类型调整buffer大小
- 支持配置参数

### 3. Multi-chunk返回
当前返回一个连续的列表。可以：
- 返回多个独立的chunks（明确标识每个chunk的时间范围）
- 为每个chunk添加元数据（agent, stage, duration）

### 4. 缓存机制
如果同一天多次运行analysis：
- 缓存提取结果
- 避免重复计算

---

## 相关文件

- **实现**: `daily_analyzer.py` (line 407-507)
- **测试**: `test_extract_conversations.py`
- **文档**: `STAGE_TRACKING_IMPLEMENTATION.md`
- **使用示例**: `example_daily_job.py`

---

## 总结

这个功能通过以下方式提升了daily analysis的质量：

1. ✅ **精确上下文**: 只提取相关对话，避免噪音
2. ✅ **完整生命周期**: 捕获request从创建到完成的所有关键对话
3. ✅ **多request支持**: 正确处理交织的多个requests
4. ✅ **Pause/resume支持**: 处理被暂停和恢复的requests
5. ✅ **Agent识别**: 利用stage_history准确识别相关agents
6. ✅ **Fallback机制**: 确保即使匹配失败也能提供合理的结果

这为retrospective analysis提供了高质量的输入，帮助LLM准确判断request的completion status和user satisfaction。
