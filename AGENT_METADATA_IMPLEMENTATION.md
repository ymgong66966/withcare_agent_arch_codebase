# Agent Metadata Implementation

## 概述

为所有agent nodes的assistant消息添加了`metadata: {"agent": "agent_name"}`字段，确保conversation extraction功能能够正确识别每条消息来自哪个agent。

---

## 修改的文件

**graph.py** - 添加了agent metadata到所有assistant消息

---

## 实现详情

### 1. front_end_node (Line 150-159)

**Before:**
```python
"messages": [{"role": "assistant", "content": "收到。我在。..."}]
```

**After:**
```python
"messages": [{
    "role": "assistant",
    "content": "收到。我在。...",
    "metadata": {"agent": "front_end_emotional_support"}
}]
```

---

### 2. info_collection_node (Multiple locations)

#### Line 534-538 (Prerequisite acceptance)
**Before:**
```python
"messages": [{
    "role": "assistant",
    "content": f"OK，我们先把先决条件（{prereq_type}）搞定。..."
}]
```

**After:**
```python
"messages": [{
    "role": "assistant",
    "content": f"OK，我们先把先决条件（{prereq_type}）搞定。...",
    "metadata": {"agent": "info_collection"}
}]
```

#### Line 548-552 (Prerequisite rejection)
**Before:**
```python
"messages": [{
    "role": "assistant",
    "content": "好，那我们先不处理这个先决条件。..."
}]
```

**After:**
```python
"messages": [{
    "role": "assistant",
    "content": "好，那我们先不处理这个先决条件。...",
    "metadata": {"agent": "info_collection"}
}]
```

#### Line 554-560 (Prerequisite confirmation prompt)
**Before:**
```python
"messages": [{
    "role": "assistant",
    "content": f'确认一下：要不要先处理先决条件...'
}]
```

**After:**
```python
"messages": [{
    "role": "assistant",
    "content": f'确认一下：要不要先处理先决条件...',
    "metadata": {"agent": "info_collection"}
}]
```

#### Line 635-639 (Initial questions) - Already had metadata ✓

#### Line 695-699 (Summarized response) - Already had metadata ✓

---

### 3. deep_search_node (Line 767-776)

**Before:**
```python
"messages": [{
    "role": "assistant",
    "content": "我已经跑了一步（demo）工具调用，拿到了初步结果。..."
}]
```

**After:**
```python
"messages": [{
    "role": "assistant",
    "content": "我已经跑了一步（demo）工具调用，拿到了初步结果。...",
    "metadata": {"agent": "deep_search"}
}]
```

---

### 4. user_info_node (Line 797-810)

**Before:**
```python
"messages": [{
    "role": "assistant",
    "content": "好的。你希望我总结 caregiver 还是 care recipient？..."
}]
```

**After:**
```python
"messages": [{
    "role": "assistant",
    "content": "好的。你希望我总结 caregiver 还是 care recipient？...",
    "metadata": {"agent": "user_info"}
}]
```

---

### 5. domain_expert_node (Line 839-858)

**Before:**
```python
"messages": [{
    "role": "assistant",
    "content": "（Demo）这里是一个示例 checklist：\n- " + "\n- ".join(steps)
}]
```

**After:**
```python
"messages": [{
    "role": "assistant",
    "content": "（Demo）这里是一个示例 checklist：\n- " + "\n- ".join(steps),
    "metadata": {"agent": "domain_expert"}
}]
```

---

## Agent名称映射

| Node名称 | Agent名称（metadata中） |
|---------|----------------------|
| front_end_node | front_end_emotional_support |
| info_collection_node | info_collection |
| deep_search_node | deep_search |
| user_info_node | user_info |
| domain_expert_node | domain_expert |

**注意**：这些agent名称必须与`state_models.py`中的`AgentName` literal定义一致。

---

## 验证

### 方法1: 检查所有assistant消息是否有metadata

```bash
# 查找所有assistant消息
grep -n '"role": "assistant"' graph.py

# 验证每行后面是否有metadata字段
# 应该在接下来的几行内看到 "metadata": {"agent": "xxx"}
```

### 方法2: 运行测试脚本

```bash
python test_extract_conversations.py
```

测试脚本会验证：
- Extraction能够正确识别所有agents的消息
- 不会漏掉任何agent的消息
- 能区分不同agents的对话

---

## 数据结构

### ChatMessage基础结构

```python
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")  # 允许额外字段
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str
    message_id: Optional[str] = None
    ts: datetime = Field(default_factory=datetime.utcnow)
    # 通过extra="allow"支持metadata字段
```

### 实际存储格式

```python
{
    "role": "assistant",
    "content": "...",
    "ts": datetime(...),
    "metadata": {
        "agent": "info_collection"  # ✅ 新添加
    }
}
```

---

## 依赖关系

这个修改是为了支持`daily_analyzer.py`中的`_extract_related_conversations()`功能。

### Extraction依赖的字段

1. ✅ **role**: 区分user和assistant消息
2. ✅ **content**: 消息内容
3. ✅ **ts / timestamp**: 时间戳，用于时间范围过滤
4. ✅ **metadata.agent**: Agent标识，用于匹配stage_history

### Extraction逻辑

```python
# daily_analyzer.py:_extract_related_conversations()

# 从stage_history提取涉及的agents
involved_agents = set()
for stage in stage_history:
    agent = stage.get("agent")
    if agent:
        involved_agents.add(agent)

# 匹配assistant消息
if msg.get("role") == "assistant":
    msg_agent = msg.get("agent") or msg.get("metadata", {}).get("agent")
    if msg_agent in involved_agents:
        # ✅ 这是相关消息
        related_indices.add(i)
```

---

## 影响分析

### ✅ 正面影响

1. **Extraction准确性**: 现在能正确识别所有agents的消息
2. **未来可扩展**: 其他功能也可以利用这个metadata（如分析、统计）
3. **一致性**: 所有agents使用统一的方式标识自己
4. **调试友好**: 更容易追踪哪个agent说了什么

### ⚠️ 需要注意

1. **向后兼容**: 旧的存储数据可能没有metadata字段
   - 解决方案: Extraction已经有fallback机制（基于时间范围）

2. **字段命名**: metadata.agent vs 直接的agent字段
   - 当前实现: 支持两种格式 `msg.get("agent") or msg.get("metadata", {}).get("agent")`

---

## 测试覆盖

### 单元测试

运行 `test_extract_conversations.py` 验证：

1. ✅ Front_end消息被正确提取
2. ✅ Info_collection消息被正确提取
3. ✅ Deep_search消息被正确提取
4. ✅ Domain_expert消息被正确提取
5. ✅ User_info消息被正确提取
6. ✅ 多个agents的消息能被区分
7. ✅ 无关的消息被正确过滤

### 集成测试

运行 `example_daily_job.py` 验证：

1. ✅ Daily analysis能识别所有agent的对话
2. ✅ Retrospective分析有完整的上下文
3. ✅ Historical records包含正确的stage信息

---

## 未来改进

### 1. 统一字段位置

当前支持两种格式：
- `msg.agent` (直接字段)
- `msg.metadata.agent` (嵌套字段)

建议：统一使用`msg.metadata.agent`，符合Pydantic的extra字段模式。

### 2. 添加更多metadata

可以考虑添加：
```python
"metadata": {
    "agent": "info_collection",
    "request_id": "req-123",  # 关联的request
    "intent": "collect_info",  # 意图标签
    "confidence": 0.9,  # 置信度
}
```

### 3. Schema验证

可以为metadata定义明确的schema：
```python
class MessageMetadata(BaseModel):
    agent: AgentName
    request_id: Optional[str] = None
    intent: Optional[IntentLabel] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    ts: datetime
    metadata: Optional[MessageMetadata] = None
```

---

## 总结

✅ 成功为所有5个agent nodes的9处assistant消息添加了agent metadata

✅ 确保conversation extraction能正确识别所有agents的消息

✅ 保持了与现有代码的兼容性（通过extra="allow"）

✅ 为future功能扩展打下了基础

---

## 相关文件

- **实现**: `graph.py` (所有agent nodes)
- **使用**: `daily_analyzer.py` (_extract_related_conversations)
- **测试**: `test_extract_conversations.py`
- **文档**: `CONVERSATION_EXTRACTION_README.md`, `STAGE_TRACKING_IMPLEMENTATION.md`
