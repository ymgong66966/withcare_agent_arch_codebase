# 踩坑记录与注意事项

> 本文档记录了在开发 WithCare Agent 架构过程中遇到的关键 bug、陷阱和设计教训。
> 主要涉及 **Pydantic 模型与 dict 交互**、**LangGraph 状态合并**、**LLM 路由决策** 三大类问题。

---

## 一、Pydantic 模型的 `None` 默认值陷阱

### 问题描述

`RequestRecord` 中定义了 `routing_hint: Optional[str] = None`。当通过 `build_request_patch` 创建请求时，没有显式设置 `routing_hint`，Pydantic 会将其初始化为 `None`。

后续在 `info_collection_node` 中读取时：

```python
# ❌ 错误写法
routing_hint = req.get("routing_hint", "deep_search")
# 返回 None，因为 key 存在，只是值为 None

# ✅ 正确写法
routing_hint = req.get("routing_hint") or "deep_search"
# None or "deep_search" → "deep_search"
```

### 根因

Python 的 `dict.get(key, default)` 只在 key **不存在** 时返回 default。如果 key 存在但值为 `None`，返回的是 `None`，不是 default。Pydantic 模型的 `Optional[str] = None` 字段在 `.model_dump()` 后会生成 `{"routing_hint": None}`，key 是存在的。

### 影响

`info_collection` 验证完成后设置 `pending_handoff.recommended_next_agent = None`，导致 `downstream_catcher` 无法路由到 `deep_search`，整个执行链断裂。用户需要额外发一条消息才能触发执行。

### 规则

> **凡是从 Pydantic 模型读取 Optional 字段并需要 fallback 的地方，一律使用 `or` 而非 `dict.get(key, default)`。**

### 受影响文件

- `graph.py` — `info_collection_node` CASE 2/3 的 `readiness == "ready"` 分支

---

## 二、LangGraph `Dict[str, Any]` 状态合并行为

### 问题描述

`GraphState` 使用 `TypedDict` 定义，没有 `Annotated` reducer：

```python
class GraphState(TypedDict, total=False):
    routing: Dict[str, Any]
    request_manager: Dict[str, Any]
    messages: List[Dict[str, Any]]
```

LangGraph 对 `Dict[str, Any]` 类型的默认合并行为是 **整体替换（replace）**，不是深度合并。

### 验证实验

```python
# node_a 返回
{"routing": {"agent": "a", "handoff": {"next": "b"}}}

# node_b 看到的 state["routing"] 是：
{"agent": "a", "handoff": {"next": "b"}}
# ← 完全是 node_a 的输出，不是与之前 state 的合并
```

### 影响

- 如果 node A 设置了 `routing.pending_handoff`，但 node B 返回 `routing` 时没有包含 `pending_handoff`，那么 `pending_handoff` 会被**丢失**。
- 每个 node 返回的 `routing` dict 必须包含所有需要保留的字段。

### 规则

> **每个 node 返回 `routing` 时，要意识到它会替换整个 `routing` dict。如果需要保留之前的字段（如 `pending_handoff`），必须在返回值中显式包含。**

### 与 `apply_node_output` 的区别

测试代码中使用 `apply_node_output`（在 `merge_utils.py` 中）来合并 node 输出到 Pydantic state。这个函数做的是 **逐字段 setattr**，不是替换：

```python
# merge_utils.py 中的逻辑
if "routing" in out:
    r = out.pop("routing") or {}
    for k, v in r.items():
        setattr(state.routing, k, v)  # ← 逐字段设置，不是替换
```

这意味着 **LangGraph 内部的状态合并** 和 **测试代码的 `apply_node_output`** 行为不同。在调试时要注意区分这两种合并方式。

---

## 三、`downstream_catcher` 的 `_catcher_next` 模式

### 设计

`downstream_catcher` 使用两步路由：
1. `downstream_catcher` 读取 `pending_handoff`，将决策保存到 `_catcher_next`，然后清除 `pending_handoff`
2. `route_after_catcher` 读取 `_catcher_next` 来决定下一个 node

### 为什么需要这个模式

如果 `downstream_catcher` 直接读取 `pending_handoff` 来路由，而路由目标 node 又设置了新的 `pending_handoff`，就会形成无限循环。通过 `_catcher_next` 中间变量，确保每个 `pending_handoff` 只被消费一次。

### 注意事项

- `downstream_catcher` 返回的 `routing` 必须同时包含 `pending_handoff`（清除）和 `_catcher_next`（设置）
- `route_after_catcher` 只读取 `_catcher_next`，不读取 `pending_handoff`

---

## 四、LLM 路由决策的不可靠性与确定性守卫

### 问题描述

`turn_router` 和 `upstream_delegator` 都依赖 LLM 做路由决策。但 LLM 的决策不总是正确的：

| 用户消息 | 期望 | LLM 实际决策 |
|---------|------|-------------|
| "没有其他要求了，帮我找吧" | continuation → info_collection | new_intent → upstream_delegator → 创建新请求 |
| "好的" (请求已 validated) | task_acknowledged | new_unrelated_task → 创建新请求 |
| "我还想了解一下养老院" | new_intent → upstream_delegator | 被确定性守卫误拦截（"了解" 匹配了 ack 短语） |

### 解决方案：确定性守卫 + LLM 决策

在关键路由点添加确定性守卫（deterministic guard），在 LLM 之前或之后做硬性检查：

#### 守卫 1：`route_from_turn_router` 中的 `awaiting_user_input` 守卫

```python
if mode == "new_intent":
    req = _get_active_request(state) or {}
    if req.get("awaiting_user_input") is True and req.get("status") in ("created", "collecting"):
        # 检查是否是明确的话题切换
        _switch_signals = ["算了", "不想", "别", "换", "之前", "继续之前", ...]
        is_explicit_switch = any(s in user_text.lower() for s in _switch_signals)
        if not is_explicit_switch:
            return "info_collection"  # 不是明确切换，继续 info_collection
    return "upstream_delegator"
```

**关键点**：
- 当 info_collection 正在收集信息时（`awaiting_user_input=True`），大部分 "new_intent" 其实是用户在回答问题
- 但如果用户说了明确的切换信号（"算了"、"之前"），必须尊重 new_intent 决策
- 切换信号列表要精心维护，避免误匹配

#### 守卫 2：`upstream_delegator` 中的确定性 ack 守卫

```python
if guard_req.get("status") in ("validated", "executed"):
    _ack_phrases = ["好的", "知道了", "谢谢", ...]
    user_stripped = user_text.strip().rstrip("。，！!.?,，")
    if len(user_stripped) <= 15 and any(p in user_stripped.lower() for p in _ack_phrases):
        # 直接返回 task_acknowledged，跳过 LLM
        return task_acknowledged_patch
```

**关键点**：
- 长度限制 `<= 15` 字符，避免误匹配长句子（如 "我还想了解一下养老院" 包含 "了解"）
- 短语列表要排除可能出现在正常句子中的词（如 "了解"、"可以"、"没有其他"）
- 只在请求已 validated/executed 时触发

### 规则

> **LLM 路由决策不可靠时，用确定性守卫兜底。但守卫的条件要严格，避免过度拦截。**
>
> **确定性守卫的设计原则：**
> 1. 条件要窄（多个条件 AND）
> 2. 短语匹配要考虑子串误匹配
> 3. 长度限制防止长句子被误拦截
> 4. 明确的切换信号要有 "逃生通道"

---

## 五、`task_complete_node` 的移除与 `task_acknowledged` 的引入

### 旧设计

```
turn_router → (task_complete) → task_complete_node → END
```

`turn_router` 有三种 mode：`continuation`、`new_intent`、`task_complete`。`task_complete_node` 是一个独立 node，负责设置 `status=executed`。

### 问题

- `task_complete` 只处理纯确认（"好的"），不处理 "用户切换话题但之前的任务已完成" 的情况
- 两个地方管理生命周期状态（`task_complete_node` 和 `upstream_delegator`），逻辑分散

### 新设计

```
turn_router → (new_intent) → upstream_delegator → (task_acknowledged) → END
```

- `turn_router` 只有两种 mode：`continuation` 和 `new_intent`
- 用户确认（"好的"）被归类为 `new_intent`，流向 `upstream_delegator`
- `upstream_delegator` 新增 `task_acknowledged` action_type 和 `mark_executed` current_request_action
- 所有生命周期状态管理集中在 `upstream_delegator`

### `upstream_delegator` 的 action_type 完整列表

| action_type | 说明 | current_request_action |
|------------|------|----------------------|
| `task_acknowledged` | 用户确认结果，任务完成 | `mark_executed` |
| `natural_progression` | 自然推进到下一阶段 | `continue` |
| `prerequisite_task` | 需要先完成前置任务 | `pause` |
| `new_unrelated_task` | 切换到无关新任务 | `pause` 或 `mark_executed` |
| `resume_existing` | 恢复之前暂停的任务 | `pause` |
| `same_agent_continue` | 继续当前对话流 | `continue` |

---

## 六、常见调试技巧

### 1. 在 `downstream_catcher` 添加日志

```python
import logging
_routing_dbg = state.get("routing") or {}
_ph_dbg = _routing_dbg.get("pending_handoff") or {}
logging.info(f"[downstream_catcher] pending_handoff={_ph_dbg}, _catcher_next={_routing_dbg.get('_catcher_next')}")
```

### 2. 检查 Pydantic 字段默认值

当 `req.get("field", "default")` 返回意外值时，检查 `state_models.py` 中对应字段的定义：

```python
# 如果定义是 Optional[str] = None
# 那么 .get("field", "default") 会返回 None，不是 "default"
```

### 3. 区分 LangGraph 内部状态和测试状态

- LangGraph 内部：`Dict[str, Any]` 整体替换
- `apply_node_output`：逐字段 setattr
- 调试时打印 LangGraph 内部状态（在 node 函数内打印 `state`），而非测试代码中的 Pydantic state

### 4. 测试 LangGraph 合并行为

```python
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict

class TestState(TypedDict, total=False):
    routing: Dict[str, Any]

def node_a(state):
    return {"routing": {"key1": "val1", "key2": "val2"}}

def node_b(state):
    print("node_b sees:", state.get("routing"))
    # → {"key1": "val1", "key2": "val2"}  (完全是 node_a 的输出)
    return {}
```

---

## 七、修改清单

| 文件 | 修改 | 原因 |
|------|------|------|
| `graph.py` L1075 | `req.get("routing_hint") or "deep_search"` | Pydantic None 默认值陷阱 |
| `graph.py` L108-126 | `route_from_turn_router` 添加 awaiting_user_input 守卫 | 防止 LLM 误判 new_intent |
| `graph.py` L217-261 | `upstream_delegator` 添加确定性 ack 守卫 | 防止为 "好的" 创建新请求 |
| `graph.py` L490-523 | 添加 `task_acknowledged` handler | 替代 `task_complete_node` |
| `graph.py` L529-549 | 添加 `mark_executed` handler | 非确认场景的任务完成标记 |
| `graph.py` 删除 | 移除 `task_complete_node` 及其图连线 | 功能合并到 upstream_delegator |
| `prompts.py` L639-707 | 移除 `task_complete` turn_mode | 简化为 continuation/new_intent |
| `prompts.py` L973-985 | 添加 `task_acknowledged` action_type | 明确的任务确认类型 |
| `prompts.py` L1016-1020 | 添加 `mark_executed` current_request_action | 任务完成标记选项 |

---

## 八、Bypass Agent 必须跳过 `upstream_delegator`，否则会创建不需要的 Request

### 问题描述 / Problem

When adding `quick_answer` — a lightweight agent that answers factual questions without creating a request — the initial implementation routed it through the standard path:

```
turn_router → (new_intent) → upstream_delegator → route_from_delegator → quick_answer
```

The problem: `upstream_delegator` runs **before** `route_from_delegator`. When the LLM decides `action_type="new_unrelated_task"` with `recommended_agent="quick_answer"`, the upstream_delegator node **creates a new request** in `request_manager` as part of its standard lifecycle management. By the time `route_from_delegator` sends the message to `quick_answer_node`, an unwanted request already exists.

### 根因 / Root Cause

The graph's routing architecture has two distinct layers:
1. **Node execution** — the node function runs and modifies state (e.g., `upstream_delegator` creates/pauses requests)
2. **Edge routing** — the conditional edge function decides where to go next

Guards in `route_from_delegator` (like skipping info_collection for `quick_answer`) only affect layer 2. They cannot undo state mutations from layer 1. If `upstream_delegator` creates a request, no downstream routing can un-create it.

### 解决方案 / Solution

Intercept bypass agents **before** they reach `upstream_delegator`:

```python
def route_from_turn_router(state):
    ...
    if mode == "new_intent":
        # quick_answer bypasses upstream_delegator entirely
        llm_recommended = routing.get("llm_recommended_agent")
        if llm_recommended == "quick_answer":
            return "quick_answer"
        ...
        return "upstream_delegator"
```

This pattern applies to any agent that should:
- Not create a request
- Not pause/modify existing requests
- Not go through info_collection

### 规则 / Rule

> **If a new agent should bypass the request lifecycle, it must be intercepted in `route_from_turn_router` before reaching `upstream_delegator`. Adding it only to `route_from_delegator` is insufficient — the delegator node will have already mutated state.**

### 检查清单 / Checklist for adding bypass agents

1. Add to `route_from_turn_router`: intercept on `new_intent` before `return "upstream_delegator"`
2. Add to `route_from_turn_router`: add to `agent_route_map` for `continuation` path
3. Add to `route_from_turn_router`: add to current-agent fallback chain
4. Add to `route_from_delegator`: skip info_collection guard (in case it's ever reached)
5. Add to `route_after_catcher`: mapping for downstream re-routing
6. Add to `build_graph()`: node, all three conditional edge dicts, downstream_catcher loop

---

## 九、Hardcoded Chinese Strings Break Language Adaptation

### 问题描述 / Problem

The agent always responded in Chinese regardless of user language. Even when the user wrote in English, hardcoded assistant messages like `"好的，还有什么我可以帮你的吗？"` and LLM prompt instructions like `"Respond in Chinese"` forced Chinese output.

### 根因 / Root Cause

Two sources of forced Chinese:
1. **~15 hardcoded Chinese strings in `graph.py`** — front_end_node, upstream_delegator ack responses, info_collection questions intro, prereq lifecycle messages, task_complete messages, fallback error messages
2. **LLM prompt instructions in `deep_search_prompts.py`** — every prompt ended with "Respond in Chinese" or "Provide your response in Chinese"

The LLM-generated responses (which are the majority of what the user sees) follow whatever language the prompt instructs, so even though conversation context was in English, the LLM would output Chinese.

### 解决方案 / Solution

1. Added `_detect_user_language(state)` helper — counts Chinese characters in the last user message; returns `"zh"` if >10% are Chinese, else `"en"`
2. Added `_msg(state, zh, en)` convenience wrapper for bilingual hardcoded messages
3. Updated all hardcoded messages to use `_msg()` or inline ternaries
4. Changed all LLM prompt language instructions from "Respond in Chinese" to "Match the language the user is writing in"
5. Added explicit `**LANGUAGE**` instructions to `make_collection_plan_prompt` (for questions) and `make_info_collection_summarize_prompt` (for suggested_response)

### 规则 / Rule

> **Never hardcode a response language. All hardcoded assistant messages must use `_msg(state, zh, en)`. All LLM prompts must include "Match the language the user is writing in" instead of specifying a fixed language.**

### 注意事项 / Notes

- `check_prereq_lifecycle` receives a Pydantic `UnifiedState`, not a dict — it can't use `_detect_user_language` directly. Language detection there reads from `state.messages` using attribute access instead.
- The keyword detection lists (switch signals, ack phrases, yes/no signals) should remain bilingual — they detect intent, not generate output.
