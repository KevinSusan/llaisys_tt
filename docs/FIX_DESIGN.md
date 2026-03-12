# 问题修复设计方案

> 日期：2026-03-13
> 作者：architect
> 基于：reviewer 审查报告（任务 #9）

---

## 修复总览

| # | 问题 | 优先级 | 修改文件 | 影响范围 |
|---|------|--------|----------|----------|
| 1 | `_session_worker` 无限增长 | 应修复 | `scheduler.py` | 调度器内部 |
| 2 | KV 路由 TOCTOU 竞态 | 可接受 | `scheduler.py` | 仅注释 |
| 3 | 异常过度吞没 + payload 污染 | 建议改进 | `scheduler.py` | `submit()` 方法 |
| 4 | 接口未被实际继承 | 建议改进 | `server.py`, `kv_cache_pool.py` | 类声明 |
| 5 | `request_stop` 两次加锁 | 建议合并 | `scheduler.py` | `request_stop()` |
| 6 | `_prompt_tokens` 泄漏到下游 | 建议清理 | `scheduler.py` | `submit()` 方法 |

---

## 问题 1：`_session_worker` 无限增长

### 根因

`_session_worker: Dict[str, int]` 在 `_choose_worker()` 和 `_bind_session()` 中只增不减。长期运行的服务会积累所有历史 session 映射，造成内存泄漏。

### 修复方案

将 `_session_worker` 从普通 `dict` 替换为带容量上限的 `OrderedDict`（LRU 语义）。

**API 变更：无。** 仅内部数据结构变化。

**新增构造参数：**

```python
def __init__(self, ..., max_sticky_sessions: int = 10000) -> None:
```

**实现要点：**

```python
from collections import OrderedDict

# __init__ 中
self._session_worker: OrderedDict[str, int] = OrderedDict()
self._max_sticky_sessions = max(100, int(max_sticky_sessions))

# 新增私有方法
def _touch_session(self, sid: str, worker_idx: int) -> None:
    """记录/更新 session->worker 映射，淘汰最旧条目。"""
    # 调用时已持有 self._lock
    if sid in self._session_worker:
        self._session_worker.move_to_end(sid)
    self._session_worker[sid] = worker_idx
    while len(self._session_worker) > self._max_sticky_sessions:
        self._session_worker.popitem(last=False)
```

**修改点：**

1. `_choose_worker()` 第 291, 321, 328 行：将 `self._session_worker[sid] = ...` 替换为 `self._touch_session(sid, ...)`
2. `_bind_session()` 第 339 行：同上
3. `debug_snapshot()` 新增字段 `"sticky_sessions": len(self._session_worker)`

**影响范围：** 仅 `scheduler.py` 内部，无外部 API 变化。

---

## 问题 2：KV 路由 TOCTOU 竞态

### 根因

`_choose_worker()` 查询 `kv_pool.query_prefix_len()` 到实际入队之间，其他线程可能改变 KV 状态。

### 决策：不修复，加注释

KV 感知路由本身是 best-effort 优化。TOCTOU 的最坏结果是路由到非最优 worker，不影响正确性。修复成本（全局锁或事务）远超收益。

**修改：** 在 `_choose_worker()` 的 KV 路由分支添加注释。

```python
# KV 感知路由是 best-effort：查询到入队之间 KV 状态可能变化，
# 最坏情况是路由到非最优 worker，不影响正确性。
```

---

## 问题 3：异常过度吞没 + payload 污染

### 根因

`submit()` 第 151-161 行有两个问题：

1. `except Exception: pass` 吞没所有异常，包括编程错误（如 `AttributeError`、`TypeError`），调试时无法发现问题。
2. `payload["_prompt_tokens"] = tokens` 修改了调用方传入的 dict（虽然 151 行做了 `payload = dict(payload)` 浅拷贝，但只在 tokens 非空时才拷贝）。

### 修复方案

**3a. 缩小 except 范围，添加日志：**

```python
import logging

logger = logging.getLogger(__name__)

# submit() 中
try:
    svc = self._services[0]
    if hasattr(svc, "tokenize_for_routing"):
        tokens = svc.tokenize_for_routing(payload)
        if tokens:
            payload = dict(payload)
            payload["_prompt_tokens"] = tokens
except Exception:
    logger.debug("tokenize_for_routing failed, falling back to default routing", exc_info=True)
```

保留 `except Exception` 是合理的，因为 `tokenize_for_routing` 可能依赖外部 tokenizer，各种异常都可能出现。关键是添加 `logger.debug` 使问题可追踪。

**3b. 确保 payload 始终拷贝后再添加内部字段：**

在 `submit()` 方法入口处统一浅拷贝：

```python
def submit(self, payload: Dict[str, Any], stream: bool) -> TaskHandle:
    payload = dict(payload)  # 防止修改调用方原始 dict

    if (self._kv_aware_routing and "_prompt_tokens" not in payload ...):
        ...
```

这也自然地解决了问题 6（`_prompt_tokens` 清理），见下文。

**影响范围：** 仅 `scheduler.py` 的 `submit()` 方法。

---

## 问题 4：接口未被实际继承

### 根因

`interfaces.py` 定义了 `IKVCachePool` 和 `IInferenceService`，但 `KVCachePool` 和 `ChatService` 都没有显式继承这些接口，依赖 duck typing。这降低了接口契约的强制性，也无法利用 `isinstance()` 检查。

### 修复方案

**4a. `KVCachePool` 继承 `IKVCachePool`：**

```python
# kv_cache_pool.py
from llaisys.interfaces import IKVCachePool

class KVCachePool(IKVCachePool):
    ...
```

`KVCachePool` 已实现所有 `IKVCachePool` 方法（`block_size`, `query_prefix_len`, `acquire_context`, `update_context`, `release_context`, `snapshot_stats`），无需新增任何方法。

注意：`block_size` 在 `IKVCachePool` 中是 `@property`，而 `KVCachePool.__init__` 中是 `self.block_size = int(block_size)` 直接赋值为实例属性。Python 中实例属性可以满足 `@property` 的读取语义，所以这不需要改动。

**4b. `ChatService` 继承 `IInferenceService`：**

```python
# server.py
from llaisys.interfaces import IInferenceService

class ChatService(IInferenceService):
    ...
```

`ChatService` 已实现所有必要方法。`kv_pool` 返回类型从 `KVCachePool` 改为 `IKVCachePool` 以匹配接口签名：

```python
@property
def kv_pool(self) -> "IKVCachePool":
    return self._kv_pool
```

**注意循环导入：** `interfaces.py` 使用 `TYPE_CHECKING` 导入 `AcquireResult`，`server.py` 导入 `interfaces.py`，`kv_cache_pool.py` 导入 `interfaces.py`。需要确认不会出现循环导入。

分析依赖链：
- `interfaces.py` → 仅在 `TYPE_CHECKING` 下导入 `kv_cache_pool.AcquireResult` ✅ 无运行时循环
- `kv_cache_pool.py` → 导入 `interfaces.IKVCachePool` ✅ `interfaces.py` 不运行时依赖 `kv_cache_pool`
- `server.py` → 导入 `interfaces.IInferenceService` ✅ 无新循环

**影响范围：** `kv_cache_pool.py` 和 `server.py` 的类声明行，无逻辑变更。

---

## 问题 5：`request_stop` 两次加锁

### 根因

`request_stop()` 第 183-186 行连续两次 `with self._lock`，应合并。

### 修复方案

```python
def request_stop(self, session_id: str) -> bool:
    sid = str(session_id or "").strip()
    if not sid:
        return False
    with self._lock:
        self._metrics["stop_requests"] += 1.0
        idx = self._session_worker.get(sid)
    if idx is not None:
        return bool(self._services[idx].request_stop(sid))
    ok = False
    for svc in self._services:
        ok = bool(svc.request_stop(sid)) or ok
    return ok
```

**影响范围：** 仅 `scheduler.py` 的 `request_stop()` 方法，无语义变化。

---

## 问题 6：`_prompt_tokens` 泄漏到下游

### 根因

`submit()` 第 158 行向 payload 添加 `_prompt_tokens`，第 168 行 `InferenceTask(payload=dict(payload), ...)` 会将此内部字段传递到 worker 和 `ChatService`，造成：
1. 下游处理不必要的数据
2. 如果下游解析 payload 时遇到未知字段可能产生困惑

### 修复方案

在路由决策完成后、创建 `InferenceTask` 前清理内部字段：

```python
def submit(self, payload: Dict[str, Any], stream: bool) -> TaskHandle:
    payload = dict(payload)  # 浅拷贝（问题 3b 已统一）

    # tokenize for routing...
    ...

    worker_idx = self._choose_worker(payload)

    # 清理路由专用的内部字段，不传递给下游
    payload.pop("_prompt_tokens", None)

    out_q: "queue.Queue[Any]" = queue.Queue()
    ...
```

**影响范围：** 仅 `scheduler.py` 的 `submit()` 方法。

---

## 实施顺序

建议按以下顺序实施，每步可独立验证：

1. **问题 5**（合并加锁）— 最简单，零风险
2. **问题 6 + 3b**（payload 拷贝 + 清理）— 一起做，改动集中在 `submit()`
3. **问题 3a**（添加 logger）— 需要在文件顶部添加 `import logging`
4. **问题 1**（LRU session map）— 最大改动，需要测试
5. **问题 4**（接口继承）— 涉及两个文件，需要验证导入
6. **问题 2**（添加注释）— 最后做，无代码变更

---

## 测试要点

| 问题 | 测试方法 |
|------|----------|
| #1 | 单测：创建超过 `max_sticky_sessions` 个 session，验证旧条目被淘汰，dict 大小不超限 |
| #3 | 单测：mock `tokenize_for_routing` 抛异常，验证 `submit()` 正常完成且 log 输出 |
| #4 | 单测：`isinstance(ChatService(...), IInferenceService)` 返回 True；`isinstance(KVCachePool(...), IKVCachePool)` 返回 True |
| #5 | 现有测试覆盖 `request_stop`，回归即可 |
| #6 | 单测：`submit()` 后检查原始 payload 不含 `_prompt_tokens`；检查 `InferenceTask.payload` 不含 `_prompt_tokens` |
