# ChatService 职责拆分设计方案

> 日期：2026-03-13
> 作者：architect
> 基于：docs/new.md Section 1 职责分析 + server.py 完整审阅

---

## 1. 现状分析

`ChatService`（server.py 第 20-671 行，约 650 行）承担了 5 个明确可分离的职责：

| 职责 | 方法 | 行数 | 状态 |
|------|------|------|------|
| **会话管理** | `_extract_messages`, `_save_context_messages`, `_get_cancel_event`, `request_stop`, `_clear_stop` | ~80 | 纯状态管理，与模型无关 |
| **KV 运行时复用** | `_release_native_kv_context`, `_find_native_kv_context_for_prefix`, `_bind_native_kv_context_for_request`, `_export_native_kv_context_after_request`, `kv_debug_snapshot` | ~100 | 依赖模型 C API，实验特性 |
| **推理执行** | `_decode_next`, `_prefill_next`, `_iter_generate_ids`, `_eos_token` | ~100 | 核心推理循环 |
| **请求编排** | `_prepare_request`, `generate`, `stream`, `generate_packed_non_stream` | ~250 | 组合上述三者 + KVCachePool |
| **文本处理** | `_render_prompt`, `_postprocess_text`, `_init_chat_template_tokenizer`, `tokenize_for_routing` | ~70 | tokenizer / template 逻辑 |

**核心问题：** 会话管理和 KV 复用是独立的关注点，却与推理执行混在一个类中，导致：
- 难以单独测试会话逻辑
- KV 复用是实验特性，开关逻辑散布在多个方法中
- `generate()` 和 `stream()` 的代码高度重复（~80% 相同结构）

---

## 2. 拆分方案

### 2.1 模块划分

```
python/llaisys/
├── server.py              # ChatService (瘦身后) + ChatHandler + main()
├── session_manager.py     # [新增] SessionManager
├── kv_runtime_bridge.py   # [新增] KVRuntimeBridge
├── kv_cache_pool.py       # [不变] KVCachePool
├── scheduler.py           # [不变] InferenceScheduler
└── interfaces.py          # [微调] 新增 ISessionManager 接口
```

### 2.2 类图（拆分后）

```
                    IInferenceService (接口)
                          │
                          │ implements
                          ▼
┌─────────────────────────────────────────────────┐
│                  ChatService                     │
│                                                  │
│  持有:                                           │
│    session_mgr: SessionManager                   │
│    kv_bridge:   KVRuntimeBridge                  │
│    kv_pool:     KVCachePool                      │
│    model:       Qwen2                            │
│    tokenizer:   Tokenizer                        │
│                                                  │
│  公开方法 (IInferenceService):                    │
│    generate(payload) → Dict                      │
│    stream(payload) → Iterable[Dict]              │
│    request_stop(session_id) → bool               │
│    kv_debug_snapshot(session_id) → Dict           │
│    kv_pool → IKVCachePool                        │
│    generate_packed_non_stream(payloads) → List    │
│    tokenize_for_routing(payload) → List[int]     │
│                                                  │
│  私有方法 (推理核心):                             │
│    _decode_next(...)                             │
│    _prefill_next(...)                            │
│    _iter_generate_ids(...)                       │
│    _eos_token()                                  │
│    _prepare_request(...)                         │
│    _render_prompt(...)                           │
│    _postprocess_text(...)                        │
└──────────────┬──────────────┬────────────────────┘
               │              │
    ┌──────────▼──┐    ┌──────▼──────────┐
    │SessionManager│   │KVRuntimeBridge  │
    │             │    │                 │
    │ 会话消息存储 │    │ 原生 KV 上下文   │
    │ 取消事件管理 │    │ 绑定/导出/查找   │
    │ 分叉编辑提取 │    │ 调试快照        │
    └─────────────┘    └─────────────────┘
```

---

## 3. 各模块详细设计

### 3.1 SessionManager（session_manager.py）

**职责：** 会话消息历史管理 + 取消事件管理

```python
class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._context_messages: Dict[str, List[Dict[str, str]]] = {}
        self._cancel_events: Dict[str, threading.Event] = {}

    def extract_messages(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """从 payload 提取 context_id 和消息列表。

        处理三种输入模式：
        - edit_from_session_id: 分叉编辑
        - messages: 直接传入消息列表
        - prompt: 追加到现有会话历史

        Returns:
            (context_id, messages)
        """

    def save_messages(
        self, context_id: str, messages: List[Dict[str, str]]
    ) -> None:
        """保存会话消息历史"""

    def get_messages(self, context_id: str) -> List[Dict[str, str]]:
        """获取会话消息历史（返回副本）"""

    def get_cancel_event(self, context_id: str) -> threading.Event:
        """获取或创建取消事件"""

    def request_stop(self, context_id: str) -> bool:
        """设置取消事件"""

    def clear_stop(self, context_id: str) -> None:
        """清除取消事件"""
```

**从 ChatService 迁移的方法：**

| ChatService 方法 | SessionManager 方法 | 变化 |
|------------------|--------------------|----|
| `_extract_messages()` | `extract_messages()` | 去掉下划线前缀，变为公开方法 |
| `_save_context_messages()` | `save_messages()` | 重命名 |
| `_get_cancel_event()` | `get_cancel_event()` | 去掉下划线前缀 |
| `request_stop()` | `request_stop()` | 直接迁移 |
| `_clear_stop()` | `clear_stop()` | 去掉下划线前缀 |

**锁策略：** `SessionManager` 拥有自己的 `threading.Lock()`，与 ChatService 的 `_model_lock` 独立。这保留了现有的锁分离设计（当前 `_context_lock` 与 `_model_lock` 就是分开的）。

---

### 3.2 KVRuntimeBridge（kv_runtime_bridge.py）

**职责：** 管理原生 C++ KV 上下文的生命周期（绑定、导出、查找、释放、调试）

```python
class KVRuntimeBridge:
    def __init__(self, model: "Qwen2", enabled: bool = False) -> None:
        self._model = model
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._native_kv_contexts: Dict[str, Any] = {}
        self._native_kv_tokens: Dict[str, Tuple[int, ...]] = {}
        self._last_kv_bind_debug: Dict[str, Dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def bind_for_request(
        self,
        context_id: str,
        prompt_ids: List[int],
        prefix_len: int,
    ) -> None:
        """为当前请求绑定最优 KV 上下文到模型。

        查找顺序：
        1. 同 context_id 的原生上下文
        2. 前缀匹配的 donor 上下文
        3. 无匹配 → set_kv_context(None)
        """

    def export_after_request(
        self,
        context_id: str,
        tokens: List[int],
        block_size: int,
    ) -> None:
        """请求完成后导出 KV 上下文供后续复用"""

    def release(self, context_id: str) -> None:
        """释放指定会话的原生 KV 上下文"""

    def debug_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """返回 KV 运行时调试信息"""
```

**从 ChatService 迁移的方法：**

| ChatService 方法 | KVRuntimeBridge 方法 | 变化 |
|------------------|---------------------|----|
| `_bind_native_kv_context_for_request()` | `bind_for_request()` | 简化名称 |
| `_export_native_kv_context_after_request()` | `export_after_request()` | 简化名称，`block_size` 作为参数传入 |
| `_release_native_kv_context()` | `release()` | 简化名称 |
| `_find_native_kv_context_for_prefix()` | `_find_for_prefix()` | 内部方法保留 |
| `kv_debug_snapshot()` 的 native 部分 | `debug_snapshot()` | 拆出 native 相关字段 |

**关键设计决策：** `KVRuntimeBridge` 持有 `model` 引用，因为它需要调用 `model.set_kv_context()`, `model.kv_context_create()`, `model.export_kv_context()` 等 C API。这是不可避免的耦合——它就是模型 KV 状态的桥接层。

---

### 3.3 ChatService（瘦身后）

**保留在 ChatService 中的职责：**
1. 推理执行（`_decode_next`, `_prefill_next`, `_iter_generate_ids`, `_eos_token`）
2. 请求编排（`_prepare_request`, `generate`, `stream`, `generate_packed_non_stream`）
3. 文本处理（`_render_prompt`, `_postprocess_text`, `tokenize_for_routing`）
4. `IInferenceService` 接口实现（门面委托）

**构造函数变化：**

```python
class ChatService(IInferenceService):
    def __init__(
        self,
        model: Qwen2,
        tokenizer: llaisys.Tokenizer,
        model_path: Optional[str] = None,
        enable_kv_runtime_reuse: bool = False,
        block_size: int = 64,
        max_blocks: int = 4096,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._model_lock = threading.RLock()

        # 文本处理
        self._chat_template_tokenizer = self._init_chat_template_tokenizer(model_path)
        self._filter_tokens = (...)
        self._filter_patterns = [...]

        # 委托组件
        self._session_mgr = SessionManager()
        self._kv_bridge = KVRuntimeBridge(model, enabled=enable_kv_runtime_reuse)
        self._kv_pool = KVCachePool(
            block_size=block_size,
            max_blocks=max_blocks,
            max_bytes=max_bytes,
        )
        self._active_tokens: List[int] = []
```

**接口方法委托示例：**

```python
def request_stop(self, context_id: str) -> bool:
    return self._session_mgr.request_stop(context_id)

def kv_debug_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
    native_info = self._kv_bridge.debug_snapshot(session_id)
    native_info["kv_pool"] = self._kv_pool.snapshot_stats()
    return native_info
```

**`generate()` 方法简化（示意）：**

```python
def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    context_id, messages, prompt_ids, sampling, max_new_tokens = self._prepare_request(payload)
    cancel_event = self._session_mgr.get_cancel_event(context_id)
    self._session_mgr.clear_stop(context_id)

    with self._model_lock:
        acquire = self._kv_pool.acquire_context(context_id, prompt_ids)
        self._kv_bridge.bind_for_request(context_id, prompt_ids, acquire.prefix_len)
        generated_ids: List[int] = []
        try:
            for token_id in self._iter_generate_ids(...):
                generated_ids.append(int(token_id))
            cancelled = cancel_event.is_set()
            if cancelled:
                self._active_tokens = list(prompt_ids)
                self._kv_pool.update_context(context_id, prompt_ids)
            else:
                self._kv_pool.update_context(context_id, self._active_tokens)
                self._kv_bridge.export_after_request(
                    context_id, self._active_tokens, self._kv_pool.block_size
                )
        except Exception:
            self._kv_pool.release_context(context_id)
            self._kv_bridge.release(context_id)
            raise

    response_text = self._postprocess_text(self.tokenizer.decode(generated_ids))
    if cancelled:
        self._session_mgr.clear_stop(context_id)
        return {"session_id": context_id, "response": response_text, "stopped": True, ...}
    messages = list(messages)
    messages.append({"role": "assistant", "content": response_text})
    self._session_mgr.save_messages(context_id, messages)
    self._session_mgr.clear_stop(context_id)
    return {"session_id": context_id, "response": response_text, ...}
```

---

## 4. 接口兼容性

### 4.1 IInferenceService 接口 —— 无变化

`ChatService` 仍然是 `IInferenceService` 的唯一实现类。所有公开方法签名不变：

| 方法 | 签名 | 状态 |
|------|------|------|
| `generate(payload)` | `Dict → Dict` | 不变 |
| `stream(payload)` | `Dict → Iterable[Dict]` | 不变 |
| `request_stop(session_id)` | `str → bool` | 委托到 SessionManager |
| `kv_debug_snapshot(session_id)` | `Optional[str] → Dict` | 组合 KVRuntimeBridge + KVCachePool |
| `kv_pool` | `→ IKVCachePool` | 不变 |
| `generate_packed_non_stream(payloads)` | `List[Dict] → Optional[List[Dict]]` | 不变 |
| `tokenize_for_routing(payload)` | `Dict → Optional[List[int]]` | 不变 |

### 4.2 HTTP API —— 无变化

`ChatHandler` 仅依赖 `InferenceScheduler`，不直接依赖 `ChatService`。以下端点不受影响：

- `POST /chat` — 通过 scheduler.submit()
- `POST /v1/chat/completions` — 同上
- `POST /chat/stop` — 通过 scheduler.request_stop()
- `GET /debug/kv` — 通过 scheduler.kv_debug_snapshot()
- `GET /debug/scheduler` — 通过 scheduler.debug_snapshot()
- `GET /health` — 无依赖

### 4.3 main() 函数 —— 无变化

`main()` 仅调用 `ChatService(model, tokenizer, ...)`，构造参数不变。

---

## 5. 不拆分的内容（及理由）

| 候选拆分 | 决策 | 理由 |
|----------|------|------|
| 文本处理独立为 `TextProcessor` | **不拆** | 仅 4 个方法，拆出后 ChatService 需要额外依赖，收益不足 |
| 推理执行独立为 `InferenceEngine` | **不拆** | `_iter_generate_ids` 与 `_active_tokens`、KV pool、KV bridge 紧密交互，拆出需要大量参数传递 |
| `generate()` 与 `stream()` 合并去重 | **不拆** | 二者逻辑相似但流式 yield 与非流式 return 的控制流不同，强行合并会引入复杂的回调/策略模式，得不偿失 |
| `ChatHandler` 拆到独立文件 | **不拆** | 它仅依赖 scheduler，已足够薄，且与 `main()` 在同一文件更便于阅读 |

---

## 6. 依赖关系与导入

```
interfaces.py          ← 无依赖
kv_cache_pool.py       ← interfaces.py (IKVCachePool)
session_manager.py     ← 无依赖（纯 Python 状态管理）
kv_runtime_bridge.py   ← 无依赖（接收 model 实例，不导入模型模块）
server.py              ← session_manager, kv_runtime_bridge, kv_cache_pool, interfaces, models, scheduler
scheduler.py           ← interfaces (TYPE_CHECKING)
```

无循环导入。`kv_runtime_bridge.py` 通过构造函数接收 `model` 实例（依赖注入），不需要导入 `Qwen2`。

---

## 7. 实施步骤

| 步骤 | 内容 | 影响文件 |
|------|------|----------|
| 1 | 创建 `session_manager.py`，从 ChatService 迁移 5 个方法 | 新文件 |
| 2 | 创建 `kv_runtime_bridge.py`，从 ChatService 迁移 5 个方法 | 新文件 |
| 3 | 修改 `ChatService`：用委托替换直接实现，删除迁移走的代码 | server.py |
| 4 | 验证 `IInferenceService` 兼容性（isinstance 检查） | - |
| 5 | 运行现有测试回归 | test/ |

**每步可独立验证：** 步骤 1 和 2 互不依赖，可以并行实施。步骤 3 在 1、2 完成后进行。

---

## 8. 预期效果

| 指标 | 拆分前 | 拆分后 |
|------|--------|--------|
| ChatService 行数 | ~650 | ~400 |
| ChatService 职责数 | 5 | 3（推理执行 + 请求编排 + 文本处理） |
| 可独立测试的模块 | 1（ChatService 整体） | 3（SessionManager, KVRuntimeBridge, ChatService） |
| 新增文件 | 0 | 2（session_manager.py, kv_runtime_bridge.py） |
| 外部 API 变更 | - | 0 |

---

## 9. 测试要点

| 模块 | 测试方法 |
|------|----------|
| `SessionManager` | 单测：消息保存/读取、分叉编辑提取、取消事件 set/clear、并发安全 |
| `KVRuntimeBridge` | 单测（需 mock model）：bind/export/release 生命周期、disabled 模式跳过、debug_snapshot 格式 |
| `ChatService` | 集成测试：验证委托正确连接，现有 test_server_kv_reuse_integration.py 回归 |
| 接口兼容 | `isinstance(ChatService(...), IInferenceService)` 仍返回 True |
