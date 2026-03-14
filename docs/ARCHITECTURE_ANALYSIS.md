# LLAISYS 架构分析与实现对比

> 文档日期：2026-03-12
> 对比基准：InfiniTensor 推理服务架构图

---

## 1. 目标架构概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              目标架构（四层设计）                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  服务层              调度层                              模型层                  │
│  ┌─────┐            ┌──────────────────────────┐       ┌─────────────┐         │
│  │用户 │───────────▶│ 请求池 ◀───▶ 调度器     │──────▶│   大模型    │         │
│  │终端 │    请求    │            ↕             │ 批次  │             │         │
│  │  ↻  │            │       KVCache池          │       │   ↻ ↻ ↻ ↻   │         │
│  └─────┘            │            ↻             │       └─────────────┘         │
│                     └──────────────────────────┘                                │
│                                                                                 │
│  张量层：  [ 运行时 ]  [ 通信 ]  [ 算子 ]                                        │
│                                                                                 │
│  ↻ = worker/线程/进程                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 架构设计要点

| 层级 | 职责 | 关键特性 |
|------|------|----------|
| **服务层** | 接收用户请求 | HTTP 服务、连接管理、协议解析 |
| **调度层** | 请求调度与资源管理 | 请求池、调度器、KVCache 池三者联动 |
| **模型层** | 模型推理执行 | 批次输入、多 worker 并行 |
| **张量层** | 底层计算基础设施 | 运行时、通信（NCCL/MPI）、算子 |

---

## 2. 当前实现状态

### 2.1 逐层对比

| 层级 | 组件 | 目标设计 | 当前实现 | 状态 |
|------|------|----------|----------|------|
| **服务层** | 终端 | HTTP 接收请求 | `server.py` ChatHandler | ✅ 完成 |
| | worker 循环 | 独立线程接收 | ThreadingHTTPServer | ✅ 完成 |
| **调度层** | 请求池 | 统一请求队列 | `scheduler.py` Queue | ✅ 完成 |
| | 调度器 | 组批 + 调度决策 | `InferenceScheduler` | ⚠️ 部分 |
| | KVCache 池 | **与调度器联动** | `kv_cache_pool.py` | ✅ 已联动 |
| | worker 循环 | 调度线程 | `_worker_loop` | ✅ 完成 |
| **模型层** | 批次 | 真正的 batch 输入 | packed prefill/decode | ⚠️ 部分 |
| | 大模型 | 共享模型实例 | 每 worker 独立副本 | ⚠️ 低效 |
| | 多 worker | 数据并行/流水线 | 模型副本并行 | ⚠️ 低效 |
| **张量层** | 运行时 | GPU 运行时 | `runtime.cpp` | ✅ 完成 |
| | 通信 | NCCL/MPI | 未实现 | ❌ 缺失 |
| | 算子 | CUDA kernels | `ops/` | ✅ 完成 |

### 2.2 整体完成度

```
服务层:  ████████████████████ 100%
调度层:  ████████████████░░░░  80%
模型层:  ██████████░░░░░░░░░░  50%
张量层:  ████████████████░░░░  80%
```

---

## 3. 关键差距详解

### 3.1 KVCache 池与调度器联动（已实现）

**目标设计：**
```
调度器 ◀───▶ KVCache 池
   │
   ├─ 调度时查询：哪些请求有可用 KV？
   ├─ 组批时考虑：KV 内存是否足够？
   └─ 决策依据：优先调度 KV 命中的请求
```

**当前实现（已完成 KV 感知路由）：**

调度器通过 `IInferenceService.kv_pool` 属性访问 KVCache 池，实现了 KV 感知的智能路由：

```python
# scheduler.py - _choose_worker() 实现 KV 感知路由
def _choose_worker(self, payload: Dict, tokens: Optional[List[int]]) -> int:
    if self._kv_aware_routing and tokens:
        best_worker = -1
        best_prefix_len = 0
        for idx, worker in enumerate(self._workers):
            # 通过接口查询各 worker 的 KV 前缀命中
            prefix_len = worker.service.kv_pool.query_prefix_len(tokens)
            if prefix_len > best_prefix_len:
                best_prefix_len = prefix_len
                best_worker = idx
        if best_worker >= 0:
            return best_worker
    # 降级到粘性路由
    return self._sticky_routing(payload)
```

**实现细节：**
1. `submit()` 自动调用 `tokenize_for_routing()` 获取 token 序列
2. `_choose_worker()` 遍历各 worker 的 `kv_pool.query_prefix_len()`
3. 选择命中最长前缀的 worker
4. 路由指标：`kv_aware_routing_attempts`, `kv_aware_routing_hits`, `kv_aware_routing_best_prefix_len_sum`

**启用方式：**
```bash
python -m llaisys.server --model /path/to/model --workers 4 --kv-aware-routing
```

**查看路由指标：**
```bash
curl http://localhost:8000/debug/scheduler | jq '.kv_routing_hit_rate'
```

---

### 3.2 批次组装不完整

**目标设计：**
```
请求池 ──▶ 调度器 ──▶ [req1, req2, req3] ──▶ 模型（一次 forward）
                           批次
```

**当前实现：**
```python
# 仅部分场景走 packed 路径
if len(packed_candidates) >= 2:
    # 非流式 + 贪心才走批量
    packed_results = svc.generate_packed_non_stream(packed_payloads)
else:
    # 其他情况走单条
```

**当前限制：**

| 场景 | 是否支持批量 | 说明 |
|------|-------------|------|
| 非流式 + 贪心 | ✅ | 走 packed prefill/decode |
| 流式请求 | ❌ | 单条处理 |
| 采样请求 | ❌ | 单条处理 |
| 批大小 | 固定 2-8 | 无动态调整 |

---

### 3.3 模型层多 Worker 设计

**目标设计（图中多个 ↻ 的可能含义）：**
- A. 单模型 + 多推理线程（共享 KVCache 池）
- B. 数据并行（多 GPU 各持一份模型）
- C. 流水线并行（模型切片分布在多 GPU）

**当前实现：**
```python
# server.py main()
for _ in range(worker_count):
    model = Qwen2(...)  # 每个 worker 独立加载完整模型！
    services.append(ChatService(model, ...))
```

**问题：**
- 内存浪费：N 个 worker = N 份模型权重
- KVCache 不共享：每个 worker 独立的 kv_cache_pool
- 无法利用多 GPU 并行

---

### 3.4 张量层通信缺失

**目标设计：**
```
张量层：[ 运行时 ] [ 通信 ] [ 算子 ]
                     ↑
                  NCCL/MPI
```

**当前状态：**
- ❌ 无通信层实现
- ❌ 项目 #5（分布式推理）未完成
- 无法支持多机多卡推理

---

## 4. KVCache 管理架构

### 4.1 当前两层设计

```
┌─────────────────────────────────────────────────────────────────┐
│  Python 层 (kv_cache_pool.py)                                   │
│  ─────────────────────────────────────────────────────────────  │
│  • Token 序列索引 (int64)                                       │
│  • 前缀匹配查找 (_prefix_index)                                 │
│  • 引用计数 (ref_count)                                         │
│  • 会话-块 映射关系 (_contexts)                                 │
│                                                                 │
│  特点：轻量级，设备无关                                          │
└─────────────────────────────────────────────────────────────────┘
                         ↓ 调用 C API
┌─────────────────────────────────────────────────────────────────┐
│  C++ 层 (Decoder 内部)                                          │
│  ─────────────────────────────────────────────────────────────  │
│  • 实际的 K/V 浮点张量                                          │
│  • CPU 内存 或 GPU 显存                                         │
│  • export/restore KVContext                                     │
│                                                                 │
│  特点：设备适配，透传 device 参数                                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 设备适配机制

**设计原则：通过 `llaisysDeviceType_t device` 参数实现设备抽象**

```cpp
// 模型创建时指定设备
Qwen2::Qwen2(..., llaisysDeviceType_t device, ...)

// 所有资源创建透传设备参数
llaisysQwen2KVBlockCreate(&meta, _device, device_id);
llaisysQwen2KVContextCreate(dtype, _device, device_id, ...);
tensorCreate(shape, ndim, dtype, _device, device_id);
```

**数据访问自动适配：**
```cpp
if (tensorGetDeviceType(tensor) == LLAISYS_DEVICE_CPU) {
    // CPU: 直接内存访问
    value = *reinterpret_cast<T*>(tensorGetData(tensor));
} else {
    // GPU: D2H memcpy
    runtime().api()->memcpy_sync(&value, tensorGetData(tensor), ...);
}
```

### 4.3 单用户多会话 KVCache 场景

**场景示例：会话分叉共享前缀**

```
用户编辑第2轮问题，创建分叉：

原会话 A: [系统][用户1][AI1][用户2-原][AI2]...  → tokens: [t1...t500]
分叉 B:   [系统][用户1][AI1][用户2-新]...      → tokens: [t1...t150, t501...]

物理存储（假设 block_size=64, 分叉点在 token 150）：

┌──────────────────────────────────────────────────────────────┐
│ Block 1: [t1...t64]     sealed, ref_count=2  ← A和B共享     │
│ Block 2: [t65...t128]   sealed, ref_count=2  ← A和B共享     │
│ Block 3: [t129...t192]  sealed, ref_count=1  ← 仅A使用      │
│ ...                                                          │
│ Block N: [新tokens]     sealed, ref_count=1  ← 仅B使用      │
└──────────────────────────────────────────────────────────────┘

逻辑视图（树形结构）：

       [Block 1] ─ [Block 2] ─┬─ [Block 3] ─ ... ─ [Block 7]  会话A
                              │
                              └─ [Block N] ─ [Block N+1]       会话B
```

---

## 5. 改进路线图

### 5.1 优先级排序

| 优先级 | 改进项 | 收益 | 复杂度 | 依赖 | 状态 |
|--------|--------|------|--------|------|------|
| **P0** | 调度器与 KVCache 联动 | 智能调度、减少重复计算 | 中 | 无 | ✅ 已完成 |
| **P1** | 流式请求走批量路径 | 吞吐提升 | 中 | 无 | 待实现 |
| **P1** | 单模型 + 多推理线程 | 内存节省 | 高 | 线程安全改造 | 待实现 |
| **P2** | 采样请求走批量路径 | 功能完整 | 低 | 无 | 待实现 |
| **P2** | KV 内存感知流控 | 稳定性 | 中 | P0 | 待实现 |
| **P3** | 通信层 (NCCL) | 分布式能力 | 高 | 无 | 待实现 |

### 5.2 目标架构演进

```
当前状态                              目标状态
─────────                            ─────────

┌─────────────────┐                 ┌─────────────────┐
│ Worker 1        │                 │                 │
│  ├─ Model       │                 │   共享模型池    │◀── 单份权重
│  ├─ KVPool      │    ────▶       │                 │
│  └─ Scheduler   │                 └────────┬────────┘
├─────────────────┤                          │
│ Worker 2        │                 ┌────────▼────────┐
│  ├─ Model       │                 │   共享 KVCache  │◀── 统一管理
│  ├─ KVPool      │                 │       池        │
│  └─ ...         │                 └────────┬────────┘
└─────────────────┘                          │
                                    ┌────────▼────────┐
                                    │  智能调度器     │
                                    │  ├─ 查 KV 状态  │ ✅ 已实现
                                    │  ├─ 组批决策    │
                                    │  └─ 内存感知    │
                                    └─────────────────┘
```

### 5.3 调度器内部架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     InferenceScheduler                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   submit()  │───▶│ tokenize_   │───▶│  _choose_   │         │
│  │             │    │ for_routing │    │   worker    │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                               │                 │
│        ┌──────────────────────────────────────┼───────┐        │
│        ▼                                      ▼       ▼        │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ...              │
│  │ Worker 0  │  │ Worker 1  │  │ Worker 2  │                   │
│  │ ├─ queue  │  │ ├─ queue  │  │ ├─ queue  │                   │
│  │ ├─ service│  │ ├─ service│  │ ├─ service│                   │
│  │ └─ kv_pool│  │ └─ kv_pool│  │ └─ kv_pool│                   │
│  └───────────┘  └───────────┘  └───────────┘                   │
│                                                                 │
│  KV 感知路由: 查询 kv_pool.query_prefix_len() 选择最优 worker   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 调度器指标监控

| 指标 | 说明 |
|------|------|
| `kv_aware_routing_attempts` | KV 感知路由尝试次数 |
| `kv_aware_routing_hits` | KV 前缀命中次数 |
| `kv_routing_hit_rate` | 命中率 (hits/attempts) |
| `kv_routing_avg_prefix_len` | 平均命中前缀长度 |

---

## 6. 相关文件索引

| 模块 | 文件路径 | 说明 |
|------|----------|------|
| 接口定义 | `python/llaisys/interfaces.py` | IKVCachePool, IInferenceService |
| 服务层 | `python/llaisys/server.py` | HTTP 服务、ChatHandler |
| 调度器 | `python/llaisys/scheduler.py` | InferenceScheduler |
| KV Cache 池 | `python/llaisys/kv_cache_pool.py` | Python 层索引管理 |
| 模型封装 | `python/llaisys/models/qwen2.py` | Python Qwen2 类 |
| C++ 模型 | `src/models/qwen2/qwen2.cpp` | Qwen2 实现 |
| Decoder | `src/models/transformer/decoder/` | Transformer Decoder |
| KV C API | `src/llaisys/models/qwen2.cpp` | KVBlock/KVContext API |
| 前端 | `frontend/` | Web 聊天界面 |
| 进度记录 | `PROGRESS.md` | 开发进度追踪 |

---

## 7. 附录：设备适配汇总

| 组件 | CPU | GPU | 实现方式 |
|------|-----|-----|----------|
| `kv_cache_pool.py` | ✅ | ✅ | 纯 Python，存 token ids，设备无关 |
| `KVBlock` 创建 | ✅ | ✅ | 透传 device 参数到 C++ |
| `KVContext` 创建 | ✅ | ✅ | 透传 device 参数到 C++ |
| K/V 张量存储 | CPU 内存 | GPU 显存 | tensorCreate 根据 device 分配 |
| 数据读取 | 直接访问 | D2H memcpy | 运行时自动判断 |
| 算子执行 | `cpu/*.cpp` | `nvidia/*.cu` | 编译时选择实现 |
