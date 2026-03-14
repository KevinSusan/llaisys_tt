# LLAISYS 项目进度总览

> 更新日期：2026-03-14
> 分支：server

---

## 项目 #1：优化 CPU 推理

### 宏观

本项目的核心目标是优化 CPU 算子性能，缩小与 PyTorch 的速度差距。优化方向包括：SIMD 向量化（AVX2/AVX-512/NEON/SVE）、OpenMP 多线程并行、以及引入第三方高性能库（Eigen/OpenBLAS/MKL）加速矩阵乘法等关键算子。

当前状态：CPU 推理链路已完整可用（作业阶段完成），所有算子功能正确，但均为朴素实现，未做任何性能优化。`linear`（矩阵乘法）是 Transformer 中最耗时的算子，也是优化的首要目标。本项目尚未开始。

### 微观

| 模块 | 状态 | 说明 |
|------|------|------|
| SIMD 向量化 | ❌ 未实现 | 未引入任何 SIMD intrinsics |
| OpenMP 并行 | ❌ 未实现 | 算子均为单线程执行 |
| 第三方库加速 | ❌ 未实现 | 未集成 Eigen/OpenBLAS/MKL |
| linear 算子优化 | ❌ 未实现 | 当前为朴素三重循环，性能远低于 PyTorch |
| 性能基准报告 | ❌ 未实现 | 未输出优化前后对比数据 |
| 已有 CPU 算子（功能） | ✅ 完成 | `add/argmax/embedding/linear/rearrange/rms_norm/rope/self_attention/swiglu`，9 个算子功能正确 |
| 算子测试 | ✅ 通过 | `test/ops/` 下全部通过 |

---

## 项目 #2：多平台 CUDA 适配

### 宏观

本项目要求在 Nvidia、天数、摩尔、沐曦四个 CUDA 或类 CUDA 平台中，至少适配两个。当前仅完成了 Nvidia CUDA 平台的适配：GPU 运行时、全部 9 个算子的 CUDA kernel、设备抽象层均已实现并测试通过。

缺失的大能力：尚未适配第二个平台（天数/摩尔/沐曦），因此本项目实际只完成了一半。此外，GPU 端到端推理的系���级回归测试（长会话、多会话、packed batch）尚未完成。

### 微观

| 模块 | 状态 | 说明 |
|------|------|------|
| Nvidia GPU 运行时 | ✅ 完成 | `src/device/nvidia/nvidia_runtime_api.cu` |
| Nvidia GPU 算子 | ✅ 完成 | 9 个算子全部有 CUDA 实现，`src/ops/*/nvidia/*.cu` |
| Nvidia GPU 算子测试 | ✅ 通过 | `test/ops_gpu/` 全量通过 |
| Nvidia GPU 运行时测试 | ✅ 通过 | `test/test_runtime.py --device nvidia` |
| 设备抽象层 | ✅ 完成 | `llaisysDeviceType_t` 参数透传，CPU/GPU 自动切换 |
| xmake CUDA 构建 | ✅ 完成 | `xmake/nvidia.lua`，`--nv-gpu=y` 开关 |
| 天数平台适配 | ❌ 未实现 | — |
| 摩尔平台适配 | ❌ 未实现 | — |
| 沐曦平台适配 | ❌ 未实现 | — |
| GPU 端到端推理回归 | ⚠️ 未完成 | 需模型文件，长会话/多会话压测未做 |

---

## 项目 #3：AI 聊天机器人

### 宏观

已构建完整的单用户 AI 聊天机器人，具备实际可用的对话能力。例如：

- 用户通过 Web UI 或 HTTP API 发送消息，服务端实时流式返回 AI 回复（SSE 协议），体验类似 ChatGPT
- 支持随机采样生成更自然的回复：可配置 temperature 控制随机性、top-k/top-p 截断低概率词、seed 固定随机种子复现结果
- 支持多轮连续对话：服务端维护每个会话的消息历史，用户可以持续追问
- 支持会话分叉编辑：用户可以修改历史某一轮的提问，AI 从该点重新生成回答，前缀 KV Cache 自动复用，避免重复计算
- 实现了 KV Cache 池（`KVCachePool`）：分块存储、引用计数、sealed 前缀匹配、0 引用回收，单用户场景下已形成完整的复用闭环
- 支持中断生成：用户可随时点击停止，服务端立即中断推理，不会将半截回复污染到下一轮上下文
- 架构经过重构：ChatService 拆分为 SessionManager（会话管理）+ KVRuntimeBridge（KV 运行���桥接）+ 瘦身后的 ChatService（推理核心），职责清晰，可独立测试

### 微观

| 模块 | 文件 | 状态 |
|------|------|------|
| HTTP 服务 | `python/llaisys/server.py`（ChatHandler + main） | ✅ 完成 |
| 聊天服务 | `python/llaisys/server.py`（ChatService，~506 行） | ✅ 完成 |
| 会话管理 | `python/llaisys/session_manager.py`（98 行） | ✅ 完成 |
| KV 运行时桥接 | `python/llaisys/kv_runtime_bridge.py`（144 行） | ✅ 完成 |
| KV Cache 池 | `python/llaisys/kv_cache_pool.py`（分块、引用计数、前缀匹配） | ✅ 完成 |
| 接口定义 | `python/llaisys/interfaces.py`（IKVCachePool, IInferenceService） | ✅ 完成 |
| Python 模型封装 | `python/llaisys/models/qwen2.py` | ✅ 完成 |
| ctypes 绑定 | `python/llaisys/libllaisys/{models,ops,runtime,tensor,tokenizer}.py` | ✅ 完成 |
| Tokenizer | `python/llaisys/tokenizer.py`, `src/tokenizer/sentencepiece/` | ✅ 完成 |
| 随机采样 | C API + Python 封装（temperature/top-k/top-p/seed） | ✅ 完成 |
| 流式响应 | SSE `/chat` 端点 | ✅ 完成 |
| 分叉编辑 | `edit_from_session_id` + `edit_message_index` | ✅ 完成 |
| 中断/取消 | `/chat/stop` 端点 | ✅ 完成 |
| 调试接口 | `/debug/kv`, `/debug/scheduler`, `/health` | ✅ 完成 |
| 前端 UI | `frontend/{index.html,app.js,style.css}` | ✅ 完成 |
| KV 复用测试 | `test/test_server_kv_reuse_integration.py` | ✅ 通过 |
| KV 池测试 | `test/test_kv_cache_pool.py` | ✅ 通过 |
| 拆分测试 | `test/test_chatservice_split.py`（19 用例） | ✅ 通过 |
| 代码审查修复测试 | `test/test_fixes.py`（19 用例） | ✅ 通过 |

---

## 项目 #4：多用户推理服务

### 宏观

已实现完整的多用户推理服务，支持多用户同时进行推理并行计算。例如：

- 当多个用户同时发送请求时，请求被加入请求池（队列），由独立的 worker 循环线程异步处理，不会互相阻塞
- 已实现 PD 分离（Prefill-Decode 两阶段调度）：新请求先经过 prefill 阶段处理完整 prompt，再进入 decode 阶段逐 token 生成，两阶段独立调度
- 已实现连续批处理（continuous batching）：每轮从池中取出若干请求组成批次（batch），通过 `Decoder::decodePacked` 执行一次批量前向推理，未完成的请求放回池中继续下一轮，最大化 GPU/CPU 利用率
- 已实现 packed prefill 批量路径：多个新请求的 prompt 拼接为一个 packed 序列，通过分段注意力（`SelfAttentionSegmented`）一次前向完成，段间隔离互不干扰
- 采样请求也已支持批量路径：不同请求可以使用不同的采样参数（temperature/top-k/top-p/seed），在同一批次中独立采样，不再回退到逐条处理
- 支持会话粘性路由：同一用户的请求优先路由到同一 worker，提高 KV Cache 命中率
- 支持 KV 感知路由：调度器查询各 worker 的 KV 前缀命中情况，将请求路由到命中最长前缀的 worker，减少重复计算
- 压测验证：稳态参数下（concurrency=2, max_new_tokens=16）成功率 100%，吞吐约 0.18 rps；packed 路径开启后吞吐提升至约 0.37 rps

缺失的大能力：流式请求尚未走批量路径（仍逐条处理）、多 worker 仍为模型副本模式（N 个 worker = N 份模型权重，内存线性增长）、无公平性/优先级/老化调度策略、无 KV 内存感知流控。

### 微观

| 模块 | 文件 | 状态 |
|------|------|------|
| 调度器 | `python/llaisys/scheduler.py`（InferenceScheduler） | ✅ 完成 |
| 请求队列 | 内置 Queue，支持 `--queue-size` 配置 | ✅ 完成 |
| 多 Worker | `--workers N`，每 worker 独立模型+KV池 | ✅ 完成（副本模式） |
| 会话粘性路由 | `_session_worker` LRU OrderedDict | ✅ 完成 |
| KV 感知路由 | `--kv-aware-routing`，查询各 worker KV 前缀命中 | ✅ 完成 |
| 连续批处理 | `--continuous-batching`，迭代级调度 | ✅ 完成 |
| PD 分离 | prefill 阶段 + decode 阶段分离调度 | ✅ 完成 |
| Packed Prefill | `generate_packed_non_stream` → `prefill_packed` | ✅ 完成 |
| Packed Decode | `Decoder::decodePacked` 单轮批前向 | ✅ 完成 |
| 分段注意力 | `llaisysSelfAttentionSegmented`（C/C++/Python） | ✅ 完成 |
| 采样批量路径 | `prefill_packed_sampling` / `step_packed_sampling` | ✅ 完成 |
| 超时/流控 | `--request-timeout-ms`，队列满 429，超时 504 | ✅ 完成 |
| 调度指标 | packed_prefill_*, kv_routing_*, batch_rounds, prefill_rounds, decode_rounds 等 | ✅ 完成 |
| 压测脚本 | `scripts/benchmark_chat_scheduler.py` | ✅ 可用 |
| 调度器测试 | `test/test_scheduler_inmemory.py` | ✅ 通过 |
| 采样批量测试 | `test/test_sampling_batch.py`（19 用例） | ✅ 通过 |
| 流式批量路径 | — | ❌ 未实现 |
| 共享模型池 | 单模型 + 多推理线程 | ❌ 未实现 |
| 共享 KV 池 | 跨 worker 统一 KVCache 管理 | ❌ 未实现 |
| KV 内存感知流控 | 根据 KV 内存压力做准入控制 | ❌ 未实现 |

---

## 项目 #5：分布式推理

### 宏观

未开始。本项目要求引入张量并行，将模型分片到多个设备上实现分布式推理。Nvidia GPU 需支持 NCCL，CPU 需支持 MPI。当前无通信层实现，无法支持多机多卡推理。张量层架构预留了通信模块的位置（运行时 + 通信 + 算子），但尚未填充。

### 微观

| 模块 | 状态 | 说明 |
|------|------|------|
| 通信层（NCCL） | ❌ 未实现 | — |
| 通信层（MPI） | ❌ 未实现 | — |
| 张量并行 | ❌ 未实现 | 模型分片策略未设计 |
| 流水线并行 | ❌ 未实现 | — |
| 多机协调 | ❌ 未实现 | — |

---

## 项目 #6：支持新模型

### 宏观

未开始。当前仅支持 Qwen2（DeepSeek-R1-Distill-Qwen-1.5B）一个模型。Transformer Decoder 层有一定通用性，但缺少模型注册/发现机制，新增模型需要手动添加 C++ 实现 + C API + Python 封装全套代码。

### 微观

| 模块 | 文件 | 状态 |
|------|------|------|
| Qwen2 C++ | `src/models/qwen2/qwen2.cpp` | ✅ 完成 |
| Qwen2 C API | `src/llaisys/models/qwen2.cpp`, `include/llaisys/models/qwen2.h` | ✅ 完成 |
| Qwen2 Python | `python/llaisys/models/qwen2.py` | ✅ 完成 |
| Transformer Decoder | `src/models/transformer/decoder/` | ✅ 完成（可复用） |
| 模型注册机制 | — | ❌ 未实现 |
| 其他模型（LLaMA 等） | — | ❌ 未实现 |
| 模型配置自动解析 | — | ❌ 未实现 |

---

## 总览

| 项目 | 完成度 | 状态 |
|------|--------|------|
| #1 优化 CPU 推理 | ░░░░░░░░░░░░░░░░░░░░ 0% | ❌ 未开始（算子功能已有，性能优化未做） |
| #2 多平台 CUDA 适配 | ██████████░░░░░░░░░░ 50% | ⚠️ 仅完成 Nvidia，需再适配一个平台 |
| #3 AI 聊天机器人 | ██████████████████░░ 90% | ✅ 核心功能完成 |
| #4 多用户推理服务 | ██████████████░░░░░░ 70% | ⚠️ 缺流式批量/共享模型 |
| #5 分布式推理 | ░░░░░░░░░░░░░░░░░░░░ 0% | ❌ 未开始 |
| #6 支持新模型 | ░░░░░░░░░░░░░░░░░░░░ 0% | ❌ 未开始 |

---

## 相关文档

| 文档 | 说明 |
|------|------|
| `docs/ARCHITECTURE_ANALYSIS.md` | 架构分析与实现对比（四层设计） |
| `docs/FIX_DESIGN.md` | 6 个代码审查问题的修复设计方案 |
| `docs/CHATSERVICE_SPLIT_DESIGN.md` | ChatService 职责拆分设计方案 |
| `docs/SAMPLING_BATCH_DESIGN.md` | 采样请求批量路径设计方案 |
| `PROGRESS.md` | 开发进度详细日志 |
