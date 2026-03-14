## 项目进度记录

- **项目名称**：LLAISYS
- **仓库路径**：`c:\Users\20307\Desktop\github\llaisys`

---

### 2026-02-27 之前
  基线记录（历史自述，待验证）
  - 完整作业阶段全部内容，测试通过。
  - 项目阶段部分完成，部分实现可能需要重构与复测。

### 2026-02-27（复核更新）

- **作业阶段复核**
  - （√）CPU 运行时与核心算子测试通过（runtime/tensor/add/argmax/embedding/linear/rms_norm/rope/self_attention/swiglu）。
  - （？）`test_infer.py` 依赖模型目录，本次未做完整对齐复测。

- **项目 #2：在 LLAISYS 中集成 CUDA**
  - （√）GPU runtime 测试通过：`test/test_runtime.py --device nvidia`。
  - （√）GPU 算子全量测试通过：`test/ops_gpu/run_all.py --device nvidia`。
  - （？）GPU 大模型推理链路仍需用目标模型目录再做一次端到端验证（`test/test_infer.py --device nvidia --model ... --test`）。

- **项目 #3：构建 AI 聊天机器人**
  - （√）随机采样：代码层已实现（top-k/top-p/temperature/seed，含 C API 与 Python 封装）。
  - （√）聊天服务器：代码层已实现（`python/llaisys/server.py`，含 `/chat`、`/v1/chat/completions`、stream）。
  - （√）前端 UI：已实现基础页面（`frontend/index.html`、`frontend/app.js`、`frontend/style.css`）。
  - （？）会话管理：已实现基础会话与模型池逻辑，仍建议继续增强（高级会话编辑/更完整复用策略）。

- **项目 #4：多用户推理服务**
  - （？）已有线程化服务与基础池化能力，但“连续批处理/完整队列调度”尚未确认完成。

- **项目 #5：分布式推理**
  - （×）未完成（当前未确认 NCCL/MPI 分布式推理链路）。

- **项目 #6：支持新模型**
  - （×）未完成。

- **环境与验证备注**
  - GPU 测试建议固定使用：`conda run -n llaisys-gpu python ...`，避免 `.venv/base` 串环境。
  - 若报 `llaisys.dll` 缺失，需要先构建并复制 DLL 到 `python/llaisys/libllaisys/`。

### 2026-02-27（KV Cache 复用链路重构）

- **会话管理与前后端联动（项目 #3，持续增强）**
  - （√）`server.py` 会话流式中断已修正为“不提交半截回复，不污染下一轮上下文”。
  - （√）支持编辑历史分叉（`edit_from_session_id` + `edit_message_index`），新分支复用公共前缀。
  - （√）新增运行时复用开关 `--kv-runtime-reuse`（默认关闭，实验特性）。

- **Python KV 池（可验证版本）**
  - （√）新增 `python/llaisys/kv_cache_pool.py`：分块存储、动态分配、引用计数、sealed 前缀匹配、0 引用回收、异常回滚。
  - （√）新增 `test/test_kv_cache_pool.py`：覆盖前缀匹配、共享引用、回收和回滚场景。
  - （√）提供统计与诊断接口：`snapshot_stats()`、`debug_context()`。

- **C++ 底层 KV Block/Context 接口与执行接线**
  - （√）`include/llaisys/models/qwen2.h` + `src/llaisys/models/qwen2.cpp` 增加 KV block/context 生命周期与模型绑定 C API。
  - （√）`src/models/transformer/decoder/*` 已接入外部 KVContext 恢复：校验参数后将 block 链恢复到 decoder 内部连续 KV cache。
  - （√）新增导出路径：可将当前 decoder KV cache 按 block 导出到 KVContext（供后续请求恢复）。
  - （√）`python/llaisys/libllaisys/models.py` 与 `python/llaisys/models/qwen2.py` 已补齐对应 ctypes 与 Python 封装。

- **运行态验证与调试能力**
  - （√）`xmake build` 多次通过，核心改动可编译。
  - （√）新增调试接口：`GET /debug/kv`（支持 `?session_id=`），可观察 prefix 命中、绑定来源会话、绑定返回码与 KVPool 统计。
  - （？）跨会话 donor 复用已接入基础匹配策略，后续仍建议补充更严格的一致性校验和端到端压力测试。

- **当前风险/待完善**
  - （？）当前 `server.py` 仍通过全局 `self._model_lock` 串行执行推理，真实高并发多用户能力需在队列/worker 方案落地后再评估。
  - （？）`--kv-runtime-reuse` 仍属实验路径，建议先小流量验证再默认开启。
  - （？）需补充 GPU 端到端回归（含长对话、分叉编辑、多次中断）确认稳定性和收益。
  - （？）后续可增加更细粒度性能指标（prefill/decode 时间、命中率分桶、导出/恢复耗时）。

### 2026-02-27（前端分叉编辑与复用测试补充）

- **前端分叉编辑能力（项目 #3）**
  - （√）`frontend/app.js` 已支持“编辑历史用户消息 -> 分叉发送”。
  - （√）发送分叉请求时会带上 `edit_from_session_id` / `edit_message_index`，并新建本地分支会话。
  - （√）新增编辑提示条与交互细节：按钮文案切换为“分叉发送”、`Esc` 取消编辑态。
  - （√）`frontend/style.css` 已补齐对应样式（用户气泡编辑按钮、编辑提示条）。

- **KV 复用集成测试（不依赖前端）**
  - （√）新增 `test/test_server_kv_reuse_integration.py`。
  - （√）覆盖同会话复用、跨会话 donor 复用、取消请求不导出脏 KV 三个关键场景。
  - （√）支持直接执行：`python test/test_server_kv_reuse_integration.py`。

- **复用可用性结论（单用户）**
  - （√）单用户 KVCache 逻辑已形成可用闭环：前缀匹配、分叉编辑、导出/恢复、取消回滚、调试观测。
  - （？）可开始推进“多用户 1.0 服务”，但建议先做队列/worker 稳定版，再灰度开启运行时复用。

- **运维/环境提醒**
  - （√）已确认 `llaisysQwen2KVBlockCreate` 报错根因是 DLL 版本不一致（构建产物未同步到 `python/llaisys/libllaisys/llaisys.dll`）。
  - （√）建议固定流程：`xmake build` 后覆盖复制 DLL，再启动服务。

### 2026-02-28（多用户调度器压测记录）

- **调度器收口能力**
  - （√）已新增请求超时参数：`--request-timeout-ms`。
  - （√）已新增调试接口：`GET /debug/scheduler`。
  - （√）队列满返回已统一：非流式返回 429；流式返回 `done=true` + `code=queue_full`。

- **压测结果（脚本：`scripts/benchmark_chat_scheduler.py`）**
  - （√）高压参数（`total=30, concurrency=10, max_new_tokens=32, timeout=60`）：
    - 成功 4/30，失败 26/30，主要为客户端超时（`-1 timed out`）。
    - 结论：该配置超过当前机器/模型可承载区间，失败主因是超时而非接口异常。
  - （√）稳态参数（`total=20, concurrency=2, max_new_tokens=16, timeout=180`）：
    - 成功 20/20，失败 0，状态码全部 200。
    - 吞吐约 0.18 rps，延迟：`avg=11122ms, p50=11131ms, p95=15863ms, p99=16265ms`。
    - 结论：多 worker + 队列方案在当前参数下稳定可用。

- **后续压测梯度建议**
  - （？）`concurrency=4, max_new_tokens=16, timeout=180`
  - （？）`concurrency=6, max_new_tokens=16, timeout=240`
  - （？）`concurrency=4, max_new_tokens=32, timeout=240`
  - 每轮同步记录 `/debug/scheduler`（`queue_full`、`timed_out`、`queues` 峰值）。

### 2026-02-28（调度器阶段总结）

- **已完成（多用户 1.0 基线）**
  - （√）新增 `python/llaisys/scheduler.py`，实现内置队列调度器（`InferenceScheduler`）。
  - （√）`server.py` 已改造为“入口线程 + 调度器 + worker”执行模式，不再直接在 Handler 内同步跑推理。
  - （√）支持多 worker 参数：`--workers`、`--queue-size`、`--request-timeout-ms`。
  - （√）实现会话粘性路由（同 `session_id` 优先落同 worker）。
  - （√）`/chat/stop` 已接入调度器路由；`/debug/kv` 与 `/debug/scheduler` 可观测调度与复用状态。
  - （√）错误语义收口：队列满（429 / `queue_full`）、超时（504 / `timeout`）。

- **验证情况**
  - （√）新增并通过：`test/test_scheduler_inmemory.py`。
  - （√）`test/test_server_kv_reuse_integration.py` 在调度器接入后仍通过。
  - （√）提供并发压测脚本：`scripts/benchmark_chat_scheduler.py`。

- **已知限制与风险**
  - （？）当前为“请求级调度”，尚未实现“迭代级连续批处理（continuous batching）”。
  - （？）worker 数增加会按模型副本线性放大资源占用；在部分机器上可能触发 `os error 1455`（页面文件不足）。
  - （？）调度策略仍偏基础（FIFO + 粘性），公平性/优先级/老化策略尚未引入。

- **下一步建议**
  - （？）在可开关前提下实现连续批处理原型（默认关闭，灰度验证）。
  - （？）补充混合场景压测（SSE + stop + 分叉编辑并发）。
  - （？）完善任务级取消与更细粒度调度指标（等待时长分布、活跃请求数、迭代批大小）。

### 2026-02-28（最小迭代调度版：降风险落地）

- **落地策略（按风险优先级）**
  - （√）新增 `--continuous-batching` 开关，默认关闭（不改变现网默认行为）。
  - （√）先在 `workers=1` 路径实现并验证迭代级调度，再扩展多 worker。
  - （√）保持协议不变：`/chat`、SSE、`/chat/stop` 均未改协议层语义。

- **代码实现**
  - （√）`python/llaisys/scheduler.py` 新增连续批分支：同一 worker 内按“每轮推进一次”轮询活跃任务（最小实现，不改底层算子）。
  - （√）新增调度指标：`batch_rounds`、`batch_last_active`、`batch_active_sum`，并补齐 `cancelled` 计数。
  - （√）`python/llaisys/server.py` 接入 `--continuous-batching` 参数并传入调度器。
  - （√）`ChatService` 锁调整为 `RLock`，保证迭代调度下同线程可重入，避免死锁风险。

- **回归验证**
  - （√）`test/test_scheduler_inmemory.py`：通过（含连续批非流式路径新增用例）。
  - （√）`test/test_kv_cache_pool.py`：通过。
  - （√）`test/test_server_kv_reuse_integration.py`：通过。
  - （√）`scripts/benchmark_chat_scheduler.py` 小规模回归：`success=4/4`，状态码全 200。
  - （！）当前环境未安装 `pytest`，本轮使用项目内直跑测试脚本完成等价回归。

- **当前边界**
  - （？）当前连续批为“最小迭代原型”，尚未引入底层算子批处理与更复杂公平性策略。
  - （？）建议下一步固定 `workers=1` 做 A/B 压测（开关开/关同参数对比），确认收益后再放大到多 worker。

### 2026-02-28（最小 PD 分离：单进程两阶段调度）

- **实现范围（低风险）**
  - （√）在连续批模式内部引入最小 PD 分离：同一 worker 内拆分为 `Prefill` 阶段与 `Decode` 阶段。
  - （√）`Prefill` 阶段采用“每轮最多接入 1 个新请求”，降低新实现对稳定性的冲击。
  - （√）`Decode` 阶段对所有活跃请求做“一轮一步”推进，保持迭代级公平轮询。
  - （√）外部协议保持不变：`/chat`、SSE、`/chat/stop` 无改动。

- **指标补充（/debug/scheduler）**
  - （√）新增：`prefill_rounds`、`decode_rounds`、`prefill_last_active`、`decode_last_active`。
  - （√）保留并继续累计：`completed`、`cancelled`、`timed_out`、`batch_rounds`、`batch_active_sum`。

- **回归验证**
  - （√）`test/test_scheduler_inmemory.py`：通过（包含 PD 指标断言）。
  - （√）`test/test_kv_cache_pool.py`：通过。
  - （√）`test/test_server_kv_reuse_integration.py`：通过。
  - （√）`scripts/benchmark_chat_scheduler.py`：通过（小规模并发，全部 200）。

- **注意事项**
  - （！）若服务进程未重启，`/debug/scheduler` 可能仍显示旧字段；重启到最新代码后可见新增 PD 指标。

### 2026-02-28（真拼批推进：阶段性总结）

- **已完成（底层能力）**
  - （√）新增分段注意力接口 `llaisysSelfAttentionSegmented`（C API + C++ 实现 + Python 封装）。
  - （√）分段注意力已支持 packed 场景的“段间隔离 + 段内因果”，避免不同请求互相看到。
  - （√）新增对照测试 `test/ops/self_attention_segmented.py`（与 torch 参考实现比对）并通过。

- **已完成（模型接口）**
  - （√）新增 `Qwen2/Decoder` packed prefill 路径（一次前向输入 packed prompts，输出每个样本 next token）。
  - （√）新增 C API：`llaisysQwen2ModelPrefillPacked(...)`。
  - （√）新增 Python 封装：`Qwen2.prefill_packed(sequences)`。

- **已完成（调度接线，受控版本）**
  - （√）连续批调度中接入 packed prefill 快路径（受限启用）：
    - 非流式请求
    - `max_new_tokens == 1`
    - 贪心路径（无 sampling）
    - 无复杂会话编辑分支
  - （√）新增调度指标：`packed_prefill_batches`、`packed_prefill_tasks`。
  - （√）新增并通过 `test/test_scheduler_inmemory.py` 的 packed prefill 覆盖用例。

- **已完成（回归）**
  - （√）`test/test_scheduler_inmemory.py` 通过。
  - （√）`test/test_server_kv_reuse_integration.py` 通过。
  - （√）`test/test_kv_cache_pool.py` 通过。
  - （√）`scripts/benchmark_chat_scheduler.py` 在服务启动状态下可通过（本轮小规模参数成功 100%）。

- **未完成（关键缺口）**
  - （？）尚未实现“算子级 fused 真拼批”内核（当前分段路径先保证正确性，性能优化待做）。
  - （？）尚未实现完整的“prefill->decode 连续迭代真拼批”全链路（目前仅落地受控 prefill 快路径）。
  - （？）尚未把 packed prefill 快路径扩展到流式、采样、多 token 连续生成与复杂会话编辑场景。
  - （？）GPU 场景下的系统化长会话/多会话压力回归仍待补齐。

- **下一步建议**
  - （？）先实现 decode 侧批量接口与调度状态机接线，形成可持续迭代的真拼批路径。
  - （？）在不改协议前提下，逐步放开 packed prefill 适用条件（多 token、采样、更多请求类型）。
  - （？）补充 A/B 压测与收益报告（开启/关闭连续批 + packed prefill + 同参数对照）。

### 真拼批里程碑（M1 / M2 / M3）

- **M1：正确性优先（已基本完成）**
  - （√）分段注意力接口与实现：`llaisysSelfAttentionSegmented`（C/C++/Python）。
  - （√）packed prefill 基础链路：`Decoder/Qwen2/C API/Python` 已可调用。
  - （√）调度器受控快路径：非流式 + 单 token + 贪心场景可走 packed prefill。
  - （√）基础回归通过：`scheduler`、`kv_reuse`、`kv_pool`、`self_attention_segmented`。

- **M2：形成可持续迭代的真拼批（进行中）**
  - （？）decode 侧批量接口：支持多请求同轮 decode 推进。
  - （？）调度状态机接线：prefill -> decode 全链路走批量接口，不再仅 prefill 快路径。
  - （？）取消/超时/stop 语义在批量模式下保持一致。
  - （？）补充调度指标：迭代批大小分布、等待时延分布、批量命中率。

- **M3：能力放开与性能收敛（未开始）**
  - （？）扩展到流式、采样、多 token 场景。
  - （？）复杂会话能力兼容：历史编辑分叉、KV donor 复用与批量路径共存。
  - （？）GPU 系统压测：长会话/多会话/中断混合回归。
  - （？）输出 A/B 报告：关闭连续批 vs 开启连续批 vs 开启 packed prefill（同参数对照）。

- **M2 完成定义（DoD）**
  - （？）非流式主路径默认可走 prefill + decode 批量链路。
  - （？）协议兼容保持不变：`/chat`、SSE、`/chat/stop`。
  - （？）关键回归全部通过，且 `workers=1` 下稳定运行。

- **风险与控制**
  - （？）风险：过早扩展到流式/采样导致行为回归。
  - （√）控制：先锁定非流式贪心路径；每步执行现有回归 + 压测脚本。
  - （？）风险：GPU 内存压力上升。
  - （√）控制：先 `workers=1` 验证，再逐步放开并记录 `queue_full/timed_out`。

### 2026-02-28（M2 近期推进与回退记录）

- **本轮完成**
  - （√）新增 `step_packed` 接口链路（C++ / C API / Python）：
    - `llaisysQwen2ModelStepPacked(...)`
    - `Qwen2.stepPacked(...)`
    - `Qwen2.step_packed(...)`
  - （√）连续批调度已支持“非流式贪心多 token”的 packed 路径（受控范围内）。
  - （√）在运行服务上验证到 packed 命中：`packed_prefill_batches`、`packed_prefill_tasks` 随请求增长。

- **关键实验结果**
  - （√）稳定版本（回退前基线）：`total=12, concurrency=4, max_new_tokens=8`
    - 吞吐约 `0.91~0.93 rps`
    - 延迟约 `avg ~4.0s, p95 ~8.1s`
    - packed 指标有命中（示例：`packed_prefill_batches=4`, `packed_prefill_tasks=11`）。
  - （！）尝试“每样本独立 KVContext 的 Python 层增量 decode”后：
    - 吞吐降至 `~0.27~0.29 rps`
    - 延迟升至 `avg ~13~14s, p95 ~25~27s, p99 ~38~41s`
    - 结论：语义可行但实现成本过高，不适合当前主路径。

- **回退与当前策略**
  - （√）已回退高开销增量实现，恢复为：
    - packed prefill + `step_packed` 批调用过渡路径（保证当前性能区间）。
  - （√）回退后回归通过：
    - `test/test_scheduler_inmemory.py`
    - `test/test_server_kv_reuse_integration.py`
    - `test/test_kv_cache_pool.py`

- **当前判断（M2）**
  - （？）M2 约完成一半：接口与调度接线已建立，但 decode 批量高性能实现仍未完成。
  - （？）下一步应转到 C++ 侧实现低开销批量 decode（避免 Python 层 per-seq set/export 循环）。

### 2026-03-01（packed 命中失败定位与修复）

- **问题定位（可观测性补齐）**
  - （√）在 `python/llaisys/scheduler.py` 为 packed 路径新增诊断指标：
    - `packed_prefill_attempts`
    - `packed_prefill_candidate_tasks`
    - `packed_prefill_none_returns`
    - `packed_prefill_exceptions`
  - （√）新增 `packed_prefill_last_error` 并通过 `/debug/scheduler` 暴露最近一次 packed 异常。
  - （√）定位结果明确：并非“未进入 packed 路径”，而是进入后在 `step_packed` 报错回退。

- **根因与修复**
  - （√）Python 侧：`generate_packed_non_stream` 原实现每轮按活跃请求缩批，导致 `step_packed` 的序列域不稳定。
    - 已改为固定 `nseq` 的 decode 轮次输入（非活跃样本保留占位输入），仅对活跃样本采纳输出。
  - （√）C++ 侧：`Decoder::runHidden(segmented)` 仍使用 KV cache 的 `past_len`，触发 segmented offset 域不一致。
    - 已在 segmented packed 路径禁用 decoder KV cache（`can_cache=false`），避免 `q_offsets end mismatch`。
  - （√）重编译并同步 DLL 后复测确认生效。

- **验证结果（同机复测）**
  - （√）修复前（6 请求，3 并发，8 token）：
    - `packed_prefill_batches=0`、`packed_prefill_exceptions=1`
    - `packed_prefill_last_error="llaisysQwen2ModelStepPacked failed with code -3"`
  - （√）修复后（同参数）：
    - `packed_prefill_batches=2`、`packed_prefill_tasks=4`
    - `packed_prefill_exceptions=0`、`packed_prefill_none_returns=0`
    - `packed_prefill_last_error=""`
  - （√）结论：packed 路径命中已恢复并稳定，不再是“命中失败”问题。

- **当前状态与下一步**
  - （？）当前修复主要解决“命中正确性与稳定性”，吞吐/尾延迟收益仍未收敛到目标。
  - （？）下一步继续推进 M2：实现更低开销的 decode 批量路径（减少重复 prefill 与无效样本计算）。

### 2026-03-01（M2：step_packed 增量路径落地，仍待真批量内核）

- **实现内容**
  - （√）`src/models/qwen2/qwen2.cpp`：
    - `prefillPacked` 后初始化每序列 KVContext 快照（为后续 decode 续跑做准备）。
    - `stepPacked` 从“每轮全量重 `prefillPacked`”改为“C++ 内部按序列 `decodeStep` + `exportKVContext` 的增量推进”。
  - （√）接口保持不变（C API / Python / 调度器无需改协议）。

- **验证结果**
  - （√）小规模：`total=6, concurrency=3, max_new_tokens=8`
    - 吞吐约 `0.25 rps`（此前同组约 `0.19 rps`）
    - 延迟约 `avg ~11.1s`（此前同组约 `~14.7s`）
  - （√）对比组：`total=12, concurrency=4, max_new_tokens=8`
    - 成功 `12/12`，吞吐 `~0.25 rps`，`avg ~15.4s`
    - `packed_prefill_batches/tasks` 持续增长，`packed_prefill_exceptions=0`

- **当前判断**
  - （√）已摆脱“每步全量重 prefill”的回退路径，decode 进入增量续跑阶段。
  - （？）该实现仍属于“C++ 内 per-seq 增量循环”，尚不是算子级单次 batched decode 前向。
  - （？）M2 下一关键点：实现真正低开销的 decode 批量前向（减少 per-seq recover/export 开销）。

### 2026-03-01（M2 试验：单 token 增量导出，已回退）

- **试验内容**
  - （√）尝试将 `step_packed` 中每步 `exportKVContext(全量导出)` 优化为“仅追加最后 1 token 到 KVContext”。

- **结果**
  - （！）在当前机器与参数下出现性能退化与超时风险上升（含 `6/3/8` 与 `12/4/8` 组的不稳定表现）。
  - （√）已确认该路径不适合作为现阶段主线优化方向。

- **处理**
  - （√）已立即回退该试验改动，恢复到上一版稳定可用实现（C++ 增量 decode + 全量导出路径）。
  - （√）回退后服务可正常启动，packed 命中与基本功能保持正常。

### 2026-03-01（M2 关键推进：Decoder 级 decode-packed 单轮批前向）

- **实现内容**
  - （√）`src/models/transformer/decoder/decoder.hpp/.cpp` 新增 `decodePacked(...)`：
    - 每轮接收 `nseq` 个新 token（当前约束：每序列每轮 1 token）。
    - 从每序列 KVContext 聚合出 packed `k/v`，并构造独立 `q_offsets`/`kv_offsets`。
    - 单轮通过 `llaisysSelfAttentionSegmented` 完成多序列 decode 注意力计算。
    - 计算后把新 token 的每层 K/V 追加回对应 KVContext。
  - （√）`src/models/qwen2/qwen2.cpp` 的 `stepPacked` 已改为调用 `Decoder::decodePacked`，不再执行 per-seq `decodeStep + exportKVContext` 循环。

- **验证结果（同机，workers=1，continuous-batching 开）**
  - （√）`total=6, concurrency=3, max_new_tokens=8`
    - `success=6/6`
    - `throughput≈0.36 rps`
    - `avg≈7.65s, p95≈13.81s`
  - （√）`total=12, concurrency=4, max_new_tokens=8`
    - `success=12/12`
    - `throughput≈0.37 rps`
    - `avg≈10.16s, p95≈19.58s`
  - （√）packed 命中稳定：`packed_prefill_batches/tasks` 正常增长，`packed_prefill_exceptions=0`。

- **阶段判断**
  - （√）decode 侧已从“C++ 内 per-seq 循环”进入“Decoder 级单轮 packed 前向”阶段，M2 主目标有实质推进。
  - （？）后续仍可继续优化：
    - 减少 layer 内部 slice/rearrange 开销；
    - 扩展到更一般的多 token/采样路径；
    - GPU 场景做更系统的长会话压测与回归。

### 2026-03-01（M2 泛化扩展：packed 路径放宽请求类型）

- **扩展内容**
  - （√）`python/llaisys/server.py` 的 `generate_packed_non_stream` 适用范围已放宽：
    - 允许常规 `session_id` 请求进入 packed 路径；
    - 允许显式 `messages` 请求进入 packed 路径；
    - 仍保持保守约束：仅非流式、仅贪心，且暂不支持 `edit_from_session_id` 分叉编辑场景。

- **意义**
  - （√）提高真实业务请求命中 packed 路径的概率，减少“条件过严导致回退”的开销。
  - （？）后续可在一致性验证充分后，继续放开到分叉编辑与采样路径。

### 2026-03-01（阶段收口：基础能力完成，可进入稳定期）

- **阶段结论**
  - （√）当前版本已完成“可用闭环”目标：调度器、KV 复用、分叉编辑、stop、中断、debug 接口、packed prefill/decode 主链路。
  - （√）批前向能力已落地到 decode 主路径（`Decoder::decodePacked`），并完成同机压测验证。
  - （√）文档口径已对齐（`PROGRESS.md` + `README.md`）。

- **建议策略（先稳后快）**
  - （√）当前建议先冻结大改，进入“稳定运行 + 观察”阶段。
  - （√）保留后续优化方向，但暂不作为当前阻塞项（采样/多 token 泛化、进一步降开销、GPU 长压测）。

- **推荐稳定启动参数（基线）**
  - （√）`--workers 1 --queue-size 128 --request-timeout-ms 120000 --continuous-batching`
  - （√）`--kv-runtime-reuse` 继续维持灰度开关，不默认强开。

### 2026-03-12（接口抽象与 KV 感知路由）

- **架构重构：接口抽象**
  - （√）新增 `python/llaisys/interfaces.py`，定义 `IKVCachePool` 和 `IInferenceService` 接口。
  - （√）`KVCachePool` 新增 `query_prefix_len()` 方法：只读查询前缀命中长度，不修改状态。
  - （√）`ChatService` 新增 `kv_pool` 属性：暴露 KVCache 池给调度器查询。
  - （√）`InferenceScheduler` 添加类型标注，依赖接口而非具体实现。

- **功能实现：KV 感知路由**
  - （√）新增 `--kv-aware-routing` 命令行参数（默认关闭）。
  - （√）`_choose_worker()` 支持 KV 感知路由：查询各 worker 的 KV 命中情况，选择命中最多的 worker。
  - （√）路由优先级：会话粘性 > KV 感知 > hash/轮询。
  - （√）新增调度指标：`kv_aware_routing_attempts`、`kv_aware_routing_hits`、`kv_aware_routing_best_prefix_len_sum`。
  - （√）`/debug/scheduler` 新增字段：`kv_aware_routing`、`kv_routing_hit_rate`、`kv_routing_avg_prefix_len`。

- **文档更新**
  - （√）新增 `docs/ARCHITECTURE_ANALYSIS.md`：架构对比分析文档。

- **使用方式**
  ```bash
  # 启用 KV 感知路由（需要 workers > 1）
  python -m llaisys.server --model "模型目录" --workers 2 --kv-aware-routing
  ```

- **自动 Tokenize 支持**
  - （√）`ChatService` 新增 `tokenize_for_routing()` 方法：轻量级构建 prompt 并 tokenize。
  - （√）`IInferenceService` 接口新增 `tokenize_for_routing()` 可选方法。
  - （√）`InferenceScheduler.submit()` 自动调用 tokenize：当启用 KV 感知路由且 payload 无 `_prompt_tokens` 时，自动尝试 tokenize。
  - （√）失败时静默回退到普通路由，不影响正常请求处理。

- **当前限制与后续方向**
  - （√）KV 感知路由现已支持自动 tokenize，无需请求手动携带 `_prompt_tokens`。
  - （？）多 worker 仍为模型副本模式，内存占用线性增长。
  - （？）后续可考虑：共享 KVCache 池、KV 感知组批、内存感知流控。

### 2026-03-13（代码审查与质量修复）

- **代码审查（reviewer 主导）**
  - （√）完成 `interfaces.py`、`kv_cache_pool.py`、`scheduler.py`、`server.py` 详细审查。
  - （√）发现 6 个问题，按风险等级分类并输出审查报告。

- **Fix #1：`_session_worker` 无限增长（scheduler.py）**
  - （√）`_session_worker` 从 `dict` 替换为 `OrderedDict`，引入 LRU 淘汰。
  - （√）新增 `_touch_session()` 方法，统一封装写入 + 淘汰逻辑。
  - （√）新增 `max_sticky_sessions` 构造参数（默认 10000，下限 100）。
  - （√）`debug_snapshot()` 新增 `sticky_sessions` 字段。

- **Fix #2：KV 路由 TOCTOU 竞态（scheduler.py）**
  - （√）不修复，添加 best-effort 注释说明 KV 感知路由是尽力近似策略。

- **Fix #3：异常过度吞没 + payload 污染（scheduler.py）**
  - （√）`submit()` 入口统一浅拷贝 `payload = dict(payload)`，保护调用方原始 dict。
  - （√）新增 `import logging` 和 `logger`，异常时 `logger.debug(exc_info=True)` 记录。

- **Fix #4：接口未被实际继承（kv_cache_pool.py, server.py）**
  - （√）`KVCachePool` 显式继承 `IKVCachePool`，`ChatService` 显式继承 `IInferenceService`。
  - （√）`block_size` 从公有实例属性改为 `self._block_size` + `@property`，满足 ABC 约束。

- **Fix #5：`request_stop` 两次加锁（scheduler.py）**
  - （√）合并为单次 `with self._lock`，减少锁开销。

- **Fix #6：`_prompt_tokens` 泄漏到下游（scheduler.py）**
  - （√）路由决策完成后 `payload.pop("_prompt_tokens", None)`，避免内部字段传递到 worker。

- **测试（qa 主导）**
  - （√）新增 `test/test_fixes.py`：19 个测试用例，覆盖全部 6 个修复点。
  - （√）既有测试套件全部通过：`test_kv_cache_pool.py`、`test_scheduler_inmemory.py`、`test_server_kv_reuse_integration.py`。
  - （√）修复既有测试中因 Fix #4 引入运行时 `interfaces` 导入的兼容问题。

- **设计文档**
  - （√）新增 `docs/FIX_DESIGN.md`：6 个问题的完整修复设计方案。

- **团队协作流程**
  - （√）使用 5 人 agent team（lead / architect / backend / qa / reviewer）完成完整开发流程。
  - （√）流程：审查报告 → 设计方案 → 代码实现 → 测试验证 → 最终审查 → 批准合入。

### 2026-03-13（ChatService 职责拆分）

- **设计方案（architect 主导）**
  - （√）分析 ChatService 5 大职责（推理执行、会话管理、KV 复用、流式生成、批量生成）。
  - （√）确定拆出 2 个独立模块，保留 3 个紧耦合职责在 ChatService 中。
  - （√）输出设计文档 `docs/CHATSERVICE_SPLIT_DESIGN.md`。

- **新增模块：SessionManager（session_manager.py，98 行）**
  - （√）会话消息历史管理：`extract_messages()`、`save_messages()`、`get_messages()`。
  - （√）取消事件管理：`get_cancel_event()`、`request_stop()`、`clear_stop()`。
  - （√）支持分叉编辑（`edit_from_session_id` + `edit_message_index`）。
  - （√）自有 `threading.Lock()`，与 ChatService 的 `_model_lock` 独立。

- **新增模块：KVRuntimeBridge（kv_runtime_bridge.py，144 行）**
  - （√）原生 C++ KV 上下文生命周期管理：`bind_for_request()`、`export_after_request()`、`release()`。
  - （√）跨会话 donor 前缀匹配：`_find_for_prefix()`。
  - （√）调试快照：`debug_snapshot()`。
  - （√）`enabled` 属性控制整个模块是否为 no-op，开关逻辑集中。

- **ChatService 瘦身（server.py）**
  - （√）从 ~726 行瘦身到 ~506 行。
  - （√）通过 `self._session_mgr` 和 `self._kv_bridge` 委托，替换原内联实现。
  - （√）`IInferenceService` 接口签名全部不变。
  - （√）HTTP API（`/chat`、SSE、`/chat/stop`、`/debug/*`）全部不变。
  - （√）`main()` 构造参数不变。

- **测试（qa 主导）**
  - （√）新增 `test/test_chatservice_split.py`：19 个测试用例。
  - （√）覆盖 SessionManager 单测（6）、KVRuntimeBridge 单测（4）、ChatService 集成（4）、接口兼容 + 回归（5）。
  - （√）既有 4 个测试套件全部通过。

- **审查结论**
  - （√）职责边界清晰，接口完全兼容，并发安全（三把锁独立，锁顺序一致无死锁风险）。
  - （√）reviewer 批准合入。
  - （？）低优先级：`generate_packed_non_stream` 未经过 `_kv_bridge`，packed 路径暂不支持 KV 复用。

### 2026-03-14（采样请求批量路径）

- **设计方案（architect 主导）**
  - （√）分析现有 `generate_packed_non_stream` 仅支持非流式+贪心的限制。
  - （√）设计 C API 扩展方案：新增 `PrefillPackedSampling` / `StepPackedSampling`，支持 per-sequence 采样参数。
  - （√）输出设计文档 `docs/SAMPLING_BATCH_DESIGN.md`。

- **实现（backend 主导）**
  - （√）`python/llaisys/libllaisys/models.py`：新增 `LlaisysSamplingParams` ctypes 结构体，新增两个 packed sampling API 绑定，`hasattr` 保护兼容旧 DLL。
  - （√）`python/llaisys/models/qwen2.py`：新增 `prefill_packed_sampling()` 和 `step_packed_sampling()` 方法，接受 per-sequence 采样参数数组。
  - （√）`python/llaisys/server.py`：重写 `generate_packed_non_stream()`，采样请求不再回退单条处理，纯贪心批次仍走原路径。
  - （√）`scheduler.py`、`interfaces.py` 签名不变，无需修改。

- **测试（qa 主导）**
  - （√）新增 `test/test_sampling_batch.py`：19 个测试用例，全部通过。
  - （√）覆盖：纯贪心回归（2）、采样进入 packed（1）、参数组合（5）、混合批次（1）、边界条件（5）、旧 DLL 回退（3）、响应格式（2）。

- **审查结论**
  - （√）正确性、向后兼容、并发安全、接口兼容均无问题。
  - （√）reviewer 批准合入。
  - （？）低优先级建议：decode 循环中已结束序列仍传入 step（浪费算力）、缺少 seed=0 测试、ctypes 构造风格不一致。

- **团队协作流程**
  - （√）使用 4 人 agent team（architect / backend / qa / reviewer）完成完整开发流程。

---

### 使用约定

- **记录频率**：建议每次进行较大修改或完成一个作业/项目阶段后更新一次。
- **记录内容**：
  - **完成事项**：简要描述完成了什么（功能、作业、优化等）。
  - **问题与风险**：记录遇到的问题、待解决的技术难点。
  - **下一步计划**：下一次要做的 1–3 件具体事情。
- **勾选规则**：用 `（√）` 表示已完成，`（×）` 表示未完成，`（？）`表示进行中或者需要重构。

