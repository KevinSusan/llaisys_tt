# LLAISYS（中文说明）

LLAISYS 是一个从零实现 AI 推理系统的学习型项目：  
后端为 C++（编译为共享库），前端与服务层为 Python。

---

## 1. 项目结构

- `include/`：C API 头文件定义
- `src/`：C++ 实现（算子、模型、运行时）
- `python/llaisys/`：Python 封装与服务代码
- `frontend/`：聊天前端页面
- `test/`：测试脚本
- `scripts/`：工具脚本（含调度器压测脚本）

---

## 2. 基础构建

```bash
# 编译 C++ 动态库
xmake build
```

> Windows 下建议每次改完 C++ 后，同步 DLL 到 Python 包目录：

```powershell
Copy-Item -Force "build/windows/x64/release/llaisys.dll" "python/llaisys/libllaisys/llaisys.dll"
```

---

## 3. 启动聊天服务

### 单 worker（推荐起步）
```powershell
C:\Users\20307\.conda\envs\llaisys-gpu\python.exe -m llaisys.server --model "你的模型目录" --device nvidia --queue-size 128

C:\Users\20307\.conda\envs\llaisys-gpu\python.exe -m llaisys.server --model "C:\Users\20307\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B\snapshots\ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562" --device nvidia --queue-size 128

```

### 多 worker

```powershell
C:\Users\20307\.conda\envs\llaisys-gpu\python.exe -m llaisys.server --model "你的模型目录" --device nvidia --workers 2 --queue-size 128
```

推荐把开关分成两层记忆：

**A. 每天常用（先记这 3 个）**

- `--workers`：推理 worker 数（默认 1）
- `--queue-size`：每个 worker 的队列大小（默认 128）
- `--request-timeout-ms`：请求超时（默认 120000）
**B. 高级/实验（按需再开）**

- `--continuous-batching`：最小迭代连续调度（默认关闭，建议先 `--workers 1` 验证）
- `--kv-runtime-reuse`：运行时 KV 复用（实验特性，默认关闭）

如果你只想“稳定可用”，建议先用这个模板（不加实验开关）：

```powershell
C:\Users\20307\.conda\envs\llaisys-gpu\python.exe -m llaisys.server --model "你的模型目录" --device nvidia --workers 1 --queue-size 128 --request-timeout-ms 120000
```

当前阶段推荐的“稳定基线”（已验证批前向主链路）：

```powershell
C:\Users\20307\.conda\envs\llaisys-gpu\python.exe -m llaisys.server --model "你的模型目录" --device nvidia --workers 1 --queue-size 128 --request-timeout-ms 120000 --continuous-batching
```

---

## 4. 健康检查与调试

- 健康检查：`GET /health`
- KV 复用状态：`GET /debug/kv`（可带 `?session_id=...`）
- 调度器状态：`GET /debug/scheduler`

`/debug/scheduler` 关键字段说明（连续批/PD 最小版）：

- `continuous_batching`：是否开启迭代连续批
- `metrics.batch_rounds`：总调度轮次
- `metrics.prefill_rounds`：Prefill 阶段轮次
- `metrics.decode_rounds`：Decode 阶段轮次
- `metrics.batch_last_active`：最近一轮总活跃请求数
- `metrics.prefill_last_active`：最近一轮 Prefill 等待数
- `metrics.decode_last_active`：最近一轮 Decode 活跃数
- `metrics.completed/cancelled/timed_out`：完成/取消/超时累计
- `metrics.packed_prefill_batches/tasks`：packed 路径命中批次数/任务数
- `metrics.packed_prefill_attempts`：packed 路径尝试次数
- `metrics.packed_prefill_exceptions`：packed 路径异常次数
- `packed_prefill_last_error`：最近一次 packed 异常（空字符串表示当前无异常）

示例：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/debug/scheduler
curl "http://127.0.0.1:8000/debug/kv?session_id=your_session_id"
```

---

## 5. 前端功能

`frontend/` 已支持：

- 连续对话
- 停止生成（`/chat/stop`）
- 历史消息编辑并分叉会话（调用后端 `edit_from_session_id` / `edit_message_index`）

---

## 6. 调度器压测

仓库提供并发压测脚本：`scripts/benchmark_chat_scheduler.py`  
输出成功率、吞吐、延迟（avg/p50/p95/p99）和 `/debug/scheduler` 快照。

```bash
python scripts/benchmark_chat_scheduler.py --endpoint http://127.0.0.1:8000 --total-requests 30 --concurrency 10 --session-mode unique --max-new-tokens 32
```

验证会话粘性（共享会话）：

```bash
python scripts/benchmark_chat_scheduler.py --endpoint http://127.0.0.1:8000 --total-requests 20 --concurrency 5 --session-mode shared --shared-session-id bench-s1
```

---

## 7. 常见问题

### 1) 启动时报 `llaisysQwen2KVBlockCreate not found`

动态库版本不一致。请重新 `xmake build` 并覆盖复制 DLL 到：

- `python/llaisys/libllaisys/llaisys.dll`

### 2) 报 `os error 1455`（页面文件太小）

是系统内存/虚拟内存不足，不是接口参数错误。可通过：

- 增大 Windows 虚拟内存（pagefile）
- 降低 `--workers`
- 减少后台占用

---

## 8. 当前状态（简述）

- 单用户 KVCache 复用链路：可用（含前缀匹配、分叉编辑、导出恢复、调试）
- 多用户调度器：已接入内置队列 + worker 架构
- 批前向（真拼批）：
  - Prefill 批前向：已实现并接入调度器 packed 路径
  - Decode 批前向：已实现 `Decoder::decodePacked`（当前每序列每轮 1 token）
- 运行时 KV 复用：实验特性，建议灰度开启
- 当前边界：采样/更一般多 token 形态仍在持续优化中

---

## 9. 阶段建议

- 当前基础能力已搭建完成，建议先进入“稳定期”（减少架构级改动）。
- 优先做基线观察：固定参数运行 + 定期记录 `/debug/scheduler` 与压测数据。
- 后续优化可按需再开：采样/多 token 泛化、decode 内部降开销、GPU 长会话压力回归。

