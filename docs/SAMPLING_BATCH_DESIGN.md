# 采样请求批量路径设计方案

## 1. 现状分析

### 1.1 当前批量路径（贪心 only）

`ChatService.generate_packed_non_stream`（`server.py:271-373`）实现了非流式批量推理，但仅限贪心解码：

```python
# server.py:307-308
if use_sampling:
    return None  # 回退到单条处理
```

当任何一个请求带有 `temperature > 0`、`top_k > 1` 或 `top_p > 0` 时，整个批次回退为 `None`，调度器随后逐条执行 `generate()`。

### 1.2 调度器如何使用批量路径

`scheduler.py:540-581` 的 continuous-batching worker 在 prefill 阶段尝试收集非流式任务调用 `generate_packed_non_stream`：

1. 收集最多 8 个非流式 `_ActiveTask` 作为 `packed_candidates`
2. 调用 `svc.generate_packed_non_stream(packed_payloads)`
3. 如果返回 `None`，回退到逐条 `_step_once`

因此，只要批次中有一个采样请求，整批回退。

### 1.3 C API 层接口现状

**贪心批量接口（已有）：**
- `llaisysQwen2ModelPrefillPacked(model, token_ids, token_offsets, nseq, out_next_tokens)` — 批量 prefill，内部对 logits 做 argmax
- `llaisysQwen2ModelStepPacked(model, token_ids, token_offsets, nseq, out_next_tokens)` — 批量 decode，同上

**单条采样接口（已有）：**
- `llaisysQwen2ModelPrefillSampling(model, token_ids, ntoken, params)` — 单条 prefill + 采样
- `llaisysQwen2ModelStepSampling(model, token_ids, ntoken, params)` — 单条 step + 采样
- `LlaisysSamplingParams` 结构体：`{top_k, top_p, temperature, seed}`

**缺失：**
- 没有 `PrefillPackedSampling` / `StepPackedSampling` — 即批量 + 每序列独立采样参数的 C API。

### 1.4 Token 选择流程

**贪心路径：**
```
forward pass → logits [nseq, vocab] → per-sequence argmax → next_tokens
```

**采样路径（单条）：**
```
forward pass → logits [1, vocab] → temperature scaling → top-k filter → top-p nucleus → multinomial sample → next_token
```

关键区别：贪心是确定性的，可以对整个 `[nseq, vocab]` 矩阵做批量 argmax；采样需要对每个序列独立应用不同的 `(temperature, top_k, top_p, seed)` 参数。

## 2. 修改方案

### 2.1 总体策略：两阶段实现

**阶段 A（Python 层采样，无需改 C/DLL）：** 利用现有 `PrefillPacked`/`StepPacked` 获取 logits，在 Python 层对每个序列独立执行采样。这要求 C 层能返回 logits 而非直接返回 argmax token。

**阶段 B（C 层原生批量采样，性能最优）：** 新增 `PrefillPackedSampling`/`StepPackedSampling` C API，在 C/CUDA 层完成批量采样。

考虑到当前 `PrefillPacked`/`StepPacked` 内部直接做 argmax 并返回 token（不暴露 logits），阶段 A 需要一个新的 C API 来返回 logits。两种路径的 C 层改动量相近，因此推荐直接走阶段 B。

### 2.2 推荐方案：C 层新增批量采样 API

#### 2.2.1 新增 C API

在 `include/llaisys/models/qwen2.h` 中新增：

```c
// 批量 prefill + 每序列独立采样
__export int32_t llaisysQwen2ModelPrefillPackedSampling(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    const int64_t *token_offsets,
    size_t nseq,
    const struct LlaisysSamplingParams *params,  // 长度为 nseq 的数组
    int64_t *out_next_tokens);

// 批量 step + 每序列独立采样
__export int32_t llaisysQwen2ModelStepPackedSampling(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    const int64_t *token_offsets,
    size_t nseq,
    const struct LlaisysSamplingParams *params,  // 长度为 nseq 的数组
    int64_t *out_next_tokens);
```

与现有 `PrefillPacked`/`StepPacked` 的唯一区别：多了一个 `params` 数组参数（长度 nseq），每个元素对应一个序列的采样参数。

**实现逻辑：**
1. 复用现有 packed forward pass 得到 `logits[nseq, vocab]`
2. 对每个序列 `i`，根据 `params[i]` 决定采样策略：
   - 如果 `params[i].top_k <= 1 && params[i].temperature <= 0`：argmax（兼容贪心）
   - 否则：temperature scaling → top-k → top-p → multinomial

#### 2.2.2 Python ctypes 绑定

在 `python/llaisys/libllaisys/models.py` 的 `load_models()` 中新增：

```python
if hasattr(lib, "llaisysQwen2ModelPrefillPackedSampling"):
    lib.llaisysQwen2ModelPrefillPackedSampling.argtypes = [
        LlaisysQwen2Model,
        POINTER(c_int64),
        POINTER(c_int64),
        c_size_t,
        POINTER(LlaisysSamplingParams),  # nseq 个元素的数组
        POINTER(c_int64),
    ]
    lib.llaisysQwen2ModelPrefillPackedSampling.restype = c_int32

if hasattr(lib, "llaisysQwen2ModelStepPackedSampling"):
    lib.llaisysQwen2ModelStepPackedSampling.argtypes = [
        LlaisysQwen2Model,
        POINTER(c_int64),
        POINTER(c_int64),
        c_size_t,
        POINTER(LlaisysSamplingParams),
        POINTER(c_int64),
    ]
    lib.llaisysQwen2ModelStepPackedSampling.restype = c_int32
```

#### 2.2.3 Qwen2 模型包装

在 `python/llaisys/models/qwen2.py` 中新增两个方法：

```python
def prefill_packed_sampling(
    self,
    sequences: Sequence[Sequence[int]],
    params_list: Sequence[LlaisysSamplingParams],
) -> list[int]:
    # 构造 flat token_ids + offsets（复用 prefill_packed 的逻辑）
    # 构造 LlaisysSamplingParams 数组
    # 调用 llaisysQwen2ModelPrefillPackedSampling
    ...

def step_packed_sampling(
    self,
    sequences: Sequence[Sequence[int]],
    params_list: Sequence[LlaisysSamplingParams],
) -> list[int]:
    # 同上，调用 llaisysQwen2ModelStepPackedSampling
    ...
```

#### 2.2.4 ChatService.generate_packed_non_stream 修改

核心改动在 `server.py:271-373`：

```python
def generate_packed_non_stream(self, payloads):
    # ... 现有校验逻辑不变 ...

    # 判断是否有采样请求
    any_sampling = False
    sampling_params_list = []
    for ctx_id, msgs, prompt_ids, sampling, max_new in prepared:
        mode = str(sampling.get("mode", "")).strip().lower()
        top_k = int(sampling.get("top_k", 1))
        top_p = float(sampling.get("top_p", 0.0))
        temperature = float(sampling.get("temperature", 0.0))
        if mode == "sample" or temperature > 0.0 or top_k > 1 or top_p > 0.0:
            any_sampling = True
        sampling_params_list.append(LlaisysSamplingParams(
            top_k=top_k, top_p=top_p,
            temperature=temperature,
            seed=int(sampling.get("seed", 0)),
        ))

    if any_sampling:
        # 检查新 API 是否可用
        if not hasattr(self.model, "prefill_packed_sampling"):
            return None  # 回退
        # 使用带采样的批量路径
        next_tokens = self.model.prefill_packed_sampling(prompts, sampling_params_list)
        # decode 循环使用 step_packed_sampling
        ...
    else:
        # 保持现有贪心路径不变
        next_tokens = self.model.prefill_packed(prompts)
        ...
```

**关键设计决策：**
- 贪心请求和采样请求可以混合在同一批次中（`params[i].top_k=1, temperature=0` 等价于 argmax）
- 如果新 C API 不可用（旧 DLL），采样请求仍然回退到单条处理，保持向后兼容

#### 2.2.5 调度器无需修改

`scheduler.py` 不需要改动。它已经将非流式任务收集后调用 `generate_packed_non_stream`，该方法内部决定是否能走批量路径。

## 3. 影响文件列表

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `include/llaisys/models/qwen2.h` | 新增 | 声明 `PrefillPackedSampling` / `StepPackedSampling` |
| C/CUDA 实现文件（`src/` 下） | 新增 | 实现批量采样逻辑 |
| `python/llaisys/libllaisys/models.py` | 修改 | 新增 ctypes 绑定 |
| `python/llaisys/models/qwen2.py` | 修改 | 新增 `prefill_packed_sampling` / `step_packed_sampling` |
| `python/llaisys/server.py` | 修改 | `generate_packed_non_stream` 移除采样回退，支持混合批次 |

不需要修改的文件：
- `scheduler.py` — 调度逻辑不变
- `interfaces.py` — `generate_packed_non_stream` 签名不变
- `session_manager.py` / `kv_runtime_bridge.py` — 不涉及

## 4. 实施步骤

### Step 1: C 层实现（需要 C/CUDA 开发者）
1. 在 `qwen2.h` 中声明两个新 API
2. 在 C 实现中，复用现有 packed forward pass
3. 将 argmax 替换为 per-sequence sampling 逻辑：
   - 对 `logits[i, :]` 应用 `temperature` 缩放
   - top-k 截断
   - top-p nucleus 截断
   - softmax → multinomial 采样（使用 `seed` 初始化 RNG）
4. 编译新 DLL

### Step 2: Python 绑定
1. `libllaisys/models.py` 中添加 `hasattr` 保护的 ctypes 声明
2. `models/qwen2.py` 中添加 `prefill_packed_sampling` / `step_packed_sampling` 包装方法

### Step 3: ChatService 集成
1. 修改 `generate_packed_non_stream`：
   - 移除 `if use_sampling: return None`
   - 构建 per-request `LlaisysSamplingParams` 数组
   - 根据 API 可用性选择 packed_sampling 或 packed（贪心）路径
   - decode 循环同理使用 `step_packed_sampling`

### Step 4: 向后兼容保护
1. 所有新 API 调用都用 `hasattr` 保���
2. 旧 DLL 下采样请求仍回退到单条处理
3. 新 DLL 下贪心请求也可以走新 API（`params` 全部设为贪心参数），但为避免性能回归，保留原有贪心快速路径

## 5. 测试要点

### 5.1 单元测试
- `prefill_packed_sampling` / `step_packed_sampling` 的 Python 包装正确性
- `LlaisysSamplingParams` 数组构造和传递
- `generate_packed_non_stream` 在以下场景的行为：
  - 全部贪心请求 → 走原有路径
  - 全部采样请求 → 走新批量采样路径
  - 混合请求（贪心 + 采样）→ 走新批量采样路径
  - 新 API 不可用时 → 采样请求回退到 `None`

### 5.2 正确性验证
- 固定 seed 下，批量采样结果应与单条采样结果一致（逐 token 对比）
- 贪心参数 `(top_k=1, temperature=0)` 通过新 API 应与 argmax 结果一致
- 不同序列使用不同采样参数时，互不干扰

### 5.3 性能测试
- 对比 N 个采样请求：批量路径 vs 逐条处理的吞吐量
- 确认贪心路径无性能回归（仍走原有 `prefill_packed`）
- 批量大小 2/4/8 下的加速比

### 5.4 边界条件
- 空批次、单条批次
- 某些序列提前遇到 EOS 而其他序列继续生成
- `max_new_tokens` 不同的混合批次
- `seed=0`（随机）和固定 seed 的混合

## 6. 风险和注意事项

1. **C 层实现复杂度**：批量采样需要在 C/CUDA 层实现 per-sequence 的 temperature/top-k/top-p/multinomial，比 argmax 复杂得多。建议先在 CPU 上实现验证正确性，再优化 CUDA kernel。

2. **RNG 状态管理**：每个序列需要独立的 RNG 状态（由 seed 初始化）。`seed=0` 表示随机，需要在 C 层生成随机种子。批量中多个 `seed=0` 的序列应使用不同的随机种子。

3. **数值一致性**：批量采样和单条采样的 softmax 精度可能略有差异（浮点运算顺序不同），但在固定 seed 下应保证 token 级别一致。

4. **内存开销**：采样需要额外的临时缓冲区（sorted logits、cumulative probabilities），批量时按 `nseq * vocab` 分配。对于大词表模型需注意内存峰值。

5. **向后兼容**：通过 `hasattr` 检测确保旧 DLL 不受影响。新 DLL 的贪心路径保持不变，不引入回归风险。
