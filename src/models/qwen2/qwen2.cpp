#include "qwen2.hpp"

#include "llaisys/ops.h"

#include "../../utils.hpp"
#include "../../core/context/context.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace llaisys::models {
Qwen2::Qwen2(const LlaisysQwen2Meta &meta,
             const LlaisysQwen2Weights &weights,
             llaisysDeviceType_t device,
             const std::vector<int> &device_ids)
    : _meta(meta),
      _weights(&weights),
      _device(device),
      _device_ids(device_ids),
      _decoder(transformer::DecoderConfig{
                   meta.dtype,
                   meta.nlayer,
                   meta.hs,
                   meta.nh,
                   meta.nkvh,
                   meta.dh,
                   meta.di,
                   meta.maxseq,
                   meta.voc,
                   meta.epsilon,
                   meta.theta},
               &weights,
               device,
               device_ids) {}

Qwen2::~Qwen2() {
}

void Qwen2::resetKVCache() {
    _decoder.resetKVCache();
}

void Qwen2::setKVCacheEnabled(bool enabled) {
    _decoder.setKVCacheEnabled(enabled);
}

//执行千问2模型推理
static int64_t argmax_from_logits(llaisysTensor_t logits,
                                  llaisysDataType_t dtype,
                                  llaisysDeviceType_t device,
                                  int device_id) {
    int64_t next_token = -1;
    size_t one_shape[1] = {1};
    llaisysTensor_t max_idx = tensorCreate(one_shape, 1, LLAISYS_DTYPE_I64, device, device_id);
    llaisysTensor_t max_val = tensorCreate(one_shape, 1, dtype, device, device_id);
    if (!max_idx || !max_val) {
        if (max_idx) tensorDestroy(max_idx);
        if (max_val) tensorDestroy(max_val);
        return -1;
    }
    ::llaisysArgmax(max_idx, max_val, logits);
    if (tensorGetDeviceType(max_idx) == LLAISYS_DEVICE_CPU) {
        next_token = *reinterpret_cast<int64_t *>(tensorGetData(max_idx));
    } else {
        int64_t host_val = -1;
        llaisys::core::context().setDevice(device, device_id);
        llaisys::core::context().runtime().api()->memcpy_sync(
            &host_val,
            tensorGetData(max_idx),
            sizeof(int64_t),
            LLAISYS_MEMCPY_D2H);
        next_token = host_val;
    }
    tensorDestroy(max_idx);
    tensorDestroy(max_val);
    return next_token;
}

static std::vector<float> logits_to_host(llaisysTensor_t logits,
                                         llaisysDataType_t dtype,
                                         llaisysDeviceType_t device,
                                         int device_id,
                                         size_t vocab) {
    std::vector<float> host(vocab, 0.0f);
    const size_t bytes = vocab * utils::dsize(dtype);
    if (device == LLAISYS_DEVICE_CPU) {
        const std::byte *src = reinterpret_cast<const std::byte *>(tensorGetData(logits));
        if (dtype == LLAISYS_DTYPE_F32) {
            const float *vals = reinterpret_cast<const float *>(src);
            for (size_t i = 0; i < vocab; ++i) {
                host[i] = vals[i];
            }
        } else if (dtype == LLAISYS_DTYPE_F16) {
            const fp16_t *vals = reinterpret_cast<const fp16_t *>(src);
            for (size_t i = 0; i < vocab; ++i) {
                host[i] = utils::cast<float>(vals[i]);
            }
        } else if (dtype == LLAISYS_DTYPE_BF16) {
            const bf16_t *vals = reinterpret_cast<const bf16_t *>(src);
            for (size_t i = 0; i < vocab; ++i) {
                host[i] = utils::cast<float>(vals[i]);
            }
        } else {
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
        return host;
    }

    std::vector<std::byte> tmp(bytes);
    llaisys::core::context().setDevice(device, device_id);
    llaisys::core::context().runtime().api()->memcpy_sync(
        tmp.data(), tensorGetData(logits), bytes, LLAISYS_MEMCPY_D2H);

    if (dtype == LLAISYS_DTYPE_F32) {
        const float *vals = reinterpret_cast<const float *>(tmp.data());
        for (size_t i = 0; i < vocab; ++i) {
            host[i] = vals[i];
        }
    } else if (dtype == LLAISYS_DTYPE_F16) {
        const fp16_t *vals = reinterpret_cast<const fp16_t *>(tmp.data());
        for (size_t i = 0; i < vocab; ++i) {
            host[i] = utils::cast<float>(vals[i]);
        }
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        const bf16_t *vals = reinterpret_cast<const bf16_t *>(tmp.data());
        for (size_t i = 0; i < vocab; ++i) {
            host[i] = utils::cast<float>(vals[i]);
        }
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    return host;
}

static int64_t sample_from_logits(const std::vector<float> &logits,
                                  const LlaisysSamplingParams *params) {
    const size_t vocab = logits.size();
    if (vocab == 0) {
        return -1;
    }

    int top_k = params ? params->top_k : 1;
    float top_p = params ? params->top_p : 0.0f;
    float temperature = params ? params->temperature : 0.0f;
    uint32_t seed = params ? params->seed : 0u;

    if (temperature <= 0.0f && top_k <= 1 && top_p <= 0.0f) {
        return static_cast<int64_t>(std::distance(logits.begin(),
            std::max_element(logits.begin(), logits.end())));
    }

    std::vector<int> indices(vocab);
    std::iota(indices.begin(), indices.end(), 0);

    if (top_k > 0 && static_cast<size_t>(top_k) < vocab) {
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        indices.resize(top_k);
    }

    const float temp = temperature > 0.0f ? temperature : 1.0f;
    std::vector<float> filtered_logits;
    filtered_logits.reserve(indices.size());
    for (int idx : indices) {
        filtered_logits.push_back(logits[idx] / std::max(temp, 1e-6f));
    }

    float max_logit = *std::max_element(filtered_logits.begin(), filtered_logits.end());
    std::vector<float> probs(filtered_logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < filtered_logits.size(); ++i) {
        probs[i] = std::exp(filtered_logits[i] - max_logit);
        sum += probs[i];
    }
    if (sum <= 0.0f) {
        return indices.front();
    }
    for (float &p : probs) {
        p /= sum;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<size_t> order(probs.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        float cumulative = 0.0f;
        size_t keep = 0;
        for (size_t idx : order) {
            cumulative += probs[idx];
            keep++;
            if (cumulative >= top_p) {
                break;
            }
        }
        std::vector<int> new_indices;
        std::vector<float> new_probs;
        new_indices.reserve(keep);
        new_probs.reserve(keep);
        for (size_t i = 0; i < keep; ++i) {
            size_t idx = order[i];
            new_indices.push_back(indices[idx]);
            new_probs.push_back(probs[idx]);
        }
        indices.swap(new_indices);
        probs.swap(new_probs);
        float new_sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (new_sum > 0.0f) {
            for (float &p : probs) {
                p /= new_sum;
            }
        }
    }

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumulative = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return indices[i];
        }
    }
    return indices.back();
}

static int64_t next_token_from_logits(llaisysTensor_t logits,
                                      llaisysDataType_t dtype,
                                      llaisysDeviceType_t device,
                                      int device_id,
                                      size_t vocab,
                                      const LlaisysSamplingParams *params) {
    if (!params) {
        return argmax_from_logits(logits, dtype, device, device_id);
    }
    auto host_logits = logits_to_host(logits, dtype, device, device_id, vocab);
    return sample_from_logits(host_logits, params);
}

int64_t Qwen2::infer(const int64_t *token_ids, size_t ntoken) {
    return prefill(token_ids, ntoken);
}

int64_t Qwen2::prefill(const int64_t *token_ids, size_t ntoken) {
    if (!token_ids || ntoken == 0) return -1;

    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    size_t logits_shape[2] = {1, _meta.voc};
    llaisysTensor_t logits = tensorCreate(logits_shape, 2, _meta.dtype, _device, device_id);
    if (!logits) return -1;
    if (!_decoder.prefill(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    int64_t next_token = argmax_from_logits(logits, _meta.dtype, _device, device_id);
    tensorDestroy(logits);

    return next_token;
}

int64_t Qwen2::step(const int64_t *token_ids, size_t ntoken) {
    if (!token_ids || ntoken == 0) return -1;

    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    size_t logits_shape[2] = {1, _meta.voc};
    llaisysTensor_t logits = tensorCreate(logits_shape, 2, _meta.dtype, _device, device_id);
    if (!logits) return -1;
    if (!_decoder.decodeStep(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    int64_t next_token = argmax_from_logits(logits, _meta.dtype, _device, device_id);
    tensorDestroy(logits);
    return next_token;
}

int64_t Qwen2::prefillSampling(const int64_t *token_ids, size_t ntoken, const LlaisysSamplingParams *params) {
    if (!token_ids || ntoken == 0) return -1;

    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    size_t logits_shape[2] = {1, _meta.voc};
    llaisysTensor_t logits = tensorCreate(logits_shape, 2, _meta.dtype, _device, device_id);
    if (!logits) return -1;
    if (!_decoder.prefill(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    int64_t next_token = next_token_from_logits(logits, _meta.dtype, _device, device_id, _meta.voc, params);
    tensorDestroy(logits);
    return next_token;
}

int64_t Qwen2::stepSampling(const int64_t *token_ids, size_t ntoken, const LlaisysSamplingParams *params) {
    if (!token_ids || ntoken == 0) return -1;

    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    size_t logits_shape[2] = {1, _meta.voc};
    llaisysTensor_t logits = tensorCreate(logits_shape, 2, _meta.dtype, _device, device_id);
    if (!logits) return -1;
    if (!_decoder.decodeStep(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    int64_t next_token = next_token_from_logits(logits, _meta.dtype, _device, device_id, _meta.voc, params);
    tensorDestroy(logits);
    return next_token;
}
} // namespace llaisys::models
