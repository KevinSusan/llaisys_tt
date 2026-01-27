#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    //千问2模型元信息
    struct LlaisysQwen2Meta {
        //数据类型
        llaisysDataType_t dtype;
        //模型参数
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        //其他参数
        float epsilon, theta;
        //特殊token
        int64_t end_token;
    };

    //千问2模型权重
    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    // 采样参数
    struct LlaisysSamplingParams {
        int32_t top_k;        // <=1 表示贪心
        float top_p;          // (0,1]，<=0 表示不启用
        float temperature;   // <=0 表示禁用温度缩放
        uint32_t seed;        // 0 表示随机
    };

    //千问2模型
    struct LlaisysQwen2Model;

    //创建千问2模型实例
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    //销毁千问2模型实例
    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    //获取千问2模型权重
    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    //执行千问2模型推理（兼容接口，建议改用 Prefill/Step）
    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    //执行千问2模型预填充（prefill）
    __export int64_t llaisysQwen2ModelPrefill(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    //执行千问2模型单步解码（step）
    __export int64_t llaisysQwen2ModelStep(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    //执行千问2模型推理（带采样参数）
    __export int64_t llaisysQwen2ModelInferSampling(struct LlaisysQwen2Model * model,
                                                    int64_t * token_ids,
                                                    size_t ntoken,
                                                    const struct LlaisysSamplingParams *params);

    //执行千问2模型推理（带采样参数，按值传递）
    __export int64_t llaisysQwen2ModelInferSamplingEx(struct LlaisysQwen2Model * model,
                                                      int64_t * token_ids,
                                                      size_t ntoken,
                                                      int32_t top_k,
                                                      float top_p,
                                                      float temperature,
                                                      uint32_t seed);

    //重置千问2模型的 KV-cache
    __export void llaisysQwen2ModelResetKVCache(struct LlaisysQwen2Model * model);

    //启用/禁用 KV-cache
    __export void llaisysQwen2ModelSetKVCacheEnabled(struct LlaisysQwen2Model * model, uint8_t enabled);
}
#endif // LLAISYS_MODELS_QWEN2_H
