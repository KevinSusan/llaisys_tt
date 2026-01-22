#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "SelfAttention: all tensors must be 3D.");

    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t dim = q->shape()[2];

    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];
    size_t kdim = k->shape()[2];
    size_t vdim = v->shape()[2];

    ASSERT(dim == kdim, "SelfAttention: q and k head dim mismatch.");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvh, "SelfAttention: v shape mismatch with k.");
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == vdim,
           "SelfAttention: output shape mismatch.");
    ASSERT(nhead % nkvh == 0, "SelfAttention: nhead must be divisible by nkvh.");

    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen,
                                   kvlen, nhead, nkvh, dim, vdim, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen,
                                   kvlen, nhead, nkvh, dim, vdim, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
