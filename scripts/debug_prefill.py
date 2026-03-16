#!/usr/bin/env python3
"""Debug script: test prefill on single GPU without TP to check next_token value."""
import os, sys, ctypes, json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llaisys.libllaisys import *
import numpy as np
import safetensors
import torch
from transformers import AutoTokenizer

model_path = Path(os.path.expanduser("~/model"))
tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
input_ids = tok.encode("What is 1+1?")
print("input_ids:", input_ids)

cfg = json.load(open(model_path / "config.json"))
eos = cfg.get("eos_token_id", -1)
end_token = int(eos[0]) if isinstance(eos, list) else int(eos)
print("eos_token_id:", end_token)

weights = {}
for f in sorted(model_path.glob("*.safetensors")):
    data_ = safetensors.safe_open(f, framework="pt", device="cpu")
    for name in data_.keys():
        arr = data_.get_tensor(name)
        if arr.dtype == torch.bfloat16:
            arr = arr.to(torch.float16)
        weights[name] = arr.cpu().numpy()

meta = LlaisysQwen2Meta(
    llaisysDataType_t(DataType.F16),
    ctypes.c_size_t(cfg["num_hidden_layers"]),
    ctypes.c_size_t(cfg["hidden_size"]),
    ctypes.c_size_t(cfg["num_attention_heads"]),
    ctypes.c_size_t(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
    ctypes.c_size_t(cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])),
    ctypes.c_size_t(cfg["intermediate_size"]),
    ctypes.c_size_t(cfg["max_position_embeddings"]),
    ctypes.c_size_t(cfg["vocab_size"]),
    ctypes.c_float(cfg.get("rms_norm_eps", 1e-6)),
    ctypes.c_float(cfg.get("rope_theta", 10000.0)),
    ctypes.c_int64(end_token),
)

device_ids = (ctypes.c_int * 1)(0)
model = LIB_LLAISYS.llaisysQwen2ModelCreate(
    ctypes.byref(meta), llaisysDeviceType_t(DeviceType.NVIDIA), device_ids, 1
)
LIB_LLAISYS.llaisysQwen2ModelSetKVCacheEnabled(model, ctypes.c_int(1))

mw = LIB_LLAISYS.llaisysQwen2ModelWeights(model).contents
WEIGHT_MAP = {
    "input_layernorm.weight": "attn_norm_w",
    "self_attn.q_proj.weight": "attn_q_w",
    "self_attn.q_proj.bias": "attn_q_b",
    "self_attn.k_proj.weight": "attn_k_w",
    "self_attn.k_proj.bias": "attn_k_b",
    "self_attn.v_proj.weight": "attn_v_w",
    "self_attn.v_proj.bias": "attn_v_b",
    "self_attn.o_proj.weight": "attn_o_w",
    "post_attention_layernorm.weight": "mlp_norm_w",
    "mlp.gate_proj.weight": "mlp_gate_w",
    "mlp.up_proj.weight": "mlp_up_w",
    "mlp.down_proj.weight": "mlp_down_w",
}

for name, arr in weights.items():
    arr = np.ascontiguousarray(arr)
    shape = (ctypes.c_size_t * arr.ndim)(*arr.shape)
    dt = DataType.F16 if "float16" in arr.dtype.name else DataType.F32
    t = LIB_LLAISYS.tensorCreate(
        shape, ctypes.c_size_t(arr.ndim),
        llaisysDataType_t(dt), llaisysDeviceType_t(DeviceType.NVIDIA), ctypes.c_int(0),
    )
    LIB_LLAISYS.tensorLoad(t, ctypes.c_void_p(arr.ctypes.data))
    if name in {"model.embed_tokens.weight"}:
        mw.in_embed = t
    elif name in {"lm_head.weight"}:
        mw.out_embed = t
    elif name in {"model.norm.weight"}:
        mw.out_norm_w = t
    elif name.startswith("model.layers."):
        parts = name.split(".")
        layer = int(parts[2])
        sub = ".".join(parts[3:])
        if sub in WEIGHT_MAP:
            getattr(mw, WEIGHT_MAP[sub])[layer] = t

if not mw.out_embed and mw.in_embed:
    mw.out_embed = mw.in_embed

token_buf = (ctypes.c_int64 * len(input_ids))(*input_ids)
params = LlaisysSamplingParams(
    ctypes.c_int(1), ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_uint32(0)
)
next_token = int(LIB_LLAISYS.llaisysQwen2ModelPrefillSampling(
    model, token_buf, ctypes.c_size_t(len(input_ids)), ctypes.byref(params),
))
print("prefill next_token:", next_token)
print("decoded:", tok.decode([next_token]))
print("is eos?", next_token == end_token)
print("is negative?", next_token < 0)
