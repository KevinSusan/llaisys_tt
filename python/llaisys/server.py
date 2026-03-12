from __future__ import annotations

import argparse
import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import llaisys
from llaisys.interfaces import IInferenceService
from llaisys.kv_cache_pool import KVCachePool
from llaisys.models import Qwen2
from llaisys.scheduler import InferenceScheduler, SchedulerQueueFullError, TaskTimeoutError


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
        self._model_path = model_path
        self._enable_kv_runtime_reuse = bool(enable_kv_runtime_reuse)
        # RLock allows cooperative iterator-level scheduling in continuous-batching mode.
        self._model_lock = threading.RLock()
        self._context_lock = threading.Lock()
        self._context_messages: Dict[str, List[Dict[str, str]]] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._native_kv_contexts: Dict[str, Any] = {}
        self._native_kv_tokens: Dict[str, Tuple[int, ...]] = {}
        self._last_kv_bind_debug: Dict[str, Dict[str, Any]] = {}
        self._kv_pool = KVCachePool(
            block_size=block_size,
            max_blocks=max_blocks,
            max_bytes=max_bytes,
        )
        self._active_tokens: List[int] = []
        self._chat_template_tokenizer = self._init_chat_template_tokenizer(model_path)

        self._filter_tokens = ("<|end_of_sentence|>",)
        self._filter_patterns = [
            re.compile(r"<\s*\|\s*end_of_sentence\s*\|\s*>", re.IGNORECASE),
            re.compile(r"<\s*\|[^>]*\|\s*>"),
            re.compile(r"<\s*[\|\uFF5C][^>]*[\|\uFF5C]\s*>"),
            re.compile(
                r"<\s*[\|\uFF5C]\s*end[\s_\u2581]*of[\s_\u2581]*sentence\s*[\|\uFF5C]\s*>",
                re.IGNORECASE,
            ),
        ]

    @property
    def kv_pool(self) -> KVCachePool:
        """暴露 KVCache 池给调度器查询"""
        return self._kv_pool

    def tokenize_for_routing(self, payload: Dict[str, Any]) -> Optional[List[int]]:
        """为 KV 感知路由进行轻量级 tokenize

        尝试从 payload 构建 prompt 并 tokenize，用于调度器查询 KV 命中。
        失败时返回 None，不影响正常请求处理。

        Args:
            payload: 请求参数

        Returns:
            token ids 列表，或 None（如果无法 tokenize）
        """
        try:
            # 尝试提取 messages
            messages = payload.get("messages")
            prompt_text = payload.get("prompt")
            system_prompt = payload.get("system_prompt")

            if messages is not None:
                if not isinstance(messages, list):
                    return None
                prompt = self._render_prompt(list(messages), str(system_prompt) if system_prompt else None)
            elif prompt_text is not None:
                # 简单 prompt，尝试获取历史
                session_id = str(payload.get("session_id") or "").strip()
                with self._context_lock:
                    history = list(self._context_messages.get(session_id, []))
                history.append({"role": "user", "content": str(prompt_text)})
                prompt = self._render_prompt(history, str(system_prompt) if system_prompt else None)
            else:
                return None

            return self.tokenizer.encode(prompt)
        except Exception:
            return None

    @staticmethod
    def _init_chat_template_tokenizer(model_path: Optional[str]):
        if not model_path:
            return None
        try:
            from transformers import AutoTokenizer
        except Exception:
            return None
        try:
            return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            return None

    def _postprocess_text(self, text: str) -> str:
        for token in self._filter_tokens:
            text = text.replace(token, "")
        for pattern in self._filter_patterns:
            text = pattern.sub("", text)
        return text

    def _extract_messages(self, payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        context_id = str(payload.get("session_id") or "").strip() or str(uuid.uuid4())
        messages = payload.get("messages")
        prompt = payload.get("prompt")
        edit_from = str(payload.get("edit_from_session_id") or "").strip()
        edit_index_raw = payload.get("edit_message_index")

        # Branch session from history edit:
        # - edit_from_session_id: source session
        # - edit_message_index: replace that user message and truncate after it
        if edit_from:
            with self._context_lock:
                source = list(self._context_messages.get(edit_from, []))
            if not source:
                raise ValueError("edit_from_session_id not found")
            if prompt is None:
                raise ValueError("prompt is required when editing history")
            if edit_index_raw is None:
                raise ValueError("edit_message_index is required when editing history")
            edit_index = int(edit_index_raw)
            if edit_index < 0 or edit_index >= len(source):
                raise ValueError("edit_message_index out of range")
            if source[edit_index].get("role") != "user":
                raise ValueError("edit_message_index must point to a user message")
            branched = source[: edit_index + 1]
            branched[edit_index] = {"role": "user", "content": str(prompt)}
            # Force new branched session id if caller didn't provide one.
            if not str(payload.get("session_id") or "").strip():
                context_id = str(uuid.uuid4())
            return context_id, branched

        if messages is not None:
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            return context_id, list(messages)

        if prompt is None:
            raise ValueError("payload must include messages or prompt")

        with self._context_lock:
            history = list(self._context_messages.get(context_id, []))
        history.append({"role": "user", "content": str(prompt)})
        return context_id, history

    def _save_context_messages(self, context_id: str, messages: List[Dict[str, str]]) -> None:
        with self._context_lock:
            self._context_messages[context_id] = list(messages)

    def _get_cancel_event(self, context_id: str) -> threading.Event:
        with self._context_lock:
            event = self._cancel_events.get(context_id)
            if event is None:
                event = threading.Event()
                self._cancel_events[context_id] = event
            return event

    def request_stop(self, context_id: str) -> bool:
        with self._context_lock:
            event = self._cancel_events.get(context_id)
            if event is None:
                event = threading.Event()
                self._cancel_events[context_id] = event
            event.set()
            return True

    def _clear_stop(self, context_id: str) -> None:
        with self._context_lock:
            event = self._cancel_events.get(context_id)
            if event:
                event.clear()

    def _release_native_kv_context(self, context_id: str) -> None:
        with self._context_lock:
            ctx = self._native_kv_contexts.pop(context_id, None)
            self._native_kv_tokens.pop(context_id, None)
            self._last_kv_bind_debug.pop(context_id, None)
        if ctx:
            self.model.kv_context_release(ctx)

    def _find_native_kv_context_for_prefix(self, prompt_ids: List[int], prefix_len: int) -> Tuple[Optional[str], Any]:
        if prefix_len <= 0:
            return None, None
        prompt_prefix = tuple(prompt_ids[:prefix_len])
        with self._context_lock:
            best_sid = None
            best_ctx = None
            best_len = -1
            for sid, ctx in self._native_kv_contexts.items():
                tokens = self._native_kv_tokens.get(sid, ())
                tlen = len(tokens)
                if tlen < prefix_len:
                    continue
                if tuple(tokens[:prefix_len]) != prompt_prefix:
                    continue
                if tlen > best_len:
                    best_len = tlen
                    best_sid = sid
                    best_ctx = ctx
            return best_sid, best_ctx

    def _bind_native_kv_context_for_request(self, context_id: str, prompt_ids: List[int], prefix_len: int) -> None:
        debug = {
            "enabled": bool(self._enable_kv_runtime_reuse),
            "session_id": context_id,
            "prefix_len": int(prefix_len),
            "bound": False,
            "source_session_id": None,
            "set_kv_context_rc": None,
        }
        if not self._enable_kv_runtime_reuse or prefix_len <= 0:
            self.model.set_kv_context(None)
            with self._context_lock:
                self._last_kv_bind_debug[context_id] = debug
            return
        with self._context_lock:
            ctx = self._native_kv_contexts.get(context_id)
        source_session_id = context_id if ctx else None
        if not ctx:
            source_session_id, ctx = self._find_native_kv_context_for_prefix(prompt_ids, prefix_len)
            if not ctx:
                self.model.set_kv_context(None)
                with self._context_lock:
                    self._last_kv_bind_debug[context_id] = debug
                return
        rc = self.model.set_kv_context(ctx)
        debug["set_kv_context_rc"] = int(rc)
        debug["source_session_id"] = source_session_id
        if rc != 0:
            self.model.set_kv_context(None)
        else:
            debug["bound"] = True
        with self._context_lock:
            self._last_kv_bind_debug[context_id] = debug

    def kv_debug_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        with self._context_lock:
            if session_id:
                last_bind = dict(self._last_kv_bind_debug.get(session_id, {}))
                native_tokens = len(self._native_kv_tokens.get(session_id, ()))
                has_native_ctx = session_id in self._native_kv_contexts
            else:
                last_bind = {}
                native_tokens = 0
                has_native_ctx = False
            native_contexts = len(self._native_kv_contexts)
            tracked_token_sessions = len(self._native_kv_tokens)
        return {
            "session_id": session_id,
            "has_native_context": has_native_ctx,
            "native_tokens": native_tokens,
            "native_contexts": native_contexts,
            "tracked_token_sessions": tracked_token_sessions,
            "last_bind": last_bind,
            "kv_pool": self._kv_pool.snapshot_stats(),
        }

    def _export_native_kv_context_after_request(self, context_id: str, tokens: List[int]) -> None:
        if not self._enable_kv_runtime_reuse:
            return
        with self._context_lock:
            ctx = self._native_kv_contexts.get(context_id)
        if not ctx:
            ctx = self.model.kv_context_create()
            if not ctx:
                return
            with self._context_lock:
                self._native_kv_contexts[context_id] = ctx
        rc = self.model.export_kv_context(ctx, self._kv_pool.block_size)
        if rc == 0:
            with self._context_lock:
                self._native_kv_tokens[context_id] = tuple(int(t) for t in tokens)

    def _render_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str]) -> str:
        templated_messages: List[Dict[str, str]] = []
        if system_prompt:
            templated_messages.append({"role": "system", "content": str(system_prompt)})
        templated_messages.extend(messages)

        if self._chat_template_tokenizer is not None:
            try:
                return self._chat_template_tokenizer.apply_chat_template(
                    templated_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                pass
        return Qwen2.build_prompt(
            messages,
            system_prompt=str(system_prompt) if system_prompt else None,
            add_generation_prompt=True,
        )

    def _eos_token(self) -> int:
        eos = getattr(self.model, "_meta", None)
        if eos is None:
            return -1
        end_token = getattr(eos, "end_token", -1)
        return int(getattr(end_token, "value", end_token))

    def _decode_next(
        self,
        token_ids: List[int],
        use_sampling: bool,
        sampling: Dict[str, Any],
    ) -> int:
        top_k = int(sampling.get("top_k", 1))
        top_p = float(sampling.get("top_p", 0.0))
        temperature = float(sampling.get("temperature", 0.0))
        seed = int(sampling.get("seed", 0))
        if use_sampling:
            return int(
                self.model.step_sampling(
                    token_ids,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seed=seed,
                )
            )
        return int(self.model.step(token_ids))

    def _prefill_next(
        self,
        prompt_ids: List[int],
        use_sampling: bool,
        sampling: Dict[str, Any],
    ) -> int:
        top_k = int(sampling.get("top_k", 1))
        top_p = float(sampling.get("top_p", 0.0))
        temperature = float(sampling.get("temperature", 0.0))
        seed = int(sampling.get("seed", 0))
        if use_sampling:
            return int(
                self.model.prefill_sampling(
                    prompt_ids,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seed=seed,
                )
            )
        return int(self.model.prefill(prompt_ids))

    def _iter_generate_ids(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        sampling: Dict[str, Any],
        prefix_len: int,
        cancel_event: threading.Event,
    ) -> Iterable[int]:
        mode = str(sampling.get("mode", "")).strip().lower()
        top_k = int(sampling.get("top_k", 1))
        top_p = float(sampling.get("top_p", 0.0))
        temperature = float(sampling.get("temperature", 0.0))
        if mode == "argmax":
            use_sampling = False
        elif mode == "sample":
            use_sampling = True
        else:
            use_sampling = temperature > 0.0 or top_k > 1 or top_p > 0.0

        if cancel_event.is_set():
            return

        can_reuse_active_prefix = (
            self._enable_kv_runtime_reuse
            and prefix_len > 0
            and len(self._active_tokens) == prefix_len
            and self._active_tokens[:prefix_len] == prompt_ids[:prefix_len]
            and len(prompt_ids) > prefix_len
        )
        if can_reuse_active_prefix:
            next_token = self._decode_next(prompt_ids[prefix_len:], use_sampling, sampling)
            self._active_tokens = list(prompt_ids)
        else:
            self.model.reset_kv_cache()
            next_token = self._prefill_next(prompt_ids, use_sampling, sampling)
            self._active_tokens = list(prompt_ids)

        if next_token < 0:
            return

        eos = self._eos_token()
        yield next_token
        self._active_tokens.append(next_token)
        for _ in range(max_new_tokens - 1):
            if cancel_event.is_set():
                break
            if eos >= 0 and next_token == eos:
                break
            next_token = self._decode_next([next_token], use_sampling, sampling)
            if next_token < 0:
                break
            yield next_token
            self._active_tokens.append(next_token)

    def _prepare_request(self, payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]], List[int], Dict[str, Any], int]:
        system_prompt = payload.get("system_prompt")
        max_new_tokens = int(payload.get("max_new_tokens", 128))
        sampling = {
            "mode": payload.get("sampling"),
            "top_k": payload.get("top_k", 1),
            "top_p": payload.get("top_p", 0.0),
            "temperature": payload.get("temperature", 0.0),
            "seed": payload.get("seed", 0),
        }

        context_id, messages = self._extract_messages(payload)
        prompt = self._render_prompt(messages, str(system_prompt) if system_prompt else None)
        prompt_ids = self.tokenizer.encode(prompt)
        return context_id, messages, prompt_ids, sampling, max_new_tokens

    def generate_packed_non_stream(self, payloads: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Best-effort packed non-stream path (greedy only).

        Current safe scope:
        - non-stream requests only
        - greedy path only (no sampling)
        - no history-edit branching fields
        """
        if not payloads:
            return []
        if not hasattr(self.model, "prefill_packed") or not hasattr(self.model, "step_packed"):
            return None

        prepared: List[Tuple[str, List[Dict[str, str]], List[int], Dict[str, Any], int]] = []
        for payload in payloads:
            if payload.get("stream", False):
                return None
            # History editing introduces branch semantics; keep packed path conservative for now.
            if payload.get("edit_from_session_id"):
                return None
            try:
                context_id, messages, prompt_ids, sampling, max_new_tokens = self._prepare_request(payload)
            except Exception:
                return None
            if max_new_tokens <= 0:
                return None
            mode = str(sampling.get("mode", "")).strip().lower()
            top_k = int(sampling.get("top_k", 1))
            top_p = float(sampling.get("top_p", 0.0))
            temperature = float(sampling.get("temperature", 0.0))
            if mode == "argmax":
                use_sampling = False
            elif mode == "sample":
                use_sampling = True
            else:
                use_sampling = temperature > 0.0 or top_k > 1 or top_p > 0.0
            if use_sampling:
                return None
            prepared.append((context_id, messages, prompt_ids, sampling, max_new_tokens))

        prompts = [it[2] for it in prepared]
        generated_all: List[List[int]] = [[] for _ in prepared]
        last_step_inputs: List[int] = [int(p[-1]) if p else 0 for p in prompts]
        max_new_tokens_list = [int(it[4]) for it in prepared]
        eos = self._eos_token()
        with self._model_lock:
            self.model.reset_kv_cache()
            next_tokens = self.model.prefill_packed(prompts)
            if len(next_tokens) != len(prepared):
                return None
            for i, tok in enumerate(next_tokens):
                t = int(tok)
                if t >= 0:
                    generated_all[i].append(t)
                    last_step_inputs[i] = t
            # Continue decode rounds for unfinished requests.
            while True:
                decode_inputs: List[List[int]] = []
                active_mask: List[bool] = []
                for i in range(len(generated_all)):
                    gen = generated_all[i]
                    is_active = True
                    if len(gen) >= max_new_tokens_list[i]:
                        is_active = False
                    elif eos >= 0 and gen and gen[-1] == eos:
                        is_active = False
                    elif not gen:
                        is_active = False
                    active_mask.append(is_active)
                    decode_inputs.append([int(last_step_inputs[i])])
                if not any(active_mask):
                    break
                step_tokens = self.model.step_packed(decode_inputs)
                if len(step_tokens) != len(generated_all):
                    return None
                for i, tok in enumerate(step_tokens):
                    if not active_mask[i]:
                        continue
                    t = int(tok)
                    if t >= 0:
                        generated_all[i].append(t)
                        last_step_inputs[i] = t

        out: List[Dict[str, Any]] = []
        for i, (context_id, messages, prompt_ids, _sampling, _max_new_tokens) in enumerate(prepared):
            generated_ids = list(generated_all[i])
            response_text = self._postprocess_text(self.tokenizer.decode(generated_ids))
            messages2 = list(messages)
            messages2.append({"role": "assistant", "content": response_text})
            self._save_context_messages(context_id, messages2)
            self._clear_stop(context_id)
            out.append(
                {
                    "session_id": context_id,
                    "response": response_text,
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(generated_ids),
                        "total_tokens": len(prompt_ids) + len(generated_ids),
                    },
                }
            )
        return out

    # Backward-compatible alias used by scheduler tests/mocks.
    def generate_packed_once(self, payloads: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        return self.generate_packed_non_stream(payloads)

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context_id, messages, prompt_ids, sampling, max_new_tokens = self._prepare_request(payload)
        cancel_event = self._get_cancel_event(context_id)
        self._clear_stop(context_id)

        with self._model_lock:
            acquire = self._kv_pool.acquire_context(context_id, prompt_ids)
            self._bind_native_kv_context_for_request(context_id, prompt_ids, acquire.prefix_len)
            generated_ids: List[int] = []
            try:
                for token_id in self._iter_generate_ids(
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    sampling=sampling,
                    prefix_len=acquire.prefix_len,
                    cancel_event=cancel_event,
                ):
                    generated_ids.append(int(token_id))
                cancelled = cancel_event.is_set()
                if cancelled:
                    # Stop requests should not commit unfinished assistant output
                    # into server-side history/context.
                    self._active_tokens = list(prompt_ids)
                    self._kv_pool.update_context(context_id, prompt_ids)
                else:
                    # Update context chain with generated continuation.
                    self._kv_pool.update_context(context_id, self._active_tokens)
                    self._export_native_kv_context_after_request(context_id, self._active_tokens)
            except Exception:
                # Release broken context to avoid leaked refs on failed request.
                self._kv_pool.release_context(context_id)
                self._release_native_kv_context(context_id)
                raise

        response_text = self._postprocess_text(self.tokenizer.decode(generated_ids))
        if cancel_event.is_set():
            self._clear_stop(context_id)
            return {
                "session_id": context_id,
                "response": response_text,
                "stopped": True,
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(generated_ids),
                    "total_tokens": len(prompt_ids) + len(generated_ids),
                },
            }
        messages = list(messages)
        messages.append({"role": "assistant", "content": response_text})
        self._save_context_messages(context_id, messages)
        self._clear_stop(context_id)
        return {
            "session_id": context_id,
            "response": response_text,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            },
        }

    def stream(self, payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        context_id, messages, prompt_ids, sampling, max_new_tokens = self._prepare_request(payload)
        cancel_event = self._get_cancel_event(context_id)
        self._clear_stop(context_id)

        generated_ids: List[int] = []
        filtered = ""
        with self._model_lock:
            acquire = self._kv_pool.acquire_context(context_id, prompt_ids)
            self._bind_native_kv_context_for_request(context_id, prompt_ids, acquire.prefix_len)
            try:
                for token_id in self._iter_generate_ids(
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    sampling=sampling,
                    prefix_len=acquire.prefix_len,
                    cancel_event=cancel_event,
                ):
                    generated_ids.append(int(token_id))
                    new_text = self.tokenizer.decode(generated_ids)
                    new_filtered = self._postprocess_text(new_text)
                    delta = new_filtered[len(filtered) :]
                    filtered = new_filtered
                    if delta:
                        yield {"session_id": context_id, "delta": delta, "done": False}
                cancelled = cancel_event.is_set()
                if cancelled:
                    self._active_tokens = list(prompt_ids)
                    self._kv_pool.update_context(context_id, prompt_ids)
                else:
                    self._kv_pool.update_context(context_id, self._active_tokens)
                    self._export_native_kv_context_after_request(context_id, self._active_tokens)
            except Exception:
                self._kv_pool.release_context(context_id)
                self._release_native_kv_context(context_id)
                raise

        if cancel_event.is_set():
            self._clear_stop(context_id)
            yield {
                "session_id": context_id,
                "done": True,
                "stopped": True,
                "response": filtered,
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(generated_ids),
                    "total_tokens": len(prompt_ids) + len(generated_ids),
                },
            }
            return

        messages = list(messages)
        messages.append({"role": "assistant", "content": filtered})
        self._save_context_messages(context_id, messages)
        self._clear_stop(context_id)
        yield {
            "session_id": context_id,
            "done": True,
            "response": filtered,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            },
        }


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    scheduler: InferenceScheduler

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _write_chunk(self, data: bytes) -> bool:
        try:
            self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if parsed.path == "/debug/kv":
            query = parse_qs(parsed.query)
            session_id = str((query.get("session_id") or [""])[0]).strip() or None
            payload = self.scheduler.kv_debug_snapshot(session_id)
            self._send_json(200, payload)
            return
        if parsed.path == "/debug/scheduler":
            self._send_json(200, self.scheduler.debug_snapshot())
            return
        self._send_json(404, {"error": "not found"})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path not in ("/chat", "/v1/chat/completions", "/chat/stop"):
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid JSON"})
            return

        if self.path == "/chat/stop":
            session_id = str(payload.get("session_id") or "").strip()
            if not session_id:
                self._send_json(400, {"error": "session_id is required"})
                return
            self.scheduler.request_stop(session_id)
            self._send_json(200, {"ok": True, "session_id": session_id})
            return

        stream = bool(payload.get("stream", False))
        if not stream:
            try:
                handle = self.scheduler.submit(payload, stream=False)
                result = handle.get_result(timeout=self.scheduler.request_timeout_seconds())
                if isinstance(result, dict) and result.get("error"):
                    code = 504 if result.get("code") == "timeout" else 400
                    self._send_json(code, {"error": str(result.get("error"))})
                    return
            except SchedulerQueueFullError as exc:
                self._send_json(429, {"error": str(exc)})
                return
            except TaskTimeoutError as exc:
                self._send_json(504, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, result)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self._set_cors_headers()
        self.end_headers()

        current_session_id = ""
        try:
            handle = self.scheduler.submit(payload, stream=True)
            for item in handle.iter_stream(timeout=self.scheduler.request_timeout_seconds()):
                current_session_id = str(item.get("session_id") or current_session_id)
                data = json.dumps(item, ensure_ascii=False).encode("utf-8")
                if not self._write_chunk(b"data: " + data + b"\n\n"):
                    if current_session_id:
                        self.scheduler.request_stop(current_session_id)
                    return
        except SchedulerQueueFullError as exc:
            data = json.dumps({"error": str(exc), "code": "queue_full", "done": True}, ensure_ascii=False).encode("utf-8")
            self._write_chunk(b"data: " + data + b"\n\n")
        except TaskTimeoutError as exc:
            if current_session_id:
                self.scheduler.request_stop(current_session_id)
            data = json.dumps({"error": str(exc), "code": "timeout", "done": True}, ensure_ascii=False).encode("utf-8")
            self._write_chunk(b"data: " + data + b"\n\n")
        except Exception as exc:
            if current_session_id:
                self.scheduler.request_stop(current_session_id)
            data = json.dumps({"error": str(exc), "done": True}, ensure_ascii=False).encode("utf-8")
            self._write_chunk(b"data: " + data + b"\n\n")
        finally:
            self._write_chunk(b"")


def _resolve_tokenizer_path(model_path: str, tokenizer_path: Optional[str]) -> str:
    if tokenizer_path:
        return tokenizer_path
    path = Path(model_path)
    sp = path / "tokenizer.model"
    if sp.exists():
        return str(sp)
    hf = path / "tokenizer.json"
    if hf.exists():
        return str(hf)
    raise FileNotFoundError(f"No tokenizer.model or tokenizer.json found under: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="model directory")
    parser.add_argument("--tokenizer", required=False, type=str, help="tokenizer file path")
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--pool-size", default=1, type=int, help="deprecated")
    parser.add_argument(
        "--kv-runtime-reuse",
        action="store_true",
        help="enable experimental runtime KV reuse fast-path",
    )
    parser.add_argument("--kv-block-size", default=64, type=int, help="kv block token size")
    parser.add_argument("--kv-max-blocks", default=4096, type=int, help="kv max block count")
    parser.add_argument("--kv-max-bytes", default=268435456, type=int, help="kv max bytes")
    parser.add_argument("--workers", default=1, type=int, help="inference worker count")
    parser.add_argument("--queue-size", default=128, type=int, help="max queued tasks per worker")
    parser.add_argument("--request-timeout-ms", default=120000, type=int, help="scheduler request timeout in milliseconds")
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="enable minimal iteration-level continuous scheduling",
    )
    parser.add_argument(
        "--kv-aware-routing",
        action="store_true",
        help="enable KV-aware worker routing (query KV pool before dispatching)",
    )
    args = parser.parse_args()

    tokenizer_path = _resolve_tokenizer_path(args.model, args.tokenizer)
    worker_count = max(1, int(args.workers))
    services: List[ChatService] = []
    for _ in range(worker_count):
        tokenizer = llaisys.Tokenizer(tokenizer_path)
        model = Qwen2(
            args.model,
            llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA,
        )
        services.append(
            ChatService(
                model,
                tokenizer,
                model_path=args.model,
                enable_kv_runtime_reuse=args.kv_runtime_reuse,
                block_size=args.kv_block_size,
                max_blocks=args.kv_max_blocks,
                max_bytes=args.kv_max_bytes,
            )
        )
    scheduler = InferenceScheduler(
        services,
        queue_size=max(1, int(args.queue_size)),
        request_timeout_ms=max(0, int(args.request_timeout_ms)),
        continuous_batching=bool(args.continuous_batching),
        kv_aware_routing=bool(args.kv_aware_routing),
    )
    scheduler.start()

    handler = ChatHandler
    handler.scheduler = scheduler
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    kv_routing_str = ", kv_aware_routing=on" if args.kv_aware_routing else ""
    print(
        f"LLAISYS chat server listening on http://{args.host}:{args.port} "
        f"(workers={worker_count}, queue_size={max(1, int(args.queue_size))}{kv_routing_str})"
    )
    try:
        server.serve_forever()
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
