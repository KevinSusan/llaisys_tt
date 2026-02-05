from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import llaisys
from llaisys.models import Qwen2


class _SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._sessions.get(session_id)
            if not state:
                return None
            return {
                "messages": list(state["messages"]),
                "model_idx": state["model_idx"],
                "tokens": list(state["tokens"]),
                "last_access": state["last_access"],
            }

    def set_state(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
        model_idx: int,
        tokens: List[int],
    ) -> None:
        with self._lock:
            self._sessions[session_id] = {
                "messages": list(messages),
                "model_idx": model_idx,
                "tokens": list(tokens),
                "last_access": time.time(),
            }

    def pop_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def get_lru_session_id(self) -> Optional[str]:
        with self._lock:
            if not self._sessions:
                return None
            return min(self._sessions.items(), key=lambda item: item[1]["last_access"])[0]


class ChatService:
    def __init__(self, models: List[Qwen2], tokenizer: llaisys.Tokenizer) -> None:
        self.models = models
        self.tokenizer = tokenizer
        self.sessions = _SessionStore()
        self._pool_lock = threading.Lock()
        self._model_locks = [threading.Lock() for _ in models]
        self._model_owner: Dict[int, str] = {}
        self._filter_tokens = ("</think>", "<|end_of_sentence|>")
        self._filter_patterns = [
            re.compile(r"<\s*\|\s*end_of_sentence\s*\|\s*>", re.IGNORECASE),
            re.compile(r"<\s*\|[^>]*\|\s*>"),
            re.compile(r"<\s*[\|\uFF5C][^>]*[\|\uFF5C]\s*>"),
            re.compile(r"<\s*[\|\uFF5C]\s*end[\s_\u2581]*of[\s_\u2581]*sentence\s*[\|\uFF5C]\s*>", re.IGNORECASE),
        ]

    def _postprocess_text(self, text: str) -> str:
        for token in self._filter_tokens:
            text = text.replace(token, "")
        for pattern in self._filter_patterns:
            text = pattern.sub("", text)
        return text

    def _extract_messages(self, payload: Dict[str, Any]) -> tuple[str, List[Dict[str, str]]]:
        session_id = str(payload.get("session_id") or "").strip()
        prompt = payload.get("prompt")
        messages = payload.get("messages")

        if messages is not None:
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            return session_id, messages

        if prompt is None:
            raise ValueError("payload must include messages or prompt")

        if session_id:
            state = self.sessions.get_state(session_id)
            history = state["messages"] if state else []
            history.append({"role": "user", "content": str(prompt)})
            return session_id, history

        return "", [{"role": "user", "content": str(prompt)}]

    def _eos_token(self, model: Qwen2) -> int:
        eos = getattr(model, "_meta", None)
        if eos is None:
            return -1
        end_token = getattr(eos, "end_token", -1)
        return int(getattr(end_token, "value", end_token))

    def _iter_generate_ids(
        self,
        model: Qwen2,
        tokens: List[int],
        prompt_ids: List[int],
        max_new_tokens: int,
        sampling: Dict[str, Any],
    ) -> Iterable[int]:
        top_k = int(sampling.get("top_k", 1))
        top_p = float(sampling.get("top_p", 0.0))
        temperature = float(sampling.get("temperature", 0.0))
        seed = int(sampling.get("seed", 0))
        mode = str(sampling.get("mode", "")).strip().lower()
        if mode == "argmax":
            use_sampling = False
        elif mode == "sample":
            use_sampling = True
        else:
            use_sampling = temperature > 0.0 or top_k > 1 or top_p > 0.0

        reuse_cache = bool(tokens) and prompt_ids[: len(tokens)] == tokens
        new_prompt = prompt_ids[len(tokens) :]
        if reuse_cache and new_prompt:
            if use_sampling:
                next_token = int(
                    model.step_sampling(
                        new_prompt,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        seed=seed,
                    )
                )
            else:
                next_token = int(model.step(new_prompt))
            tokens[:] = list(prompt_ids)
        else:
            model.reset_kv_cache()
            tokens[:] = list(prompt_ids)
            if use_sampling:
                next_token = int(
                    model.prefill_sampling(
                        prompt_ids,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        seed=seed,
                    )
                )
            else:
                next_token = int(model.prefill(prompt_ids))
        if next_token < 0:
            return
        eos = self._eos_token(model)
        yield next_token
        tokens.append(next_token)
        for _ in range(max_new_tokens - 1):
            if eos >= 0 and next_token == eos:
                break
            if use_sampling:
                next_token = int(
                    model.step_sampling(
                        [next_token],
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        seed=seed,
                    )
                )
            else:
                next_token = int(model.step([next_token]))
            if next_token < 0:
                break
            yield next_token
            tokens.append(next_token)

    def _assign_model(self, session_id: str) -> int:
        for idx in range(len(self.models)):
            if idx not in self._model_owner:
                self._model_owner[idx] = session_id
                return idx
        lru_session = self.sessions.get_lru_session_id()
        if lru_session is None:
            raise RuntimeError("No available model slots")
        state = self.sessions.pop_state(lru_session)
        if not state:
            raise RuntimeError("Failed to evict session")
        evicted_idx = state["model_idx"]
        self._model_owner[evicted_idx] = session_id
        return evicted_idx

    def _prepare_session(self, session_id: str, messages: List[Dict[str, str]]) -> Tuple[int, List[int]]:
        with self._pool_lock:
            state = self.sessions.get_state(session_id)
            if state is None:
                model_idx = self._assign_model(session_id)
                tokens: List[int] = []
            else:
                model_idx = state["model_idx"]
                tokens = state["tokens"]
            self.sessions.set_state(session_id, messages, model_idx, tokens)
            return model_idx, tokens

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = payload.get("system_prompt")
        max_new_tokens = int(payload.get("max_new_tokens", 128))
        sampling = {
            "mode": payload.get("sampling"),
            "top_k": payload.get("top_k", 1),
            "top_p": payload.get("top_p", 0.0),
            "temperature": payload.get("temperature", 0.0),
            "seed": payload.get("seed", 0),
        }

        session_id, messages = self._extract_messages(payload)
        if not session_id:
            session_id = str(uuid.uuid4())
        prompt = Qwen2.build_prompt(
            messages,
            system_prompt=str(system_prompt) if system_prompt else None,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt)

        generated_ids: List[int] = []
        model_idx, tokens = self._prepare_session(session_id, messages)
        model = self.models[model_idx]
        with self._model_locks[model_idx]:
            for token_id in self._iter_generate_ids(
                model, tokens, prompt_ids, max_new_tokens, sampling
            ):
                generated_ids.append(int(token_id))

        response_text = self._postprocess_text(self.tokenizer.decode(generated_ids))

        messages = list(messages)
        messages.append({"role": "assistant", "content": response_text})
        self.sessions.set_state(session_id, messages, model_idx, tokens)

        return {
            "session_id": session_id,
            "response": response_text,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            },
        }

    def stream(self, payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        system_prompt = payload.get("system_prompt")
        max_new_tokens = int(payload.get("max_new_tokens", 128))
        sampling = {
            "mode": payload.get("sampling"),
            "top_k": payload.get("top_k", 1),
            "top_p": payload.get("top_p", 0.0),
            "temperature": payload.get("temperature", 0.0),
            "seed": payload.get("seed", 0),
        }

        session_id, messages = self._extract_messages(payload)
        prompt = Qwen2.build_prompt(
            messages,
            system_prompt=str(system_prompt) if system_prompt else None,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt)

        if not session_id:
            session_id = str(uuid.uuid4())

        generated_ids: List[int] = []
        decoded = ""
        filtered = ""
        model_idx, tokens = self._prepare_session(session_id, messages)
        model = self.models[model_idx]
        with self._model_locks[model_idx]:
            for token_id in self._iter_generate_ids(
                model, tokens, prompt_ids, max_new_tokens, sampling
            ):
                generated_ids.append(int(token_id))
                new_text = self.tokenizer.decode(generated_ids)
                new_filtered = self._postprocess_text(new_text)
                delta = new_filtered[len(filtered) :]
                decoded = new_text
                filtered = new_filtered
                if delta:
                    yield {"session_id": session_id, "delta": delta, "done": False}

        messages = list(messages)
        messages.append({"role": "assistant", "content": filtered})
        self.sessions.set_state(session_id, messages, model_idx, tokens)

        yield {
            "session_id": session_id,
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
    service: ChatService

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

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path not in ("/chat", "/v1/chat/completions"):
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid JSON"})
            return

        stream = bool(payload.get("stream", False))
        if not stream:
            try:
                result = self.service.generate(payload)
            except Exception as exc:
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

        try:
            for item in self.service.stream(payload):
                data = json.dumps(item, ensure_ascii=False).encode("utf-8")
                self._write_chunk(b"data: " + data + b"\n\n")
        except Exception as exc:
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
    parser.add_argument("--pool-size", default=1, type=int, help="model instance pool size")
    args = parser.parse_args()

    tokenizer_path = _resolve_tokenizer_path(args.model, args.tokenizer)
    tokenizer = llaisys.Tokenizer(tokenizer_path)
    models = [
        Qwen2(
            args.model,
            llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA,
        )
        for _ in range(max(1, int(args.pool_size)))
    ]

    handler = ChatHandler
    handler.service = ChatService(models, tokenizer)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    print(f"LLAISYS chat server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
