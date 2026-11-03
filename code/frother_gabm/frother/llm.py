"""LLM access: chat, JSON chat (optionally with images), embeddings.

Two real backends and one mock:
  * "ollama"  -> native Ollama HTTP API (http://localhost:11434). Used locally.
  * "openai"  -> OpenAI-compatible API (vLLM on ARC, or any /v1 server).
  * "mock"    -> deterministic fake used by the smoke test; no network.

Every call is cached in a sqlite file keyed by the full request, so re-running
a job never re-spends a call, and seeds are passed through so results are
reproducible where the server honours them.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time

import numpy as np
import requests


# ==========================================================================
# cache
# ==========================================================================

class Cache:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        with sqlite3.connect(path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str)
                              .encode()).hexdigest()

    def get(self, k: str):
        with self.lock, sqlite3.connect(self.path) as c:
            row = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v):
        with self.lock, sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v)))


# ==========================================================================
# clients
# ==========================================================================

class BaseClient:
    def __init__(self, cfg, cache: Cache):
        self.cfg, self.cache = cfg, cache
        self.n_calls = 0
        self.n_cached = 0
        self.lock = threading.Lock()

    # -- public API ---------------------------------------------------------
    def chat(self, messages, seed: int, *, temperature=None, max_tokens=None,
             json_mode=False, images=None, model=None) -> str:
        model = model or self.cfg.model
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.cfg.max_tokens
        k = Cache.key("chat", self.cfg.backend, model, messages, seed,
                      temperature, max_tokens, json_mode,
                      [hashlib.sha256(i.encode()).hexdigest() for i in (images or [])])
        hit = self.cache.get(k)
        if hit is not None:
            with self.lock:
                self.n_cached += 1
            return hit
        text = self._chat_with_retry(messages, seed, temperature, max_tokens,
                                     json_mode, images, model)
        self.cache.put(k, text)
        with self.lock:
            self.n_calls += 1
        return text

    def embed(self, texts: list[str]) -> np.ndarray:
        out = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            k = Cache.key("embed", self.cfg.backend, self.cfg.embed_model, t)
            hit = self.cache.get(k)
            if hit is not None:
                out[i] = hit
            else:
                todo.append((i, k, t))
        for start in range(0, len(todo), 32):
            batch = todo[start:start + 32]
            vecs = self._embed_with_retry([t for _, _, t in batch])
            for (i, k, _), v in zip(batch, vecs):
                out[i] = v
                self.cache.put(k, v)
        return np.asarray(out, dtype=float)

    # -- retry wrappers -------------------------------------------------------
    def _chat_with_retry(self, messages, seed, temperature, max_tokens,
                         json_mode, images, model):
        last = None
        for attempt in range(self.cfg.max_retries):
            try:
                text = self._chat(messages, seed + attempt, temperature,
                                  max_tokens * (2 if attempt else 1),
                                  json_mode, images, model)
                if text and text.strip():
                    return text
                last = RuntimeError("empty completion")
            except Exception as e:      # noqa: BLE001
                last = e
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after retries: {last}")

    def _embed_with_retry(self, texts):
        last = None
        for attempt in range(self.cfg.max_retries):
            try:
                return self._embed(texts)
            except Exception as e:      # noqa: BLE001
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"embedding call failed after retries: {last}")

    def _chat(self, *a, **k):
        raise NotImplementedError

    def _embed(self, texts):
        raise NotImplementedError


class OllamaClient(BaseClient):
    def _chat(self, messages, seed, temperature, max_tokens, json_mode, images, model):
        msgs = [dict(m) for m in messages]
        if images:
            msgs[-1]["images"] = images
        body = {"model": model, "messages": msgs, "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": temperature, "seed": int(seed) % (2**31),
                            "num_predict": int(max_tokens), "num_ctx": 8192}}
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{self.cfg.base_url}/api/chat", json=body,
                          timeout=self.cfg.timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _embed(self, texts):
        r = requests.post(f"{self.cfg.embed_base_url}/api/embed",
                          json={"model": self.cfg.embed_model, "input": texts,
                                "keep_alive": "30m"},
                          timeout=self.cfg.timeout_s)
        r.raise_for_status()
        return r.json()["embeddings"]


class OpenAICompatClient(BaseClient):
    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _chat(self, messages, seed, temperature, max_tokens, json_mode, images, model):
        msgs = [dict(m) for m in messages]
        if images:
            content = [{"type": "text", "text": msgs[-1]["content"]}]
            for b64 in images:
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}})
            msgs[-1]["content"] = content
        body = {"model": model, "messages": msgs, "temperature": temperature,
                "max_tokens": int(max_tokens), "seed": int(seed) % (2**31)}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.cfg.chat_kwargs:
            body["chat_template_kwargs"] = json.loads(self.cfg.chat_kwargs)
        r = requests.post(f"{self.cfg.base_url}/v1/chat/completions", json=body,
                          headers=self._headers(), timeout=self.cfg.timeout_s)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return msg.get("content") or ""

    def _embed(self, texts):
        r = requests.post(f"{self.cfg.embed_base_url}/v1/embeddings",
                          json={"model": self.cfg.embed_model, "input": texts},
                          headers=self._headers(), timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


class MockClient(BaseClient):
    """Behaviourally plausible fake: copies features it is shown with some
    probability, so the smoke test exercises every measurement path."""

    def __init__(self, cfg, cache, schema):
        super().__init__(cfg, cache)
        self.schema = schema

    def _chat(self, messages, seed, temperature, max_tokens, json_mode, images, model):
        rng = random.Random(seed)
        user = messages[-1]["content"]
        if json_mode:
            # extraction: pick features whose label words appear in the text
            text = user.split("CONCEPT", 1)[-1].lower()
            present = [f.fid for f in self.schema.features
                       if any(w in text for w in f.mock_words())]
            return json.dumps({"features": present})
        # generation: 3-5 random features, plus copy of any feature named in prompt
        feats = rng.sample(self.schema.features, 4)
        shown = [f for f in self.schema.features
                 if f.label.lower() in user.lower() and rng.random() < 0.6]
        feats = list({f.fid: f for f in feats + shown}.values())[:6]
        sents = " ".join(f.mock_sentence() for f in feats)
        return (f"Concept Title: Mock {rng.randint(1, 999)}\n"
                f"Concept Description: {sents}\n"
                f"Rationale: I combined ideas I found sensible.")

    def _embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
            out.append(rng.normal(size=64).tolist())
        return out


def make_client(cfg, schema=None) -> BaseClient:
    cache = Cache(f"{cfg.cache_dir}/llm_cache.sqlite")
    if cfg.backend == "ollama":
        return OllamaClient(cfg, cache)
    if cfg.backend == "openai":
        return OpenAICompatClient(cfg, cache)
    if cfg.backend == "mock":
        return MockClient(cfg, cache, schema)
    raise ValueError(f"unknown backend {cfg.backend}")


def check_endpoint(cfg) -> str:
    """One tiny real call so a misconfigured endpoint fails in seconds, not hours."""
    if cfg.backend == "mock":
        return "mock backend"
    try:
        if cfg.backend == "ollama":
            r = requests.get(f"{cfg.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            missing = [m for m in (cfg.model, cfg.embed_model) if m not in names
                       and m.split(":")[0] not in [n.split(":")[0] for n in names]]
            if missing:
                return (f"ERROR: models not found in Ollama: {missing}. "
                        f"Installed: {names}. Run `ollama pull <name>`.")
            return f"ollama OK, models installed: {names}"
        r = requests.get(f"{cfg.base_url}/v1/models", timeout=10,
                         headers={"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {})
        r.raise_for_status()
        return "openai-compatible endpoint OK: " + \
            ", ".join(m["id"] for m in r.json().get("data", []))
    except Exception as e:          # noqa: BLE001
        return f"ERROR: cannot reach {cfg.base_url}: {e}"
