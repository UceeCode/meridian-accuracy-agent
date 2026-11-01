"""LLM backends. The agent loop speaks ONE neutral format; each backend translates it to a provider's API.

Neutral history items:
  {"role": "user",      "text": str}
  {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...]}
  {"role": "tool",      "results": [{"id", "name", "content"}, ...]}

Backends: AnthropicBackend (paid API) and OllamaBackend (free, local, no key). The deterministic engine and the safeguards do not
depend on which backend is used, so a weak local model can make the investigation worse but cannot make the numbers or the
safety rules worse."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

from . import config as C


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Reply:
    text: str = ""
    tool_calls: list = field(default_factory=list)


class Backend:
    label = "base"
    result_char_limit = C.TOOL_RESULT_CHAR_LIMIT

    def chat(self, system: str, history: list, tools: list) -> Reply:
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ Anthropic
class AnthropicBackend(Backend):
    def __init__(self, client, model: str, max_tokens: int = C.MAX_TOKENS):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.label = f"anthropic:{model}"

    def _convert(self, history):
        msgs = []
        for h in history:
            if h["role"] == "user":
                msgs.append({"role": "user", "content": h["text"]})
            elif h["role"] == "assistant":
                blocks = [{"type": "text", "text": h["text"]}] if h.get("text") else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in h.get("tool_calls", [])]
                msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no output)"}]})
            else:
                msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]} for r in h["results"]]})
        return msgs

    def chat(self, system, history, tools):
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system, tools=tools, messages=self._convert(history))
        texts = [b.text for b in resp.content if b.type == "text" and b.text.strip()]
        calls = [ToolCall(b.id, b.name, dict(b.input or {})) for b in resp.content if b.type == "tool_use"]
        return Reply("\n".join(texts), calls)


# ------------------------------------------------------------------------------------------------ Ollama
def _as_dict(args) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            v = json.loads(args)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def salvage_text_tool_call(text: str, tool_names: set):
    """Small local models sometimes print a tool call as JSON text instead of using the tool-call channel. Recover it if it is clean."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    if not (t.startswith("{") and t.endswith("}")):
        return None
    try:
        d = json.loads(t)
    except json.JSONDecodeError:
        return None
    name = d.get("name") or (d.get("function") or {}).get("name")
    args = d.get("arguments", d.get("parameters", d.get("input", (d.get("function") or {}).get("arguments", {}))))
    if name in tool_names:
        return ToolCall(f"call_{uuid.uuid4().hex[:8]}", name, _as_dict(args))
    return None


class OllamaBackend(Backend):
    """Talks to a local Ollama server (https://ollama.com). Uses only the standard library, so there is nothing extra to install."""

    def __init__(self, model: str, host: str = C.OLLAMA_HOST, num_ctx: int = C.OLLAMA_NUM_CTX, temperature: float = 0.0,
                 timeout: int = 600, post=None):
        self.model, self.host, self.num_ctx, self.temperature, self.timeout = model, host.rstrip("/"), num_ctx, temperature, timeout
        self._post = post or self._http_post
        self.label = f"ollama:{model}"
        self.result_char_limit = 6000      # small models have small context windows; keep tool output short

    # --- transport -----------------------------------------------------------------------------
    def _http_post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 404:
                raise RuntimeError(f"Ollama does not have the model '{self.model}'. Run:  ollama pull {self.model}") from e
            if "does not support tools" in body:
                raise RuntimeError(f"The model '{self.model}' does not support tool calling. Use one that does, e.g. qwen2.5:7b or llama3.1:8b.") from e
            raise RuntimeError(f"Ollama returned HTTP {e.code}: {body[:300]}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise RuntimeError(f"Cannot reach Ollama at {self.host} ({e}). Install it from https://ollama.com, start it (`ollama serve`), "
                               f"and run `ollama pull {self.model}`. Or use --offline.") from e

    # --- translation ---------------------------------------------------------------------------
    def _convert(self, system, history):
        msgs = [{"role": "system", "content": system}]
        for h in history:
            if h["role"] == "user":
                msgs.append({"role": "user", "content": h["text"]})
            elif h["role"] == "assistant":
                m = {"role": "assistant", "content": h.get("text") or ""}
                if h.get("tool_calls"):
                    m["tool_calls"] = [{"function": {"name": c.name, "arguments": c.input}} for c in h["tool_calls"]]
                msgs.append(m)
            else:
                msgs += [{"role": "tool", "tool_name": r["name"], "content": r["content"]} for r in h["results"]]
        return msgs

    def chat(self, system, history, tools):
        payload = {"model": self.model, "stream": False, "messages": self._convert(system, history),
                   "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools],
                   "options": {"temperature": self.temperature, "num_ctx": self.num_ctx}}
        msg = self._post("/api/chat", payload).get("message", {})
        text = (msg.get("content") or "").strip()
        calls = [ToolCall(f"call_{uuid.uuid4().hex[:8]}", c["function"]["name"], _as_dict(c["function"].get("arguments")))
                 for c in msg.get("tool_calls") or []]
        if not calls:
            salvaged = salvage_text_tool_call(text, {t["name"] for t in tools})
            if salvaged:
                calls, text = [salvaged], ""
        return Reply(text, calls)

    # --- setup helper --------------------------------------------------------------------------
    def check(self) -> str:
        try:
            req = urllib.request.Request(self.host + "/api/tags")
            with urllib.request.urlopen(req, timeout=10) as r:
                names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
        except Exception as e:
            return f"FAIL  Cannot reach Ollama at {self.host}: {e}\n      Install from https://ollama.com and start it (`ollama serve`)."
        have = any(n == self.model or n.split(":")[0] == self.model for n in names)
        if not have:
            return f"FAIL  Ollama is running but model '{self.model}' is not installed. Run: ollama pull {self.model}\n      Installed: {names or 'none'}"
        return f"OK    Ollama reachable at {self.host}; model '{self.model}' is installed."


# ------------------------------------------------------------------------------------------------ factory
def make_backend(provider: str | None = None, model: str | None = None, client=None) -> Backend:
    provider = (provider or C.PROVIDER).lower()
    if client is not None:                      # tests / advanced use: an Anthropic-style client object
        return AnthropicBackend(client, model or C.ANTHROPIC_MODEL)
    if provider == "ollama":
        return OllamaBackend(model or C.OLLAMA_MODEL)
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Put it in .env, or use the free local option: --provider ollama, or --offline.")
        import anthropic
        return AnthropicBackend(anthropic.Anthropic(api_key=key), model or C.ANTHROPIC_MODEL)
    raise RuntimeError(f"Unknown provider '{provider}'. Use 'ollama' or 'anthropic'.")
