"""Tests for the free local-model path (Ollama). A tiny fake Ollama HTTP server is used so the real HTTP code runs."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from meridian_agent import agent
from meridian_agent.llm import OllamaBackend, ToolCall, make_backend, salvage_text_tool_call
from meridian_agent.tools import TOOL_SCHEMAS


class FakeOllama:
    """Serves scripted /api/chat replies and a /api/tags listing on a random local port."""

    def __init__(self, script, models=("qwen2.5:7b",)):
        self.script, self.models, self.requests = list(script), list(models), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                self._send(200, {"models": [{"name": m} for m in outer.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if body["model"] not in outer.models:
                    return self._send(404, {"error": "model not found"})
                self._send(200, {"message": outer.script.pop(0)})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.host = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self): self.server.shutdown()


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_full_run_against_fake_ollama(ds, tmp_path):
    fake = FakeOllama([
        call("get_overview"),
        call("investigate_error", txn_id="TX-0169"),
        # small-model sloppiness: evidence_ids sent as a plain string, not a list
        call("raise_escalation", title="Unbriefed rounding change", severity="HIGH",
             detail="PROC-FR-07 v3 was published with staff briefing deferred and 8 variances follow it.", evidence_ids="proc:PROC-FR-07@v3"),
        call("submit_report", executive_summary="Most FR consolidation variances follow an unbriefed procedure change.",
             key_findings=["8 of 9 variances used v3 as published."]),
    ])
    try:
        backend = OllamaBackend("qwen2.5:7b", host=fake.host)
        res = agent.run_assessment(ds.dir, tmp_path, backend=backend)
    finally:
        fake.close()
    assert res["model"] == "ollama:qwen2.5:7b" and res["loop"]["stop_reason"] == "submitted"
    assert any(e["raised_by"] == "model" for e in res["escalations"])        # string evidence_ids were coerced to a list
    first = fake.requests[0]
    assert first["options"]["num_ctx"] >= 8192 and first["options"]["temperature"] == 0
    assert first["messages"][0]["role"] == "system" and first["tools"][0]["type"] == "function"
    # tool results came back in Ollama's format
    assert any(m["role"] == "tool" and m["tool_name"] == "get_overview" for m in fake.requests[1]["messages"])


def test_text_style_tool_call_is_salvaged():
    names = {t["name"] for t in TOOL_SCHEMAS}
    c = salvage_text_tool_call('{"name": "get_overview", "arguments": {}}', names)
    assert isinstance(c, ToolCall) and c.name == "get_overview"
    c = salvage_text_tool_call('```json\n{"name": "investigate_error", "parameters": {"txn_id": "TX-0206"}}\n```', names)
    assert c and c.input == {"txn_id": "TX-0206"}
    assert salvage_text_tool_call("I will look at the data now.", names) is None
    assert salvage_text_tool_call('{"name": "delete_everything", "arguments": {}}', names) is None


def test_string_arguments_are_parsed():
    fake = FakeOllama([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_overview", "arguments": "{\"a\": 1}"}}]}])
    try:
        r = OllamaBackend("qwen2.5:7b", host=fake.host).chat("sys", [{"role": "user", "text": "hi"}], TOOL_SCHEMAS)
    finally:
        fake.close()
    assert r.tool_calls[0].input == {"a": 1}


def test_missing_model_gives_actionable_message():
    fake = FakeOllama([], models=("llama3.1:8b",))
    try:
        with pytest.raises(RuntimeError, match="ollama pull qwen2.5:7b"):
            OllamaBackend("qwen2.5:7b", host=fake.host).chat("s", [{"role": "user", "text": "x"}], TOOL_SCHEMAS)
        assert OllamaBackend("qwen2.5:7b", host=fake.host).check().startswith("FAIL")
        assert OllamaBackend("llama3.1:8b", host=fake.host).check().startswith("OK")
    finally:
        fake.close()


def test_ollama_not_running_gives_actionable_message():
    
    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        OllamaBackend("qwen2.5:7b", host="http://127.0.0.1:9", timeout=2).chat("s", [{"role": "user", "text": "x"}], TOOL_SCHEMAS)


def test_weak_model_that_never_submits_still_yields_full_report(ds, tmp_path):
    fake = FakeOllama([{"role": "assistant", "content": "I looked around and I am done."}] * 2)
    try:
        res = agent.run_assessment(ds.dir, tmp_path, backend=OllamaBackend("qwen2.5:7b", host=fake.host))
    finally:
        fake.close()
    assert res["model_narrative"] is None and res["summary"]["total_flagged"] == 76
    assert "no recommendation on a salary deduction" in (tmp_path / "report.md").read_text()


def test_weak_model_cannot_break_safeguards(ds, tmp_path):
    fake = FakeOllama([
        call("record_review", txn_id="TX-0206", category="EMPLOYEE", confidence="high", evidence_ids=["log:LG-0001"], rationale="Employee should have noticed the stale rate."),
        call("submit_report", executive_summary="Recommend applying the deduction to FR.", key_findings=["x"]),
        call("submit_report", executive_summary="FR is at 24.4% as flagged.", key_findings=["See report."]),
    ])
    try:
        res = agent.run_assessment(ds.dir, tmp_path, backend=OllamaBackend("qwen2.5:7b", host=fake.host))
    finally:
        fake.close()
    assert {v["txn_id"]: v for v in res["verdicts"]}["TX-0206"]["category"] == "SYSTEM"
    assert [e["type"] for e in res["safeguard_events"]] == ["REVIEW_REJECTED", "REPORT_TEXT_REJECTED"]


def test_factory_defaults_and_errors(monkeypatch):
    assert isinstance(make_backend("ollama"), OllamaBackend)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_backend("anthropic")
    with pytest.raises(RuntimeError, match="Unknown provider"):
        make_backend("openai")
