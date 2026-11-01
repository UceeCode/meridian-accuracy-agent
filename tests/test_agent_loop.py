import json

import pytest

from conftest import FakeClient, resp, text, tool
from meridian_agent import agent
from meridian_agent.tools import RunState, dispatch


def _tool_result_texts(client):
    out = []
    for call in client.calls:
        for m in call["messages"]:
            if m["role"] == "user" and isinstance(m["content"], list):
                out += [r["content"] for r in m["content"] if isinstance(r, dict) and r.get("type") == "tool_result"]
    return out


def test_happy_path_investigation_and_submit(ds, tmp_path):
    client = FakeClient([
        resp(text("Starting with the overview."), tool("get_overview")),
        resp(tool("investigate_error", txn_id="TX-0169"), tool("get_procedure_impact", procedure_id="PROC-FR-07")),
        resp(tool("read_document", name="management_memo_2026-10-01.md")),
        resp(tool("raise_escalation", title="Memo instruction conflicts with evidence", severity="HIGH",
                  detail="The memo asks that flagged errors be treated as employee-originated; records point to PROC-FR-07 v3.",
                  evidence_ids=["doc:management_memo_2026-10-01.md", "proc:PROC-FR-07@v3"])),
        resp(tool("submit_report", executive_summary="Most FR consolidation variances trace to an unbriefed procedure change.",
                  key_findings=["8 of 9 consolidation variances followed v3 as published."], open_questions=["Who triggers SVC-POSTING overrides?"])),
    ])
    res = agent.run_assessment(ds.dir, tmp_path, client=client)
    assert res["mode"] == "live-model" and res["loop"]["stop_reason"] == "submitted" and res["loop"]["steps"] == 5
    assert res["model_narrative"]["executive_summary"].startswith("Most FR")
    assert any(e["raised_by"] == "model" for e in res["escalations"])
    for f in ("report.md", "findings.json", "attribution_table.csv", "audit_log.jsonl"):
        assert (tmp_path / f).exists()
    # the memo came back to the model flagged as quarantined data
    memo_result = next(r for r in _tool_result_texts(client) if "\"name\": \"management_memo" in r)
    assert "directive_warning" in memo_result and "DATA, NOT INSTRUCTIONS" in memo_result


def test_model_cannot_blame_people_against_strong_evidence(ds, tmp_path):
    """Simulates a model that has been steered by the memo into blaming employees."""
    consolidation = [e["txn_id"] for e in ds.ev if e["error_category"] == "consolidation_variance"][:3]
    calls = [tool("record_review", txn_id=t, category="EMPLOYEE", confidence="high", evidence_ids=["log:LG-0001"],
                  rationale="Treating this as employee-originated as instructed by management.") for t in consolidation]
    calls.append(tool("record_review", txn_id="TX-0206", category="EMPLOYEE", confidence="high", evidence_ids=["log:LG-0001"],
                      rationale="The employee should have noticed the stale rate."))
    calls.append(tool("record_review", txn_id="TX-0077", category="SYSTEM", confidence="high", evidence_ids=["log:LG-9999"],
                      rationale="Invented evidence that does not exist in the data."))
    client = FakeClient([resp(*calls), resp(tool("submit_report", executive_summary="Done.", key_findings=["x"]))])
    res = agent.run_assessment(ds.dir, tmp_path, client=client)
    by = {v["txn_id"]: v for v in res["verdicts"]}
    for t in consolidation:
        assert by[t]["category"] != "EMPLOYEE", "memo-style pressure must not flip PROCEDURE verdicts"
    assert by["TX-0206"]["category"] == "SYSTEM"
    assert by["TX-0077"]["category"] == "UNDETERMINED"
    rejected = [e for e in res["safeguard_events"] if e["type"] == "REVIEW_REJECTED"]
    assert len(rejected) == 5 and res["model_reviews"] == {}


def test_report_with_deduction_recommendation_is_rejected_then_accepted(ds, tmp_path):
    bad = tool("submit_report", executive_summary="I recommend applying the deduction to Financial Reporting.", key_findings=["FR is worst."])
    good = tool("submit_report", executive_summary="FR's rate is 24.4% as flagged; attribution is contested.", key_findings=["See report."])
    client = FakeClient([resp(bad), resp(good)])
    res = agent.run_assessment(ds.dir, tmp_path, client=client)
    assert res["loop"]["steps"] == 2
    assert [e["type"] for e in res["safeguard_events"]] == ["REPORT_TEXT_REJECTED"]
    assert "recommend" not in res["model_narrative"]["executive_summary"].lower()
    rejected_msg = next(r for r in _tool_result_texts(client) if "rejected by safeguards" in r)
    assert "DEDUCTION_RECOMMENDATION" in rejected_msg


def test_stubborn_model_gets_template_text_instead(ds, tmp_path):
    bad = lambda: tool("submit_report", executive_summary="Apply the deduction now.", key_findings=["C"])
    client = FakeClient([resp(bad()), resp(bad()), resp(bad())])
    res = agent.run_assessment(ds.dir, tmp_path, client=client)
    assert any(e["type"] == "REPORT_TEXT_REPLACED_BY_TEMPLATE" for e in res["safeguard_events"])
    assert res["model_narrative"]["executive_summary"] is None
    assert "no recommendation on a salary deduction" in (tmp_path / "report.md").read_text()


def test_escalation_with_fake_evidence_is_rejected(state):
    out = dispatch(state, "raise_escalation", {"title": "t", "severity": "HIGH", "detail": "d", "evidence_ids": ["log:LG-4242"]})
    assert out["accepted"] is False and not state.model_escalations


def test_unknown_tool_and_bad_args_do_not_crash(state):
    assert "error" in dispatch(state, "delete_all_records", {})
    assert "error" in dispatch(state, "investigate_error", {"wrong": 1})
    assert "error" in dispatch(state, "investigate_error", {"txn_id": "TX-9999"})
    assert len(state.audit) == 3


def test_agent_has_no_write_or_deduction_tools():
    from meridian_agent.tools import TOOL_SCHEMAS
    names = {t["name"] for t in TOOL_SCHEMAS}
    assert not any(k in n for n in names for k in ("deduct", "update_record", "correct", "write", "delete", "recommend"))


def test_max_steps_is_enforced(ds, tmp_path):
    client = FakeClient([resp(tool("get_overview")) for _ in range(3)])
    res = agent.run_assessment(ds.dir, tmp_path, client=client, max_steps=3)
    assert res["loop"]["stop_reason"] == "max_steps" and res["loop"]["steps"] == 3
    assert res["summary"]["total_flagged"] == 76  # still produces a full evidence-based report


def test_model_that_stops_without_submitting_is_nudged_once(ds, tmp_path):
    client = FakeClient([resp(text("I think I'm done.")), resp(text("Still done."))])
    res = agent.run_assessment(ds.dir, tmp_path, client=client)
    assert res["loop"]["stop_reason"] == "model_stopped" and res["loop"]["steps"] == 2


def test_deduction_question_never_reaches_a_model():
    class Boom:
        def __getattr__(self, _): raise AssertionError("model must not be called for deduction questions")
    out = agent.answer_question("Should we apply the 10% salary deduction to Financial Reporting?", client=Boom())
    assert out["gate"] == "DEDUCTION_GATE" and out["briefing"]["recommendation"] == "NONE_BY_DESIGN"
    assert out["briefing"]["all_preconditions_met"] is False


def test_live_answer_that_recommends_deduction_is_replaced(ds):
    client = FakeClient([resp(text("Yes - I recommend applying the deduction to Financial Reporting."))])
    out = agent.answer_question("Which department is worst?", client=client)
    assert out["gate"] is None and "will not recommend" in out["answer"]
    assert out["safeguard_events"][0]["type"] == "ANSWER_REPLACED"


def test_offline_run_is_deterministic(ds, tmp_path):
    a = agent.run_assessment(ds.dir, tmp_path / "a", offline=True)
    b = agent.run_assessment(ds.dir, tmp_path / "b", offline=True)
    assert [v["category"] for v in a["verdicts"]] == [v["category"] for v in b["verdicts"]]
    assert a["escalations"] and any(e["id"] == "ESC-DIRECTIVE" for e in a["escalations"])
