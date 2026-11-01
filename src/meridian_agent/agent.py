"""The agent loop. The model decides what to investigate and calls tools; code enforces the safeguards.

run_assessment(...) works in two modes:
  * live    - Anthropic API model drives the investigation via tool use
  * offline - no model; the deterministic engine produces the full report (used for tests, CI, and if no API key is present)
In both modes the numbers come from code, and the same safeguards apply."""
from __future__ import annotations

import time

from . import analysis as A
from . import config as C
from . import deduction as D
from . import escalation as ESC
from . import report as R
from . import safeguards as S
from .data import Dataset
from .llm import make_backend
from .prompts import SYSTEM_PROMPT, TASK_PROMPT
from .tools import TOOL_SCHEMAS, RunState, dispatch, serialize


def investigate(state: RunState, backend, max_steps: int = C.MAX_STEPS, on_event=None,
                task: str = TASK_PROMPT, require_submit: bool = True) -> dict:
    """The tool-use loop. Each turn: the model replies; if it asked for tools we run them (through the safeguards) and feed the
    results back; we stop when it submits its report, stops asking for tools, or hits the step cap."""
    history = [{"role": "user", "text": task}]
    last_text, steps, nudged, stop = "", 0, False, "max_steps"
    while steps < max_steps:
        steps += 1
        reply = backend.chat(SYSTEM_PROMPT, history, TOOL_SCHEMAS)
        history.append({"role": "assistant", "text": reply.text, "tool_calls": reply.tool_calls})
        if reply.text.strip():
            last_text = reply.text.strip()
            if on_event:
                on_event("thought", last_text)
        if not reply.tool_calls:
            if require_submit and state.submission is None and not nudged:
                nudged = True
                history.append({"role": "user", "text": "You have not submitted yet. Call submit_report when your investigation is complete."})
                continue
            stop = "model_stopped"
            break
        results = []
        for c in reply.tool_calls:
            out = dispatch(state, c.name, c.input)
            if on_event:
                on_event("tool", f"{c.name}({_short(c.input)})")
            results.append({"id": c.id, "name": c.name, "content": serialize(out, backend.result_char_limit)})
        history.append({"role": "tool", "results": results})
        if require_submit and state.submission is not None:
            stop = "submitted"
            break
    return {"steps": steps, "stop_reason": stop, "last_text": last_text}


def _short(d):
    s = str(d)
    return s if len(s) < 90 else s[:87] + "..."


def build_result(state: RunState, mode: str, model: str | None, loop: dict | None) -> dict:
    ds = state.ds
    verdicts = state.final_verdicts()
    ledger = state.ledger
    esc = ESC.deterministic_escalations(ds, verdicts, ledger) + state.model_escalations
    result = {
        "mode": mode, "model": model, "loop": loop, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": A.attribution_summary(verdicts),
        "rates": A.all_rates(ds, verdicts),
        "verdicts": [v.to_dict() for v in verdicts.values()],
        "claims": ledger,
        "escalations": esc,
        "procedure_impact": {pid: A.procedure_impact(ds, pid) for pid in ("PROC-FR-07",)},
        "employee_review": A.employee_review_list(ds, verdicts),
        "deduction_briefing": D.deduction_briefing(ds, verdicts, esc),
        "model_reviews": state.reviews,
        "safeguard_events": state.safeguard_events,
        "model_narrative": state.submission,
        "dept_category_matrix": _matrix(ds, verdicts),
    }
    return result


def _matrix(ds, verdicts):
    out = {}
    for v in verdicts.values():
        out.setdefault(v.department, {}).setdefault(v.category, 0)
        out[v.department][v.category] += 1
    return out


def run_assessment(data_dir=C.DATA_DIR, out_dir=C.OUT_DIR, offline=False, client=None, backend=None, provider=None, model=None,
                   max_steps=C.MAX_STEPS, on_event=None) -> dict:
    ds = Dataset(data_dir)
    state = RunState.new(ds)
    loop, label = None, None
    if not offline:
        backend = backend or make_backend(provider, model, client)
        label = backend.label
        loop = investigate(state, backend, max_steps, on_event)
    result = build_result(state, "offline-deterministic" if offline else "live-model", label, loop)
    R.write_outputs(result, state.audit, out_dir)
    return result


# ----------------------------------------------------------------------------------------- Q&A about the results
def answer_question(question: str, data_dir=C.DATA_DIR, client=None, backend=None, provider=None, model=None, offline=False, on_event=None) -> dict:
    """Ask the agent a question.
    - Deduction questions NEVER reach a model: they get the deterministic briefing (the gate is code, not a prompt).
    - Other questions go to the live agent with the same tools; its answer is scanned before it is returned."""
    ds = Dataset(data_dir)
    state = RunState.new(ds)
    verdicts = state.final_verdicts()
    esc = ESC.deterministic_escalations(ds, verdicts, state.ledger)
    if S.is_deduction_request(question):
        state.audit.append({"tool": "deduction_gate", "input": question, "ok": True, "output_preview": "refused to recommend; briefing returned"})
        return {"gate": "DEDUCTION_GATE", "briefing": D.deduction_briefing(ds, verdicts, esc), "audit": state.audit}
    if offline:
        return {"gate": None, "answer": "Offline mode cannot answer free-form questions. Run `meridian-agent run --offline` and read outputs/report.md.", "audit": []}
    backend = backend or make_backend(provider, model, client)
    loop = investigate(state, backend, 20, on_event,
                       task=f"A colleague asks: {question}\n\nAnswer using the tools. Cite evidence ids. If the question cannot be answered from the data, say so.",
                       require_submit=False)
    text = loop["last_text"]
    viol = S.scan_text(text)
    if viol:
        state.safeguard_events.append({"type": "ANSWER_REPLACED", "violations": viol})
        text = D.REFUSAL if any(v["type"] == "DEDUCTION_RECOMMENDATION" for v in viol) else "[Answer withheld: it labelled a named person. See outputs/report.md for evidence-based patterns.]"
    return {"gate": None, "answer": text, "safeguard_events": state.safeguard_events, "audit": state.audit}
