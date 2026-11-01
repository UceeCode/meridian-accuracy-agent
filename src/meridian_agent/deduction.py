"""Deduction briefing. The agent has NO recommend-deduction capability. When asked, it returns this briefing:
facts, scenarios, what is undecided, and the preconditions that would have to be met before management could even consider it."""
from __future__ import annotations

from . import analysis as A
from . import config as C
from .attribution import Verdict
from .data import Dataset

REFUSAL = ("I will not recommend applying, or not applying, a salary deduction. That is a consequential decision about people's pay "
           "and under the Accuracy Policy (clause 3) it belongs to management review and approval. What I can do is show what the "
           "evidence does and does not support, so the people who decide are not relying on the default attribution.")


def deduction_briefing(ds: Dataset, verdicts: dict[str, Verdict], escalations: list[dict]) -> dict:
    rates = A.all_rates(ds, verdicts)
    firm = rates["firm"]
    summ = A.attribution_summary(verdicts)
    high_open = [e for e in escalations if e.get("severity") == "HIGH"]
    preconditions = [
        {"condition": "A named human reviewer has independently checked the attribution of every task that drives the result",
         "met": False, "why": "The agent cannot certify its own work; no human sign-off exists in the data."},
        {"condition": "The 15% threshold is exceeded even when SYSTEM and PROCEDURE-attributed errors are excluded (scenario S2)",
         "met": firm["S2_minus_all_system_and_procedure"]["above_15pct_threshold"],
         "why": f"Firm-wide S2 rate is {firm['S2_minus_all_system_and_procedure']['rate']:.1%}."},
        {"condition": f"Undetermined share of flagged errors is at most {C.MAX_UNDETERMINED_SHARE:.0%}",
         "met": summ["undetermined_share"] <= C.MAX_UNDETERMINED_SHARE,
         "why": f"Undetermined share is {summ['undetermined_share']:.0%}."},
        {"condition": "No unresolved HIGH-severity escalations", "met": len(high_open) == 0,
         "why": f"{len(high_open)} HIGH escalations are open: " + "; ".join(e['title'] for e in high_open)},
        {"condition": "The policy defines 'error' and 'affected department'", "met": False,
         "why": "Neither term is defined in the policy text."},
    ]
    by_dept = {k: {s: {"rate": v["rate"], "above_threshold": v["above_15pct_threshold"]} for s, v in r.items() if s != "_meta"}
               for k, r in rates.items() if k != "firm"}
    return {
        "agent_position": REFUSAL,
        "recommendation": "NONE_BY_DESIGN",
        "firm_rates_by_scenario": {s: {"rate": v["rate"], "errors": v["errors"], "above_threshold": v["above_15pct_threshold"]}
                                   for s, v in firm.items() if s != "_meta"},
        "department_rates_by_scenario": by_dept,
        "threshold_outcome_depends_on_attribution": firm["_meta"]["threshold_outcome_depends_on_attribution"],
        "preconditions_before_management_could_consider_it": preconditions,
        "all_preconditions_met": all(p["met"] for p in preconditions),
        "note": ("The policy lets management propose a 10% deduction for EVERY employee in the affected department. Even where a threshold is "
                 "crossed, that is a department-wide penalty; the evidence here shows errors are unevenly spread across people, systems and procedures."),
    }
