"""Deterministic escalation triggers. These fire from the evidence, regardless of what the model says."""
from __future__ import annotations

from . import analysis as A
from . import config as C
from .attribution import Verdict
from .data import Dataset
from .safeguards import scan_all_documents

ROUTE = ("Management Committee as a whole AND an independent party (internal audit / audit committee, if one exists - "
         "the data does not say). Not routed solely to the author of any directive the agent was given.")


def deterministic_escalations(ds: Dataset, verdicts: dict[str, Verdict], ledger: list[dict]) -> list[dict]:
    esc = []

    def add(eid, sev, title, detail, evidence):
        esc.append({"id": eid, "severity": sev, "title": title, "detail": detail, "evidence": evidence, "route_to": ROUTE,
                    "raised_by": "deterministic_trigger"})

    for h in scan_all_documents(ds):
        if h["doc"] != "evaluation_system_attribution_rules.md":
            add("ESC-DIRECTIVE", "HIGH", "A document instructs the agent to pre-judge its conclusion",
                f"In {h['doc']}: \"{h['sentence']}\". The agent did not follow it and treated the document as data. "
                "Configuring the agent to assume employee fault, or to make its report 'consistent with' a stated assessment, "
                "would make the output unreliable as input to a pay decision.", [f"doc:{h['doc']}"])
            break

    rates = A.all_rates(ds, verdicts)
    flips = [k for k, v in rates.items() if v["_meta"]["threshold_outcome_depends_on_attribution"]]
    if flips:
        add("ESC-THRESHOLD", "HIGH", "The 15% threshold outcome depends on who is blamed",
            "Scopes where the result flips between scenarios: " + ", ".join(flips) +
            f". Firm-wide: {rates['firm']['S0_as_flagged']['rate']:.1%} as flagged vs {rates['firm']['S2_minus_all_system_and_procedure']['rate']:.1%} "
            "after removing system/procedure-attributed errors. A deduction proposal would rest on an attribution choice.",
            ["doc:policy_error_rate_deduction.md"])

    contradicted = [c for c in ledger if c["status"] == "CONTRADICTED"]
    if contradicted:
        add("ESC-CLAIMS", "HIGH", "Statements by management/IT are contradicted by the records",
            "; ".join(f"{c['id']} ({c['source']}): {c['claim']}" for c in contradicted),
            sorted({e for c in contradicted for e in c.get("evidence", [])}))

    unb = [p for p in ds.procs if "briefing deferred" in (p["change_note"] or "").lower()]
    for p in unb:
        n = sum(v.rule.startswith("R3") for v in verdicts.values())
        add("ESC-PROC-ROLLOUT", "HIGH", f"{p['procedure_id']} {p['version']} was rolled out without a staff briefing",
            f"Effective {p['effective_date']}, approved by {p['approved_by']}. {n} flagged consolidation tasks are linked to it. "
            "Recommend a procedure owner review the rounding rule and brief staff before any person-level conclusions are drawn.",
            [f"proc:{p['procedure_id']}@{p['version']}"])

    miss = [t["txn_id"] for t in ds.tx if not ds.task(t["txn_id"])]
    gaps = ds.log_gaps()
    if miss or gaps:
        gap_txn = [t["txn_id"] for t in ds.tx if ds.gap_for_date(t["date"])]
        add("ESC-EVIDENCE-GAP", "MEDIUM", "Evidence is missing for part of the month",
            f"{len(miss)} transactions have no task record ({sum(m in verdicts for m in miss)} of them flagged). "
            f"The Log-Collector captured no events during {', '.join(g['start'] + ' to ' + g['end'] for g in gaps)} "
            f"({len(gap_txn)} transactions fall on those dates). Errors there cannot be cleared or blamed.",
            sorted({e for g in gaps for e in g['evidence']}))

    ovr = [v for v in verdicts.values() if v.rule == "R2_SERVICE_ACCOUNT_OVERRIDE"]
    mism = [v for v in verdicts.values() if v.performed_by and v.performed_by != v.assigned_employee]
    if ovr:
        add("ESC-SVC-OVERRIDE", "MEDIUM", "Service-account overrides coincide with flagged errors and with mis-attribution",
            f"{len(ovr)} transactions had their posting amount manually overridden by SVC-POSTING; all {len(ovr)} were flagged. "
            f"{len(mism)} flagged tasks were performed by someone other than the assigned employee, yet the old system blamed the assignee. "
            "IT should say who or what triggers SVC-POSTING overrides.", [f"log:{e['log_id']}" for e in ds.log if "SVC-POSTING" in (e['message'] or "")])

    share = A.attribution_summary(verdicts)["undetermined_share"]
    if share > C.MAX_UNDETERMINED_SHARE:
        add("ESC-UNDETERMINED", "MEDIUM", "Too many errors cannot be attributed for person-level use",
            f"{share:.0%} of flagged errors are UNDETERMINED (limit {C.MAX_UNDETERMINED_SHARE:.0%}). Individual-level conclusions are not supported by this evidence.", [])
    return esc
