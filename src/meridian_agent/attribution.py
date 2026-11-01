"""Evidence-based attribution engine.

The company's old evaluation system blames the assigned employee by default and only looks for FAILED events.
This engine instead asks, for every flagged task, *what evidence exists for each possible source*
(employee / automated system / management-approved procedure) and says UNDETERMINED when the evidence
does not support a conclusion. Every verdict lists the evidence ids it rests on, so a human can check it.

Rules are applied in priority order; the first that fires decides the verdict.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from . import config as C
from .data import Dataset


@dataclass
class Verdict:
    txn_id: str
    department: str
    error_category: str
    assigned_employee: str
    performed_by: str | None            # from task record; may differ from assigned
    old_attribution: str                # what the existing evaluation system said
    category: str                       # EMPLOYEE | SYSTEM | PROCEDURE | SHARED | UNDETERMINED
    confidence: str                     # low | medium | high
    responsible_employee: str | None    # only set when category is EMPLOYEE or SHARED
    rule: str
    reasoning: str
    evidence: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    employee_indicator: bool = False    # True only if there is positive evidence of an employee-side lapse
    reviewed_by_model: bool = False

    def to_dict(self):
        return asdict(self)


def _events(ds: Dataset, tid: str, needle: str):
    return [e for e in ds.events_for_txn(tid) if needle in (e["message"] or "")]


def attribute(ds: Dataset, tid: str) -> Verdict:
    t, ev, rec = ds.txn(tid), ds.eval_row(tid), ds.task(tid)
    cat = ev["error_category"]
    assigned = t["assigned_employee"]
    performed = rec["employee_id"] if rec else None
    evid = [f"txn:{tid}", f"eval:{tid}"]
    caveats: list[str] = []

    def make(category, confidence, rule, reasoning, responsible=None, extra=(), indicator=False):
        return Verdict(
            txn_id=tid, department=t["department"], error_category=cat, assigned_employee=assigned,
            performed_by=performed, old_attribution=ev["attributed_to"], category=category, confidence=confidence,
            responsible_employee=responsible, rule=rule, reasoning=reasoning,
            evidence=list(dict.fromkeys(evid + list(extra))), caveats=caveats, employee_indicator=indicator)

    if rec:
        evid.append(f"task:{tid}")
        if performed and performed != assigned:
            caveats.append(f"Task record shows {performed} performed the work, not the assigned employee {assigned}; "
                           f"the old system blamed {ev['attributed_to']}.")
    who = performed or assigned

    # R0 - no task record: we cannot see who did the work, which procedure version, or the checklist.
    if not rec:
        caveats.append("No task record exists, so performer, procedure version applied and checklist status are unknown.")
        return make("UNDETERMINED", "low", "R0_NO_TASK_RECORD",
                    "No task record exists for this transaction, so the evidence needed to attribute it is absent. "
                    "The old system blamed the assigned employee by default (its rule 4); that is not evidence.")

    # R1 - FX posted on a stale cached rate during the FX-Rate-Sync failure.
    cached = _events(ds, tid, "cached FX rate")
    if cat == "fx_rate_variance" and cached:
        failed = [e for e in ds.events_window(date_from=t["date"], date_to=t["date"], process="FX-Rate-Sync", severity="FAILED")]
        return make("SYSTEM", "high", "R1_STALE_FX_CACHE",
                    "Ledger-Posting used a cached FX rate while the FX-Rate-Sync job had failed; the variance is consistent "
                    "with the stale rate, not with the employee's work.",
                    extra=[f"log:{e['log_id']}" for e in cached + failed])

    # R2 - a service account manually overrode the posted amount.
    ovr = _events(ds, tid, "manual override of posting amount by SVC-POSTING")
    if ovr:
        caveats.append("Who or what triggered the SVC-POSTING override is not recorded; IT must confirm.")
        return make("SYSTEM", "medium", "R2_SERVICE_ACCOUNT_OVERRIDE",
                    "A service account (SVC-POSTING) manually overrode the posting amount on this transaction. "
                    "The amount that was posted was therefore not (only) the employee's work.",
                    extra=[f"log:{e['log_id']}" for e in ovr])

    # R3 - consolidation variance under a procedure version rolled out without staff briefing.
    if cat == "consolidation_variance":
        pf = ds.procedure_in_force(t["procedure_id"], t["date"])
        applied = rec["procedure_version_applied"]
        if pf and "briefing deferred" in (pf["change_note"] or "").lower():
            pev = f"proc:{pf['procedure_id']}@{pf['version']}"
            note = rec.get("notes")
            if note:
                caveats.append(f"Staff note on task: '{note}'.")
            if applied == pf["version"] and rec["checklist_signed"] == "Y":
                return make("PROCEDURE", "high", "R3_UNBRIEFED_PROCEDURE_APPLIED_AS_PUBLISHED",
                            f"The employee applied {pf['procedure_id']} {applied} as published and signed the checklist. "
                            f"That version took effect {pf['effective_date']} (approved by {pf['approved_by']}) with 'staff briefing deferred'. "
                            "Consolidation variances began with this change.", extra=[pev])
            if applied != pf["version"]:
                caveats.append(f"Employee applied {applied}, which had been superseded by {pf['version']} on {pf['effective_date']}.")
                return make("SHARED", "medium", "R3_SUPERSEDED_VERSION_USED",
                            "The employee used a superseded procedure version, but the replacement was rolled out with the "
                            "staff briefing deferred, so the employee may not have been told. Responsibility is shared.",
                            responsible=who, extra=[pev], indicator=True)
            return make("SHARED", "medium", "R3_UNBRIEFED_PROCEDURE_CHECKLIST_UNSIGNED",
                        "Employee applied the new, unbriefed procedure version but did not sign the checklist.",
                        responsible=who, extra=[pev], indicator=True)

    # R4 - auto-coding engine used a fallback tax-code mapping.
    fb = _events(ds, tid, "tax-code mapping fallback")
    if cat == "wrong_account_code" and fb:
        extra = [f"log:{e['log_id']}" for e in fb]
        if rec["checklist_signed"] == "N":
            return make("SHARED", "medium", "R4_AUTOCODE_FALLBACK_AND_CHECKLIST_UNSIGNED",
                        "The Auto-Coding-Engine used a fallback mapping on this task, and the checklist was not signed.",
                        responsible=who, extra=extra, indicator=True)
        return make("SYSTEM", "medium", "R4_AUTOCODE_FALLBACK",
                    "The Auto-Coding-Engine logged a fallback tax-code mapping on this task, which is a plausible cause of a "
                    "wrong account code. The checklist was signed.", extra=extra)

    # R5 - duplicate detector warned and a person overrode / dismissed the warning.
    dup = _events(ds, tid, "possible duplicate")
    if cat == "duplicate_entry" and dup:
        e = dup[0]
        m = re.search(r"\b(E\d{2})\b", e["message"])
        reviewer = m.group(1) if m else who
        explicit = "user override recorded" in e["message"]
        caveats.append("The Duplicate-Detector raised 22 warnings in September; only 8 of those transactions were later flagged, "
                       "so dismissing a warning is usually correct. This is a judgement call, not proof of carelessness.")
        return make("EMPLOYEE", "medium" if explicit else "low", "R5_WARNING_OVERRIDDEN",
                    f"The Duplicate-Detector warned about this transaction and {reviewer} "
                    f"{'overrode the warning' if explicit else 'dismissed it after review'}; it was later flagged as a duplicate.",
                    responsible=reviewer, extra=[f"log:{e['log_id']}"], indicator=True)

    # R6 - evidence collection was down on this date and the error type could be system-caused.
    gap = ds.gap_for_date(t["date"])
    if gap and cat in C.SYSTEM_PLAUSIBLE:
        caveats.append(f"Log-Collector captured no events {gap['start']} to {gap['end']}.")
        return make("UNDETERMINED", "low", "R6_LOG_GAP_SYSTEM_PLAUSIBLE",
                    "Automated-system events are missing for this date and this error type could be system-caused, "
                    "so a system cause can neither be shown nor ruled out.", extra=gap["evidence"])
    if gap:
        caveats.append(f"Log gap on this date ({gap['start']} to {gap['end']}); system involvement cannot be fully excluded.")

    # R7 - checklist not signed: a positive employee-side process lapse.
    if rec["checklist_signed"] == "N":
        conf = "medium" if cat in C.HUMAN_TYPE else "low"
        if conf == "low":
            caveats.append("Error type is system-plausible and no causal link to the missed checklist is shown.")
        return make("EMPLOYEE", conf, "R7_CHECKLIST_UNSIGNED",
                    "The checklist for this task was not signed, which is an employee-side process lapse "
                    "(correlation with the error, not proof of cause).", responsible=who, indicator=True)

    # R8 - human-type error, procedure followed, no external cause found.
    if cat in C.HUMAN_TYPE:
        return make("EMPLOYEE", "low", "R8_HUMAN_TYPE_RESIDUAL",
                    "This is a typically human-type error; the checklist was signed and no system or procedure cause was found. "
                    "Attribution to the employee is by elimination only.", responsible=who, indicator=False)

    # R9 - nothing decisive either way.
    return make("UNDETERMINED", "low", "R9_NO_DECISIVE_EVIDENCE",
                "This error type could be caused by a person or a system; no event, procedure change or checklist lapse "
                "points either way.")


def attribute_all(ds: Dataset) -> dict[str, Verdict]:
    return {tid: attribute(ds, tid) for tid in ds.flagged_ids()}
