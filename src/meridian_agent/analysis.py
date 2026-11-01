"""Deterministic analysis helpers: error-rate scenarios, procedure impact, employee profiles.
All numbers shown to the model or in the report come from here, never from the model's own arithmetic."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from . import config as C
from .attribution import Verdict
from .data import Dataset

SCENARIOS = {
    "S0_as_flagged": "Every flagged task counts (what management used).",
    "S1_minus_high_confidence_nonemployee": "Remove flagged tasks with HIGH-confidence SYSTEM or PROCEDURE evidence.",
    "S2_minus_all_system_and_procedure": "Remove all flagged tasks attributed to SYSTEM or PROCEDURE (medium or high).",
    "S3_person_involved_medium_plus_only": "Count only tasks with medium-or-higher evidence of an employee-side cause (EMPLOYEE or SHARED).",
}


def _in_scenario(v: Verdict, scen: str) -> bool:
    if scen == "S0_as_flagged": return True
    if scen == "S1_minus_high_confidence_nonemployee":
        return not (v.category in ("SYSTEM", "PROCEDURE") and v.confidence == "high")
    if scen == "S2_minus_all_system_and_procedure":
        return v.category not in ("SYSTEM", "PROCEDURE")
    if scen == "S3_person_involved_medium_plus_only":
        return v.category in ("EMPLOYEE", "SHARED") and C.CONF_RANK[v.confidence] >= 2
    raise ValueError(scen)


def error_rates(ds: Dataset, verdicts: dict[str, Verdict], department: str | None = None) -> dict:
    depts = ds.dept_counts()
    total = sum(d["tasks"] for d in depts.values()) if not department else depts[department]["tasks"]
    out = {}
    for scen, desc in SCENARIOS.items():
        vs = [v for v in verdicts.values() if (not department or v.department == department) and _in_scenario(v, scen)]
        rate = len(vs) / total
        out[scen] = {"description": desc, "errors": len(vs), "tasks": total, "rate": round(rate, 4),
                     "above_15pct_threshold": rate > C.ERROR_RATE_THRESHOLD}
    out["_meta"] = {"scope": department or "firm-wide", "threshold": C.ERROR_RATE_THRESHOLD,
                    "threshold_outcome_depends_on_attribution": len({s["above_15pct_threshold"] for k, s in out.items() if k != "_meta"}) > 1,
                    "undetermined_errors": sum(1 for v in verdicts.values() if v.category == "UNDETERMINED"
                                               and (not department or v.department == department))}
    return out


def all_rates(ds, verdicts):
    r = {"firm": error_rates(ds, verdicts)}
    for d in ds.dept_counts():
        r[d] = error_rates(ds, verdicts, d)
    return r


def attribution_summary(verdicts: dict[str, Verdict]) -> dict:
    by = Counter(v.category for v in verdicts.values())
    by_conf = Counter(f"{v.category}/{v.confidence}" for v in verdicts.values())
    old = Counter("SYSTEM" if v.old_attribution == "SYSTEM" else "EMPLOYEE" for v in verdicts.values())
    return {"total_flagged": len(verdicts), "new_category_counts": dict(by), "new_category_confidence_counts": dict(by_conf),
            "old_system_counts": dict(old), "undetermined_share": round(by.get("UNDETERMINED", 0) / max(1, len(verdicts)), 3)}


# ---- procedure impact -------------------------------------------------------------------------------
def procedure_impact(ds: Dataset, procedure_id: str) -> dict:
    flagged = set(ds.flagged_ids())
    rows = [t for t in ds.tx if t["procedure_id"] == procedure_id]
    vers = ds.procedure_versions(procedure_id)
    by_applied: dict[str, list] = defaultdict(list)
    for t in rows:
        rec = ds.task(t["txn_id"])
        by_applied[rec["procedure_version_applied"] if rec else "no_record"].append(t["txn_id"] in flagged)
    applied = {k: {"tasks": len(v), "flagged": sum(v), "rate": round(sum(v) / len(v), 3)} for k, v in sorted(by_applied.items())}
    result = {"procedure_id": procedure_id,
              "versions": [{k: p[k] for k in ("version", "effective_date", "approved_by", "change_note")} for p in vers],
              "flag_rate_by_version_applied": applied}
    # before/after the newest version that has data on both sides
    pre_post = []
    for p in vers[1:]:
        pre = [t["txn_id"] in flagged for t in rows if t["date"] < p["effective_date"]]
        post = [t["txn_id"] in flagged for t in rows if t["date"] >= p["effective_date"]]
        if pre and post:
            pre_post.append({"change": p["version"], "effective": p["effective_date"],
                             "before": {"tasks": len(pre), "flagged": sum(pre)},
                             "after": {"tasks": len(post), "flagged": sum(post)}})
    result["before_after_by_effective_date"] = pre_post
    if not pre_post:
        result["note"] = "No tasks on both sides of any version change, so no before/after comparison is possible."
    mc = [t for t in rows if "+" in t["currency"]]
    if mc:
        eff = vers[-1]["effective_date"] if vers else None
        split = {}
        for label, grp in (("multi_currency", mc), ("single_currency", [t for t in rows if "+" not in t["currency"]])):
            split[label] = {"before_latest_change": {"tasks": 0, "flagged": 0}, "after_latest_change": {"tasks": 0, "flagged": 0}}
        # use the unbriefed version's date if one exists, else the latest
        for p in vers:
            if "briefing deferred" in (p["change_note"] or "").lower():
                eff = p["effective_date"]
        for label, grp in (("multi_currency", mc), ("single_currency", [t for t in rows if "+" not in t["currency"]])):
            for t in grp:
                k = "after_latest_change" if t["date"] >= eff else "before_latest_change"
                split[label][k]["tasks"] += 1
                split[label][k]["flagged"] += t["txn_id"] in flagged
        result["currency_split"] = {"split_date": eff, **split}
    return result


# ---- employee profiles ------------------------------------------------------------------------------
def employee_profile(ds: Dataset, verdicts: dict[str, Verdict], eid: str) -> dict:
    e = ds.employee(eid)
    if not e:
        return {"error": f"unknown employee {eid}"}
    recs = [r for r in ds.tasks if r["employee_id"] == eid]
    unsigned = sum(r["checklist_signed"] == "N" for r in recs)
    own = [v for v in verdicts.values() if v.performed_by == eid]
    resp = [v for v in verdicts.values() if v.responsible_employee == eid]
    resp_mp = [v for v in resp if v.category in ("EMPLOYEE", "SHARED") and C.CONF_RANK[v.confidence] >= 2]
    old = [v for v in verdicts.values() if v.old_attribution == eid]
    return {
        "employee_id": eid, "name": e["name"], "department": e["department"], "role": e["role"],
        "tasks_performed_per_task_records": len(recs),
        "flagged_tasks_they_performed": len(own),
        "their_flagged_tasks_by_new_category": dict(Counter(v.category for v in own)),
        "old_system_blamed_them_for": len(old),
        "old_blamed_but_not_responsible_now": sum(1 for v in old if v.responsible_employee != eid),
        "medium_plus_employee_side_errors": len(resp_mp),
        "unsigned_checklist_tasks": unsigned,
        "unsigned_checklist_rate": round(unsigned / len(recs), 3) if recs else None,
        "caution": "Counts are small. Treat as a pattern for human review, never as a finding of negligence.",
    }


def employee_review_list(ds: Dataset, verdicts: dict[str, Verdict]) -> dict:
    profiles = [employee_profile(ds, verdicts, e["employee_id"]) for e in ds.emp]
    firm_unsigned = sum(r["checklist_signed"] == "N" for r in ds.tasks) / len(ds.tasks)
    flagged_unsigned = sum(1 for r in ds.tasks if r["checklist_signed"] == "N" and r["txn_id"] in verdicts)
    flagged_recs = sum(1 for r in ds.tasks if r["txn_id"] in verdicts)
    peers = [p["unsigned_checklist_rate"] for p in profiles if p["unsigned_checklist_rate"] is not None]
    candidates = []
    for p in profiles:
        if p["medium_plus_employee_side_errors"] >= C.MIN_EMPLOYEE_ERRORS_FOR_REVIEW:
            cats = p["their_flagged_tasks_by_new_category"]
            und = cats.get("UNDETERMINED", 0) / max(1, p["flagged_tasks_they_performed"])
            candidates.append({**p, "undetermined_share_of_their_flagged": round(und, 2),
                               "peer_median_unsigned_rate": round(median(peers), 3),
                               "status": "FOR_HUMAN_REVIEW_ONLY"})
    return {"profiles": profiles, "review_candidates": candidates,
            "firm_unsigned_checklist_rate": round(firm_unsigned, 3),
            "unsigned_rate_among_flagged_tasks": round(flagged_unsigned / max(1, flagged_recs), 3),
            "minimum_errors_for_listing": C.MIN_EMPLOYEE_ERRORS_FOR_REVIEW}
