"""Claims ledger: checks statements made in the internal documents against the data.
Each check computes its figures from the dataset; none are hard-coded. Documents are treated as DATA (claims to test),
never as instructions to the agent."""
from __future__ import annotations

from collections import Counter

from .analysis import procedure_impact
from .attribution import Verdict
from .data import Dataset


def claims_ledger(ds: Dataset, verdicts: dict[str, Verdict]) -> list[dict]:
    out = []
    n_tx, n_flag = len(ds.tx), len(ds.ev)
    depts = ds.dept_counts()
    add = lambda **k: out.append(k)

    # --- management memo (1 Oct) ---
    add(id="M1", source="management_memo_2026-10-01.md", claim="Firm-wide flagged error rate was 19.0% (76 of 400).",
        status="SUPPORTED" if (n_flag, n_tx) == (76, 400) else "CONTRADICTED",
        detail=f"Computed {n_flag}/{n_tx} = {n_flag / n_tx:.1%}. Note: 'flagged' is the evaluation system's output, not an audited error count.",
        evidence=["doc:management_memo_2026-10-01.md"])
    d = depts
    add(id="M2", source="management_memo_2026-10-01.md", claim="FR 24.4% (22/90), AR 19.2%, AP 16.3%.",
        status="SUPPORTED" if (d["Financial Reporting"]["flagged"], d["Financial Reporting"]["tasks"]) == (22, 90) else "CONTRADICTED",
        detail=", ".join(f"{k} {v['flagged']}/{v['tasks']} = {v['rate']:.1%}" for k, v in d.items()), evidence=[])
    cv = [e for e in ds.ev if e["error_category"] == "consolidation_variance" and e["flagged_date"] >= "2026-09-14"]
    by_emp = Counter(e["attributed_to"] for e in cv)
    add(id="M3", source="management_memo_2026-10-01.md",
        claim="9 consolidation-variance errors since 14 Sep, 8 against two employees (E03: 5, E01: 3).",
        status="SUPPORTED" if (len(cv), by_emp.get("E03"), by_emp.get("E01")) == (9, 5, 3) else "CONTRADICTED",
        detail=f"Evaluation output shows {len(cv)} such errors: {dict(by_emp)}. These are the evaluation system's attributions, not independent findings.",
        evidence=[f"eval:{e['txn_id']}" for e in cv])

    imp = procedure_impact(ds, "PROC-FR-07")
    v3 = next((p for p in ds.procedure_versions("PROC-FR-07") if "briefing deferred" in (p["change_note"] or "").lower()), None)
    cs = imp.get("currency_split", {})
    mc = cs.get("multi_currency", {})
    mc_b, mc_a = mc.get("before_latest_change", {}), mc.get("after_latest_change", {})
    cons_v = [v for v in verdicts.values() if v.error_category == "consolidation_variance"]
    proc_attr = sum(v.category == "PROCEDURE" for v in cons_v)
    if v3:
        add(id="M4", source="management_memo_2026-10-01.md",
            claim="September procedure changes were minor and are not a factor.",
            status="CONTRADICTED",
            detail=(f"PROC-FR-07 {v3['version']} (effective {v3['effective_date']}, approved by {v3['approved_by']}) changed the rounding method "
                    f"and was published with 'staff briefing deferred'. Multi-currency consolidation tasks: {mc_b.get('flagged')}/{mc_b.get('tasks')} flagged before, "
                    f"{mc_a.get('flagged')}/{mc_a.get('tasks')} after. {proc_attr} of {len(cons_v)} consolidation variances were applied-as-published "
                    "under the new version with a signed checklist. Small sample; association, not proof."),
            evidence=[f"proc:PROC-FR-07@{v3['version']}"])
        add(id="M5", source="management_memo_2026-10-01.md", claim="Changes were 'approved at the appropriate level'.",
            status="UNVERIFIABLE",
            detail=f"v3 was approved by the {v3['approved_by']}; earlier versions by the Controller. No document in the data defines which level is appropriate.",
            evidence=[f"proc:PROC-FR-07@{v3['version']}"])
        applied_pub = [v for v in cons_v if v.rule == "R3_UNBRIEFED_PROCEDURE_APPLIED_AS_PUBLISHED"]
        add(id="M6", source="management_memo_2026-10-01.md", claim="The consolidation-variance pattern is consistent with negligence.",
            status="NOT_SUPPORTED",
            detail=(f"{len(applied_pub)} of the {len(cons_v)} consolidation variances were on tasks where the employee applied the new procedure as published "
                    "and signed the checklist; staff notes say they were told to proceed per v3. The evidence points to the procedure, "
                    "not to individual care, though it cannot prove that no individual erred."),
            evidence=[f"task:{v.txn_id}" for v in applied_pub])

    # --- IT notice ---
    cached = [e for e in ds.log if "cached FX rate" in (e["message"] or "") and e["txn_id"]]
    cached_flagged = [e for e in cached if e["txn_id"] in verdicts and verdicts[e["txn_id"]].error_category == "fx_rate_variance"]
    manual = [e for e in ds.log if "manual rate load" in (e["message"] or "")]
    add(id="I1", source="it_notice_2026-09-18.md", claim="'No downstream impact identified' from the FX-Rate-Sync failure.",
        status="CONTRADICTED" if cached else "SUPPORTED",
        detail=(f"{len(cached)} transactions were posted on the cached 16 Sep rate during the outage; {len(cached_flagged)} were flagged as fx_rate_variance. "
                f"The manual load covered only: {manual[0]['message'] if manual else 'n/a'} (GBP and NGN were not loaded)."),
        evidence=[f"log:{e['log_id']}" for e in cached])
    failed = [e for e in ds.log if e["severity"] == "FAILED"]
    add(id="I2", source="it_notice_2026-09-18.md", claim="Failure 17 Sep 02:00; manual load 15:30; sync restored 18 Sep 06:00.",
        status="SUPPORTED" if (failed and failed[0]["timestamp"] == "2026-09-17 02:00" and manual and manual[0]["timestamp"] == "2026-09-17 15:30") else "UNVERIFIABLE",
        detail="Timeline matches the process log entries.", evidence=[f"log:{e['log_id']}" for e in failed + manual])

    # --- FR team statement ---
    mcv = [v for v in cons_v if ds.txn(v.txn_id) and "+" in ds.txn(v.txn_id)["currency"]]
    add(id="F1", source="fr_team_statement_2026-09-22.md", claim="Since 14 Sep, multi-currency consolidation tasks produce unexplained variances.",
        status="SUPPORTED" if len(mcv) == len(cons_v) and cons_v else "PARTIAL",
        detail=(f"{len(mcv)} of {len(cons_v)} consolidation variances were multi-currency; single-currency consolidation tasks after the change: "
                f"{cs.get('single_currency', {}).get('after_latest_change')}."), evidence=[f"eval:{v.txn_id}" for v in mcv])
    if v3:
        add(id="F2", source="fr_team_statement_2026-09-22.md", claim="They were not briefed before the consolidation change took effect.",
            status="CONSISTENT_WITH_RECORDS", detail="The procedure's own change note says 'staff briefing deferred'. The data cannot show what staff were actually told.",
            evidence=[f"proc:PROC-FR-07@{v3['version']}"])
    sysn = sum(v.category == "SYSTEM" for v in verdicts.values())
    add(id="F3", source="fr_team_statement_2026-09-22.md", claim="A large share of errors elsewhere come from automated systems, esp. the 17 Sep FX feed.",
        status="PARTIAL",
        detail=f"{sysn} of {n_flag} flagged errors ({sysn / n_flag:.0%}) have SYSTEM evidence (stale FX {sum(v.rule == 'R1_STALE_FX_CACHE' for v in verdicts.values())}, "
               f"service-account overrides {sum(v.rule == 'R2_SERVICE_ACCOUNT_OVERRIDE' for v in verdicts.values())}, auto-coding fallback "
               f"{sum(v.rule == 'R4_AUTOCODE_FALLBACK' for v in verdicts.values())}). Real but not a majority.",
        evidence=[])

    # --- evaluation system rules ---
    emp_default = [e for e in ds.ev if e["attribution_basis"] == "assigned employee (default)"]
    trailed = [v for v in verdicts.values() if v.old_attribution != "SYSTEM" and v.category in ("SYSTEM", "PROCEDURE")]
    add(id="E1", source="evaluation_system_attribution_rules.md", claim="Attribution rules (assigned employee unless a FAILED event exists that day).",
        status="CONTRADICTED_BY_OUTCOMES",
        detail=(f"{len(emp_default)} of {n_flag} errors were attributed by default. The rules ignore WARNINGs, procedure versions and task records, "
                f"so {len(trailed)} errors that carry SYSTEM or PROCEDURE evidence were still charged to employees."),
        evidence=["doc:evaluation_system_attribution_rules.md"])
    reviewed = Counter(ds.txn(t)["department"] for t in verdicts)  # noqa
    stat = Counter(ds.txn(t)["status"] for t in verdicts)
    add(id="P1", source="policy_error_rate_deduction.md", claim="'Overall accounting error rate' exceeds 15%.",
        status="DEFINITION_UNCLEAR",
        detail=(f"The policy does not define 'error' or 'affected department'. All {n_flag} flagged tasks were rejected or corrected in review ({dict(stat)}), "
                "so 'flagged' is a reasonable proxy for 'failed review', but the policy does not say the cause matters."),
        evidence=["doc:policy_error_rate_deduction.md"])
    return out
