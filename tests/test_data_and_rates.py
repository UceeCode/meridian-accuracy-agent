from meridian_agent import analysis as A
from meridian_agent.attribution import attribute_all


def test_headline_numbers_reproduce(ds):
    assert (len(ds.tx), len(ds.ev)) == (400, 76)
    d = ds.dept_counts()
    assert (d["Financial Reporting"]["flagged"], d["Financial Reporting"]["tasks"]) == (22, 90)
    assert round(d["Accounts Receivable"]["rate"], 3) == 0.192
    assert round(d["Accounts Payable"]["rate"], 3) == 0.163


def test_old_system_defaulted_to_employee_72_times(ds):
    assert sum(e["attribution_basis"].startswith("assigned") for e in ds.ev) == 72


def test_log_gap_detected(ds):
    g = ds.log_gaps()
    assert len(g) == 1 and g[0]["start"].startswith("2026-09-08") and g[0]["end"].startswith("2026-09-09")
    assert ds.gap_for_date("2026-09-08") and not ds.gap_for_date("2026-09-10")


def test_procedure_in_force(ds):
    assert ds.procedure_in_force("PROC-FR-07", "2026-09-13")["version"] == "v2"
    assert ds.procedure_in_force("PROC-FR-07", "2026-09-14")["version"] == "v3"
    assert ds.procedure_in_force("PROC-FR-07", "2026-09-29")["version"] == "v3.1"


def test_scenarios_and_threshold_flip(ds):
    v = attribute_all(ds)
    r = A.error_rates(ds, v)
    assert r["S0_as_flagged"]["errors"] == 76 and r["S0_as_flagged"]["above_15pct_threshold"]
    assert r["S2_minus_all_system_and_procedure"]["errors"] == 54 and not r["S2_minus_all_system_and_procedure"]["above_15pct_threshold"]
    assert r["_meta"]["threshold_outcome_depends_on_attribution"] is True


def test_fr_consolidation_pattern(ds):
    imp = A.procedure_impact(ds, "PROC-FR-07")
    mc = imp["currency_split"]["multi_currency"]
    assert mc["before_latest_change"] == {"tasks": 8, "flagged": 0}
    assert mc["after_latest_change"] == {"tasks": 17, "flagged": 9}
    assert imp["currency_split"]["single_currency"]["after_latest_change"]["flagged"] == 0
