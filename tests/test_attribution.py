import pytest

from meridian_agent.attribution import attribute, attribute_all


@pytest.mark.parametrize("tid,cat,conf,rule", [
    ("TX-0206", "SYSTEM", "high", "R1_STALE_FX_CACHE"),                          # FX outage, cached rate
    ("TX-0018", "SYSTEM", "medium", "R2_SERVICE_ACCOUNT_OVERRIDE"),              # service account overrode amount
    ("TX-0169", "PROCEDURE", "high", "R3_UNBRIEFED_PROCEDURE_APPLIED_AS_PUBLISHED"),
    ("TX-0291", "SHARED", "medium", "R3_SUPERSEDED_VERSION_USED"),
    ("TX-0032", "UNDETERMINED", "low", "R0_NO_TASK_RECORD"),
    ("TX-0308", "UNDETERMINED", "low", "R0_NO_TASK_RECORD"),
    ("TX-0077", "UNDETERMINED", "low", "R6_LOG_GAP_SYSTEM_PLAUSIBLE"),            # 8 Sep, logs missing
    ("TX-0134", "EMPLOYEE", "medium", "R5_WARNING_OVERRIDDEN"),
])
def test_known_cases(ds, tid, cat, conf, rule):
    v = attribute(ds, tid)
    assert (v.category, v.confidence, v.rule) == (cat, conf, rule)


def test_misattributed_when_someone_else_did_the_work(ds):
    v = attribute(ds, "TX-0018")
    assert v.assigned_employee == "E04" and v.performed_by == "E06"
    assert v.old_attribution == "E04" and v.responsible_employee is None
    assert any("performed the work" in c for c in v.caveats)


def test_nobody_is_blamed_without_a_task_record(ds):
    for v in attribute_all(ds).values():
        if v.performed_by is None:
            assert v.category == "UNDETERMINED" and v.responsible_employee is None


def test_system_and_procedure_verdicts_never_name_an_employee(ds):
    for v in attribute_all(ds).values():
        if v.category in ("SYSTEM", "PROCEDURE", "UNDETERMINED"):
            assert v.responsible_employee is None


def test_every_verdict_cites_evidence_that_exists(ds):
    for v in attribute_all(ds).values():
        assert v.evidence and all(ds.evidence_exists(e) for e in v.evidence)


def test_low_confidence_employee_is_never_high(ds):
    for v in attribute_all(ds).values():
        if v.category == "EMPLOYEE":
            assert v.confidence in ("low", "medium")  # the engine never has 'high' proof of employee fault


def test_totals_add_up(ds):
    v = attribute_all(ds)
    assert len(v) == 76
    assert sum(1 for x in v.values() if x.category == "SYSTEM") == 14
    assert sum(1 for x in v.values() if x.category == "PROCEDURE") == 8
