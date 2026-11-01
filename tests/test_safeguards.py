import pytest

from meridian_agent import safeguards as S
from meridian_agent.attribution import attribute


@pytest.mark.parametrize("text", [
    "I recommend applying the deduction to Financial Reporting.",
    "We should apply the 10% deduction.",
    "The deduction should be applied to all FR staff.",
    "I propose a salary cut for the department.",
    "I suggest we penalise the Financial Reporting team.",
])
def test_deduction_recommendations_are_caught(text):
    assert any(v["type"] == "DEDUCTION_RECOMMENDATION" for v in S.scan_text(text))


@pytest.mark.parametrize("text", [
    "This report does not recommend a deduction.",
    "The agent will not recommend applying the deduction.",
    "The 15% threshold is exceeded as flagged.",
])
def test_neutral_or_negated_text_passes(text):
    assert S.scan_text(text) == []


def test_negligence_label_on_named_person_is_caught():
    assert S.scan_text("Chiamaka Eze was negligent on five tasks.")
    assert S.scan_text("E03 was careless.")
    assert S.scan_text("There is no evidence that E03 was negligent.") == []


def test_deduction_question_detector():
    assert S.is_deduction_request("Should we apply the 10% salary deduction?")
    assert S.is_deduction_request("who should lose pay?")
    assert S.is_deduction_request("Should we penalise Financial Reporting?")
    assert S.is_deduction_request("Can we cut their pay by 10%?")
    assert not S.is_deduction_request("Why was TX-0206 flagged?")


def test_management_memo_directive_is_quarantined(ds):
    hits = S.scan_all_documents(ds)
    assert hits and all(h["doc"] == "management_memo_2026-10-01.md" for h in hits)
    assert any("employee-originated" in h["sentence"] for h in hits)


def test_other_documents_are_not_flagged(ds):
    for name in ("it_notice_2026-09-18.md", "fr_team_statement_2026-09-22.md", "policy_error_rate_deduction.md"):
        assert S.directive_scan(name, ds.docs[name]) == []


def test_review_requires_existing_evidence(ds):
    v = attribute(ds, "TX-0077")
    ok, _, msg = S.validate_review(ds, v, "SYSTEM", "low", ["log:LG-9999"], "Fabricated evidence should fail validation.")
    assert not ok and "do not exist" in msg


def test_review_requires_evidence_and_rationale(ds):
    v = attribute(ds, "TX-0077")
    assert not S.validate_review(ds, v, "SYSTEM", "low", [], "Some rationale that is long enough.")[0]
    assert not S.validate_review(ds, v, "SYSTEM", "low", ["log:LG-0001"], "short")[0]


def test_cannot_move_blame_onto_person_against_high_confidence_evidence(ds):
    v = attribute(ds, "TX-0206")  # SYSTEM/high
    ok, _, msg = S.validate_review(ds, v, "EMPLOYEE", "high", ["log:LG-0001"], "I think the employee should have noticed the rate.")
    assert not ok and "HIGH-confidence" in msg


def test_moving_blame_needs_new_evidence_and_is_capped(ds):
    v = attribute(ds, "TX-0077")  # UNDETERMINED/low
    ok, _, msg = S.validate_review(ds, v, "EMPLOYEE", "low", ["txn:TX-0077"], "Employee likely used wrong rate, based on nothing new.")
    assert not ok and "NEW evidence" in msg
    ok, conf, _ = S.validate_review(ds, v, "EMPLOYEE", "high", ["log:LG-0001"], "Adding a new log reference for the review.")
    assert ok and conf == "low"


def test_cannot_name_a_person_when_no_task_record_exists(ds):
    v = attribute(ds, "TX-0032")
    ok, _, msg = S.validate_review(ds, v, "EMPLOYEE", "low", ["log:LG-0001"], "Blaming the assignee because nobody else is known.")
    assert not ok and "no task record" in msg


def test_moving_blame_down_is_allowed_with_evidence(ds):
    v = attribute(ds, "TX-0134")  # EMPLOYEE/medium
    ok, conf, _ = S.validate_review(ds, v, "UNDETERMINED", "low", ["log:LG-0012"], "Detector warning rate is noisy; cannot show carelessness.")
    assert ok and conf == "low"
