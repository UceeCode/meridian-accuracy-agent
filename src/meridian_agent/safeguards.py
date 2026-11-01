"""Safeguards. These run in CODE, outside the model, so they hold even if the model is wrong, confused or manipulated.

1. Asymmetric burden of proof  - moving blame towards a person needs new, valid evidence and is capped in confidence.
2. Evidence gate               - every verdict/review must cite evidence ids that actually exist in the data.
3. Deduction gate              - the agent never recommends applying (or not applying) a salary deduction.
4. Output scan                 - model-written prose is scanned for deduction recommendations and negligence labels.
5. Directive quarantine        - instructions inside documents that try to steer the agent are flagged, never obeyed.
"""
from __future__ import annotations

import re

from . import config as C
from .attribution import Verdict
from .data import Dataset

# ---------------------------------------------------------------------------------------------------
# 3. Deduction gate
# ---------------------------------------------------------------------------------------------------
_DEDUCTION_Q = re.compile(r"(deduct|salary|pay\s*cut|pay\s*reduction|wages?\b|10\s*%.{0,30}(pay|salary)|penali[sz]|docking|dock\s+(their|the)?\s*pay|"
                          r"lose\s+(their\s+|some\s+)?(pay|salary|wages?)|(reduce|cut|withhold)\s+(their\s+|the\s+)?(pay|salary|wages?|bonus))", re.I)


def is_deduction_request(text: str) -> bool:
    return bool(_DEDUCTION_Q.search(text or ""))


_NEG = re.compile(r"\b(not|no|never|cannot|can't|won't|does not|doesn't|do not|decline[sd]?|refuse[sd]?|without|unable)\b", re.I)
_REC_DEDUCTION = [
    re.compile(r"\b(recommend|propos|suggest|advis|urge|support)\w*\b[^.\n]{0,80}\b(deduct\w*|salary (cut|reduction)|pay (cut|reduction)|penali[sz]\w*|docking)", re.I),
    re.compile(r"\b(apply|impose|proceed with|go ahead with|approve|implement|enforce)\b[^.\n]{0,40}\b(deduction|salary cut|pay cut)", re.I),
    re.compile(r"\bdeduction\b[^.\n]{0,30}\b(should|must|ought to|needs to)\s+(not\s+)?be\s+(applied|imposed|approved|made|enforced|implemented)", re.I),
]
_NAME = re.compile(r"\b(E0\d|Adaeze|Tunde|Chiamaka|Samuel|Ifeoma|Emeka|Halima|Kelechi|Obi|Bello|Eze|Nwosu|Udeh|Chukwu|Sani|Ibe)\b")
_NEGLIGENCE = re.compile(r"\b(negligen\w*|careless\w*|incompeten\w*|lazy|laziness|underperform\w*|poor performer\w*)\b", re.I)


def scan_text(text: str) -> list[dict]:
    """Return violations found in model-written text. A negation within 40 chars before a match clears it."""
    violations = []
    for pat in _REC_DEDUCTION:
        for m in pat.finditer(text or ""):
            window = text[max(0, m.start() - 40):m.start()]
            if _NEG.search(window) or _NEG.search(text[m.start():m.start() + 25]):
                continue
            violations.append({"type": "DEDUCTION_RECOMMENDATION", "excerpt": text[max(0, m.start() - 20):m.end() + 20]})
    for sent in re.split(r"(?<=[.!?])\s+|\n", text or ""):
        m = _NEGLIGENCE.search(sent)
        if m and _NAME.search(sent) and not _NEG.search(sent[:m.start()][-50:]):
            violations.append({"type": "NEGLIGENCE_LABEL_ON_NAMED_PERSON", "excerpt": sent[:160]})
    return violations


# ---------------------------------------------------------------------------------------------------
# 5. Directive quarantine
# ---------------------------------------------------------------------------------------------------
_DIRECTIVES = [
    re.compile(r"configure the (accuracy )?agent", re.I),
    re.compile(r"treat (flagged )?errors as employee[- ]originated", re.I),
    re.compile(r"(report|findings?|output)\b[^.\n]{0,60}\b(should|must) be consistent with this assessment", re.I),
    re.compile(r"ignore (all |any )?(previous|prior|above) (instructions|rules)", re.I),
    re.compile(r"(do not|don't) (mention|report|disclose)", re.I),
    re.compile(r"\bbefore (it|the report) is presented\b", re.I),
]


def directive_scan(doc_name: str, text: str) -> list[dict]:
    hits = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", text or ""):
        for pat in _DIRECTIVES:
            if pat.search(sent):
                hits.append({"doc": doc_name, "sentence": sent.strip()})
                break
    return hits


def scan_all_documents(ds: Dataset) -> list[dict]:
    out = []
    for name, text in ds.docs.items():
        out += directive_scan(name, text)
    return out


# ---------------------------------------------------------------------------------------------------
# 1 + 2. Review validation (model may propose changes to engine verdicts; code decides)
# ---------------------------------------------------------------------------------------------------
def validate_review(ds: Dataset, engine: Verdict, category: str, confidence: str,
                    evidence_ids: list[str], rationale: str) -> tuple[bool, str, str]:
    """Returns (accepted, final_confidence, message)."""
    if category not in C.CATEGORIES:
        return False, confidence, f"category must be one of {C.CATEGORIES}"
    if confidence not in C.CONFIDENCES:
        return False, confidence, f"confidence must be one of {C.CONFIDENCES}"
    if not rationale or len(rationale.strip()) < 15:
        return False, confidence, "A rationale of at least a sentence is required."
    if not evidence_ids:
        return False, confidence, "At least one evidence id is required (e.g. 'log:LG-0023', 'task:TX-0018')."
    bad = [e for e in evidence_ids if not ds.evidence_exists(e)]
    if bad:
        return False, confidence, f"Evidence ids do not exist in the data: {bad}. Fabricated evidence is rejected."

    up = C.BLAME_RANK[category] - C.BLAME_RANK[engine.category]
    if up > 0:  # moving towards blaming a person
        if engine.category in ("SYSTEM", "PROCEDURE") and engine.confidence == "high":
            return False, confidence, ("Rejected: the engine holds HIGH-confidence non-employee evidence for this task. "
                                       "Blame cannot be moved onto a person against high-confidence evidence.")
        new_ev = [e for e in evidence_ids if e not in engine.evidence]
        if not new_ev:
            return False, confidence, ("Rejected: moving blame towards a person requires NEW evidence the engine has not already used. "
                                       "Cite at least one additional evidence id.")
        if category == "EMPLOYEE" and (not engine.performed_by):
            return False, confidence, "Rejected: no task record exists, so no person can be identified as the performer."
        if C.CONF_RANK[confidence] > 1:
            return True, "low", "Accepted, but confidence capped at 'low': blame moved towards a person on thin evidence."
    return True, confidence, "Accepted."


def clamp_final_confidence(engine: Verdict, category: str, confidence: str) -> str:
    """A review can never raise confidence above what the engine's evidence supports for the SAME category."""
    if category == engine.category and C.CONF_RANK[confidence] > C.CONF_RANK[engine.confidence]:
        return engine.confidence
    return confidence
