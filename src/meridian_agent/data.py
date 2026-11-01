"""Read-only access layer over the seed dataset. Nothing here ever writes to the data."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _clean(rec: dict) -> dict:
    return {k: (None if (v is None or (isinstance(v, float) and pd.isna(v))) else v) for k, v in rec.items()}


def _read(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return [_clean(r) for r in df.to_dict("records")]


GAP_RE = re.compile(r"no events captured between (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) and (\d{4}-\d{2}-\d{2} \d{2}:\d{2})")


class Dataset:
    def __init__(self, data_dir: str | Path):
        d = Path(data_dir)
        self.dir = d
        self.tx = _read(d / "transactions.csv")
        self.tasks = _read(d / "task_records.csv")
        self.log = _read(d / "automated_process_log.csv")
        self.procs = _read(d / "procedures.csv")
        self.ev = _read(d / "evaluation_system_output.csv")
        self.emp = _read(d / "employees.csv")
        docs_dir = d / "documents"
        self.docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(docs_dir.glob("*.md"))} if docs_dir.exists() else {}

        for t in self.tx:
            t["amount"] = float(t["amount"]) if t["amount"] else None
        for e in self.ev:
            e["discrepancy_amount"] = float(e["discrepancy_amount"]) if e["discrepancy_amount"] else None

        self._tx = {t["txn_id"]: t for t in self.tx}
        self._task = {t["txn_id"]: t for t in self.tasks}
        self._ev = {e["txn_id"]: e for e in self.ev}
        self._emp = {e["employee_id"]: e for e in self.emp}
        self._log_by_id = {r["log_id"]: r for r in self.log}
        self._log_by_txn: dict[str, list[dict]] = {}
        for r in self.log:
            if r["txn_id"]:
                self._log_by_txn.setdefault(r["txn_id"], []).append(r)

    # ---- lookups ----------------------------------------------------------
    def txn(self, tid): return self._tx.get(tid)
    def task(self, tid): return self._task.get(tid)
    def eval_row(self, tid): return self._ev.get(tid)
    def employee(self, eid): return self._emp.get(eid)
    def events_for_txn(self, tid): return list(self._log_by_txn.get(tid, []))
    def flagged_ids(self): return [e["txn_id"] for e in self.ev]

    def events_window(self, date_from=None, date_to=None, process=None, severity=None, txn_id=None):
        out = []
        for r in self.log:
            day = r["timestamp"][:10]
            if date_from and day < date_from: continue
            if date_to and day > date_to: continue
            if process and r["process"] != process: continue
            if severity and r["severity"] != severity: continue
            if txn_id and r["txn_id"] != txn_id: continue
            out.append(r)
        return out

    # ---- procedures -------------------------------------------------------
    def procedure_versions(self, procedure_id):
        return sorted([p for p in self.procs if p["procedure_id"] == procedure_id], key=lambda p: p["effective_date"])

    def procedure_in_force(self, procedure_id, date):
        vs = [p for p in self.procedure_versions(procedure_id) if p["effective_date"] <= date]
        return vs[-1] if vs else None

    # ---- log coverage -----------------------------------------------------
    def log_gaps(self):
        gaps = []
        for r in self.log:
            m = GAP_RE.search(r["message"] or "")
            if m:
                suspended = [x["log_id"] for x in self.log
                             if "event collection suspended" in (x["message"] or "") and x["timestamp"] <= m.group(1)]
                gaps.append({"start": m.group(1), "end": m.group(2),
                             "evidence": [f"gap:{r['log_id']}"] + [f"log:{i}" for i in suspended[-1:]]})
        return gaps

    def gap_for_date(self, date):
        day_start, day_end = f"{date} 00:00", f"{date} 23:59"
        for g in self.log_gaps():
            if g["start"] <= day_end and g["end"] >= day_start:
                return g
        return None

    # ---- evidence ids -----------------------------------------------------
    def evidence_exists(self, eid: str) -> bool:
        if ":" not in eid:
            return False
        kind, ref = eid.split(":", 1)
        if kind in ("log", "gap"): return ref in self._log_by_id
        if kind == "task": return ref in self._task
        if kind == "txn": return ref in self._tx
        if kind == "eval": return ref in self._ev
        if kind == "doc": return ref in self.docs
        if kind == "proc":
            pid, _, ver = ref.partition("@")
            return any(p["procedure_id"] == pid and p["version"] == ver for p in self.procs)
        return False

    # ---- basic stats ------------------------------------------------------
    def dept_counts(self):
        out = {}
        flagged = set(self.flagged_ids())
        for t in self.tx:
            d = out.setdefault(t["department"], {"tasks": 0, "flagged": 0})
            d["tasks"] += 1
            d["flagged"] += t["txn_id"] in flagged
        for d in out.values():
            d["rate"] = d["flagged"] / d["tasks"]
        return out
