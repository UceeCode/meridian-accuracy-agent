"""Re-derive every headline number from the raw CSVs with plain pandas (independent of the agent's code).
Run:  python scripts/explore_findings.py
Use this to understand the data yourself before defending the agent."""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"
tx, tr = pd.read_csv(D / "transactions.csv"), pd.read_csv(D / "task_records.csv")
lg, pr = pd.read_csv(D / "automated_process_log.csv"), pd.read_csv(D / "procedures.csv")
ev = pd.read_csv(D / "evaluation_system_output.csv")

m = tx.merge(tr, on="txn_id", how="left").merge(ev, on="txn_id", how="left")
m["flag"] = m.error_category.notna()

print("1. Error rates (memo claims 19.0% / FR 24.4% / AR 19.2% / AP 16.3%)")
print(m.groupby("department").flag.agg(flagged="sum", tasks="count", rate="mean").round(3), f"\nFirm: {m.flag.sum()}/{len(m)} = {m.flag.mean():.1%}\n")

print("2. How the OLD system attributed errors")
print(ev.attribution_basis.value_counts().to_string(), "\n")

print("3. Process log: severity by process")
print(lg.groupby(["process", "severity"]).size().to_string(), "\n")

print("4. The FX failure on 17 Sep - transactions posted on a stale cached rate")
c = lg[lg.message.str.contains("cached FX", na=False)].txn_id
print(m[m.txn_id.isin(c)][["txn_id", "currency", "error_category", "attributed_to"]].to_string(index=False), "\n")

print("5. Service-account overrides vs. WHO actually did the work (task record) vs. who was blamed")
o = lg[lg.message.str.contains("SVC-POSTING", na=False)].txn_id
print(m[m.txn_id.isin(o)][["txn_id", "assigned_employee", "employee_id", "error_category", "attributed_to"]].rename(columns={"employee_id": "performed_by"}).to_string(index=False), "\n")

print("6. FR consolidation: multi-currency vs single-currency, before/after PROC-FR-07 v3 (14 Sep)")
c = m[m.task_type == "consolidation"].copy()
c["multi_ccy"], c["after_14sep"] = c.currency.str.contains(r"\+"), c.date >= "2026-09-14"
print(c.groupby(["multi_ccy", "after_14sep"]).agg(tasks=("txn_id", "count"), flagged=("flag", "sum")).to_string(), "\n")
print(pr[pr.procedure_id == "PROC-FR-07"][["version", "effective_date", "approved_by", "change_note"]].to_string(index=False), "\n")

print("7. Missing evidence")
print("transactions with no task record:", sorted(set(tx.txn_id) - set(tr.txn_id)))
print(lg[lg.message.str.contains("collector|disk quota", case=False, na=False)][["timestamp", "message"]].to_string(index=False), "\n")

print("8. Checklist not signed: flagged vs all tasks")
print(f"flagged tasks: {(m[m.flag].checklist_signed == 'N').mean():.1%} | all tasks: {(m.checklist_signed.dropna() == 'N').mean():.1%}")
