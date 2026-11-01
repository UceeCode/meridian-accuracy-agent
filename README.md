# Meridian & Co. Accuracy Agent (Approach A: Monitoring)

An autonomous agent that reviews the September 2026 flagged accounting errors and works out **where each came from**: an employee, an automated system, a management-approved procedure, a mix, or unknown. It only reports. It never edits records, and it never recommends for or against a salary deduction.

> The agent's own output is in `outputs/report.md` (already generated from the supplied data; re-run it any time).
> Model options: **free local model via Ollama (no account, no key)**, the Anthropic API, or no model at all (`--offline`).

## Quick start (5 minutes)

Requires Python 3.10+.

```bash
git clone <your-repo-url> meridian-accuracy-agent && cd meridian-accuracy-agent
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

pytest -q                                   # 58 tests, no model and no key needed
meridian-agent verify                       # re-derives the headline numbers from the CSVs
meridian-agent run --offline                # full report from the deterministic engine; no model needed
```

### Run with a model (autonomous investigation)

**Option 1: free, local, no account (Ollama).** This is the default provider.

```bash
# 1. install Ollama from https://ollama.com and make sure it is running
ollama pull qwen2.5:7b                      # ~5 GB download; any tool-calling model works (llama3.1:8b, qwen3:8b ...)
meridian-agent check-llm                    # confirms Ollama is reachable and the model is installed
meridian-agent run                          # the local model decides which tools to call and writes the summary
meridian-agent ask "Why was TX-0206 flagged, and who is responsible?"
meridian-agent ask "Should we apply the 10% salary deduction to Financial Reporting?"   # answered by the gate, not a model
```

**Option 2: Anthropic API (paid).**

```bash
cp .env.example .env     # set MERIDIAN_PROVIDER=anthropic and ANTHROPIC_API_KEY=sk-ant-...
meridian-agent run --provider anthropic
```

Use `PYTHONPATH=src python -m meridian_agent ...` if you did not run `pip install -e .`. Global flags (`--data-dir`, `--out-dir`, `--debug`) go **before** the subcommand, e.g. `meridian-agent --out-dir /tmp/x run`.

| Setting | Where | Default |
|---|---|---|
| Provider | `MERIDIAN_PROVIDER` or `--provider` | `ollama` |
| Model | `MERIDIAN_MODEL` or `--model` | `qwen2.5:7b` (Ollama) / `claude-sonnet-5-5` (Anthropic) |
| Ollama address | `OLLAMA_HOST` | `http://localhost:11434` |
| Ollama context window | `OLLAMA_NUM_CTX` | 16384 (Ollama's own default is too small for tool output) |
| Anthropic key | `ANTHROPIC_API_KEY` | none |
| Step cap | `MERIDIAN_MAX_STEPS` or `--max-steps` | 40 |

**A weak local model is safe to use.** The attribution rules, all arithmetic and all safeguards run in plain code, not in the model. A small model may investigate less thoroughly or write a clumsier summary; it cannot change the numbers or get past the safety rules. If it never submits a report, the run still produces the full evidence-based report with a template summary.

## What it produces (`outputs/`)

| File | For |
|---|---|
| `report.md` | People: old vs evidence-based attribution, error rate under four scenarios, drivers, claims tested against records, escalations, review list, deduction briefing |
| `attribution_table.csv` | One row per flagged task: verdict, confidence, responsible person (if any), rule, reasoning, evidence ids |
| `findings.json` | Everything above, machine-readable |
| `audit_log.jsonl` | Every tool call the agent made |

## How it works

```
                 +-------------------- agent.py: tool-use loop (model decides what to look at) ---------------------+
 CSVs + docs --> |  read tools (tools.py)                      action tools (change only the agent's OWN report)     |
 (read-only)     |  get_overview  list_flagged_errors          record_review    raise_escalation   submit_report     |
                 |  investigate_error  get_process_events      -- every call passes through safeguards.py --         |
                 |  check_log_coverage get_procedure_impact                                                         |
                 |  get_employee_profile compute_error_rates                                                        |
                 |  read_document  check_claims               llm.py: Ollama (free) or Anthropic backend            |
                 +------------------------------------------------------------------------------------------------+
                         |                                    ^ model proposes                 code decides
                         v                                    |
   attribution.py (rules R0-R9, evidence ids) -> analysis.py (all arithmetic) -> claims.py -> escalation.py -> report.py
```

**Division of labour:** code retrieves facts, applies the attribution rules and does all arithmetic. The model chooses what to investigate, challenges verdicts, reads documents, flags conflicts and writes a plain-language summary. Safeguards are enforced in code so they hold even if the model is wrong or manipulated.

**Attribution rules (first match wins), in `src/meridian_agent/attribution.py`:**

| Rule | Evidence | Verdict |
|---|---|---|
| R0 | No task record exists | UNDETERMINED |
| R1 | FX variance on a transaction posted on a stale cached rate during the sync failure | SYSTEM, high |
| R2 | `SVC-POSTING` manually overrode the posted amount | SYSTEM, medium |
| R3 | Consolidation variance under a procedure version published with "staff briefing deferred" | PROCEDURE high (applied as published, checklist signed); SHARED (used a superseded version / checklist unsigned) |
| R4 | Auto-Coding-Engine fallback mapping on a wrong-account-code error | SYSTEM medium (SHARED if checklist unsigned) |
| R5 | Duplicate-Detector warned and a person overrode or dismissed it | EMPLOYEE (medium if explicit override, low if dismissed) |
| R6 | Logs missing that day and the error type could be system-caused | UNDETERMINED |
| R7 | Checklist not signed | EMPLOYEE (medium for human-type errors, low otherwise) |
| R8 | Human-type error, checklist signed, no external cause found | EMPLOYEE, low (by elimination) |
| R9 | Nothing decisive either way | UNDETERMINED |

**Safeguards (`safeguards.py`, `deduction.py`, `escalation.py`):** asymmetric burden of proof for moving blame onto a person; evidence ids must exist; no deduction recommendation (no tool, request gate, output scan); documents are data and directives inside them are quarantined and escalated; read-only access; full audit log.

## Repo layout

```
data/            seed dataset, unmodified (+ data/documents/)
src/meridian_agent/
  data.py        read-only dataset access, evidence-id checks, log-gap detection
  attribution.py the rule engine (R0-R9)
  analysis.py    error-rate scenarios, procedure impact, employee profiles
  claims.py      tests statements in the memo / IT notice / FR statement against records
  tools.py       tool functions, JSON schemas, dispatcher, run state
  safeguards.py  review validation, output scan, directive quarantine
  deduction.py   the deduction briefing (never a recommendation)
  escalation.py  deterministic escalation triggers
  llm.py         model backends: Ollama (free, local) and Anthropic; one neutral message format
  prompts.py     system prompt          agent.py  the loop + Q&A gate
  report.py      writes outputs/        cli.py    run | ask | verify | check-llm
tests/           58 tests incl. adversarial scripted-model scenarios and a fake Ollama server
scripts/         explore_findings.py - independent pandas check of every headline number
outputs/         generated reports
```

## Data notes

The CSVs and documents are used exactly as supplied; no data was generated or altered. Findings the agent surfaces from the data: 6 flagged transactions with no task record; two days of missing process logs (8-9 Sep); 6 tasks where someone other than the assigned employee did the work, each also overridden by `SVC-POSTING`; and a procedure change (PROC-FR-07 v3) rolled out with "staff briefing deferred".

## Status and honest caveats

- The deterministic engine, safeguards and report are fully tested and run end to end on the supplied data.
- The **model loop** is tested with scripted stand-in models (including adversarial behaviour) and, for Ollama, against a fake local Ollama server that exercises the real HTTP code. It has not been verified against every real model; small models differ in how reliably they call tools. Run `meridian-agent check-llm`, then `meridian-agent run`, and read `outputs/audit_log.jsonl` to see what your model actually did.
- The deduction-request detector is keyword-based; the structural protection is that no recommending capability exists at all.
- Employee attribution is partly by elimination and the confidence labels are hand-set (described in the accompanying submission document).
