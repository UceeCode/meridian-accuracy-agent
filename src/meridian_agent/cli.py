"""Command line: run | ask | verify"""
from __future__ import annotations

import argparse
import json
import sys

from . import config as C


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(C.ROOT / ".env")
    except ImportError:
        pass


def main(argv=None):
    _load_env()
    ap = argparse.ArgumentParser(prog="meridian-agent", description="Meridian & Co. Accuracy Monitoring Agent (Approach A)")
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--out-dir", default=str(C.OUT_DIR))
    ap.add_argument("--debug", action="store_true", help="show full tracebacks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="investigate all flagged errors and write outputs/")
    r.add_argument("--offline", action="store_true", help="no model; deterministic engine only (nothing to install, no key)")
    r.add_argument("--provider", default=C.PROVIDER, choices=["ollama", "anthropic"], help="which LLM runs the investigation")
    r.add_argument("--model", default=None, help="model name for the chosen provider")
    r.add_argument("--max-steps", type=int, default=C.MAX_STEPS)
    r.add_argument("--quiet", action="store_true")
    a = sub.add_parser("ask", help="ask the agent a question (deduction questions are answered by the gate, not a model)")
    a.add_argument("question")
    a.add_argument("--offline", action="store_true")
    a.add_argument("--provider", default=C.PROVIDER, choices=["ollama", "anthropic"])
    a.add_argument("--model", default=None)
    cl = sub.add_parser("check-llm", help="check that the chosen LLM backend is reachable (Ollama running, model installed)")
    cl.add_argument("--provider", default=C.PROVIDER, choices=["ollama", "anthropic"])
    cl.add_argument("--model", default=None)
    sub.add_parser("verify", help="check that headline numbers in the memo are reproduced from the data")
    args = ap.parse_args(argv)
    try:
        _dispatch(args)
    except RuntimeError as e:          # e.g. missing API key
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:             # API/network/auth problems: say what to do, keep the traceback for --debug only
        if getattr(args, "debug", False):
            raise
        print(f"Error talking to the model: {type(e).__name__}: {e}\n"
              "Run `meridian-agent check-llm` to diagnose, or use --offline (no model needed).", file=sys.stderr)
        sys.exit(3)


def _dispatch(args):
    from . import agent
    if args.cmd == "run":
        def on_event(kind, text):
            if not args.quiet:
                print(f"  [{kind}] {text[:200]}", flush=True)
        res = agent.run_assessment(args.data_dir, args.out_dir, offline=args.offline, provider=args.provider, model=args.model, max_steps=args.max_steps, on_event=on_event)
        s = res["summary"]
        print(f"\nMode: {res['mode']}  |  flagged: {s['total_flagged']}  |  categories: {s['new_category_counts']}")
        print(f"Escalations: {len(res['escalations'])} ({sum(e['severity'] == 'HIGH' for e in res['escalations'])} HIGH)  |  safeguard events: {len(res['safeguard_events'])}")
        print(f"Wrote {args.out_dir}/report.md, findings.json, attribution_table.csv, audit_log.jsonl")
    elif args.cmd == "ask":
        out = agent.answer_question(args.question, args.data_dir, provider=args.provider, model=args.model, offline=args.offline)
        if out["gate"]:
            b = out["briefing"]
            print("[DEDUCTION GATE]\n" + b["agent_position"] + "\n")
            print("Firm error rate by scenario:")
            for k, v in b["firm_rates_by_scenario"].items():
                print(f"  {k:42s} {v['rate']:.1%}  ({v['errors']} errors)  {'above' if v['above_threshold'] else 'below'} 15%")
            print("\nPreconditions before management could even consider it:")
            for p in b["preconditions_before_management_could_consider_it"]:
                print(f"  [{'x' if p['met'] else ' '}] {p['condition']}\n        -> {p['why']}")
        else:
            print(out["answer"])
    elif args.cmd == "check-llm":
        from .llm import OllamaBackend, make_backend
        b = make_backend(args.provider, args.model)
        print(b.check() if isinstance(b, OllamaBackend) else f"OK    Anthropic client created for {b.label} (key present; not tested with a call).")
    elif args.cmd == "verify":
        sys.exit(_verify(args.data_dir))


def _verify(data_dir) -> int:
    from .attribution import attribute_all
    from .claims import claims_ledger
    from .data import Dataset
    ds = Dataset(data_dir)
    v = attribute_all(ds)
    led = {c["id"]: c for c in claims_ledger(ds, v)}
    checks = [("400 tasks, 76 flagged", (len(ds.tx), len(ds.ev)) == (400, 76)),
              ("Memo rates reproduce (M1, M2, M3)", all(led[k]["status"] == "SUPPORTED" for k in ("M1", "M2", "M3"))),
              ("Old system blamed employees by default for 72", sum(e["attribution_basis"].startswith("assigned") for e in ds.ev) == 72),
              ("Memo 'procedure not a factor' is contradicted (M4)", led["M4"]["status"] == "CONTRADICTED"),
              ("IT 'no downstream impact' is contradicted (I1)", led["I1"]["status"] == "CONTRADICTED"),
              ("14 SYSTEM, 8 PROCEDURE", (sum(x.category == "SYSTEM" for x in v.values()), sum(x.category == "PROCEDURE" for x in v.values())) == (14, 8)),
              ("Every verdict cites evidence that exists", all(ds.evidence_exists(e) for x in v.values() for e in x.evidence))]
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    main()
