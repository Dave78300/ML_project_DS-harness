"""Run the no-file rules of an agent file on the blind audit sets, without the model.

Usage:  python audit/run_nofile_audit.py agent/agent_v10.py
        python audit/run_nofile_audit.py agent/experiments/agent_v10_guard.py

For each set it prints how many variants the rules answer correctly, answer wrongly, or leave to the LLM. With the
guarded agent it also prints how many rule answers the guard flags (the rule then votes instead of deciding).
"""
import collections, contextlib, importlib.util, io, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nofile_audit_1, nofile_audit_2  # noqa: E402

spec = importlib.util.spec_from_file_location("agent_under_audit", sys.argv[1])
A = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(A)
guarded = hasattr(A, "calc_rules_guarded")

for name, tasks in (("audit 1", nofile_audit_1.T), ("audit 2", nofile_audit_2.T)):
    count = collections.Counter()
    for t in tasks:
        obj = t["objective"]
        if A._THOUSANDS_RE.search(obj):
            obj = A._norm_thousands(obj)
        try:
            value, flagged = A.calc_rules_guarded(obj) if guarded else (A.calc_rules(obj), False)
        except Exception:
            value, flagged = None, False
        if value is None or not math.isfinite(value):
            count["left to the LLM"] += 1
            continue
        nd = t["nd"]
        right = abs(round(value, nd) - round(t["answer"], nd)) <= 10 ** -nd + 1e-9
        count[("right" if right else "WRONG") + (", flagged by the guard" if flagged else "")] += 1
    print(name, f"({len(tasks)} variants):", dict(sorted(count.items())))
