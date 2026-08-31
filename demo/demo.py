"""Zero-dependency CLI demo.

Usage:
    python demo/demo.py [path/to/events.jsonl]     # print report + plan
    python demo/demo.py --html report.html         # self-contained HTML dashboard
    python demo/demo.py --accept                   # auto-approve and write exports
    python demo/demo.py --history history.jsonl    # compare vs personal baseline
    python demo/demo.py --out outdir               # where exports are written

Runs the full loop — **Granite proposes -> LoadGuard validates -> the human
approves -> LoadGuard acts -> impact is measured**. No third-party packages or
API keys are required (the decision agent and guardian fall back to the
deterministic engine without a model).
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loadguard.actions import export_ics, export_tasks_csv  # noqa: E402
from loadguard.baseline import load_history  # noqa: E402
from loadguard.sample_data import sample_tasks  # noqa: E402
from loadguard.signals import load_events  # noqa: E402
from loadguard.workflow import run_workflow  # noqa: E402

SAMPLE = Path(__file__).resolve().parent / "sample_events.jsonl"

FACTOR_LABELS = {
    "context_switches_per_hour": "Context switches / hour",
    "meeting_ratio": "Meeting density",
    "notification_rate": "Notifications / hour",
    "focus_ratio": "Focus time",
    "multitasking_index": "Multitasking",
}

MODEL_LABEL = {
    "watsonx": "Granite via watsonx",
    "ollama": "Granite via Ollama",
    "heuristic": "Deterministic fallback",
    "deterministic": "Deterministic fallback",
}


def _parse_args(argv: list[str]) -> dict:
    opts: dict = {
        "events": SAMPLE,
        "tasks": None,
        "html": None,
        "accept": False,
        "history": None,
        "out": Path("."),
        "role": None,
    }
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--html" and i + 1 < len(argv):
            opts["html"] = Path(argv[i + 1])
            i += 2
        elif arg == "--accept":
            opts["accept"] = True
            i += 1
        elif arg == "--tasks" and i + 1 < len(argv):
            opts["tasks"] = Path(argv[i + 1])
            i += 2
        elif arg == "--history" and i + 1 < len(argv):
            opts["history"] = Path(argv[i + 1])
            i += 2
        elif arg == "--out" and i + 1 < len(argv):
            opts["out"] = Path(argv[i + 1])
            i += 2
        elif arg == "--role" and i + 1 < len(argv):
            opts["role"] = argv[i + 1]
            i += 2
        else:
            opts["events"] = Path(arg)
            i += 1
    return opts


def _guard_summary(result) -> str:
    g = result.guardian
    if g is None:
        return "guard not run"
    return f"{g.summary()}{' (narrative sanitized)' if g.sanitized else ''}"


def _proposal_summary(result) -> str:
    if result.proposal is None:
        return "deterministic plan (no LLM proposal or proposal rejected by gate)"
    p = result.proposal
    parts = []
    if p.priority_task_id:
        parts.append(f"prioritize {p.priority_task_id}")
    if p.delegate_task_ids:
        parts.append(f"delegate {len(p.delegate_task_ids)} task(s)")
    if p.inserts:
        parts.append(f"insert {len(p.inserts)} block(s)")
    return "; ".join(parts) or "no adjustments"


def _print_report(result, role: str | None = None) -> None:
    report = result.load_report
    print("=" * 64)
    print(" LoadGuard — Cognitive-Load-Aware AI Co-Worker")
    print("=" * 64)
    if role:
        print(f" Role profile: {role.title()}")
    print(f" Score: {report.score:.0f}/100  [{report.level.upper()}]")
    print(f" {report.explanation}")
    if result.trend:
        print(f" Trend: {result.trend.summary}")
    print()
    print(" Signals (proxies):")
    for name, label in FACTOR_LABELS.items():
        print(f"   - {label:28s} {report.factors[name]}")
    print()
    print(
        f" Granite Decision Agent: {MODEL_LABEL.get(result.plan.proposed_by, result.plan.proposed_by)}"
    )
    print(f"   proposal: {_proposal_summary(result)}")
    print(f" Safety gate ({_guard_summary(result)})")
    print()
    print(" Proposed plan:")
    for item in result.plan.items:
        print(f"   {item.position:2d}. {item.action.upper():11s} {item.title}")
        if item.rationale:
            print(f"       -> {item.rationale}")
    print()
    print(
        f" Note ({MODEL_LABEL.get(result.plan.generated_by, result.plan.generated_by)}): {result.plan.note}"
    )
    print()
    imp = result.impact
    print(
        f" Impact (projected): {imp.before_score:.0f} -> {imp.after_score:.0f} "
        f"(reduction of {imp.delta:.0f} points, level {imp.before_level} -> {imp.after_level})"
    )
    print(f" Plan status: {result.plan.status}  (plan id: {result.plan.plan_id})")
    print(f" Disclaimer: {report.disclaimer}")


def _render_html(result) -> str:
    report = result.load_report
    imp = result.impact

    def gauge(score: float) -> str:
        color = "#34d399" if score < 50 else "#fbbf24" if score < 75 else "#f87171"
        return (
            f'<div style="width:{max(2, score)}%;background:{color};height:100%;'
            f'border-radius:8px;transition:width .4s"></div>'
        )

    factor_rows = "".join(
        f"<tr><td>{html.escape(FACTOR_LABELS[k])}</td><td>{report.factors[k]}</td></tr>"
        for k in FACTOR_LABELS
    )

    item_rows = "".join(
        f'<li><span class="badge {html.escape(i.action)}">{html.escape(i.action)}</span>'
        f'<b>{html.escape(i.title)}</b><div class="why">{html.escape(i.rationale)}</div></li>'
        for i in result.plan.items
    )

    trend_html = ""
    if result.trend:
        trend_html = f'<div class="trend">{html.escape(result.trend.summary)}</div>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>LoadGuard — Cognitive Load Report</title>
<style>
:root{{color-scheme:dark}}
body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0b1020;color:#e5e7eb;
margin:0;padding:32px}}
.wrap{{max-width:860px;margin:0 auto}}
h1{{margin:0 0 4px;font-size:26px}}
.sub{{color:#9ca3af;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:640px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:20px}}
.card h2{{margin:0 0 12px;font-size:16px;color:#d1d5db}}
.big{{font-size:44px;font-weight:700}}
.bar{{background:#1f2937;border-radius:8px;height:14px;overflow:hidden;margin:8px 0}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
td{{padding:6px 0;border-bottom:1px solid #1f2937}}
td:last-child{{text-align:right;font-variant-numeric:tabular-nums}}
ul{{list-style:none;padding:0;margin:0}}
li{{padding:10px 0;border-bottom:1px solid #1f2937}}
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
text-transform:uppercase;margin-right:8px;background:#374151;color:#e5e7eb}}
.badge.do{{background:#065f46}}.badge.delegate{{background:#7c2d12}}
.badge.focus_block{{background:#1e3a8a}}.badge.break{{background:#4c1d95}}
.badge.batch{{background:#0e7490}}
.why{{color:#9ca3af;font-size:13px;margin-top:2px}}
.level{{text-transform:uppercase;letter-spacing:.08em;font-size:13px}}
.delta{{font-size:14px;color:#34d399}}
.trend{{font-size:13px;color:#93c5fd;margin-top:8px}}
.note{{font-size:15px;line-height:1.5}}
.meta{{font-size:12px;color:#9ca3af;margin-bottom:4px}}
.disc{{color:#6b7280;font-size:12px;margin-top:20px}}
</style></head><body><div class="wrap">
<h1>🧠 LoadGuard — Cognitive Load Report</h1>
<div class="sub">Cognitive-Load-Aware AI Co-Worker · IBM AI Builders Challenge 2026</div>
<div class="meta">Decision: {html.escape(MODEL_LABEL.get(result.plan.proposed_by, result.plan.proposed_by))} ·
Guardian: {html.escape(result.guardian.engine if result.guardian else "n/a")} ·
Narrator: {html.escape(MODEL_LABEL.get(result.plan.generated_by, result.plan.generated_by))} ·
Status: {html.escape(result.plan.status)}</div>
{trend_html}
<div class="grid">
  <div class="card">
    <h2>Before</h2>
    <div class="big">{imp.before_score:.0f}<span style="font-size:20px">/100</span></div>
    <div class="level">{imp.before_level}</div>
    <div class="bar">{gauge(imp.before_score)}</div>
  </div>
  <div class="card">
    <h2>After following the plan</h2>
    <div class="big">{imp.after_score:.0f}<span style="font-size:20px">/100</span></div>
    <div class="level">{imp.after_level}</div>
    <div class="bar">{gauge(imp.after_score)}</div>
    <div class="delta">↓ {imp.delta:.0f} points projected</div>
  </div>
</div>
<div class="grid" style="margin-top:20px">
  <div class="card"><h2>Signals (proxies)</h2><table>{factor_rows}</table></div>
  <div class="card"><h2>Proposed plan</h2><ul>{item_rows}</ul></div>
</div>
<div class="card" style="margin-top:20px"><h2>Note</h2>
<div class="note">{html.escape(result.plan.note)}</div></div>
<div class="disc">{html.escape(report.disclaimer)}</div>
</div></body></html>"""


def _load_tasks(path: Path) -> list:
    import json
    from loadguard.models import Task

    tasks: list[Task] = []
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
        for item in data:
            tasks.append(Task(**item))
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            tasks.append(Task(**item))
    return tasks


def main() -> None:
    opts = _parse_args(sys.argv)
    events = load_events(opts["events"])
    history = load_history(opts["history"]) if opts["history"] else None
    approval = "accepted" if opts["accept"] else None
    tasks = _load_tasks(opts["tasks"]) if opts.get("tasks") else sample_tasks()

    result = run_workflow(
        events,
        tasks,
        history=history,
        approval=approval,
        role=opts.get("role"),
    )

    _print_report(result, role=opts.get("role"))

    if result.plan.status == "accepted":
        out = opts["out"]
        out.mkdir(parents=True, exist_ok=True)
        ics_path = out / f"loadguard-{result.plan.plan_id}.ics"
        csv_path = out / f"loadguard-{result.plan.plan_id}.csv"
        ics_path.write_text(
            export_ics(result.plan, tasks, existing_events=events), encoding="utf-8"
        )
        csv_path.write_text(export_tasks_csv(result.plan, tasks), encoding="utf-8")
        print(f"\n ✅ Plan accepted — protected calendar written to {ics_path}")
        print(f"    Resequenced task list written to {csv_path}")
    else:
        print("\n Plan is pending human approval. Re-run with --accept to apply and export.")

    if opts["html"]:
        opts["html"].write_text(_render_html(result), encoding="utf-8")
        print(f" HTML dashboard written to {opts['html']}")


if __name__ == "__main__":
    main()
