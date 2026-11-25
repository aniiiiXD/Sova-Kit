"""Render a self-contained HTML migration report."""

from __future__ import annotations

import html
import json

from switchit.evaluation import VERDICTS


def render_html(report: dict) -> str:
    esc = html.escape

    def verdict_class(verdict):
        return {
            "pass": "v-pass", "fail": "v-fail", "error": "v-error", "uncertain": "v-uncertain",
        }.get(verdict, "v-uncertain")

    verdicts = report.get("verdicts") or {}
    total_cases = sum(verdicts.get(v, 0) for v in VERDICTS)
    usage = report.get("usage") or {}

    def usage_rows(side):
        rows = []
        data = (usage.get(side) or {})
        for basis in ("recorded", "estimated", "unknown"):
            bucket = data.get(basis) or {}
            rows.append(
                f"<tr><td>{esc(basis)}</td><td>{bucket.get('calls', 0)}</td>"
                f"<td>{bucket.get('input_tokens_known', 0):,}</td>"
                f"<td>{bucket.get('output_tokens_known', 0):,}</td>"
                f"<td>{bucket.get('calls_missing_either', 0)}</td></tr>"
            )
        rows.append(
            f"<tr class='muted'><td>requests without any usage</td>"
            f"<td colspan='4'>{data.get('requests_without_usage', 0)}</td></tr>"
        )
        return "".join(rows)

    case_cards = []
    for case in report.get("cases") or []:
        checks = []
        for check in case.get("checks_target") or []:
            status_class = {"pass": "c-pass", "fail": "c-fail", "unknown": "c-unknown"}.get(check.get("status"), "c-unknown")
            checks.append(
                f"<li class='{status_class}'><code>{esc(str(check.get('type')))}</code> "
                f"{'required' if check.get('required') else 'optional'} — {esc(check.get('status', ''))}: "
                f"{esc(check.get('detail', ''))}</li>"
            )
        tools = []
        for finding in case.get("tool_findings") or []:
            tools.append(f"<li>{esc(finding.get('kind', ''))}: {esc(finding.get('detail', ''))}</li>")
        uncertainties = "".join(f"<li>{esc(note)}</li>" for note in case.get("uncertainty") or [])
        case_notes = "".join(f"<li>{esc(note)}</li>" for note in case.get("notes") or [])
        chain_info = ""
        if case.get("chain"):
            chain_meta = case["chain"]
            chain_info = (
                f"<p class='muted'>chained replay: {len(chain_meta.get('turns') or [])} turn(s), "
                f"fixtures used: {esc(', '.join(chain_meta.get('fixtures_used') or []) or 'none')}"
                + (f", missing: {esc(', '.join(chain_meta['fixtures_missing']))}" if chain_meta.get("fixtures_missing") else "")
                + "</p>"
            )
        errors = ""
        if case.get("target_error"):
            errors = f"<p class='error-text'>target error: {esc(case['target_error'])}</p>"
        request_lines = "".join(
            f"<li><strong>{esc(message.get('role', ''))}</strong>: {esc(message.get('content', ''))}</li>"
            for message in case.get("request_messages") or []
        )
        case_cards.append(
            f"<details class='case {verdict_class(case.get('verdict', ''))}'>"
            f"<summary><span class='verdict {verdict_class(case.get('verdict', ''))}'>"
            f"{esc(case.get('verdict', ''))}</span> {esc(case.get('case_id', ''))} "
            f"<span class='muted'>— workflow {esc(str(case.get('workflow_id')))}, "
            f"trace {esc(str(case.get('trace_id')))}, origins: {esc(', '.join(case.get('origins') or []))}</span></summary>"
            f"<div class='case-body'>"
            f"<h4>Request (redacted)</h4><ul class='messages'>{request_lines}</ul>"
            f"{errors}"
            f"<h4>Checks (target)</h4><ul class='checks'>{''.join(checks) or '<li>none</li>'}</ul>"
            f"<h4>Tool findings</h4><ul class='tools'>{''.join(tools) or '<li>none</li>'}</ul>"
            f"<h4>Uncertainty</h4><ul class='uncertainty'>{uncertainties or '<li>none</li>'}</ul>"
            f"<h4>Comparison notes</h4><ul class='notes'>{case_notes or '<li>none</li>'}</ul>"
            f"{chain_info}"
            f"<h4>Excerpts</h4>"
            f"<p><strong>baseline:</strong> {esc(case.get('baseline_excerpt') or '(empty)')}</p>"
            f"<p><strong>target:</strong> {esc(case.get('target_excerpt') or '(empty)')}</p>"
            f"<p class='muted'>latency baseline {case.get('baseline_latency_ms')} ms vs target "
            f"{case.get('target_latency_ms')} ms</p>"
            f"</div></details>"
        )

    fixes = []
    for fix in report.get("prioritized_fixes") or []:
        fixes.append(
            f"<li><strong>P{fix.get('priority')} — {esc(fix.get('title', ''))}</strong><br>"
            f"{esc(fix.get('suggestion', ''))}<br>"
            f"<span class='muted'>cases: {esc(', '.join(fix.get('case_ids') or []) or 'n/a')}</span></li>"
        )

    groups = []
    for group in report.get("failure_groups") or []:
        groups.append(
            f"<li><code>{esc(group.get('key', ''))}</code> × {group.get('count', 0)} "
            f"<span class='muted'>({esc(', '.join(group.get('case_ids') or []))})</span></li>"
        )

    gaps = report.get("capability_gaps") or []
    gaps_html = (
        "".join(f"<li><strong>{esc(gap.get('capability', ''))}</strong> — {esc(gap.get('detail', ''))}</li>" for gap in gaps)
        or "<li>none flagged</li>"
    )

    boundaries = "".join(f"<li>{esc(boundary)}</li>" for boundary in report.get("boundaries") or [])
    notes = "".join(f"<li>{esc(note)}</li>" for note in report.get("notes") or [])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SwitchIt! migration report</title>
<style>
  body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 2rem auto; max-width: 1080px;
         padding: 0 1.5rem; color: #333; background: #fafafa; line-height: 1.55; }}
  h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.2rem; margin-top: 2.2rem; }} h4 {{ margin: 0.6rem 0 0.2rem; }}
  .muted {{ color: #777; }} code {{ background: #eee; padding: 0 4px; border-radius: 3px; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
  .card {{ border: 2px solid #999; border-radius: 8px; padding: 0.7rem 1.1rem; background: #fff; min-width: 110px; }}
  .card .n {{ font-size: 1.5rem; font-weight: 700; }}
  .v-pass {{ color: #1e7d3c; }} .v-fail {{ color: #b3541e; }} .v-error {{ color: #b03030; }} .v-uncertain {{ color: #8a6d00; }}
  .c-pass {{ color: #1e7d3c; }} .c-fail {{ color: #b03030; }} .c-unknown {{ color: #8a6d00; }}
  table {{ border-collapse: collapse; margin: 0.5rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; font-size: 0.85rem; }}
  details.case {{ border: 1px solid #bbb; border-radius: 6px; padding: 0.5rem 0.9rem; margin: 0.5rem 0; background: #fff; }}
  summary {{ cursor: pointer; font-weight: 600; }}
  .verdict {{ border: 1px solid currentColor; border-radius: 4px; padding: 0 6px; font-size: 0.8rem; }}
  ul {{ padding-left: 1.2rem; }}
  .error-text {{ color: #b03030; }}
  footer {{ margin-top: 3rem; border-top: 1px solid #ccc; padding-top: 1rem; }}
</style>
</head>
<body>
<h1>SWITCHIT! / MIGRATION REPORT</h1>
<p class="muted">generated {esc(report.get('generated_at', ''))} · runner {esc(str(report.get('runner_version')))}
 · fingerprint {esc(str(report.get('fingerprint'))[:16])}…</p>
<p class="muted">baseline: {esc(json.dumps(report.get('baseline_spec') or {}))}<br>
target: {esc(json.dumps(report.get('target_spec') or {}))}</p>

<h2>Summary</h2>
<div class="cards">
  <div class="card"><div class="n">{total_cases}</div>cases</div>
  <div class="card"><div class="n v-pass">{verdicts.get('pass', 0)}</div>pass</div>
  <div class="card"><div class="n v-fail">{verdicts.get('fail', 0)}</div>fail</div>
  <div class="card"><div class="n v-error">{verdicts.get('error', 0)}</div>error</div>
  <div class="card"><div class="n v-uncertain">{verdicts.get('uncertain', 0)}</div>uncertain</div>
</div>

<h2>Capability gaps</h2>
<ul>{gaps_html}</ul>

<h2>Failure groups</h2>
<ul>{''.join(groups) or '<li>none</li>'}</ul>

<h2>Prioritized fixes</h2>
<ol>{''.join(fixes) or '<li>none — every case passed its checks</li>'}</ol>

<h2>Usage by provenance</h2>
<h4>Baseline ({esc(str((report.get('baseline_spec') or {}).get('kind')))})</h4>
<table><tr><th>basis</th><th>calls</th><th>input tokens</th><th>output tokens</th><th>missing either</th></tr>
{usage_rows('baseline')}</table>
<h4>Target ({esc(str((report.get('target_spec') or {}).get('kind')))})</h4>
<table><tr><th>basis</th><th>calls</th><th>input tokens</th><th>output tokens</th><th>missing either</th></tr>
{usage_rows('target')}</table>

<h2>Cases</h2>
{''.join(case_cards) or '<p>no cases</p>'}

<footer>
<h2>Boundaries</h2>
<ul>{boundaries}</ul>
<h2>Notes</h2>
<ul>{notes}</ul>
</footer>
</body>
</html>
"""
