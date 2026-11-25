---
name: switchit
description: Discover agent traces, reconstruct evidence-backed workflows, and verify a move between vendors using fixed checks and reproducible reports. Prepare and verify SwitchUP instruction updates from recorded corrections. Operate authorized orchestration plans and export reviewed workflows as skills.
---

# SwitchIt!

Move a user's workflow from a source agent or provider to a target while preserving its required outcomes. The shared workflow representation is the center: importing Codex, Claude Code, Devin, Grok, or Muse records does not itself establish that the target can execute that workflow.

## Working discipline

1. Inspect the repository, authorized source locations, runtime configuration, and existing success criteria. Reuse authorization and constraints already present in the session or orchestration plan; ask only for missing information that materially affects execution. Do not repeatedly request approval already granted.
2. Discover and import evidence with source IDs, revisions, tool identities, usage provenance, and diagnostics intact. Keep credentials out of artifacts. Keep raw personal trace files separate from public release artifacts; use synthetic or explicitly authorized sanitized fixtures for distribution.
3. Reconstruct the execution graph. Distinguish recorded relationships from inferred order and retain missing/ambiguous links. Metadata-only records are discovery evidence, not executable recordings. Missing tool results must stay missing.
4. Build a draft workflow, inspect its dependencies and branches, and establish explicit outcome checks. Parameterize inputs where recordings contain concrete values. An observed answer is not a success criterion by itself. Review must reflect the user's requirements or the standing plan policy; do not mark a draft reviewed merely to unblock a run.
5. Validate with fixtures first. Use provider or command runtimes only within the authorized destinations, data-transfer scope, tool permissions, and budgets. Verify the target's current documented capabilities before implementing a native adapter; unsupported execution must be reported explicitly.
6. Compare complete outcomes, tool sequence and arguments, artifacts/state where observable, and whole-run usage/latency. Keep pass, fail, error, and uncertain distinct. Never weaken fixed checks or move held-out cases into training to make a candidate pass.
7. Export or publish only within the user's requested release scope. Exporting a skill, running a workflow, and publishing a package/deployment are different operations. Respect an existing no-push/no-publish instruction until the user changes that scope.

## Harness access

Use Python switchit-harness 0.2.0 or later for the workflow commands below. The Python package is the execution engine. The npm `switchit-harness` package is a thin CLI/MCP launcher, not a second implementation; it requires Python 3.10+ and the separately installed Python engine. Set `SWITCHIT_PYTHON` to the intended interpreter when using that launcher. It does not install anything automatically.

For MCP-capable agents, register `switchit-mcp` and inspect its advertised tools. Its stdout is reserved for protocol messages. Use the CLI when a required operation is not exposed by the connected MCP server. The examples below run the Python engine directly from a checkout or installed environment.

## SwitchUP improvement projects

Use `improve status` to inspect the active recipe, generation, and recent rounds. `improve prepare --project PROJECT --output learning.json` exports recorded learning evidence for a pinned round. Use only that packet to author an instruction update. Submit JSON with `system_prompt`, `rationale`, and `evidence_ids` through `improve propose --proposal`. Cite actual event IDs and preserve the reviewed tool scope and outcomes.

`improve verify` runs both versions using the pinned suite and requires explicit authorization for the configured model connection. Read the saved decision and replay evidence with `improve show`. Activate a verified round with `improve activate --expected-generation` only within the user's release scope. An activation is used for subsequent connected requests. See `SwitchUp/docs/switchup.md` in Sova-Kit for project setup, capture, replay, and rollback. MCP tools with the `switchup_` prefix expose the same project state.

## Evidence to reviewed workflow

```sh
# Legacy offline smoke test:
python3 -m switchit.cli run --demo --state-dir /tmp/switchit-demo

# Build from a canonical events JSON artifact:
python3 -m switchit.cli workflow build --events events.json --output .switchit/workflow-build

# Execute a reviewed workflow and one matching case; fixture is the default runtime:
python3 -m switchit.cli workflow run --workflow workflow.json --case case.json --output run.json --runtime fixture

# Export the reviewed workflow as a standalone skill:
python3 -m switchit.cli workflow export-skill --workflow workflow.json --output ./exported-skill
```

Use the artifact paths reported by `workflow build`; inspect the generated draft and diagnostics before execution. A changed workflow revision invalidates cases and recipes pinned to the previous revision. `--runtime provider` and `--runtime command` select additional execution modes; inspect `workflow run --help` for their configuration and authorization flags. Never treat selecting a runtime as permission to transmit private data or execute business tools.

For explicitly scoped local trace roots, inspect `sources inventory --help` and `sources import --help`; inventory metadata before choosing recordings to import. For a reviewed workflow and its pinned evaluation suite, inspect `harness propose`, `harness compare`, `harness promote`, `harness rollback`, and `harness status` with `--help`. Keep recipes tied to the exact workflow revision and fixed checks. Local recipe promotion does not publish a package or authorize live tools.

The older stages remain useful for imported model-call recordings:

```sh
python3 -m switchit.cli import --jsonl traces.jsonl --state-dir .switchit
python3 -m switchit.cli profile --state-dir .switchit
python3 -m switchit.cli map --keep-edits --state-dir .switchit
python3 -m switchit.cli cases --state-dir .switchit
python3 -m switchit.cli replay --state-dir .switchit
python3 -m switchit.cli report --state-dir .switchit
```

After mapping, correct uncertain groups rather than guessing. After case generation, replace default checks with explicit required outcomes. Historical behavior is a reference, not ground truth. Legacy name-only tool fixtures do not prove full workflow fidelity.

## Authorized orchestration plans

```sh
python3 -m switchit.cli orchestrate run --plan plan.json --output .switchit/orchestration
python3 -m switchit.cli orchestrate status --output .switchit/orchestration
python3 -m switchit.cli orchestrate run --plan plan.json --output .switchit/orchestration --resume
python3 -m switchit.cli orchestrate cancel --output .switchit/orchestration
```

Inspect the plan before execution: declared nodes, dependencies, owned paths, evaluator contracts, attempt limits, resource budgets, and release policy must fit the user's scope. A resumable run must use matching plan and artifact revisions. Route failures to the owning module with evidence; stop at exhausted budgets or unresolved requirements instead of looping indefinitely. Coders must not rewrite independent evaluator criteria or publish around a failed release gate.

To inspect and operate the local platform:

```sh
python3 -m switchit.cli serve --host 127.0.0.1 --port 8787 --data-dir .switchit/platform
```

Keep the default loopback binding for local use. Do not represent a local server or static marketing deployment as a production multi-user service.

## Report honestly

Explain the workflow, source evidence, target capabilities, checks performed, observed results, unresolved gaps, and next concrete action. Identify which results came from fixtures and which came from live executions. Usage estimates are not invoices; absent state, cost, or artifact evidence cannot be a passing check. Configuration rollback does not undo external tool effects.

For configured coder/verifier repair, use `orchestrate repair --config REPAIR.json --output RUN_DIR` (add `--resume` for an interrupted run). The bounded loop preserves verifier and acceptance-file digests and feeds failure evidence into the next coder round. Worker commands and source scope must be configured; the harness does not infer native vendor CLI flags.

For GitHub work evidence, use `sources github --file EXPORT.json --output DIR` offline, or `sources github --repo OWNER/REPO --pr NUMBER --remote --output DIR` for explicitly selected API reads (`--issue` and `--run` also supported). Credentials come from GH_TOKEN/GITHUB_TOKEN. Inspect coverage diagnostics. PRs, commits, reviews and CI summaries are activity evidence, not captured model/tool execution; do not claim CI success proves workflow portability. MCP exposes `switchit_sources_github` with the same controls.

To connect imported GitHub activity with agent sessions, use `sources correlate --agent-events AGENT.json --github-events GITHUB.json --output DIR`, or MCP `switchit_sources_correlate`. Read correlation.json: exact references are noncausal and do not prove authorship; missing/ambiguous matches stay unresolved. Prefer active revisioned stores to mixed historical snapshots. Do not substitute repository metadata for executable steps or holdout cases.
