# Local workflow platform

Run the full local workspace with Python 3.10+ and no frontend build:

```sh
python3 -m switchit.cli serve --host 127.0.0.1 --port 8787
```

In Sova-Kit, local jobs and captured events are stored under `Sova-Kit-data/workspace/platform/` alongside the checkout. Pass `--data-dir` to use a custom platform workspace.

Open <http://127.0.0.1:8787>. The Python server serves both its UI and API. The existing Next.js marketing site remains separate. This long-running local HTTP server is not a Vercel serverless deployment.

The [SwitchUp improvement workspace](../../SwitchUp/docs/switchup.md) is available at `/improvements`. Its loopback API displays project versions and replay evidence, accepts captures and corrections, and applies verified activations or rollbacks. Use `--allow-live-models` to enable the connected model invocation route.

## Evidence to reviewed verification

1. Import a JSON/JSONL file containing canonical SwitchIt events or calls. Accepted shapes are an array, `{"events": [...]}`, or `{"calls": [...]}`. Native vendor logs must first pass through a SwitchIt CLI importer. The platform does not scan your home directory.
2. Inspect the execution graph. Solid edges represent recorded relationships; dashed edges represent inferred ordering. Selecting a node shows its payload and provenance. Graph issues remain visible. The graph displays the first 80 events; downloadable artifacts contain the whole graph.
3. Inspect the workflow draft and its individual steps. Recorded arguments are proposals; optional JSON overrides allow explicit `inputs`, `outputs`, `steps`, and `allowed_tools`. Bind parameters with the workflow module's `{"$ref":"inputs.task"}` syntax.
4. Enter a reviewer and explicit success criteria. The simple form requires an exact final output (`output_equals`). Advanced JSON supports other existing outcome checks. `no_error` alone is insufficient. There is no automatic business review.
5. Verify the reviewed workflow against whole-segment cases using recorded model outputs and exact tool fixtures. Inspect per-case checks, failures, missing evidence, and downloadable artifacts.

**Recorded fixture verification is a baseline check. It does not launch a target vendor, a live model, or a business tool and does not prove vendor migration or production equivalence.** A job can finish with failing, uncertain, or error verdicts. A completed job does not mean its criteria passed. Cases labelled holdout exclude the selected construction segment, but discovery used the source corpus: these are not independent unseen benchmarks.

“Explore sample traces” creates visibly labelled synthetic evidence. It still requires your own reviewer and criteria. The sample's recorded final output is `The account is active`.

## JSON API

All mutation requests use `Content-Type: application/json`. Requests are limited to 2,000,000 bytes, 50 nesting levels, and unique JSON fields. Non-finite numbers are rejected. Submission accepts 1–3,000 records; each job produces at most 100 fixture cases. Two workers share a queue limited to 24 active jobs.

| Method | Route | Request / response |
| --- | --- | --- |
| GET | `/api/health` | Local execution mode and health |
| POST | `/api/jobs` | `{name?, events:[...]}`, `{name?, calls:[...]}`, or `{name?, demo:true}` → job, HTTP 202 |
| GET | `/api/jobs` | `{jobs:[...]}`, newest first |
| GET | `/api/jobs/{id}` | Status, phase, counts, progress, available artifacts, errors and final verdict counts |
| POST | `/api/jobs/{id}/review` | `{reviewer, checks, candidate_id?, inputs?, outputs?, steps?, allowed_tools?}` → queued job |
| POST | `/api/jobs/{id}/cancel` | `{}` → cooperative cancellation |
| POST | `/api/jobs/{id}/resume` | `{}` → restart stopped task from saved input/review |
| GET | `/api/jobs/{id}/artifacts/{name}` | Allowlisted JSON artifact; append `?download=1` for attachment |

Artifact names: `input.json`, `graph.json`, `dependencies.json`, `candidates.json`, `draft.json`, `workflow.json`, `cases.json`, `evaluation.json`.

Example explicit review:

```json
{
  "reviewer": "Your name",
  "checks": [
    {"type": "no_error"},
    {"type": "output_equals", "text": "The account is active"}
  ]
}
```

Available advanced check types: `no_error`, `output_equals`, `output_contains`, `output_min_chars`, `tool_called`, `tool_args_include`, `tool_calls_equal`, `tool_sequence`, `allowed_tools`, `step_before`, `state_equals`, `artifact_equals`, `artifact_exists`, `latency_ms_max`, `usage_input_max`, `usage_output_max`, and `cost_max`. Missing evidence remains unknown. Arbitrary regular expressions are excluded from this HTTP API to prevent a check from monopolizing a worker.

There is no endpoint for commands, arbitrary runtime adapters, provider credentials, external URLs, or filesystem input paths.

## Durability and local boundaries

Jobs and normalized evidence persist under `<data-dir>/jobs/<uuid>/`. Artifact writes and job metadata writes are atomic. One server process owns a data directory through an OS lock; a second server cannot alter an active server's jobs. In-flight jobs recovered on restart become `interrupted` and can be resumed. Resume restarts the analysis or verification task from durable input; it is not a partial instruction replay. Cancellation is checked between phases and cases, and allows the current bounded operation to finish.

The server defaults to loopback. Host and same-origin validation, JSON-only mutations, static artifact allowlists, path containment checks, and a restrictive content security policy protect the local browser boundary. It does not provide authentication, TLS, shared workspaces, a public deployment, or multi-user isolation. Keep it local. Request logs omit imported content.

Imported events pass through the project's existing normalization/redaction routines. Redaction is best-effort: local artifacts can still contain sensitive evidence. `.switchit/` is ignored by Git; choose another data directory with care. The UI makes no requests to analytics, fonts, vendor APIs, or remote assets.

## Verification

```sh
python3 -m unittest discover -s tests -p test_platform.py -v
node --check switchit/platform/static/app.js
```

Integration tests exercise import, explicit review, binding overrides, passing and failing recorded outcomes, artifact retrieval, malformed requests, origin/Host checks, traversal and symlink containment, cancellation, single-process ownership, and interrupted job recovery. Browser verification covers sample creation, graph inspection, workflow inspection, user-entered review, actual fixture results, JSON file import, and desktop/mobile layout.
