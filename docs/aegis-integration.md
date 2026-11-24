# Local Aegis integration

SwitchIt can read an Aegis task export or collect one selected task from a running
local Aegis daemon. The integration is read-only: it does not launch agents,
execute imported commands, change Aegis state, or send data to hosted services.

## Import an exported task

Export a task from Aegis, then run from the SwitchIt checkout:

```sh
python3 -m switchit.cli sources aegis \
  --file /path/to/task.aegis.json \
  --output .switchit/aegis
```

Both Aegis `aegis-task` version 1 bundles and SwitchIt's `aegis-snapshot` version 1
files are accepted. The latter wraps `bundle`, `runs`, `events`, `verifications`
and `coverage`. The checked-in `tests/fixtures/aegis-task.json` is **synthetic**;
it can be substituted for the file above to try the pipeline without Aegis.

## Read a running daemon

Use the URL opened by Aegis (its port is assigned at launch) and the selected task
ID. Task IDs are available from Aegis's `GET /api/tasks` response.

```sh
python3 -m switchit.cli sources aegis \
  --url http://127.0.0.1:PORT --task TASK_ID \
  --output .switchit/aegis
```

The adapter uses GET requests for `/api/tasks/:id/bundle`, `/runs`,
`/verifications`, and `/api/events?task_id=...&before_sequence=...&limit=500`.
It walks event history backwards using persisted sequence cursors. `--max-pages`
defaults to 10 and accepts 1–20. A reached page limit saves the collected evidence
and returns exit code 1 with `limited` coverage. Read failures fail the command;
they do not silently replace previous artifacts with incomplete responses.

Only explicit loopback HTTP URLs are accepted. Redirects and environment HTTP
proxies are disabled. Reads have time and response-size limits. Aegis's bundle
and verification APIs are bounded; even a complete event traversal does not mean
every artifact or provider trace was captured. The separate GETs are not an
atomic snapshot, so import after an agent run finishes when possible.

## Outputs and integration surfaces

- `events.json`: canonical events accepted by SwitchIt's graph, workflow discovery,
  correlation and local platform.
- `graph.json`: task → run → observed-event hierarchy, plus inferred ordering
  within each run. Missing runs remain unresolved; they are never invented.
- `store.json`: revisioned evidence. Pass `--store PREVIOUS/store.json` on later
  imports to retain history; identical imports are idempotent. Bundle and daemon
  snapshot scopes are distinct sources because they contain different evidence.
- `export.json`: a redacted, bounded copy of the input for inspection. Redaction
  changes content, so reimporting this copy can have a different revision hash.
- `diagnostics.json`: collection coverage, omissions, unresolved runs and replay
  limitations. CLI success means import success, not workflow portability.

MCP exposes `switchit_sources_aegis` with the same arguments as the CLI. The npm
harness bridge forwards `sources aegis` to its Python backend; the backend must
include this adapter. There is no separate npm implementation.

To use the graph and discovery pipeline directly:

```sh
python3 -m switchit.cli workflow build \
  --events .switchit/aegis/events.json --output .switchit/aegis/workflow
```

The local platform's **Import evidence** dialog accepts Aegis task bundles and
snapshots directly. Its `POST /api/jobs` accepts `{ "name": "Aegis task", "aegis": BUNDLE_OR_SNAPSHOT }`
as well as canonical `{ "name": "Aegis task", "events": [...] }` submissions.
It displays these imports as `evidence_only` and does
not manufacture an executable workflow. To correlate exact GitHub references:

```sh
python3 -m switchit.cli sources correlate \
  --agent-events .switchit/aegis/events.json \
  --github-events .switchit/github/events.json \
  --output .switchit/aegis-linked
```

## Fidelity boundary

Task prompts/context, run metadata/provider session IDs, semantic messages,
command observations, findings and verification argv/output/exit codes are
preserved where present. All records remain activity evidence (`metadata_only`)
for workflow execution purposes. Assistant text is not a complete model request;
command text is not a structured tool invocation/result pair. Unknown event
types are retained as state observations.

Imported verification results are always historical, even if Aegis reported
`current`; the original value is retained as `observed_freshness`. Review findings
never become execution approvals. Advisory working profiles never become routing
instructions. Exact replay still requires original provider transcripts or richer
capture plus a reviewed target runtime adapter.

## Compatibility and validation

Implemented against Aegis commit
[`02bef5b`](https://github.com/AnujJha27/Aegis/tree/02bef5b5eeb4f885df8e42c1deb8bb1fd1e38831),
especially its [JSON protocol](https://github.com/AnujJha27/Aegis/blob/02bef5b5eeb4f885df8e42c1deb8bb1fd1e38831/daemon/protocol/json.h),
[task routes](https://github.com/AnujJha27/Aegis/blob/02bef5b5eeb4f885df8e42c1deb8bb1fd1e38831/daemon/api/routes/tasks.cpp)
and [bundle exporter](https://github.com/AnujJha27/Aegis/blob/02bef5b5eeb4f885df8e42c1deb8bb1fd1e38831/daemon/session/store.cpp).

Tests exercise an actual loopback HTTP connection to a synthetic protocol fixture,
CLI/MCP imports, hierarchy, redaction, pagination, immutable revisions, validation,
and the platform's evidence-only behavior. This does not establish compatibility
with an installed/running Aegis binary; that live check remains pending.
