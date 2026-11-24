# Local Aegis evidence bridge

`python3 -m switchit.aegis_bridge --store ROOT` reads one JSON request from stdin
and emits one versioned JSON response. `--request-file FILE` reads the same request
from a daemon-created file. The Aegis process supplies `ROOT`; it is not a
browser-selectable filesystem destination. Importing the module does not start a
server, read private sessions, or mutate storage.

The bridge is local development code following SwitchIt 0.2.0; the already
published 0.2.0 artifacts are not changed by this app integration. Use the local
checkout or an interpreter with that checkout installed.

## Protocol

```json
{"bridge_version":1,"operation":"capture","snapshot":{"format":"aegis-snapshot","version":1,"bundle":{},"runs":[],"events":[],"verifications":[],"coverage":[]}}
```

The example shows the envelope only; `bundle` must be a valid Aegis task export.
All requests include `bridge_version: 1` and one operation:

| Operation | Other request fields | Result |
| --- | --- | --- |
| `capture` | `snapshot` (Aegis task bundle or daemon snapshot) | Stored task detail |
| `list` | None | `{tasks:[summary]}` |
| `get` | `task_id`, optional `artifact` | Task detail, or `{artifact,document}` |
| `sources` | None | Configured native source IDs and display labels |
| `attach_trace` | `task_id`, either `events` or `source_id`, optional `run_id`, `attachment_id` | Task detail with attached trace evidence |
| `build_workflow` | `task_id` | Draft workflows from eligible attached evidence |
| `review_workflow` | `task_id`, `workflow_id`, `reviewer`, `checks`, optional `revision`, `bindings` | Reviewed contract and recorded-fixture evaluation |
| `export_skill` | `task_id`, `workflow_id`, optional `revision`, `include_cases` | Task detail plus `{export:{path,manifest,includes_private_evidence}}` |

Success is `{bridge_version:1,ok:true,result:{...}}`; failure is
`{bridge_version:1,ok:false,error:{code,message}}`, with a nonzero exit code.
Protocol stdout contains no logging banners. Requests are limited to 8 MiB,
50 JSON nesting levels, unique fields, and finite numbers. A task can have at
most 3,000 active events. Review accepts at most 30 supported checks and 100
cases; workflow drafts are limited to 100 steps.

Task detail includes `task`, up to 200 `events`, a bounded `graph`, `workflows`,
configured `sources`, and explicit `limitations`. The summary has task/prompt,
run/event/native-event counts, evidence status, workflow/reviewed counts, update
time and evidence revision. Workflow entries include the contract, revision,
evidence revision, staleness flag, fixture verdict counts and evaluation.
`artifact` is allowlisted to `store`, `graph`, `candidates`, `drafts`, and
`metadata`; it cannot be a filesystem path.

## Source boundary

Aegis semantic messages, commands, findings and historical verifications remain
`metadata_only`. Capture alone cannot create an executable workflow. Native
canonical event documents may be explicitly uploaded; their provenance adds
`caller_supplied_trace`, rather than treating upload as certified capture.

For native Codex, Claude Code or Cursor session files, trusted daemon startup
configuration may pass `--sources CONFIG.json`:

```json
{"selected-session":{"agent":"codex","path":"/explicit/session.jsonl","label":"Selected session"}}
```

The request then selects `source_id: "selected-session"`. Native file sources
must be explicitly configured files no larger than 32 MiB. The bridge never
scans a home directory, accepts an arbitrary browser filesystem path, or uses
an agent importer to authorize live execution. A supplied `run_id` must belong
to the previously captured task; attaching traces is explicit association, not
proof that Aegis generated those traces.

## Review and storage

Storage is rooted under `ROOT/tasks/<hashed-task-id>/`. Task IDs do not become
path segments. Capture revisions and reviewed contracts remain available while
active source pointers advance. File writes are atomic and a POSIX file lock
serializes bridge operations. Store-relative linked components and symlink
leaves are rejected; store files with multiple hard links are rejected. This is
local application storage, not an OS sandbox for hostile same-user processes.

Changed evidence marks a reviewed workflow stale. Rebuilding identical evidence
preserves its reviewed contract. Supplying a revision selects an exact stored
contract; exporting a stale contract is rejected. Review requires attributable
criteria beyond `no_error` and saves pass/fail/error/uncertain recorded-fixture
results. The bridge does not execute remote models, coding agents, or business
tools.

Skill exports stay under the store root. Contract-only export is the default;
bundled case evidence requires `include_cases: true`. Exported fixture verification
establishes consistency with recorded evidence, not live cross-agent portability.

## Verification

```sh
python3 -m unittest tests.test_aegis_bridge -v
```

Focused tests cover the subprocess request-file protocol, explicit native file
selection, uploaded provenance and run association, metadata-only rejection,
fixed checks, positive/negative fixture outcomes, exported verifier execution,
source revision staleness, idempotent rebuild, and external symlink sentinels.
