# GitHub activity evidence

SwitchIt imports GitHub issues, pull requests, discussion comments, review submissions, inline review comments, commits, Actions runs, jobs, and step summaries. Use it to inspect the request, implementation, feedback and CI outcome surrounding agent work.

GitHub activity is **not a full agent execution trace**. Every imported record is marked metadata-only. No hidden prompts, reasoning, model/token usage or business-tool calls are invented. CI success alone does not prove workflow portability. Agent sessions and separately exported trace artifacts supply that additional evidence.

## Read selected resources

```sh
python3 -m switchit.cli sources github --repo OWNER/REPO --pr 123 --remote --output .switchit/github-pr
python3 -m switchit.cli sources github --repo OWNER/REPO --issue 456 --remote --output .switchit/github-issue
python3 -m switchit.cli sources github --repo OWNER/REPO --run 789 --remote --output .switchit/github-run
```

Private repositories require an appropriately scoped read token in `GH_TOKEN` or `GITHUB_TOKEN`. Public resources may be read without a token. Tokens are never CLI arguments or saved export fields. API access is GitHub.com only, uses GET, refuses redirects, and never executes repository code. GitHub Enterprise endpoints are not included in this implementation.

A selected PR collects its comments, reviews, commits, and Actions runs for its current head SHA. Job collection uses each run's exact attempt. Earlier head SHAs and earlier attempts are not scanned automatically. Combine `--pr`, `--issue`, and `--run` to select explicit additional resources. There is no repository-wide crawl.

Collection defaults to 3 pages per collection and 60 total requests. Adjust `--max-pages` (1–20) and `--max-requests` (1–200) explicitly. Partial pagination, access failures on child collections, and request exhaustion retain available evidence and appear in diagnostics; the command exits 1. Failure to fetch an explicitly requested root resource stops collection. Snapshot reads are not a transaction across GitHub endpoints.

## Offline exports

```sh
python3 -m switchit.cli sources github --file github-export.json --output .switchit/github
```

The versioned bundle wraps GitHub REST records, with explicit thread IDs on comments/reviews/commits and run attempts on jobs:

```json
{
  "github_export_version": 1,
  "repository": "example/project",
  "scope": {"pr": 42},
  "pull_requests": [{"number": 42, "title": "Fix search", "state": "open"}],
  "comments": [{"id": 100, "issue_number": 42, "body": "Please add a regression test"}],
  "reviews": [{"id": 101, "pull_number": 42, "state": "APPROVED"}],
  "commits": [{"sha": "abc123", "pull_number": 42, "commit": {"message": "Fix search"}}],
  "runs": [{"id": 200, "run_attempt": 1, "head_sha": "abc123", "conclusion": "success", "pull_requests": [{"number": 42}]}],
  "jobs": [{"id": 201, "run_id": 200, "run_attempt": 1, "name": "test", "steps": [{"number": 1, "name": "Run tests", "conclusion": "success"}]}]
}
```

Optional arrays: `issues`, `review_comments`; optional `coverage` entries describe acquisition completeness. The API collector produces this format. Arbitrary webhook payloads, GraphQL response shapes and copied connector responses must first be converted to this explicit schema; they are not silently guessed.

## Outputs and relationships

- `export.json`: redacted, bounded snapshot suitable for offline re-import.
- `events.json`: canonical events, importable through the local platform's file picker.
- `store.json`: revisioned evidence store. Pass a previous `--store` to retain history.
- `graph.json`: recorded containment plus explicitly noncausal PR/commit/run references.
- `relationships.json`: observed cross-record references, separate from execution dependencies.
- `diagnostics.json`: API coverage, unresolved references and evidence limitations.

Changes to a snapshot create new event revisions; unchanged imports are idempotent. The current source revision replaces the same source scope's active event set, while history stays in the store. Use separate scope values for separate offline selections. Raw API collection order is never presented as execution ordering. Reviews do not become executable approval steps, and Actions jobs do not become reusable tool fixtures.

In the platform, metadata-only imports end at **Evidence imported** with an inspectable graph. They require additional model/tool execution evidence before a replayable workflow can be built. The MCP equivalent is `switchit_sources_github`, with the same options and explicit `remote: true` gate.

Logs, artifact contents, full commit patches and issue timeline transitions are not collected yet. Local agent folders are not scanned automatically; the correlation command below uses explicitly supplied evidence files. No evidence was read from a live repository while implementing this feature; tests use synthetic REST fixtures and mocked transport.

API references: [pull requests and their issue relationships](https://docs.github.com/en/rest/pulls/pulls), [Actions jobs and attempt-specific jobs](https://docs.github.com/en/rest/actions/workflow-jobs), [Actions runs](https://docs.github.com/en/rest/actions/workflow-runs). The collector pins API version `2026-03-10`.

## Correlate with agent execution evidence

Once both sources have been imported, combine their event documents:

```sh
python3 -m switchit.cli sources correlate --agent-events .switchit/agent/events.json --github-events .switchit/github/events.json --output .switchit/linked
```

The MCP equivalent is `switchit_sources_correlate`. It accepts `agent_events`, `github_events`, and `output` paths. Inputs can also be canonical call stores or revisioned event stores; only active revisions are selected from a store. Identical events are deduplicated; conflicting event IDs reject the merge.

The output directory contains combined `events.json`, `graph.json`, and `correlation.json`. Import the combined events into the platform to inspect both sources together. Graph construction recomputes the same reference report, so references survive file import without embedding derived links into the original evidence.

Matching uses exact recorded references:

- Full GitHub PR, issue, commit, or Actions-run URLs in recorded payloads. PR file/check/commit views resolve to the same PR identity.
- Explicit `payload.github_reference` metadata: `repository` plus `pull_number`, `issue_number`, `commit_sha`, or `run_id`.
- Repository-scoped commit references require a full 40- or 64-character hexadecimal SHA.

For example, an importer may preserve a structured reference as:

```json
{"github_reference":{"repository":"example/project","pull_number":42}}
```

There is no matching by title, branch, author, timestamp, bare issue number or abbreviated SHA. If two imported snapshots or run attempts match the same reference, the report marks it `ambiguous_snapshot` and creates no link. A resource outside the import is `not_imported`; correlation never fetches it automatically.

A matched reference establishes only that the recorded payload mentions that imported resource. It does **not** establish that the agent authored the commit, opened the PR, passed CI or reproduced a workflow. The graph marks these edges `causal: false`; workflow construction excludes them from step dependencies. The UI shows them as noncausal references and surfaces unresolved counts.

When metadata-only GitHub segments and executable agent segments describe the same task, draft construction selects executable evidence. Case construction excludes metadata-only segments and records their IDs and an uncertainty flag. Repository activity cannot supply fake replay cases or an artificial holdout.
