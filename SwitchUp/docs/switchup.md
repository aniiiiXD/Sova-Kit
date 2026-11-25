# SwitchUP

SwitchUP runs a local improvement loop around a connected text model. Captures and corrections supply the evidence. A model or operator proposes new instructions. SwitchIt replays the active and proposed versions against the same reviewed cases. A candidate becomes active when it fixes an observed failure, passes the full suite, and the operator activates it.

## Connect a model

Install the Python package with `python -m pip install ./SwitchIt` from the Sova-Kit root. It exposes `switchup`, `switchup-mcp`, and the SwitchIt entry points. In Sova-Kit, the workspace defaults to `Sova-Kit-data/workspace/platform/switchup/` alongside the kit. Set `SOVA_KIT_DATA_DIR` to choose another data root. Use `--state-dir` on `improve` commands and `--switchup-dir` on `serve` to share a custom workspace.

Save a target connection as `target.json`. The options are `openai`, `anthropic`, `bedrock`, or a SwitchIt provider preset such as `groq`, `ollama`, or `vllm`. For a custom OpenAI-compatible endpoint, use `kind` set to `openai` and give the API base URL. The model ID is pinned in the project.

```json
{
  "kind": "openai",
  "model": "your-local-model-id",
  "base_url": "http://127.0.0.1:8000/v1"
}
```

The example connects to a keyless loopback server. Hosted endpoints use provider environment variables or `api_key_env` to name a specific variable. Bedrock uses AWS SDK credentials and the optional `region` and `profile` fields. Store keys in the environment.

A new chat project takes an explicit, reviewed `checks.json` array. For a support workflow that should request confirmation before cancellation, one possible check is `[{"type":"output_contains","text":"CONFIRMATION_REQUIRED"}]`. Supply the actual policy and response format through `--prompt` if they belong in the baseline instructions. You can also supply a JSON tool allowlist with `--tools`.

```sh
switchup improve init --project support --connection target.json --checks checks.json --reviewer USER --prompt instructions.txt
```

To start with a reviewed, multi-step SwitchIt workflow, supply `--workflow WORKFLOW.json --cases CASES.json` in place of `--checks` and `--reviewer`. Bind model requests and tool arguments to the case inputs or prior steps so the replay uses each case’s actual inputs. See [the workflow quickstart](../../SwitchIt/docs/workflow-quickstart.md).

## Capture requests and corrections

Import canonical events, a SwitchIt event store, or normalized calls. The CLI imports large files in batches and deduplicates stable event IDs. You can also append events from your application through the local API.

```sh
switchup improve ingest --project support --events EVENTS.json
switchup improve feedback --project support --trace TRACE_ID --correction 'Ask for confirmation before cancelling an order.'
```

For connected requests, launch the workspace with `switchup serve --allow-live-models` and send a JSON POST to `/api/switchup/projects/support/invoke`. The CLI accepts the same request document through `switchup improve invoke --project support --request REQUEST.json --live`.

```json
{
  "request_id": "req-001",
  "trace_id": "trace-001",
  "session_id": "session-001",
  "request": {
    "messages": [{"role": "user", "content": "Cancel my order."}]
  }
}
```

Each request uses the active recipe at the time it is accepted. The response includes the recipe ID, generation, and normalized model result. Supply unique, opaque request, trace, and session IDs. A repeated request ID returns a conflict so a retry cannot silently repeat an upstream model call. Capture keeps redacted evidence of the input, reply, and tool proposals. The calling app executes its approved tools and can append observed tool results through `POST /api/switchup/projects/support/events` with `{"events":[...]}`. To record corrections from the app, post `trace_id`, `correction`, and optional `event_id` to `/api/switchup/projects/support/feedback`.

The connected endpoint accepts canonical text messages and optional `tools` with `name`, `description`, and `parameters`. Tool history uses explicit `tool_call_id` values. Other agent runtimes can integrate through canonical evidence ingestion and read the active prompt from `GET /api/switchup/projects/support/active`.

## Pin a replay suite

For a captured chat project, collect separate sessions for learning and held-out verification. Put the holdout trace IDs in a JSON array and pin the suite. SwitchUP selects one usable request per session and retains the suite by digest.

```sh
switchup improve snapshot --project support --holdout holdout-traces.json --expected-generation 0
```

Use `--cases CASES.json` to pin or refresh a reviewed workflow suite. A case can include an `expected_checks` array for outcomes specific to its input. Both versions face those checks and the workflow’s shared checks. Evidence seen during proposal preparation stays in the learning partition. Held-out trace and session identities stay out of the proposal queue. Use additional unseen cases to measure generalization after iterating on a pinned suite.

## Propose and verify

`prepare` pins the active generation, connection, workflow, suite, and a bounded batch of recorded learning evidence. Corrections and failures take priority within each pass over the evidence. Give the exported training packet to Codex or another operator to write `proposal.json`.

```sh
switchup improve prepare --project support --output learning.json
```

```json
{
  "system_prompt": "Before proposing an order cancellation, confirm the order and get explicit consent.",
  "rationale": "The recorded correction requires confirmation before cancellation.",
  "evidence_ids": ["EVENT_ID_FROM_LEARNING_JSON"]
}
```

```sh
switchup improve propose --project support --round ROUND_ID --proposal proposal.json
switchup improve verify --project support --round ROUND_ID --live --max-model-calls 80
```

You can give `propose` an `--optimizer OPTIMIZER.json --live` connection instead. To generate and verify one candidate in the same invocation, use `switchup improve cycle --project support --optimizer OPTIMIZER.json --live`. Add `--activate` to authorize activation if that candidate passes the gate. The optimizer receives the current instructions, fixed outcome checks, and bounded learning excerpts. The proposal can change the instruction text. It cites the evidence and inherits the target model, connection, tool permissions, and evaluator.

Inspect a round with `switchup improve show --project support --round ROUND_ID`. It includes the proposal, budget, both runs for each replay case, and the decision. A failed or interrupted process leaves its saved phase visible. `improve abandon` closes an unfinished round. A new round uses a new operation ID and starts from the current generation.

## Replay tool effects

For a multi-step workflow, review the alternative steps and attach a `tool_simulation` rule for each modeled tool. Supply an `initial_state` object in each case. The replay runner copies that state independently for each version. Use `state_equals` to check an actual value in the resulting state.

```json
{
  "tool_simulation": {
    "cancel_order": {
      "operation": "set",
      "path": ["orders", {"$ref": "arguments.id"}, "status"],
      "value": "cancelled"
    }
  }
}
```

Supported operations are `read`, `set`, and `delete`. A branch can use `when.on_missing` set to `skip` for an optional tool proposal. Set `requires_proposal` to `true` on model-driven tool steps so they require an exact matching proposal. Unmodeled tools use exact recorded fixtures. Include simulated reads for any state a modeled write can change. The comparison reports the first differing executed call and whether the two final states match. This replay environment consists of the reviewed JSON operations and recorded fixtures.

## Activate and roll back

Open `/improvements` on `switchup serve` to inspect the instruction diff, per-case results, and release controls. The status command shows the current generation. A generation check prevents activating an update after another operator changes the active recipe or suite.

```sh
switchup improve status --project support
switchup improve activate --project support --round ROUND_ID --expected-generation GENERATION
switchup improve rollback --project support --expected-generation NEW_GENERATION
```

New connected requests use the selected recipe. Activation and rollback write a history record in the same SQLite transaction as the project update. The workspace retains the captured events, replay suites, proposals, and run evidence across restarts.

## Compute policy

The local machine stores evidence, assembles packets, and coordinates model calls. Proposals and replay use the model endpoints configured by the operator. Inspect the destinations, data scope, upper bound on calls, and expected cost before authorizing each `--live` run. A cycle uses one optimizer call and up to the configured number of replay model calls. Time and optional token limits stop subsequent replay steps. An in-flight call can exceed a soft budget. Use the normal teardown policy for any separately provisioned GPU or model endpoint.
