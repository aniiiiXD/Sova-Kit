# Independent workflow modules

These modules are callable as Python APIs and through the new workflow CLI/MCP surface. The original `switchit run` replay pipeline and compatibility imports remain available. See [workflow quickstart](workflow-quickstart.md), [orchestration](orchestration.md), and [platform](platform.md) for integrated entry points.

The new modules use versioned JSON dictionaries. Invalid versions, conflicting identities, dependency cycles, incompatible runtime versions, and changed workflow revisions are rejected explicitly. Use the public functions below rather than importing implementation helpers.

## Ownership, APIs, and tests

| Responsibility | Module and public entry points | Focused tests |
| --- | --- | --- |
| Source inventory | [`source_inventory.py`](../switchit/source_inventory.py): `inventory_sources`, `source_digest` | `tests.test_source_pipeline` |
| Audited incremental imports | [`import_pipeline.py`](../switchit/import_pipeline.py): `import_inventory`, `active_events` | `tests.test_source_pipeline` |
| Canonical evidence | [`events.py`](../switchit/events.py): `normalize_event`, `validate_event`, `events_from_calls` | `tests.test_events_graph` |
| Revisioned event storage | [`event_store.py`](../switchit/event_store.py): `empty_event_store`, `merge_events`, `ingest_events` | `tests.test_events_graph` |
| Execution graph | [`execution_graph.py`](../switchit/execution_graph.py): `build_execution_graph` | `tests.test_events_graph` |
| Dependency manifest | [`dependencies.py`](../switchit/dependencies.py): `profile_dependencies` | `tests.test_workflow_modules.DependencyTests` |
| Task segmentation and grouping | [`workflow_patterns.py`](../switchit/workflow_patterns.py): `discover_workflows` | `tests.test_workflow_modules.WorkflowDiscoveryTests` |
| Workflow specification | [`workflow_builder.py`](../switchit/workflow_builder.py): `build_workflow`, `review_workflow`, `validate_workflow` | `tests.test_workflow_modules.BuilderTests` |
| Exact tool fixtures | [`fixtures.py`](../switchit/fixtures.py): `build_fixtures`, `FixtureSession` | `tests.test_workflow_modules.FixtureTests` |
| Whole-workflow cases | [`workflow_cases.py`](../switchit/workflow_cases.py): `build_workflow_cases` | `tests.test_workflow_modules.WorkflowCaseTests` |
| Runtime adapters | [`runtime_adapters.py`](../switchit/runtime_adapters.py): `FixtureRuntime`, `ProviderRuntime`, `CommandRuntime`, `RuntimeRegistry` | `tests.test_workflow_runtime` |
| Controlled execution | [`workflow_runner.py`](../switchit/workflow_runner.py): `run_workflow` | `tests.test_workflow_runtime` |
| Independent evaluation | [`workflow_evaluation.py`](../switchit/workflow_evaluation.py): `evaluate_workflow` | `tests.test_workflow_runtime.EvaluationTests` |
| Immutable recipes | [`recipes.py`](../switchit/recipes.py): `make_recipe`, `RecipeRegistry` | `tests.test_composer` |
| Bounded composition and trials | [`composer.py`](../switchit/composer.py): `propose_recipes`, `evaluate_candidates` | `tests.test_composer` |
| Activation history | [`promotions.py`](../switchit/promotions.py): `validate_selection`, `PromotionStore` | `tests.test_composer` |
| Shared validation | [`contracts.py`](../switchit/contracts.py): `ContractError`, `fingerprint`, `topological_order` | Exercised at each contract boundary |

Existing agent parsers live under [`importers/`](../switchit/importers). Existing model-provider adapters remain in [`providers.py`](../switchit/providers.py). Existing aggregation and JSON/HTML reports remain separate modules; experiment output has its own JSON schema rather than pretending to be a legacy migration report.

## Run the modules

Run a single module's tests or the whole suite:

```sh
python3 -m unittest tests.test_events_graph
python3 -m unittest tests.test_source_pipeline
python3 -m unittest tests.test_workflow_modules
python3 -m unittest tests.test_workflow_runtime
python3 -m unittest tests.test_composer
python3 -m unittest tests.test_replay_correctness
python3 -m unittest discover -s tests
```

An offline composition example is available without credentials or personal session imports:

```sh
python3 -m switchit.module_demo --output /tmp/switchit-modules
```

It writes ten synthetic events, their graph, a dependency manifest, workflow candidates and draft, an explicitly reviewed fixture specification, two cases from separate sessions, two recipe candidates, four trials, and local recipe/promotion history. A deliberately incorrect control fails; the correct recorded control passes. This demonstrates the evaluation gate, not performance improvements in a real model or agent.

For real authorized local sources, keep each stage explicit:

```python
from pathlib import Path
from switchit.source_inventory import inventory_sources
from switchit.import_pipeline import import_inventory, active_events
from switchit.execution_graph import build_execution_graph
from switchit.workflow_patterns import discover_workflows
from switchit.workflow_builder import build_workflow

inventory = inventory_sources({"codex": [Path("/path/to/exported/sessions")]})
imported = import_inventory(inventory)
graph = build_execution_graph(active_events(imported["store"]))
candidates = discover_workflows(graph)
draft = build_workflow(candidates["candidates"][0], graph)
# Inspect diagnostics and evidence, parameterize the draft, then call review_workflow.
```

For incremental ingestion, pass the previous inventory to `inventory_sources` and the previous event store to `import_inventory`. A source revision is content-addressed. Unchanged revisions are skipped; old revisions are retained. `active_events` selects the latest imported revision per source. Missing sources are reported by discovery; absence from a limited scan is not treated as deletion. File-backed stores are single-writer local artifacts, not concurrent databases.

## Evidence and hierarchy contract

Each event carries identity, source provenance, trace/session IDs, source order, kind, payload, evidence level, and uncertainty. Optional parent, span, agent, retry, handoff, and tool-call IDs remain explicit. Original timestamp values are preserved; source index supplies ordering, so unknown timestamp units are not silently converted.

The graph resolves explicit IDs, reports missing/ambiguous references, rejects cyclic edges, and labels chronological sequence edges as inferred. A tool result is linked by ID within its trace. Parsing a subagent file does not establish its parent when the source omitted the relationship. Legacy call conversion cannot recover information already lost upstream. Devin/catalog metadata remains metadata-only and cannot independently create an executable workflow.

Codex and Claude Code imports preserve tool/result identities and explicit record relationships. Muse and Grok preserve available tool identities. JSONL/OTLP/Langfuse imports retain available parent/time fields. Coverage still varies by source format and version; the importer fixtures are evidence of supported samples, not universal schema coverage.

## Workflow and fixture contract

Discovery segments explicit user turns and groups task text conservatively. It retains tool-sequence variants and failure markers. Grouping is heuristic, not measured semantic accuracy.

The builder creates a **draft** from a selected observed segment. Steps carry source evidence. Dependencies bridge intervening non-executable events, such as tool results. Inputs, outputs, ordering, tool permissions, variants, and success criteria need review and, where appropriate, parameterization. Use `{"$ref": "inputs.account"}` to bind step arguments to case inputs. A condition is `{"path": "inputs.enabled", "equals": true}`; unknown condition inputs make the run uncertain. This is a bounded declarative format, not arbitrary executable branch code.

`review_workflow` requires attribution and at least one required outcome check beyond `no_error`. Changing a specification changes its revision, invalidating cases/recipes pinned to the old revision. This API records caller-supplied review; it is not an identity or approval authentication system.

Whole-workflow cases retain complete segments. Train/holdout splitting groups traces by session where available and keeps workflow-construction evidence in training. Discovery still used the source corpus, which is reported in suite uncertainty; a blind release holdout must be supplied separately. Fixtures match tool name, exact arguments, and occurrence within a trace. Missing, ambiguous, failed, or truncated results are not manufactured. Model outputs in fixture runs are recorded baseline evidence. They cannot establish a live target agent's performance.

## Runtime and evaluation contract

A runtime implements `capabilities`, `execute`, and optionally `snapshot_state`. Registry entries must match their pinned identity/version. Runtime factories construct a fresh instance for each trial.

- `FixtureRuntime` executes only explicit recorded outputs and tool fixtures.
- `ProviderRuntime` uses the existing model adapters and fixture-backed tool steps. Supported recipe settings are a system prompt, allowed tools, `max_attempts` (1–3), `context_policy` (`full` or `last_user`), and `include_step_outputs`. Retry usage is accumulated.
- `CommandRuntime` is an opt-in JSON stdin/stdout bridge for an explicitly configured executable. The request is `{step, inputs, context}`; the response is a JSON object containing output/usage/error fields. Caller-supplied environment variables are not inherited automatically. Vendor-specific CLI wrappers still need implementing and testing; no Codex/Claude Code/Devin/Grok/Muse execution support is inferred from their importer names.

The runner uses a fresh temporary workspace, validates artifact paths, resolves input bindings, enforces tool scope, and checks model proposals against the approved tool step before consuming fixtures. Unconsumed proposals remain uncertain. Step/time/token budgets stop subsequent work. An in-flight provider call may exceed a soft budget; this is not a hard billing ceiling. Command adapters have subprocess timeouts. A temporary working directory is not an OS security sandbox.

The evaluator owns checks separately from recipe selection. It supports existing output/usage checks plus exact ordered tool sequences, step ordering, allowed tools, captured artifacts, observed state, and reported cost. Missing state/cost evidence stays unknown. A failed, incomplete, budget-exceeded, or unevaluable run cannot become a passing configuration.

## Recipe selection and promotion

`propose_recipes` composes a bounded set from caller-supplied variants, preserving workflow revision and permissions. It can filter against runtime capabilities. It does not generate or install arbitrary plugins.

`evaluate_candidates` runs fixed checks against every case, enforces candidate/trial limits, requires independent train/holdout groups by default, and chooses among passing candidates using mean reported cost or latency. Missing objective measurements disqualify selection. This is empirical selection among supplied recipes, not proof of a globally optimal setup.

`PromotionStore` verifies experiment fingerprints, coverage, pinned recipe identity, and recomputed evaluations before recording activation. Accepted recipes and complete experiments are persisted before updating the active pointer. Generation checks detect stale callers. Rollback restores configuration history; it cannot undo external business effects. Evidence hashes detect changed artifacts but are not signed attestations against a hostile caller with write access.

## Corrections to the original replay pipeline

Repeated tool calls and repeated check types are compared individually. Chained usage/latency includes all turns. Continuation messages share tool-call IDs; preserved IDs use native Anthropic/Bedrock tool blocks. Bedrock text content uses content-block arrays. See the official [Bedrock message format](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Message.html), [content blocks](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html), and [Anthropic tool documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools).

Legacy name-only fixtures remain available with an uncertain verdict. Exact fixture documents can be supplied through `fixture_set`. The internal replay fingerprint version is now `0.2.0`, so old checkpoints cannot silently resume under changed semantics. This is not a published package release.

## Validation limits

The local suite covers the legacy pipeline, independent workflow modules, public CLI/MCP integration, exported-skill verification, orchestration, local platform, and npm packaging. Optional Parquet/zstd tests skip when their dependencies are unavailable. Wheel checks run both demo paths from isolated package contents. No live provider calls, real-agent executions, paid model runs, or production tool effects are part of these checks. Real vendor wrappers, app-specific fixtures and success criteria, larger labeled workflow evals, stronger process isolation, cross-process persistence for standalone event/promotion stores, and production deployment remain separate work. The local platform and orchestration coordinator enforce their own ownership boundaries; they do not turn every standalone JSON store into a multi-writer database.
