# Local workflow platform and packages

SwitchIt has one Python execution engine. The npm harness launches that engine; the npm skill installer installs agent instructions. Exported workflow skills carry a particular reviewed workflow contract. The platform uses the same Python workflow modules.

These changes are local development work, not a claim that registry releases or the hosted site have been updated.

## Run the platform

```sh
python3 -m switchit.cli serve --host 127.0.0.1 --port 8787 --data-dir .switchit/platform
```

Open http://127.0.0.1:8787. Import a JSON event/call document or start the synthetic example. Inspect the reconstructed evidence, supply a reviewer and outcome checks, then verify against recorded fixtures. The server persists jobs, progress and evidence. Fixture success measures consistency with recorded evidence; a live target requires a separate provider or configured command run.

## Complete workflow through the CLI

The commands below use the bundled synthetic evidence. For real evidence, use an authorized export or the explicit source inventory commands below. The demo's reviewer and checks are synthetic test data.

```sh
python3 -m switchit.module_demo --output .switchit/example
python3 -m switchit.cli workflow build --events .switchit/example/events.json --output .switchit/bundle
```

`build` writes `events.json`, `graph.json`, `dependencies.json`, `candidates.json` and `drafts.json`. Select a draft if multiple candidates exist. Review its steps, bindings, tool permissions and output paths. Put meaningful business success checks into a JSON array file; no_error alone is insufficient.

```sh
python3 -m switchit.cli workflow review --workflow .switchit/bundle/drafts.json --checks CHECKS.json --reviewer REVIEWER --output .switchit/workflow.json
python3 -m switchit.cli workflow cases --workflow .switchit/workflow.json --graph .switchit/bundle/graph.json --candidates .switchit/bundle/candidates.json --output .switchit/cases.json
python3 -m switchit.cli workflow run --workflow .switchit/workflow.json --case .switchit/cases.json --case-id CASE_ID --output .switchit/result.json
python3 -m switchit.cli workflow export-skill --workflow .switchit/workflow.json --cases .switchit/cases.json --output .switchit/exported-skill
```

`--bindings FILE` on review accepts `inputs`, `outputs`, `steps`, and `allowed_tools` edits. A reviewed revision is immutable by digest; changing the contract requires reviewing it and rebuilding cases. The run command returns nonzero for fail, uncertain or error while retaining the full result artifact. A single case JSON is also accepted without `--case-id`.

The exported folder contains SKILL.md, the contract, an optional private case suite, a file-hash manifest and `scripts/verify.py`. Run that script with the Python engine installed to verify bundled fixtures. Export does not install the skill or publish private evidence. Choose a new export directory for a new revision.

## Source discovery and trace hierarchy

Create a roots JSON mapping supported importer names to explicit directories, for example `{"codex":["/path/to/authorized/codex/sessions"],"claude-code":["/path/to/authorized/claude/projects"]}`. Use the importer names in `switchit/importers/registry.py`.

```sh
python3 -m switchit.cli sources inventory --roots ROOTS.json --output .switchit/inventory.json
python3 -m switchit.cli sources import --inventory .switchit/inventory.json --output .switchit/imported
python3 -m switchit.cli workflow build --events .switchit/imported/events.json --output .switchit/bundle
```

Pass `--previous` to inventory and `--store` to import for incremental revisions. The store retains historical events; workflow build selects active source revisions. Missing files, malformed records, unsupported schemas and inferred relationships remain explicit. Generic Downloads exports still need an explicit format/import route; this does not infer every undocumented vendor format.

## Provider and configured agent execution

`workflow run --runtime provider --provider NAME --model MODEL --remote` uses a supported provider and environment credentials. Tools use exact recorded fixtures. `--recipe RECIPE.json` pins a runtime recipe. Live model output is evaluated against the same reviewed outcome checks.

For a caller-supplied agent wrapper, use `--runtime command --command-config CONFIG.json --allow-execution`. Config contains `runtime_id`, `version`, `argv`, optional `remote`, explicit `env`, and `models`. The wrapper receives `{step, inputs, context}` as JSON on stdin and returns one result object on stdout. A remote wrapper also requires `--remote`. See runtime contracts for result fields. A temporary working directory is not an OS sandbox.

Imported Codex, Claude Code, Devin, Grok or Muse logs do not imply native unattended execution adapters for those vendors. Configure a compatible wrapper where the vendor permits one.

## Harness composition

Use the Python `make_recipe` API to create content-addressed baseline recipes. `harness propose` applies bounded explicit variants; it cannot change the workflow or its success checks. `harness compare` evaluates complete cases with independent session groups. Fixture runtime is built in; `--runtimes` optionally configures a provider connection. Credentials remain in the environment.

```sh
python3 -m switchit.cli harness propose --workflow WORKFLOW.json --baseline RECIPE.json --variants VARIANTS.json --output PROPOSALS.json
python3 -m switchit.cli harness compare --workflow WORKFLOW.json --cases CASES.json --recipes PROPOSALS.json --output EXPERIMENT.json
python3 -m switchit.cli harness promote --experiment EXPERIMENT.json --store .switchit/promotions.json --expected-generation 0
```

Promotion validates the experiment and persists the recipe plus evaluation evidence before advancing a local pointer. It does not deploy an agent or publish a package. `harness status` reads the pointer; `harness rollback --workflow-id ID --expected-generation N` restores a prior local configuration. Cases keep the workflow-construction trace in training, but discovery sees the imported source corpus. Comparison reuses its supplied holdout: keep a separate release holdout outside iterative tuning.

## Autonomous development graph

See [orchestration](orchestration.md) for the durable DAG, configurable worker wrappers and local smoke plan. The scheduler runs independent commands in parallel, preserves successful artifacts, bounds failures and resumes interrupted work. Shared-cwd tasks serialize. Configure separate checkouts to allow coders to run concurrently.

The workflow evidence graph and the development DAG are distinct. The scheduler supplies durable execution; configured investigators/coders/evaluators supply the reasoning. It does not invent business acceptance criteria or vendor credentials, automatically merge worktrees, or publish releases by default.

Generate and run a credential-free fanout/join graph:

```sh
python3 -m switchit.cli orchestrate smoke-plan --workspace .switchit/graph-workers --output .switchit/smoke-plan.json
python3 -m switchit.cli orchestrate run --plan .switchit/smoke-plan.json --output .switchit/graph-run
python3 -m switchit.cli orchestrate run --plan .switchit/smoke-plan.json --output .switchit/graph-run --resume
```

For configured agents, `orchestrate plan --workers WORKERS.json --output PLAN.json` accepts a document containing `workers`, explicit `sources`, optional `max_parallel`, and optional per-lane `repairs` with fixed acceptance-file paths. Configured repair lanes execute coder/verifier rounds before the independent downstream test gate. It creates no publish task. `orchestrate repair --config REPAIR.json --output RUN_DIR` runs bounded coder/verifier rounds while checking frozen acceptance files. See the orchestration contract for worker roles and repair configuration.
