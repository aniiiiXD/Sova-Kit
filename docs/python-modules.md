# SwitchUP Python modules

Install SwitchUP from the repository with `python -m pip install .` or download the wheel from [GitHub Releases](https://github.com/aniiiiXD/SwitchIt/releases/tag/v0.3.0). Both modules live in the same Python distribution and can be imported independently.

## Traces

`switchit.switchup.traces` imports model and tool events into a local SQLite workspace, skips identical event IDs, and builds holdouts by session. Import [canonical evidence](module-contracts.md) from your application. Captured corrections reference the trace they correct.

```python
from switchit.switchup import traces

workspace = traces.Workspace("./switchup-state")
counts = traces.ingest_events(workspace, "support", traces.read_events("events.json"))
workspace.feedback("support", "train-001", "Ask for confirmation before cancelling.")
project = traces.pin_chat_suite(workspace, "support", ["holdout-001"], expected_generation=0)
```

Initialize the project first with `loop.initialize` and the reviewed workflow, model connection, and initial instructions. The sample trace IDs must exist in the supplied events. For raw event validation independent of a SwitchUP project, use `switchit.events.normalize_event` and `switchit.event_store.ingest_events`.

## Improvement Loop

`switchit.switchup.loop` verifies a proposed instruction revision against the same pinned workflow and cases as the active one. An application can call the individual stages or run one bounded round. To repeat the loop, collect fresh corrections or failures and run another round from the active version.

```python
from switchit.switchup import loop

round_record = loop.run_round(
    workspace,
    "support",
    optimizer=optimizer_connection,
    allow_live=True,
    max_model_calls=40,
)
if round_record["status"] == "verified":
    loop.activate(workspace, "support", round_record["id"],
                  expected_generation=round_record["generation"])
```

The optimizer connection is supplied by your application. For a manually authored revision, call `workspace.prepare_round`, read its evidence, and call `loop.propose(..., patch=patch)` followed by `loop.verify`. The patch contains `system_prompt`, `rationale`, and `evidence_ids` from that round. `workspace.rollback` restores the prior active recipe for future invocations. See the [SwitchUP guide](switchup.md) for connection configuration, replay constraints, and the full lifecycle.
