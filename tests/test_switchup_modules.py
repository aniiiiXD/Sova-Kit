"""Exercise the public SwitchUP APIs with a deterministic provider fixture."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from switchit.switchup import loop, traces
from switchit.switchup.contracts import chat_workflow


class FixtureProvider:
    remote = False

    def __init__(self, model):
        self.model = model

    def replay(self, request, case=None):
        if self.model == "optimizer":
            packet = json.loads(request["messages"][-1]["content"])
            proposal = {
                "system_prompt": "Always confirm a cancellation.",
                "rationale": "Follow the recorded correction.",
                "evidence_ids": [item["event_id"] for item in packet["evidence"]],
            }
            return {"text": json.dumps(proposal), "tool_calls": []}
        revised = any("Always confirm" in msg["content"] for msg in request["messages"] if msg["role"] == "system")
        return {"text": "CONFIRMATION_REQUIRED" if revised else "Proceeding.",
                "tool_calls": [], "model": self.model, "latency_ms": 1,
                "usage": {"basis": "recorded", "input_tokens": 10, "output_tokens": 2}}


class SwitchUPModuleTests(unittest.TestCase):
    def test_trace_to_verified_revision_and_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = traces.Workspace(folder)
            spec = chat_workflow("support", [{"type": "output_contains", "text": "CONFIRMATION_REQUIRED"}], "fixture")
            connection = {"kind": "openai", "model": "target", "base_url": "http://127.0.0.1:8000/v1"}
            loop.initialize(workspace, "support", connection, spec, "Respond to the user.")
            events = [
                {"event_id": key, "trace_id": key, "session_id": key,
                 "kind": "model_call", "source": {"importer": "fixture"},
                 "payload": {"request": {"messages": [{"role": "user", "content": "Cancel my order."}]}}}
                for key in ("learning", "holdout")
            ]
            path = Path(folder) / "events.json"
            path.write_text(json.dumps(events))
            self.assertEqual(traces.ingest_events(workspace, "support", traces.read_events(path)),
                             {"added": 2, "duplicates": 0})
            self.assertEqual(traces.ingest_events(workspace, "support", events),
                             {"added": 0, "duplicates": 2})
            workspace.feedback("support", "learning", "Ask before cancelling.")
            project = traces.pin_chat_suite(workspace, "support", ["holdout"], expected_generation=0)
            optimizer = {**connection, "model": "optimizer"}

            def adapter(settings, **kwargs):
                return FixtureProvider(settings["model"])

            with patch("switchit.switchup.loop.model_adapter", side_effect=adapter):
                record = loop.run_round(workspace, "support", optimizer=optimizer,
                                        allow_live=True, max_model_calls=4)
            self.assertEqual(record["status"], "verified", record.get("decision"))
            self.assertFalse(any(e["trace_id"] == "holdout" for e in record["evidence"]))
            self.assertEqual(workspace.project("support")["active_recipe"], project["active_recipe"])
            active = loop.activate(workspace, "support", record["id"], expected_generation=record["generation"])
            self.assertIn("Always confirm", active["active_recipe"]["system_prompt"])
            rolled_back = workspace.rollback("support", active["generation"])
            self.assertEqual(rolled_back["active_recipe"], project["active_recipe"])
