"""Graph regressions for convergence, recovery and artifact versioning."""
import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
from threading import Barrier
import unittest

from kvverify.agent_graph import Controller, GraphFailed, Node, RetryableError, canonical


class ControllerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "graph.sqlite3"

    def run_graph(self, nodes, spec=None):
        return asyncio.run(Controller(self.database, nodes, spec or {"version": 1}).run())

    def test_independent_modules_overlap_and_resume_from_verified_artifacts(self):
        fork = Barrier(2, timeout=2)

        def work(_):
            fork.wait()
            return 5

        graph = [Node("a", (), work), Node("b", (), work),
                 Node("join", ("a", "b"), lambda inputs: inputs["a"] + inputs["b"])]
        self.assertEqual(self.run_graph(graph)["join"], 10)
        # A resumed controller must not call either successful node again.
        def must_not_run(_):
            self.fail("completed node was started again")
        restored = [Node(node.name, node.dependencies, must_not_run) for node in graph]
        self.assertEqual(self.run_graph(restored)["join"], 10)

    def test_transient_retry_can_advance_but_permanent_error_blocks_its_dependents(self):
        calls = []

        def flaky(_):
            calls.append(1)
            if len(calls) == 1:
                raise RetryableError("transient local error")
            return {"passed": True}

        def fail(_):
            raise AssertionError("gate failed")

        def must_not_run(_):
            self.fail("dependency failure was ignored")

        graph = [Node("flaky", (), flaky, max_attempts=2),
                 Node("gate", ("flaky",), fail),
                 Node("dependent", ("gate",), must_not_run)]
        with self.assertRaisesRegex(GraphFailed, "gate failed"):
            self.run_graph(graph)
        with self.assertRaisesRegex(GraphFailed, "gate failed"):
            self.run_graph(graph)
        with sqlite3.connect(self.database) as db:
            states = db.execute("SELECT node, status FROM attempts ORDER BY rowid").fetchall()
        self.assertEqual(states, [("flaky", "retry"), ("flaky", "passed"), ("gate", "failed")])
        self.assertEqual(len(calls), 2)

    def test_restart_reclaims_interrupted_attempt_and_fences_old_publication(self):
        graph = [Node("a", (), lambda _: 1)]
        controller = Controller(self.database, graph, {"version": 1})
        with sqlite3.connect(self.database) as db:
            controller._init_database(db)
            db.execute("INSERT INTO attempts (node, attempt, inputs, status) VALUES (?, ?, ?, ?)",
                       ("a", 1, canonical({}), "running"))
        self.assertEqual(self.run_graph(graph), {"a": 1})
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("SELECT attempt, status FROM attempts ORDER BY attempt").fetchall(),
                             [(1, "interrupted"), (2, "passed")])
            with self.assertRaisesRegex(RuntimeError, "ownership fence"):
                controller._finish(db, "a", 1, "passed", output=999)

    def test_mismatched_spec_or_artifact_cannot_resume(self):
        graph = [Node("a", (), lambda _: {"version": 1})]
        self.run_graph(graph)
        with self.assertRaisesRegex(ValueError, "different input"):
            self.run_graph(graph, {"version": 2})
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE attempts SET output = ? WHERE node = 'a'", (json.dumps({"version": 2}),))
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            self.run_graph(graph)


if __name__ == "__main__":
    unittest.main()
