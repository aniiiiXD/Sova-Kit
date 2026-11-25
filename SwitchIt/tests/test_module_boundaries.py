"""Guard extraction boundaries and legacy imports without duplicating algorithms."""

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from switchit import agent_sessions, source_discovery
from switchit.importers import registry


class ModuleBoundaryTests(unittest.TestCase):
    def test_focused_modules_do_not_load_legacy_facades_or_execution(self):
        # Fresh interpreters reveal accidental eager imports hidden by test order.
        for module in (
            "source_discovery", "importers.codex", "importers.claude_code",
            "hierarchy", "workflow_discovery", "evaluation",
        ):
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-c",
                     "import importlib, json, sys; "
                     "importlib.import_module(sys.argv[1]); "
                     "print(json.dumps(sorted(sys.modules)))",
                     f"switchit.{module}"],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True, text=True, check=True,
                )
                loaded = set(json.loads(completed.stdout))
                self.assertFalse(loaded.intersection({
                    "switchit.core", "switchit.runner", "switchit.agent_sessions",
                    "switchit.cli", "switchit.mcp_server", "switchit.providers",
                    "switchit.replay", "switchit.reporting",
                }))

    def test_implementations_do_not_depend_on_compatibility_facades(self):
        package = Path(__file__).resolve().parents[1] / "switchit"
        forbidden = {"core", "runner", "agent_sessions"}
        for path in package.rglob("*.py"):
            if path.stem in forbidden or path.name == "__init__.py":
                continue
            with self.subTest(module=path.relative_to(package)):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotIn(alias.name, {f"switchit.{n}" for n in forbidden})
                    elif isinstance(node, ast.ImportFrom):
                        self.assertNotIn(node.module, {f"switchit.{n}" for n in forbidden})
                        if node.module == "switchit" or (node.level and node.module is None):
                            self.assertTrue(forbidden.isdisjoint(a.name for a in node.names))

    def test_legacy_function_imports_resolve_to_owning_modules(self):
        for facade, owner, name in (
            ("core", "normalization", "make_call"),
            ("core", "hierarchy", "build_traces"),
            ("core", "workflow_discovery", "group_workflows"),
            ("core", "cases", "select_cases"),
            ("agent_sessions", "importers.codex", "import_codex"),
            ("agent_sessions", "importers.muse", "import_muse_tree"),
            ("runner", "evaluation", "evaluate_check"),
            ("runner", "replay", "run_replay"),
            ("runner", "reporting", "write_report"),
        ):
            with self.subTest(name=name):
                old = importlib.import_module(f"switchit.{facade}")
                new = importlib.import_module(f"switchit.{owner}")
                self.assertIs(getattr(old, name), getattr(new, name))

    def test_legacy_discovery_override_does_not_mutate_new_module(self):
        original = source_discovery.AGENT_DEFAULTS
        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder) / "session.jsonl"
            session.write_text("{}\n")
            defaults = {"claude-code": [Path(folder)]}
            with patch.object(agent_sessions, "AGENT_DEFAULTS", defaults):
                self.assertEqual(agent_sessions.discover_agent_sessions(), [
                    {"agent": "claude-code", "path": str(session)},
                ])
                self.assertEqual(
                    agent_sessions.discover_agent_sessions(),
                    source_discovery.discover_agent_sessions(defaults=defaults),
                )
            self.assertIs(source_discovery.AGENT_DEFAULTS, original)

    def test_legacy_importer_override_does_not_mutate_new_registry(self):
        original = registry.IMPORTERS_BY_AGENT
        fixture_importers = {"fixture": lambda path: [{"source": path}]}
        with patch.object(agent_sessions, "IMPORTERS_BY_AGENT", fixture_importers):
            self.assertEqual(
                agent_sessions.import_agent_sessions([{"agent": "fixture", "path": "fixture.jsonl"}]),
                [{"source": "fixture.jsonl"}],
            )
        self.assertIs(registry.IMPORTERS_BY_AGENT, original)


if __name__ == "__main__":
    unittest.main()
