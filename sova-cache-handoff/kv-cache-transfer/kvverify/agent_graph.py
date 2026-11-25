"""Bounded async module execution with a durable SQLite artifact journal."""
import asyncio
import fcntl
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class RetryableError(RuntimeError):
    """A temporary failure where repeating the same inputs is safe."""


@dataclass(frozen=True)
class Node:
    name: str
    dependencies: tuple
    work: object
    max_attempts: int = 1


class GraphFailed(RuntimeError):
    pass


class Controller:
    def __init__(self, database, nodes, spec):
        self.database = Path(database)
        nodes = tuple(nodes)
        self.nodes = {node.name: node for node in nodes}
        self.spec = {"graph": [(n.name, n.dependencies, n.max_attempts) for n in nodes],
                     "inputs": spec}
        if len(self.nodes) != len(nodes) or not nodes:
            raise ValueError("Unique nonempty graph required")
        known = set()
        while len(known) < len(self.nodes):
            ready = {n.name for n in nodes if set(n.dependencies) <= known}
            if ready <= known:
                raise ValueError("Graph has an unknown dependency or cycle")
            known |= ready
        if any(node.max_attempts < 1 for node in nodes):
            raise ValueError("Positive attempt limits required")

    def _init_database(self, db):
        db.executescript("""
            CREATE TABLE IF NOT EXISTS run_spec (
                id INTEGER PRIMARY KEY CHECK(id = 1), value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                node TEXT NOT NULL, attempt INTEGER NOT NULL, inputs TEXT NOT NULL,
                status TEXT NOT NULL, output TEXT, output_sha TEXT, error TEXT,
                PRIMARY KEY (node, attempt)
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, node TEXT NOT NULL,
                attempt INTEGER NOT NULL, status TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        wanted = canonical(self.spec)
        with db:
            db.execute("INSERT OR IGNORE INTO run_spec VALUES (1, ?)", (wanted,))
            saved = db.execute("SELECT value FROM run_spec WHERE id = 1").fetchone()[0]
            if saved != wanted:
                raise ValueError("Run directory belongs to different input or code versions")
            interrupted = db.execute("SELECT node, attempt FROM attempts WHERE status = 'running'").fetchall()
            for node, attempt in interrupted:
                self._finish(db, node, attempt, "interrupted", error="Previous process stopped")

    @staticmethod
    def _finish(db, node, attempt, status, *, output=None, error=None):
        payload = canonical(output) if status == "passed" else None
        checksum = digest(output) if status == "passed" else None
        cursor = db.execute("UPDATE attempts SET status = ?, output = ?, output_sha = ?, error = ? "
                            "WHERE node = ? AND attempt = ? AND status = 'running'",
                            (status, payload, checksum, error, node, attempt))
        if cursor.rowcount != 1:
            raise RuntimeError("Attempt lost its ownership fence")
        db.execute("INSERT INTO events (node, attempt, status) VALUES (?, ?, ?)",
                   (node, attempt, status))

    def _restore(self, db):
        artifacts = {}
        passed = {row["node"]: row for row in db.execute("SELECT * FROM attempts WHERE status = 'passed'")}
        while passed:
            progressed = False
            for name, row in list(passed.items()):
                node = self.nodes[name]
                if not set(node.dependencies) <= artifacts.keys():
                    continue
                value = json.loads(row["output"])
                inputs = {dep: artifacts[dep][0] for dep in node.dependencies}
                if row["output_sha"] != digest(value) or row["inputs"] != canonical(inputs):
                    raise ValueError("Corrupt or mismatched cached artifact for " + name)
                artifacts[name] = (row["output_sha"], value)
                del passed[name]
                progressed = True
            if not progressed:
                raise ValueError("Persisted artifact dependencies are incomplete")
        return artifacts

    async def _execute(self, db):
        artifacts = self._restore(db)
        tasks = {}
        failed = {}
        while len(artifacts) < len(self.nodes):
            for name, node in self.nodes.items():
                if (name in artifacts or name in tasks or name in failed
                        or not set(node.dependencies) <= artifacts.keys()):
                    continue
                attempts = db.execute("SELECT attempt, status, error FROM attempts WHERE node = ? "
                                      "ORDER BY attempt", (name,)).fetchall()
                completed_failures = [row for row in attempts if row["status"] in ("failed", "retry")]
                if (any(row["status"] == "failed" for row in attempts)
                        or len(completed_failures) >= node.max_attempts):
                    failed[name] = completed_failures[-1]["error"]
                    continue
                if completed_failures:
                    await asyncio.sleep(min(.05 * 2**(len(completed_failures)-1), 1))
                index = attempts[-1]["attempt"] + 1 if attempts else 1
                values = {dep: json.loads(canonical(artifacts[dep][1])) for dep in node.dependencies}
                inputs = {dep: artifacts[dep][0] for dep in node.dependencies}
                with db:
                    db.execute("INSERT INTO attempts (node, attempt, inputs, status) "
                               "VALUES (?, ?, ?, 'running')", (name, index, canonical(inputs)))
                    db.execute("INSERT INTO events (node, attempt, status) VALUES (?, ?, 'running')",
                               (name, index))
                tasks[name] = (index, asyncio.create_task(asyncio.to_thread(node.work, values)))
            if not tasks:
                raise GraphFailed(canonical(failed))
            done, _ = await asyncio.wait([task for _, task in tasks.values()],
                                         return_when=asyncio.FIRST_COMPLETED)
            for name, (index, task) in list(tasks.items()):
                if task not in done:
                    continue
                del tasks[name]
                try:
                    output = task.result()
                    checksum = digest(output)
                except Exception as exc:
                    status = "retry" if isinstance(exc, RetryableError) else "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    with db:
                        self._finish(db, name, index, status, error=error)
                else:
                    with db:
                        self._finish(db, name, index, "passed", output=output)
                    artifacts[name] = (checksum, output)
        return {key: value for key, (_, value) in artifacts.items()}

    async def run(self):
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.database.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GraphFailed("Another process owns this local graph") from exc
            db = sqlite3.connect(self.database)
            db.row_factory = sqlite3.Row
            try:
                self._init_database(db)
                return await self._execute(db)
            finally:
                db.close()
