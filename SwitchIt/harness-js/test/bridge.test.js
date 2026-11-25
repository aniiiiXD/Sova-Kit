import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const cli = path.join(packageRoot, "bin", "switchit.js");
const mcp = path.join(packageRoot, "bin", "switchit-mcp.js");
const python = process.env.SWITCHIT_TEST_PYTHON || "python3";

function fixture(t, body, moduleName = "cli") {
  const root = mkdtempSync(path.join(tmpdir(), "switchit-node-bridge-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(path.join(root, "switchit"));
  writeFileSync(path.join(root, "switchit", "__init__.py"), "");
  writeFileSync(path.join(root, "switchit", `${moduleName}.py`),
    "import json,sys,signal,time\nif __name__ == '__main__':\n" +
    body.split("\n").map(line => "    " + line).join("\n") + "\n");
  return { root, env: { ...process.env, SWITCHIT_PYTHON: python, PYTHONPATH: root, PYTHONNOUSERSITE: "1" } };
}

function run(entry, args, { root, env }, input) {
  return spawnSync(process.execPath, [entry, ...args], {
    cwd: root, env, input, encoding: "utf8", timeout: 15_000,
  });
}

test("CLI forwards the exact argument array without shell interpretation", t => {
  const setup = fixture(t, "print(json.dumps(sys.argv[1:]))");
  const args = ["workflow", "build", "--events", "file with spaces.json", "$(never-run)", "a; b", "--output", "path!"];
  const result = run(cli, args, setup);
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), args);
  assert.equal(result.stderr, "");
});

test("MCP preserves stdin/stdout bytes and keeps diagnostics on stderr", t => {
  const setup = fixture(t, 'data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data); print("engine diagnostic",file=sys.stderr)', "mcp_server");
  const input = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"text":"hello 🌍"}}\n';
  const result = run(mcp, [], setup, input);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, input);
  assert.equal(result.stderr, "engine diagnostic\n");
});

test("engine exit code is preserved", t => {
  const setup = fixture(t, 'print("completed"); sys.exit(23)');
  const result = run(cli, [], setup);
  assert.equal(result.status, 23);
  assert.equal(result.stdout, "completed\n");
});

test("missing interpreter gives actionable stderr without touching stdout", t => {
  const setup = fixture(t, "pass");
  setup.env.SWITCHIT_PYTHON = path.join(setup.root, "missing-python");
  const result = run(cli, [], setup);
  assert.equal(result.status, 127);
  assert.equal(result.stdout, "");
  assert.match(result.stderr, /SWITCHIT_PYTHON/);
  assert.match(result.stderr, /never installs/);
});

test("interpreter without an importable engine exits with a configuration error", t => {
  const setup = fixture(t, "pass");
  // Node is deliberately not a Python interpreter; the import probe must fail cleanly.
  setup.env.SWITCHIT_PYTHON = process.execPath;
  const result = run(cli, [], setup);
  assert.equal(result.status, 78);
  assert.equal(result.stdout, "");
  assert.match(result.stderr, /could not import switchit.cli/);
});

test("SIGTERM reaches the engine and its handled exit code is preserved", { skip: process.platform === "win32", timeout: 10_000 }, async t => {
  const setup = fixture(t, [
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(42))",
    "signal.signal(signal.SIGALRM, lambda *_: sys.exit(90))",
    "signal.alarm(5)",
    "print('ready', flush=True)",
    "while True: time.sleep(0.05)",
  ].join("\n"));
  const child = spawn(process.execPath, [cli], { cwd: setup.root, env: setup.env, stdio: ["ignore", "pipe", "pipe"] });
  t.after(() => { if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL"); });
  const outcome = new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => resolve({ code, signal }));
  });
  await new Promise((resolve, reject) => {
    let output = "";
    child.stdout.on("data", chunk => {
      output += chunk.toString();
      if (output.includes("ready\n")) resolve();
    });
    child.once("error", reject);
    child.once("exit", code => { if (!output.includes("ready\n")) reject(new Error(`engine exited before ready: ${code}`)); });
  });
  child.kill("SIGTERM");
  assert.deepEqual(await outcome, { code: 42, signal: null });
});

test("unhandled Python signal termination propagates to the launcher caller", { skip: process.platform === "win32" }, t => {
  const setup = fixture(t, "import os; os.kill(os.getpid(), signal.SIGTERM)");
  const result = run(cli, [], setup);
  assert.equal(result.status, null);
  assert.equal(result.signal, "SIGTERM");
});

test("npm tarball contains only launchers, their library, skill and metadata", t => {
  const destination = mkdtempSync(path.join(tmpdir(), "switchit-node-pack-"));
  t.after(() => rmSync(destination, { recursive: true, force: true }));
  const npm = process.platform === "win32" ? "npm.cmd" : "npm";
  const packed = spawnSync(npm, ["pack", "--json", "--ignore-scripts", "--pack-destination", destination], {
    cwd: packageRoot, encoding: "utf8", timeout: 20_000,
    env: { ...process.env, npm_config_cache: path.join(destination, "cache"), npm_config_offline: "true" },
  });
  assert.equal(packed.status, 0, packed.stderr);
  const [artifact] = JSON.parse(packed.stdout);
  const files = artifact.files.map(item => item.path);
  assert.ok(files.includes("bin/switchit.js"));
  assert.ok(files.includes("bin/switchit-mcp.js"));
  assert.ok(files.includes("lib/python-bridge.js"));
  assert.ok(files.includes("skills/switchit/SKILL.md"));
  assert.ok(files.every(name => !name.startsWith("src/") && !name.startsWith("dist/") && !name.startsWith("test/")));
  assert.ok(readFileSync(path.join(destination, artifact.filename)).length > 0);
});
