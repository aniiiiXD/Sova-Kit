const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const packageRoot = path.resolve(__dirname, "..");
const installer = path.join(packageRoot, "bin", "install.js");
const source = fs.readFileSync(path.join(packageRoot, "skills", "switchit", "SKILL.md"));

function temporary(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "switchit-skill-test-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}

test("explicit destination gets the exact bundled skill and repeat install is unchanged", t => {
  const root = temporary(t);
  const destination = path.join(root, "isolated skills", "switchit");
  const result = spawnSync(process.execPath, [installer, "--dest", destination], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(fs.readFileSync(path.join(destination, "SKILL.md")), source);
  const before = fs.statSync(path.join(destination, "SKILL.md")).mtimeMs;
  const again = spawnSync(process.execPath, [installer, "--dest", destination], { encoding: "utf8" });
  assert.equal(again.status, 0, again.stderr);
  assert.match(again.stdout, /already up to date/);
  assert.equal(fs.statSync(path.join(destination, "SKILL.md")).mtimeMs, before);
  assert.deepEqual(fs.readdirSync(destination), ["SKILL.md"]);
});

test("installer updates an old skill atomically without leaving temporary files", t => {
  const root = temporary(t);
  fs.writeFileSync(path.join(root, "SKILL.md"), "old skill");
  const result = spawnSync(process.execPath, [installer, "--dest", root], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(fs.readFileSync(path.join(root, "SKILL.md")), source);
  assert.deepEqual(fs.readdirSync(root), ["SKILL.md"]);
});

test("missing or unknown options fail without writing a skill", t => {
  const root = temporary(t);
  for (const args of [["--dest"], ["--unknown"], ["--dest", "--help"]]) {
    const result = spawnSync(process.execPath, [installer, ...args], { cwd: root, encoding: "utf8" });
    assert.equal(result.status, 2);
  }
  assert.deepEqual(fs.readdirSync(root), []);
});

test("an invalid destination reports a failure", t => {
  const root = temporary(t);
  const destination = path.join(root, "a-file");
  fs.writeFileSync(destination, "not a directory");
  const result = spawnSync(process.execPath, [installer, "--dest", destination], { encoding: "utf8" });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /switchit-skill:/);
  assert.equal(fs.readFileSync(destination, "utf8"), "not a directory");
});

test("npm tarball contains installer and bundled skill without personal data or tests", t => {
  const root = temporary(t);
  const npm = process.platform === "win32" ? "npm.cmd" : "npm";
  const result = spawnSync(npm, ["pack", "--json", "--ignore-scripts", "--pack-destination", root], {
    cwd: packageRoot, encoding: "utf8", timeout: 20_000,
    env: { ...process.env, npm_config_cache: path.join(root, "cache"), npm_config_offline: "true" },
  });
  assert.equal(result.status, 0, result.stderr);
  const [artifact] = JSON.parse(result.stdout);
  const files = artifact.files.map(item => item.path);
  assert.ok(files.includes("bin/install.js"));
  assert.ok(files.includes("skills/switchit/SKILL.md"));
  assert.ok(files.every(name => /^(bin\/|skills\/|README\.md$|package\.json$|LICENSE$)/.test(name)));
});
