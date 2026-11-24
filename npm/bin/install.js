#!/usr/bin/env node
/** Install the bundled skill only; never install or download a harness. */
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { randomUUID } = require("node:crypto");

function main(args = process.argv.slice(2)) {
  let destination = path.join(os.homedir(), ".agents", "skills", "switchit");
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--help" || args[index] === "-h") {
      console.log("Usage: switchit-skill [--dest DIRECTORY]\n\nInstalls SKILL.md directly inside DIRECTORY.\nDefault: ~/.agents/skills/switchit\nNo Python or package installation is performed.");
      return 0;
    }
    if (args[index] !== "--dest" || !args[index + 1] || args[index + 1].startsWith("--")) {
      console.error("switchit-skill: expected --dest DIRECTORY; use --help for usage.");
      return 2;
    }
    destination = path.resolve(args[++index]);
  }
  const source = path.join(__dirname, "..", "skills", "switchit", "SKILL.md");
  const target = path.join(destination, "SKILL.md");
  let temporary;
  try {
    const content = fs.readFileSync(source);
    fs.mkdirSync(destination, { recursive: true });
    if (fs.existsSync(target) && fs.readFileSync(target).equals(content)) {
      console.log(`The switchit skill is already up to date at ${target}`);
      return 0;
    }
    temporary = path.join(destination, `.SKILL.md.${randomUUID()}.tmp`);
    fs.writeFileSync(temporary, content, { flag: "wx", mode: 0o644 });
    fs.renameSync(temporary, target);
    temporary = undefined;
    console.log(`Installed the switchit skill to ${target}`);
    console.log("Invoke $switchit from a compatible agent. The Python harness is installed separately.");
    return 0;
  } catch (error) {
    console.error(`switchit-skill: ${error.message}`);
    return 1;
  } finally {
    if (temporary) fs.rmSync(temporary, { force: true });
  }
}

module.exports = { main };
if (require.main === module) process.exitCode = main();
