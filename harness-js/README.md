# SwitchIt Node launchers

This npm package forwards `switchit` and `switchit-mcp` to the **single Python SwitchIt engine**. It contains no separate parser, evaluator, replay implementation, or model-provider SDK.

Requirements: Node.js 20+, Python 3.10+, and the Python `switchit-harness` package installed in the selected interpreter. There are no Node runtime dependencies. Installing this npm package does **not** install Python or the engine, and it makes no provider calls by itself.

## Upgrade to 0.2.0

**Breaking change from the 0.1.x npm package:** the independent TypeScript engine has been replaced by CLI/MCP launchers for the Python engine. Install or upgrade the Python engine before upgrading this package:

```sh
python3 -m pip install --upgrade "switchit-harness>=0.2.0"
npm install --global switchit-harness@latest
switchit --version
```

Use a Python virtual environment if your system Python requires it, and set `SWITCHIT_PYTHON` to its interpreter. Existing Python legacy replay commands remain available. New workflow contracts have their own schema and explicit review requirements; a legacy `workflows.json` grouping is not an executable workflow contract.

## Select the engine

Install the Python engine explicitly from a trusted release or a local checkout:

```sh
python3 -m pip install /path/to/SwitchIt
export SWITCHIT_PYTHON=/path/to/venv/bin/python
```

`SWITCHIT_PYTHON` is one executable name or path, not a shell command. Paths with spaces are supported. Without the variable, the bridge uses `python3` (`python` on Windows). The chosen interpreter must be able to import `switchit.cli` and `switchit.mcp_server`; activating a virtual environment also works.

From this checkout, no npm installation or transpilation is needed:

```sh
node bin/switchit.js --help
node bin/switchit.js run --demo --state-dir /tmp/switchit-demo
node bin/switchit-mcp.js
```

Once this package is installed, the equivalent commands are `switchit` and `switchit-mcp`. The Python and npm distributions expose the same command names; check your PATH if both are installed. Both routes ultimately execute Python.

## Forwarding contract

- CLI invokes `PYTHON -m switchit.cli` with the original argument array.
- MCP invokes `PYTHON -m switchit.mcp_server` with inherited stdin/stdout/stderr. No banner or diagnostics are written to protocol stdout by the bridge.
- The current directory and environment are preserved. Credentials remain an engine/runtime concern; the launcher does not read or persist them.
- Exit codes are preserved. SIGINT/SIGTERM (and SIGHUP on POSIX) are forwarded to the Python child. POSIX signal termination is propagated back to the launcher's caller.
- A silent, bounded import check runs before starting the selected entry point. Missing Python exits with code 127; an unavailable/incompatible engine exits with code 78 and an actionable stderr message.
- Nothing is installed, upgraded, downloaded, published, or sent to providers automatically.

All workflow, orchestration, and platform command availability comes from the installed Python engine. Use `switchit --help` and the relevant subcommand's `--help`; an older engine will not acquire newer capabilities merely by updating this launcher.

## Development and packaging

```sh
npm run build  # syntax checks; there is no transpiled duplicate engine
npm test       # argv/stdin/stdout/exit/signal forwarding and npm-pack checks
npm pack --ignore-scripts
```

The packed artifact contains `bin/`, `lib/`, the synchronized companion skill, and this README. It excludes tests, old TypeScript engine sources, generated state, and personal traces.

The repository's canonical companion skill is `skills/switchit/SKILL.md`; synchronize its copy here before a release. See the repository's `docs/distribution.md` for the three distinct distribution surfaces.
