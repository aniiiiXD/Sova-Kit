# Distribution surfaces

SwitchIt has one Python execution engine and separate delivery surfaces. The Node package now launches Python; the duplicate TypeScript engine has been removed from `harness-js/`.

| Artifact | Source | Responsibility | Runtime requirements |
| --- | --- | --- | --- |
| Python harness | `pyproject.toml`, `switchit/` | Canonical import, workflow, execution, evaluation, CLI/MCP and platform code | Python 3.10+; optional extras for selected providers/formats |
| Node launcher | `harness-js/` (`switchit-harness`) | Forward `switchit` and `switchit-mcp` to the installed Python engine | Node 20+, separately installed Python engine |
| Companion skill installer | `npm/` (`switchit-skill`) | Copy the bundled general SwitchIt skill into a supported agent's skill directory | Node 18+; engine installed separately |
| Canonical companion skill | `skills/switchit/SKILL.md` | Guide discovery, workflow review, verification and authorized orchestration | A compatible host agent and harness access |
| Exported workflow skill | `workflow export-skill` output | Package one reviewed workflow's instructions/artifacts | Requirements declared by that workflow |
| Local platform | `switchit serve` | Expose the engine's local platform interface | Installed Python engine; not supplied by the Node launcher alone |

The static `marketing-next/` site is a separate marketing distribution. A successful marketing deployment is not evidence that a workflow execution service is running.

## Node bridge contract

The bridge forwards the original argument array, current directory, environment, stdin, stdout, stderr, exit code, and termination signals. It does not parse the Python command's options and therefore does not need its own copy of workflow or orchestration logic.

`SWITCHIT_PYTHON` selects a single executable name or path. It is never split as a shell command. The default is `python3` on POSIX and `python` on Windows. A silent, bounded preflight checks the Python version and imports the selected engine entry point; this does not consume input or write to MCP stdout. Missing interpreters exit 127; unavailable/incompatible engines exit 78 with installation guidance on stderr.

No automatic engine/package installation, package download, provider request, or publication is performed. An older Python engine remains older even when the launcher package is updated; inspect that engine's `--help` output for supported namespaces.

## Skill synchronization and isolated installation

The canonical skill must match both packaged copies byte-for-byte:

```text
skills/switchit/SKILL.md
npm/skills/switchit/SKILL.md
harness-js/skills/switchit/SKILL.md
```

The source and npm installers support explicit destinations. The destination is the exact directory containing `SKILL.md`, not a parent to which another `switchit` component is appended:

```sh
./scripts/install-skill.sh --dest /tmp/switchit-shell-skill
node npm/bin/install.js --dest /tmp/switchit-node-skill
```

Without `--dest`, both retain the existing default `~/.agents/skills/switchit`. Repeated installs with identical content are no-ops. Updates use a temporary file and atomic replacement. Installers do not modify an agent's configuration or install the Python engine.

## CLI namespace in the companion skill

The synchronized skill documents these engine interfaces:

```sh
python3 -m switchit.cli workflow build --events events.json --output ./build
python3 -m switchit.cli workflow run --workflow workflow.json --case case.json --output run.json --runtime fixture
python3 -m switchit.cli workflow export-skill --workflow workflow.json --output ./exported-skill
python3 -m switchit.cli orchestrate run --plan plan.json --output ./orchestration
python3 -m switchit.cli orchestrate run --plan plan.json --output ./orchestration --resume
python3 -m switchit.cli orchestrate status --output ./orchestration
python3 -m switchit.cli orchestrate cancel --output ./orchestration
python3 -m switchit.cli serve --host 127.0.0.1 --port 8787 --data-dir ./platform-data
```

Fixture execution is the documented default. Provider/command runtime details and authorization flags come from the engine's help output. The skill reuses existing session/plan authorization rather than asking for it again, while respecting explicit no-push/no-publish instructions.

## Verification and release inputs

```sh
npm --prefix harness-js run build
npm --prefix harness-js test
npm --prefix npm test
python3 -m unittest tests.test_packaging
```

Tests use synthetic Python entry points and temporary skill destinations. They verify exact argument forwarding, MCP byte preservation, engine exit codes, missing-engine errors, POSIX signal forwarding, installer idempotency/errors, synchronized skill copies, and `npm pack` contents. Packaging runs offline with scripts disabled and writes tarballs to temporary directories.

The Node launcher tarball must include its bins, bridge library, skill, and metadata. It must exclude the retired TypeScript implementation, generated replay state, tests, and personal traces. The skill installer tarball contains its installer, skill, and metadata only.

Before a real release, the release controller must additionally verify the actual installed Python engine through the launcher, the newly documented CLI interfaces, version metadata consistency, and the intended publication/deployment destinations. Package names and local source metadata are not proof of publication. Changes here are prepared locally; these instructions do not perform or authorize a release.
