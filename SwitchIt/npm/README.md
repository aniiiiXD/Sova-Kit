# switchit-skill

An installer for the **$switchit** companion agent skill. The skill guides trace discovery, workflow reconstruction, fixed-check evaluation, authorized orchestration, and reviewed workflow export.

The installer only copies its bundled `SKILL.md`. It does not install Python, the SwitchIt engine, npm dependencies, or native agent integrations.

## Install

Install or refresh the companion skill and its separately installed engine:

```sh
python3 -m pip install --upgrade "switchit-harness>=0.2.0"
npx switchit-skill@latest
```

From this checkout:

```sh
node npm/bin/install.js
node npm/bin/install.js --dest /tmp/switchit-skill-check
```

The default destination is `~/.agents/skills/switchit`. `--dest DIRECTORY` installs `SKILL.md` directly inside that exact directory, allowing isolated installation tests. Repeating an installation with identical content is a no-op; updates replace the skill atomically.

After installing this package, use `switchit-skill` with the same arguments. Invoke `$switchit` in an agent that supports skills from the chosen location. The installer does not configure an agent's search paths.

## Engine and launchers

The Python engine is installed separately, for example from a local checkout:

```sh
python3 -m pip install /path/to/SwitchIt
python3 -m switchit.cli run --demo
```

The npm package named `switchit-harness` is a thin Node launcher for that Python engine. It preserves the `switchit` and `switchit-mcp` command names and supports explicit interpreter selection through `SWITCHIT_PYTHON`. It is different from this skill installer.

The skill documents newer `workflow`, `orchestrate`, and `serve` commands. Their availability depends on the Python engine version, not this installer version. Verify `python3 -m switchit.cli --help` before use. Use engine 0.2.0 or later for the commands in this skill.

## Development

```sh
npm --prefix npm test
npm --prefix npm pack --ignore-scripts
```

The canonical source is `skills/switchit/SKILL.md` at the repository root. Its copies in this package and `harness-js/skills/` must match before distribution. Tests cover explicit destinations, repeat installs, error handling, and package contents.

Repository: [aniiiiXD/SwitchIt](https://github.com/aniiiiXD/SwitchIt)

MIT license.
