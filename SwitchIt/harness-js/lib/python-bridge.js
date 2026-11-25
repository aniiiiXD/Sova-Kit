import { spawn, spawnSync } from "node:child_process";
import { constants } from "node:os";

const MODULES = new Set(["switchit.cli", "switchit.mcp_server"]);

function engineError(python, reason) {
  process.stderr.write(
    `switchit: Python engine unavailable using ${JSON.stringify(python)}.\n` +
    `${reason}\n` +
    "Install the Python engine explicitly from your SwitchIt checkout or a trusted package release.\n" +
    "For a checkout: python3 -m pip install /path/to/SwitchIt\n" +
    "Set SWITCHIT_PYTHON to that Python 3.10+ executable, for example /path/to/venv/bin/python.\n" +
    "This launcher never installs Python or packages automatically.\n",
  );
}

/** Forward one existing CLI surface to the single Python implementation. */
export function launchPython(moduleName, args = process.argv.slice(2)) {
  if (!MODULES.has(moduleName)) {
    throw new Error(`Unsupported Python entry point: ${moduleName}`);
  }
  const python = process.env.SWITCHIT_PYTHON || (process.platform === "win32" ? "python" : "python3");
  // Do not use a shell or split SWITCHIT_PYTHON: paths containing spaces remain one executable.
  // Import under the module's normal name so its __main__ guard does not consume stdin.
  const probe = spawnSync(
    python,
    ["-c", `import sys, importlib; sys.exit(10) if sys.version_info < (3, 10) else importlib.import_module(${JSON.stringify(moduleName)})`],
    { stdio: ["ignore", "ignore", "pipe"], timeout: 10_000, maxBuffer: 64 * 1024, windowsHide: true },
  );
  if (probe.error || probe.status !== 0) {
    const missing = probe.error?.code === "ENOENT";
    const reason = missing ? "The selected Python executable could not be found." :
      probe.status === 10 ? "The selected interpreter is older than Python 3.10." :
      probe.error?.code === "ETIMEDOUT" ? "The Python engine availability check timed out." :
      `The selected interpreter could not import ${moduleName}.`;
    engineError(python, reason);
    return Promise.resolve({ code: missing ? 127 : 78, signal: null });
  }
  return new Promise((resolve) => {
    const child = spawn(python, ["-m", moduleName, ...args], {
      stdio: "inherit", shell: false, windowsHide: true,
    });
    const handlers = new Map();
    const signals = process.platform === "win32" ? ["SIGINT", "SIGTERM"] : ["SIGINT", "SIGTERM", "SIGHUP"];
    for (const signal of signals) {
      const handler = () => { if (child.exitCode === null && child.signalCode === null) child.kill(signal); };
      handlers.set(signal, handler);
      process.on(signal, handler);
    }
    function cleanup() {
      for (const [signal, handler] of handlers) process.removeListener(signal, handler);
    }
    child.once("error", (error) => {
      cleanup();
      engineError(python, `Could not start the Python engine (${error.code || "spawn error"}).`);
      resolve({ code: error.code === "ENOENT" ? 127 : 78, signal: null });
    });
    child.once("exit", (code, signal) => {
      cleanup();
      resolve({ code, signal });
    });
  });
}

/** Keep both ordinary exit codes and POSIX signal termination visible to callers. */
export function exitLikePython({ code, signal }) {
  if (signal) {
    process.exitCode = 128 + (constants.signals[signal] || 1);
    if (process.platform !== "win32") {
      process.kill(process.pid, signal);
    }
    return;
  }
  process.exitCode = code ?? 1;
}
