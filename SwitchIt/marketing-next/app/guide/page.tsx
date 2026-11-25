import type { Metadata } from "next";
export const metadata: Metadata = { title: "Getting started | SwitchUP" };
export default function Guide() {
  return (
    <main id="main-content" className="guide">
      <a className="guide-back" href="/">
        ← Back to SwitchUP
      </a>
      <h1>Get started with SwitchUP</h1>
      <p className="intro">
        Connect a model and define the outcomes its instruction updates must pass.
      </p>
      <section id="switchup">
        <h2>Create an improvement project</h2>
        <p>
          Install the Python package from the repository with <code>pip install .</code>.
          Register a model connection and reviewed checks, import traces, and pin a replay suite.
        </p>
        <pre><code>{['switchup improve init --help', 'switchup serve'].join('\n')}</code></pre>
        <p>
          Open <code>http://127.0.0.1:8787/improvements</code> to review instruction changes,
          replay evidence, and the active version. The repository guide in <code>docs/switchup.md</code>
          describes capture, proposal, verification, activation, and rollback.
        </p>
      </section>
      <section>
        <h2>Install the replay harness</h2>
        <p>
          Install the Python harness first (Python 3.10+). The optional npm
          package provides CLI and MCP launchers and requires Node 20+ plus the
          Python harness. A virtual environment keeps the Python install
          separate from your other projects.
        </p>
        <pre>
          <code>
            {[
              "# Python harness — install or upgrade from PyPI",
              'pip install --upgrade "switchit-harness>=0.2.0"',
              "",
              "# Optional Node CLI bridge — requires the Python harness",
              "npm install --global switchit-harness@latest",
              "",
              "# Agent skill — one command, any compatible agent",
              "npx switchit-skill@latest",
            ].join("\n")}
          </code>
        </pre>
        <p>
          Version 0.2.0 replaces the standalone npm engine with a Python bridge.
          If you used npm 0.1.x, install the Python engine before upgrading. Set{" "}
          <code>SWITCHIT_PYTHON</code> to your virtual environment’s Python
          executable when needed. The skill installer requires Node 18+.
        </p>
        <p>
          Building from source? Run <code>pip install .</code> from the
          repository root. For the optional Node bridge, run{" "}
          <code>npm install &amp;&amp; npm run build</code> inside{" "}
          <code>harness-js/</code>.
        </p>
      </section>
      <section>
        <h2>Run the offline demo</h2>
        <pre>
          <code>switchit run --demo --state-dir .switchit-demo</code>
        </pre>
        <p>
          Open <code>.switchit-demo/run/report.html</code> in your browser. The
          bundled traces and mock target let you explore the workflow without
          sending data to a provider. The report includes passes, failures,
          errors, and uncertainties.
        </p>
      </section>
      <section>
        <h2>Bring your own workflows</h2>
        <p>
          Import authorized local coding-agent sessions, then review the
          workflow map and define success checks before replaying.
        </p>
        <pre>
          <code>
            {[
              "switchit import --agents --state-dir .switchit",
              "switchit profile --state-dir .switchit",
              "switchit map --state-dir .switchit",
              "switchit cases --state-dir .switchit",
              "# Review workflows.json and cases.json in .switchit first.",
              "switchit replay --state-dir .switchit",
              "switchit report --state-dir .switchit",
            ].join("\n")}
          </code>
        </pre>
        <p>
          Local imports support Codex, Claude Code, OpenCode, Cursor, Cline,
          Gemini CLI, and Copilot CLI. Exported JSON/JSONL, OTLP, Langfuse, and
          Parquet are supported too; Parquet needs the optional dependency.
        </p>
      </section>
      <section>
        <h2>Choose a live target when you’re ready</h2>
        <p>
          Live replay requires explicit opt-in with <code>--remote</code>. It
          sends the redacted cases to your chosen provider and can incur usage
          charges. Supply credentials through environment variables.
          OpenAI-compatible APIs, Anthropic Messages, and Bedrock Converse are
          supported, along with local Ollama and vLLM targets.
        </p>
        <p>
          The baseline comes from recorded history. Provider replay compares
          tool calls against historical fixtures and does not execute your live
          business tools. A passing check only establishes the behavior that
          check measures. Separately configured command runtimes require
          explicit execution authorization.
        </p>
      </section>
      <section>
        <h2>Let your agent drive</h2>
        <p>
          Register the built-in MCP server in your agent’s configuration.
          Version 0.2.0 exposes the original import, profile, map, cases, replay
          and report tools alongside workflow, source and harness operations.
          Inspect the server’s advertised tools for the installed version.
        </p>
        <pre>
          <code>
            {JSON.stringify(
              {
                mcpServers: { switchit: { command: "switchit-mcp", args: [] } },
              },
              null,
              2,
            )}
          </code>
        </pre>
        <p>
          The companion <code>$switchit</code> skill guides the review of
          workflows, success checks, and migration findings.
        </p>
      </section>
    </main>
  );
}
