"use client";
import { useEffect, useState } from "react";
import styles from "./product-story.module.css";

const steps = [
  {
    title: "Capture",
    heading: "Start with what actually happened.",
    description:
      "Import recorded agent sessions. Keep the prompts, tool calls, results and source relationships that the trace provides.",
    file: "recorded-session.jsonl",
    nodes: [
      ["User request", "Find the customer’s latest order"],
      ["Tool call", "lookup_order(customer_id)"],
      ["Recorded result", "Order #1042 · dispatched"],
      ["Agent response", "Return the order and next step"],
    ],
    note: "Recorded execution · illustrative trace",
    detail: "source: codex  /  session: sample-042",
  },
  {
    title: "Reconstruct",
    heading: "Give the work a structure.",
    description:
      "Turn the captured events into a graph. Separate observed relationships from inferred ordering, and keep missing evidence visible.",
    file: "execution-graph.json",
    nodes: [
      ["Task", "Find the latest order"],
      ["Dependency", "Account lookup → order lookup"],
      ["Workflow candidate", "Resolve a customer order"],
      ["Evidence gap", "Unrecorded tool state stays unknown"],
    ],
    note: "A candidate to inspect, not an automatic approval",
    detail: "relationships: recorded + inferred",
  },
  {
    title: "Review",
    heading: "Decide what must stay true.",
    description:
      "Review the workflow, bind its tools and define the outcomes that matter. A plausible answer alone is not a passing result.",
    file: "reviewed-workflow.json",
    nodes: [
      ["Input contract", "A valid customer identifier"],
      ["Tool scope", "lookup_account · lookup_order"],
      ["Required check", "Order belongs to the customer"],
      ["Required check", "Answer includes the delivery status"],
    ],
    note: "Human-reviewed contracts and explicit checks",
    detail: "criteria: fixed before comparison",
  },
  {
    title: "Replay",
    heading: "Take the same work through a new gate.",
    description:
      "Run selected cases against the target runtime. Compare the observed outcome with the checks, then inspect every difference.",
    file: "migration-report.json",
    nodes: [
      ["Reference", "Recorded source behaviour"],
      ["Target", "Your selected provider"],
      ["Comparison", "Tools · outputs · required checks"],
      ["Verdict", "Pass · fail · error · uncertain"],
    ],
    note: "Offline fixtures first. Live providers are opt-in.",
    detail: "a pass covers only the checks you defined",
  },
];

export function SourceStrip() {
  return (
    <section
      id="sources"
      className={styles.sources}
      aria-labelledby="sources-heading"
    >
      <div className={styles.inner}>
        <div className={styles.sourceIntro}>
          <h2 id="sources-heading">Begin where your work lives.</h2>
          <span>Import recorded sessions</span>
        </div>
        <div className={styles.sourceNames}>
          {[
            "Codex",
            "Claude Code",
            "Cursor",
            "OpenCode",
            "Gemini CLI",
            "Copilot CLI",
          ].map((name) => (
            <span key={name}>{name}</span>
          ))}
        </div>
        <p className={styles.sourceNote}>
          Also bring Aegis task history and GitHub activity. Metadata enriches
          the evidence; executable replay needs model and tool traces.
        </p>
      </div>
    </section>
  );
}

export function WorkflowSection() {
  const [active, setActive] = useState(0);
  const [compact, setCompact] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(max-width:700px)");
    const update = () => setCompact(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  const step = steps[active];
  return (
    <section id="how" className={styles.workflow}>
      <div className={styles.inner}>
        <div className={styles.workflowHead}>
          <p className="section-label">The shape of the work</p>
          <h2>
            A conversation ends.
            <br />
            <em>A workflow continues.</em>
          </h2>
          <p>
            The useful part is the sequence: what was asked, which tools were
            used, and what made the result correct.
          </p>
        </div>
        <div className={styles.explorer}>
          <div
            className={styles.tabs}
            role="tablist"
            aria-label="Workflow stages"
            aria-orientation={compact ? "horizontal" : "vertical"}
          >
            {steps.map((item, index) => (
              <button
                key={item.title}
                type="button"
                role="tab"
                id={`workflow-tab-${index}`}
                aria-selected={active === index}
                aria-controls="workflow-panel"
                tabIndex={active === index ? 0 : -1}
                onClick={() => setActive(index)}
                onKeyDown={(event) => {
                  let next = index;
                  if (["ArrowDown", "ArrowRight"].includes(event.key))
                    next = (index + 1) % steps.length;
                  else if (["ArrowUp", "ArrowLeft"].includes(event.key))
                    next = (index + steps.length - 1) % steps.length;
                  else if (event.key === "Home") next = 0;
                  else if (event.key === "End") next = steps.length - 1;
                  else return;
                  event.preventDefault();
                  setActive(next);
                  document.getElementById(`workflow-tab-${next}`)?.focus();
                }}
              >
                <span className={styles.stepNumber}>0{index + 1}</span>
                <span>
                  <strong>{item.title}</strong>
                  <small>
                    {
                      [
                        "Bring the evidence",
                        "Find the relationships",
                        "Define a good outcome",
                        "Compare what survives",
                      ][index]
                    }
                  </small>
                </span>
                <span className={styles.tabArrow} aria-hidden="true">
                  ↗
                </span>
              </button>
            ))}
          </div>
          <div
            id="workflow-panel"
            className={styles.explorerBody}
            role="tabpanel"
            aria-labelledby={`workflow-tab-${active}`}
            tabIndex={0}
          >
            <div className={styles.stepCopy}>
              <h3>{step.heading}</h3>
              <p>{step.description}</p>
            </div>
            <div className={styles.traceVisual}>
              <div className={styles.traceTop}>
                <span>{step.file}</span>
                <span>Illustrative sample</span>
              </div>
              <ol className={styles.traceNodes}>
                {step.nodes.map(([label, content], index) => (
                  <li key={`${index}-${label}`}>
                    <span className={styles.nodeMark} aria-hidden="true">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <span>{label}</span>
                      <strong>{content}</strong>
                    </div>
                    {index === step.nodes.length - 1 && (
                      <span className={styles.endMark} aria-hidden="true">
                        ↗
                      </span>
                    )}
                  </li>
                ))}
              </ol>
              <div className={styles.traceFoot}>
                <code>{step.detail}</code>
              </div>
            </div>
            <p className={styles.workflowNote}>{step.note}</p>
          </div>
        </div>
      </div>
    </section>
  );
}

const outcomes = [
  {
    name: "Pass",
    symbol: "✓",
    text: "The result meets the defined checks.",
    tone: "pass",
  },
  {
    name: "Fail",
    symbol: "×",
    text: "An outcome differs in a way that matters.",
    tone: "fail",
  },
  {
    name: "Error",
    symbol: "!",
    text: "Execution could not complete.",
    tone: "error",
  },
  {
    name: "Uncertain",
    symbol: "?",
    text: "The available evidence cannot decide.",
    tone: "uncertain",
  },
];
export function OutcomeSection() {
  return (
    <section id="verdicts" className={styles.outcomes}>
      <div className={styles.outcomeInner}>
        <div className={styles.outcomeCopy}>
          <p className="section-label">What comes through the gate</p>
          <h2>
            Keep the work.
            <br />
            <em>Keep the proof.</em>
          </h2>
          <p>
            A migration deserves more than “looks good.” Get a record of what
            held, what changed, and what still needs a closer look.
          </p>
          <a className="text-link" href="/guide/">
            Understand the checks <span aria-hidden="true">↗</span>
          </a>
          <span className={styles.paperNote}>
            A historical answer is evidence.
            <br />
            Your checks determine success.
          </span>
        </div>
        <div className={styles.receipt}>
          <div className={styles.receiptHead}>
            <span>SWITCHIT / MIGRATION RECORD</span>
            <span>Illustrative report</span>
          </div>
          <div className={styles.receiptTitle}>
            <span>
              One workflow.
              <br />
              Every outcome accounted for.
            </span>
            <span className={styles.receiptSeal} aria-hidden="true">
              検<br />証
            </span>
          </div>
          <div className={styles.receiptRows}>
            {outcomes.map((item) => (
              <div key={item.name} className={styles.receiptRow}>
                <span
                  className={`${styles.verdictSymbol} ${styles[item.tone]}`}
                >
                  {item.symbol}
                </span>
                <div>
                  <strong>{item.name}</strong>
                  <p>{item.text}</p>
                </div>
              </div>
            ))}
          </div>
          <div className={styles.receiptTotal}>
            <span>DECISION BASIS</span>
            <strong>Your criteria. Recorded results.</strong>
          </div>
          <div className={styles.receiptFoot}>
            <span>Inspectable · exportable · yours</span>
            <span>↗</span>
          </div>
        </div>
      </div>
    </section>
  );
}

export function TargetsSection() {
  return (
    <section id="targets" className={styles.targets}>
      <div className={styles.inner}>
        <div className={styles.targetsHead}>
          <p className="section-label">Choose the next environment</p>
          <h2>
            Many destinations.
            <br />
            <em>One reviewed workflow.</em>
          </h2>
          <p>
            Choose a provider adapter for replay. Agent-session import and
            target execution are separate capabilities.
          </p>
        </div>
        <div className={styles.destinationList}>
          <div>
            <span className={styles.destinationNumber}>01</span>
            <h3>Compatible APIs</h3>
            <p>
              OpenAI · xAI · Groq · DeepSeek · Mistral
              <br />
              OpenRouter · Together · Fireworks
            </p>
            <span className={styles.destinationTag}>OpenAI-compatible</span>
          </div>
          <div>
            <span className={styles.destinationNumber}>02</span>
            <h3>Native protocols</h3>
            <p>
              Anthropic Messages
              <br />
              Amazon Bedrock Converse
            </p>
            <span className={styles.destinationTag}>Provider adapters</span>
          </div>
          <div>
            <span className={styles.destinationNumber}>03</span>
            <h3>On your own ground</h3>
            <p>
              Ollama · vLLM
              <br />
              Recorded fixtures · mock targets
            </p>
            <span className={styles.destinationTag}>Local & offline</span>
          </div>
        </div>
        <p className={styles.targetNote}>
          Live replay is an explicit choice. Start offline, inspect the
          evidence, then connect your provider.
        </p>
      </div>
    </section>
  );
}
export default function ProductStory() {
  return (
    <>
      <SourceStrip />
      <WorkflowSection />
      <OutcomeSection />
      <TargetsSection />
    </>
  );
}
