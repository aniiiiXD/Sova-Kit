import Hero from "../components/hero";
import {
  SourceStrip,
  WorkflowSection,
  OutcomeSection,
  TargetsSection,
} from "../components/product-story";
import { InstallPanel, MigrationPreview } from "../components/migration-demo";
export default function Home() {
  return (
    <main id="main-content">
      <Hero />
      <SourceStrip />
      <WorkflowSection />
      <section id="demo" className="replay-section">
        <div className="replay-scenery" aria-hidden="true" />
        <div className="replay-inner">
          <div className="replay-heading">
            <p className="section-label">The crossing</p>
            <h2>
              Take another route.
              <br />
              <em>See what comes through.</em>
            </h2>
            <p>
              Choose a target. Replay the same cases. Inspect the result, one
              check at a time.
            </p>
          </div>
          <MigrationPreview />
          <div className="replay-principles">
            <span>
              Illustrative outcomes · synthetic data · no provider calls
            </span>
            <a href="/guide/">Run a comparison with your own traces ↗</a>
          </div>
        </div>
      </section>
      <OutcomeSection />
      <TargetsSection />
      <InstallPanel />
    </main>
  );
}
