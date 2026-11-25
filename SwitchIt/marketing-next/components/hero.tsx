import styles from "./hero.module.css";

export default function Hero() {
  return (
    <section className={styles.hero} aria-labelledby="hero-heading">
      <img
        className={styles.landscape}
        src="/images/switchit-gate.webp"
        alt="A vermilion gate over a misty blue lake, with one stone path continuing through it toward the mountains."
        fetchPriority="high"
        width="1672"
        height="941"
      />
      <div className={styles.wash} aria-hidden="true" />
      <div className={styles.mist} aria-hidden="true" />
      <div className={styles.inner}>
        <h1 id="hero-heading">
          Improve your agent
          <br />
          <em>with replay.</em>
        </h1>
        <p className={styles.description}>
          Turn real failures into replay cases. Compare an instruction update
          with the active version before you use it.
        </p>
        <div className={styles.actions}>
          <a className="button button-red" href="/guide/#switchup">
            Set up SwitchUP <span aria-hidden="true">↗</span>
          </a>
          <a
            className="text-link"
            href="https://github.com/AnujJha27/Aegis"
            target="_blank"
            rel="noreferrer"
          >
            Try Aegis <span aria-hidden="true">↗</span>
          </a>
        </div>
        <nav className={styles.resources} aria-label="SwitchUP resources">
          <span className={styles.resourceLabel}>Resources</span>
          <div className={styles.resourceLinks}>
            <a
              href="https://pypi.org/project/switchit-harness/"
              target="_blank"
              rel="noreferrer"
            >
              PyPI <span aria-hidden="true">↗</span>
            </a>
            <a
              href="https://www.npmjs.com/package/switchit-harness"
              target="_blank"
              rel="noreferrer"
            >
              npm <span aria-hidden="true">↗</span>
            </a>
            <a
              href="https://www.npmjs.com/package/switchit-skill"
              target="_blank"
              rel="noreferrer"
            >
              Agent skill <span aria-hidden="true">↗</span>
            </a>
            <a
              href="https://github.com/aniiiiXD/SwitchIt"
              target="_blank"
              rel="noreferrer"
            >
              GitHub <span aria-hidden="true">↗</span>
            </a>
            <a href="/guide/">
              Guide <span aria-hidden="true">↗</span>
            </a>
          </div>
        </nav>
      </div>
    </section>
  );
}
