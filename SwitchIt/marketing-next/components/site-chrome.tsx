"use client";
import { useEffect, useRef, useState } from "react";

export function Brand() {
  return (
    <a className="wordmark" href="/" aria-label="SwitchUP home">
      <svg
        viewBox="0 0 32 32"
        width="29"
        height="29"
        fill="none"
        aria-hidden="true"
      >
        <path
          d="M3 6c8 3 18 3 26 0M7 11h18M9 10 8 27M23 10l1 17M6 17h20"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="square"
        />
      </svg>
      <span>
        Switch<span className="wordmark-stop">UP</span>
      </span>
    </a>
  );
}
const links = [
  { href: "/#how", label: "The workflow" },
  { href: "/#demo", label: "The crossing" },
  { href: "/#verdicts", label: "The evidence" },
];
export function Header() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const toggle = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const sentinel = document.getElementById("nav-sentinel");
    if (!sentinel) return;
    const observer = new IntersectionObserver(([entry]) =>
      setScrolled(!entry.isIntersecting),
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape" && open) {
        setOpen(false);
        toggle.current?.focus();
      }
    };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, [open]);
  return (
    <>
      <div id="nav-sentinel" aria-hidden="true" />
      <header
        className={`site-header ${scrolled || open ? "header-solid" : ""}`}
      >
        <div className="nav-inner">
          <Brand />
          <nav className="desktop-nav" aria-label="Main navigation">
            {links.map((link) => (
              <a key={link.href} href={link.href}>
                {link.label}
              </a>
            ))}
          </nav>
          <div className="nav-actions">
            <a
              href="https://github.com/aniiiiXD/SwitchIt"
              className="github-link"
              target="_blank"
              rel="noreferrer"
            >
              GitHub ↗
            </a>
            <a href="/#install" className="nav-cta">
              Start your switch <span aria-hidden="true">↗</span>
            </a>
            <button
              ref={toggle}
              className="menu-toggle"
              aria-expanded={open}
              aria-controls="mobile-navigation"
              aria-label={open ? "Close menu" : "Open menu"}
              onClick={() => setOpen(!open)}
            >
              <span />
              <span className={open ? "menu-cross" : ""} />
            </button>
          </div>
        </div>
        {open && (
          <nav
            className="mobile-menu"
            id="mobile-navigation"
            aria-label="Mobile navigation"
            onClick={() => setOpen(false)}
          >
            {links.map((link) => (
              <a key={link.href} href={link.href}>
                {link.label}
              </a>
            ))}
            <a href="/#sources">Sources & integrations</a>
            <a href="/guide/">Read the guide</a>
          </nav>
        )}
      </header>
    </>
  );
}
export function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-main">
        <div>
          <Brand />
          <p>The model changes. The path continues.</p>
        </div>
        <nav className="footer-links" aria-label="Footer">
          <a href="/guide/">Documentation</a>
          <a
            href="https://github.com/aniiiiXD/SwitchIt"
            target="_blank"
            rel="noreferrer"
          >
            GitHub ↗
          </a>
          <a href="/LICENSE.txt">MIT license</a>
        </nav>
      </div>
      <div className="footer-bottom">
        <span>Built for the freedom to switch.</span>
        <span>Open source · Local by default</span>
        <a href="#main-content">Back to the beginning ↑</a>
      </div>
    </footer>
  );
}
