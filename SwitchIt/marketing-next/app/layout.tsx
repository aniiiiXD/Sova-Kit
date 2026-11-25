import type { Metadata } from "next";
import { Manrope, Cormorant_Garamond, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Header, Footer } from "../components/site-chrome";

const geistSans = Manrope({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});
const display = Cormorant_Garamond({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["400", "500"],
  style: ["normal", "italic"],
});
const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:4321",
  ),
  icons: { icon: "/icon.svg" },
  title: "SwitchUP — Agent improvement through replay",
  openGraph: {
    title: "SwitchUP — Agent improvement through replay",
    description:
      "Capture agent failures, compare instruction updates on reviewed cases, and activate passing versions.",
    images: [
      {
        url: "/images/switchit-gate.webp",
        width: 1672,
        height: 941,
        alt: "A continuous path through a gate over a misty blue lake",
      },
    ],
  },
  description:
    "SwitchUP uses recorded agent traces and replay to verify instruction updates and provider changes.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${jetbrainsMono.variable} ${display.variable}`}
    >
      <body className="font-sans">
        <a href="#main-content" className="skip-link">
          Skip to content
        </a>
        <Header />
        {children}
        <Footer />
      </body>
    </html>
  );
}
