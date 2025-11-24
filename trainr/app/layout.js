"use client";
import Link from "next/link";

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <title>Trainr</title>
        <meta name="Therapy" content="Helps revlive pain in your muscles" />
      </head>
      <body>
        <nav>
          <Link href="/">Home</Link>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
