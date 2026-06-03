"""Render the Markdown report to a self-contained, print-ready HTML file.

Single responsibility: convert REPORT.md to REPORT.html with figures inlined as
base64 (so the file is fully portable) and clean print CSS. Open it in a browser
and "Save as PDF" to get the PDF deliverable; no system tools required.
"""

from __future__ import annotations

import base64
import re

import markdown

from . import paths

_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       max-width: 880px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 1.9rem; border-bottom: 2px solid #C44E52; padding-bottom: .3rem; }
h2 { font-size: 1.35rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }
h3 { font-size: 1.1rem; }
table { border-collapse: collapse; width: 100%; font-size: .82rem; margin: 1rem 0; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
th { background: #f4f4f4; }
img { max-width: 100%; height: auto; display: block; margin: 1rem auto; }
em { color: #555; font-size: .9rem; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
@media print { body { margin: 0; } h2 { page-break-after: avoid; } img { page-break-inside: avoid; } }
"""


def _inline_images(html: str) -> str:
    def repl(match: re.Match) -> str:
        rel = match.group(1)
        img_path = paths.REPORT_DIR / rel
        if not img_path.exists():
            return match.group(0)
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    return re.sub(r'src="(figures/[^"]+)"', repl, html)


def build() -> str:
    md_text = paths.REPORT_FILE.read_text()
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    body = _inline_images(body)
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Voice Across the Menstrual Cycle</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )
    out = paths.REPORT_DIR / "REPORT.html"
    out.write_text(html)
    return str(out)
