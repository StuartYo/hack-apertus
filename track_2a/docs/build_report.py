"""Build docs/BookletCheck_Report.pdf from technical_report.md (pandoc -> HTML -> headless Chrome).

    python docs/build_report.py [--browser PATH]

Run scripts/make_results.py first so the tables are current. Prints the page count; the
challenge allows at most 6 pages.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "BookletCheck_Report.pdf"
CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Source Serif 4", Georgia, "Times New Roman", serif; font-size: 9.6pt; line-height: 1.38;
       color: #111; max-width: none; margin: 0; }
h1 { font-size: 16pt; margin: 0 0 6pt; }
h2 { font-size: 11.5pt; margin: 11pt 0 4pt; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; }
p, li { margin: 3pt 0; }
ul { padding-left: 14pt; margin: 3pt 0; }
code { font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 8.4pt; }
pre { font-size: 7.6pt; line-height: 1.25; background: #f5f5f5; padding: 5pt 6pt; border-radius: 2pt;
      white-space: pre; overflow: hidden; page-break-inside: avoid; }
table { border-collapse: collapse; font-size: 7.6pt; margin: 4pt 0; width: 100%; page-break-inside: avoid; }
th, td { border: 0.5pt solid #bbb; padding: 2pt 3pt; text-align: left; vertical-align: top; }
th { background: #eee; }
"""


def find_browser(explicit):
    cands = [explicit] if explicit else []
    cands += [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              shutil.which("chromium") or "", shutil.which("google-chrome") or ""]
    for c in cands:
        if c and Path(c).exists():
            return c
    sys.exit("no Chrome/Edge/Chromium found; pass --browser")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser")
    args = ap.parse_args()
    md = (ROOT / "technical_report.md").read_text(encoding="utf-8")
    md = re.sub(r"<!-- [A-Z]+:(START|END) -->\n?", "", md)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.md"
        src.write_text(md, encoding="utf-8")
        css = Path(tmp) / "report.css"
        css.write_text(CSS, encoding="utf-8")
        html = Path(tmp) / "report.html"
        subprocess.run(["pandoc", str(src), "-f", "gfm", "-t", "html5", "-s", "--embed-resources",
                        "--css", str(css), "--metadata", "pagetitle=booklet-check technical report",
                        "-o", str(html)], check=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([find_browser(args.browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={OUT}", html.as_uri()],
                       check=True, capture_output=True)
    from pypdf import PdfReader

    pages = len(PdfReader(str(OUT)).pages)
    print(f"{OUT} ({pages} pages){' TOO LONG' if pages > 6 else ''}")


if __name__ == "__main__":
    main()
