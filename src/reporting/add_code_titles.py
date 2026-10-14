"""Add a short title above every code cell of the code-only notebooks.

Titles come from the section headings of the detailed notebooks (same code cells, same order):
  * the first code cell of a notebook is titled "Setup";
  * the first code cell of a section gets the section heading ("## 4. Structural Missing Values");
  * further code cells of the same section get "### 4.2 <first comment line of the cell>", or "<section> (continued)".
Inserted cells carry metadata {"code_title": true}, so the script is idempotent (it removes its own cells first).
Notebook outputs and code are never changed.

Run from the project root:
    python3 -m src.reporting.add_code_titles --dry-run        # print the titles only
    python3 -m src.reporting.add_code_titles                  # add them
    python3 -m src.reporting.add_code_titles --remove         # take them out again
"""
import argparse
import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "notebooks"
OVERRIDES = ROOT / "src" / "reporting" / "code_title_overrides.json"   # {"<notebook>": {"<code cell index>": "<title>"}}
SOURCE = TARGET / "notebooks_legacy"


def _code(cell):
    return cell.cell_type == "code" and cell.source.strip()


def section_plan(detailed_path):
    """For each non-empty code cell of the detailed notebook: (section heading or None, sub-heading or None)."""
    nb = nbformat.read(detailed_path, as_version=4)
    sec = sub = None
    plan = []
    for c in nb.cells:
        if c.cell_type == "markdown":
            for line in c.source.splitlines():
                m = re.match(r"^(#{2,3}) (.+)", line)
                if m:
                    if len(m.group(1)) == 2:
                        sec, sub = m.group(2).strip(), None
                    else:
                        sub = m.group(2).strip()
        elif _code(c):
            plan.append((sec, sub))
    return plan


def clean(text, limit=70):
    text = text.strip().replace("`", "")
    cuts = [(text.index(sep), sep) for sep in (". ", "? ", ": ", "; ", " - ", " => ") if sep in text]
    question = False
    if cuts:                                                               # keep only the first clause
        pos, sep = min(cuts)
        question = sep == "? "
        text = text[:pos]
    text = text.strip().rstrip(".:;?") + ("?" if question else "")
    text = text[0].upper() + text[1:] if text else text
    return text if len(text) <= limit else text[: limit - 3].rsplit(" ", 1)[0] + "..."


def build_titles(code_cells, plan):
    titles, seen, count = [], {}, {}
    for i, (cell, (sec, sub)) in enumerate(zip(code_cells, plan)):
        first = next(l.strip() for l in cell.source.splitlines() if l.strip())
        if i == 0:
            titles.append("## Setup: imports, paths and data loading")
            continue
        key = sec or "(no section)"
        count[key] = count.get(key, 0) + 1
        num = re.match(r"^(\d+)\.", sec or "")
        name = re.sub(r"^\d+\.\s*", "", sec or "").strip() or "Code"
        if count[key] == 1:
            titles.append(f"## {sec}" if sec else f"## {name}")
            continue
        label = f"{num.group(1)}.{count[key]}" if num else f"{count[key]}"
        if sub and sub != seen.get(key):
            descr = sub
        elif first.startswith("#"):
            descr = clean(first.lstrip("#"))
        else:
            descr = f"{name} (continued)"
        seen[key] = sub
        if re.match(r"^\d+[a-zA-Z]\.\s", descr):                      # sub-heading already carries its own label, e.g. "11b. ..."
            titles.append(f"### {descr}")
        else:
            titles.append(f"### {label} {descr}")
    return titles


def process(name, dry_run, remove):
    path = TARGET / name
    nb = nbformat.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if not c.get("metadata", {}).get("code_title")]       # idempotent: drop our own cells
    code_cells = [c for c in nb.cells if _code(c)]
    if remove:
        if not dry_run:
            nbformat.write(nb, path)
        return []
    plan = section_plan(SOURCE / name)
    assert len(plan) == len(code_cells), f"{name}: {len(code_cells)} code cells vs {len(plan)} in the detailed notebook"
    titles = build_titles(code_cells, plan)
    if OVERRIDES.exists():
        for idx, text in json.loads(OVERRIDES.read_text(encoding="utf-8")).get(name, {}).items():
            titles[int(idx)] = text
    out, k = [], 0
    for c in nb.cells:
        if _code(c):
            t = nbformat.v4.new_markdown_cell(titles[k]); t["metadata"]["code_title"] = True; t["id"] = f"code-title-{k:03d}"   # fixed ids: re-runs give identical files
            out.append(t); k += 1
        out.append(c)
    nb.cells = out
    if not dry_run:
        nbformat.validate(nb)
        nbformat.write(nb, path)
    return titles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    for p in sorted(TARGET.glob("[0-9]*.ipynb")):
        titles = process(p.name, a.dry_run, a.remove)
        print(f"{p.name}: {'removed titles' if a.remove else f'{len(titles)} titles'}{' (dry run)' if a.dry_run else ''}")
        if a.dry_run and not a.remove:
            for i, t in enumerate(titles):
                print(f"   {i:2d} {t}")


if __name__ == "__main__":
    main()
