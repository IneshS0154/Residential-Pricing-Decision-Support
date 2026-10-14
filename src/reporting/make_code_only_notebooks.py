"""Create code-only copies of the notebooks in notebooks/code_only/.

Each copy keeps every code cell and its stored outputs, drops all explanation (markdown) cells, and starts with
one short pointer cell that links to the matching document in docs/notebooks/. Narrative that a notebook generates as
markdown output (display(Markdown(...))) is dropped too. The originals are never modified.
Run from the project root:  python3 -m src.reporting.make_code_only_notebooks
"""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "notebooks"
OUT_DIR = SRC_DIR / "code_only"


def drop_markdown_outputs(cell):
    """Remove outputs that are rendered markdown text; return how many were removed."""
    keep = [o for o in cell.get("outputs", []) if not ("data" in o and "text/markdown" in o["data"])]
    removed = len(cell.get("outputs", [])) - len(keep)
    cell["outputs"] = keep
    return removed


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for path in sorted(SRC_DIR.glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        title = next((l.lstrip("# ").strip() for c in nb.cells if c.cell_type == "markdown"
                      for l in c.source.splitlines() if l.startswith("# ")), path.stem)
        code = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
        dropped = sum(drop_markdown_outputs(c) for c in code)
        pointer = nbformat.v4.new_markdown_cell(
            f"**{title}** (code-only copy). Explanations: [`docs/notebooks/{path.stem}.md`](../../docs/notebooks/{path.stem}.md)")
        slim = nbformat.v4.new_notebook(cells=[pointer, *code], metadata=nb.metadata)
        nbformat.validate(slim)
        nbformat.write(slim, OUT_DIR / path.name)
        print(f"{path.name:40s} {len(nb.cells):3d} cells -> {len(slim.cells):3d} "
              f"({len(code)} code + 1 pointer, {dropped} markdown outputs dropped)  "
              f"{path.stat().st_size / 1e6:.1f} MB -> {(OUT_DIR / path.name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
