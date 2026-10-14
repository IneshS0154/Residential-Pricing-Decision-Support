"""Build the per-member folders in "Github Submission/" (one stand-alone repository folder per group member).

Each folder contains: that member's notebooks as code-only files (no explanation cells, no markdown outputs),
the shared source code (src/, without the notebook-conversion tools), tests that do not depend on documents,
the stored results the notebooks read (artifacts/, reports/), requirements.txt, .gitignore and a short README.
It contains no explanation documents (no docs/*.md) and no notebooks with explanations. Dataset/ is left empty.
Run from the project root:  python3 -m src.reporting.build_github_submission
"""
import re
import shutil
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Github Submission"
SRC_NB = ROOT / "notebooks"

MEMBERS = {   # folder name -> (id, name, role, notebooks)
    "IT24103124_Walawwaththage": ("IT24103124", "Walawwaththage H.D.N", "Business problem framing, lens selection, EDA",
                                  ["01_eda"]),
    "IT24102584_Samarasinghe": ("IT24102584", "Samarasinghe G.K.D.T", "Data preparation, feature engineering, leakage audit",
                                ["02_preprocessing", "03_feature_selection"]),
    "IT24102616_Chandrasekaran": ("IT24102616", "Chandrasekaran I.S", "Model development, quantile models, hyperparameter tuning",
                                  ["04_baseline_model_comparison", "05_hyperparameter_tuning"]),
    "IT24103989_Gunarathna": ("IT24103989", "Gunarathna H.M.S.S", "Evaluation, calibration, routing, recommendation",
                              ["06_final_evaluation", "07_prediction_intervals", "08_segment_temporal_analysis",
                               "09_auto_escalate_routing", "10_business_interpretation"]),
}
SKIP_TESTS = {"test_project_consistency.py", "test_report_quality.py"}     # check project-wide documents


def code_only(nb_path, dest):
    nb = nbformat.read(nb_path, as_version=4)
    cells = []
    for c in nb.cells:
        if c.cell_type != "code" or not c.source.strip():
            continue
        c.source = re.sub(r"\(ROOT / ['\"]CLAUDE\.md['\"]\)", "(ROOT / 'requirements.txt')", c.source)   # root marker present in every repo
        c["outputs"] = [o for o in c.get("outputs", []) if not ("data" in o and "text/markdown" in o["data"])]
        cells.append(c)
    slim = nbformat.v4.new_notebook(cells=cells, metadata=nb.metadata)
    nbformat.validate(slim)
    nbformat.write(slim, dest)


def readme(folder, ident, name, role, notebooks):
    rows = "\n".join(f"| `notebooks/{n}.ipynb` |" for n in notebooks)
    return f"""# {name} ({ident})

IT3091 Machine Learning, group 2026-DS-38. Predictive modelling of residential property prices (Ames Housing).
Role: {role}.

## Notebooks in this repository

| Notebook |
|---|
{rows}

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The data is not included. Download the Kaggle competition files into `Dataset/` (`train.csv`, `test.csv`, `data_description.txt`):

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
```

## Run

Run everything from the repository root. Stored results used by the notebooks are in `artifacts/` and `reports/`.

```bash
python -m unittest discover -s tests
jupyter nbconvert --to notebook --execute --inplace notebooks/<notebook>.ipynb
```
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    for folder, (ident, name, role, nbs) in MEMBERS.items():
        repo = OUT / folder
        shutil.copytree(ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "reporting"))
        shutil.copytree(ROOT / "tests", repo / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", *SKIP_TESTS))
        shutil.copytree(ROOT / "artifacts", repo / "artifacts", ignore=ignore)
        shutil.copytree(ROOT / "reports", repo / "reports", ignore=ignore)
        shutil.copy(ROOT / "requirements.txt", repo / "requirements.txt")
        (repo / "notebooks").mkdir()
        for n in nbs:
            code_only(SRC_NB / f"{n}.ipynb", repo / "notebooks" / f"{n}.ipynb")
        for d in ("Dataset", "docs"):
            (repo / d).mkdir()
            (repo / d / ".gitkeep").write_text("")
        (repo / ".gitignore").write_text(".DS_Store\n__pycache__/\n*.pyc\n.venv/\n.ipynb_checkpoints/\nCLAUDE.md\n")
        (repo / "README.md").write_text(readme(folder, ident, name, role, nbs), encoding="utf-8")
        size = sum(f.stat().st_size for f in repo.rglob("*") if f.is_file()) / 1e6
        print(f"{folder:30s} notebooks {len(nbs)}  {size:5.1f} MB")
    rows = "\n".join(f"| `{f}` | {i} | {n} | {', '.join(nb.split('_')[0] for nb in nbs)} |" for f, (i, n, _, nbs) in MEMBERS.items())
    (OUT / "README.md").write_text(f"""# Github Submission

One folder per group member. Each folder is a stand-alone repository: give a member their folder and they upload it from their own GitHub account.

| Folder | ID | Member | Notebooks |
|---|---|---|---|
{rows}

Upload (run by the member, inside their folder, logged in as themselves):

```bash
git init
git add .
git commit -m "Add notebooks and supporting code"
git branch -M main
git remote add origin https://github.com/<their-username>/<repository-name>.git
git push -u origin main
```

`Dataset/` is empty on purpose (Kaggle files are not redistributed); see each folder's README for the download command.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
