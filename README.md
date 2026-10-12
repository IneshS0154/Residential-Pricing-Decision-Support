# Samarasinghe G.K.D.T (IT24102584)

IT3091 Machine Learning, group 2026-DS-38. Predictive modelling of residential property prices (Ames Housing).
Role: Data preparation, feature engineering, leakage audit.

## Notebooks in this repository

| Notebook |
|---|
| `notebooks/02_preprocessing.ipynb` |
| `notebooks/03_feature_selection.ipynb` |

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
