"""Build a colour-coded reference workbook from 'train 2-Reference.csv'.

CSV cannot store colours, so the categorised copy is written as .xlsx next to
the CSV. Training always uses Dataset/train.csv; this file is for reading only.
Run from the project root: python3 -m src.data.make_reference_xlsx
"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from src.features.feature_groups import COLUMN_TO_GROUP, FEATURE_GROUPS, GROUP_COLORS

from pathlib import Path

SRC = "Dataset/train 2-Reference.csv" if Path("Dataset/train 2-Reference.csv").exists() else "Dataset/train.csv"
OUT = "Dataset/train 2-Reference.xlsx"

df = pd.read_csv(SRC, keep_default_na=False)
missing = set(df.columns) - set(COLUMN_TO_GROUP)
extra = set(COLUMN_TO_GROUP) - set(df.columns)
assert not missing and not extra, (missing, extra)

fill = lambda hex_: PatternFill("solid", start_color=hex_, end_color=hex_)
thin = Side(style="thin", color="808080")

wb = Workbook()
ws = wb.active
ws.title = "Data"
for j, col in enumerate(df.columns, 1):
    g = COLUMN_TO_GROUP[col]
    top = ws.cell(row=1, column=j, value=g)
    hdr = ws.cell(row=2, column=j, value=col)
    for cell, bold in ((top, False), (hdr, True)):
        cell.fill = fill(GROUP_COLORS[g])
        cell.font = Font(bold=bold, size=9 if not bold else 11)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=thin, right=thin)
    ws.column_dimensions[get_column_letter(j)].width = 15
for i, row in enumerate(df.itertuples(index=False), 3):
    for j, v in enumerate(row, 1):
        if isinstance(v, str) and v.lstrip("-").replace(".", "", 1).isdigit():
            v = float(v) if "." in v else int(v)
        ws.cell(row=i, column=j, value=v)
ws.row_dimensions[1].height = 32
ws.freeze_panes = "B3"

lg = wb.create_sheet("Legend")
lg.append(["Category", "Colour", "No. of columns", "Columns"])
for c in lg[1]:
    c.font = Font(bold=True)
for g, cols in FEATURE_GROUPS.items():
    lg.append([g, "", len(cols), ", ".join(cols)])
    lg.cell(row=lg.max_row, column=2).fill = fill(GROUP_COLORS[g])
    lg.cell(row=lg.max_row, column=1).fill = fill(GROUP_COLORS[g])
lg.append(["Total", "", sum(len(c) for c in FEATURE_GROUPS.values())])
for col, w in zip("ABCD", (30, 10, 16, 120)):
    lg.column_dimensions[col].width = w

wb.save(OUT)
print("saved", OUT, df.shape)
