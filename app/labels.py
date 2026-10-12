"""Display labels for coded categorical values (CSV spellings, which differ from data_description.txt). Display only: the model always receives the code."""
QUAL = {"Ex": "Excellent", "Gd": "Good", "TA": "Average / typical", "Fa": "Fair", "Po": "Poor"}

LABELS = {
    "Neighborhood": {
        "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale", "BrkSide": "Brookside", "ClearCr": "Clear Creek",
        "CollgCr": "College Creek", "Crawfor": "Crawford", "Edwards": "Edwards", "Gilbert": "Gilbert", "IDOTRR": "Iowa DOT and Rail Road",
        "MeadowV": "Meadow Village", "Mitchel": "Mitchell", "NAmes": "North Ames", "NoRidge": "Northridge", "NPkVill": "Northpark Villa",
        "NridgHt": "Northridge Heights", "NWAmes": "Northwest Ames", "OldTown": "Old Town", "SWISU": "South & West of Iowa State University",
        "Sawyer": "Sawyer", "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook", "Timber": "Timberland", "Veenker": "Veenker",
    },
    "BldgType": {"1Fam": "Single-family detached", "2fmCon": "Two-family conversion", "Duplex": "Duplex",
                 "Twnhs": "Townhouse, inside unit", "TwnhsE": "Townhouse, end unit"},
    "HouseStyle": {"1Story": "One story", "1.5Fin": "1.5 story, 2nd level finished", "1.5Unf": "1.5 story, 2nd level unfinished",
                   "2Story": "Two story", "2.5Fin": "2.5 story, 2nd level finished", "2.5Unf": "2.5 story, 2nd level unfinished",
                   "SFoyer": "Split foyer", "SLvl": "Split level"},
    "KitchenQual": QUAL, "ExterQual": QUAL,
    "CentralAir": {"Y": "Yes", "N": "No"},
}


def label(col, code):
    return f"{LABELS.get(col, {}).get(code, code)}"
