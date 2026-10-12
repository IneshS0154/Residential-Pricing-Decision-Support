# Data Dictionary (Ames Housing, train.csv, 81 columns)

Types: Cont = continuous number, Count = integer count, Year = calendar year, Nom = nominal category, Ord = ordinal category, Bin = binary.
Quality scale (Ord): None < Po < Fa < TA < Gd < Ex = No feature / Poor / Fair / Typical-Average / Good / Excellent.
"NA" in categorical columns means the feature is absent (no alley, no garage, ...), not missing data.

## Identifier and target
| Column | Full name | Type |
|---|---|---|
| Id | Row identifier (drop) | - |
| SalePrice | Sale price in USD (target) | Cont |

## Location and lot
| Column | Full name | Type |
|---|---|---|
| Neighborhood | Neighborhood (physical location within Ames city limits) | Nom, 25 levels |
| MSZoning | Zoning classification of the sale | Nom |
| LotFrontage | Lot frontage: linear feet of street connected to the property | Cont |
| LotArea | Lot area in square feet | Cont |
| LotShape | Lot shape (Reg, IR1, IR2, IR3) | Ord |
| LotConfig | Lot configuration (inside, corner, cul-de-sac, FR2, FR3) | Nom |
| LandContour | Land contour: flatness of the property | Nom |
| LandSlope | Land slope (Gtl, Mod, Sev) | Ord |
| Street | Street access type (gravel/paved) | Bin |
| Alley | Alley access type (gravel/paved/none) | Nom |
| Condition1 | Proximity to roads, railroads or positive features (primary) | Nom |
| Condition2 | Proximity condition (second, if more than one applies) | Nom |
| Utilities | Utilities available | Ord |
| PavedDrive | Paved driveway (Y, P, N) | Ord |

## Dwelling type and structure
| Column | Full name | Type |
|---|---|---|
| MSSubClass | Building class: dwelling type involved in the sale (numeric code, 16 values, but nominal) | Nom |
| BldgType | Building type (1Fam, 2fmCon, Duplex, Twnhs, TwnhsE) | Nom |
| HouseStyle | House style (1Story, 2Story, SFoyer, ...) | Nom |
| RoofStyle | Roof style | Nom |
| RoofMatl | Roof material | Nom |
| Foundation | Foundation type | Nom |
| Exterior1st | Exterior covering on house (primary) | Nom, 15 levels |
| Exterior2nd | Exterior covering on house (second material, if any) | Nom, 16 levels |
| MasVnrType | Masonry veneer type | Nom |
| MasVnrArea | Masonry veneer area in square feet | Cont |
| Functional | Home functionality rating (Typ ... Sal) | Ord |
| Electrical | Electrical system | Nom/Ord |
| Heating | Heating type | Nom |
| CentralAir | Central air conditioning (Y/N) | Bin |

## Size
| Column | Full name | Type |
|---|---|---|
| GrLivArea | Above-grade (ground) living area in square feet | Cont |
| 1stFlrSF | First floor area in square feet | Cont |
| 2ndFlrSF | Second floor area in square feet | Cont |
| LowQualFinSF | Low-quality finished area, all floors, in square feet | Cont |
| TotalBsmtSF | Total basement area in square feet | Cont |
| BsmtFinSF1 | Basement finished area, type 1, in square feet | Cont |
| BsmtFinSF2 | Basement finished area, type 2, in square feet | Cont |
| BsmtUnfSF | Basement unfinished area in square feet | Cont |
| TotRmsAbvGrd | Total rooms above grade (excludes bathrooms) | Count |
| BedroomAbvGr | Bedrooms above grade (excludes basement bedrooms) | Count |
| KitchenAbvGr | Kitchens above grade | Count |
| FullBath | Full bathrooms above grade | Count |
| HalfBath | Half bathrooms above grade | Count |
| BsmtFullBath | Basement full bathrooms | Count |
| BsmtHalfBath | Basement half bathrooms | Count |
| GarageArea | Garage size in square feet | Cont |
| GarageCars | Garage size in car capacity | Count |

## Quality and condition
| Column | Full name | Type |
|---|---|---|
| OverallQual | Overall material and finish quality (1-10) | Ord |
| OverallCond | Overall condition of the house (1-10) | Ord |
| ExterQual | Exterior material quality | Ord |
| ExterCond | Exterior material present condition | Ord |
| BsmtQual | Basement quality (height of basement) | Ord |
| BsmtCond | Basement general condition | Ord |
| BsmtExposure | Basement exposure (walkout or garden-level walls: Gd, Av, Mn, No) | Ord |
| BsmtFinType1 | Basement finished area rating, type 1 | Ord |
| BsmtFinType2 | Basement finished area rating, type 2 | Ord |
| HeatingQC | Heating quality and condition | Ord |
| KitchenQual | Kitchen quality | Ord |
| FireplaceQu | Fireplace quality | Ord |
| GarageQual | Garage quality | Ord |
| GarageCond | Garage condition | Ord |
| GarageFinish | Garage interior finish (Fin, RFn, Unf, none) | Ord |
| PoolQC | Pool quality | Ord |
| Fence | Fence quality (privacy / wood-wire) | Nom |

## Age
| Column | Full name | Type |
|---|---|---|
| YearBuilt | Original construction year | Year |
| YearRemodAdd | Remodel / addition year (equals YearBuilt if none) | Year |
| GarageYrBlt | Year the garage was built | Year |

## Amenities
| Column | Full name | Type |
|---|---|---|
| Fireplaces | Number of fireplaces | Count |
| GarageType | Garage location/type (attached, detached, built-in, carport, ...) | Nom |
| WoodDeckSF | Wood deck area in square feet | Cont |
| OpenPorchSF | Open porch area in square feet | Cont |
| EnclosedPorch | Enclosed porch area in square feet | Cont |
| 3SsnPorch | Three-season porch area in square feet | Cont |
| ScreenPorch | Screen porch area in square feet | Cont |
| PoolArea | Pool area in square feet | Cont |
| MiscFeature | Miscellaneous feature (elevator, 2nd garage, shed, tennis court, other) | Nom |
| MiscVal | Dollar value of the miscellaneous feature | Cont |

## Sale context
| Column | Full name | Type |
|---|---|---|
| MoSold | Month sold (1-12) | Count |
| YrSold | Year sold (2006-2010) | Year |
| SaleType | Type of sale (warranty deed, new, court officer deed, contract, ...) | Nom |
| SaleCondition | Condition of sale (normal, abnormal, partial, family, ...) | Nom |

## Data-vs-description discrepancies
- `BedroomAbvGr`, `KitchenAbvGr` appear as `Bedroom`, `Kitchen` in `data_description.txt`.
- The description spells some codes differently from the CSV. The CSV uses `NAmes` (description: `Names`), `2fmCon` (`2FmCon`), `Twnhs` (`TwnhsI`, inside unit), `Duplex` (`Duplx`), `CmentBd` (`CemntBd`), `Wd Shng` (`WdShing`), `Brk Cmn` (`BrkComm`), `C (all)` (`C`).
- Some description codes never occur in train (MSZoning `A`, `I`, `RP`; `PreCast` in `Exterior2nd`). `MSSubClass` 150 occurs only in test.
