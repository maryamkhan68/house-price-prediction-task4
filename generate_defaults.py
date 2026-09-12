"""
One-time script: computes sensible default values for every feature
the model expects, so the Streamlit app can fill in the ~224 fields
the user isn't asked about, using realistic values instead of zeros.
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/train.csv")
df = df.drop(columns=["Id"])
df = df.drop_duplicates()

none_means_no_feature = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]
for col in none_means_no_feature:
    df[col] = df[col].fillna("None")
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))]

quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
ordinal_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                 "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
for col in ordinal_cols:
    df[col] = df[col].map(quality_map)

nominal_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])
df["HasPool"] = (df["PoolArea"] > 0).astype(int)
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

X = df.drop(columns=["SalePrice"])

# For every column: median if continuous, mode if binary/dummy
binary_like_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
defaults = {}
for col in X.columns:
    if col in binary_like_cols:
        defaults[col] = X[col].mode()[0]
    else:
        defaults[col] = X[col].median()

joblib.dump(defaults, "models/feature_defaults.pkl")
print(f"Saved feature_defaults.pkl with {len(defaults)} default values")