"""
Trains and saves all 4 models (not just the best one) so the Streamlit
app can offer "multiple model selection" as a bonus feature.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

print("Loading and processing data...")
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
y = df["SalePrice"]
binary_like_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
cols_to_scale = [c for c in X.select_dtypes(include=[np.number]).columns if c not in binary_like_cols]
scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training all 4 models...")
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.1, max_depth=2,
        min_samples_split=10, min_samples_leaf=2, subsample=0.8, random_state=42
    ),  # using the tuned hyperparameters we found earlier
}

all_models = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    all_models[name] = m
    print(f"  {name} trained.")

joblib.dump(all_models, "models/all_models.pkl")
print("Saved all_models.pkl with 4 trained models")