"""
Retrain and re-save the model locally to fix the sklearn version mismatch
from Colab. Run this once from the project root folder.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("data/train.csv")
df = df.drop(columns=["Id"])
df = df.drop_duplicates()

# Handle missing values
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

# Outliers
df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))]

# Encoding
quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
ordinal_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                 "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
for col in ordinal_cols:
    df[col] = df[col].map(quality_map)
nominal_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# Feature engineering
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

# Scaling
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]
binary_like_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
cols_to_scale = [c for c in X.select_dtypes(include=[np.number]).columns if c not in binary_like_cols]
scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

print(f"Final shape: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
print("Running RandomizedSearchCV (this takes 1-3 minutes)...")
param_dist = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.7, 0.8, 0.9, 1.0],
}
random_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=30, scoring="neg_root_mean_squared_error",
    cv=5, random_state=42, n_jobs=-1, verbose=1
)
random_search.fit(X_train, y_train)
tuned_model = random_search.best_estimator_

# Evaluate
preds = tuned_model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
print(f"R2: {r2_score(y_test, preds):.4f}")

# Save everything fresh, using local sklearn version
joblib.dump(tuned_model, "models/final_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(cols_to_scale, "models/cols_to_scale.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
print("Saved all model files locally with matching sklearn version.")