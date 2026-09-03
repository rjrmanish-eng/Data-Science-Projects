# ================================
# 1. Import Libraries
# ================================
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# ================================
# 2. Load Dataset
# ================================
housing = pd.read_csv("housing.csv")


# ================================
# 3. Stratified Train-Test Split
# ================================
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idx].drop("income_cat", axis=1)
    strat_test_set  = housing.loc[test_idx].drop("income_cat", axis=1)


# ================================
# 4. Separate Features & Labels
# ================================
housing = strat_train_set.copy()

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)


# ================================
# 5. Columns
# ================================
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]


# ================================
# 6. Pipelines
# ================================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])


# ================================
# 7. MODELS WITH PIPELINE (IMPORTANT)
# ================================

# ---- Linear Regression ----
lin_pipeline = Pipeline([
    ("preprocess", full_pipeline),
    ("model", LinearRegression())
])

lin_rmse_scores = -cross_val_score(
    lin_pipeline,
    housing,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("Linear Regression RMSE")
print(pd.Series(lin_rmse_scores).describe())


# ---- Decision Tree ----
tree_pipeline = Pipeline([
    ("preprocess", full_pipeline),
    ("model", DecisionTreeRegressor(random_state=42))
])

tree_rmse_scores = -cross_val_score(
    tree_pipeline,
    housing,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("\nDecision Tree RMSE")
print(pd.Series(tree_rmse_scores).describe())


# ---- Random Forest ----
forest_pipeline = Pipeline([
    ("preprocess", full_pipeline),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

forest_rmse_scores = -cross_val_score(
    forest_pipeline,
    housing,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("\nRandom Forest RMSE")
print(pd.Series(forest_rmse_scores).describe())
