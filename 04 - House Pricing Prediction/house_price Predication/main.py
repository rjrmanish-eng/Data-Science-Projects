# import os
# import joblib
# import pandas as pd
# import numpy as np

# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor


# MODEL_FILE = "model.pkl"
# PIPELINE_FILE = "pipeline.pkl"
# TRAIN_INPUT_FILE = "train_input.csv"
# PREDICT_INPUT_FILE = "predict_input.csv"
# OUTPUT_FILE = "output.csv"


# # ---------------- PIPELINE ----------------
# def build_pipeline(num_attribs, cat_attribs):

#     num_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])

#     cat_pipeline = Pipeline([
#         ("onehot", OneHotEncoder(handle_unknown="ignore"))
#     ])

#     full_pipeline = ColumnTransformer([
#         ("num", num_pipeline, num_attribs),
#         ("cat", cat_pipeline, cat_attribs)
#     ])

#     return full_pipeline


# # ---------------- TRAIN / PREDICT ----------------
# if not os.path.exists(MODEL_FILE):

#     print("Training model...")

#     housing = pd.read_csv("housing.csv")

#     housing["income_cat"] = pd.cut(
#         housing["median_income"],
#         bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
#         labels=[1, 2, 3, 4, 5]
#     )

#     split = StratifiedShuffleSplit(
#         n_splits=1, test_size=0.2, random_state=42
#     )

#     for train_idx, _ in split.split(housing, housing["income_cat"]):
#         housing = housing.loc[train_idx].drop("income_cat", axis=1)

#     # Save sample input for prediction testing
#     housing.drop("median_house_value", axis=1).head(10).to_csv(
#         TRAIN_INPUT_FILE, index=False
#     )

#     # Separate features & labels
#     housing_labels = housing["median_house_value"].copy()
#     housing_features = housing.drop("median_house_value", axis=1)

#     num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
#     cat_attribs = ["ocean_proximity"]

#     pipeline = build_pipeline(num_attribs, cat_attribs)
#     housing_prepared = pipeline.fit_transform(housing_features)

#     model = RandomForestRegressor(
#         n_estimators=100,
#         random_state=42
#     )
#     model.fit(housing_prepared, housing_labels)

#     joblib.dump(model, MODEL_FILE)
#     joblib.dump(pipeline, PIPELINE_FILE)

#     print("✅ Model trained and saved successfully")


# else:
#     print("Loading model & making predictions...")

#     model = joblib.load(MODEL_FILE)
#     pipeline = joblib.load(PIPELINE_FILE)

#     # Make sure this file exists
#     input_data = pd.read_csv(TRAIN_INPUT_FILE)

#     transformed_input = pipeline.transform(input_data)
#     predictions = model.predict(transformed_input)

#     input_data["median_house_value"] = predictions
#     input_data.to_csv(OUTPUT_FILE, index=False)

#     print("✅ Prediction completed & saved to output.csv")


import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# ================= ABSOLUTE PATH FIX =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
PIPELINE_FILE = os.path.join(BASE_DIR, "pipeline.pkl")
INPUT_FILE = os.path.join(BASE_DIR, "input.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "output.csv")
HOUSING_FILE = os.path.join(BASE_DIR, "housing.csv")

print("BASE_DIR:", BASE_DIR)
print("MODEL EXISTS:", os.path.exists(MODEL_FILE))


# ================= PIPELINE =================
def build_pipeline(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])


# ================= MAIN LOGIC =================
if not os.path.exists(MODEL_FILE):

    print("🔵 TRAINING MODE")

    housing = pd.read_csv(HOUSING_FILE)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )

    for train_idx, _ in split.split(housing, housing["income_cat"]):
        housing = housing.loc[train_idx].drop("income_cat", axis=1)

    housing_labels = housing["median_house_value"]
    housing_features = housing.drop("median_house_value", axis=1)

    # save input sample for prediction
    housing_features.head(10).to_csv(INPUT_FILE, index=False)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("✅ MODEL TRAINED & SAVED")


else:
    print("🟢 PREDICTION MODE")

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv(INPUT_FILE)

    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)

    input_data["median_house_value"] = predictions
    input_data.to_csv(OUTPUT_FILE, index=False)

    print("✅ OUTPUT SAVED TO output.csv")
