import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():

    df = pd.read_csv("../Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.dropna(inplace=True)

    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    df.to_csv("../Data/processed/clean_data.csv", index=False)

    print("yeh maine print kiya hai", df) 