import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="House Price Prediction API",
    docs_url="/docs",
    redoc_url="/redoc"
)


from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
PIPELINE_FILE = os.path.join(BASE_DIR, "pipeline.pkl")

# ================= LOAD MODEL =================
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# ================= FASTAPI =================
app = FastAPI(title="House Price Prediction API")

# ================= INPUT SCHEMA =================
class HouseInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

# ================= ROUTES =================
@app.get("/")
def home():
    return {"message": "House Price Prediction API Running 🚀"}

@app.post("/predict")
def predict_price(data: HouseInput):

    input_df = pd.DataFrame([data.dict()])
    transformed = pipeline.transform(input_df)
    prediction = model.predict(transformed)

    return {
        "predicted_house_price": float(prediction[0])
    }
