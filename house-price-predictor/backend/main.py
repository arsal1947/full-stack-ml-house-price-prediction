import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

app = FastAPI()

MODEL_PATH = "house_model.joblib"
FEATURES_PATH = "house_features.joblib"

REPO_ID = "arsal1947/house-price-model"

# =========================
# 1. DOWNLOAD MODEL FIRST
# =========================
if not os.path.exists(MODEL_PATH):
    print("Downloading model from HuggingFace...")
    MODEL_PATH = hf_hub_download(
        repo_id=REPO_ID,
        filename="house_model.joblib"
    )

# =========================
# 2. LOAD MODEL SAFELY
# =========================
model = joblib.load(MODEL_PATH)

# =========================
# 3. LOAD FEATURES SAFELY
# =========================
if os.path.exists(FEATURES_PATH):
    features = joblib.load(FEATURES_PATH)
else:
    print("Warning: features file not found")
    features = []

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================
# INPUT SCHEMA
# =========================
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0)
    HouseAge: float = Field(ge=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(gt=0)
    AveOccup: float = Field(gt=0)
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "California home prediction API"}

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "loaded",
        "features": features
    }

@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([house.dict()])

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:.2f} hundred thousand",
            "confidence_range": f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
