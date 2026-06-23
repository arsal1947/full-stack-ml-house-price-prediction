import os
import joblib
import pandas as pd
import gdown
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# =========================
# CONFIG
# =========================
MODEL_PATH = "house_model.joblib"
FEATURES_PATH = "house_features.joblib"

# Google Drive file ID (from your link)
FILE_ID = "11_R8EhbLg5UTMTJIHkz7PJbyOV6CY4r3"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# =========================
# DOWNLOAD MODEL IF NOT EXISTS
# =========================
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# (Optional) features file handling
if not os.path.exists(FEATURES_PATH):
    print("Features file not found. Make sure you upload it to Drive too.")

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

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
        "model": "RandomForestRegressor",
        "features": features
    }

@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:.2f} hundred thousand",
            "confidence_range": f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
