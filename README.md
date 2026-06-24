# 🏠 CalPriceAI — Full-Stack ML House Price Prediction System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Netlify-00C7B7?style=for-the-badge&logo=netlify)](https://shiny-toffee-61a83c.netlify.app)
[![Backend API](https://img.shields.io/badge/Backend%20API-Railway-0B0D0E?style=for-the-badge&logo=railway)](https://full-stack-ml-house-price-prediction-production.up.railway.app)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://full-stack-ml-house-price-prediction-production.up.railway.app/docs)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/arsal1947/full-stack-ml-house-price-prediction)

> A production-grade, end-to-end Machine Learning web application that predicts California house prices in real time using a trained **Random Forest Regressor** model, served via a **FastAPI** backend and a sleek **vanilla HTML/CSS/JS** frontend.

---

## 🌐 Live Links

| Resource | URL |
|---|---|
| 🖥️ Frontend (Netlify) | https://shiny-toffee-61a83c.netlify.app |
| ⚡ Backend API (Railway) | https://full-stack-ml-house-price-prediction-production.up.railway.app |
| 📖 Interactive API Docs | https://full-stack-ml-house-price-prediction-production.up.railway.app/docs |

---

## 🚀 Key Features

- 🔮 **Real-time house price prediction** via REST API
- ⚡ **FastAPI backend** for high-performance ML inference
- 🌐 **Vanilla JS frontend** with fetch API integration
- 📊 **Trained on California Housing Dataset** (20,640 samples, sklearn)
- 🌲 **Random Forest Regressor** — 100 estimators, ~$39K MAE
- 🔒 **Input validation** using Pydantic v2
- 🌍 **CORS-enabled** for cross-origin requests
- 🤗 **Model hosted on HuggingFace Hub** (auto-downloaded at startup)
- ☁️ **Fully deployed** — Railway (backend) + Netlify (frontend)
- 🎨 **Dark UI** with quick presets (SF, LA, Suburb, Rural)

---

## 🧠 Machine Learning Model

### Dataset
- **California Housing Dataset** from `sklearn.datasets`
- 20,640 block groups from the 1990 US Census

### Algorithm
- **Random Forest Regressor** (100 estimators)

### Features Used

| Feature | Description |
|---|---|
| `MedInc` | Median income of the block group (×$10K) |
| `HouseAge` | Median age of houses in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude (32°–42°N) |
| `Longitude` | Block group longitude (-125°–-114°W) |

### Evaluation Metrics
- **MAE:** ~$39,000
- **R² Score:** ~0.81

---

## 🏗️ Project Structure

```
full-stack-ml-house-price-prediction/
│
├── house-price-predictor/
│   │
│   ├── backend/
│   │   ├── main.py                 # FastAPI app — routes, model loading, inference
│   │   ├── train.py                # Model training pipeline
│   │   ├── explore.py              # Dataset EDA
│   │   └── house_features.joblib   # Saved feature list
│   │
│   ├── frontend/
│   │   └── index.html              # Full frontend — HTML + CSS + JS (single file)
│   │
│   ├── Procfile                    # Railway/Render start command
│   ├── requirements.txt            # Python dependencies
│   └── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/arsal1947/full-stack-ml-house-price-prediction.git
cd full-stack-ml-house-price-prediction/house-price-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (first time only)

```bash
python backend/train.py
```

### 4. Run the FastAPI server

```bash
uvicorn backend.main:app --reload
```

Server starts at:
```
http://127.0.0.1:8000
```

### 5. Open the frontend

Simply open `frontend/index.html` in your browser. Make sure the `API` variable in the script points to `http://127.0.0.1:8000` for local development.

---

## 🌐 API Endpoints

### `GET /`
Returns basic API status.

**Response:**
```json
{
  "message": "California House Price Prediction API"
}
```

---

### `GET /health`
Returns model status and loaded features.

**Response:**
```json
{
  "status": "running",
  "model": "loaded",
  "features": ["MedInc", "HouseAge", "AveRooms", ...]
}
```

---

### `POST /predict`
Predicts house price from input features.

**Request Body:**
```json
{
  "MedInc": 8.3,
  "HouseAge": 30,
  "AveRooms": 6.0,
  "AveBedrms": 1.0,
  "Population": 1500,
  "AveOccup": 3.0,
  "Latitude": 34.0,
  "Longitude": -118.0
}
```

**Response:**
```json
{
  "predicted_price": "$450,000",
  "predicted_price_short": "$4.50 hundred thousand",
  "confidence_range": "$411,000 to $489,000"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn (Random Forest) |
| Model Storage | HuggingFace Hub |
| Backend | FastAPI + Pydantic v2 + Uvicorn |
| Data Processing | Pandas + NumPy |
| Model Serialization | Joblib |
| Frontend | Vanilla HTML + CSS + JavaScript |
| Backend Deployment | Railway |
| Frontend Deployment | Netlify |

---

## ☁️ Deployment

### Backend — Railway

| Setting | Value |
|---|---|
| Root Directory | `house-price-predictor` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |

### Frontend — Netlify

Deployed via Netlify Drop — single `index.html` file with no build step required.

---

## 🎯 Project Goal

To demonstrate a complete production-style machine learning system:

```
Data Collection → EDA → Model Training → API Development → Web Integration → Cloud Deployment
```

This project showcases skills across the full ML engineering lifecycle — not just model building, but serving, validating, and deploying it in a real production environment.

---

## 👨‍💻 Author

**Arsal** — AI/ML Engineer & CS Student at PUCIT, Lahore

- 🔗 [GitHub](https://github.com/arsal1947)
- 💼 Specializing in Agentic AI, LangGraph, FastAPI, and full-stack ML systems
