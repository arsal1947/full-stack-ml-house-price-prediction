# 🏠 Full-Stack ML House Price Prediction System

A production-style end-to-end Machine Learning web application that predicts California house prices using a trained **Random Forest Regressor** model.  

This project demonstrates the complete ML lifecycle:  
**Data → Training → Model → FastAPI Backend → Deployment-ready System**

---

## 🚀 Key Features

- 🔮 Real-time house price prediction via REST API  
- ⚡ FastAPI backend for high-performance inference  
- 🌐 Frontend integration using JavaScript (fetch API)  
- 📊 Trained on California Housing Dataset (Scikit-learn)  
- 🧠 Random Forest Regression model  
- 🔒 Input validation using Pydantic  
- 🌍 CORS-enabled for cross-origin requests  
- ☁️ Deployment-ready architecture (Render + Vercel)

---

## 🧠 Machine Learning Model

### 📌 Dataset
- California Housing Dataset (Scikit-learn)

### 📌 Algorithm
- Random Forest Regressor

### 📌 Features Used
- Median Income  
- House Age  
- Average Rooms  
- Average Bedrooms  
- Population  
- Average Occupancy  
- Latitude  
- Longitude  

### 📊 Evaluation Metrics
- Mean Absolute Error (MAE)  
- R² Score  

---

## 🏗️ Project Structure

```

house-price-api/
│
├── explore.py                # Dataset exploration (EDA)
├── train.py                  # Model training pipeline
├── main.py                   # FastAPI backend
│
├── house_model.joblib        # Trained ML model
├── house_features.joblib     # Feature list for inference
│
├── frontend/
│   └── index.html            # Simple frontend UI
│
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/full-stack-ml-house-price-prediction.git
cd full-stack-ml-house-price-prediction
````

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run FastAPI server

```bash
uvicorn main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

---

## 🌐 API Endpoints

### 🔹 Home

```
GET /
```

Returns basic API status.

---

### 🔹 Health Check

```
GET /health
```

Returns model status and loaded features.

---

### 🔹 Predict House Price

```
POST /predict
```

---

### 📌 Request Body

```json
{
  "MedInc": 8.3,
  "HouseAge": 30,
  "AveRooms": 6,
  "AveBedrms": 1,
  "Population": 1500,
  "AveOccup": 3,
  "Latitude": 34,
  "Longitude": -118
}
```

---

### 📌 Response

```json
{
  "predicted_price": "$450,000",
  "predicted_price_short": "$4.50 hundred thousand",
  "confidence_range": "$410,000 to $490,000"
}
```

---

## 🌍 Deployment

This project is designed for cloud deployment.

### 🚀 Backend Deployment

* Render
* Railway

### 🌐 Frontend Deployment

* Vercel
* Netlify

---

## 🛠️ Tech Stack

* Python 🐍
* FastAPI ⚡
* Scikit-learn 🤖
* Pandas 📊
* NumPy 🔢
* HTML / JavaScript 🌐
* Joblib 💾

---

## 🎯 Project Goal

To demonstrate a complete production-style machine learning system:

> Data Collection → Model Training → API Development → Web Integration → Deployment

---

## 👨‍💻 Author

Built as a full-stack machine learning portfolio project showcasing ML engineering, backend development, and deployment skills.

---
