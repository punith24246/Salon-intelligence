# Salon Intelligence Platform

**Salon Intelligence** is a data-driven salon operations dashboard and AI toolkit built with Python, Streamlit, FastAPI, and machine learning. It combines predictive analytics, customer segmentation, demand forecasting, review sentiment analysis, and no-show risk modeling into a single salon intelligence solution.

## 🚀 Project Highlights

- **Interactive dashboard** built in `Streamlit` for salon managers and operations teams.
- **No-show risk prediction** using a trained machine learning model and SHAP-style feature explanations.
- **Customer segmentation** with RFM-based CLV and loyalty analytics for retention campaigns.
- **Demand forecasting and dynamic pricing** recommendations to optimize salon capacity and revenue.
- **Review sentiment analysis** for customer feedback, rating trends, and service quality monitoring.
- **A/B testing experiment analysis** for promotional and pricing optimization.
- **FastAPI backend** for production-ready prediction endpoints and external integration.

## 📁 Key Files

- `app.py` — Streamlit dashboard for data exploration, model predictions, and business insights.
- `Salon_Intelligence_Platform.py` — end-to-end data science project script that includes data simulation, exploration, and analytics.
- `train_models.py` — model training workflow for customer segmentation and demand forecasting.
- `api_and_deployment.py` — FastAPI service with endpoints for no-show risk, customer segmentation, and demand forecasting.
- `customers.csv`, `bookings.csv`, `revenue.csv`, `reviews.csv` — sample datasets powering the dashboard and models.
- `demand_model.pkl`, `segment_model.pkl`, `segment_scaler.pkl` — pre-trained model artifacts used in the dashboard and API.
- `screenshots/` — example dashboard visuals and results snapshots.

## 🗂️ Folder Structure

```
Salon-intelligence/
├─ app.py
├─ api_and_deployment.py
├─ Salon_Intelligence_Platform.py
├─ train_models.py
├─ customers.csv
├─ bookings.csv
├─ revenue.csv
├─ reviews.csv
├─ demand_model.pkl
├─ segment_model.pkl
├─ segment_scaler.pkl
├─ screenshots/
│   ├─ dashboard.png
│   └─ api_docs.png
├─ requirements.txt
└─ README.md
```

## 📊 Features

- KPI overview for bookings, revenue, customer satisfaction, and no-show performance.
- Visual analytics for revenue trend, no-show distribution, sentiment mix, and booking patterns.
- Predictive insights for appointment no-shows and suggested engagement actions.
- Demand forecast scenario planning with pricing adjustment recommendations.
- A/B testing analysis and results tracking for data-driven marketing and promotion optimization.
- Turnkey API endpoints for ML-powered salon decision support.

## 🛠️ Tech Stack

- Python
- Streamlit
- FastAPI
- Pandas / NumPy
- Plotly
- Scikit-learn
- XGBoost / Prophet (model workflow)
- Joblib / Pickle

## ▶️ Setup & Run Locally

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd Salon-intelligence
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

4. Start the API server (optional):
   ```bash
   uvicorn api_and_deployment:app --reload
   ```

## 📌 Usage

- Open the Streamlit app to view the salon analytics dashboard and navigate between sections like: Overview, No-show Predictor, Customer Segments, Demand Forecast, Review Sentiment, and A/B Test Results.
- Use the API endpoints to consume predictions from external applications or dashboards.

## 💡 Resume-friendly Summary

This repository demonstrates a complete salon intelligence solution with:
- full-stack analytics and visualization,
- predictive modeling for operational decisions,
- A/B testing and business experiment analysis,
- API deployment readiness,
- and real-world business use cases for customer retention, revenue growth, and appointment management.

## ✅ Notes

- The dashboard expects the provided CSV files to exist in the repository root.
- Model artifacts such as `segment_model.pkl`, `segment_scaler.pkl`, and `demand_model.pkl`

---


