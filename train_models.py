
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# =========================
# CUSTOMER SEGMENT MODEL
# =========================

customers = pd.read_csv("customers.csv")

# Encode categorical columns
le_gender = LabelEncoder()
le_loyalty = LabelEncoder()

customers["gender"] = le_gender.fit_transform(customers["gender"])
customers["loyalty_tier"] = le_loyalty.fit_transform(customers["loyalty_tier"])

# Features
X_segment = customers[["age", "gender", "loyalty_tier"]]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_segment)

# Train clustering model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Save segmentation model
joblib.dump(kmeans, "segment_model.pkl")
joblib.dump(scaler, "segment_scaler.pkl")

print("✅ Segment model saved")


# =========================
# DEMAND FORECAST MODEL
# =========================

demand = pd.read_csv("revenue.csv")

# Features
X_demand = demand[[
    "avg_price",
    "day_of_week",
    "month",
    "is_weekend",
    "seasonal_factor",
    "promo_active"
]]

# Target
y_demand = demand["bookings_count"]

# Train regression model
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_demand, y_demand)

# Save demand model
joblib.dump(rf, "demand_model.pkl")

print("✅ Demand model saved")

print("\n🎉 ALL MODELS CREATED SUCCESSFULLY")
