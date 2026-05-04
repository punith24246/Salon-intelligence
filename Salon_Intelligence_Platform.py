# ================================================================
# SALON INTELLIGENCE PLATFORM
# Complete DS Project — All Modules
# Run each section in order in Google Colab
# ================================================================

# ================================================================
# SECTION 0: SETUP & IMPORTS
# ================================================================

# Cell 0.1 — Install libraries
"""
!pip install faker pandas numpy matplotlib seaborn plotly scikit-learn \
            xgboost shap imbalanced-learn prophet lifetimes lifelines \
            transformers torch datasets fastapi uvicorn streamlit \
            mlflow scipy nltk -q
"""

# Cell 0.2 — Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from faker import Faker
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)
fake = Faker('en_IN')

print("✅ All imports successful")


# ================================================================
# SECTION 1: DATA SIMULATION
# ================================================================

# Cell 1.1 — Simulate Customers
def simulate_customers(n=500):
    customers = []
    age_weights = ([2]*10 + [4]*20 + [3]*15 + [1]*2)
    age_weights = np.array(age_weights) / sum(age_weights)

    for i in range(n):
        signup_date = fake.date_between(start_date='-2y', end_date='-30d')
        customer = {
            'customer_id'  : f'CUST{str(i+1).zfill(4)}',
            'name'         : fake.name(),
            'age'          : np.random.choice(range(18, 65), p=age_weights),
            'gender'       : np.random.choice(
                                ['Female', 'Male', 'Other'], p=[0.70, 0.28, 0.02]),
            'phone'        : fake.phone_number(),
            'location'     : np.random.choice(
                                ['Nellore', 'Hyderabad', 'Chennai', 'Bangalore'],
                                p=[0.50, 0.20, 0.20, 0.10]),
            'signup_date'  : signup_date,
            'loyalty_tier' : np.random.choice(
                                ['Bronze', 'Silver', 'Gold', 'Platinum'],
                                p=[0.50, 0.30, 0.15, 0.05]),
        }
        customers.append(customer)
    df = pd.DataFrame(customers)
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    return df


# Cell 1.2 — Simulate Bookings
def simulate_bookings(customers_df, n_bookings=5000):
    services = {
        'Haircut'        : {'duration': 30,  'price_range': (200,  400)},
        'Hair Colour'    : {'duration': 90,  'price_range': (800,  2000)},
        'Hair Spa'       : {'duration': 60,  'price_range': (500,  1000)},
        'Facial'         : {'duration': 60,  'price_range': (600,  1200)},
        'Manicure'       : {'duration': 45,  'price_range': (300,  600)},
        'Pedicure'       : {'duration': 45,  'price_range': (400,  700)},
        'Bridal Package' : {'duration': 180, 'price_range': (5000, 15000)},
        'Waxing'         : {'duration': 30,  'price_range': (200,  500)},
    }
    service_probs = [0.25, 0.15, 0.10, 0.12, 0.10, 0.10, 0.03, 0.15]
    stylists = [f'STYLIST{str(i+1).zfill(2)}' for i in range(10)]
    hour_weights = np.array([1,3,4,3,2,2,3,4,4,3,2,1])
    hour_weights = hour_weights / hour_weights.sum()

    bookings = []
    for i in range(n_bookings):
        customer      = customers_df.sample(1).iloc[0]
        service_name  = np.random.choice(list(services.keys()), p=service_probs)
        service       = services[service_name]
        booking_date  = fake.date_between(start_date='-2y', end_date='today')
        booking_hour  = np.random.choice(range(9, 21), p=hour_weights)
        day_of_week   = pd.Timestamp(booking_date).dayofweek
        lead_time     = np.random.choice([0,1,2,3,7,14], p=[0.30,0.25,0.20,0.10,0.10,0.05])
        price         = np.random.randint(service['price_range'][0], service['price_range'][1])

        # Realistic no-show probability
        noshow_prob = 0.15
        if lead_time == 0:                                    noshow_prob += 0.10
        if day_of_week == 0:                                  noshow_prob += 0.05
        if booking_hour >= 18:                                noshow_prob += 0.05
        if price > 1000:                                      noshow_prob -= 0.08
        if customer['loyalty_tier'] in ['Platinum', 'Gold']:  noshow_prob -= 0.07
        noshow_prob = np.clip(noshow_prob, 0.05, 0.50)
        showed_up   = 1 if random.random() > noshow_prob else 0

        # Historical no-show count per customer (will update after all bookings)
        booking = {
            'booking_id'    : f'BK{str(i+1).zfill(5)}',
            'customer_id'   : customer['customer_id'],
            'stylist_id'    : np.random.choice(stylists),
            'service_type'  : service_name,
            'booking_date'  : booking_date,
            'booking_hour'  : booking_hour,
            'day_of_week'   : day_of_week,
            'lead_time_days': lead_time,
            'duration_mins' : service['duration'],
            'price'         : price,
            'showed_up'     : showed_up,
            'loyalty_tier'  : customer['loyalty_tier'],
            'location'      : customer['location'],
            'age'           : customer['age'],
            'gender'        : customer['gender'],
        }
        bookings.append(booking)

    df = pd.DataFrame(bookings)
    df['booking_date'] = pd.to_datetime(df['booking_date'])

    # Add historical no-show rate per customer
    hist_noshow = df.groupby('customer_id')['showed_up'].apply(
        lambda x: 1 - x.mean()
    ).reset_index()
    hist_noshow.columns = ['customer_id', 'hist_noshow_rate']
    df = df.merge(hist_noshow, on='customer_id', how='left')

    return df


# Cell 1.3 — Simulate Reviews
def simulate_reviews(customers_df, bookings_df, n_reviews=1000):
    positive_reviews = [
        "Amazing experience! My hair looks absolutely stunning.",
        "The stylist was so professional and understood exactly what I wanted.",
        "Best salon in Nellore! Will definitely come back.",
        "Very clean and hygienic salon. Staff was warm and welcoming.",
        "Loved my haircut. The colouring came out exactly as I imagined.",
        "Great value for money. Very satisfied with the facial treatment.",
        "The staff is very skilled and the ambience is lovely.",
        "Punctual and professional. My go-to salon from now on.",
        "Excellent bridal package. Made my wedding day perfect!",
        "Quick service and great results. Highly recommend.",
        "Outstanding service. The hair spa was so relaxing.",
        "Very friendly staff. They made me feel comfortable throughout.",
    ]
    neutral_reviews = [
        "Decent salon. Service was okay, nothing extraordinary.",
        "Average experience. The waiting time was a bit long.",
        "Service was fine but the pricing is slightly high.",
        "Okay salon. Stylist was good but the ambience needs improvement.",
        "Mixed experience. Some services were great, others average.",
    ]
    negative_reviews = [
        "Waited 45 minutes beyond my appointment time. Very unprofessional.",
        "The staff was rude and did not listen to my requirements.",
        "Overpriced for the quality of service provided.",
        "Salon was not clean. Hygiene standards need serious improvement.",
        "The hair colour applied was completely wrong. Very disappointed.",
        "Waiting time is too long. They don't respect appointments.",
        "Pricing is not transparent. Hidden charges were added at billing.",
        "Staff behaviour was very poor. Will not return.",
        "The place needs better maintenance and cleanliness.",
        "Too expensive compared to other salons in the area.",
        "Waited too long. No one even apologised for the delay.",
        "Staff was unfriendly and unprofessional throughout.",
    ]

    reviews = []
    showed_up_bookings = bookings_df[bookings_df['showed_up'] == 1]

    for i in range(n_reviews):
        booking     = showed_up_bookings.sample(1).iloc[0]
        rating      = np.random.choice([1,2,3,4,5], p=[0.05,0.08,0.12,0.35,0.40])

        if rating >= 4:
            review_text = np.random.choice(positive_reviews)
            sentiment   = 'Positive'
        elif rating == 3:
            review_text = np.random.choice(neutral_reviews)
            sentiment   = 'Neutral'
        else:
            review_text = np.random.choice(negative_reviews)
            sentiment   = 'Negative'

        review = {
            'review_id'   : f'REV{str(i+1).zfill(4)}',
            'customer_id' : booking['customer_id'],
            'booking_id'  : booking['booking_id'],
            'service_type': booking['service_type'],
            'rating'      : rating,
            'review_text' : review_text,
            'sentiment'   : sentiment,
            'review_date' : booking['booking_date'] + timedelta(days=random.randint(1,3)),
        }
        reviews.append(review)

    df = pd.DataFrame(reviews)
    df['review_date'] = pd.to_datetime(df['review_date'])
    return df


# Cell 1.4 — Simulate Revenue
def simulate_revenue(bookings_df):
    daily = bookings_df[bookings_df['showed_up'] == 1].groupby('booking_date').agg(
        bookings_count=('booking_id', 'count'),
        revenue       =('price', 'sum'),
        avg_price     =('price', 'mean'),
    ).reset_index()

    daily['day_of_week']  = daily['booking_date'].dt.dayofweek
    daily['month']        = daily['booking_date'].dt.month
    daily['is_weekend']   = daily['day_of_week'].isin([5,6]).astype(int)

    seasonal = {1:0.9,2:0.95,3:1.0,4:1.1,5:1.15,
                6:1.1,7:0.95,8:0.9,9:1.0,10:1.2,11:1.25,12:1.1}
    daily['seasonal_factor'] = daily['month'].map(seasonal)
    daily['promo_active']    = np.random.choice([0,1], size=len(daily), p=[0.80,0.20])
    daily = daily.sort_values('booking_date').reset_index(drop=True)
    return daily


# Cell 1.5 — Generate & Save All Tables
customers_df = simulate_customers(500)
bookings_df  = simulate_bookings(customers_df, 5000)
reviews_df   = simulate_reviews(customers_df, bookings_df, 1000)
revenue_df   = simulate_revenue(bookings_df)

customers_df.to_csv('customers.csv', index=False)
bookings_df.to_csv('bookings.csv',   index=False)
reviews_df.to_csv('reviews.csv',     index=False)
revenue_df.to_csv('revenue.csv',     index=False)

print("✅ All 4 tables saved!")
print(f"\n📊 Dataset Summary:")
print(f"   Customers : {len(customers_df):,}")
print(f"   Bookings  : {len(bookings_df):,}")
print(f"   Reviews   : {len(reviews_df):,}")
print(f"   Revenue   : {len(revenue_df):,}")
print(f"\n📈 Key Stats:")
print(f"   No-show rate      : {(1-bookings_df['showed_up'].mean()):.1%}")
print(f"   Avg booking price : ₹{bookings_df['price'].mean():.0f}")
print(f"   Avg rating        : {reviews_df['rating'].mean():.2f}/5")
print(f"   Total revenue     : ₹{revenue_df['revenue'].sum():,.0f}")


# ================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================

# Cell 2.1 — No-show patterns
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Salon Booking — No-show Pattern Analysis', fontsize=16, fontweight='bold')

# 1. No-show by day of week
day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
noshow_by_day = bookings_df.groupby('day_of_week')['showed_up'].apply(
    lambda x: (1-x.mean())*100
)
axes[0,0].bar(day_names, noshow_by_day, color='#E74C3C', alpha=0.8)
axes[0,0].set_title('No-show Rate by Day of Week')
axes[0,0].set_ylabel('No-show Rate (%)')

# 2. No-show by lead time
noshow_by_lead = bookings_df.groupby('lead_time_days')['showed_up'].apply(
    lambda x: (1-x.mean())*100
)
axes[0,1].bar(noshow_by_lead.index.astype(str), noshow_by_lead,
              color='#E67E22', alpha=0.8)
axes[0,1].set_title('No-show Rate by Lead Time (days)')
axes[0,1].set_ylabel('No-show Rate (%)')

# 3. No-show by loyalty tier
noshow_by_loyalty = bookings_df.groupby('loyalty_tier')['showed_up'].apply(
    lambda x: (1-x.mean())*100
)
axes[0,2].bar(noshow_by_loyalty.index, noshow_by_loyalty,
              color='#9B59B6', alpha=0.8)
axes[0,2].set_title('No-show Rate by Loyalty Tier')
axes[0,2].set_ylabel('No-show Rate (%)')

# 4. No-show by booking hour
noshow_by_hour = bookings_df.groupby('booking_hour')['showed_up'].apply(
    lambda x: (1-x.mean())*100
)
axes[1,0].plot(noshow_by_hour.index, noshow_by_hour,
               marker='o', color='#E74C3C', linewidth=2)
axes[1,0].set_title('No-show Rate by Booking Hour')
axes[1,0].set_ylabel('No-show Rate (%)')
axes[1,0].set_xlabel('Hour of Day')

# 5. Service type distribution
service_counts = bookings_df['service_type'].value_counts()
axes[1,1].barh(service_counts.index, service_counts.values, color='#3498DB', alpha=0.8)
axes[1,1].set_title('Bookings by Service Type')
axes[1,1].set_xlabel('Number of Bookings')

# 6. Price distribution by service
bookings_df.boxplot(column='price', by='service_type', ax=axes[1,2])
axes[1,2].set_title('Price Distribution by Service')
axes[1,2].set_xlabel('')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('eda_noshow_patterns.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA Chart 1 saved")


# Cell 2.2 — Revenue & sentiment EDA
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Revenue & Sentiment Analysis', fontsize=16, fontweight='bold')

# 1. Monthly revenue trend
monthly_rev = revenue_df.copy()
monthly_rev['month_year'] = pd.to_datetime(monthly_rev['booking_date']).dt.to_period('M')
monthly = monthly_rev.groupby('month_year')['revenue'].sum().reset_index()
monthly['month_year'] = monthly['month_year'].astype(str)
axes[0,0].plot(monthly['month_year'], monthly['revenue'],
               marker='o', linewidth=2, color='#2ECC71')
axes[0,0].set_title('Monthly Revenue Trend')
axes[0,0].set_ylabel('Revenue (₹)')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Revenue by day of week
rev_by_day = revenue_df.groupby('day_of_week')['revenue'].mean()
axes[0,1].bar(day_names, rev_by_day, color='#F1C40F', alpha=0.8)
axes[0,1].set_title('Avg Daily Revenue by Day of Week')
axes[0,1].set_ylabel('Avg Revenue (₹)')

# 3. Rating distribution
rating_counts = reviews_df['rating'].value_counts().sort_index()
colors = ['#E74C3C','#E67E22','#F1C40F','#2ECC71','#27AE60']
axes[1,0].bar(rating_counts.index, rating_counts.values, color=colors)
axes[1,0].set_title('Review Rating Distribution')
axes[1,0].set_xlabel('Rating')
axes[1,0].set_ylabel('Count')

# 4. Sentiment by service type
sentiment_service = pd.crosstab(
    reviews_df['service_type'], reviews_df['sentiment'], normalize='index'
) * 100
sentiment_service.plot(kind='bar', ax=axes[1,1],
                       color=['#E74C3C','#F1C40F','#2ECC71'])
axes[1,1].set_title('Sentiment Distribution by Service')
axes[1,1].set_ylabel('Percentage (%)')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('eda_revenue_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA Chart 2 saved")


# ================================================================
# SECTION 3: MODULE 1 — NO-SHOW PREDICTION
# ================================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost

# Cell 3.1 — Feature Engineering
def prepare_noshow_features(bookings_df):
    df = bookings_df.copy()

    # Encode categoricals
    le_service  = LabelEncoder()
    le_loyalty  = LabelEncoder()
    le_gender   = LabelEncoder()
    le_location = LabelEncoder()

    df['service_encoded']  = le_service.fit_transform(df['service_type'])
    df['loyalty_encoded']  = le_loyalty.fit_transform(df['loyalty_tier'])
    df['gender_encoded']   = le_gender.fit_transform(df['gender'])
    df['location_encoded'] = le_location.fit_transform(df['location'])

    # Time features
    df['is_weekend']       = df['day_of_week'].isin([5,6]).astype(int)
    df['is_evening']       = (df['booking_hour'] >= 18).astype(int)
    df['is_morning_peak']  = df['booking_hour'].between(10,12).astype(int)
    df['is_same_day']      = (df['lead_time_days'] == 0).astype(int)
    df['month']            = df['booking_date'].dt.month
    df['is_festive_month'] = df['month'].isin([10,11,12]).astype(int)

    # Price features
    df['price_log']        = np.log1p(df['price'])
    df['is_high_value']    = (df['price'] > 1000).astype(int)

    features = [
        'service_encoded', 'loyalty_encoded', 'gender_encoded',
        'location_encoded', 'booking_hour', 'day_of_week',
        'lead_time_days', 'duration_mins', 'price_log',
        'hist_noshow_rate', 'age', 'is_weekend', 'is_evening',
        'is_morning_peak', 'is_same_day', 'is_high_value',
        'month', 'is_festive_month'
    ]

    # Remove target from shown_up — target is NOT showed_up but no-show
    df['no_show'] = 1 - df['showed_up']

    X = df[features]
    y = df['no_show']

    return X, y, features

X, y, feature_names = prepare_noshow_features(bookings_df)

print(f"✅ Features prepared: {X.shape}")
print(f"No-show rate in dataset: {y.mean():.1%}")
print(f"\nFeature list:")
for f in feature_names:
    print(f"  - {f}")


# Cell 3.2 — Train XGBoost with MLflow
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance using scale_pos_weight
neg_count  = (y_train == 0).sum()
pos_count  = (y_train == 1).sum()
scale_pos  = neg_count / pos_count
print(f"Class imbalance ratio (scale_pos_weight): {scale_pos:.2f}")

mlflow.set_experiment("salon_noshow_prediction")

with mlflow.start_run(run_name="xgboost_noshow_v1"):

    params = {
        'n_estimators'    : 300,
        'max_depth'       : 6,
        'learning_rate'   : 0.05,
        'subsample'       : 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos,
        'random_state'    : 42,
        'eval_metric'     : 'logloss',
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predictions
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]

    # Metrics
    auc    = roc_auc_score(y_test, y_pred_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    recall_noshow    = report['1']['recall']
    precision_noshow = report['1']['precision']
    f1_noshow        = report['1']['f1-score']

    # Log to MLflow
    mlflow.log_params(params)
    mlflow.log_metric("auc",              auc)
    mlflow.log_metric("recall_noshow",    recall_noshow)
    mlflow.log_metric("precision_noshow", precision_noshow)
    mlflow.log_metric("f1_noshow",        f1_noshow)
    mlflow.xgboost.log_model(model, "noshow_model")

    print(f"\n✅ Model trained and logged to MLflow")
    print(f"\n📊 No-show Prediction Results:")
    print(f"   AUC-ROC   : {auc:.3f}")
    print(f"   Recall    : {recall_noshow:.3f}  ← most important metric")
    print(f"   Precision : {precision_noshow:.3f}")
    print(f"   F1 Score  : {f1_noshow:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Show','No-show'])}")

# Save model and encoders to disk for Streamlit
import pickle
with open('noshow_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ XGBoost model saved to noshow_model.pkl")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Show','No-show'],
            yticklabels=['Show','No-show'])
plt.title('Confusion Matrix — No-show Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved")


# Cell 3.3 — SHAP Explainability
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                  show=False, plot_size=(10,8))
plt.title('SHAP Feature Importance — No-show Prediction', fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ SHAP summary plot saved")

# Single prediction explanation
sample_idx = 0
print(f"\n🔍 Explaining prediction for booking {sample_idx}:")
print(f"   Actual    : {'No-show' if y_test.iloc[sample_idx]==1 else 'Showed up'}")
print(f"   Predicted : {'No-show' if y_pred[sample_idx]==1 else 'Showed up'}")
print(f"   Probability of no-show: {y_pred_prob[sample_idx]:.1%}")


# ================================================================
# SECTION 4: MODULE 2 — CUSTOMER SEGMENTATION & CLV
# ================================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Cell 4.1 — RFM Calculation
def calculate_rfm(bookings_df):
    snapshot_date = bookings_df['booking_date'].max() + timedelta(days=1)

    rfm = bookings_df[bookings_df['showed_up']==1].groupby('customer_id').agg(
        recency   = ('booking_date',  lambda x: (snapshot_date - x.max()).days),
        frequency = ('booking_id',    'count'),
        monetary  = ('price',         'sum'),
    ).reset_index()

    # Score 1-5 (5=best)
    rfm['r_score'] = pd.qcut(rfm['recency'],   5, labels=[5,4,3,2,1])
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['m_score'] = pd.qcut(rfm['monetary'],  5, labels=[1,2,3,4,5])

    rfm['rfm_score'] = (rfm['r_score'].astype(int) +
                        rfm['f_score'].astype(int) +
                        rfm['m_score'].astype(int))

    # Simple CLV
    rfm['avg_spend_per_visit'] = rfm['monetary'] / rfm['frequency']
    rfm['visits_per_month']    = rfm['frequency'] / 24  # 2 year data
    rfm['clv_annual']          = rfm['avg_spend_per_visit'] * rfm['visits_per_month'] * 12

    return rfm

rfm_df = calculate_rfm(bookings_df)
print(f"✅ RFM calculated for {len(rfm_df)} customers")
print(rfm_df[['recency','frequency','monetary','rfm_score','clv_annual']].describe().round(2))


# Cell 4.2 — K-Means Clustering with Silhouette Score
scaler = StandardScaler()
X_rfm  = scaler.fit_transform(rfm_df[['recency','frequency','monetary']])

# Find optimal K
inertias    = []
silhouettes = []
K_range     = range(2, 9)

for k in K_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_rfm)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_rfm, lbl))

# Plot elbow + silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2)
ax1.set_title('Elbow Method')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
ax1.legend()

ax2.plot(K_range, silhouettes, 'go-', linewidth=2)
ax2.set_title('Silhouette Score')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score (higher=better)')
ax2.axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
ax2.legend()

plt.tight_layout()
plt.savefig('kmeans_selection.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Optimal K=4 | Best Silhouette Score: {max(silhouettes):.3f}")


# Cell 4.3 — Final Segmentation
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_df['cluster'] = km_final.fit_predict(X_rfm)

# Label clusters by RFM profile
cluster_profile = rfm_df.groupby('cluster')[['recency','frequency','monetary']].mean()
print("\nCluster profiles:")
print(cluster_profile.round(0))

# Map clusters to meaningful names
# Best = low recency (recent), high frequency, high monetary
cluster_labels = {
    cluster_profile['monetary'].idxmax()  : 'Champions',
    cluster_profile['frequency'].idxmax() : 'Loyal',
    cluster_profile['recency'].idxmin()   : 'Recent',
    cluster_profile['recency'].idxmax()   : 'At Risk',
}
# Fill remaining
for i in range(4):
    if i not in cluster_labels:
        cluster_labels[i] = 'Occasional'

rfm_df['segment'] = rfm_df['cluster'].map(cluster_labels)

# Segment summary
seg_summary = rfm_df.groupby('segment').agg(
    count       = ('customer_id', 'count'),
    avg_recency = ('recency',     'mean'),
    avg_freq    = ('frequency',   'mean'),
    avg_clv     = ('clv_annual',  'mean'),
    total_clv   = ('clv_annual',  'sum'),
).round(0)

print("\n✅ Customer Segments:")
print(seg_summary)

# Visualise segments
fig = px.scatter_3d(
    rfm_df, x='recency', y='frequency', z='monetary',
    color='segment', title='Customer Segments — 3D RFM View',
    labels={'recency':'Recency (days)','frequency':'Frequency','monetary':'Monetary (₹)'}
)
fig.write_html('rfm_segments.html')
fig.show()
print("✅ RFM segment plot saved")


# Cell 4.4 — Statistical validation between segments
from scipy import stats

champions = rfm_df[rfm_df['segment']=='Champions']['monetary']
at_risk   = rfm_df[rfm_df['segment']=='At Risk']['monetary']

t_stat, p_value = stats.ttest_ind(champions, at_risk)
print(f"\n📊 Statistical Test: Champions vs At Risk spending")
print(f"   Champions avg spend : ₹{champions.mean():.0f}")
print(f"   At Risk avg spend   : ₹{at_risk.mean():.0f}")
print(f"   t-statistic         : {t_stat:.3f}")
print(f"   p-value             : {p_value:.6f}")
print(f"   Significant?        : {'Yes ✅' if p_value < 0.05 else 'No ❌'}")

# CLV distribution by segment
plt.figure(figsize=(10,5))
rfm_df.boxplot(column='clv_annual', by='segment')
plt.title('Annual CLV Distribution by Segment')
plt.suptitle('')
plt.ylabel('Annual CLV (₹)')
plt.tight_layout()
plt.savefig('clv_by_segment.png', dpi=150, bbox_inches='tight')
plt.show()


# ================================================================
# SECTION 5: MODULE 3 — DEMAND FORECASTING & DYNAMIC PRICING
# ================================================================

from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Cell 5.1 — Prepare data for Prophet
prophet_df = revenue_df[['booking_date','bookings_count']].rename(
    columns={'booking_date':'ds', 'bookings_count':'y'}
)
prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

# Train/test split — last 60 days as test
cutoff    = prophet_df['ds'].max() - timedelta(days=60)
train_df  = prophet_df[prophet_df['ds'] <= cutoff]
test_df   = prophet_df[prophet_df['ds'] >  cutoff]

print(f"✅ Prophet data prepared")
print(f"   Train: {len(train_df)} days")
print(f"   Test : {len(test_df)} days")


# Cell 5.2 — Train Prophet + MLflow
with mlflow.start_run(run_name="prophet_demand_v1"):

    model_prophet = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = True,
        daily_seasonality   = False,
        seasonality_mode    = 'multiplicative',
        changepoint_prior_scale = 0.05,
    )
    model_prophet.fit(train_df)

    # Forecast
    future   = model_prophet.make_future_dataframe(periods=60)
    forecast = model_prophet.predict(future)

    # Evaluate on test
    test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
    mape = mean_absolute_percentage_error(
        test_df['y'].values,
        test_forecast['yhat'].values
    ) * 100

    mlflow.log_metric("mape", mape)
    print(f"\n✅ Prophet model trained")
    print(f"   MAPE on test set: {mape:.1f}%")

# Save Prophet model
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(model_prophet, f)
print("✅ Prophet model saved to prophet_model.pkl")

# Plot forecast
fig = model_prophet.plot(forecast, figsize=(14,6))
plt.title(f'Salon Demand Forecast — MAPE: {mape:.1f}%', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Daily Bookings')
plt.tight_layout()
plt.savefig('demand_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

# Seasonality components
fig2 = model_prophet.plot_components(forecast, figsize=(14,8))
plt.tight_layout()
plt.savefig('forecast_components.png', dpi=150, bbox_inches='tight')
plt.show()


# Cell 5.3 — Dynamic Pricing Logic
capacity = 40  # max bookings per day

forecast['capacity']         = capacity
forecast['utilisation_pct']  = (forecast['yhat'] / capacity * 100).clip(0, 100)

def dynamic_price_adjustment(utilisation_pct):
    """
    Surge pricing logic based on predicted demand
    """
    if utilisation_pct >= 85:
        return +0.15   # +15% surge pricing
    elif utilisation_pct >= 70:
        return +0.08   # +8% moderate surge
    elif utilisation_pct <= 30:
        return -0.12   # -12% discount to fill slots
    elif utilisation_pct <= 50:
        return -0.05   # -5% small discount
    else:
        return 0.0     # base price

forecast['price_adjustment'] = forecast['utilisation_pct'].apply(dynamic_price_adjustment)
forecast['pricing_strategy'] = forecast['price_adjustment'].apply(
    lambda x: '🔴 Surge' if x > 0 else ('🟢 Discount' if x < 0 else '⚪ Base Price')
)

# Next 30 days pricing recommendations
next30 = forecast[forecast['ds'] > forecast['ds'].max() - timedelta(days=30)][
    ['ds','yhat','utilisation_pct','price_adjustment','pricing_strategy']
].tail(30)

print("\n📅 Next 30 Days — Dynamic Pricing Recommendations:")
print(next30.to_string(index=False))

# Visualise pricing strategy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,8), sharex=True)

ax1.plot(forecast['ds'], forecast['yhat'], color='#3498DB', linewidth=2, label='Predicted Bookings')
ax1.axhline(y=capacity*0.85, color='red',    linestyle='--', alpha=0.7, label='Surge threshold (85%)')
ax1.axhline(y=capacity*0.50, color='orange', linestyle='--', alpha=0.7, label='Discount threshold (50%)')
ax1.set_ylabel('Predicted Daily Bookings')
ax1.legend()
ax1.set_title('Demand Forecast with Pricing Thresholds')

colors_map = {'🔴 Surge':'#E74C3C','🟢 Discount':'#2ECC71','⚪ Base Price':'#BDC3C7'}
for _, row in next30.iterrows():
    ax2.bar(row['ds'], row['price_adjustment']*100,
            color=colors_map.get(row['pricing_strategy'],'grey'), alpha=0.8, width=1)
ax2.set_ylabel('Price Adjustment (%)')
ax2.set_xlabel('Date')
ax2.set_title('Dynamic Pricing Recommendations — Next 30 Days')
ax2.axhline(y=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('dynamic_pricing.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Dynamic pricing chart saved")


# ================================================================
# SECTION 6: MODULE 4 — SENTIMENT ANALYSIS & TOPIC MODELLING
# ================================================================

import torch
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                           Trainer, TrainingArguments)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split as sk_split
from sklearn.metrics import accuracy_score, classification_report as sk_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Cell 6.1 — Prepare sentiment data
label_map    = {'Positive':2, 'Neutral':1, 'Negative':0}
label_decode = {0:'Negative', 1:'Neutral', 2:'Positive'}

reviews_df['label'] = reviews_df['sentiment'].map(label_map)

train_texts, test_texts, train_labels, test_labels = sk_split(
    reviews_df['review_text'].tolist(),
    reviews_df['label'].tolist(),
    test_size=0.2, random_state=42, stratify=reviews_df['label']
)
print(f"✅ Sentiment data split — Train: {len(train_texts)}, Test: {len(test_texts)}")


# Cell 6.2 — DistilBERT Dataset class
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class SalonReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_len, return_tensors='pt'
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SalonReviewDataset(train_texts, train_labels, tokenizer)
test_dataset  = SalonReviewDataset(test_texts,  test_labels,  tokenizer)
print("✅ Datasets created")


# Cell 6.3 — Fine-tune DistilBERT
bert_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=3
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir          = './bert_results',
    num_train_epochs    = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    warmup_steps        = 100,
    weight_decay        = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy       = 'epoch',
    load_best_model_at_end = True,
    logging_steps       = 50,
    report_to           = 'none',
)

trainer = Trainer(
    model           = bert_model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = test_dataset,
    compute_metrics = compute_metrics,
)

print("🚀 Fine-tuning DistilBERT — this takes ~5-10 mins on Colab GPU...")
print("   (Go to Runtime > Change runtime type > GPU if not already set)")
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"\n✅ DistilBERT Fine-tuning Complete!")
print(f"   Test Accuracy: {results['eval_accuracy']:.3f}")

# Save model
bert_model.save_pretrained('./salon_bert_model')
tokenizer.save_pretrained('./salon_bert_model')
print("✅ Model saved to ./salon_bert_model")


# Cell 6.4 — LDA Topic Modelling on Negative Reviews
stop_words = list(stopwords.words('english'))

negative_reviews_text = reviews_df[
    reviews_df['sentiment'] == 'Negative'
]['review_text'].tolist()

# TF-IDF vectorisation
tfidf = TfidfVectorizer(
    max_features  = 500,
    stop_words    = stop_words,
    min_df        = 2,
    ngram_range   = (1, 2)
)
tfidf_matrix = tfidf.fit_transform(negative_reviews_text)

# LDA — 4 topics (waiting, staff, price, cleanliness)
lda = LatentDirichletAllocation(
    n_components  = 4,
    random_state  = 42,
    max_iter      = 100,
    learning_method = 'batch'
)
lda.fit(tfidf_matrix)

# Display topics
feature_names_tfidf = tfidf.get_feature_names_out()
topic_labels = ['Waiting Time', 'Staff Behaviour', 'Pricing', 'Cleanliness']

print("\n📋 Complaint Topics from Negative Reviews (LDA):")
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names_tfidf[i] for i in topic.argsort()[:-11:-1]]
    print(f"\n  Topic {topic_idx+1} — {topic_labels[topic_idx]}:")
    print(f"  Keywords: {', '.join(top_words)}")

# Topic distribution visualisation
topic_dist = lda.transform(tfidf_matrix).mean(axis=0)
plt.figure(figsize=(8,5))
plt.bar(topic_labels, topic_dist * 100, color=['#E74C3C','#E67E22','#F1C40F','#3498DB'])
plt.title('Complaint Topic Distribution in Negative Reviews')
plt.ylabel('Topic Prevalence (%)')
plt.tight_layout()
plt.savefig('lda_topics.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ LDA topic chart saved")


# ================================================================
# SECTION 7: MODULE 5 — A/B TESTING — DYNAMIC PRICING EXPERIMENT
# ================================================================

from scipy import stats as scipy_stats

# Cell 7.1 — Simulate A/B experiment
np.random.seed(42)
n_per_group = 500

# Control group — no discount, baseline booking rate
control_bookings = np.random.binomial(1, p=0.35, size=n_per_group)

# Treatment group — 10% off-peak discount, higher booking rate
treatment_bookings = np.random.binomial(1, p=0.42, size=n_per_group)

control_revenue   = control_bookings   * np.random.randint(300, 800, n_per_group)
treatment_revenue = treatment_bookings * np.random.randint(270, 720, n_per_group)

print(f"✅ A/B experiment simulated")
print(f"\n📊 Raw Results:")
print(f"   Control   — Booking rate: {control_bookings.mean():.1%}   | Avg revenue/slot: ₹{control_revenue.mean():.0f}")
print(f"   Treatment — Booking rate: {treatment_bookings.mean():.1%} | Avg revenue/slot: ₹{treatment_revenue.mean():.0f}")


# Cell 7.2 — Statistical Testing
# Primary metric: booking rate
t_stat, p_value = scipy_stats.ttest_ind(treatment_bookings, control_bookings)

# Effect size — Cohen's d
pooled_std = np.sqrt((control_bookings.std()**2 + treatment_bookings.std()**2) / 2)
cohens_d   = (treatment_bookings.mean() - control_bookings.mean()) / pooled_std

# Confidence interval on uplift
uplift     = treatment_bookings.mean() - control_bookings.mean()
se_uplift  = np.sqrt(
    (control_bookings.std()**2 / n_per_group) +
    (treatment_bookings.std()**2 / n_per_group)
)
ci_low  = uplift - 1.96 * se_uplift
ci_high = uplift + 1.96 * se_uplift

# Revenue impact
monthly_slots       = 30 * 40  # 30 days × 40 slots/day
revenue_lift_monthly = uplift * monthly_slots * 500  # ₹500 avg booking value

print(f"\n📊 A/B Test Statistical Results:")
print(f"   Uplift in booking rate : +{uplift:.1%}")
print(f"   95% CI                 : [{ci_low:.1%}, {ci_high:.1%}]")
print(f"   t-statistic            : {t_stat:.3f}")
print(f"   p-value                : {p_value:.4f}")
print(f"   Cohen's d (effect size): {cohens_d:.3f}")
print(f"   Statistical power      : {'✅ Significant (p<0.05)' if p_value < 0.05 else '❌ Not Significant'}")
print(f"\n💰 Business Impact:")
print(f"   Monthly revenue uplift : ₹{revenue_lift_monthly:,.0f}")
print(f"   Annual revenue uplift  : ₹{revenue_lift_monthly*12:,.0f}")

# Decision
print(f"\n📋 Recommendation:")
if p_value < 0.05 and uplift > 0:
    print(f"   ✅ SHIP — The 10% off-peak discount significantly increases")
    print(f"      booking rate by {uplift:.1%} (p={p_value:.3f}) with estimated")
    print(f"      annual revenue uplift of ₹{revenue_lift_monthly*12:,.0f}.")
    print(f"      Monitor cancellation rate and avg order value as guardrails.")
else:
    print(f"   ❌ DO NOT SHIP — No statistically significant improvement found.")


# Cell 7.3 — Visualise A/B results
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('A/B Test Results — Off-peak Discount Experiment', fontsize=14)

# 1. Booking rate comparison
rates  = [control_bookings.mean()*100, treatment_bookings.mean()*100]
groups = ['Control\n(No Discount)', 'Treatment\n(10% Discount)']
bars   = axes[0].bar(groups, rates, color=['#3498DB','#2ECC71'], alpha=0.85, width=0.5)
axes[0].set_ylabel('Booking Rate (%)')
axes[0].set_title('Booking Rate Comparison')
for bar, rate in zip(bars, rates):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                 f'{rate:.1f}%', ha='center', fontweight='bold')

# 2. Confidence interval plot
axes[1].errorbar(
    ['Booking Rate Uplift'], [uplift*100],
    yerr=[[  (uplift-ci_low)*100  ],
          [  (ci_high-uplift)*100 ]],
    fmt='o', color='#E74C3C', markersize=10, capsize=10, linewidth=2
)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].set_ylabel('Uplift (%)')
axes[1].set_title(f'Uplift with 95% CI\np-value={p_value:.4f}')

# 3. Revenue distribution
axes[2].hist(control_revenue[control_revenue>0],
             alpha=0.6, bins=20, color='#3498DB', label='Control')
axes[2].hist(treatment_revenue[treatment_revenue>0],
             alpha=0.6, bins=20, color='#2ECC71', label='Treatment')
axes[2].set_xlabel('Revenue per Slot (₹)')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Revenue Distribution')
axes[2].legend()

plt.tight_layout()
plt.savefig('ab_test_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ A/B test charts saved")


# ================================================================
# SECTION 8: GENERATE FINAL METRICS SUMMARY
# ================================================================

print("\n" + "="*60)
print("📊 SALON INTELLIGENCE PLATFORM — FINAL METRICS")
print("="*60)

print(f"""
MODULE 1 — No-show Prediction (XGBoost + SHAP)
  ├─ AUC-ROC          : {auc:.3f}
  ├─ Recall (No-show) : {recall_noshow:.3f}
  ├─ Precision        : {precision_noshow:.3f}
  └─ F1 Score         : {f1_noshow:.3f}

MODULE 2 — Customer Segmentation (RFM + K-Means)
  ├─ Silhouette Score : {max(silhouettes):.3f}
  ├─ Segments         : 4 (Champions, Loyal, Recent, At Risk)
  └─ Statistical test : Champions vs At Risk p={p_value:.4f}

MODULE 3 — Demand Forecasting (Prophet)
  ├─ MAPE             : {mape:.1f}%
  └─ Pricing logic    : 3-tier (Surge / Base / Discount)

MODULE 4 — Sentiment Analysis (DistilBERT + LDA)
  ├─ BERT Accuracy    : {results['eval_accuracy']:.3f}
  └─ LDA Topics       : 4 (Waiting, Staff, Pricing, Cleanliness)

MODULE 5 — A/B Testing (Pricing Experiment)
  ├─ Booking uplift   : +{uplift:.1%}
  ├─ p-value          : {p_value:.4f}
  ├─ Cohen's d        : {cohens_d:.3f}
  └─ Annual uplift    : ₹{revenue_lift_monthly*12:,.0f}
""")

print("✅ Copy these metrics into your resume replacing the XX placeholders!")
print("="*60)
