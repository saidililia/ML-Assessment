import pandas as pd
import numpy as np
import xgboost as xgb

print("Training model to populate December predictions...")
# 1. Load training data
train_df = pd.read_csv("data/train-test-engineered.csv")

# Feature engineering for training
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['month'] = train_df['date'].dt.month
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['is_weekend'] = train_df['day_of_week'].isin([5, 6]).astype(int)
train_df['weight_per_mile'] = train_df['weight'] / train_df['distance'].replace(0, np.nan)
train_df['market_distance_interaction'] = train_df['market_index'] * train_df['distance']
train_df['lane'] = train_df['pickup'].astype(str) + " -> " + train_df['delivery'].astype(str)
train_df = train_df.dropna()

train_lanes = set(train_df['lane'].astype(str).unique())
train_lanes.add('Unknown')
train_equip = set(train_df['equipment'].astype(str).unique())
train_equip.add('Unknown')

train_df['lane'] = pd.Categorical(train_df['lane'].astype(str), categories=list(train_lanes))
train_df['equipment'] = pd.Categorical(train_df['equipment'].astype(str), categories=list(train_equip))

drop_cols = ['load_id', 'pickup', 'delivery', 'date', 'posted_rate']
X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
y_train = train_df['posted_rate']

# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=150, 
    learning_rate=0.08, 
    max_depth=6, 
    random_state=42, 
    n_jobs=-1,
    enable_categorical=True
)
model.fit(X_train, y_train)

# 2. Load December chart inputs
dec_df = pd.read_csv("data/december-chart-inputs.csv")

# Apply identical feature engineering
dec_df['date'] = pd.to_datetime(dec_df['date'])
dec_df['month'] = dec_df['date'].dt.month
dec_df['day_of_week'] = dec_df['date'].dt.dayofweek
dec_df['is_weekend'] = dec_df['day_of_week'].isin([5, 6]).astype(int)
dec_df['weight_per_mile'] = dec_df['weight'] / dec_df['distance'].replace(0, np.nan)
dec_df['lane'] = dec_df['pickup'].astype(str) + " -> " + dec_df['delivery'].astype(str)

# Handle unseen lanes/equipment in December data
dec_df.loc[~dec_df['lane'].astype(str).isin(train_lanes), 'lane'] = 'Unknown'
dec_df.loc[~dec_df['equipment'].astype(str).isin(train_equip), 'equipment'] = 'Unknown'

dec_df['lane'] = pd.Categorical(dec_df['lane'].astype(str), categories=list(train_lanes))
dec_df['equipment'] = pd.Categorical(dec_df['equipment'].astype(str), categories=list(train_equip))

# Drop non-feature columns if present
dec_features_to_drop = [col for col in ['pickup', 'delivery', 'date', 'predicted_rate'] if col in dec_df.columns]
X_dec = dec_df.drop(columns=dec_features_to_drop)

# --- INTELLIGENT COLUMN ALIGNMENT & IMPUTATION ---
# If any column expected by the model is missing from dec_df, fill it using train medians or defaults
for col in X_train.columns:
    if col not in X_dec.columns:
        if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
            X_dec[col] = train_df[col].median()
        else:
            X_dec[col] = 0

# Ensure exact column order match with X_train
X_dec = X_dec[X_train.columns]

# 3. Predict and update December file
dec_predictions = model.predict(X_dec)
dec_df['predicted_rate'] = dec_predictions

# Save back to the exact path expected by score.py
dec_df.to_csv("data/december-chart-inputs.csv", index=False)
print("Success! 'data/december-chart-inputs.csv' updated with valid predicted rates.")