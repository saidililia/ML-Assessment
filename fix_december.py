import pandas as pd
import numpy as np
import xgboost as xgb

print("Training model to populate December predictions...")
# 1. Load training data and train model
train_df = pd.read_csv("data/train-test-cleaned.csv")
train_df = train_df.dropna(subset=["weight", "market_index"])

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

# 2. Load December chart inputs safely and force-strip any unwanted extra columns
dec_df = pd.read_csv("data/december-chart-inputs.csv")
expected_original_cols = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']

# Keep only the original columns that actually exist in the file
existing_originals = [col for col in expected_original_cols if col in dec_df.columns]
dec_df = dec_df[existing_originals]

# Create a temporary copy to engineer features for prediction purposes only
dec_features = dec_df.copy()
dec_features['date'] = pd.to_datetime(dec_features['date'])
dec_features['month'] = dec_features['date'].dt.month
dec_features['day_of_week'] = dec_features['date'].dt.dayofweek
dec_features['is_weekend'] = dec_features['day_of_week'].isin([5, 6]).astype(int)
dec_features['weight_per_mile'] = dec_features['weight'] / dec_features['distance'].replace(0, np.nan)
dec_features['lane'] = dec_features['pickup'].astype(str) + " -> " + dec_features['delivery'].astype(str)

# Handle unseen categories
dec_features.loc[~dec_features['lane'].astype(str).isin(train_lanes), 'lane'] = 'Unknown'
dec_features.loc[~dec_features['equipment'].astype(str).isin(train_equip), 'equipment'] = 'Unknown'

dec_features['lane'] = pd.Categorical(dec_features['lane'].astype(str), categories=list(train_lanes))
dec_features['equipment'] = pd.Categorical(dec_features['equipment'].astype(str), categories=list(train_equip))

# Drop non-feature columns
dec_features_to_drop = [col for col in ['pickup', 'delivery', 'date', 'predicted_rate'] if col in dec_features.columns]
X_dec = dec_features.drop(columns=dec_features_to_drop)

# Impute missing columns if any are absent
for col in X_train.columns:
    if col not in X_dec.columns:
        if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
            X_dec[col] = train_df[col].median()
        else:
            X_dec[col] = 0

X_dec = X_dec[X_train.columns]

# 3. Predict rates
dec_predictions = model.predict(X_dec)

# 4. Update predicted_rate and ensure strict 7-column order
dec_df['predicted_rate'] = dec_predictions
dec_df = dec_df[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']]

# Save back cleanly
dec_df.to_csv("data/december-chart-inputs.csv", index=False)
print("Success! 'data/december-chart-inputs.csv' updated with strict 7-column schema.")