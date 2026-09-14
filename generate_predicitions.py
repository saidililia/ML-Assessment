import pandas as pd
import numpy as np
import xgboost as xgb

print("--- Step 1: Training Final Model on Full Development Data ---")
# 1. Load and clean full training data
train_df = pd.read_csv("data/train-test.csv")
train_df = train_df.dropna(subset=["weight", "market_index"])

# Feature engineering pipeline for training data
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['month'] = train_df['date'].dt.month
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['is_weekend'] = train_df['day_of_week'].isin([5, 6]).astype(int)
train_df['weight_per_mile'] = train_df['weight'] / train_df['distance'].replace(0, np.nan)
train_df['market_distance_interaction'] = train_df['market_index'] * train_df['distance']
train_df['lane'] = train_df['pickup'].astype(str) + " -> " + train_df['delivery'].astype(str)
train_df = train_df.dropna()

print("\n--- Step 2: Processing Validation Dataset ---")
# 2. Load validation dataset
val_df = pd.read_csv("data/validation.csv")

# Apply identical feature engineering pipeline
val_df['date'] = pd.to_datetime(val_df['date'])
val_df['month'] = val_df['date'].dt.month
val_df['day_of_week'] = val_df['date'].dt.dayofweek
val_df['is_weekend'] = val_df['day_of_week'].isin([5, 6]).astype(int)
val_df['weight_per_mile'] = val_df['weight'] / val_df['distance'].replace(0, np.nan)
val_df['market_distance_interaction'] = val_df['market_index'] * val_df['distance']
val_df['lane'] = val_df['pickup'].astype(str) + " -> " + val_df['delivery'].astype(str)

# --- HANDLE UNSEEN CATEGORIES ---
# Collect all valid training categories and add 'Unknown' as a fallback safety net
train_lanes = set(train_df['lane'].astype(str).unique())
train_lanes.add('Unknown')

train_equip = set(train_df['equipment'].astype(str).unique())
train_equip.add('Unknown')

# Map any unseen validation lanes/equipment to 'Unknown'
val_df.loc[~val_df['lane'].astype(str).isin(train_lanes), 'lane'] = 'Unknown'
val_df.loc[~val_df['equipment'].astype(str).isin(train_equip), 'equipment'] = 'Unknown'

# Enforce identical categorical types with explicit categories shared between train and val
train_df['lane'] = pd.Categorical(train_df['lane'].astype(str), categories=list(train_lanes))
val_df['lane'] = pd.Categorical(val_df['lane'].astype(str), categories=list(train_lanes))

train_df['equipment'] = pd.Categorical(train_df['equipment'].astype(str), categories=list(train_equip))
val_df['equipment'] = pd.Categorical(val_df['equipment'].astype(str), categories=list(train_equip))

# Define X_train and y_train
drop_cols = ['load_id', 'pickup', 'delivery', 'date', 'posted_rate']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['posted_rate']

# Train final XGBoost model
print("Training final XGBoost model...")
final_model = xgb.XGBRegressor(
    n_estimators=150, 
    learning_rate=0.08, 
    max_depth=6, 
    random_state=42, 
    n_jobs=-1,
    enable_categorical=True
)
final_model.fit(X_train, y_train)
print("Final model training complete!")

# Prepare X_val ensuring feature columns match X_train exactly
val_features_to_drop = [col for col in ['load_id', 'pickup', 'delivery', 'date'] if col in val_df.columns]
X_val = val_df.drop(columns=val_features_to_drop)
X_val = X_val[X_train.columns]

print("\n--- Step 3: Generating Predictions and Formatting Output ---")
# 3. Predict rates for the 12,000 loads
val_predictions = final_model.predict(X_val)

# Load the template file
template_df = pd.read_csv("data/validation-predictions-template.csv")

# Assign predictions and save
template_df['predicted_rate'] = val_predictions
template_df.to_csv("validation_predictions.csv", index=False)

print("Success! File saved as 'validation_predictions.csv'. Ready for submission.")