import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/train-test-cleaned.csv") 

print("Starting shape:", df.shape)

# --- 1. TEMPORAL FEATURES ---
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter

# --- 2. LANE & SPATIAL FEATURES ---
df['lane'] = df['pickup'].astype(str) + " -> " + df['delivery'].astype(str)

# --- 3. NUMERIC INTERACTION FEATURES ---
df['weight_per_mile'] = df['weight'] / df['distance'].replace(0, np.nan)
df['market_distance_interaction'] = df['market_index'] * df['distance']

# --- 4. CATEGORICAL ENCODING FOR RANDOM FOREST ---
print("Applying One-Hot Encoding to equipment...")
# One-hot encode equipment to avoid false numeric ordering
df = pd.get_dummies(df, columns=['equipment'], drop_first=True)

# For lane, we keep it as a clean string for now. 
# (In your model script, you can use Target Encoding or let XGBoost handle it natively).

print("\nFinal Engineered Shape:", df.shape)

# Save the feature-engineered dataset
df.to_csv("data/train-test-engineered.csv", index=False)
print("Successfully saved engineered dataset!")