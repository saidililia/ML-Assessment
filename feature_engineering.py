import pandas as pd
import numpy as np

# Load the cleaned dataset (or use df directly if running sequentially)
df = pd.read_csv("data/train-test-cleaned.csv") 

# For demonstration, assuming df is loaded from your pipeline:
# df = pd.read_csv("data/train-test.csv")
df = df.dropna(subset=["weight", "market_index"]) # Apply cleaning drop

print("Starting shape:", df.shape)

# --- 1. TEMPORAL FEATURES ---
print("Extracting temporal features...")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter

# --- 2. LANE & SPATIAL FEATURES ---
print("Creating lane features...")
# Combining origin and destination into a single route lane
df['lane'] = df['pickup'].astype(str) + " -> " + df['delivery'].astype(str)

# --- 3. NUMERIC INTERACTION FEATURES ---
print("Calculating interaction features...")
# Avoid division by zero by adding a tiny epsilon or checking distance > 0
df['weight_per_mile'] = df['weight'] / df['distance'].replace(0, np.nan)

# Interaction between market index and distance
df['market_distance_interaction'] = df['market_index'] * df['distance']

# --- 4. CATEGORICAL ENCODING PREPARATION ---
# Equipment and lane are high-cardinality or categorical. 
# We can map them or leave them ready for One-Hot Encoding / Target Encoding later.
print("Categorical columns ready for encoding:", ['equipment', 'lane'])

print("\nFinal Engineered Shape:", df.shape)
print("\nNew columns added:")
print([col for col in df.columns if col not in ['load_id', 'pickup', 'delivery', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'distance', 'equipment', 'weight', 'date', 'market_index', 'quote_signal', 'posted_rate']])

# Save the feature-engineered dataset
df.to_csv("data/train-test-engineered.csv", index=False)