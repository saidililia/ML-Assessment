import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
df['market_distance_interaction'] = df['market_index'] * df['distance'] # market index serves as a proxy for market demand, and distance is a proxy for transportation cost. The interaction term captures the combined effect of market demand and transportation cost on the posted rate.

# --- 4. TRAIN / TEST SPLIT ---
train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# --- 5. ONE-HOT ENCODING: EQUIPMENT ---
train = pd.get_dummies(train, columns=['equipment'], dtype=int)
test = pd.get_dummies(test, columns=['equipment'], dtype=int)

# Make sure train and test have the same columns
test = test.reindex(columns=train.columns, fill_value=0)

# --- 6. TARGET ENCODING: LANE ---
# Calculate lane averages using TRAINING DATA ONLY
lane_mean = train.groupby('lane')['posted_rate'].mean()

# Apply those averages to both datasets
train['lane_target_encoded'] = train['lane'].map(lane_mean)
test['lane_target_encoded'] = test['lane'].map(lane_mean)

# For lanes that only appear in test, use the overall training mean
train_mean = train['posted_rate'].mean()

train['lane_target_encoded'] = train['lane_target_encoded'].fillna(train_mean)
test['lane_target_encoded'] = test['lane_target_encoded'].fillna(train_mean)

# Drop original lane
train = train.drop(columns=['lane'])
test = test.drop(columns=['lane'])

print("\nTraining shape:", train.shape)
print("Testing shape:", test.shape)

# Save
train.to_csv("data/train-engineered.csv", index=False)
test.to_csv("data/test-engineered.csv", index=False)

print("Successfully saved engineered datasets!")
