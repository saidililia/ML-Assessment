import pandas as pd
import numpy as np
import joblib


# 1. Loading our trained random forest

print("Loading trained Random Forest...")

model = joblib.load(
    "data/random_forest_model.pkl"
)

print("Random Forest loaded.")


# 2. Feature engineering our december dataset

train_df = pd.read_csv("data/train-engineered.csv")
december_df = pd.read_csv("data/december-chart-inputs.csv")

# Drop columns that are not features for the model, and use X_train value for our feature engineering
X_train = train_df.drop(
    columns=[
        'load_id',
        'pickup',
        'delivery',
        'date',
        'posted_rate'
    ]
)

y_train = train_df['posted_rate']

# December dataframe processed features
X_december = december_df.copy()

X_december['date'] = pd.to_datetime(X_december['date'])

X_december['month'] = X_december['date'].dt.month
X_december['day_of_week'] = X_december['date'].dt.dayofweek
X_december['is_weekend'] = (
    X_december['day_of_week'].isin([5, 6]).astype(int)
)
X_december['quarter'] = X_december['date'].dt.quarter

X_december['lane'] = (
    X_december['pickup'].astype(str)
    + " -> "
    + X_december['delivery'].astype(str)
)

X_december['weight_per_mile'] = (
    X_december['weight']
    / X_december['distance'].replace(0, np.nan)
)

# December does not contain market_index or quote_signal, so these features will be filled with training medians.
X_december['market_index'] = X_train['market_index'].median()
X_december['quote_signal'] = X_train['quote_signal'].median()

X_december['market_distance_interaction'] = (
    X_december['market_index']
    * X_december['distance']
)



# Target Lane Encoding. Use only training data to calculate lane averages
cleaned_df = pd.read_csv(
    "data/train-test-cleaned.csv"
)

cleaned_df['lane'] = (
    cleaned_df['pickup'].astype(str)
    + " -> "
    + cleaned_df['delivery'].astype(str)
)

# Use the lane_target_encoded values already calculated in train-engineered.csv to create the mapping.
lane_mapping = (train_df[['lane_target_encoded']])

# Since the raw lane is not in train-engineered.csv, calculate December lane encoding from the training split.
from sklearn.model_selection import train_test_split

train_cleaned, _ = train_test_split(
    cleaned_df,
    test_size=0.2,
    random_state=42
)

lane_means = (
    train_cleaned
    .groupby('lane')['posted_rate']
    .mean()
)

X_december['lane_target_encoded'] = (
    X_december['lane']
    .map(lane_means)
    .fillna(y_train.mean())
)

# One-hot encode equipment
X_december = pd.get_dummies(
    X_december,
    columns=['equipment'],
    dtype=int
)


# Prepare model input

X_december = X_december.drop(
    columns=[
        'pickup',
        'delivery',
        'date',
        'lane',
        'predicted_rate'
    ]
)


# Match training columns exactly
X_december = X_december.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Handle missing values
X_december = X_december.replace(
    [np.inf, -np.inf],
    np.nan
)

X_december = X_december.fillna(
    X_train.median(numeric_only=True)
)


# 3. December predictions
predictions = model.predict(X_december)


# Update only predicted_rate

december_df['predicted_rate'] = predictions

# Explicitly keep only the required columns
december_df = december_df[
    [
        'pickup',
        'delivery',
        'distance',
        'equipment',
        'weight',
        'date',
        'predicted_rate'
    ]
]


# Save the results

december_df.to_csv(
    "data/december-chart-inputs.csv",
    index=False
)

print("Success!")
print("December file updated.")
print(december_df.head())
