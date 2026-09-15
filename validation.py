import pandas as pd
import numpy as np
import joblib


# 1. Loading our trained random forest

print("Loading trained Random Forest...")

model = joblib.load(
    "data/random_forest_model.pkl"
)

print("Random Forest loaded.")


# 1. Load engineered training data
train_df = pd.read_csv("data/train-engineered.csv")

X_train = train_df.drop(columns=['load_id', 'pickup', 'delivery', 'date', 'posted_rate'])
y_train = train_df['posted_rate']


# 3. Load validation data
validation_df = pd.read_csv("data/validation.csv")

# 4. Apply the same feature engineering

validation_df['date'] = pd.to_datetime(validation_df['date'])

validation_df['month'] = validation_df['date'].dt.month
validation_df['day_of_week'] = validation_df['date'].dt.dayofweek
validation_df['is_weekend'] = (
    validation_df['day_of_week'].isin([5, 6]).astype(int)
)
validation_df['quarter'] = validation_df['date'].dt.quarter

validation_df['lane'] = (
    validation_df['pickup'].astype(str)
    + " -> "
    + validation_df['delivery'].astype(str)
)

validation_df['weight_per_mile'] = (
    validation_df['weight']
    / validation_df['distance'].replace(0, np.nan)
)

validation_df['market_distance_interaction'] = (
    validation_df['market_index']
    * validation_df['distance']
)

# 5. Target encode lane using TRAINING DATA ONLY

from sklearn.model_selection import train_test_split


# Get the lane from the cleaned training data to ensure consistency
cleaned_df = pd.read_csv("data/train-test-cleaned.csv")

train_cleaned, test_cleaned = train_test_split(
    cleaned_df,
    test_size=0.2,
    random_state=42
)

train_cleaned['lane'] = (
    train_cleaned['pickup'].astype(str)
    + " -> "
    + train_cleaned['delivery'].astype(str)
)

lane_mean = train_cleaned.groupby('lane')['posted_rate'].mean()


validation_df['lane_target_encoded'] = (
    validation_df['lane']
    .map(lane_mean)
    .fillna(y_train.mean())
)

# 6. One-hot encode equipment

validation_df = pd.get_dummies(
    validation_df,
    columns=['equipment'],
    dtype=int
)

# 7. Prepare validation features

X_validation = validation_df.drop(
    columns=[
        'load_id',
        'pickup',
        'delivery',
        'date',
        'lane'
    ]
)

# Make sure validation has exactly the same features as training
X_validation = X_validation.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Handle missing values
X_validation = X_validation.replace(
    [np.inf, -np.inf],
    np.nan
)

X_validation = X_validation.fillna(
    X_train.median(numeric_only=True)
)

# 8. Generate predictions

predictions = model.predict(X_validation)

# 9. Fill prediction template

predictions_df = pd.read_csv(
    "data/validation-predictions-template.csv"
)

predictions_df['predicted_rate'] = predictions

predictions_df.to_csv(
    "data/validation-predictions-template.csv",
    index=False
)

print("Success! Predicted rates filled in.")
print(predictions_df.head())

