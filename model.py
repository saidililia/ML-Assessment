# Train and validate your model using data/train_test.csv. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# 1. Load engineered training and testing data
train = pd.read_csv("data/train-engineered.csv")
test = pd.read_csv("data/test-engineered.csv")

print("Loading processed data...")

# 2. Define Features (X) and Target (y)
# Drop target identifier, category and already processed columns
drop_cols = ['load_id', 'pickup', 'delivery', 'date', 'posted_rate']

X_train = train.drop(columns=drop_cols)
y_train = train['posted_rate']

X_test = test.drop(columns=drop_cols)
y_test = test['posted_rate']

# 3. Train Random Forest
print("Training Random Forest Regressor...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
joblib.dump(model, "data/random_forest_model.pkl")
print("Random Forest model saved.")


# 4. Make Predictions
predictions = model.predict(X_test)

# 5. Evaluate Performance
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n--- Model Performance ---")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# 6. Extract Feature Importances
importances = model.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n--- Top 5 Most Important Features ---")
print(importance_df.head(5))

# 7. Plot Feature Importances
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df.head(10)['Feature'],
    importance_df.head(10)['Importance'],
    color='purple'
)

plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()

plt.savefig('data/feature_importance.png')

print("\nPlot successfully saved as 'feature_importance.png'!")

plt.show()
