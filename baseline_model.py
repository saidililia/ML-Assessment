import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data and apply cleaning
print("Loading and cleaning data...")
df = pd.read_csv("data/train-test-cleaned.csv")
df = df.dropna(subset=["weight", "market_index"])

# 2. Apply Feature Engineering
print("Engineering features...")
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['weight_per_mile'] = df['weight'] / df['distance'].replace(0, np.nan)
df['market_distance_interaction'] = df['market_index'] * df['distance']

# Drop rows with any remaining NaNs from division
df = df.dropna()

# 3. Define Features (X) and Target (y)
# Drop non-predictive columns or raw text strings for this baseline
drop_cols = ['load_id', 'pickup', 'delivery', 'date', 'posted_rate']
X = df.drop(columns=drop_cols)

# One-hot encode categorical features like 'equipment'
X = pd.get_dummies(X, columns=['equipment'], drop_first=True)

y = df['posted_rate']

# 4. Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Baseline Model
print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate performance
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"\n--- Model Performance ---")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# 6. Extract and Plot Feature Importances
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Most Important Features ---")
print(importance_df.head(5))

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title('Top 10 Feature Importances (Baseline Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nPlot successfully saved as 'feature_importance.png'!")
plt.show()