import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Load data after cleaning and feature engineering
df = pd.read_csv("data/train-test-engineered.csv")
print("Loading processed data...")

# 2. Define Features (X) and Target (y)
# Drop raw string/date columns, including original 'pickup', 'delivery', and text 'lane'
drop_cols = ['load_id', 'pickup', 'delivery', 'lane', 'date', 'posted_rate']
X = df.drop(columns=drop_cols)

# One-hot encode categorical features like 'equipment'
X = pd.get_dummies(X, columns=['equipment'], drop_first=True)

y = df['posted_rate']

# 3. Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Baseline Model
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