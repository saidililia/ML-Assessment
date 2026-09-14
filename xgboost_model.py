import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

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

# Create lane feature
df['lane'] = df['pickup'].astype(str) + " -> " + df['delivery'].astype(str)

# Drop rows with any remaining NaNs from division
df = df.dropna()

# Convert text/categorical features to pandas 'category' dtype 
# so XGBoost can process them natively
df['lane'] = df['lane'].astype('category')
df['equipment'] = df['equipment'].astype('category')

# 3. Define Features (X) and Target (y)
drop_cols = ['load_id', 'pickup', 'delivery', 'date', 'posted_rate']
X = df.drop(columns=drop_cols)
y = df['posted_rate']

# 4. Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train XGBoost Regressor
print("Training XGBoost Regressor...")
model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=6, 
    random_state=42, 
    n_jobs=-1,
    enable_categorical=True # <--- Tells XGBoost to handle category types natively
)
model.fit(X_train, y_train)

# Evaluate performance
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"\n--- XGBoost Model Performance ---")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# 6. Extract and Plot Feature Importances
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Most Important Features ---")
print(importance_df.head(5))

# Plotting (Fixed palette/hue warning)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), hue='Feature', palette='viridis', legend=False)
plt.title('Top 10 Feature Importances (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png')
print("\nPlot successfully saved as 'xgboost_feature_importance.png'!")
plt.show()