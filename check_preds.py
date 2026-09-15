import pandas as pd
import numpy as np

# Load your predictions file
preds = pd.read_csv("validation_predictions.csv")

print("Shape:", preds.shape)
print("Missing values (NaN):\n", preds.isna().sum())
print("Infinite values:\n", np.isinf(preds['predicted_rate']).sum())
print("Negative values count:", (preds['predicted_rate'] < 0).sum())

# Check data type
print("Data type of predicted_rate:", preds['predicted_rate'].dtype)
print("\nFirst 10 values:\n", preds['predicted_rate'].head(10))