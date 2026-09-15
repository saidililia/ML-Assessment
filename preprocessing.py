import pandas as pd

df = pd.read_csv("data/train-test.csv")

print("Initial Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values before cleaning:")
print(df.isna().sum())  # Aka Null values

print("\nDuplicate rows:", df.duplicated().sum())

print("\nValues with unexpected Python types:")
for col in df.columns:
    expected_type = df[col].dropna().map(type).mode().iloc[0]
    wrong_type_count = (df[col].dropna().map(type) != expected_type).sum()
    print(f"{col}: {wrong_type_count}")


# Dropping rows where 'weight' or 'market_index' have null values
initial_row_count = len(df)
df = df.dropna(subset=["weight", "market_index"])
dropped_count = initial_row_count - len(df)

print("\n--- Cleaning Complete ---")
print(f"Dropped {dropped_count} rows with missing values.")
print("Final Cleaned Shape:", df.shape)
print("Missing values after cleaning:\n", df.isna().sum())


# Save the cleaned dataset for your modeling workflow
df.to_csv("data/train-test-cleaned.csv", index=False)