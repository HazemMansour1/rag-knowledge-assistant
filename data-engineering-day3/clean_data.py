import pandas as pd 

df = pd.read_csv("documents.csv")

# Check missing values
print("Missing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check data types
print("\nData types:")
print(df.dtypes)

df = df.drop_duplicates()

df["source"] = df["source"].fillna("Unknown")
df["file_size_mb"] = pd.to_numeric(df["file_size_mb"], errors="coerce")
df["file_size_mb"] = df["file_size_mb"].fillna(0)
df["status"] = df["status"].str.lower().str.strip()
print(df["status"].unique())
print(df["status"].value_counts())
print(df.head(10))
print(df["file_size_mb"].dtype)

# Ensure no duplicate rows remain
assert df.duplicated().sum() == 0

# Ensure file_size_mb contains no missing values
assert df["file_size_mb"].isnull().sum() == 0

# Ensure all statuses are valid
assert df["status"].isin(["ready", "pending"]).all()

print("All validation checks passed!")

df.to_csv("cleaned_document.csv",index=False)
