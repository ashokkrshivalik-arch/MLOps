
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "tourism.csv")

# 2. Load dataset
df = pd.read_csv(RAW_PATH)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)

# 3. Remove unnecessary columns
print("Step 2: Performing data cleaning...")

cols_to_drop = ["CustomerID", "Unnamed: 0"]

df = df.drop(
    columns=[col for col in cols_to_drop if col in df.columns]
)

# 4. Standardize categorical values
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace(
        {"Fe Male": "Female"}
    )

if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace(
        {"Single": "Unmarried"}
    )

# 5. Define target
target_col = "ProdTaken"

if target_col not in df.columns:
    raise ValueError(f"Target column {target_col} not found!")

# 6. Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Keep categorical features as strings.
# OneHotEncoder in train.py will handle encoding.

# 7. Train-test split
print("Step 3: Splitting into train and test sets...")

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Save processed datasets
Xtrain.to_csv(
    os.path.join(DATA_DIR, "Xtrain.csv"),
    index=False
)

Xtest.to_csv(
    os.path.join(DATA_DIR, "Xtest.csv"),
    index=False
)

ytrain.to_csv(
    os.path.join(DATA_DIR, "ytrain.csv"),
    index=False
)

ytest.to_csv(
    os.path.join(DATA_DIR, "ytest.csv"),
    index=False
)

print("\nData preparation completed successfully!")
print("Training features:", Xtrain.shape)
print("Testing features:", Xtest.shape)
print("Training target:", ytrain.shape)
print("Testing target:", ytest.shape)

print("\nProcessed files saved in:", DATA_DIR)
