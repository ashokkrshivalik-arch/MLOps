# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/insurance.csv")
print("Dataset loaded from hugging face successfully.")


# 2.1 Remove unnecessary columns
print("Step 2: Performing data cleaning...")

# Remove unnecessary columns -> customerID is Primary Key and there exists an unnamed column which appears to be Serial number. Both are not useful for modeling
cols_to_drop = ['CustomerID', 'Unnamed: 0']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 2.2 Standardize Categorical Values
if 'Gender' in df.columns:
  df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
  
if 'MaritalStatus' in df.columns:
# Merging 'Single' into 'Unmarried' to simplify the categories
  df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')

# 2.3 Label Encoding for categorical variables
# This converts columns like 'Occupation' from strings to numbers
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
  df[col] = le.fit_transform(df[col].astype(str))
  print(f" Encoded column: {col}")

print("Step 3: Splitting into train and test sets...")

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
# Stratified split to ensure equal proportion of buyers in both sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42,stratify=y)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")








    

 



