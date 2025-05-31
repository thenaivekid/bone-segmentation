import pandas as pd
import warnings

from pandas._libs.hashtable import value_count

warnings.filterwarnings("ignore")

df = pd.read_csv("TASK_2/train_set.csv")
# print(f"{df.head(10)=}")

print(f"{df.shape=}")
# CLASS train
# 0    191
# 1    124 

# CLASS test
# 0    58
# 1    42

print(f"{df['CLASS'].value_counts()}=")
# print(f"{df.columns=}")

# print(f"{df.info()=}")

# print(f"{df.describe()=}")

# print(f" null values{df.isnull().sum()}")

null_counts = df.isnull().sum()
missing_cols = null_counts[null_counts > 0]

print("Columns with missing values and their counts:")
print(missing_cols.sort_values(ascending=False))
print("------------------------------------------")
# for col in df.columns:
#     print(f"{col}: {df[col].nunique()}, {df[col].value_counts()}")

### filter out the culumns with too many unique values
threshold = 150

high_cardinality_cols = []
constant_cols = []
useful_cols = []

for col in df.columns:
    unique_vals = df[col].nunique(dropna=False)
    if unique_vals == 1:
        constant_cols.append(col)
    elif unique_vals > threshold:
        high_cardinality_cols.append(col)
    else:
        useful_cols.append(col)

print(f"High-cardinality features (> {threshold} unique values): {len(high_cardinality_cols)}")
print(f"Constant features (only one unique value): {len(constant_cols)}")
print(f"Useful features (≤ {threshold} unique values): {len(useful_cols)}")

print("------------------------------------------")
