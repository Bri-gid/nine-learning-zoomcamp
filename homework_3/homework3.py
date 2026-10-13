import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('course_lead_scoring_2026.csv')
df.head()
print(df.head())
print(df.dtypes)

print(df.isnull().sum())
# Select all string/object columns and fill NaNs with 'NA'
cat_cols = df.select_dtypes(include=['object', 'str']).columns
df[cat_cols] = df[cat_cols].fillna('NA')


num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(0.0)


print(df.isnull().sum())

print(df["industry"].mode()) # technology

correlation_matrix = df[num_cols].corr()

print(correlation_matrix)


# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
plt.savefig('correlation_matrix.png') #interaction_count and lead_score


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.converted.values
y_val = df_val.converted.values
y_test = df_test.converted.values

num_cols = [col for col in num_cols if col != 'converted']
cat_cols = [col for col in cat_cols if col != 'converted']

del df_train['converted']
del df_val['converted']
del df_test['converted']

def mutual_info_converted(series):
    return mutual_info_score(series, df_full_train.converted)

mi = df_full_train[cat_cols].apply(mutual_info_converted)
mi_sorted_rounded = mi.sort_values(ascending=False).round(2)

print(mi_sorted_rounded) # lead_source


# 1. Convert training and validation dataframes to dictionaries
train_dict = df_train[cat_cols + num_cols].to_dict(orient='records')
val_dict = df_val[cat_cols + num_cols].to_dict(orient='records')

# 2. One-hot encode using DictVectorizer
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)

# 3. Initialize and fit the model (using identical rows for X_train and y_train)
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 4. Predict on validation dataset and calculate accuracy
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

# 5. Print and round the result
print(f"Validation Accuracy: {round(accuracy, 2)}") # 0.65




