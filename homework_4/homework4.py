import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold

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

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)


# List of numerical features to evaluate
numerical_features = [
    'lead_score', 
    'number_of_courses_viewed', 
    'interaction_count', 
    'annual_income'
]

# Dictionary to store the final AUC scores
auc_scores = {}

for col in numerical_features:
    #  Extract the feature score
    score = df_train[col]
    
    #  Compute initial ROC AUC
    auc = roc_auc_score(df_train['converted'], score)
    
    #  If AUC < 0.5, invert the variable to flip the correlation direction
    if auc < 0.5:
        score = -df_train[col]
        auc = roc_auc_score(df_train['converted'], score)
    
    # Store the corrected AUC score
    auc_scores[col] = auc

# Find the variable with the highest AUC
highest_variable = max(auc_scores, key=auc_scores.get)

# Display results
print("ROC AUC Scores for each feature:")
for feature, score in auc_scores.items():
    print(f" - {feature}: {score:.4f}")

print(f"\nThe feature with the highest AUC is: {highest_variable}")





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



# Convert training and validation dataframes to dictionaries
train_dict = df_train[cat_cols + num_cols].to_dict(orient='records')
val_dict = df_val[cat_cols + num_cols].to_dict(orient='records')

# One-hot encode using DictVectorizer
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)

# Initialize and fit the model (using identical rows for X_train and y_train)
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Compute the ROC AUC score
val_auc = roc_auc_score(y_val, y_pred_proba)

# Print the result rounded to 3 decimal places
print(f"Validation ROC AUC Score: {val_auc:.3f}")


# Get the predicted probabilities for the validation set
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Define the thresholds from 0.0 to 1.0 with a step of 0.01
thresholds = np.linspace(0, 1, 101)

# Arrays to store the calculated metrics
precisions = []
recalls = []

best_threshold = None
min_diff = float('inf')

# Evaluate precision and recall for each threshold
for t in thresholds:
    # Make hard predictions (0 or 1) based on the threshold
    y_pred = (y_pred_proba >= t).astype(int)
    
    # Calculate metrics (zero_division=0 prevents warnings when precision is 0/0)
    p = precision_score(y_val, y_pred, zero_division=0)
    r = recall_score(y_val, y_pred, zero_division=0)
    
    precisions.append(p)
    recalls.append(r)
    
    # Ignore thresholds where both metrics are zero
    if p == 0 and r == 0:
        continue
        
    # Find the threshold with the smallest absolute difference
    diff = abs(p - r)
    if diff < min_diff:
        min_diff = diff
        best_threshold = t

print(f"Precision and Recall curves intersect around threshold: {best_threshold:.2f}")

# Plot the precision and recall curves
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, label='Precision', color='blue')
plt.plot(thresholds, recalls, label='Recall', color='orange')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Intersection (~{best_threshold:.2f})')

plt.title('Precision and Recall vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()





best_threshold = None
max_f1 = -1

# Evaluate F1-score for each threshold
for t in thresholds:
    # Make binary predictions based on the threshold
    y_pred = (y_pred_proba >= t).astype(int)
    
    # Calculate precision and recall
    p = precision_score(y_val, y_pred, zero_division=0)
    r = recall_score(y_val, y_pred, zero_division=0)
    
    # Calculate F1-score (handle case where both P and R are 0 to avoid division by zero)
    if (p + r) == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
        
    # Track the threshold that gives the highest F1 score
    if f1 > max_f1:
        max_f1 = f1
        best_threshold = t

# Display the results
print(f"Maximal F1-Score: {max_f1:.4f}")
print(f"F1-score is maximal at threshold: {best_threshold:.2f}")


#  Initialize KFold with 5 splits
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

# List to store AUC scores from each fold iteration
auc_scores = []

# Iterate over the 5 different folds of df_full_train
for train_idx, val_idx in kfold.split(df_full_train):
    
    # Split the original full train dataframe into fold train and fold validation sets
    df_train_fold = df_full_train.iloc[train_idx]
    df_val_fold = df_full_train.iloc[val_idx]
    
    # Isolate targets (y)
    y_train_fold = df_train_fold['converted'].values
    y_val_fold = df_val_fold['converted'].values
    
    # Convert feature subsets to dictionaries for one-hot encoding
    # (Excluding the target variable 'converted' automatically by using feature lists)
    features = cat_cols + num_cols
    train_dict = df_train_fold[features].to_dict(orient='records')
    val_dict = df_val_fold[features].to_dict(orient='records')
    
    # One-hot encode using DictVectorizer
    dv = DictVectorizer(sparse=False)
    X_train_fold = dv.fit_transform(train_dict)
    X_val_fold = dv.transform(val_dict)
    
    # Train the model with the exact specified parameters
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train_fold, y_train_fold)
    
    # Predict probabilities and calculate AUC on the validation fold
    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, y_pred_proba)
    
    auc_scores.append(auc)

#  Calculate the standard deviation of the scores
auc_std = np.std(auc_scores)

print(f"AUC scores across 5 folds: {[round(score, 4) for score in auc_scores]}")
print(f"Mean AUC score: {np.mean(auc_scores):.4f}")
print(f"Standard deviation of AUC scores: {auc_std:.3f}")


# 1. Define the C values to tune
c_values = [0.000001, 0.001, 1]

# Initialize KFold with identical parameters as before
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
features = cat_cols + num_cols

# Dictionary to store the results for each C
tuning_results = {}

# Iterate over the different C values
for C in c_values:
    auc_scores = []
    
    # 5-Fold cross-validation loop
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train_fold = df_full_train.iloc[train_idx]
        df_val_fold = df_full_train.iloc[val_idx]
        
        y_train_fold = df_train_fold['converted'].values
        y_val_fold = df_val_fold['converted'].values
        
        train_dict = df_train_fold[features].to_dict(orient='records')
        val_dict = df_val_fold[features].to_dict(orient='records')
        
        dv = DictVectorizer(sparse=False)
        X_train_fold = dv.fit_transform(train_dict)
        X_val_fold = dv.transform(val_dict)
        
        # Train the model using the current loop's C parameter
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_fold, y_train_fold)
        
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        auc_scores.append(auc)
        
    #  Compute mean and std, rounded to 3 decimal places
    mean_auc = round(np.mean(auc_scores), 3)
    std_auc = round(np.std(auc_scores), 3)
    
    # Store results
    tuning_results[C] = {'mean': mean_auc, 'std': std_auc}
    print(f"C = {C:<8} -> Mean AUC: {mean_auc:.3f} | Std: {std_auc:.3f}")

# Apply selection logic (Highest mean -> Lowest std -> Smallest C)
best_C = None
best_score = (-1, float('inf'), float('inf'))  # (max_mean, min_std, min_C)

for C, results in tuning_results.items():
    # We want to maximize mean, minimize std, and minimize C value itself
    # By negating std and C, we can maximize the whole tuple using standard sorting
    score_tuple = (results['mean'], -results['std'], -C)
    
    if score_tuple > best_score:
        best_score = score_tuple
        best_C = C

print(f"\nThe best parameter is C = {best_C}")
