import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#checking for tail
df = pd.read_csv("/workspaces/nine-learning-zoomcamp/homework2/car_fuel_efficiency_2026.csv")
plt.figure(figsize=(8,4))
sns.histplot(df['fuel_efficiency_mpg'], kde= True, bins= 30)
plt.title("Distribution of Fuel Fuel Efficiency(mpg)")
plt.xlabel("MPG")
plt.ylabel("Count")
plt.savefig("/workspaces/nine-learning-zoomcamp/homework2/fuel_efficiency_plot.png")
print("Plot saved successfully!")

cols = ['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']
df = df[cols]

print(df.isna().sum()) #1 horsepower

print(df["horsepower"].median()) #2 254.0

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

y_train = df_train.fuel_efficiency_mpg.values
y_val   = df_val.fuel_efficiency_mpg.values

features = ['engine_displacement','horsepower','vehicle_weight','model_year']

# Step 4: Linear regression without regularization (from the lessons)
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

# Step 5: RMSE (from the lessons)
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    return np.sqrt(se.mean())

# Step 6: Helper that prepares features with a given fill value
def prepare_X(df, fill_value):
    return df[features].fillna(fill_value).values

# Step 7: Option 1 - fill missing values with 0
X_train_0 = prepare_X(df_train, 0)
X_val_0   = prepare_X(df_val, 0)

w0, w = train_linear_regression(X_train_0, y_train)
y_pred_0 = w0 + X_val_0.dot(w)
score_0 = round(rmse(y_val, y_pred_0), 3)

# Step 8: Option 2 - fill with the mean (computed on TRAIN only)
horsepower_mean = df_train['horsepower'].mean()

X_train_mean = prepare_X(df_train, horsepower_mean)
X_val_mean   = prepare_X(df_val, horsepower_mean)

w0, w = train_linear_regression(X_train_mean, y_train)
y_pred_mean = w0 + X_val_mean.dot(w)
score_mean = round(rmse(y_val, y_pred_mean), 3)

# Step 9: Compare
print("RMSE with 0:   ", score_0)      # 2.205
print("RMSE with mean:", score_mean)   # 2.202
print("Better option:", "With mean" if score_mean < score_0 else "With 0")


# Step 1: Regularized linear regression (ridge) from the lessons
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])   # add r to the diagonal

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

# Step 2: Prepare data, filling NAs with 0
X_train = df_train[features].fillna(0).values
X_val   = df_val[features].fillna(0).values

# Step 3: Try each r and evaluate on the validation set
results = {}
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w0 + X_val.dot(w)
    results[r] = round(rmse(y_val, y_pred), 4)
    print(r, results[r])

# Step 4: Pick the best r (smallest r wins ties)
best_score = min(results.values())
best_r = min(r for r, s in results.items() if s == best_score)
print("Best r:", best_r)

# Step 1: Loop over seeds, split, train without regularization, collect RMSE
scores = []

for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    # 60/20/20 split with this seed
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_shuffled = df.iloc[idx]
    df_train = df_shuffled.iloc[:n_train]
    df_val   = df_shuffled.iloc[n_train:n_train + n_val]

    y_train = df_train.fuel_efficiency_mpg.values
    y_val   = df_val.fuel_efficiency_mpg.values

    # Fill NAs with 0
    X_train = df_train[features].fillna(0).values
    X_val   = df_val[features].fillna(0).values

    # Train without regularization (the function from the first question)
    w0, w = train_linear_regression(X_train, y_train)

    # Evaluate on validation
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    scores.append(score)
    print(seed, score)

# Step 2: Standard deviation of the scores
std = np.std(scores)
print(round(std, 3))

# Step 1: Split with seed 9 (60/20/20)
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]
df_train = df_shuffled.iloc[:n_train]
df_val   = df_shuffled.iloc[n_train:n_train + n_val]
df_test  = df_shuffled.iloc[n_train + n_val:]

# Step 2: Combine train and validation
df_full_train = pd.concat([df_train, df_val])
y_full_train = df_full_train.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values

# Step 3: Fill NAs with 0
X_full_train = df_full_train[features].fillna(0).values
X_test = df_test[features].fillna(0).values

# Step 4: Train with r=0.001 (the regularized function from the second question)
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

# Step 5: Evaluate on the test set
y_pred = w0 + X_test.dot(w)
test_score = rmse(y_test, y_pred)
print(round(test_score, 3))