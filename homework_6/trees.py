import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb




#1 'model_year'
#2 val RMSE: 1.8336350509302553
#3 The value of n_estimators with the lowest validation RMSE is: 100 (RMSE: 1.768)
#Best Configuration:
# 4 -> max_depth: 10
#-> n_estimators: 100
#-> Lowest RMSE: 1.746
# 5 The most important feature among the four is: **vehicle_weight**
# 6 The eta that leads to the best (lowest) validation RMSE score is: **0.1**

df = pd.read_csv('car_fuel_efficiency_2026.csv')
df.head()
print(df.head())
print(df.dtypes)
df.fillna(0, inplace=True)
print(df.isnull().sum())


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values


del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

# Initialize the DictVectorizer with sparse=False to get a dense array output
dv = DictVectorizer(sparse=False)

# Fit on training data and transform both sets into sparse matrices
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
rmse_train = root_mean_squared_error(y_train, y_pred_train)
print('train RMSE:', rmse_train)

# Evaluate on validation data
y_pred_val = dt.predict(X_val)
rmse_val = root_mean_squared_error(y_val, y_pred_val)
print('val RMSE:', rmse_val)

# This text output line will now run perfectly without throwing errors
print(export_text(dt, feature_names=list(dv.get_feature_names_out())))


# 1. Get the raw text tree layout
tree_rules = export_text(dt, feature_names=list(dv.get_feature_names_out()))
print("--- Tree Structure ---")
print(tree_rules)

# 2. Extract the engineered feature name from the first line of export_text
# This finds the feature inside a string like "|--- origin=USA <= 0.50"
first_line = tree_rules.split("\n")[0]
split_feature = None

for feature in dv.get_feature_names_out():
    if feature in first_line:
        split_feature = feature
        break

# 3. Map it back to the original column name
if split_feature:
    original_feature = (
        split_feature.split("=")[0] if "=" in split_feature else split_feature
    )
    print("--------------------------------------------------")
    print(f"Engineered feature used for split: {split_feature}")
    print(f"Original DataFrame column name   : {original_feature}")
    print("--------------------------------------------------")
else:
    print("Could not find the splitting feature in the tree text layout.")






df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values


del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

# Initialize the DictVectorizer with sparse=False to get a dense array output
dv = DictVectorizer(sparse=False)

# Fit on training data and transform both sets into sparse matrices
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

dt = RandomForestRegressor(n_jobs=-1,n_estimators=10, random_state=1)
dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
rmse_train = root_mean_squared_error(y_train, y_pred_train)
print('train RMSE:', rmse_train)

# Evaluate on validation data
y_pred_val = dt.predict(X_val)
rmse_val = root_mean_squared_error(y_val, y_pred_val)
print('val RMSE:', rmse_val)



# Define the parameter values to test
estimators_list = [10, 50, 100, 150]
results = {}

for n in estimators_list:
    # Initialize and fit the random forest regressor
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predict and calculate validation RMSE
    y_pred_val = rf.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred_val)
    
    # Store the result rounded to 3 decimal places
    results[n] = round(rmse, 3)
    print(f"n_estimators = {n:3d} | Validation RMSE: {rmse:.3f}")

# Find the best n_estimators
best_n = min(results, key=results.get)
print(f"\nThe value of n_estimators with the lowest validation RMSE is: {best_n} (RMSE: {results[best_n]})")


# Define hyperparameter ranges
depths = [10, 15, 20, 25]
#estimators_list = [10, 50, 100, 150]

# List to collect structural results
all_results = []

print("depth | n_est | Validation RMSE")
print("-" * 32)

for depth in depths:
    for n in estimators_list:
        # Initialize and train the Random Forest
        rf = RandomForestRegressor(
            max_depth=depth,
            n_estimators=n,
            random_state=1,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Predict on validation data and compute RMSE
        y_pred = rf.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        
        # Store results
        all_results.append({
            'max_depth': depth,
            'n_estimators': n,
            'rmse': round(rmse, 3)
        })
        print(f"{depth:5d} | {n:5d} | {rmse:.3f}")

# Find the configuration with the lowest validation RMSE
best_config = min(all_results, key=lambda x: x['rmse'])

print("-" * 32)
print(f"Best Configuration:")
print(f"-> max_depth: {best_config['max_depth']}")
print(f"-> n_estimators: {best_config['n_estimators']}")
print(f"-> Lowest RMSE: {best_config['rmse']:.3f}")


# Train the Random Forest with the specified parameters
rf_importance = RandomForestRegressor(
    n_estimators=10, max_depth=20, random_state=1, n_jobs=-1
)
rf_importance.fit(X_train, y_train)

# Extract feature importances
importances = rf_importance.feature_importances_

# Pair importances with feature names from the DictVectorizer
feature_names = list(dv.get_feature_names_out())
importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
)

# Sort features by importance score in descending order
importance_df = importance_df.sort_values(by="importance", ascending=False)

print("--- Feature Importances ---")
print(importance_df.head(10).to_string(index=False))

#  Identify the absolute top feature
most_important = importance_df.iloc[0]["feature"]
print("\n--------------------------------------------------")
print(f"The most important feature is: {most_important}")
print("--------------------------------------------------")


# List of the 4 target features to compare
target_features = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']

# Filter the dataframe to only include these 4 features
subset_importance = importance_df[importance_df['feature'].isin(target_features)]

# Sort them to find the top one
subset_importance = subset_importance.sort_values(by='importance', ascending=False)

print("--- Comparison Subset ---")
print(subset_importance.to_string(index=False))

best_subset_feature = subset_importance.iloc[0]['feature']
print(f"\nThe most important feature among the four is: **{best_subset_feature}**")


#  Create DMatrix for train and validation datasets
# Note: XGBoost natively supports feature names when columns are strings
feature_names = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

# Create a watchlist to monitor performance on both sets
watchlist = [(dtrain, 'train'), (dval, 'val')]

# Base configurations
base_params = {
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 0 # Turned down to keep print statements clean
}

#  Train and evaluate with eta = 0.3
print("--- Training with eta = 0.3 ---")
params_03 = base_params.copy()
params_03['eta'] = 0.3

model_03 = xgb.train(
    params_03,
    dtrain,
    num_boost_round=100,
    evals=watchlist,
    verbose_eval=10
)
y_pred_03 = model_03.predict(dval)
rmse_03 = root_mean_squared_error(y_val, y_pred_03)

#  Train and evaluate with eta = 0.1
print("\n--- Training with eta = 0.1 ---")
params_01 = base_params.copy()
params_01['eta'] = 0.1

model_01 = xgb.train(
    params_01,
    dtrain,
    num_boost_round=100,
    evals=watchlist,
    verbose_eval=10
)
y_pred_01 = model_01.predict(dval)
rmse_01 = root_mean_squared_error(y_val, y_pred_01)

# Final Summary Comparison
print("\n=== Final Validation Results ===")
print(f"eta = 0.3 -> Validation RMSE: {rmse_03:.4f}")
print(f"eta = 0.1 -> Validation RMSE: {rmse_01:.4f}")

best_eta = "0.3" if rmse_03 < rmse_01 else "0.1"
print(f"\nThe eta that leads to the best (lowest) validation RMSE score is: **{best_eta}**")