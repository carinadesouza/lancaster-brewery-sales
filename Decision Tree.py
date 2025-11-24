# ======================================================
# Decision Tree Regression for Lancaster Brewery Sales
# Group 21 - SCC450 Coursework
# ======================================================

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the dataset ---
file_path = "LB SALES.xlsx"     # Adjust path if needed
df = pd.read_excel(file_path)

# --- Select relevant columns (same as MLR model) ---
features = [
    'Order Type', 'Barrels', 'GL Desc', 'Qty',
    'Product Cost Price', 'Total Cost Price',
    'Discount', 'Duty Element'
]
target = 'Price Paid'

# --- Create working dataframe ---
df_tree = df[features + [target]].dropna().copy()

# --- Encode categorical columns ---
for col in ['Order Type', 'GL Desc']:
    if df_tree[col].dtype == 'object':
        encoder = LabelEncoder()
        df_tree[col] = encoder.fit_transform(df_tree[col].astype(str))

# --- Split data ---
X = df_tree[features]
y = df_tree[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ======================================================
# 1️⃣ Baseline Decision Tree
# ======================================================
baseline_tree = DecisionTreeRegressor(
    random_state=42,
    max_depth=5,
    min_samples_leaf=50
)
baseline_tree.fit(X_train, y_train)

# --- Evaluate baseline ---
y_pred = baseline_tree.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("=== Baseline Decision Tree ===")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# ======================================================
# 2️⃣ Hyperparameter Tuning with GridSearchCV
# ======================================================
param_grid = {
    'max_depth': [4, 6, 8, 10, None],
    'min_samples_leaf': [10, 25, 50, 100],
    'min_samples_split': [2, 10, 50, 100],
    'max_features': [None, 'sqrt', 'log2']
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
tree = DecisionTreeRegressor(random_state=42)

grid = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    scoring='r2',
    cv=cv,
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_tree = grid.best_estimator_
print("\n=== Tuned Decision Tree ===")
print("Best Parameters:", grid.best_params_)

# --- Evaluate tuned model ---
y_pred_tuned = best_tree.predict(X_test)
r2_tuned = r2_score(y_test, y_pred_tuned)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

print(f"R²: {r2_tuned:.4f}")
print(f"RMSE: {rmse_tuned:.2f}")
print(f"MAE: {mae_tuned:.2f}")

# ======================================================
# 3️⃣ Feature Importance
# ======================================================
importances = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7,5))
sns.barplot(x=importances, y=importances.index, palette="crest")
plt.title('Decision Tree Feature Importance (Sales Prediction)')
plt.xlabel('Importance (0–1)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ======================================================
# 4️⃣ (Optional) Small Tree Visualisation
# ======================================================
small_tree = DecisionTreeRegressor(
    random_state=42, max_depth=3, min_samples_leaf=100
)
small_tree.fit(X_train, y_train)

plt.figure(figsize=(16,8))
plot_tree(
    small_tree,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title('Simplified Decision Tree (depth=3)')
plt.show()

# ======================================================
# 5️⃣ Cross-validation Robustness Check
# ======================================================
cv_r2 = cross_val_score(best_tree, X_train, y_train, cv=5, scoring='r2')
print("\nCross-Validation R² Scores:", cv_r2.round(3))
print("Mean CV R²:", cv_r2.mean().round(3))
