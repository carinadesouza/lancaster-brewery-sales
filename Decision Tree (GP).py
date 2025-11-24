# ======================================================
# Decision Tree Regression for Gross Profit (Improved)
# ======================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("LB SALES.xlsx")

# Fix inconsistent names
df.rename(columns={"Qty": "Quantity"}, inplace=True)

# ======================================================
# 1️⃣ Feature Engineering — KEEP all categories
# ======================================================

df["Outlet Category"] = df["Outlet Name"].str.title()
df["Product Category"] = df["Product Group"].str.title()
df["Order Type"] = df["Order Type"].str.title()
df["Sales Category"] = df["Sales Category"].str.title()

# ======================================================
# 2️⃣ Select Features for GP
# ======================================================

features = [
    "Quantity",
    "Barrels",
    "Product Cost Price",
    "Total Cost Price",
    "Discount",
    "Duty Element",
    "Outlet Category",
    "Product Category",
    "Sales Category",
    "Order Type"
]

target = "GP"

df_model = df[features + [target]].dropna().copy()

# One-hot encode ALL categories
df_model = pd.get_dummies(df_model, columns=[
    "Outlet Category", "Product Category", "Sales Category", "Order Type"
], drop_first=True)

X = df_model.drop(columns=[target])
y = df_model[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ======================================================
# 3️⃣ Base Model
# ======================================================

base = DecisionTreeRegressor(
    random_state=42,
    max_depth=6,
    min_samples_leaf=50
)
base.fit(X_train, y_train)

y_pred = base.predict(X_test)

print("\nBASELINE GP MODEL")
print("R²:", round(r2_score(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

# ======================================================
# 4️⃣ Hyperparameter Tuning
# ======================================================

param_grid = {
    "max_depth": [6, 8, 10, 12, None],
    "min_samples_leaf": [10, 25, 50],
    "min_samples_split": [2, 10, 50],
    "max_features": [None, "sqrt", "log2"]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_tree = grid.best_estimator_

y_tuned = best_tree.predict(X_test)

print("\nTUNED GP MODEL")
print("Best Params:", grid.best_params_)
print("R²:", round(r2_score(y_test, y_tuned), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_tuned)), 2))
print("MAE:", round(mean_absolute_error(y_test, y_tuned), 2))

# ======================================================
# 5️⃣ Feature Importance
# ======================================================

importances = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(10, 12))
sns.barplot(x=importances, y=importances.index, palette="viridis")
plt.title("Feature Importance — Gross Profit Model")
plt.tight_layout()
plt.show()

# ======================================================
# 6️⃣ Simple Tree Visualization
# ======================================================

simple_tree = DecisionTreeRegressor(
    random_state=42, max_depth=3, min_samples_leaf=50
)
simple_tree.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(simple_tree, feature_names=X.columns, filled=True, rounded=True, fontsize=6)
plt.show()

# ======================================================
# 7️⃣ Cross Validation
# ======================================================

cv_scores = cross_val_score(best_tree, X_train, y_train, cv=5, scoring="r2")
print("\nCross-val R² scores:", cv_scores.round(3))
print("Mean CV R²:", round(cv_scores.mean(), 3))
