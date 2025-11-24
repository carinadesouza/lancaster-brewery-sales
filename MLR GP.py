# ==========================
# MULTIPLE LINEAR REGRESSION (with Clean Labels)
# ==========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --------------------------
# Load dataset
# --------------------------
df = pd.read_excel("LB SALES.xlsx")

# Clean dataset (drop missing)
df = df[['Price Paid', 'Qty', 'Product Cost Price', 'Total Cost Price', 'Duty Element',
         'Discount', 'Barrels', 'Outlet Name', 'Product Group', 'Order Type', 'GP']].dropna()

# --------------------------
# Identify top 4 outlets & top 4 product groups
# --------------------------
top_outlets = df.groupby("Outlet Name")["GP"].sum().nlargest(4).index
top_products = df.groupby("Product Group")["GP"].sum().nlargest(4).index

df["Outlet Name"] = df["Outlet Name"].apply(lambda x: x if x in top_outlets else "Other Outlets")
df["Product Group"] = df["Product Group"].apply(lambda x: x if x in top_products else "Other Products")

# --------------------------
# Numeric / Categorical split
# --------------------------
numeric_features = ['Qty', 'Product Cost Price', 'Total Cost Price', 'Duty Element', 'Discount', 'Barrels']
categorical_features = ['Outlet Name', 'Product Group', 'Order Type']

# --------------------------
# Build preprocessing
# --------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# --------------------------
# Build full pipeline
# --------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# --------------------------
# Train/Test Split
# --------------------------
X = df[numeric_features + categorical_features]
y = df['GP']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Fit model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# =============================
# Extract cleaned feature names
# =============================

# 1. Get numeric feature names (already clean)
num_labels = numeric_features.copy()

# 2. Get dummy variable names and clean them
ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
cat_labels_raw = ohe.get_feature_names_out(categorical_features)

clean_labels = []
for label in cat_labels_raw:
    # Remove prefixes (Outlet Name_, Product Group_, Order Type_, GL Desc_)
    clean = label
    clean = clean.replace("Outlet Name_", "")
    clean = clean.replace("Product Group_", "")
    clean = clean.replace("Order Type_", "")

    # Convert to Title Case
    clean = clean.title()

    clean_labels.append(clean)

# Combine into full feature list
all_features = num_labels + clean_labels

# =============================
# Compute metrics
# =============================
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n====================")
print("MODEL PERFORMANCE")
print("====================")
print(f"R² Score:        {r2:.4f}")
print(f"Adjusted R²:     {1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(all_features) - 1):.4f}")
print(f"MSE:             {mse:.2f}")
print(f"RMSE:            {rmse:.2f}")
print("====================\n")

# ==========================
# Coefficient analysis
# ==========================
coefs = model.named_steps["regressor"].coef_
coef_df = pd.DataFrame({"Feature": all_features, "Coefficient": coefs})

# Sort
coef_df_sorted = coef_df.sort_values(by="Coefficient", ascending=False)

print("\nTop Positive Drivers of GP:\n")
print(coef_df_sorted.head(10).to_string(index=False))

print("\nTop Negative Drivers of GP:\n")
print(coef_df_sorted.tail(10).to_string(index=False))

# ==========================
# Residual Analysis
# ==========================
residuals = y_test - y_pred

# --- Residuals vs Predicted Plot ---
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
plt.axhline(0, linestyle='--', color='red')
plt.xlabel("Predicted Gross Profit")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.show()

# --- QQ Plot ---
plt.figure(figsize=(7, 5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.tight_layout()
plt.show()

# --- Actual vs Predicted ---
plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle="--", color="red")
plt.xlabel("Actual Gross Profit")
plt.ylabel("Predicted Gross Profit")
plt.title("Actual vs Predicted GP")
plt.tight_layout()
plt.show()