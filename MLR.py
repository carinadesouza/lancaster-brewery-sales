# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
file_path = "LB SALES.xlsx"
df = pd.read_excel(file_path)

# Checking for nulls
print(df.info())

# Select the key independent variables
selected_features = [
    'Order Type', 'Barrels', 'GL Desc', 'Qty',
    'Product Cost Price', 'Total Cost Price',
    'Discount', 'Duty Element'
]

# Define the target (sales = Price Paid)
target = 'Price Paid'

# Create a working dataframe
df_sales = df[selected_features + [target]].copy()

# Drop missing values
df_sales.dropna(inplace=True)

print("Shape after cleaning:", df_sales.shape)
df_sales.head()

# Encode non-numeric fields (e.g., 'Order Type' or 'GL Desc')
encoder = LabelEncoder()
for col in ['Order Type', 'GL Desc']:
    if df_sales[col].dtype == 'object':
        df_sales[col] = encoder.fit_transform(df_sales[col].astype(str))

# Separate predictors (X) and target (y)
X = df_sales[selected_features]
y = df_sales[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardise only numeric columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
mlr_sales = LinearRegression()
mlr_sales.fit(X_train_scaled, y_train)

# Make predictions
y_pred = mlr_sales.predict(X_test_scaled)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
n, p = X_test.shape
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': mlr_sales.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance (Effect on Sales):")
print(coeff_df)

import matplotlib.pyplot as plt
import seaborn as sns

# Sort coefficients
coeff_df_sorted = coeff_df.sort_values(by='Coefficient', ascending=True)

plt.figure(figsize=(8,5))
sns.barplot(x='Coefficient', y='Feature', data=coeff_df_sorted, palette='coolwarm')
plt.title('Feature Importance: Effect of Predictors on Sales (Price Paid)', fontsize=12)
plt.xlabel('Regression Coefficient (Impact on Sales)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='teal', edgecolor='white')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Price Paid (Sales)')
plt.ylabel('Predicted Price Paid (Sales)')
plt.title('Actual vs Predicted Sales (MLR Model)', fontsize=12)
plt.tight_layout()
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,4))
sns.histplot(residuals, bins=30, kde=True, color='orange')
plt.title('Distribution of Residuals (Model Errors)', fontsize=12)
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

metrics = pd.DataFrame({
    'Metric': ['R²', 'Adjusted R²', 'MSE', 'RMSE'],
    'Value': [r2, adj_r2, mse, rmse]
})
print(metrics)