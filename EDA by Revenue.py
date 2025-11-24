import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# Exploratory Data Analysis (EDA) - Lancaster Brewery Sales
# ======================================================

# ======================================================
# 1️⃣ LOAD DATASET
# ======================================================

df = pd.read_excel("LB SALES.xlsx")

print("Dataset loaded successfully!")
print("Shape of data:", df.shape)
print(df.head())

# ======================================================
# 2️⃣ RANDOM SAMPLING FOR VISUAL CLARITY
# ======================================================

sample_df = df.sample(n=2000, random_state=42)

# ======================================================
# 3️⃣ SUMMARY STATISTICS
# ======================================================

key_cols = ['Price Paid', 'Total Cost Price', 'Product Cost Price',
            'Discount', 'Duty Element', 'Qty']

print("\nSummary statistics for key financial variables:")
print(df[key_cols].describe())

# Boxplot of Price Paid (sampled)
plt.figure(figsize=(12, 5))
sns.boxplot(
    x=df["Price Paid"].clip(df["Price Paid"].quantile(0.01),
                            df["Price Paid"].quantile(0.99)),
    color="skyblue"
)

price_min = df["Price Paid"].quantile(0.01)
price_max = df["Price Paid"].quantile(0.99)

tick_range = np.arange(start=0, stop=price_max+30, step=30)
plt.xticks(tick_range, rotation=45)

plt.title("Boxplot of Price Paid (1st–99th Percentile)", fontsize=16)
plt.xlabel("Price Paid (£)", fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# Boxplot for Total Cost Price
plt.figure(figsize=(12, 5))
sns.boxplot(
    x=df["Total Cost Price"].clip(df["Total Cost Price"].quantile(0.01),
                            df["Total Cost Price"].quantile(0.99)),
    color="skyblue"
)
price_min = df["Total Cost Price"].quantile(0.01)
price_max = df["Total Cost Price"].quantile(0.99)

tick_range = np.arange(start=0, stop=price_max+30, step=30)
plt.xticks(tick_range, rotation=45)
plt.title("Boxplot of Total Cost Price (1st–99th Percentile)", fontsize=16)
plt.xlabel("Total Cost Price (£)", fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ======================================================
# 6️⃣ HISTOGRAM OF PRICE PAID (TRIMMED TO 99TH PERCENTILE)
# ======================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

price = df["Price Paid"].clip(
    df["Price Paid"].quantile(0.01),
    df["Price Paid"].quantile(0.99)
)

plt.figure(figsize=(12, 5))

sns.histplot(price, bins=40, kde=True, color="teal", edgecolor="black", alpha=0.35)

xmin, xmax = int(price.min()), int(price.max())

# Tick every £20 for readability
tick_range = np.arange(start=0, stop=xmax + 30, step=30)
plt.xticks(tick_range)

plt.title("Distribution of Sales (Price Paid) – Clipped at 99th Percentile", fontsize=16)
plt.xlabel("Price Paid (£)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# ======================================================
# 8️⃣ MONTHLY SALES TREND (HISTORICAL ANALYSIS)
# ======================================================

if 'SRDate' in df.columns:
    df['SRDate'] = pd.to_datetime(df['SRDate'], errors='coerce')
    df_time = df.groupby(df['SRDate'].dt.to_period('M'))['Price Paid'].sum().reset_index()
    df_time['SRDate'] = df_time['SRDate'].astype(str)

    plt.figure(figsize=(10, 4))
    sns.lineplot(x='SRDate', y='Price Paid', data=df_time, color='coral', lw=2)
    plt.title('Monthly Sales Trend Over Time')
    plt.xlabel('Month')
    plt.ylabel('Total Sales (£)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'SRDate' not found. Skipping monthly trend analysis.")




sns.set(style="whitegrid")

# Prepare data
top_outlets = df.groupby("Outlet Name")["Price Paid"].sum().sort_values(ascending=False).head(20)
highlight_n = 5  # highlight top 5

# Build color list: first N = strong color, rest = faded
colors = ["#1f77b4"] * highlight_n + ["#1f77b480"] * (len(top_outlets) - highlight_n)
# "80" at the end makes it 50% transparent (faded)

plt.figure(figsize=(14,6))
plt.bar(top_outlets.index, top_outlets.values, color=colors)
plt.title("Top 20 Outlets by Total Sales (Highlighted)")
plt.ylabel("Total Sales (£)")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()



top_products = df.groupby("Product Group")["Price Paid"].sum().sort_values(ascending=False).head(15)
highlight_n = 4  # highlight top 4 products with the highest number of sales

colors = ["#ff7f0e"] * highlight_n + ["#ff7f0e80"] * (len(top_products) - highlight_n)
# "80" makes it semi-transparent

plt.figure(figsize=(14,6))
plt.bar(top_products.index, top_products.values, color=colors)
plt.title("Top 15 Product Groups by Total Sales (Highlighted)")
plt.ylabel("Total Sales (£)")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

top_products = df.groupby("Product Name")["Price Paid"].sum().sort_values(ascending=False).head(15)
highlight_n = 4  # highlight top 4 products with the highest number of sales

colors = ["#ff7f0e"] * highlight_n + ["#ff7f0e80"] * (len(top_products) - highlight_n)
# "80" makes it semi-transparent

plt.figure(figsize=(14,6))
plt.bar(top_products.index, top_products.values, color=colors)
plt.title("Top 15 Product Names by Total Sales (Highlighted)")
plt.ylabel("Total Sales (£)")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()