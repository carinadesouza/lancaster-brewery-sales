'''
SCC450 Group Project 21 - K-means Clustering for LB Gross Profit GP

Firstly start with cleaning the data (if necessary) and finding the
optimum k value using WCSS and the Elbow Method.

Notes in week 2 lecture and week 3 practical.

Note: This clustering does not use a sample of the data so takes a 
few seconds to load. 
'''

# All necessary imports here
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

#import LB sales excel as a pandas dataframe
df = pd.read_excel('LB SALES.xlsx')
# extract the data for k-means clustering, using GP 
LB_clusterdf = df[['Outlet Name', 'Product Group', 'GP']]

# check if the data needs cleaning
#print(LB_clusterdf.info())
# we do not have non-null types in the data which need to be removed





# ====================================================================
#       Elbow Method to find the optimum k for GP clustering.
# ====================================================================

# From Week 3 Practical Notes
wcss = [] # make a within-cluster sum of squares list
# define our input feature GP 
X = LB_clusterdf[['GP']]

# Iterate through k= 1, ..., 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=1)
    kmeans.fit(X)
    # Add WCSS value to the list
    wcss.append(kmeans.inertia_)

# Plot results and use the elbow method to find k
plt.plot(range(1, 11), wcss)
plt.title('The elbow method for Gross Profit (GP)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# The plot shows that the elbow sharply changes at k=3
# Hence choose k=3 for our number of clusters



# ==================================================================
#     Build a K-means model for GP using k=3
# ==================================================================

# From Week 3 Practical LectureNotes
# Initialise the training model and use k=3 from Elbow Method
LB_kmeans = KMeans(n_clusters=3, random_state=1)
# Let model learn from the data
LB_kmeans.fit(X)

# use the trained model to find clusters
labels = LB_kmeans.predict(X)

# Find Cluster Centroids
centroids = LB_kmeans.cluster_centers_

# Plot a figure to show the clusters
fig = plt.figure(figsize=(8, 6))
# select the cluster colours
colors = ['r', 'b', 'g']
# Plot data points (from lecture notes)
for i in range(len(X)):
    plt.plot(i, X.iloc[i, 0], color=colors[labels[i]], marker='o', alpha=0.3)
# Plot centroids
for n, y in enumerate(centroids):
    plt.plot(len(X)/2, y, marker='^', color=colors[n], ms=10)
#Add labels
plt.title('K-means Clustering Model for Gross Profit (GP)')
plt.xlabel("index")
plt.ylabel("Gross Profit (GP)")
plt.show()


