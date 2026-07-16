import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("Task-02/Mall_Customers.csv")

print("\nFirst 5 Rows of Dataset:\n")
print(df.head())

# ==========================
# CHECK DATASET
# ==========================

print("\nDataset Information:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

# ==========================
# SELECT FEATURES
# ==========================

X = df.iloc[:, [3, 4]].values

# ==========================
# FIND OPTIMAL NUMBER OF CLUSTERS
# (ELBOW METHOD)
# ==========================

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# ==========================
# ELBOW GRAPH
# ==========================

plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("WCSS")

plt.grid(True)

plt.show()

plt.savefig("Task-02/elbow_method.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# TRAIN K-MEANS MODEL
# ==========================

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

y_kmeans = kmeans.fit_predict(X)

# ==========================
# VISUALIZE CLUSTERS
# ==========================

plt.figure(figsize=(10,6))

plt.scatter(
    X[y_kmeans==0,0],
    X[y_kmeans==0,1],
    s=80,
    c='red',
    label='Cluster 1'
)

plt.scatter(
    X[y_kmeans==1,0],
    X[y_kmeans==1,1],
    s=80,
    c='blue',
    label='Cluster 2'
)

plt.scatter(
    X[y_kmeans==2,0],
    X[y_kmeans==2,1],
    s=80,
    c='green',
    label='Cluster 3'
)

plt.scatter(
    X[y_kmeans==3,0],
    X[y_kmeans==3,1],
    s=80,
    c='cyan',
    label='Cluster 4'
)

plt.scatter(
    X[y_kmeans==4,0],
    X[y_kmeans==4,1],
    s=80,
    c='magenta',
    label='Cluster 5'
)

# Centroids

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=250,
    c='yellow',
    marker='*',
    edgecolors='black',
    label='Centroids'
)

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.title("Customer Segmentation using K-Means Clustering")

plt.legend()

plt.grid(True)

plt.show()

# ==========================
# ADD CLUSTER COLUMN
# ==========================

df["Cluster"] = y_kmeans

print("\nDataset with Cluster Column:\n")

print(df.head())

plt.savefig("Task-02/customer_clusters.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# SAVE RESULT
# ==========================

df.to_csv(
    "Task-02/Customer_Segmentation_Result.csv",
    index=False
)

print("\nCustomer_Segmentation_Result.csv has been saved successfully.")