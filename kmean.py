import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Iris dataset from CSV
# -----------------------------
df = pd.read_csv("Iris.csv")

# Drop Id column
df = df.drop(columns=["Id"])

# Convert Species to binary (Setosa = 0, Others = 1)
df["Species"] = (df["Species"] != "Iris-setosa").astype(int)

# -----------------------------
# Select features
# -----------------------------
X = df[["SepalLengthCm", "SepalWidthCm"]].values
k = 3

# -----------------------------
# Initialize centroids randomly
# -----------------------------
centroids = X[np.random.choice(len(X), k, replace=False)]

# -----------------------------
# K-Means algorithm
# -----------------------------
for _ in range(50):
    distances = np.linalg.norm(X[:, None] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    for i in range(k):
        centroids[i] = X[labels == i].mean(axis=0)

# -----------------------------
# Plot clusters
# -----------------------------
colors = ['r', 'g', 'b']

for i in range(k):
    plt.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        c=colors[i],
        label=f"Cluster {i+1}"
    )

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='black',
    marker='x',
    s=100,
    label="Centroids"
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()
plt.show()
