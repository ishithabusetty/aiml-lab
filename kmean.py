
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("Iris.csv")

# Drop Id column
df = df.drop(columns=["Id"])

# Split into X and y
X = df.iloc[:, :-1].values   # features
y = df.iloc[:, -1].values    # labels (not used)

# -------------------------------
# 2. K-Means parameters
# -------------------------------
k = 3
centroids = X[np.random.choice(len(X), k, replace=False)]

# -------------------------------
# 3. K-Means algorithm
# -------------------------------
for _ in range(100):

    labels = []

    # Assign points to nearest centroid
    for x in X:
        distances = [np.linalg.norm(x - c) for c in centroids]
        labels.append(np.argmin(distances))

    labels = np.array(labels)

    # Update centroids
    for i in range(k):
        centroids[i] = X[labels == i].mean(axis=0)

# -------------------------------
# 4. Silhouette Score
# -------------------------------
score = silhouette_score(X, labels)
print("Silhouette Score:", score)

# -------------------------------
# 5. Plot clusters (first 2 features)
# -------------------------------
colors = ['r', 'g', 'b']

for i in range(k):
    plt.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        c=colors[i],
        label=f'Cluster {i+1}'
    )

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='black',
    marker='x',
    label='Centroids'
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means with Silhouette Score")
plt.legend()
plt.show()
