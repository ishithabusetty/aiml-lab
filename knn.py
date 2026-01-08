import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("placementdata.csv")

# Drop Id column
df = df.drop(columns=["Id"])

# Convert text columns to numbers
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# Features and labels
# -----------------------------
X = df.drop(columns=["Placed_Yes"]).values
y = df["Placed_Yes"].values

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

k = 3  # number of neighbors

# -----------------------------
# KNN Prediction
# -----------------------------
y_pred = []

for x_test in X_test:
    distances = []

    for x_train in X_train:
        dist = np.linalg.norm(x_test - x_train)
        distances.append(dist)

    # Get k nearest neighbors
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]

    # Majority voting
    prediction = Counter(k_labels).most_common(1)[0][0]
    y_pred.append(prediction)

y_pred = np.array(y_pred)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# -----------------------------
# Confusion Matrix
# -----------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
