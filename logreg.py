import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# SIGMOID FUNCTION
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------
# LOGISTIC REGRESSION USING GRADIENT DESCENT
# -----------------------------
def logistic_regression(X, y, iterations=200, lr=0.001):
    w = np.zeros(X.shape[1])   # initialize weights

    for _ in range(iterations):
        z = np.dot(X, w)       # linear combination
        h = sigmoid(z)         # prediction
        error = h - y          # error
        gradient = np.dot(X.T, error) / y.shape[0]
        w = w - lr * gradient  # update weights

    return w

# -----------------------------
# LOAD DATASET FROM CSV
# -----------------------------
import pandas as pd

df = pd.read_csv("iris.csv")

# Drop Id column
df = df.drop(columns=["Id"])

# Convert species to binary (IMPORTANT)
df["Species"] = (df["Species"] != "Iris-setosa").astype(int)

# Split into X and y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=9
)

# -----------------------------
# FEATURE SCALING
# -----------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------
# TRAIN MODEL
# -----------------------------
weights = logistic_regression(X_train, y_train)

# -----------------------------
# PREDICTION & ACCURACY
# -----------------------------
y_pred = sigmoid(np.dot(X_test, weights)) > 0.5
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)