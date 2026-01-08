import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("Iris.csv")

# Drop Id column
df = df.drop(columns=["Id"])

# Convert Species to binary using astype(int)
df["Species"] = (df["Species"] != "Iris-setosa").astype(int)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# -------------------------------
# 2. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# -------------------------------
# 3. Compute Class Statistics
# -------------------------------
classes = np.unique(y_train)

means = []
vars_ = []
priors = []

for c in classes:
    X_c = X_train[y_train == c]
    means.append(X_c.mean(axis=0))
    vars_.append(X_c.var(axis=0))
    priors.append(X_c.shape[0] / len(y_train))

means = np.array(means)
vars_ = np.array(vars_)
priors = np.array(priors)

# -------------------------------
# 4. Gaussian Probability
# -------------------------------
def gaussian_pdf(x, mean, var):
    return np.exp(-(x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)

# -------------------------------
# 5. Prediction
# -------------------------------
def predict(X):
    predictions = []

    for x in X:
        posteriors = []

        for i in range(len(classes)):
            prior = np.log(priors[i])
            likelihood = np.sum(np.log(gaussian_pdf(x, means[i], vars_[i])))
            posteriors.append(prior + likelihood)

        predictions.append(classes[np.argmax(posteriors)])

    return np.array(predictions)

# -------------------------------
# 6. Evaluate
# -------------------------------
y_pred = predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)