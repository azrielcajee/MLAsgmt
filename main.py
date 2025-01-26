import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the datasets
logisticX = pd.read_csv('logisticX.csv', header=None)
logisticY = pd.read_csv('logisticY.csv', header=None)

# Extract features and labels
X = logisticX.values
y = logisticY.values.ravel()

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e6)
model.fit(X_scaled, y)

# Extract coefficients, intercept, and calculate cost function
coefficients = model.coef_[0]
intercept = model.intercept_[0]
y_pred_proba = model.predict_proba(X_scaled)[:, 1]
cost_function_value = log_loss(y, y_pred_proba)

# Print results
print("Coefficients:", coefficients)
print("Intercept:", intercept)
print("Cost Function Value (Log-Loss):", cost_function_value)

# Plot cost function vs. iterations
from sklearn.linear_model import SGDClassifier

# Custom training loop for SGD
sgd_model = SGDClassifier(loss='log', learning_rate='constant', eta0=0.1, max_iter=50, tol=None, random_state=42)
costs = []

for epoch in range(1, 51):
    sgd_model.partial_fit(X_scaled, y, classes=np.unique(y))
    y_pred_proba_sgd = sgd_model.predict_proba(X_scaled)[:, 1]
    cost = log_loss(y, y_pred_proba_sgd)
    costs.append(cost)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), costs, marker='o', color='blue', label='Cost (Log-Loss)')
plt.title("Cost Function vs. Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost Function Value (Log-Loss)")
plt.legend()
plt.grid()
plt.show()

# Plot dataset and decision boundary
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(8, 6))

    # Scatter plot of data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')

    # Plot decision boundary
    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_values = -(model.coef_[0][0] * x_values + model.intercept_[0]) / model.coef_[0][1]
    plt.plot(x_values, y_values, color='green', label='Decision Boundary')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()

plot_decision_boundary(X_scaled, y, model, "Original Dataset with Decision Boundary")

# Add squared features
X_new = np.hstack((X_scaled, X_scaled ** 2))

# Train logistic regression on new dataset
model_new = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e6)
model_new.fit(X_new, y)

plot_decision_boundary(X_new[:, :2], y, model_new, "Dataset with Squared Features and Decision Boundary")

# Confusion Matrix and Metrics
y_pred = model.predict(X_scaled)
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
