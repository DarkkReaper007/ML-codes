import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("C://Users//Shasmeet//Desktop//Housing.csv")
data = data.fillna(0)  # Replace NaNs with 0

# Convert categorical to numerical
data['mainroad'] = data['mainroad'].replace({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].replace({'yes': 1, 'no': 0})
data['basement'] = data['basement'].replace({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].replace({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].replace({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].replace({'yes': 1, 'no': 0})
data['furnishingstatus'] = data['furnishingstatus'].replace({'furnished': 1, 'semi-furnished': 0.5, 'unfurnished': 0})

# Split features and target
X = data.iloc[:, 1:]
y = data['price'].values.astype(float)

# Normalize features
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Function to compute MSE
def mse(y_true, y_pred):
    n = len(y_true)
    return 1/n * np.sum((y_true - y_pred) ** 2)

# Function to perform gradient descent
def gradient_descent(X, y, learning_rate=0.1, iterations=50000):
    n_samples, n_features = X.shape
    weights = np.random.rand(n_features)  # Initialize weights
    bias = 0

    for _ in range(iterations):
        # Calculate predictions
        y_pred = np.dot(X, weights) + bias

        # Calculate gradients
        dw = -(2/n_samples) * np.dot(X.T, (y - y_pred))
        db = -(2/n_samples) * np.sum(y - y_pred)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

# Function to make predictions
def predict(X, weights, bias):
    return np.dot(X, weights) + bias

# Perform gradient descent
weights, bias = gradient_descent(X, y)
print("Weights:", weights)
print("Bias:", bias)

# Make predictions
predictions = predict(X, weights, bias)

# Calculate MSE
mse1 = mse(y, predictions)
print("Mean Squared Error:", mse1)

# Plot actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, color="blue", label="Predicted vs Actual")
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--", label="Perfect Fit Line")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.show()
