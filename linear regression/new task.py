import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, lr=0.001, iter=1000):
        self.learning_rate = lr
        self.iterations = iter
        self.weights = None
        self.bias = None
        self.loss = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            cost = (1 / n_samples) * np.sum((y - y_pred) ** 2)
            self.loss.append(cost)
            dw = -(2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2 / n_samples) * np.sum(y - y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def plot_loss(self):
        plt.plot(range(self.iterations), self.loss, color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


train_data = pd.read_csv('train.csv')
train_data.fillna(train_data.mean(), inplace=True)
features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt', 'YearRemodAdd']
X_train = train_data[features].values
y_train = train_data['SalePrice'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
test_data = pd.read_csv('test.csv')

X_test = test_data[features].values
y_test = test_data['SalePrice'].values
X_test = scaler.transform(X_test)

model = LinearRegression(lr=0.001, iter=10000)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error on Test Set: {mse_test}")


model.plot_loss()
