import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
features = ['OverallQual','GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','YearBuilt','YearRemodAdd']
X_train = train_data[features].values
y_train = train_data['SalePrice'].values
X_test = test_data[features].values
y_test = test_data['SalePrice'].values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error on Test Set: {mse_test}")
