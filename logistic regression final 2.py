import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

full_data = pd.read_csv("User_Data.csv")
#print(full_data)

full_data = full_data.drop("User ID", axis = 1)
#print(full_data)

full_data['Gender'] = full_data['Gender'].replace({'Male':0, 'Female':1})
#print(full_data)

X_full_data = full_data[['Gender','Age','EstimatedSalary','Purchased']].values
#print(X_full_data)
X_training = X_full_data[:200,:]
print(X_training)

Y_full_data = full_data[['Purchased']].values
Y_training = Y_full_data[:200,:]
print(Y_training)
#print(Y_full_data)

observations,features = X_full_data.shape
#print(observations,features)
print(X_full_data.shape)
print(Y_full_data.shape)
X_training_final = X_training.T
Y_training_final = Y_training.T
X_final_data = X_full_data.T
Y_final_data = Y_full_data.T



def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression(iter,learning_rate,Y,X):
    m,n = X_full_data.shape
    w = np.zeros((n,1))
    b = 0
    cost_list = []
    for i in range(iter):
        linear_prediction = np.dot(w.T,X) + b
        prediction = sigmoid(linear_prediction)

        cost = -(1/m)*np.sum( Y*np.log(prediction) + (1-Y)*np.log(1-prediction))
        dW = (1/m)*np.dot(prediction-Y, X.T)
        dB = (1/m)*np.sum(prediction - Y)

        w = w - learning_rate*dW.T
        b = b - learning_rate*dB

        cost_list.append(cost)
    return w, b, cost_list

lr = 0.000000000005
iter = 10000
w, b, cost_list = logistic_regression(iter,lr,Y_training_final,X_training_final) 
plt.plot(np.arange(iter), cost_list)
plt.show() 

def accuracy(X, Y, W, B):
    
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    
    A = A > 0.5
    
    A = np.array(A, dtype = 'int64')
    
    acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
    
    print("Accuracy of the model is : ", round(acc, 2), "%")
accuracy(X_final_data, Y_final_data, w, b)