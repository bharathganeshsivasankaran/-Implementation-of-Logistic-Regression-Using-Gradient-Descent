# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm
```
1.Import the data file and import numpy, matplotlib and scipy.
2.Visulaize the data and define the sigmoid function, cost function and gradient descent.
3.Plot the decision boundary .
4.Calculate the y-prediction.
```
## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Bharathganesh.S
RegisterNumber:  212222230022

```
```

#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/classes/ML/New folder/Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('sl_no',axis=1)

#catogorising col for further labegling
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the colums
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

#display dataset
dataset

#selecting the features and labels
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#display dependent variables
Y

#initialize the model parameter
theta = np.random.randn(X.shape[1])
y=Y

#define the sigmoid function 
def sigmoid(z):
    return 1/(1+np.exp(-z))

#define the loss function 
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

#define the gradient descent algorithm
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta

#train the model
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

# make the predictions
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)

#evaluate the model
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy : ",accuracy)
print(y_pred)
print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```

## Output:
### dataset:
![image](https://github.com/bharathganeshsivasankaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119478098/24b4d12a-fff6-4739-9515-b3b32674e342)
### dataset.dtypes:
![image](https://github.com/bharathganeshsivasankaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119478098/bd420e17-3557-4fc7-94b8-47c405ba868d)
### dataset:
![image](https://github.com/bharathganeshsivasankaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119478098/71b6c22d-7114-4fa2-943e-d6e6c22877a4)
### Y:
![image](https://github.com/bharathganeshsivasankaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119478098/6005f905-a14f-4d4f-a54e-76efd734a784)
### Accuracy:
![image](https://github.com/bharathganeshsivasankaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119478098/09ce8877-6c0b-4b2f-9cce-498497e18475)
### y_pred:
![image](https://github.com/bharathganeshsivasankaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119478098/b2e2dffb-00c0-42c8-9115-97903283e3c2)
### Y:
![image](https://github.com/bharathganeshsivasankaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119478098/e9fb1fbf-3df5-4524-af15-b7be228f4eaa)
### y_prednew:
![image](https://github.com/bharathganeshsivasankaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119478098/3b130eb0-c97f-4afe-b697-64d9202222c1)
### y_prednew:
![image](https://github.com/bharathganeshsivasankaran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119478098/c391a569-a460-4cb8-97d2-d730ade88e5f)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

