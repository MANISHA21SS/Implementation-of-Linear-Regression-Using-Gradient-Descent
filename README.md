# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import required libraries
Import numpy as np for numerical operations (like arrays, matrix multiplications).
Import matplotlib.pyplot as plt to visualize data and the cost function.

Step 2: Load and understand the dataset

Step 3: Preprocess the dataset
Separate the input feature X (e.g., city population) and output label y (e.g., city profit).
Reshape X and y properly if needed (ensure they are column vectors).

Step 4: Initialize model parameters
Initialize the parameters (also called weights or coefficients) theta:
Set all initial values of theta to zeros (or small random numbers).

Step 5: Define the cost function
Define the Mean Squared Error (MSE) cost function

Step 6: Implement the Gradient Descent algorithm

Step 7: Train the model

Step 8: Make predictions

Step 9: Evaluate the model
Plot the regression line over the scatter plot of the data.
Plot the cost function value versus number of iterations to check if the cost is decreasing and if convergence is reached.

Step 10: Display final outputs
Print the final learned parameters theta.



## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Manisha selvakumari.S.S.
RegisterNumber: 212223220055 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(x).dot(theta).reshape(-1,1)
        errors =(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta

data=pd.read_csv('50_Startups.csv',header=None)
print("Name: Manisha selvakumari.S.S.")
print("Reg No: 212223220055")
print(data.head())

x=(data.iloc[1:,:-2].values)
print(x)

x1=x.astype(float)
scaler=StandardScaler()

y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)

theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"predicted value: {pre}")

```

## Output:

![Screenshot 2025-04-28 213338](https://github.com/user-attachments/assets/bd37c1cc-6787-4941-8457-fe3d85d10281)


![Screenshot (248)](https://github.com/user-attachments/assets/bdf98e16-a946-4918-9267-3302d6e44609)


![Screenshot (249)](https://github.com/user-attachments/assets/ec0e0d1f-9b0b-469a-b2c6-84623acddd7c)


![Screenshot (250)](https://github.com/user-attachments/assets/c77eeeb9-4473-4423-88b4-b84bcdb41805)


![Screenshot (251)](https://github.com/user-attachments/assets/355a804d-63ae-49c7-9a31-60ef4f6631f9)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
