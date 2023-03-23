import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Generating data
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0, 0, 1, 1, 1])
# Reshaping x array
x = x.reshape((-1, 1))
# Creating a logistic regression model and fitting the data
model = LogisticRegression().fit(x, y)
# # Printing the coefficients of the model
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)
# Plotting the data and logistic regression curve
plt.scatter(x, y)
plt.plot(x,y)
plt.title("Logistic regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
