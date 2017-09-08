import numpy as np
from math import sqrt

# Calculate the mean
def mean(values):
    return sum(values)/float(len(values))

# Calculate the variance
def variance(values, mean):
    return sum([(x - mean)**2 for x in values])


# Calculate Covariance
def covariance(x, y, x_mean, y_mean):
    covar = 0
    for i in range(len(x)):
        covar += (x[i] - x_mean)*(y[i] - y_mean)
    return covar

# Calculate coefficient
def coefficients(datasets):
    X = [row[0] for row in datasets]
    y = [row[1] for row in datasets]
    x_mean, y_mean = mean(X), mean(y)
    b1 = covariance(X, y, x_mean, y_mean)/variance(X, x_mean)
    b0 = y_mean - b1*x_mean
    return [b0, b1]

# Simple Linear Regression Algorithm
def simple_linear_regression(train):
    prediction = list()
    b0, b1 = coefficients(train)
    for row in train:
        yhat = b0 + b1*row[0]
        prediction.append(yhat)
    return prediction

# Calculate root mean square error
def rmse_metrics(actual, prediction):
    sum_error = 0
    for i in range(len(actual)):
        predication_error = prediction[i] - actual[i]
        sum_error += (predication_error**2)
    mean_error = sum_error/float(len(actual))
    return sqrt(mean_error)

# Evaluate the simple linear algorithm
def evaluate_algorithm(dataset, algorithm):
    prediction = simple_linear_regression(dataset)
    print(prediction)
    actual = [row[-1] for row in dataset]
    rmse = rmse_metrics(actual, prediction)
    return rmse


dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
rmse = evaluate_algorithm(dataset, simple_linear_regression)
print("RMSE:", rmse)
