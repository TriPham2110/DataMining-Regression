"""
Ref: https://www.youtube.com/watch?v=sDv4f4s2SB8 (Gradient Descent step-by-step by Josh Starmer)
     https://becominghuman.ai/univariate-linear-regression-with-mathematics-in-python-8ebac73f9b12
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class UnivariateRegression:

    def __init__(self, learning_rate=0.01, random_state=42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_iteration = None
        self.X = None       # original X
        self.X_norm = None  # normalized X
        self.y = None

        # make m a list of slopes and b a list of intercepts to store results
        self.m = []
        self.b = []

    def gradient(self, index):
        n = len(self.X_norm)
        derivative_wrt_m = 0
        derivative_wrt_b = 0

        for i in range(n):
            derivative_wrt_m = derivative_wrt_m + (-2) * self.X_norm[i] * (self.y[i] - (self.m[index] * self.X_norm[i] + self.b[index]))
            derivative_wrt_b = derivative_wrt_b + (-2) * (self.y[i] - (self.m[index] * self.X_norm[i] + self.b[index]))

        derivative_wrt_m = (1/n) * derivative_wrt_m
        derivative_wrt_b = (1/n) * derivative_wrt_b

        return derivative_wrt_m, derivative_wrt_b

    def train(self, X, y, max_iteration=1000):
        self.max_iteration = max_iteration
        self.X = X
        self.X_norm = normalize(X, 'univariate')
        self.y = y

        # initial slope and intercept
        self.m.append(1)
        self.b.append(0)

        for i in range(max_iteration):
            # Plug slope and intercept into the gradient
            derivative_wrt_m, derivative_wrt_b = self.gradient(i)

            # Calculate step sizes
            step_size_m = self.learning_rate * derivative_wrt_m
            step_size_b = self.learning_rate * derivative_wrt_b

            # Calculate new slop and intercept
            self.m.append(self.m[i] - step_size_m)
            self.b.append(self.b[i] - step_size_b)

    def plot(self, feature, target):
        plt.scatter(self.X, self.y)
        plt.plot(self.X, self.m[-1] * self.X_norm + self.b[-1], c='red')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()

    def MSE(self):
        mse = 0
        n = len(self.y)
        for i in range(n):
            mse = mse + (self.y[i] - (self.m[-1] * self.X_norm[i] + self.b[-1]))**2
        mse = (1/n) * mse
        return mse

    @staticmethod
    def subplot(model1, model2, model3, features, target):
        plt.subplot(2, 3, 1)
        plt.scatter(model1.X, model1.y)
        plt.plot(model1.X, model1.m[-1] * model1.X_norm + model1.b[-1], c='red')
        plt.xlabel(features[0])
        plt.ylabel(target)

        plt.subplot(2, 3, 2)
        plt.scatter(model2.X, model2.y)
        plt.plot(model2.X, model2.m[-1] * model2.X_norm + model2.b[-1], c='red')
        plt.xlabel(features[1])
        plt.ylabel(target)

        plt.subplot(2, 3, 3)
        plt.scatter(model3.X, model3.y)
        plt.plot(model3.X, model3.m[-1] * model3.X_norm + model3.b[-1], c='red')
        plt.xlabel(features[2])
        plt.ylabel(target)

        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()


def normalize(X, datatype):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
    if datatype == 'univariate':
        X = X.reshape(-1, 1)
        return MinMaxScaler().fit_transform(X)
    if datatype == 'multivariate':
        return MinMaxScaler().fit_transform(X)


class MultivariateRegression:

    def __init__(self, learning_rate=0.01, random_state=42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_iteration = None
        self.X = None       # original X (vector of features with length p)
        self.X_norm = None  # normalized X
        self.y = None
        self.a = None       # (a0, a1, ..., ap)^T, a0 in place of b in model

    def gradient(self):
        n = len(self.X_norm)

        dot_products = np.dot(self.a, np.transpose(self.X_norm))

        derivative_wrt_as = []
        for i in range(n):
            derivative_wrt_a = []
            for j in range(len(self.X.columns)):
                derivative_wrt_a.append(0)
            derivative_wrt_as.append(derivative_wrt_a)

        for i in range(n):
            for j in range(len(self.X.columns)):
                derivative_wrt_as[i][j] = (-2) * self.X_norm[i][j] * (self.y[i] - dot_products[i])

        derivative_wrt_as = (1/n) * np.sum(derivative_wrt_as, axis=0)
        return derivative_wrt_as

    def train(self, X, y, max_iteration=1000):
        self.max_iteration = max_iteration
        self.X = X
        self.X_norm = normalize(X, 'multivariate')

        # add bias term to input x and x_norm
        self.X.insert(loc=0, column='Bias term', value=1)
        self.X_norm = np.append([[1.]] * (len(X)), self.X_norm, axis=1)
        self.y = y

        p = len(X.columns)
        self.a = np.array([0] * p)

        for i in range(max_iteration):
            derivative_wrt_a = self.gradient()

            # Calculate step sizes
            step_size_a = self.learning_rate * derivative_wrt_a

            # Calculate new slop and intercept
            self.a = self.a - step_size_a   # an array of size 9 corresponding to 9 params

    def MSE(self):
        n = len(self.y)
        mse = np.sum((self.y - np.dot(self.a, np.transpose(self.X_norm))) ** 2) / n
        return mse