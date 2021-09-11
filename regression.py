"""
Ref: https://www.youtube.com/watch?v=sDv4f4s2SB8 (Gradient Descent step-by-step by Josh Starmer)
     https://becominghuman.ai/univariate-linear-regression-with-mathematics-in-python-8ebac73f9b12
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class UnivariateRegression:

    def __init__(self, learning_rate=0.01, random_state=42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_iteration = None
        self.X = None
        self.y = None

        # make m a list of slopes and b a list of intercepts to store results
        self.m = []
        self.b = []

    def gradient(self, index):
        n = len(self.X)
        derivative_wrt_m = 0
        derivative_wrt_b = 0

        for i in range(n):
            derivative_wrt_m = derivative_wrt_m + (-2) * self.X[i] * (self.y[i] - (self.m[index] * self.X[i] + self.b[index]))
            derivative_wrt_b = derivative_wrt_b + (-2) * (self.y[i] - (self.m[index] * self.X[i] + self.b[index]))

        derivative_wrt_m = (1/n) * derivative_wrt_m
        derivative_wrt_b = (1/n) * derivative_wrt_b

        return derivative_wrt_m, derivative_wrt_b

    def train(self, X, y, max_iteration=1000):
        self.max_iteration = max_iteration
        self.X = UnivariateRegression.normalize(X, 'univariate')
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

    def plot(self):
        plt.scatter(self.X, self.y)
        plt.plot(self.X, self.m[-1] * self.X + self.b[-1], c='red')
        plt.show()

    @staticmethod
    def normalize(X, datatype):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if datatype == 'univariate':
            X = X.reshape(-1, 1)
        return StandardScaler().fit_transform(X)