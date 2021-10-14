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

        # list of slopes, list of intercepts, list of costs (each index corresponds to an iteration)
        self.m = []
        self.b = []
        self.C = []

    def gradient(self, iter):
        n = len(self.X_norm)
        derivative_wrt_m = 0
        derivative_wrt_b = 0

        for i in range(n):
            derivative_wrt_m = derivative_wrt_m + (-2) * self.X_norm[i] * (self.y[i] - (self.m[iter] * self.X_norm[i] + self.b[iter]))
            derivative_wrt_b = derivative_wrt_b + (-2) * (self.y[i] - (self.m[iter] * self.X_norm[i] + self.b[iter]))

        derivative_wrt_m = (1/n) * derivative_wrt_m
        derivative_wrt_b = (1/n) * derivative_wrt_b

        return derivative_wrt_m, derivative_wrt_b

    def train(self, X, y, max_iteration=1000):
        self.max_iteration = max_iteration
        self.X = X
        self.X_norm = normalize(X, 'univariate')
        self.y = y

        # initial slope, intercept, and cost
        self.m.append(1)
        self.b.append(0)
        self.C.append(0)

        for i in range(max_iteration):
            # Plug slope and intercept into the gradient
            derivative_wrt_m, derivative_wrt_b = self.gradient(i)

            # Calculate step sizes
            step_size_m = self.learning_rate * derivative_wrt_m
            step_size_b = self.learning_rate * derivative_wrt_b

            # Calculate new slop and intercept
            new_m = self.m[i] - step_size_m
            new_b = self.b[i] - step_size_b

            # Check if the change in the cost function between consecutive iteration is negligible
            n = len(self.y)
            new_C = (np.sum(np.array(self.y) - (new_m * np.array(self.X_norm) + new_b)) ** 2) / n
            if abs(self.C[i] - new_C) < 1e-5:
                print("Change in the objective function between steps is negligeable starting at iteration ", i)
                break
            else:
                self.C.append(new_C)
                self.m.append(new_m)
                self.b.append(new_b)

    def plot(self, feature, target):
        plt.scatter(self.X, self.y)
        plt.plot(self.X, self.m[-1] * self.X_norm + self.b[-1], c='red')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title('Univariate regression plot (training data)')
        plt.show()

    # MSE after training
    def MSE(self):
        mse = 0
        n = len(self.y)
        for i in range(n):
            mse = mse + (self.y[i] - (self.m[-1] * self.X_norm[i] + self.b[-1]))**2
        mse = (1/n) * mse
        return mse[0]

    def test(self, X_test):
        X_test_norm = normalize(X_test, 'univariate')
        y_pred = []
        for i in range(len(X_test_norm)):
            pred = self.m[-1] * X_test_norm[i] + self.b[-1]
            y_pred.append(pred)
        return y_pred

    def info(self, name, y_pred, y_true):
        print("############ Univariate regression for feature " + name + " ################")
        print("Max iterations:", self.max_iteration)
        print("Learning rate:", self.learning_rate)
        print("MSE training:", self.MSE())
        print("MSE testing:", MSE_test(y_pred, y_true))
        print("############ ------------ ################\n")

    @staticmethod
    def subplot(model1=None, model2=None, model3=None, features=None, target=None):
        if model1:
            plt.subplot(2, 3, 1)
            plt.scatter(model1.X, model1.y)
            plt.plot(model1.X, model1.m[-1] * model1.X_norm + model1.b[-1], c='red')
            plt.xlabel(features[0])
            plt.ylabel(target)

        if model2:
            plt.subplot(2, 3, 2)
            plt.scatter(model2.X, model2.y)
            plt.plot(model2.X, model2.m[-1] * model2.X_norm + model2.b[-1], c='red')
            plt.xlabel(features[1])
            plt.ylabel(target)
            plt.suptitle('Univariate regression plots for individual features (training data)', fontsize=18)

        if model3:
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


# MSE for test data
def MSE_test(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    n = len(y_true)
    mse = np.sum((y_pred - y_true) ** 2) / n
    return mse


def closed_form(X_train, y_train, X_test, y_test):
    X_train = X_train.values
    XTX_inverse = np.linalg.inv(np.dot(np.transpose(X_train), X_train))
    closed_form_sol = np.dot(XTX_inverse, np.transpose(X_train)).dot(y_train)
    mse_train = np.sum((y_train - np.dot(closed_form_sol, np.transpose(X_train))) ** 2) / len(y_train)

    X_test = X_test.values
    mse_test = np.sum((y_test - np.dot(closed_form_sol, np.transpose(X_test))) ** 2) / len(y_test)
    return closed_form_sol, mse_train, mse_test


class MultivariateRegression:

    def __init__(self, learning_rate=0.01, random_state=42, degree=1):
        self.learning_rate = learning_rate
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_iteration = None
        self.X = None           # original X (vector of features)
        self.X_norm = None      # normalized X
        self.y = None
        self.a = None           # (a0, a1, ..., ap)^T, a0 in place of b in model
        self.C = []
        self.degree = degree    # handles polynomial regression

    def gradient(self, iter):
        n = len(self.X_norm)

        dot_products = np.dot(self.a, np.transpose(self.X_norm))

        derivative_wrt_as = []
        for i in range(n):
            derivative_wrt_a = []
            for j in range(len(self.X_norm[0])):
                derivative_wrt_a.append(0)
            derivative_wrt_as.append(derivative_wrt_a)

        # No need to worry about checking degree here
        # since x_norm has been processed to include columns that contain results of x^2
        for i in range(n):
            for j in range(len(self.X_norm[0])):
                if j == 0:  # derivative wrt bias b (or param a0)
                    derivative_wrt_as[i][j] = (-2) * (self.y[i] - dot_products[i])
                else:
                    derivative_wrt_as[i][j] = (-2) * self.X_norm[i][j] * (self.y[i] - dot_products[i])

        derivative_wrt_as = (1/n) * np.sum(derivative_wrt_as, axis=0)
        return derivative_wrt_as

    def train(self, X, y, max_iteration=1000):
        self.max_iteration = max_iteration
        X_copy = X.copy()
        self.X = X_copy.values

        # concatenate all necessary x before normalizing
        if self.degree >= 2:
            degrees = np.arange(2, (self.degree + 1), 1)
            for d in degrees:
                self.X = np.append(self.X, X_copy.values ** d, axis=1)

        # normalize x to x_norm and add bias term to input x_norm
        self.X_norm = normalize(self.X, 'multivariate')
        p = len(self.X_norm[0])

        self.X_norm = np.append([[1.]] * (len(self.X_norm)), self.X_norm, axis=1)
        self.y = y

        # (p + 1) here contains all parameters
        self.a = np.array([0] * (p + 1))
        self.C.append(0)

        for i in range(max_iteration):
            derivative_wrt_a = self.gradient(i)

            # Calculate step sizes
            step_size_a = self.learning_rate * derivative_wrt_a

            # Calculate new params
            new_a = self.a - step_size_a

            # Check if the change in the cost function between consecutive iteration is negligible
            n = len(self.y)
            new_C = (np.sum(np.array(self.y) - np.dot(new_a, np.transpose(self.X_norm))) ** 2) / n
            if abs(self.C[i] - new_C) < 1e-5:
                print("Change in the objective function between steps is negligeable starting at iteration ", i)
                break
            else:
                self.C.append(new_C)
                self.a = new_a

    # MSE after training
    def MSE(self):
        n = len(self.y)
        mse = np.sum((self.y - np.dot(self.a, np.transpose(self.X_norm))) ** 2) / n
        return mse

    def test(self, X_test):
        X_test_copy = X_test.copy()
        X_test_ = X_test_copy

        if self.degree >= 2:
            degrees = np.arange(2, self.degree + 1, 1)
            for d in degrees:
                X_test_ = np.append(X_test_, X_test_copy.values ** d, axis=1)

        X_test_norm = normalize(X_test_, 'multivariate')
        X_test_norm = np.append([[1.]] * (len(X_test_norm)), X_test_norm, axis=1)
        return np.dot(self.a, np.transpose(X_test_norm))

    def info(self, y_pred, y_true):
        if self.degree == 1:
            print("############ Multivariate linear regression for all features ################")
        elif self.degree == 2:
            print("############ Multivariate quadratic regression for all features ################")
        else:
            print("############ Multivariate polynomial of degree", self.degree, "regression for all features ################")
        print("Max iterations:", self.max_iteration)
        print("Learning rate:", self.learning_rate)
        print("MSE training:", self.MSE())
        print("MSE testing:", MSE_test(y_pred, y_true))
        print("############ ------------ ################\n")