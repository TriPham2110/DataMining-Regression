from pandas import read_excel

import regression
from regression import UnivariateRegression, MultivariateRegression


def UnivariateFeature(X_train, y_train, X_test, y_test, col, learning_rate=0.01, max_iteration=100):
    if col < 0 or col >= X.shape[1]:
        return "Incorrect column index!"
    univariate_feature = X_train.iloc[:, col]
    model = UnivariateRegression(learning_rate=learning_rate)
    model.train(univariate_feature, y_train, max_iteration=max_iteration)
    y_pred = model.test(X_test.iloc[:, col])
    model.info(name=X.columns[col], y_pred=y_pred, y_true=y_test)
    # individual plot for univariate regression of a feature
    # model.plot(feature=X.columns[0], target=dataset.columns[-1])
    return model


if __name__ == '__main__':
    dataset = read_excel('data/Concrete_Data.xls')
    dataset.dropna(axis="columns", how="any", inplace=True)

    X = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1]

    # first 900 instances for training and the last 130 instances for testing
    X_train = X[:900]
    X_test = X[900:]
    y_train = y[:900]
    y_test = y[900:]

    '''
    Univariate regression for individual feature of Concrete Data
    '''
    model1 = UnivariateFeature(X_train, y_train, X_test, y_test, col=0, learning_rate=0.1, max_iteration=100)
    model2 = UnivariateFeature(X_train, y_train, X_test, y_test, col=1, learning_rate=0.1, max_iteration=100)
    model3 = UnivariateFeature(X_train, y_train, X_test, y_test, col=2, learning_rate=0.1, max_iteration=100)
    model4 = UnivariateFeature(X_train, y_train, X_test, y_test, col=3, learning_rate=0.1, max_iteration=100)
    model5 = UnivariateFeature(X_train, y_train, X_test, y_test, col=4, learning_rate=0.1, max_iteration=100)
    model6 = UnivariateFeature(X_train, y_train, X_test, y_test, col=5, learning_rate=0.1, max_iteration=100)
    model7 = UnivariateFeature(X_train, y_train, X_test, y_test, col=6, learning_rate=0.1, max_iteration=100)
    model8 = UnivariateFeature(X_train, y_train, X_test, y_test, col=7, learning_rate=0.1, max_iteration=100)

    # group of 3 plots for univariate regression of corresponding model obtained above
    # UnivariateRegression.subplot(model1, model2, model3, features=X.columns[0:3], target=dataset.columns[-1])
    # UnivariateRegression.subplot(model4, model5, model6, features=X.columns[3:6], target=dataset.columns[-1])
    # UnivariateRegression.subplot(model7, model8, features=X.columns[6:], target=dataset.columns[-1])

    '''
    Multivariate regression for all features of Concrete Data
    '''
    model9 = MultivariateRegression(learning_rate=0.1)
    model9.train(X_train, y_train, max_iteration=100)
    y_pred = model9.test(X_test)
    model9.info(y_pred=y_pred, y_true=y_test)

    print("############ MSE results for all models on training data ################")
    print("Model 1:", model1.MSE())
    print("Model 2:", model2.MSE())
    print("Model 3:", model3.MSE())
    print("Model 4:", model4.MSE())
    print("Model 5:", model5.MSE())
    print("Model 6:", model6.MSE())
    print("Model 7:", model7.MSE())
    print("Model 8:", model8.MSE())
    print("Model 9:", model9.MSE())
    print("############ ------------------------------------------- ################\n")

    print("############ MSE results for all models on testing data ################")
    print("Model 1:", regression.MSE_test(model1.test(X_test.iloc[:, 0]), y_test))
    print("Model 2:", regression.MSE_test(model2.test(X_test.iloc[:, 1]), y_test))
    print("Model 3:", regression.MSE_test(model3.test(X_test.iloc[:, 2]), y_test))
    print("Model 4:", regression.MSE_test(model4.test(X_test.iloc[:, 3]), y_test))
    print("Model 5:", regression.MSE_test(model5.test(X_test.iloc[:, 4]), y_test))
    print("Model 6:", regression.MSE_test(model6.test(X_test.iloc[:, 5]), y_test))
    print("Model 7:", regression.MSE_test(model7.test(X_test.iloc[:, 6]), y_test))
    print("Model 8:", regression.MSE_test(model8.test(X_test.iloc[:, 7]), y_test))
    print("Model 9:", regression.MSE_test(model9.test(X_test), y_test))
    print("############ ------------------------------------------ ################\n")

    '''
    # For comparison with scikit-learn
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    scikit_model = LinearRegression()
    # x_train = X_train.iloc[:, 0].values.reshape(-1, 1)    # univarariate
    scikit_model.fit(X_train, y_train)
    # plt.scatter(X_all, y)
    # plt.plot(X_all, test.coef_ * X_all + test.intercept_, c='green')
    # plt.show()
    # x_test = X_test.iloc[:, 0].values.reshape(-1, 1)        # univarariate
    # train_mse = mean_squared_error(y_train, scikit_model.coef_ * x_train + scikit_model.intercept_)
    # test_mse = mean_squared_error(y_test, scikit_model.coef_ * x_test + scikit_model.intercept_)
    train_mse = mean_squared_error(y_train, scikit_model.predict(X_train))
    test_mse = mean_squared_error(y_test, scikit_model.predict(X_test))
    print(train_mse, ', ', test_mse)
    '''

