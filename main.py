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
    model = MultivariateRegression(learning_rate=0.1)
    model.train(X_train, y_train, max_iteration=100)
    y_pred = model.test(X_test)
    model.info(y_pred=y_pred, y_true=y_test)

    '''
    # For comparison with scikit-learn
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    test = LinearRegression()
    X_ = dataset.iloc[:, 0].values.reshape(-1, 1)
    test.fit(X, y)
    print(test.coef_, test.intercept_)
    print('Mean squared error: %.2f'
          % mean_squared_error(y, X.dot(test.coef_) + test.intercept_))
    # plt.scatter(X_all, y)
    # plt.plot(X_all, test.coef_ * X_all + test.intercept_, c='green')
    # plt.show()
    # mse_scikit = mean_squared_error(y, test.coef_ * X_ + test.intercept_)
    # print(mse_scikit)
    '''

