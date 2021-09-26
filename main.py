from pandas import read_excel
from regression import UnivariateRegression


def UnivariateFeature(X, y, col, learning_rate=0.01, max_iteration=100):
    if col < 0 or col >= X.shape[1]:
        return "Incorrect column index!"
    univariate_feature = X.iloc[:, col]
    model = UnivariateRegression(learning_rate=learning_rate)
    model.train(univariate_feature, y, max_iteration=max_iteration)
    model.plot(feature=X.columns[col], target=dataset.columns[-1])
    return model


if __name__ == '__main__':
    dataset = read_excel('data/Concrete_Data.xls')
    dataset.dropna(axis="columns", how="any", inplace=True)

    X = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1]

    '''
    Univariate regression for individual feature of Concrete Data
    '''
    # model1 = UnivariateFeature(X, y, col=0, learning_rate=0.1, max_iteration=100)
    # model2 = UnivariateFeature(X, y, col=1, learning_rate=0.1, max_iteration=100)
    model3 = UnivariateFeature(X, y, col=2, learning_rate=0.1, max_iteration=100)

    # UnivariateRegression.subplot(model1, model2, model3, features=X.columns[0:3], target=dataset.columns[-1])

    '''
    For comparison with scikit-learn
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    test = LinearRegression()
    X_ = dataset.iloc[:, 2].values.reshape(-1, 1)
    test.fit(X_, y)
    plt.scatter(X_, y)
    plt.plot(X_, test.coef_ * X_ + test.intercept_, c='green')
    plt.show()
    '''

