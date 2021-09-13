from pandas import read_excel
from regression import UnivariateRegression


def UnivariateFeature(X, y, col, max_iteration=100):
    if col < 0 or col >= X.shape[1]:
        return "Incorrect column index!"
    univariate_feature = X.iloc[:, col]
    model = UnivariateRegression()
    model.train(univariate_feature, y, max_iteration=max_iteration)
    model.plot()


if __name__ == '__main__':
    dataset = read_excel('data/Concrete_Data.xls')
    dataset.dropna(axis="columns", how="any", inplace=True)

    X = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1]

    '''
    Univariate regression for individual feature of Concrete Data
    '''
    UnivariateFeature(X, y, col=0, max_iteration=100)

    '''
    For comparison with scikit-learn
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    test = LinearRegression()
    cement_ = cement.values.reshape(-1, 1)
    test.fit(cement_, y)
    plt.scatter(cement_, y)
    plt.plot(cement_, test.coef_ * cement_ + test.intercept_, c='green')
    plt.show()
    '''

