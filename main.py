from pandas import read_excel
from regression import UnivariateRegression

if __name__ == '__main__':
    dataset = read_excel('data/Concrete_Data.xls')
    dataset.dropna(axis="columns", how="any", inplace=True)

    X = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1]

    cement = X.iloc[:, 0]

    model_1 = UnivariateRegression()

    model_1.train(cement, y, max_iteration=100)
    model_1.plot()

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

