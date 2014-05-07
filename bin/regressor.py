__author__ = 'roz016'

import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math


def test(file_path):
    """

    :param file_path:
    """
    df = pandas.read_csv(file_path, na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]

    print('___________')
    print(me_no_regression(df, 'MetarwindSpeed', 'WindSpd'))
    print(rmse_no_regression(df, 'MetarwindSpeed', 'WindSpd'))
    simple_regression(df, 'MetarwindSpeed', 'WindSpd')


def me_no_regression(df, col1, col2):
    """

    :param file_path:
    """
    values = (df[col1] - df[col2])**2

    values = values.apply(np.sqrt)

    return values.sum()/df.shape[0]


def rmse_no_regression(df, col1, col2):
    """

    :param file_path:
    """
    values = (df[col1] - df[col2])**2

    return math.sqrt(values.sum()/df.shape[0])


def plot_simple_regression(df, col1, col2):

    df = df[[col2, col1]]
    data = np.matrix(df)

    org = LinearRegression()
    x, y = data[:,0], data[:,1]
    org.fit(x, y)

    print(org.coef_)
    print(org.intercept_)

    original_score = '{0:.3f}'.format(org.score(x, y))

    plt.xlim(df[col2].min()-1, df[col2].max()+1)
    plt.ylim(df[col1].min()-1, df[col1].max()+1)

    plt.scatter(df[col2], df[col1], marker='x', color='b')
    plt.xlabel('GFS Wind Speed [knots]')
    plt.ylabel('Metar Wind Speed[knots]')

    testo = np.arange(df[col2].min(), df[col2].max(), 0.1)
    testo = np.array(np.matrix(testo).T)

    plt.plot(testo, org.predict(testo), 'k')
    plt.title('Linear Regression')
    plt.grid()

    plt.show()


def simple_regression(df, col1, col2):

    df = df[[col2, col1]]
    data = np.matrix(df)

    y = np.matrix(np.array(df[col1]))
    x0 = np.array(df[col2])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    b = y * x.T * np.linalg.inv(x*x.T)
    return b


def degrees_distance(angleA, angleB):

    return min(math.fabs(angleA-angleB), 360-math.fabs(angleA-angleB))



if __name__ == "__main__":

    test('../data/LEBBData.csv')
    test('../data/LESOData.csv')
    test('../data/LEVTData.csv')