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
    #print('ME no regression: {}'.format(me_no_regression(df, 'MetarwindSpeed', 'WindSpd')))
    #print('RMSE no regression: {}'.format(rmse_no_regression(df, 'MetarwindSpeed', 'WindSpd')))
    #print('ME simple regression: {}'.format(me_simple_regression(df, 'MetarwindSpeed', 'WindSpd')))
    #print('RMSE simple regression: {}'.format(rmse_simple_regression(df, 'MetarwindSpeed', 'WindSpd')))

    params = weighted_double_linear_regression(df, 'MetarwindSpeed', 'WindSpd', 'SWRad', 8, 2)
    print(params)
    print('{}, {}, {}'.format(params[0, 0], params[0, 1], params[0, 2]))

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
    se = ((df[col1] - df[col2])**2).sum()

    return math.sqrt(se/df.shape[0])


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


def simple_linear_regression(df, y_name, x_name):

    df = df[[x_name, y_name]]
    data = np.matrix(df)

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x_name])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    b = y * x.T * np.linalg.inv(x*x.T)
    return b

def me_simple_linear_regression(df, y_name, x_name):

    params = simple_linear_regression(df, y_name, x_name)

    e = 0

    for index, row in df.iterrows():
        #print(index)
        e += math.fabs(row[y_name]-(row[x_name]*params[0, 0]+params[0, 1]))

    return e/df.shape[0]


def rmse_simple_linear_regression(df, y_name, x_name):

    params = simple_linear_regression(df, y_name, x_name)

    se = 0

    for index, row in df.iterrows():
        #print(index)
        se += math.fabs(row[y_name]-(row[x_name]*params[0, 0]+params[0, 1]))**2

    return math.sqrt(se/df.shape[0])


def weighted_simple_linear_regression(df, y_name, x_name, x_centre, width):

    df = df[[x_name, y_name]]
    data = np.matrix(df)

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x_name])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    w = np.matrix(np.zeros(shape=(x0.shape[0], x0.shape[0])))

    for i in range(x0.shape[0]):
        for j in range(x0.shape[0]):
            if i == j:
                if math.fabs(x0[i]-x_centre) < width:
                    w[i, j] = 1.0
                else:
                    w[i, j] = 0.0

    b = y * w * x.T * np.linalg.inv(x * w * x.T)
    return b


def double_linear_regression(df, y_name, x0_name, x1_name):

    df = df[[x0_name, x1_name, y_name]]
    data = np.matrix(df)

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x0_name])
    x1 = np.array(df[x1_name])
    x = np.matrix(np.vstack((x0, x1, np.ones(x0.shape[0]))))

    b = y * x.T * np.linalg.inv(x*x.T)
    return b


def weighted_double_linear_regression(df, y_name, x0_name, x1_name, x_centre, width):

    df = df[[x0_name, x1_name, y_name]]
    data = np.matrix(df)

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x0_name])
    x1 = np.array(df[x1_name])
    x = np.matrix(np.vstack((x0, x1, np.ones(x0.shape[0]))))

    w = np.matrix(np.zeros(shape=(x0.shape[0], x0.shape[0])))

    for i in range(x0.shape[0]):
        for j in range(x0.shape[0]):
            if i == j:
                if math.fabs(x0[i]-x_centre) < width:
                    w[i, j] = 1.0
                else:
                    w[i, j] = 0.0

    b = y * w * x.T * np.linalg.inv(x * w * x.T)
    return b


def direction_weighted_simple_linear_regression(df, y_name, x_name, wind_dir_name, wind_dir_centre, wind_dir_span):
    df = df[[x_name, wind_dir_name, y_name]]
    data = np.matrix(df)

    y = np.matrix(np.array(df[y_name]))
    x0 = np.array(df[x_name])
    x = np.matrix(np.vstack((x0, np.ones(x0.shape[0]))))

    w = np.matrix(np.zeros(shape=(x0.shape[0], x0.shape[0])))

    for i in range(x0.shape[0]):
        for j in range(x0.shape[0]):
            if i == j:
                distance = degrees_distance(wind_dir_centre, df[wind_dir_name][i])
                if distance < wind_dir_span:
                    # Kernel cuadratic
                    w[i, j] = 70.0/81.0 * (1-math.fabs(distance/wind_dir_span)**3)**3
                else:
                    w[i, j] = 0.0

    b = y * w * x.T * np.linalg.inv(x * w * x.T)
    return b


def degrees_distance(angleA, angleB):

    return min(math.fabs(angleA-angleB), 360-math.fabs(angleA-angleB))


if __name__ == "__main__":

    test('../data/LEBBData.csv')
    test('../data/LESOData.csv')
    test('../data/LEVTData.csv')