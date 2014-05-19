__author__ = 'SmartWombat'

import pandas
import random
from regressor import *


datasets = ['../data/LEBBData.csv', '../data/LESOData.csv', '../data/LEVTData.csv']

for dataset in datasets:

    print(dataset[-12:])

    df = pandas.read_csv(dataset, na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    f = open('../data/results1_{}'.format(dataset[-12:]), 'w')
    f.write('me_no_regr,me_simpl_regr,me_dir_w_simpl_regr,width\n')

    for i in range(10):

        print(i)

        rows = random.sample(df.index, int(df.shape[0]*.75))
        train_df = df.ix[rows]
        test_df = df.drop(rows)

        for width in [5, 10, 20, 30, 40, 60, 90, 120, 180]:
            f.write('{},'.format(me_no_regression(test_df, 'MetarwindSpeed', 'WindSpd')))
            f.write('{},'.format(me_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd')))
            f.write('{},{}\n'.format(me_direction_weighted_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd', 'WindDir', width), width))

    f.close()


    """
for i in range(20):

    df = pandas.read_csv("./web/static/data/EGLL.csv")
    df = df[['time', 'windDir', 'windSpeed', 'temp', 'dewPoint', 'pressure']]

    rows = random.sample(df.index, 23000)
    train_df = df.ix[rows]
    test_df = df.drop(rows)

    3#train_df2 = train_df[['windDir', 'windSpeed', 'temp', 'dewPoint', 'pressure']]
    test_df2 = test_df[['windDir', 'windSpeed', 'temp', 'dewPoint', 'pressure']]
    """
