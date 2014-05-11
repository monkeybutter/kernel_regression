__author__ = 'SmartWombat'

import pandas
import random
from regressor import *


datasets = ['../data/LEBBData.csv', '../data/LESOData.csv', '../data/LEVTData.csv']

for dataset in datasets:

    df = pandas.read_csv(dataset, na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    print(dataset)

    for i in range(5):

        rows = random.sample(df.index, int(df.shape[0]*.75))
        train_df = df.ix[rows]
        test_df = df.drop(rows)

        print('___________')
        print('ME no regression: {}'.format(me_no_regression(test_df, 'MetarwindSpeed', 'WindSpd')))
        print('ME simple regression: {}'.format(me_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd')))
        print('ME direction weighted simple regression: {}'.format(me_direction_weighted_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd', 'WindDir', 35)))
        print('ME direction speed weighted simple regression: {}'.format(me_direction_speed_weighted_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd', 'WindDir', 35, 5)))

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
