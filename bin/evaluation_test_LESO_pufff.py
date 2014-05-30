__author__ = 'SmartWombat'

import pandas
import random
from regressor import *


datasets = ['../data/LESOData.csv']

for dataset in datasets:

    print(dataset[-12:])

    df = pandas.read_csv(dataset, na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    f = open('../data/results_puff_{}'.format(dataset[-12:]), 'w')
    f.write('rmse_no_regr,rmse_simpl_regr,rmse_w_simpl_regr(10),rmse_dir_w_simpl_regr(45),rmse_dir_w_simpl_regr(45,10)\n')

    for i in range(1):

        print(i)

        rows = random.sample(df.index, int(df.shape[0]*.75))
        train_df = df.ix[rows]
        test_df = df.drop(rows)

        f.write('{},'.format(rmse_no_regression(test_df, 'MetarwindSpeed', 'WindSpd')))
        print(0)
        f.write('{},'.format(rmse_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd')))
        print(1)
        f.write('{},'.format(rmse_weigthed_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd', 10)))
        print(2)
        f.write('{},'.format(rmse_direction_weighted_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd', 'WindDir', 45)))
        print(3)
        f.write('{}\n'.format(rmse_direction_speed_weighted_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd', 'WindDir', 45, 10)))
        print(4)

    f.close()