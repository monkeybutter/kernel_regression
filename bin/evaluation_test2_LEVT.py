__author__ = 'SmartWombat'

import pandas
import random
from regressor import *
import multiprocessing

print(multiprocessing.cpu_count())


datasets = ['../data/LEVTData.csv']

for dataset in datasets:

    print(dataset[-12:])

    df = pandas.read_csv(dataset, na_values=['NaN', ' NaN'])
    df = df[np.isfinite(df['MetarwindSpeed'])]
    df = df[np.isfinite(df['WindSpd'])]
    df = df[np.isfinite(df['WindDir'])]

    f = open('../data/results2_{}'.format(dataset[-12:]), 'w')
    f.write('me_no_regr,me_simpl_regr,me_dir_w_simpl_regr,width\n')

    for i in range(10):

        print(i)

        rows = random.sample(df.index, int(df.shape[0]*.75))
        train_df = df.ix[rows]
        test_df = df.drop(rows)

        for width2 in [.5, 1, 2, 4, 8]:
            f.write('{},'.format(rmse_no_regression(test_df, 'MetarwindSpeed', 'WindSpd')))
            f.write('{},'.format(rmse_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd')))
            f.write('{},{}\n'.format(rmse_direction_speed_weighted_simple_linear_regression(test_df, train_df, 'MetarwindSpeed', 'WindSpd', 'WindDir', 25, width2), width2))

    f.close()