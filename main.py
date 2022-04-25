import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    # load training data
    df = pd.read_csv(args.training,names=["open", "high", "low", "close"])
    # df["close"].plot()
    # plt.savefig('training.png')
    # print(df["close"])

    X_train, y_train , X_test, sc = utils.ts_train_test(df,5,1)
    my_rnn_model, rnn_predictions = model.simple_rnn_model(X_train, y_train, X_test,sc)

    utils.actual_pred_plot(df,rnn_predictions)

    # load test data
    df_test = pd.read_csv(args.testing,names=["open", "high", "low", "close"])
    df_test = df_test['close']

    action = []
    flag = 0
    for i in range(1,len(df_test)):
        if (df_test.iloc[i] < df_test.iloc[i-1] and flag == 1):
            action.append(-1)
            flag = 0
        elif (df_test.iloc[i] == df_test.iloc[i-1]):
            action.append(0)
        elif (df_test.iloc[i] > df_test.iloc[i-1] and flag == 0):
            action.append(1)
            flag = 1
        else:
            action.append(0)

    output = pd.DataFrame(action)
    output.to_csv("output.csv",index=False,header=None)


