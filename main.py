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
    parser.add_argument('--model',
                        default='rnn',
                        help='choose model')                    
    args = parser.parse_args()

    # load training data
    df = pd.read_csv(args.training,names=["open", "high", "low", "close"])
    # df_close = df['close']
    # df_close.plot()
    # plt.savefig('figures/close.png')

    # prepare training and testing data
    X_train, y_train , X_test, sc = utils.ts_train_test(df,5,1)

    if (args.model == "rnn"):
        # rnn training
        my_model, rnn_predictions = model.simple_rnn_model(X_train, y_train, X_test,sc)
        utils.actual_pred_plot(df,rnn_predictions)
    elif(args.model == "lstm"):
        # LSTM training
        my_model, LSTM_predictions = model.LSTM_model(X_train, y_train, X_test, sc)
        utils.actual_pred_plot(df,LSTM_predictions)

    # load test data
    df_test = pd.read_csv(args.testing,names=["open", "high", "low", "close"])
    df_test = df_test['close']

    action = []
    stock = 0
    for i in range(0,len(df_test)-1):
        if (i >= 4):
            # make prediction for pass 5 days
            final_test = df_test.iloc[i-4:i+1]
            final_test = utils.iterator_test(final_test,sc)
            final_predictions = my_model.predict(final_test)
            final_predictions = sc.inverse_transform(final_predictions)
            tomorrow = float(final_predictions)
        else:
            tomorrow = df_test[:i+1].mean()
        today = df_test.iloc[i]
        # print('day: ',i+1)
        # print('today',today)
        # print('tomorrow',tomorrow)

        # sell
        if (today > tomorrow ):
            if stock == 0:
                action.append(-1)
                stock -= 1
            elif stock == 1:
                action.append(-1)
                stock -= 1
            elif stock == -1:
                action.append(0)
        # hold
        elif (today == tomorrow ):
            action.append(0)
        # buy
        else:
            if stock == 0:
                action.append(1)
                stock += 1
            elif stock == 1:
                action.append(0)
            elif stock == -1:
                action.append(1)
                stock += 1

    output = pd.DataFrame(action)
    output.to_csv(args.output,index=False,header=None)


