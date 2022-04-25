import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt

def ts_train_test(all_data,time_steps,for_periods):
    '''
    input:
      data: dataframe with dates and price data
    output:
      X_train, y_train: data from 2013/1/1-2018/12/31
      X_test:  data from 2019 -
      sc:      insantiated MinMaxScaler object fit to the training data
    '''
    # create training and test set
    ts_train, ts_test = train_test_split(all_data,test_size=0.2, shuffle=False)
    ts_train = ts_train.iloc[:,3:4].values
    ts_test  = ts_test.iloc[:,3:4].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # scale the data
    sc = MinMaxScaler(feature_range=(0,1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-1):
        X_train.append(ts_train_scaled[i-time_steps:i,0])
        y_train.append(ts_train_scaled[i:i+for_periods,0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    inputs = pd.DataFrame(all_data['close']).values
    inputs = inputs[len(inputs)-len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1,1)
    inputs  = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(time_steps,ts_test_len+time_steps-for_periods):
        X_test.append(inputs[i-time_steps:i,0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return X_train, y_train , X_test, sc

def actual_pred_plot(all_data,preds):
    actual_pred = pd.DataFrame(columns = ['close', 'prediction'])
    ts_train, ts_test = train_test_split(all_data,test_size=0.2, shuffle=False)
    actual_pred['close'] = ts_test['close'][0:len(preds)]
    actual_pred['prediction'] = preds[:,0]

    actual_pred.plot()
    plt.savefig('figures/prediction.png')