import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    # load training data
    df = pd.read_csv(args.training)
    df.columns =["open", "high", "low", "close"]
    #df["close"].plot()
    #plt.savefig('training.png')
    #print(df["close"])

