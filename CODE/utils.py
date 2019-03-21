import pandas as pd
import os
import numpy as np
from tensorflow.keras.utils import Sequence

def generate_windows(df, window_size, scaler=None):
    n_windows = df.values.shape[0]-window_size
    X = list()
    Y = list()
    for i in range(n_windows):
        window = df.values[i:i+window_size,:]
        if scaler is not None:
            window = scaler.transform( window )
        X.append(window[:,1:])
        Y.append(window[:, 0].reshape(window_size,1))
    return np.array(X), np.array(Y)

def load_dfs():
    folder = 'DATA'
    dfCsvs = os.listdir(folder)
    listDf = list()
    for file in dfCsvs:
        if file[0] is not '.':
            df = pd.read_csv('DATA/{0}'.format(file), index_col='Date')['Close'].sort_index(ascending=True).to_frame()
            df.columns = [file.split('.')[0]]
            listDf.append(df)

    df = listDf[0]
    for dfs in listDf[1:]:
        df = df.join(dfs)
    df = df.dropna()
    return df

def generate_windows_for_two_factor(df, window_size, scaler=None):
    n_windows = df.values.shape[0]-window_size
    X1 = list()
    X2 = list()
    Y = list()
    for i in range(n_windows):
        window = df.values[i:i+window_size,:]
        if scaler is not None:
            window = scaler.transform( window )
        X1.append(window[:,1:])
        X2.append(window[:-1, 0].reshape(window_size-1,1))
        Y.append(window[1:, 0].reshape(-1,1))
    return np.array(X1), np.array(X2), np.array(Y)

class TwoFactorGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x[0]) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x1 = self.x[0][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x2 = self.x[1][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [batch_x1, batch_x2], batch_y