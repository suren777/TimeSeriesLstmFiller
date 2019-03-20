from tensorflow import keras as k
from utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

window_size = 20
n_units = 200


df = load_dfs()
diff_df = df.pct_change(1).iloc[1:,:]

layers = [k.layers.Bidirectional(k.layers.LSTM(n_units, input_shape=(window_size,3), return_sequences=True)),
         k.layers.TimeDistributed(k.layers.Dense(1))
          ]

model = k.models.Sequential(layers)
model.compile(optimizer='rmsprop', loss='mae')

scaler = MinMaxScaler()
scaler.fit(df.values)

X, Y = generate_windows(df, window_size,scaler)

model.fit(x=X, y=Y, batch_size=512, epochs=100, validation_split=0.05, verbose=2)

id = 1000

Y_pred = model.predict(X[id].reshape(1,window_size,3))

plt.plot(Y_pred.flatten(), label='pred')
plt.plot(Y[id].flatten(), label='act')
plt.legend()
plt.show()