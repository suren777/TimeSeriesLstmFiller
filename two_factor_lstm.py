from tensorflow import keras as kf
from utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_2f_model(n_units, input_shape, file=None):
    window_size,input_dim = input_shape
    input_1 = kf.layers.Input(shape=(window_size,input_dim-1))
    input_2 = kf.layers.Input(shape=(window_size - 1, 1))

    lstm_1= kf.layers.LSTM(n_units, return_state=True)
    output,  output_h, output_c = lstm_1(input_1)
    encoder_states = [output_h, output_c]

    lstm_2 = kf.layers.LSTM(n_units)
    lstm_out = lstm_2(input_2, initial_state=encoder_states)

    dense_in = kf.layers.Dense(n_units//2,activation='tanh')(lstm_out)
    dense_out = kf.layers.Dense(1, activation='linear', use_bias=False)(dense_in)

    model = kf.models.Model(inputs=[input_1, input_2], outputs=[dense_out])
    model.compile(optimizer='rmsprop', loss='mse')

    if file == file:
        try:
            model.load_weights(file)
        except:
            pass
    return model

window_size = 5
n_units = 200

df = load_dfs()

# scaler = MinMaxScaler()
diff_df = df.pct_change(1).iloc[1:,:]
# scaler.fit(df.values)
#X1, X2, Y = generate_windows_for_two_factor(df, window_size,scaler=scaler)
X1, X2, Y = generate_windows_for_two_factor(diff_df, window_size)

file = 'FILES/Models/lstm_2f_model.hdf5'
input_shape = (window_size,df.values.shape[1])
model = create_2f_model(n_units, input_shape, file)

epochs = 0
for i in range(epochs):
    hist = model.fit(x=[X1, X2],
              y=Y,
              batch_size=512,
              verbose=2,
              epochs=1,
              validation_split=0.05,
              callbacks=[])
    if i > 0:
        if hist.history['val_loss'][-1]<val_loss:
            val_loss = hist.history['val_loss'][-1]
            print("New val_loss: {}\t epoch: {}".format(val_loss,i))
            model.save_weights(file)
    else:
        val_loss =  hist.history['val_loss'][-1]

start = 100
test_df = df.iloc[0:300,0].to_frame()
test_df['GAP'] = test_df.iloc[:,0]
test_df['GAP'].iloc[start:start+100]=np.nan
test_df = test_df.interpolate()
test_df.plot()

Y_pred, Y_actual=list(), list()


for id in range(start,start+100):
    if id == 0:
        x2 = X2[id].reshape(-1, window_size - 1, 1)
    x1 = X1[id].reshape(-1,window_size,input_shape[1]-1)
    Y_pred.append(model.predict([x1, x2]).flatten())
    Y_actual.append(Y[id].flatten())
    x2[0, :-1, 0] = x2[0, 1:, 0]
    x2[0,-1,0] = Y_pred[-1]

plt.plot(np.array(Y_pred), label='pred')
plt.plot(np.array(Y_actual), label='act')
plt.legend()
plt.show()