from CODE.models import create_2f_model
from CODE.utils import *


window_size = 5
n_units = 200
corr_threshold = 0.2

df = load_dfs()
diff_df = df.pct_change(1).iloc[1:,:]

correlations = diff_df.corr().values[0]
filteres_ids = np.argwhere(abs(correlations)> corr_threshold).flatten()
diff_df = diff_df.iloc[:,filteres_ids]
print(diff_df.head())

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

Y_pred, Y_actual=list(), list()


for id in range(start,start+100):
    if id == start:
        x2 = X2[id].reshape(-1, window_size - 1, 1)
    x1 = X1[id].reshape(-1,window_size,input_shape[1]-1)
    Y_pred.append(model.predict([x1, x2]).flatten())
    Y_actual.append(Y[id].flatten())
    x2[0, :-1, 0] = x2[0, 1:, 0]
    x2[0,-1,0] = Y_pred[-1]

test_df['Predicted']=np.nan
test_df['Predicted'].iloc[start:start+100]=Y_pred
test_df.plot()