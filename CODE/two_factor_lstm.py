from CODE.models import create_2f_model
from CODE.utils import *


window_size = 5
n_units = 512
corr_threshold = 0.2

df = load_dfs()
diff_df = df.pct_change(1).iloc[1:,:]

correlations = diff_df.corr().values[0]
filteres_ids = np.argwhere(abs(correlations)> corr_threshold).flatten()
diff_df = diff_df.iloc[:,filteres_ids]
print(diff_df.head())

X1, X2, Y = generate_windows_for_two_factor(diff_df, window_size)
batch_size = 512
generator = TwoFactorGenerator([X1, X2], Y, batch_size=batch_size)

file = 'FILES/Models/lstm_2f_model.hdf5'
rf_num = diff_df.values.shape[1]
input_shape = (window_size,rf_num)
model = create_2f_model(n_units, input_shape, file)

epochs = 100

# for i in range(epochs):
#     hist = model.fit(x=[X1, X2],
#               y=Y,
#               batch_size=512,
#               verbose=2,
#               epochs=1,
#               validation_split=0.05,
#               callbacks=[])
#     if i > 0:
#         if hist.history['val_loss'][-1] !=  hist.history['val_loss'][-1]:
#             print('NaN loss exiting')
#             break
#         if hist.history['val_loss'][-1]<val_loss:
#             val_loss = hist.history['val_loss'][-1]
#             print("New val_loss: {}\t epoch: {}".format(val_loss,i))
#             model.save_weights(file)
#     else:
#         val_loss =  hist.history['val_loss'][-1]

model.fit_generator(generator, steps_per_epoch=1, epochs=epochs)



start = 100
window = 100
test_df = df.iloc[0:300,0].to_frame()
test_df.columns=['Actual']
test_df['Interp'] = test_df.iloc[:,0]
test_df['Interp'].iloc[start:start+window]=np.nan
test_df = test_df.interpolate()
start_value = test_df.iloc[start-1,0]
end_value = test_df.iloc[start+window,0]


Y_pred, Y_actual=list(), list()
for id in range(start, start+window):
    if id == start:
        x2 = X2[id].reshape(-1, window_size - 1, 1)
    x1 = X1[id].reshape(-1,window_size,input_shape[1]-1)
    Y_pred.append(model.predict([x1, x2]).flatten())
    Y_actual.append(Y[id].flatten())
    x2[0, :-1, 0] = x2[0, 1:, 0]
    x2[0,-1,0] = Y_pred[-1]

test_df['Predicted']=np.nan
fixed_values = np.array(Y_pred)+1
for i, value in enumerate(fixed_values):
    if i == 0:
        fixed_values[0]=value*start_value
    else:
        fixed_values[i]=fixed_values[i-1]*value
multiplier = end_value/fixed_values[-1]
singlemultipliers = np.array([multiplier**(x/(window+1)) for x in range(window+2)])
#singlemultipliers = np.array([multiplier**(0) for x in range(window+2)])
test_df['Predicted'].iloc[start:start+100]=fixed_values[:, 0]*singlemultipliers[:-2, 0]
test_df.plot()

test_df['Predicted']=np.nan
test_df['Predicted'].iloc[start:start+100]=Y_pred
test_df.plot()

compare_df = test_df.dropna().copy()
compare_df['LinInterpDiffs']=abs(compare_df['Actual']-compare_df['Interp'])/compare_df['Actual']
compare_df['LstmPredDiffs']=abs(compare_df['Actual']-compare_df['Predicted'])/compare_df['Actual']
print(compare_df.head())

compare_df.loc[:,['LinInterpDiffs', 'LstmPredDiffs'] ].plot()

print("Lstm Average Error: {}".format(compare_df['LstmPredDiffs'].mean()))
print("Interpolation Average Error: {}".format(compare_df['LinInterpDiffs'].mean()))

from scipy import stats

resLstm = stats.ks_2samp(compare_df['Actual'].values, compare_df['Predicted'].values)
print("Lstm KS result: \n \t - Statistic: {0} \n \t - pvalue: {1}".format(resLstm.statistic, resLstm.pvalue))
resInterp = stats.ks_2samp(compare_df['Actual'].values, compare_df['Interp'].values)
print("Interp KS result: \n \t - Statistic: {0} \n \t - pvalue: {1}".format(resInterp.statistic, resInterp.pvalue))