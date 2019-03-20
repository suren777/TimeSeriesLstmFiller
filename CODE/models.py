import tensorflow.keras as kf

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