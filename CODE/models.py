import tensorflow.keras as kf

def create_2f_model(n_units, input_shape, file=None):
    window_size,input_dim = input_shape
    input_1 = kf.layers.Input(shape=(window_size, input_dim - 1))
    input_2 = kf.layers.Input(shape=(window_size - 1, 1))

    # lstm_1= kf.layers.LSTM(n_units, return_state=True)
    # output,  output_h, output_c = lstm_1(input_1)

    # lstm_1 = kf.layers.Bidirectional(kf.layers.CuDNNLSTM(n_units, return_state=True))
    lstm_1 = kf.layers.Bidirectional(kf.layers.LSTM(n_units, return_state=True, return_sequences=True))(input_1)
    _, output_h1, output_c1, output_h2, output_c2 = kf.layers.Bidirectional(kf.layers.LSTM(n_units, return_state=True))(lstm_1)
    output_h = kf.layers.Concatenate()([output_h1, output_h2])
    output_c = kf.layers.Concatenate()([output_c1, output_c2])

    dense_h = kf.layers.Dense(n_units,activation='tanh')(output_h)
    dense_C = kf.layers.Dense(n_units,activation='tanh')(output_c)

    encoder_states = [dense_h, dense_C]

    #lstm_2 = kf.layers.CuDNNLSTM(n_units, return_sequences=True)
    lstm_2 = kf.layers.LSTM(n_units, return_sequences=True)
    lstm_out = lstm_2(input_2, initial_state=encoder_states)

    #
    # dense_in = kf.layers.Dense(n_units // 2, activation='tanh')(lstm_out)
    # dense_out = kf.layers.Dense(1)(dense_in)
    dense_out = kf.layers.TimeDistributed(kf.layers.Dense(1))(lstm_out)
    model = kf.models.Model(inputs=[input_1, input_2], outputs=[dense_out])
    optimizer = kf.optimizers.RMSprop(lr=0.001, clipnorm=5)
    model.compile(optimizer=optimizer, loss='mse')

    if file == file:
        try:
            model.load_weights(file)
        except:
            pass
    return model


def create_2f_model_sum(n_units, input_shape, file=None):
    window_size,input_dim = input_shape
    input_1 = kf.layers.Input(shape=(window_size, input_dim - 1))
    input_2 = kf.layers.Input(shape=(window_size - 1, 1))

    lstm_1 = kf.layers.Bidirectional(kf.layers.LSTM(n_units, return_state=True, dropout=0.1, recurrent_dropout=0.1, unroll=True))
    _, output_h1, output_c1, output_h2, output_c2 = lstm_1(input_1)
    output_h = kf.layers.Concatenate()([output_h1, output_h2])
    output_c = kf.layers.Concatenate()([output_c1, output_c2])

    dense_h = kf.layers.Dense(n_units,activation='relu')(output_h)
    dense_h1 = kf.layers.Dense(n_units//2, activation='relu')(dense_h)
    dense_h2 = kf.layers.Dense(n_units, activation='relu')(dense_h1)

    dense_c = kf.layers.Dense(n_units,activation='relu')(output_c)
    dense_c1 = kf.layers.Dense(n_units // 2, activation='relu')(dense_c)
    dense_c2 = kf.layers.Dense(n_units, activation='relu')(dense_c1)

    encoder_states = [dense_h2, dense_c2]

    lstm_2 = kf.layers.LSTM(n_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
    lstm_out = lstm_2(input_2, initial_state=encoder_states)


    dense_out = kf.layers.TimeDistributed(kf.layers.Dense(1))(lstm_out)
    model = kf.models.Model(inputs=[input_1, input_2], outputs=[dense_out])
    optimizer = kf.optimizers.Adam(lr=0.001, clipnorm=5)
    model.compile(optimizer=optimizer, loss='mae')

    if file == file:
        try:
            model.load_weights(file)
        except:
            pass
    return model


def create_2f_model_c(n_units, input_shape, file=None):
    window_size,input_dim = input_shape
    input_1 = kf.layers.Input(shape=(window_size, input_dim - 1))
    input_2 = kf.layers.Input(shape=(window_size - 1, 1))

    lstm_cell = kf.layers.LSTM(n_units,
                               activation='relu',
                               return_state=True,
                               dropout=0.2,
                               recurrent_dropout=0.2,
                               unroll=True)
    lstm_1 = kf.layers.Bidirectional(lstm_cell)
    _, output_h1, output_c1, output_h2, output_c2 = lstm_1(input_1)
    output_h = kf.layers.Concatenate()([output_h1, output_h2])
    output_c = kf.layers.Concatenate()([output_c1, output_c2])

    dense_h = kf.layers.Dense(n_units,activation='tanh',
                              activity_regularizer=kf.regularizers.l2(l=0.001)
                              )(output_h)
    dense_c = kf.layers.Dense(n_units,activation='tanh',
                              activity_regularizer=kf.regularizers.l2(l=0.001)
                              )(output_c)

    encoder_states = [dense_h, dense_c]

    lstm_2 = kf.layers.LSTM(n_units, return_sequences=True)

    lstm_out = lstm_2(input_2, initial_state=encoder_states)

    dense_out = kf.layers.TimeDistributed(kf.layers.Dense(1,activity_regularizer=kf.regularizers.l2(l=0.001))
                                          )(lstm_out)
    model = kf.models.Model(inputs=[input_1, input_2], outputs=[dense_out])
    optimizer = kf.optimizers.RMSprop(lr=0.001, clipnorm=5)
    model.compile(optimizer=optimizer, loss='mse')

    if file == file:
        try:
            model.load_weights(file)
        except:
            pass
    return model