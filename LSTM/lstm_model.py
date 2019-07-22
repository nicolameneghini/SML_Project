from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM, Input, Dropout, TimeDistributed, RepeatVector
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas
import matplotlib.pyplot as plt

# *************** Simple LSTM ***************

# Error metrics


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


# Scales the values of the time series between 0 and 1 for fitting
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0,  1))
    scaler = scaler.fit(train)
    # transform train
    train_scaled = scaler.transform(train)
    # transform test
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# Rescale the time series back
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# Divides the series into train and test
def series_split(data, size):

    size = int(size)
    train = data[0:size]  # .dropna()
    test = data[size:]  # .dropna()

    return train, test


# Creates the arrays to be fed to the lstm layer: lstm.fit(X,y)
# X: 3D array containing all the observation grouped into arrays of lag-dimension.
# y: 2D array of the same observation but shifted of a time-laps equal to the lag
def create_dataset(train, last, look_back=1):
    dataX, dataY = [], []
    for i in range(len(train)-look_back):
        a = train[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(train[i + look_back, :])

    dataX.append(train[-1:, :])
    dataY.append(last)
    return np.array(dataX), np.array(dataY)


def fit_simple_lstm(train, last, units, batch_size, nb_epochs, lookback):

    X, y = create_dataset(train, last, lookback)
    X = np.reshape(X, (X.shape[0], lookback, X.shape[2]))

    model = Sequential()
    model.add(LSTM(units=units,
                   input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X, y, epochs=nb_epochs, validation_split=0.20, batch_size=batch_size, verbose=0,
                        shuffle=False)

    return model, history


def plot_keras(train, test, X_test, predictions, lookback, page_name=None):
    plt.figure(figsize=(15, 7))
    labels = ['train', 'reality', 'prediction']
    plt.plot(train.index, train, color='b', linewidth=1.5)
    plt.plot(test.index[0:len(test)-lookback+1],
             X_test[0:, 0],  color='orange', linewidth=1.5)
    plt.plot(test.index[0:len(test)-lookback+1],
             predictions,  color='teal', linewidth=1.5)
    plt.ylabel('Series')
    plt.title('Keras prediction: ' + page_name)
    plt.legend(labels)
    plt.show()


# *************** Seq2Seq LSTM ***************

# Encoder_input: 3D train array for the encoder. It spans from Date:0 to Date:N
# Decocoder_input: 3D array for the decoder. It spans from Date:N-1 to Date:N-1+timestep
# Decoder_target: 3D array. It spans form Date:N to Date:N+ timestep
def get_encod_decod(page, train_start, train_end, train_pred_start, train_pred_end):

    N_of_pages = max(page, key=int)+1

    encoder_input_data = np.zeros((N_of_pages, train_end-train_start, 1))

    decoder_target_data = np.zeros(
        (N_of_pages, train_pred_end-train_pred_start, 1))

    decoder_input_data = np.zeros(
        (N_of_pages, train_pred_end-train_pred_start, 1))

    for i in page.keys():
        encoder_input_data[i] = page[i].values[train_start:train_end]
        decoder_target_data[i] = page[i].values[train_pred_start: train_pred_end]

    scaler = {}

    for i in page.keys():
        scaler[i], encoder_input_data[i], decoder_target_data[i] = scale(
            encoder_input_data[i], decoder_target_data[i])

    decoder_input_data[:, 1:, 0] = decoder_target_data[:, :-1, 0]
    decoder_input_data[:, 0, 0] = encoder_input_data[:, -1, 0]

    return scaler, encoder_input_data, decoder_input_data, decoder_target_data


# Adapted from the Keras Documentation. See https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

def fit_se2seq(encoder_input_data, decoder_input_data, decoder_target_data,
               latent_dim=50, epochs=100, batch_size=20):

    encoder_inputs = Input(shape=(None, 1))
    encoder = LSTM(units=latent_dim, dropout=0.2, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, 1))
    decoder_lstm = LSTM(units=latent_dim, dropout=0.2,
                        return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)

    decoder_dense = Dense(1)  # 1 continuous output at each timestep
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(loss='mean_absolute_error', optimizer='adam')
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2, shuffle=False)

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)

    return model, history, encoder_model, decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model, timestep):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, 1))

    target_seq[0, 0, 0] = input_seq[0, -1, 0]

    decoded_seq = np.zeros((1, timestep, 1))

    for i in range(timestep):
        output, h, c = decoder_model.predict([target_seq] + states_value)
        decoded_seq[0, i, 0] = output[0, 0, 0]
        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, 0] = output[0, 0, 0]
        states_value = [h, c]

    return decoded_seq
