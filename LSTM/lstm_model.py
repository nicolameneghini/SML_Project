from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM, Input, Dropout, TimeDistributed, RepeatVector
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas
import matplotlib.pyplot as plt


def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train_scaled = scaler.transform(train)
    # transform test
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def series_split(data, size):

    size = int(size)
    train = data[0:size]  # .dropna()
    test = data[size:]  # .dropna()

    return train, test


def create_dataset(train, last, look_back=1):
    dataX, dataY = [], []
    for i in range(len(train)-look_back):
        a = train[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(train[i + look_back, :])

    dataX.append(train[-1:, :])
    dataY.append(last)
    return np.array(dataX), np.array(dataY)


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


def fit_univariate_lstm(train, last, units, batch_size, nb_epochs, lookback):

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


def fit_multivariate_lstm(encoder_input_data, decoder_input_data, decoder_target_data,
                          units=50, epochs=100, batch_size=20):

    encoder_inputs = Input(shape=(None, 1))
    encoder = LSTM(units=units, dropout=0.2, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, 1))
    decoder_lstm = LSTM(units=units, dropout=0.2,
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
                        validation_split=0.2,
                        verbose=0, shuffle=False)

    return model, history


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def plot_keras(train, test, X_test, predictions, lookback, page_name=None):
    plt.figure(figsize=(15, 7))
    labels = ['reality', 'predictions']
    plt.plot(train.index, train, color='b', linewidth=1.5)
    plt.plot(test.index[0:len(test)-lookback+1],
             X_test[0:, 0], label=labels[0], color='b', linewidth=1.5)
    plt.plot(test.index[0:len(test)-lookback+1],
             predictions, label=labels[1], color='orange', linewidth=1.5)
    plt.ylabel('Series')
    plt.legend(labels)
    plt.title('Keras prediction: ' + page_name)
    plt.show()


#encoder_inputs = Input(shape=(None, 1))
#
#_, state_h, state_c = encoder(encoder_inputs)
#encoder_states = [state_h, state_c]
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
 #                                    initial_state=encoder_states)
