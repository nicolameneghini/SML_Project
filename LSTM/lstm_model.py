from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas
import matplotlib.pyplot as plt


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back+1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back - 1, :])

    return np.array(dataX), np.array(dataY)


def series_split(data, size):

    size = int(size)
    train = data[0:size].dropna()
    test = data[size:].dropna()

    return train, test


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


def fit_lstm(train, units, batch_size, nb_epochs, lookback):
    X, y = create_dataset(train, lookback)
    X = np.reshape(X, (X.shape[0], lookback, X.shape[2]))

    model = Sequential()
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True,
                   input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=units))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(nb_epochs):
        model.fit(X, y, epochs=1, batch_size=batch_size,   verbose=0,
                  shuffle=False)
        model.reset_states()

    return model


def forecast(model, batch_size, row):
    X = row[0:-1]
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def plot_keras(train, test, X_test, predictions, lookback, page_name=None):
    plt.figure(figsize=(15, 7))
    titles = ['reality', 'predictions']
    plt.plot(train.index, train, color='b', linewidth=1.5)
    plt.plot(test.index[0:len(test)-lookback+1],
             X_test[0:, 0], label=titles[0], color='b', linewidth=1.5)
    plt.plot(test.index[0:len(test)-lookback+1],
             predictions, label=titles[1], color='orange', linewidth=1.5)
    plt.ylabel('Series')
    plt.legend(titles)
    plt.title('Keras prediction ' + page_name)
    plt.show()
