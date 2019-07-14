from keras.models import Model, load_model
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, Concatenate
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from wavenet_preprocessing import preprocessing
import datetime
import pandas as pd
import pathlib
import gc


def get_time_block_series(series_array, date_to_index, start_date, end_date):
    inds = date_to_index[start_date:end_date]
    return series_array[:, inds]


def transform_series_encode(series_array):
    series_array = np.log1p(np.nan_to_num(series_array))  # filling NaN with 0
    series_mean = series_array.mean(axis=1).reshape(-1, 1)
    series_std = series_array.std(axis=1).reshape(-1, 1)
    epsilon = 1e-6
    series_array = (series_array - series_mean)/(series_std+epsilon)
    series_array = series_array.reshape(
        (series_array.shape[0], series_array.shape[1], 1))
    return series_array, series_mean, series_std


def untransform_series_decode(series_array, encode_series_mean, encdoe_series_std):
    series_array = series_array.reshape(
        series_array.shape[0], series_array.shape[1])
    series_array = series_array*encdoe_series_std + encode_series_mean
    # unlog the data, clip the negative part if smaller than 0
    np.clip(np.power(10., series_array) - 1.0, 0.0, None)
    return series_array


def transform_series_decode(series_array, encode_series_mean, encode_series_std):
    series_array = np.log1p(np.nan_to_num(series_array))  # filling NaN with 0
    epsilon = 1e-6  # prevent numerical error in the case std = 0
    series_array = (series_array - encode_series_mean) / \
        (encode_series_std+epsilon)
    series_array = series_array.reshape(
        (series_array.shape[0], series_array.shape[1], 1))
    return series_array


def predict_sequences(input_sequences, batch_size):
    history_sequences = input_sequences.copy()
    print(history_sequences.shape)
    # initialize output (pred_steps time steps)
    pred_sequences = np.zeros((history_sequences.shape[0], pred_steps, 1))
    print(pred_sequences.shape)
    for i in range(pred_steps):
        # record next time step prediction (last time step of model output)
        last_step_pred = model.predict(history_sequences, batch_size)[:, -1, 0]
        print("last step prediction first 10 channels")
        print(last_step_pred[0:10])
        print(last_step_pred.shape)
        pred_sequences[:, i, 0] = last_step_pred

        # add the next time step prediction to the history sequence
        history_sequences = np.concatenate([history_sequences,
                                            last_step_pred.reshape(-1, 1, 1)], axis=1)

    return pred_sequences


def predict_and_plot(encoder_input_data, sample_ind, batch_size, enc_tail_len=50, decoder_target_data=1):
    encode_series = encoder_input_data[sample_ind:sample_ind + 1, :, :]
    pred_series = predict_sequences(encode_series, batch_size)

    encode_series = encode_series.reshape(-1, 1)
    pred_series = pred_series.reshape(-1, 1)

    if isinstance(decoder_target_data, np.ndarray):
        target_series = decoder_target_data[sample_ind, :, :1].reshape(-1, 1)
        encode_series_tail = np.concatenate(
            [encode_series[-enc_tail_len:], target_series[:1]])
    else:
        encode_series_tail = encode_series[-enc_tail_len:]

    x_encode = encode_series_tail.shape[0]

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, x_encode + 1), encode_series_tail)

    plt.plot(range(x_encode, x_encode + pred_steps),
             pred_series, color='teal', linestyle='--')

    plt.title(
        'Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)

    if isinstance(decoder_target_data, np.ndarray):
        plt.plot(range(x_encode, x_encode + pred_steps),
                 target_series, color='orange')
        plt.legend(['Encoding Series', 'Target Series', 'Predictions'])
    else:
        plt.legend(['Encoding Series', 'Predictions'])


def build_model(n_filters, filter_width):
    dilation_rates = [2**i for i in range(12)]

    # define an input history series and pass it through a stack of dilated causal convolutions
    history_seq = Input(shape=(None, 1))
    x = history_seq

    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
                   kernel_size=filter_width,
                   padding='causal',
                   dilation_rate=dilation_rate)(x)
    # for Dense Layer:
    # Input shape
    # nD tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).
    #
    # Output shape
    # nD tensor with shape: (batch_size, ..., units).
    # For instance, for a 2D input with shape  (batch_size, input_dim), the output would have shape (batch_size, units).

    x = Dense(128, activation='relu')(x)
    x = Dropout(.8)(x)
    x = Dense(64)(x)
    x = Dense(1)(x)

    # extract the last 14 time steps as the training target

    def slice(x, seq_length):
        return x[:, -seq_length:, :]

    pred_seq_train = Lambda(slice, arguments={'seq_length': 24})(x)

    model = Model(history_seq, pred_seq_train)

    return model
