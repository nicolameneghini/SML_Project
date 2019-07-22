
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import re

import seaborn as sns
sns.set()

def plot_random_series(df, n_series):
    sample = df.sample(n_series, random_state=8)
    page_labels = sample['Page'].tolist()
    series_samples = sample.loc[:,data_start_date:data_end_date]

    plt.figure(figsize=(10,6))

    for i in range(series_samples.shape[0]):
        np.log1p(pd.Series(series_samples.iloc[i]).astype(np.float64)).plot(linewidth=1.5)

    plt.title('Randomly Selected Wikipedia Page Daily Views Over Time (Log(views) + 1)')
    plt.legend(page_labels)


def get_time_block_series(series_array, date_to_index, start_date, end_date):
    inds = date_to_index[start_date:end_date]
    return series_array[:,inds]

def transform_series_encode(series_array):
    #series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_array = np.nan_to_num(series_array) # filling NaN with 0
    series_array = np.log1p(series_array)

    series_mean = series_array.mean(axis=1).reshape(-1,1)
    #series_array = series_array - series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    return series_array, series_mean

def transform_series_decode(series_array, encode_series_mean):
    #series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0

    series_array = np.nan_to_num(series_array) # filling NaN with 0
    series_array = np.log1p(series_array)

    #series_array = series_array - encode_series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))

    return series_array

def predict_sequence(input_sequence):
    history_sequence = input_sequence.copy()
    pred_sequence = np.zeros((1,pred_steps,1)) # initialize output (pred_steps time steps)

    for i in range(pred_steps):

        # record next time step prediction (last time step of model output)
        last_step_pred = model.predict(history_sequence)[0,-1,0]
        pred_sequence[0,i,0] = last_step_pred

        # add the next time step prediction to the history sequence
        history_sequence = np.concatenate([history_sequence,last_step_pred.reshape(-1,1,1)], axis=1)

    return pred_sequence

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def predict_and_plot(encoder_input_data, decoder_target_data, sample_ind, enc_tail_len=50):

    encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:]
    pred_series = predict_sequence(encode_series)

    encode_series = encode_series.reshape(-1,1)
    pred_series = pred_series.reshape(-1,1)
    target_series = decoder_target_data[sample_ind,:,:1].reshape(-1,1)

    #print('Sample ind: ', sample_ind,'Test Mean Absolute Error :', mean_absolute_error(target_series,pred_series), 'Smape: ',smape(target_series, pred_series))
    print('Sample ind: ', sample_ind,'Smape: ',smape(target_series, pred_series))

    encode_series = np.expm1(encode_series)
    pred_series = np.expm1(pred_series)
    target_series = np.expm1(target_series)
    encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])
    x_encode = encode_series_tail.shape[0]

    plt.figure(figsize=(10,6))
    plt.plot(range(1,x_encode+1),encode_series_tail, color='b')
    plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='orange')
    plt.plot(range(x_encode,x_encode+pred_steps),target_series, color='b')
    plt.title("Page: %s " %df['Page'][sample_ind])
    plt.legend(['Real Serie','Predictions'])


def predict_and_plot_only(encoder_input_data, decoder_target_data, sample_ind, enc_tail_len=50):
    encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:]
    pred_series = predict_sequence(encode_series)

    encode_series = encode_series.reshape(-1,1)
    pred_series = pred_series.reshape(-1,1)
    target_series = decoder_target_data[sample_ind,:,:1].reshape(-1,1)
    encode_series = np.expm1(encode_series)
    pred_series = np.expm1(pred_series)
    target_series = np.expm1(target_series)
    encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])
    x_encode = encode_series_tail.shape[0]

    plt.figure(figsize=(10,6))
    plt.plot(range(1,x_encode+1),encode_series_tail, color='b')
    plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='orange')
    plt.plot(range(x_encode,x_encode+pred_steps),target_series, color='b')
    plt.title("Page: %s " %df['Page'][sample_ind])
    plt.legend(['Real Serie','Predictions'])

def smape_tot(encoder_input_data, decoder_target_data, n):
    smape_tot = {}
    for i in range(n):
        encode_series = encoder_input_data[i:i+1,:,:]
        pred_series = predict_sequence(encode_series)
        target_series = decoder_target_data[i,:,:1].reshape(-1,1)
        #print("working on..... ",i)
        smape_tot[i] = smape(target_series, pred_series)
    min_smape = np.array(list(smape_tot.values())).min()
    mean_smape = np.array(list(smape_tot.values())).mean()
    max_smape = np.array(list(smape_tot.values())).max()
    return smape_tot, min_smape, mean_smape, max_smape

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error Loss')
    plt.title('Loss Over Time')
    plt.legend(['Train','Valid'])


from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from keras.optimizers import Adam
def fit_wavenet(n_filters, filter_width, dilation_rates, batch_size,epochs):
    # define an input history series and pass it through a stack of dilated causal convolution blocks.
    history_seq = Input(shape=(None, 1))
    x = history_seq

    skips = []
    for dilation_rate in dilation_rates:

        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same', activation='relu')(x)

        # filter convolution
        x_f = Conv1D(filters=n_filters,
                     kernel_size=filter_width,
                     padding='causal',
                     dilation_rate=dilation_rate)(x)

        # gating convolution
        x_g = Conv1D(filters=n_filters,
                     kernel_size=filter_width,
                     padding='causal',
                     dilation_rate=dilation_rate)(x)

        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])

        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same', activation='relu')(z)

        # residual connection
        x = Add()([x, z])

        # collect skip connections
        skips.append(z)

    # add all skip connection outputs
    out = Activation('relu')(Add()(skips))

    # final time-distributed dense layers
    out = Conv1D(128, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(.2)(out)
    out = Conv1D(1, 1, padding='same')(out)

    # extract the last 60 time steps as the training target
    def slice(x, seq_length):
        return x[:,-seq_length:,:]

    pred_seq_train = Lambda(slice, arguments={'seq_length':60})(out)

    model = Model(history_seq, pred_seq_train)
    model.compile(Adam(), loss='mean_absolute_error')
    model.summary()
    history = model.fit(encoder_input_data, decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2)
    return history, model
