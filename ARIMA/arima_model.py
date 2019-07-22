import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def series_split(data, size):

    size = int(size)
    train = data['count'][0:size].dropna()
    test = data['count'][size:len(data)].dropna()

    return train, test


def difference(dataset, interval=30):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


def inverse_difference(history, yhat, interval=30):
    return yhat + history[-interval]


def rolling_forecast(train, test, model_name, p, d, q):
    history = [x for x in train]
    predictions = list()
    output = []
    for t in range(len(test)):
        model = model_name(history, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        history = history[1:]

    return predictions


def rolling_forecast_var(train, test, model_name, p, q):
    history = train.copy()
    predictions = pd.DataFrame(columns=train.columns)
    for t in range(len(test)):
        model = model_name(history, order=(p, q))
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()
        predictions = predictions.append(yhat)
        obs = test.iloc[t, :]
        history = history.append(obs)
        history = history[1:]

    return predictions


def plot_arima(train, test, predictions, title):
    plt.figure(figsize=(17, 9))
    titles = ['prediction', 'reality']
    plt.plot(test.index, predictions, color='orange', linewidth=1.5)
    plt.plot(test.index, test, color='b', linewidth=1.5)
    plt.plot(train.index, train, color='b', linewidth=1.5)
    plt.legend(titles)
    plt.ylabel('Series')
    plt.title('Performance of predictions - Benchmark Predictions vs Reality ' + title)
    plt.show()
    
    
def smape(A, F):
    A, F = np.array(A), np.array(F)
    return np.mean(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))*100

def perform_var_pred(dataframe):
    train_size = int(dataframe.shape[0]*0.66)
    train = dataframe.iloc[0:train_size,:]
    test = dataframe.iloc[train_size:,:]
    predictions = pd.DataFrame(columns = dataframe.columns)
    for t in range(test.shape[0]):
        model = VAR(train)
        model_fit = model.fit(maxlags = 10, ic = 'aic')
        lag_order = model_fit.k_ar
        yhat = model_fit.forecast(train.values[-lag_order:], steps = 1)
        predictions = predictions.append(pd.DataFrame(list(yhat), columns = dataframe.columns))
        train = train.append(test.iloc[t,:])
        
    #error = mean_squared_error(test.values, predictions.values, multioutput='raw_values')
    predictions.index = test.index
    return predictions 
