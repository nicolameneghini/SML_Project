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

    return predictions


def plot_arima(train, test, predictions):
    plt.figure(figsize=(17, 9))
    titles = ['prediction', 'reality']
    plt.plot(test.index, predictions, color='orange', linewidth=1.5)
    plt.plot(test.index, test, color='b', linewidth=1.5)
    plt.plot(train.index, train, color='b', linewidth=1.5)
    plt.legend(titles)
    plt.ylabel('Series')
    plt.title('Performance of predictions - Benchmark Predictions vs Reality')
    plt.show()
