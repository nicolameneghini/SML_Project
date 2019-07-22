import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt 
import pymc3 as pm  
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt

def auto_ar(series, max_order):
    best_score = float('inf')
    best_order = 0
    train_size = int(0.66*series.shape[0])
    train = series[:train_size]
    test = series[train_size:]
    for i in range(0, max_order):
        try:
            predictions = rolling_forecast(train, test, ARIMA, i, 0, 0)
            predictions = pd.DataFrame(predictions)
            error = smape(test, predictions[0])
            if error < best_score:
                best_score = error
                best_order = i
        except:
            continue
    print('the best ar order is ' + str(best_order) + 'with a SMAPE of {:10.4f}'.format(best_score))
    
   
 
def my_trace_plot(trace_df):
    for i in range(len(trace_df.columns)):
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        ax[0].plot(trace_df.iloc[:,i])
        ax[0].set_title(trace_df.columns[i])
        ax[1] = trace_df.iloc[:,i].plot.kde()
        ax[1].set_title('kde')
        plt.show()
        
def get_coef_from_trace(trace):
    trace_df = pm.trace_to_dataframe(trace)
    params = trace_df.mean()
    return params


def predict_with_no_updates(test_series, ar_order, model_fit):

    pre_sample = test_series
    predictions = list()
    for i in range(test_series.shape[0]-ar_order):
        values = pre_sample[:ar_order]
        values = values[::-1]
        yhat = np.dot(model_fit.arparams, values)
        predictions.append(yhat)
        pre_sample = pre_sample[1:]

    predictions = pd.DataFrame(predictions)
    predictions.set_index(test[ar_order:].index, inplace=True, drop=True)
    return predictions


def get_residuals(observed_series, predicted_series):
    # input are series of observed and predicted with the same index.
    # return the series of residuals
    observed = test[predicted_series.index].values
    predicted = predicted_series.values
    residuals = [observed[i]-predicted[i] for i in range(len(predictions))]
    residuals = pd.DataFrame(residuals)[0]
    residuals.index = predictions.index
    return residuals




def standardize_pd_series(series):
    scaler = StandardScaler()
    values = series.values
    values = values.reshape((len(values), 1))
    scaler = scaler.fit(values)
    standardized = scaler.transform(values)
    standardized = pd.DataFrame(standardized, index=series.index)[0]
    return standardized

def predict_with_no_updates_mcmc(test_series, ar_order, trace):

    pre_sample = test_series
    predictions = list()
    for i in range(test_series.shape[0]-ar_order):
        values = pre_sample[:ar_order]
        values = values[::-1]
        yhat = np.dot(get_coef_from_trace(trace), values)
        predictions.append(yhat)
        pre_sample = pre_sample[1:]

    predictions = pd.DataFrame(predictions)
    predictions.set_index(test_series[ar_order:].index, inplace=True, drop=True)
    return predictions


def ar_model_pred_advi_dynamic(X, ar_order):
    # prepare training dataset
    train_size = int(X.shape[0] * 0.66)
    train, test = X.iloc[0:train_size], X.iloc[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(test.shape[0]):
        tau = 0.001

        model = pm.Model()
        with model:
    
            beta = pm.Uniform('beta',lower = -1, upper = 1,  shape = ar_order)
            y_obs = pm.AR('y_obs', rho = beta, tau=tau, observed=history)
            #trace = pm.sample(2000, tune=1000)
            step = step = pm.ADVI()
    
            n_draws, n_chains = 3000, 3
            n_sim = n_draws*n_chains
            
            advi_fit = pm.fit(method=pm.ADVI(), n=30000)
    
            # Consider 3000 draws and 2 chains.
            advi_trace = advi_fit.sample(10000)
            
        values =history[len(history)-ar_order:]
        values = values[::-1]
        yhat = np.dot(get_coef_from_trace(advi_trace), values)
        predictions.append(yhat)
        history.append(test[t])
        history = history[1:]
    # calculate out of sample error
    #error = mean_squared_error(test, predictions)
    predictions = pd.DataFrame(predictions)
    predictions.set_index(X[train_size:X.shape[0]].index, inplace=True, drop=True)
    return predictions[0]


def plot_actual_vs_pred(test, predictions, title):
    df = pd.DataFrame()
    test = test[predictions.index]
    df['Actual'] = test
    df['Predicted'] = predictions
    df.plot(figsize=[14,5])
    plt.title('Actual vs Predicted- ' + title)
    plt.show()
    
    
def create_lags(series, n_lags):
    y = series.reset_index().iloc[:,1]
    data = pd.DataFrame()
    for i in range(0,n_lags+1):
        x = y[n_lags-i:len(series)-i]
        x = x.reset_index().iloc[:,1]
        data = pd.concat([data, x], axis = 1)
    data.columns = ['lag_{}'.format(i) for i in range(0,len(data.columns))]
    data.index = series[n_lags:].index
    return data

def plot_elbo(advi_fit):
    advi_elbo = pd.DataFrame(
    {'log-ELBO': -np.log(advi_fit.hist),
     'n': np.arange(advi_fit.hist.shape[0])})
    plt.figure(figsize=(15, 6))
    plt.plot(advi_elbo['n'].values, advi_elbo['log-ELBO'].values)
    plt.xlabel('n')
    plt.ylabel('log-ELBO')
    plt.show()