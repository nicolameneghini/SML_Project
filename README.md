# Web Traffic Time Serie Forecasting
This folder contains the final project for the **Statistical Machine Learning course**. 

It is based on the Kaggle competition [Web Traffic Time Serie Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting), that requires to forecast the future values of multiple time series. 

## Data
The data consists of approximately 145,000 Wikipedia. Each of these time series represent a number of daily views of a different Wikipedia article, starting from **July, 1st, 2015** up until **December 31st, 2016**.

Each row corresponds to a particular article and each column correspond to a particular date. Some entries are missing data. The page names contain: 
* the Wikipedia project (e.g. en.wikipedia.org)
* type of access (e.g. desktop) 
* type of agent (e.g. spider) 

In other words, each article name has the following format: 'name_project_access_agent' (e.g. 'AKB48_zh.wikipedia.org_all-access_spider').

An example of 6 random time series can be seen in the graph below. 
![](series.png)


## Approches 
We compared three different approches to solve this problem: 
* [ARIMA model](https://github.com/alessiapaoletti/SML_Project/tree/master/ARIMA)
* [LSTM](https://github.com/alessiapaoletti/SML_Project/tree/master/LSTM)
* [Wavenet](https://github.com/alessiapaoletti/SML_Project/tree/master/Wavenet)

All the code, implemented in Python, can be found in the folders. 

## Contributors 
* [Gianluca Iacubino](https://github.com/IacubinoGianluca)
* [Nicola Meneghini](https://github.com/nicolameneghini)
* [Alessia Paoletti](https://github.com/alessiapaoletti) 

