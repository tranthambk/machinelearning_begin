import sys
# scipy
import scipy
# numpy
import numpy
# matplotlib
import matplotlib
# pandas
import pandas
# scikit-learn
import sklearn
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import csv
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMAResults
from math import sqrt
from pandas import Grouper
from statsmodels.tsa.stattools import adfuller
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
from scipy.stats import boxcox

url1 = 'tbl_monitor_data_export.csv'
url2 = 'tbl_monitors_export.csv'
series = pandas.read_csv(url1,header=0)
series['ts']=pandas.to_datetime(series['ts'])
series=series.set_index('ts')
series['Day']=series.index.day
print(series['Day'].head(4))
dataset1= pandas.read_csv(url2)

#Dataset for each id
numMax=series['monitor_id'].max()
for i in range(1,numMax+1):
    df=series.loc[series['monitor_id'] == i]
    print(df.describe())
    a=df.describe()
    print(dataset1.loc[dataset1['id']==i]['label'])
    print(dataset1.loc[dataset1['id']==1]['host_uri'])
    print("have speed: Mean=%f and std = %f "%(a.loc['mean']['result_float2'],a.loc['std']['result_float2']))
dataseta=series.loc[series['monitor_id'] == 1]
#Validation Dataset
split_point=int(len(dataseta)*0.8)
dataset, validation=dataseta[0:split_point],dataseta[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# Peristence
X = dataset['result_float2'].values
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
i=0
for i in range(len(test)):
    # predict
    yhat = history[-1]
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    print(i)
    i=i+1
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
#Try to draw it on X Y

#PLOT
# dataseta['result_float2'].plot()
# plt.show()
# dataseta['result_float2'].hist()
# plt.show()
# dataseta['result_float2'].plot(kind='kde')
# plt.show()
#box and whisker plots
#groups=dataseta['19':'29'].groupby(Grouper('A'))

# dataseta['result_float2'].plot(kind='box', subplots=True, layout=(2,2), sharex=False,sharey=False)
# plt.show()
# dataseta.hist(ax=plt.gca())
# plt.show()
# autocorrelation_plot(dataseta['result_float2'])
# plt.show()
#timegrouper
print('id1')
print(dataseta['result_float2'].head())
print(dataseta.shape)
df=pandas.read_csv('~/Downloads/sales.csv')
print(df.dtypes)
#ARIMA Model
## manually configured ARIMA
def difference(dataset, interval=1):
    diff =list()
    for i in range (interval, len(dataset)):
        value = dataset[i]-dataset[i-interval]
        diff.append(value)
    return diff
# X= dataset['result_float2'].values
# months_in_year=12
# stationary = difference(X, months_in_year)
# stationary.index = dataset.index[months_in_year:]
# #check if stationary
# result = adfuller(stationary)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
#     print('\t%s: %.3f' % (key, value))
# save
# stationary.to_csv('stationary.csv')
# # plot
# stationary.plot()
# plt.show()
# # Draw ACF and PACF Autocorrelation Function and Partial
# plt.figure()
# plt.subplot(211)
# plot_acf(stationary, ax=plt.gca())
# plt.subplot(212)
# plot_pacf(stationary, ax=plt.gca())
# plt.show()
#Try  1 1 1
#create a differenced series
def inverse_difference(history, yhat, interval=1):
    return yhat+history[-interval]

#Walk-forward validation
# for i in range(len(test)):
#     # difference data
#     months_in_year =12
#     diff=difference(history, months_in_year)
#     #predict
#     model=ARIMA(diff, order=(0,0,1))
#     model_fit=model.fit(trend='nc', disp=0)
#     yhat=model_fit.forecast()[0]
#     yhat=inverse_difference(history, yhat, months_in_year)
#     predictions.append(yhat)
#     #observation
#     obs=test[i]
#     history.append(obs)
#     print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# # report performance
# mse = mean_squared_error(test, predictions)
# rmse = sqrt(mse)
# print('RMSE: %.3f' % rmse)

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
# def evaluate_arima_model(X, arima_order):
#     #prepare training dataset
#     train_size = int(len(X)*0.5)
#     train, test = X[0:train_size], X[train_size:]
#     history =[x for x in train]
#     #make prediction
#     predictions=list()
#     for t in range(len(test)):
#         #difference data
#         months_in_year=12
#         diff = difference(history, months_in_year)
#         model = ARIMA(diff, order=arima_order)
#         model_fit=model.fit(trend='nc', disp=0)
#         yhat = model_fit.forecast()[0]
#         yhat = inverse_difference(history, yhat, months_in_year)
#         predictions.append(yhat)
#         history.append(test[t])
#     mse = mean_squared_error(test, predictions)
#     rmse=sqrt(mse)
#     return rmse
# evaluate combinations of p, d and q values for an ARIMA model
# def evaluate_models(dataset, p_values, d_values, q_values):
#     best_score, best_cfg = float('inf'), None
#     for p in p_values:
#         for d in d_values:
#             for q in q_values:
#                 order = (p,d,q)
#                 try:
#                     mse = evaluate_arima_model(dataset, order)
#                     if mse < best_score:
#                         best_score, best_cfg =mse, order
#                     print('ARIMA%s RMSE=%.3f' %(order, mse))
#                 except:
#                     continue
#     print('Best Arima%s RMSE=%.3f' % (best_cfg, best_score))
#
# #load dataset
# p_values = range(0,7)
# d_values = range(0,3)
# q_values = range(0,7)
# warnings.filterwarnings("ignore")
# evaluate_models(dataset['result_float2'][0:5000], p_values, d_values, q_values)

####################
# #find bias
# X = dataset['result_float2'].values
# train_size = int(len(X) * 0.50)
# train, test = X[0:train_size], X[train_size:]
# # walk-forward validation
# # history = [x for x in train]
# # predictions = list()
# # for i in range(len(test)):
# #     months_in_year=12
# #     diff=difference(history, months_in_year)
# #     #predict
# #     model = ARIMA(diff, order=(0,0,1))
# #     model_fit=model.fit(trend='nc', disp=0)
# #     yhat=model_fit.forecast()[0]
# #     yhat=inverse_difference(history, yhat, months_in_year)
# #     predictions.append(yhat)
# #     obs=test[i]
# #     history.append(obs)
# # residuals=[test[i]-predictions[i] for i in range(len(test))]
# # residuals = DataFrame(residuals)
# # print(residuals.describe())
# # bias = residuals.describe()[1]
# # print(bias)
# # ## Finalize Model
# # def __getnewargs__(self):
# #     return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
# #
# # ARIMA.__getnewargs__ = __getnewargs__
# # # create a differenced series
# # #load data
# minutes_in_hour=60
# diff= difference(X, minutes_in_hour)
# # #fit model
# # model = ARIMA(diff, order=(0,0,1))
# # model_fit = model.fit(trend='nc', disp=0)
# # #bias constant
# # #save model
# # model_fit.save('model.pkl')
# # numpy.save('model_bias.npy',[bias])
# # yhat=float(model_fit.forecast()[0])
# # yhat= bias + inverse_difference(X, yhat, months_in_year)
# # print('Predicted: %.3f' %yhat)
# #
# #
# #
# #load and prepare datasets
# X = dataset['result_float2'].values
# train_size = int(len(X) * 0.8)
# train, test = X[0:train_size], X[train_size:]
# # walk-forward validation
# history = [x for x in train]
# #predictions =
#
# bias= -0.000014
# y=validation['result_float2'].values
# #load model
# model = ARIMA(diff, order=(0,0,1))
# model_fit = model.fit(disp=0)
# model_fit.save('model.pkl')
# predictions=list()
# #print(model_fit)
# #predictions=model_fit.forecast()
# #make first prediction
# predictions = list()
# yhat=model_fit.forecast()[0]
# yhat   = bias+inverse_difference(history, yhat, minutes_in_hour)
# predictions.append(yhat)
# history.append(y[0])
# print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# #rolling forecasts
# for i in range(1, len(y)):
#     #difference data
#     months_in_year =12
#     diff = difference(history, minutes_in_hour)
#     #predict
#     model = ARIMA(diff, order=(0,0,1))
#     model_fit=model.fit(trend='nc',disp=0)
#     yhat = model_fit.forecast()[0]
#     yhat = bias + inverse_difference(history, yhat, minutes_in_hour)
#     predictions.append(yhat)
#     #observation
#     obs = y[i]
#     history.append(obs)
#     print(i)
#     print('>predicted=%.3f, Expected=%.3f' %(yhat, obs))
# #report performance
# mse=mean_squared_error(y, predictions)
# rmse=sqrt(mse)
# print('RMSE: %.3f' %rmse)
# #
# start_index = len(diff)
# end_index = start_index + 8640
# forecast = model_fit.predict(start=start_index, end=end_index)
# # invert the differenced forecast to something usable
# day = 1
# for yhat in forecast:
#     inverted = bias +inverse_difference(history, yhat, minutes_in_hour)
#     print('Day %d: %f' % (day, inverted))
#     history.append(inverted)
#     predictions.append(inverted)
#     day += 1
# #model_fit.save('model1.pkl')
# #print(model_fit)
# #predictions=model_fit.forecast()
# #make first prediction
# plt.plot(y)
# plt.plot(predictions, color='red')
# plt.show()
# #try
# # model = ARIMA(diff, order=(0,0,1))
# # model_fit = model.fit(disp=0)
# # multi-step out-of-sample forecast

##################
