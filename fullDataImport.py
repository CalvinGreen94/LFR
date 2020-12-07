import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,TimeSeriesSplit
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# %matplotlib inline
import ssl
import json
import ast
import os
import bitfinex
api_v2 = bitfinex.bitfinex_v2.api_v2()
result = api_v2.candles()
import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
#
data = yf.download("AR", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/AR.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('AR')))

data = yf.download("CHK", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/CHK.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('CHK"')))

data = yf.download("PCG", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/PCG.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('PCG"')))

data = yf.download("SPY", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/SPY.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('SPY')))

data = yf.download("AAPL", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/AAPL.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('AAPL')))

data = yf.download("EA", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/EA.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('EA')))

data = yf.download("FB", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/FB.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('FB')))

data = yf.download("ROKU", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/ROKU.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('ROKU')))

data = yf.download("SIRI", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/SIRI.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('SIRI')))

data = yf.download("XOM", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/XOM.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('XOM')))

data = yf.download("^GSPC", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/^GSPC.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('^GSPC')))

data = yf.download("MSFT", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/MSFT.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('MSFT')))


data = yf.download("TSLA", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/TSLA.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('TSLA')))

data = yf.download("GRPN", start="2010-01-03", end="2020-11-3")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/stocks_portfolio/GRPN.csv')
print('RETRIEVING linkLY STOCK DATA FOR {}'.format(str('GRPN')))

# plotly.tools.set_credentials_file(username='Gamma-AI1011', api_key='KoXH9I7ffpwUueVaa7TT')

#
# Cov = pd.read_csv("data/futures/6B 03-21.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] # "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/futures_csv/6B03-21.Last.csv')
# data = pd.read_csv('data/futures/futures_csv/6B03-21.Last.csv')
# data = data.drop(['Unnamed: 0'], axis=1)
# df0,df1 = data.shape[0], data.shape[1]
# # data = data.drop(['Unnamed: 0'],axis=1)
# print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES \n'.format(df0,df1))
# data = data.drop(['yyyyMMdd'],axis=1)
# high = data['high price']
# low = data['low price']
# open = data['open price']
# close = data['close price']
# plt.subplot(2, 1, 2)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')
# plt.legend(['6B'])
# plt.show()
# plt.close()
# #data.describe()
# X= data.drop(['open price'],axis=1)
# y= data['open price']
# X_close= data.drop(['close price'],axis=1)
# y_close= data['close price']
# mini = MinMaxScaler()
# X = mini.fit_transform(X)
# X_close = mini.fit_transform(X_close)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
# X_t,X_tt,y_t,y_tt = train_test_split(X_close,y_close,test_size=.67,shuffle=False)
# reg = LinearRegression(normalize=True,n_jobs=-1)
# fit = reg.fit(X_train,y_train)
# fit2 = reg.fit(X_t,y_t)
#     # score = reg.score(X_test,y_test)
# pred = reg.predict(X_test[[-1]])
# pred_close = reg.predict(X_tt[[-1]])
# # fit = reg.fit(X_train,y_train)
# # score = reg.score(X_test,y_test)
# # print('score for 6B {}'.format(score))
# # pred = reg.predict(X_test[[-1]])
# # pred1 = reg.predict(X_test[[-1]])
# # dataFrame = pd.DataFrame(X_test[-1])
# # dataFrame.to_csv('6bpred.csv',dataFrame)
# # print('6B 03-21 score {}'.format(score))
# print('6B  PREVIOUS OPEN --> $ {}'.format(open[6622:]))
# print('6B PREDICTED OPEN {}\n'.format(pred))
# print('6B   PREVIOUS CLOSE--> $ {}'.format(close[6622:]))
# print('6B PREDICTED CLOSE {}\n'.format(pred_close))
# #    print('tick gain/loss {}'.format((pred[[-1]]-X_test[[-1]])))
# print('PREDICTED GAIN/LOSS $ {}\n'.format((6.25*3*5)+25000))
#
# Cov = pd.read_csv("data/futures/RTY 06-21.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] # "yyyyMMdd" = MINUTE BY MINUTE
#
# data = Cov.to_csv('data/futures/futures_csv/RTY06-21.Last.csv')
# data = pd.read_csv('data/futures/futures_csv/RTY06-21.Last.csv')
# data = data.drop(['Unnamed: 0'], axis=1)
# df0,df1 = data.shape[0], data.shape[1]
# print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES \n'.format(df0,df1))
# data = data.drop(['yyyyMMdd'],axis=1)
# high1= data['high price']
# low1 = data['low price']
# open = data['open price']
# close = data['close price']
# plt.subplot(2, 1, 2)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')
# plt.legend(['RTY'])
# plt.show()
# # plt.savefig('20-100_moving_average')
# plt.close()
#
# # data.describe()
# def train_test1():
#     X= data.drop(['open price'],axis=1)
#     y= data['open price']
#     X_close= data.drop(['close price'],axis=1)
#     y_close= data['close price']
#     returns = data.pct_change()[1:]
#     mini = MinMaxScaler()
#     X = mini.fit_transform(X)
#     X_close = mini.fit_transform(X_close)
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
#     X_t,X_tt,y_t,y_tt = train_test_split(X_close,y_close,test_size=.67,shuffle=False)
#     reg = LinearRegression(normalize=True,n_jobs=-1)
#     fit = reg.fit(X_train,y_train)
#     fit2 = reg.fit(X_t,y_t)
#     # score = reg.score(X_test,y_test)
#     pred = reg.predict(X_test[[-1]])
#     pred_close = reg.predict(X_tt[[-1]])
#     # score = reg.score(X_test,y_test)
#     # pred = reg.predict(X_test[[-1]])
#     # print('RTY score {}'.format(score))
#     # print('previous open {}'.format(data[[-1]]))
#
#     print('RTY  PREVIOUS OPEN --> $ {}'.format(open[5118:]))
#     print('RTY PREDICTED OPEN {}\n'.format(pred))
#     print('RTY   PREVIOUS CLOSE--> $ {}'.format(close[5118:]))
#     print('RTY PREDICTED CLOSE {}\n'.format(pred_close))
#     #    print('tick gain/loss {}'.format((pred[[-1]]-X_test[[-1]])))
#     print('PREDICTED GAIN/LOSS $ {}\n'.format((5*3*20)+25000))
#     return X_train,X_test,y_train,y_test,pred
# train_test1()
#
# Cov = pd.read_csv("data/futures/CL 04-21.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/futures_csv/CL04-21.Last.csv')
# data = pd.read_csv('data/futures/futures_csv/CL04-21.Last.csv')
# data = data.drop(['Unnamed: 0'],axis=1)
# df0,df1 = data.shape[0], data.shape[1]
# print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES \n'.format(df0,df1))
# data = data.drop(['yyyyMMdd'],axis=1)
# high2 =data['high price']
# low2 = data['low price']
# open = data['open price']
# close = data['close price']
# plt.subplot(2, 1, 2)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')
# plt.legend(['CL'])
# plt.show()
# # plt.savefig('20-100_moving_average')
# plt.close()
# plt.savefig('futures_graphs/CL')
# # data.describe()
# def train_test2():
#     X= data.drop(['open price'],axis=1)
#     y= data['open price']
#     X_close= data.drop(['close price'],axis=1)
#     y_close= data['close price']
#     mini = MinMaxScaler()
#     X = mini.fit_transform(X)
#     X_close = mini.fit_transform(X_close)
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
#     X_t,X_tt,y_t,y_tt = train_test_split(X_close,y_close,test_size=.67,shuffle=False)
#     reg = LinearRegression(normalize=True,n_jobs=-1)
#     fit = reg.fit(X_train,y_train)
#     fit2 = reg.fit(X_t,y_t)
#     # score = reg.score(X_test,y_test)
#     pred = reg.predict(X_test[[-1]])
#     pred_close = reg.predict(X_tt[[-1]])
#
#     print('CL 4-21  PREVIOUS OPEN --> $ {}'.format(open[10769:]))
#     print('CL 4-21 PREDICTED OPEN {}\n'.format(pred))
#     print('CL 4-21   PREVIOUS CLOSE--> $ {}'.format(close[10769:]))
#     print('CL 4-21 PREDICTED CLOSE {}\n'.format(pred_close))
#     #    print('tick gain/loss {}'.format((pred[[-1]]-X_test[[-1]])))
#     print('PREDICTED GAIN/LOSS $ {}\n'.format((10*3*3)+25000))
#     return X_train,X_test,y_train,y_test,fit,pred
# train_test2()
#
# Cov = pd.read_csv("data/futures/ES 06-21.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/futures_csv/ES06-21.Last.csv')
# data = pd.read_csv('data/futures/futures_csv/ES06-21.Last.csv')
# data = data.drop(['Unnamed: 0'],axis=1)
# df0,df1 = data.shape[0], data.shape[1]
# print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES \n'.format(df0,df1))
# data = data.drop(['yyyyMMdd'],axis=1)
# high3= data['high price']
# low3= data['low price']
# open = data['open price']
# close = data['close price']
# plt.subplot(2, 1, 2)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')
# plt.legend(['ES'])
# plt.show()
# plt.close()
# plt.savefig('futures_graphs/ES')
# # data.describe()
# def train_test3():
#     X= data.drop(['open price'],axis=1)
#     y= data['open price']
#     X_close= data.drop(['close price'],axis=1)
#     y_close= data['close price']
#     mini = MinMaxScaler()
#     X = mini.fit_transform(X)
#     X_close = mini.fit_transform(X_close)
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
#     X_t,X_tt,y_t,y_tt = train_test_split(X_close,y_close,test_size=.67,shuffle=False)
#     reg = LinearRegression(normalize=True,n_jobs=-1)
#     fit = reg.fit(X_train,y_train)
#     fit2 = reg.fit(X_t,y_t)
#     # score = reg.score(X_test,y_test)
#     pred = reg.predict(X_test[[-1]])
#     pred_close = reg.predict(X_tt[[-1]])
#     # pred = reg.predict(X_test[[-1]])
#     # print('ES 06-21 score {}'.format(score))
#     print('ES  PREVIOUS OPEN --> $ {}'.format(open[8972:]))
#     print('ES PREDICTED OPEN {}\n'.format(pred))
#     print('ES   PREVIOUS CLOSE--> $ {}'.format(close[8972:]))
#     print('ES PREDICTED CLOSE {}\n'.format(pred_close))
#     #    print('tick gain/loss {}'.format((pred[[-1]]-X_test[[-1]])))
#     print('PREDICTED GAIN/LOSS $ {}\n'.format((12.50*3*5)+25000))
#     return X_train,X_test,y_train,y_test,fit,pred
# train_test3()
#
# Cov = pd.read_csv("data/futures/GC 04-21.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/futures_csv/GC04-21.Last.csv')
# data = pd.read_csv('data/futures/futures_csv/GC04-21.Last.csv')
# data.head()
# data = data.drop(['Unnamed: 0'],axis=1)
# df0,df1 = data.shape[0], data.shape[1]
# print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES \n'.format(df0,df1))
# data = data.drop(['yyyyMMdd'],axis=1)
# high4= data['high price']
# low4 = data['low price']
# open = data['open price']
# close = data['close price']
# plt.subplot(2, 1, 2)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')
# plt.legend(['GC'])
# plt.show()
# plt.close()
# plt.savefig('futures_graphs/GC')
# # data.describe()
# def train_test4():
#     X= data.drop(['open price'],axis=1)
#     y= data['open price']
#     X_close= data.drop(['close price'],axis=1)
#     y_close= data['close price']
#     mini = MinMaxScaler()
#     X = mini.fit_transform(X)
#     X_close = mini.fit_transform(X_close)
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
#     X_t,X_tt,y_t,y_tt = train_test_split(X_close,y_close,test_size=.67,shuffle=False)
#     reg = LinearRegression(normalize=True,n_jobs=-1)
#     fit = reg.fit(X_train,y_train)
#     fit2 = reg.fit(X_t,y_t)
#     # score = reg.score(X_test,y_test)
#     pred = reg.predict(X_test[[-1]])
#     pred_close = reg.predict(X_tt[[-1]])
#     # print('GC 04-21 score {}'.format(score))
#     print('GC  PREVIOUS OPEN --> $ {}'.format(open[10639:]))
#     print('GC PREDICTED OPEN {}\n'.format(pred))
#     print('GC   PREVIOUS CLOSE--> $ {}'.format(close[10639:]))
#     print('GC PREDICTED CLOSE {}\n'.format(pred_close))
#     print('PREDICTED GAIN/LOSS $ {}\n'.format((10*3*-7)+25000))
#     return X_train,X_test,y_train,y_test,fit,pred
# train_test4()
# #
# Cov = pd.read_csv("data/futures/NQ 06-21.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/futures_csv/NQ06-21.Last.csv')
# data = pd.read_csv('data/futures/futures_csv/NQ06-21.Last.csv')
# data = data.drop(['Unnamed: 0'],axis=1)
# data = data.drop(['yyyyMMdd'],axis=1)
# open = data['open price']
# close = data['close price']
# plt.subplot(2, 1, 2)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')
# plt.legend(['NQ'])
# plt.show()
# plt.close()
# df0,df1 = data.shape[0], data.shape[1]
# print('NQ futures Data Has {} Transactions with {} FEATURES \n'.format(df0,df1))
# # #data.describe()
# X= data.drop(['open price'],axis=1)
# y= data['open price']
# X_close= data.drop(['close price'],axis=1)
# y_close= data['close price']
# mini = MinMaxScaler()
# X = mini.fit_transform(X)
# X_close = mini.fit_transform(X_close)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
# X_t,X_tt,y_t,y_tt = train_test_split(X_close,y_close,test_size=.67,shuffle=False)
# reg = LinearRegression(normalize=True,n_jobs=-1)
# fit = reg.fit(X_train,y_train)
# fit2 = reg.fit(X_t,y_t)
#     # score = reg.score(X_test,y_test)
# pred = reg.predict(X_test[[-1]])
# pred_close = reg.predict(X_tt[[-1]])
# # print('NQ 06-21 score  {}'.format(reg.score(X_test,y_test)))
# print('NQ  PREVIOUS OPEN --> $ {}'.format(open[7492:]))
# print('NQ PREDICTED OPEN {}\n'.format(pred))
# print('NQ   PREVIOUS CLOSE--> $ {}'.format(close[7492:]))
# print('NQ PREDICTED CLOSE {}\n'.format(pred_close))
# #    print('tick gain/loss {}'.format((pred[[-1]]-X_test[[-1]])))
# print('PREDICTED GAIN/LOSS $ {}\n'.format((5*3*2)+25000))
#
#
#
# Cov = pd.read_csv("data/futures/ZB 06-21.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/futures_csv/ZB06-21.Last.csv')
# data = pd.read_csv('data/futures/futures_csv/ZB06-21.Last.csv')
# data = data.drop(['Unnamed: 0'],axis=1)
# df0,df1 = data.shape[0], data.shape[1]
# print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES \n'.format(df0,df1))
# data = data.drop(['yyyyMMdd'],axis=1)
# high5 =  data['high price']
# low5 = data['low price']
# open = data['open price']
# close = data['close price']
#
# plt.subplot(2, 1, 2)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')
# plt.legend(['ZB'])
# plt.show()
# # plt.savefig('20-100_moving_average')
# plt.close()
# plt.savefig('futures_graphs/ZB')
# # data.describe()
# def train_test5():
#     X= data.drop(['open price'],axis=1)
#     y= data['open price']
#     X_close= data.drop(['close price'],axis=1)
#     y_close= data['close price']
#     mini = MinMaxScaler()
#     X = mini.fit_transform(X)
#     X_close = mini.fit_transform(X_close)
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
#     X_t,X_tt,y_t,y_tt = train_test_split(X_close,y_close,test_size=.67,shuffle=False)
#     reg = LinearRegression(normalize=True,n_jobs=-1)
#     fit = reg.fit(X_train,y_train)
#     fit2 = reg.fit(X_t,y_t)
#     # score = reg.score(X_test,y_test)
#     pred = reg.predict(X_test[[-1]])
#     pred_close = reg.predict(X_tt[[-1]])    # print('ZB 06-21 score {}'.format(score))
#     print('ZB  PREVIOUS OPEN --> $ {}'.format(open[10726:]))
#     print('ZB PREDICTED OPEN {}\n'.format(pred))
#     print('ZB   PREVIOUS CLOSE--> $ {}'.format(close[10726:]))
#     print('ZB PREDICTED CLOSE {}\n'.format(pred_close))
#     print('PREDICTED GAIN/LOSS $ {} \n'.format((31.25*3*3)+25000))
#     # print('ACTUAL GAIN/LOSS ${}\n'.format(31.25*3*(pred-)+25000)
#     return X_train,X_test,y_train,y_test,fit,pred
# train_test5()

# plt.subplot(2, 1, 1)
# plt.plot(data['open price'])
# plt.xlabel('time')
# plt.ylabel('price')
#
#
# plt.subplot(2, 1, 1)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')
#
# plt.subplot(2, 1, 1)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')
#
# plt.subplot(2, 1, 1)
# plt.plot(data['open price'])
# plt.xlabel('time (m)')
# plt.ylabel('price')


print('DOWNLOADING BITCOIN PAIRS DATA')
api_v2 = bitfinex.bitfinex_v2.api_v2()
result = api_v2.candles()
time_step = 60000000
 # Define query parameters
pair = 'btcusd' # Currency pair of interest
bin_size = '15m' # This will return minute data
limit = 1000    # We want the maximum of 1000 data points
# Define the start date
t_start = datetime.datetime(2020,10, 31, 0, 0)
t_start = time.mktime(t_start.timetuple()) * 1000
# Define the end date
t_stop = datetime.datetime(2020,11, 3, 0, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000
result = api_v2.candles(symbol=pair, interval=bin_size,
                        limit=limit, start=t_start, end=t_stop)
# result =  pd.DataFrame(list_of_rows,columns=['PRICES','PRICE:'])
def fetch_data(start, stop, symbol, interval, tick_limit, step):
    api_v2 = bitfinex.bitfinex_v2.api_v2()
    data = []
    start = start - step
    while start < stop:
        start = start + step
        end = start + step
        res = api_v2.candles(symbol=symbol, interval=interval,
                             limit=tick_limit, start=start,
                             end=end)
        data.extend(res)
        time.sleep(2)
    return data
api_v1 = bitfinex.bitfinex_v1.api_v1()
pairs = ['ethusd']#api_v1.symbols() #['btcusd','xtzusd','oxtusd','dntusd','xtzusd','xrpusd','zecusd','ethusd','etcusd','xlmusd','oxtusd','dntusd','linkusd','eosusd']
save_path = 'data/crypto/crypto_portfolio/15m'
if os.path.exists(save_path) is False:
    os.mkdir(save_path)
for pair in pairs:
    pair_data = fetch_data(start=t_start, stop=t_stop, symbol=pair, interval=bin_size, tick_limit=limit, step=time_step)
    # Remove error messages
    ind = [np.ndim(x) != 0 for x in pair_data]
    pair_data = [i for (i, v) in zip(pair_data, ind) if v]
    #Create pandas data frame and clean data
    names = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(pair_data, columns=names)
    df.drop_duplicates(inplace=True)
    # df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    print('Done downloading data. Saving to .csv.')
    df.to_csv('{}/bitfinex_{}.csv'.format(save_path, pair))
    print('Done saving pair{}. Moving to next pair.'.format(pair))
    # df.drop(['volume'],axis=1)
print('Done retrieving data')

print('DOWNLOADING BITCOIN PAIRS DATA')
api_v2 = bitfinex.bitfinex_v2.api_v2()
result = api_v2.candles()
time_step = 60000000
 # Define query parameters
pair = 'btcusd' # Currency pair of interest
bin_size = '5m' # This will return minute data
limit = 1000    # We want the maximum of 1000 data points
# Define the start date
t_start = datetime.datetime(2020,10, 31, 0, 0)
t_start = time.mktime(t_start.timetuple()) * 1000
# Define the end date
t_stop = datetime.datetime(2020,11, 3, 0, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000
result = api_v2.candles(symbol=pair, interval=bin_size,
                        limit=limit, start=t_start, end=t_stop)
# result =  pd.DataFrame(list_of_rows,columns=['PRICES','PRICE:'])
def fetch_data(start, stop, symbol, interval, tick_limit, step):
    api_v2 = bitfinex.bitfinex_v2.api_v2()
    data = []
    start = start - step
    while start < stop:
        start = start + step
        end = start + step
        res = api_v2.candles(symbol=symbol, interval=interval,
                             limit=tick_limit, start=start,
                             end=end)
        data.extend(res)
        time.sleep(2)
    return data
api_v1 = bitfinex.bitfinex_v1.api_v1()
pairs = ['ethusd']#api_v1.symbols() #['btcusd','xtzusd','oxtusd','dntusd','xtzusd','xrpusd','zecusd','ethusd','etcusd','xlmusd','oxtusd','dntusd','linkusd','eosusd']
save_path = 'data/crypto/crypto_portfolio/5m'
if os.path.exists(save_path) is False:
    os.mkdir(save_path)
for pair in pairs:
    pair_data = fetch_data(start=t_start, stop=t_stop, symbol=pair, interval=bin_size, tick_limit=limit, step=time_step)
    # Remove error messages
    ind = [np.ndim(x) != 0 for x in pair_data]
    pair_data = [i for (i, v) in zip(pair_data, ind) if v]
    #Create pandas data frame and clean data
    names = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(pair_data, columns=names)
    df.drop_duplicates(inplace=True)
    # df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    print('Done downloading data. Saving to .csv.')
    df.to_csv('{}/bitfinex_{}.csv'.format(save_path, pair))
    print('Done saving pair{}. Moving to next pair.'.format(pair))
    # df.drop(['volume'],axis=1)
print('Done retrieving data')

print('DOWNLOADING BITCOIN PAIRS DATA')
api_v2 = bitfinex.bitfinex_v2.api_v2()
result = api_v2.candles()
time_step = 60000000
 # Define query parameters
pair = 'btcusd' # Currency pair of interest
bin_size = '1m' # This will return minute data
limit = 1000    # We want the maximum of 1000 data points
# Define the start date
t_start = datetime.datetime(2020,11, 1,0, 0)
t_start = time.mktime(t_start.timetuple()) * 1000
# Define the end date
t_stop = datetime.datetime(2020,11, 3, 0, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000
result = api_v2.candles(symbol=pair, interval=bin_size,
                        limit=limit, start=t_start, end=t_stop)
# result =  pd.DataFrame(list_of_rows,columns=['PRICES','PRICE:'])
def fetch_data(start, stop, symbol, interval, tick_limit, step):
    api_v2 = bitfinex.bitfinex_v2.api_v2()
    data = []
    start = start - step
    while start < stop:
        start = start + step
        end = start + step
        res = api_v2.candles(symbol=symbol, interval=interval,
                             limit=tick_limit, start=start,
                             end=end)
        data.extend(res)
        time.sleep(2)
    return data
api_v1 = bitfinex.bitfinex_v1.api_v1()
pairs = ['ethusd']#api_v1.symbols() #['btcusd','xtzusd','oxtusd','dntusd','xtzusd','xrpusd','zecusd','ethusd','etcusd','xlmusd','oxtusd','dntusd','linkusd','eosusd']
save_path = 'data/crypto/crypto_portfolio/1m'
if os.path.exists(save_path) is False:
    os.mkdir(save_path)
for pair in pairs:
    pair_data = fetch_data(start=t_start, stop=t_stop, symbol=pair, interval=bin_size, tick_limit=limit, step=time_step)
    # Remove error messages
    ind = [np.ndim(x) != 0 for x in pair_data]
    pair_data = [i for (i, v) in zip(pair_data, ind) if v]
    #Create pandas data frame and clean data
    names = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(pair_data, columns=names)
    df.drop_duplicates(inplace=True)
    # df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    print('Done downloading data. Saving to .csv.')
    df.to_csv('{}/bitfinex_{}.csv'.format(save_path, pair))
    print('Done saving pair{}. Moving to next pair.'.format(pair))
    # df.drop(['volume'],axis=1)
print('Done retrieving data')

#
# print('DOWNLOADING BITCOIN PAIRS DATA')
# api_v2 = bitfinex.bitfinex_v2.api_v2()
# result = api_v2.candles()
# time_step = 60000000
#  # Define query parameters
# pair = 'btcusd' # Currency pair of interest
# bin_size = '1d' # This will return minute data
# limit = 1000    # We want the maximum of 1000 data points
# # Define the start date
# t_start = datetime.datetime(2014, 1,  4, 0, 0)
# t_start = time.mktime(t_start.timetuple()) * 1000
# # Define the end date
# t_stop = datetime.datetime(2020,9, 26, 0, 0)
# t_stop = time.mktime(t_stop.timetuple()) * 1000
# result = api_v2.candles(symbol=pair, interval=bin_size,
#                         limit=limit, start=t_start, end=t_stop)
# # result =  pd.DataFrame(list_of_rows,columns=['PRICES','PRICE:'])
# def fetch_data(start, stop, symbol, interval, tick_limit, step):
#     api_v2 = bitfinex.bitfinex_v2.api_v2()
#     data = []
#     start = start - step
#     while start < stop:
#         start = start + step
#         end = start + step
#         res = api_v2.candles(symbol=symbol, interval=interval,
#                              limit=tick_limit, start=start,
#                              end=end)
#         data.extend(res)
#         time.sleep(2)
#     return data
# api_v1 = bitfinex.bitfinex_v1.api_v1()
# pairs = ['btcusd','xtzusd','xrpusd','eosusd','kncusd','xlmusd','oxtusd','dntusd']#api_v1.symbols() #['btcusd','xtzusd','oxtusd','dntusd','xtzusd','xrpusd','zecusd','ethusd','etcusd','xlmusd','oxtusd','dntusd','linkusd','eosusd']
# save_path = 'data/crypto/crypto_portfolio/1d'
# if os.path.exists(save_path) is False:
#     os.mkdir(save_path)
# for pair in pairs:
#     pair_data = fetch_data(start=t_start, stop=t_stop, symbol=pair, interval=bin_size, tick_limit=limit, step=time_step)
#     # Remove error messages
#     ind = [np.ndim(x) != 0 for x in pair_data]
#     pair_data = [i for (i, v) in zip(pair_data, ind) if v]
#     #Create pandas data frame and clean data
#     names = ['time', 'open', 'high', 'low', 'close', 'volume']
#     df = pd.DataFrame(pair_data, columns=names)
#     df.drop_duplicates(inplace=True)
#     # df['time'] = pd.to_datetime(df['time'], unit='ms')
#     df.set_index('time', inplace=True)
#     df.sort_index(inplace=True)
#     print('Done downloading data. Saving to .csv.')
#     df.to_csv('{}/bitfinex_{}.csv'.format(save_path, pair))
#     print('Done saving pair{}. Moving to next pair.'.format(pair))
#     # df.drop(['volume'],axis=1)
# print('Done retrieving data')
