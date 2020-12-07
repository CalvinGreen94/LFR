import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd
import datetime as dt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,TimeSeriesSplit
# apiKey = "1a824261af5b56f17e51f86097bacd51"
# apiSecret = "yh68uZG5ufDpJNJD5B1oIN+uQZW/XDKGVqGP8rg/9d73bWFv4I4PpRWZk17RjTAkSebU97t/DITKZPLtJFbBwg=="
# passphrase = "t9jtvccivy"

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
# prints formatted price

def moving_20average(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    # np.savetxt("20avgHigh.csv", ret[n:], delimiter=",")
    return ret[n - 1:] / n

def moving_100average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# auth_client = cbpro.AuthenticatedClient(apiKey,apiSecret,passphrase)
# auth_client_df = pd.DataFrame(auth_client.get_accounts())
# auth_client_df.to_csv('Current_infochanning.csv')
# current_info = pd.read_csv('Current_infochanning.csv')
# current_info = current_info.drop(['Unnamed: 0'],axis=1)
# pie = current_info['available'][[11]] #9,13,18,20,22,23,24,26
# plt.pie(pie)
# plt.legend([current_info['currency'][11]]) #]current_info['currency'][9],current_info['currency'][13],current_info['currency'][18],current_info['currency'][20],current_info['currency'][22],current_info['currency'][23],current_info['currency'][24],current_info['currency'][26]
# plt.show()
# plt.close()
# current_info
# # def vol_ind(data):
# #     data = pd.read_csv(data)
# #     volume = data['volume']
# #     volume = pd.DataFrame(volume)
# #     volume = volume.to_csv('volume.csv')
# #     plt.plot(volume)
# #     plt.savefig('images/volume.png')
# #     return volume
# # a = data
# # print(moving_average(a))
# # low_moving = pd.DataFrame(moving_average(a))
# # low_moving = low_moving.to_csv('LOW_SIMPLE_MOVING_AVG.csv')
# def current_price(currency):
#     currency = currency
#     Period = 3600
#     historicData = auth_client.get_product_historic_rates(currency, granularity=Period)
#     #     print(historicData)
#             # Make an array of the historic price data from the matrix
#     price = np.squeeze(np.asarray(np.matrix(historicData)[:,4]))
#             # Wait for 1 second, to avoid API limit
#     time.sleep(1)
#             # Get latest data and show to the user for reference
#     newData = auth_client.get_product_ticker(product_id=currency)
#     currentPrice=newData['price']
#     print('currency: {}'.format(currency))
#     print('current_price {} \n\n'.format(currentPrice))
#     return currentPrice
#
# def history(currency):
#     currency = currency
#     Period = 300
#     historicData = auth_client.get_product_historic_rates(currency, granularity=Period)
#     historicData = pd.DataFrame(historicData,columns=['time','open','high','low','close','volume'])
#     price = historicData['high']
#     price.to_csv('data/crypto/crypto_portfolio/{}_high.csv'.format(currency))
#             # Wait for 1 second, to avoid API limit
#     time.sleep(1)
#     return historicData
# history('etc-usd')
def profit_target(token,current_holdings,target_percentage):
    token = token
    print('\n\n {} target'.format(token))
    current_holdings = current_holdings
    target_percentage = current_holdings * .3
    total_target = current_holdings+target_percentage
    print('{} profit target {}, == {}'.format(token,target_percentage,total_target))
    return target_percentage
def loss(token,current_holdings,loss):
    token = token
    print('\n\n {} loss'.format(token))
    current_holdings = current_holdings
    target_percentage = current_holdings * .1
    total_loss = current_holdings-target_percentage
    print('{} stop loss {}, == {}'.format(token,target_percentage,total_loss))
    return target_percentage


def moving_20average(a, n=25) :
    ret = pd.DataFrame.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_100average(a, n=100) :
    ret = pd.DataFrame.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# current_priceetc = current_price('etc-usd')
#
#
# auth_clientetc = current_info['available'][11]
# print('available {}: {}\n\n'.format(current_info['currency'][11],auth_clientetc))
#
#
# currentetc = float(current_priceetc) * float(auth_clientetc)
# print('current etc balance: {}\n\n'.format(currentetc))
#
#
# print('-->PROFIT TARGETS:')
# etc_tar = profit_target('etc',currentetc, .3)
#
# etc_loss = loss('etc',currentetc, .1)

def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.4f}".format(abs(n))
# returns the vector containing stock data from a fixed file

def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()
	print(len(lines))

	for line in lines[48:]:
		vec.append(float(line.split(",")[6])) #HIGH
		# print('initializing 20 second moving average')
		# a = moving_20average(vec)
		# print('initializing 100 second moving average')
		# b = moving_100average(vec)
	# for ma in ma20:
	# 	vec.append(float(line.split(',')[4]))
	# for ma1, in ma100:
	# 	vec.append(float(line.split(',')[4]))

	ax1.clear()
	ax1.plot(vec)
	# ax1.plot(a)
	# ax1.plot(b)
	ax1
#     ani = animation.FuncAnimation(fig, vec, interval=1000)
	plt.show()
	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])
