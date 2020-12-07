import argparse
import asyncio
import logging
import os
import random
import IPython.display as ipd
import pandas as pd
import librosa
import keras
import librosa.display
# %pylab inline
import glob
import plotly.graph_objects as go
import cv2
from av import VideoFrame

from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import BYE, ApprtcSignaling
from web3 import Web3
import matplotlib.pyplot as plt
import pandas as pd
import hashlib, datetime,json, hashlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

market = pd.read_csv('data/crypto/crypto_portfolio/5m/bitfinex_ethusd.csv')
shares = balance
print('Current Ethereum Portfolio')
cap = market['open'] * shares 
plt.hist(cap)
plt.show()
plt.close()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import AdaBoostRegressor
a0 = market
b0 = a0['low']
c0 = a0['high']
d0 = a0['close']
e0 = a0['open']
plt.plot(b0)
plt.plot(c0)
plt.plot(d0) 
plt.plot(e0)
plt.legend(labels=['low','high','close','open'])
print('current balance when you own this amount of shares \n{}'.format(cap[-1:]))
g1 = pd.DataFrame(e0*.001) 
g1 = g1.rename(columns={"open": "1LRF"})
mb750 = pd.DataFrame(e0*.0075) 
mb750 = mb750.rename(columns={"open": ".75LRF"})
mb500 = pd.DataFrame(e0*.0050) 
mb500 = mb500.rename(columns={"open": ".5LRF"})
mb250 = pd.DataFrame(e0*.0025)  
mb250 = mb250.rename(columns={"open": ".25LRF"})
# rooms = np.random.randint(3, size=(3714)) #1=music ,2=news, 3=research
# rooms = pd.DataFrame(rooms)
# rooms.columns = ['rooms']
mini = MinMaxScaler()
bc = pd.concat([a0,g1,mb750,mb500,mb250],axis=1) 
# plt.hist(bc)
# plt.show()
bc.to_csv('Bandwidth_Capital.csv',index=False)
Xg,X7,X5,X2 = bc[['1LRF']],bc[['.75LRF']],bc[['.5LRF']],bc[['.25LRF']],
Xg,X7,X5,X2 = mini.fit_transform(Xg),mini.fit_transform(X7),mini.fit_transform(X5),mini.fit_transform(X2)
yg,y7 = bc.drop(['1LRF'],axis=1),bc.drop(['.75LRF'],axis=1)
y5,y2 = bc.drop(['.5LRF'],axis=1),bc.drop(['.25LRF'],axis=1), 
Xg_train,Xg_test,yg_train,yg_test=train_test_split(Xg,yg,test_size=.45)
X7_train,X7_test,y7_train,y7_test=train_test_split(X7,y7,test_size=.45)
X5_train,X5_test,y5_train,y5_test=train_test_split(X5,y5,test_size=.45)
X2_train,X2_test,y2_train,y2_test= train_test_split(X2,y2,test_size=.45)
print(Xg_train[-1:]*.0075)
def bc_fuk_me_rite(data):
    a = pd.read_csv(data)
    b = a['low']
    c = a['high']
    d = a['close']
    e = d = a['open']
    plt.plot(b)
    plt.plot(c)
    plt.plot(d)
    plt.plot(e)
    plt.legend(labels=['low','high','close','open'])
    avg=np.average(d)   
    avg1=np.average(b) 
    avg2=np.average(c)
    avg3=np.average(e)
    print('avg low for 1m: {}, avg high for 1m: {}, avg close for 1m: {}, avg open for 1m: {}\n\n'.format(avg,avg1,avg2,avg3))   
    return 
d=bc_fuk_me_rite('bitfinex_ethusd.csv')

avg=np.average(a0['open'])
print('avg open: {}'.format(avg))
import plotly.express as px
vo= dict(
    number=[b0, c0, d0, e0, ],
    stage=[ "Open", "High", "Low", "Close"])
fig = px.funnel(vo, x=a0['volume'], y=a0['open'])
fig.show()
to = dict(
    number=[b0, c0, d0, e0, ],
    stage=[ "Open", "High", "Low", "Close"])
fig = px.funnel(to, x=a0['time'], y=a0['open'])
fig.show()
tv = dict(
    number=[b0, c0, d0, e0,a0['volume']],
    stage=[ "Open", "High", "Low", "Close",'volume'])
fig = px.funnel(tv, x=a0['time'], y=a0['volume'])
fig.show()
fig.write_html("eth_signal.html")
background = a0['volume']
x = a0['time']
y = a0['open']
x_df = pd.DataFrame(x) 
y_df = pd.DataFrame(y) 
background_df = pd.DataFrame(background) 
x = x_df 
y = y_df 
background = background_df
extract = x.join(background) 
extract = extract.join(y)
extract 
data = extract.to_csv('extractioneth.csv',index=False) 
data = pd.read_csv('extractioneth.csv')
# data = data.drop(['Unnamed: 0'],axis=1) 
data = pd.concat([bc]) 
data = data.to_csv('anotherconcat.csv',index=False)
data = pd.read_csv('anotherconcat.csv') 
# data = data.drop(['Unnamed: 0'],axis=1)
from sklearn.decomposition import PCA, FastICA
X= data['time'] 
y = data['open'] 
background = data['volume'] 
true = True 
false = False
plt.hist2d(data['time'],data['open']) 

data = np.squeeze(np.asarray(np.matrix(data)[:,1])) 
sam_rate = np.squeeze(np.asarray(np.matrix(data)[:,0])) 
D = np.abs(librosa.stft(data))**2
S = librosa.feature.melspectrogram(data,sr=sam_rate,S=D,n_mels=128)
log_S1 = librosa.power_to_db(S,ref=np.max)

plt.figure(figsize=(12,4))
librosa.display.specshow(log_S1,sr=sam_rate,x_axis='time',y_axis='mel')
plt.title('MEL POWER SPECTOGRAM')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()

librosa.get_duration(data, sam_rate)
h_l = 256 
f_l = 512
first_iteration = pd.read_csv('bitfinex_ethusd.csv') 
time = first_iteration['time']
y= first_iteration['open']
X = first_iteration.drop(['open'],axis=1)  
reg = LinearRegression(n_jobs=-1, normalize=True ) 
reg1 = LinearRegression(n_jobs=-1, normalize=True ) 
reg2 = LinearRegression(n_jobs=-1, normalize=True ) 
reg3 = LinearRegression(n_jobs=-1, normalize=True ) 
reg4 = LinearRegression(n_jobs=-1, normalize=True ) 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.45,shuffle=False) 
reg.fit(X_train,y_train)
reg.predict(X_test[-1:]) 
X_test_time = X_train.shape[0] - X_test.shape[0]
X_test_time
import plotly.graph_objects as go
y_low= first_iteration['low']
X_low = first_iteration.drop(['low'],axis=1) 
Xl_train,Xl_test,yl_train,yl_test = train_test_split(X_low,y_low,test_size=.45,shuffle=False) 
reg1.fit(Xl_train,yl_train)
reg1.predict(Xl_test[-1:]) 
y_high= first_iteration['high']
X_high = first_iteration.drop(['high'],axis=1) 
Xh_train,Xh_test,yh_train,yh_test = train_test_split(X_high,y_high,test_size=.45,shuffle=False) 
reg2.fit(Xh_train,yh_train)
reg2.predict(Xh_test[-1:]) 
y_volume= first_iteration['volume']
X_volume = first_iteration.drop(['volume'],axis=1) 
Xv_train,Xv_test,yv_train,yv_test = train_test_split(X_volume,y_volume,test_size=.45,shuffle=False) 
reg3.fit(Xv_train,yv_train)
reg3.predict(Xv_test[-1:]) 

energy = np.array([
        sum(abs(data[i:i+f_l]**2))
        for i in range(0, len(data), h_l)
    ]) 
    
energy_r0 = np.array([
        sum(abs(reg.predict(X_test[i:i+f_l])**2))
        for i in range(0, len(reg.predict(X_test)), h_l)
    ])  

energy_r1 = np.array([
        sum(abs(reg1.predict(Xl_test[i:i+f_l])**2))
        for i in range(0, len(reg.predict(Xl_test)), h_l) 
    ])  

energy_r2 = np.array([
        sum(abs(reg2.predict(Xh_test[i:i+f_l])**2))
        for i in range(0, len(reg2.predict(Xh_test)), h_l)
    ])   
energy_r3 = np.array([
        sum(abs(reg3.predict(Xv_test[i:i+f_l])**2))
        for i in range(0, len(reg2.predict(Xv_test)), h_l)
    ])

print('predicted energy open: {}'.format(energy_r0))
print('predicted energy low: {}'.format(energy_r1))
print('predicted energy high: {}'.format(energy_r2))
print('predicted energy volume: {}'.format(energy_r3))
rmse_o = librosa.feature.rms(reg.predict(X_test), frame_length=f_l, hop_length=h_l, center=True)
rmse_l = librosa.feature.rms(reg.predict(Xl_test), frame_length=f_l, hop_length=h_l, center=True)
rmse_h = librosa.feature.rms(reg.predict(Xh_test), frame_length=f_l, hop_length=h_l, center=True)
rmse_v = librosa.feature.rms(reg.predict(Xv_test), frame_length=f_l, hop_length=h_l, center=True)
# ro = pd.DataFrame(rmse_o,columns=['RMS_O'])
# rl = pd.DataFrame(rmse_h,columns=['RMS_L'])
# rh = pd.DataFrame(rmse_l,columns=['RMS_H'])
# rv = pd.DataFrame(rmse_o,columns=['RMS_V']) 
# energy.to_csv('rmse.csv',index=False)
print('predicted root mean squared energy open: {}'.format(rmse_o))
print('predicted root mean squared energy low: {}'.format(rmse_l))
print('predicted root mean squared energy high: {}'.format(rmse_v))
print('predicted root mean squared energy volume: {}'.format(energy_r3))
frames = range(len(energy))
t = librosa.frames_to_time(frames, sr=sam_rate, hop_length=h_l) 
plt.scatter(frames,t)

sig = dict( 
    number=[b0, c0, d0, e0,a0['open']],
    stage=[ "Open", "High", "Low", "Close",])
fig = px.funnel(data, x=t, y=frames)
fig.show()


def strip(x, frame_length, hop_length):

    # Compute RMSE.
    rmse = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)
    
    # Identify the first frame index where RMSE exceeds a threshold.
    thresh = 0.01
    frame_index = 0
    while rmse[0][frame_index] < thresh:
        frame_index += 1
        
    # Convert units of frames to samples.
    start_sample_index = librosa.frames_to_samples(frame_index, hop_length=hop_length)
    
    # Return the trimmed signal.
    return x[start_sample_index:]
y = strip(reg.predict(X_test), f_l, h_l)
plt.plot(y)
yl = strip(reg1.predict(Xl_test), f_l, h_l)
plt.plot(yl)
yh = strip(reg2.predict(Xh_test), f_l, h_l)
plt.plot(yh)
yv = strip(reg3.predict(Xv_test), f_l, h_l)
plt.plot(yv)
fig = go.Figure(
    data=[go.Scatter(x=[[sam_rate*.45]], y=[[yv_test]])],
    layout=go.Layout(
        xaxis=dict(range=[sam_rate.min()*.45, sam_rate.max()*.45], autorange=True),
        yaxis=dict(range=[yv_test.min(),yv_test.max()], autorange=True),
        title="Locating Open Recieving Signals",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    
    frames=[
            go.Frame(data=[go.Scatter(x=[time[166:-1]], y=[[Xv_test[0:-1]]])]), 
            go.Frame(data=[go.Scatter(x=[time[176:-1]], y=[[Xv_test[10:-1]]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:-1]], y=[[Xv_test[20:-1]]])]),
            go.Frame(data=[go.Scatter(x=[time[193:-1]], y=[[Xv_test[27:-1]]])]),
            go.Frame(data=[go.Scatter(x=[time[171:-1]], y=[[reg.predict(Xv_test[5:-1])]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:-1]], y=[[reg.predict(Xv_test[10:-1])]])]), 
            go.Frame(data=[go.Scatter(x=[time[191:-1]], y=[[reg.predict(Xv_test[15:-1])]])]), 
            go.Frame(data=[go.Scatter(x=[time[196:-1]], y=[[reg.predict(Xv_test[20:-1])]])]),     
            go.Frame(data=[go.Scatter(x=[time[196:-1]], y=[[Xv_test[196:-1]]])],
                     
                     
                     layout=go.Layout(title_text="End of Signals"))]
    
)


fig.show()

fig = go.Figure(
    data=[go.Scatter(x=[[sam_rate]], y=[[yl_test]])],
    layout=go.Layout(
        xaxis=dict(range=[sam_rate.min(), sam_rate.max()], autorange=True),
        yaxis=dict(range=[yl_test.min(),yl_test.max()], autorange=True),
        title="Transmitting Buy Signals",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    
    frames=[
            go.Frame(data=[go.Scatter(x=[time[166:176]], y=[[Xl_test[0:10]]])]), 
            go.Frame(data=[go.Scatter(x=[time[176:186]], y=[[Xl_test[10:20]]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:193]], y=[[Xl_test[20:27]]])]),
            go.Frame(data=[go.Scatter(x=[time[193:195]], y=[[Xl_test[27:29]]])]),
            go.Frame(data=[go.Scatter(x=[time[171:181]], y=[[reg1.predict(X_test[5:10])]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:191]], y=[[reg1.predict(X_test[10:15])]])]), 
            go.Frame(data=[go.Scatter(x=[time[191:196]], y=[[reg1.predict(X_test[15:20])]])]), 
            go.Frame(data=[go.Scatter(x=[time[196:201]], y=[[reg1.predict(X_test[20:25])]])]),
#             go.Frame(data=[go.Scatter(x=[a1[['time']][79:85]], y=[[reg.predict(X_test[79:201])]])]),
#             go.Frame(data=[go.Scatter(x=[a1[['time']][85:92]], y=[[reg.predict(X_test[201:207])]])]),        
            go.Frame(data=[go.Scatter(x=[time[201:-1]], y=[[reg1.predict(Xl_test[25:-1])]])],
                     
                     
                     layout=go.Layout(title_text="End of Receiving Low Signal"))]
    
)


fig.show()
fig.write_html("fileeth.html")


fig = go.Figure(
    data=[go.Scatter(x=[[sam_rate]], y=[[yh_test]])],
    layout=go.Layout(
        xaxis=dict(range=[sam_rate.min(), sam_rate.max()], autorange=True),
        yaxis=dict(range=[yh_test.min(),yh_test.max()], autorange=True),
        title="Transmitting  high,open,low Signals",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    
    frames=[
            go.Frame(data=[go.Scatter(x=[time[166:176]], y=[[X_test[0:10]]])]), 
            go.Frame(data=[go.Scatter(x=[time[176:186]], y=[[X_test[10:20]]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:193]], y=[[X_test[20:27]]])]),
            go.Frame(data=[go.Scatter(x=[time[193:195]], y=[[X_test[27:29]]])]),
            go.Frame(data=[go.Scatter(x=[time[166:176]], y=[[Xl_test[0:10]]])]), 
            go.Frame(data=[go.Scatter(x=[time[176:186]], y=[[Xl_test[10:20]]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:193]], y=[[Xl_test[20:27]]])]),
            go.Frame(data=[go.Scatter(x=[time[193:195]], y=[[Xl_test[27:29]]])]),
            go.Frame(data=[go.Scatter(x=[time[166:176]], y=[[Xh_test[0:10]]])]), 
            go.Frame(data=[go.Scatter(x=[time[176:186]], y=[[Xh_test[10:20]]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:193]], y=[[Xh_test[20:27]]])]),
            go.Frame(data=[go.Scatter(x=[time[193:195]], y=[[Xh_test[27:29]]])]),
            go.Frame(data=[go.Scatter(x=[time[171:181]], y=[[reg.predict(X_test[5:10])]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:191]], y=[[reg.predict(X_test[10:15])]])]), 
            go.Frame(data=[go.Scatter(x=[time[191:196]], y=[[reg.predict(X_test[15:20])]])]), 
            go.Frame(data=[go.Scatter(x=[time[196:201]], y=[[reg.predict(X_test[20:25])]])]),
            go.Frame(data=[go.Scatter(x=[time[171:181]], y=[[reg1.predict(Xl_test[5:10])]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:191]], y=[[reg1.predict(Xl_test[10:15])]])]), 
            go.Frame(data=[go.Scatter(x=[time[191:196]], y=[[reg1.predict(Xl_test[15:20])]])]), 
            go.Frame(data=[go.Scatter(x=[time[196:201]], y=[[reg1.predict(Xl_test[20:25])]])]),
            go.Frame(data=[go.Scatter(x=[time[171:181]], y=[[reg2.predict(Xh_test[5:10])]])]), 
            go.Frame(data=[go.Scatter(x=[time[186:191]], y=[[reg2.predict(Xh_test[10:15])]])]), 
            go.Frame(data=[go.Scatter(x=[time[191:196]], y=[[reg2.predict(Xh_test[15:20])]])]), 
            go.Frame(data=[go.Scatter(x=[time[196:201]], y=[[reg2.predict(Xh_test[20:25])]])]),
#             go.Frame(data=[go.Scatter(x=[a1[['time']][79:85]], y=[[reg.predict(X_test[79:201])]])]),
#             go.Frame(data=[go.Scatter(x=[a1[['time']][85:92]], y=[[reg.predict(X_test[201:207])]])]),        
            go.Frame(data=[go.Scatter(x=[time[201:-1]], y=[[reg2.predict(Xh_test[25:-1])]])],
                     
                     
                     layout=go.Layout(title_text="End of Predicting All Received Signals based on high,open,low"))]
    
)


fig.show()
print('predicted market cap per signal {}'.format((reg.predict(X_test[0:10])*background[0:10])))
print('predicted market cap per signal {}'.format((reg.predict(X_test[10:20])*background[10:20])))
print('predicted market cap per signal {}'.format((reg.predict(X_test[20:27])*background[20:27]))) 
print('predicted market cap per signal {}'.format((reg.predict(X_test[27:29])*background[27:29])))
import os

if not os.path.exists("images"):
    os.mkdir("images")

reg2.predict(Xh_test[25:-1])
fig.write_html("high_low_open.html")
reg.predict(X_test[25:-1])
reg1.predict(Xl_test[25:-1])
plt.hist2d(yl_test[25:-1],reg1.predict(Xl_test[25:-1]))
plt.hist2d(yh_test[25:-1],reg1.predict(Xh_test[25:-1]))
plt.hist2d(y_test[25:-1],reg.predict(X_test[25:-1]))
abi = [{"constant":true,"inputs":[],"name":"mintingFinished","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_from","type":"address"},{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transferFrom","outputs":[],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[],"name":"unpause","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_amount","type":"uint256"}],"name":"mint","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"paused","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[],"name":"finishMinting","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":false,"inputs":[],"name":"pause","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"owner","outputs":[{"name":"","type":"address"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_amount","type":"uint256"},{"name":"_releaseTime","type":"uint256"}],"name":"mintTimelocked","outputs":[{"name":"","type":"address"}],"payable":false,"type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"remaining","type":"uint256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"payable":false,"type":"function"},{"anonymous":false,"inputs":[{"indexed":true,"name":"to","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Mint","type":"event"},{"anonymous":false,"inputs":[],"name":"MintFinished","type":"event"},{"anonymous":false,"inputs":[],"name":"Pause","type":"event"},{"anonymous":false,"inputs":[],"name":"Unpause","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"name":"owner","type":"address"},{"indexed":true,"name":"spender","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"name":"from","type":"address"},{"indexed":true,"name":"to","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Transfer","type":"event"}]
print('Using OMG token/OMG Wallet Address("0xd26114cd6EE289AccF82350c8d8487fedB8A0C07") for simulation based on ERC20 protocols.')
address = "0xd26114cd6EE289AccF82350c8d8487fedB8A0C07"
print("creating OMG token smart contract representation.")
contract = web3.eth.contract(address=address, abi=abi)
totalSupply = contract.functions.totalSupply().call()
print('total supply of {} {}'.format(contract.functions.name().call(),web3.fromWei(totalSupply, 'ether')))
# print(contract.functions.symbol().call())
balance = contract.functions.balanceOf(address).call()
print('total supply of OMG tokens{}'.format(web3.fromWei(balance, 'ether')))
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
web3.eth.defaultAccount = web3.eth.accounts[0]
account_1 = '0xBF309c4DFf319F65dAa6492dc8Bb2ca728dEdAeB'
account_2 = '0x0c8F7deA9f1E4c876ad600554E4b56bf8Fba46c5'
private_key = ''
