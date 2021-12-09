from agent2.agent import Agent
from functions import *
import sys
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import csv
import datetime
os.environ['KERAS_BACKEND' ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from matplotlib import style
import pandas as pd
import asyncio
import logging
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
import webbrowser
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import BYE, ApprtcSignaling
from web3 import Web3
# import matplotlib.pyplot as plt
# import pandas as pd
# import hashlib, datetime,json, hashlib
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# infura_url = "https://mainnet.infura.io/v3/89f69d97c5c44c35959cc4d15c0f0531"
# web3 = Web3(Web3.HTTPProvider(infura_url))
# print(web3.isConnected())
# print(web3.eth.blockNumber)
# # account = '0xF0Cab2c1b9E289D188A4994F98801118448cCB34'
# balance1 = web3.eth.getBalance(account)
# print(web3.fromWei(balance1, "ether"))
# market = pd.read_csv('data/anotherconcat.csv')
# shares1 = balance1
# print('Current Ethereum Portfolio')
# cap = market['open'] * shares1 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import AdaBoostRegressor
true = True 
false = False
#DISCOUNT RETURN EX:
from agent2.agent import Agent
from functions import *
import sys
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import csv
import datetime
os.environ['KERAS_BACKEND' ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from matplotlib import style
import pandas as pd
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
infura_url = "https://ropsten.infura.io/v3/89f69d97c5c44c35959cc4d15c0f0531"
web3 = Web3(Web3.HTTPProvider(infura_url))
print(web3.isConnected())
print(web3.eth.blockNumber)
web3.eth.defaultAccount,web3.eth.acct = input('enter Ethereum address for house'),input('Confirm Ethereum address for house')
account = str(web3.eth.acct)
stake_acct='0xEa9Aafc2f6bc5c82BF3652909f26455b0f39dBFa'
web3.eth.stake_acct=stake_acct
balance1 = web3.eth.getBalance(account)
print(web3.fromWei(balance1, "ether"))
market = pd.read_csv('data/anotherconcat.csv')
shares1 = balance1
print('Current Ethereum Portfolio')
cap = market['open'] * shares1 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import AdaBoostRegressor
true = True 
false = False
abi = [{"constant":true,"inputs":[],"name":"mintingFinished","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_from","type":"address"},{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transferFrom","outputs":[],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[],"name":"unpause","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_amount","type":"uint256"}],"name":"mint","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"paused","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[],"name":"finishMinting","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":false,"inputs":[],"name":"pause","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"owner","outputs":[{"name":"","type":"address"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_amount","type":"uint256"},{"name":"_releaseTime","type":"uint256"}],"name":"mintTimelocked","outputs":[{"name":"","type":"address"}],"payable":false,"type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"remaining","type":"uint256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"payable":false,"type":"function"},{"anonymous":false,"inputs":[{"indexed":true,"name":"to","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Mint","type":"event"},{"anonymous":false,"inputs":[],"name":"MintFinished","type":"event"},{"anonymous":false,"inputs":[],"name":"Pause","type":"event"},{"anonymous":false,"inputs":[],"name":"Unpause","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"name":"owner","type":"address"},{"indexed":true,"name":"spender","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"name":"from","type":"address"},{"indexed":true,"name":"to","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Transfer","type":"event"}]
address = "0xd26114cd6EE289AccF82350c8d8487fedB8A0C07"
contract = web3.eth.contract(address=address, abi=abi)
# totalSupply = contract.functions.totalSupply().call()
# print('total supply of {} {}'.format(contract.functions.name().call(),web3.fromWei(totalSupply, 'ether')))
# print(contract.functions.symbol().call())
# balance = contract.functions.balanceOf(address).call()
# print('total supply of OMG tokens{}'.format(web3.fromWei(balance, 'ether')))
# ganache_url = "http://127.0.0.1:7545"
# web3 = Web3(Web3.HTTPProvider(ganache_url))
# web3.eth.defaultAccount = web3.eth.accounts[8]
# web3.eth.defaultStakeAccount = web3.eth.accounts[7]
private_key = input('House, Enter private key')

players= input('how many players ?')
if players > '5':
    print('too many players ,**cough cough covid i"m dying gn..') 
if players == '1':
    print('enter ethereum address , ily its all fun and games')
import gym
env = gym.make('Blackjack-v0')
obs_space = env.observation_space
act_space = env.action_space
print(obs_space)
print(act_space)
print('AGENT HAS {} POTENTIAL ACTIONS'.format(act_space))
print('LETS PLAY 21 ROUNDS WITH THIS POLICY AGENT ^__^')
def generate_episode_from_limit(bljck_env):
    episode = []
    state = bljck_env.reset()
    while True:
        action = 0 if state[0] >18 else 1
        next_state, reward, done, info = bljck_env.step(action)
        episode.append((state,action,reward))
        state = next_state
        if done:
            print('THE GAME IS OVER !', reward)
            print('PLAYER WON THE GAME ^____^ \n') if reward >0 else print('THE AGENT WONT THE GAME  X__x')
            break
    return episode
for i in range(21):
    print(generate_episode_from_limit(env))
from keras.preprocessing import  image

image.load_img('pseudotmp3.png')
from collections import defaultdict
import numpy as np
import sys
act_space = env.action_space
def monte(env,num_episodes, gen_ep, gamma=.37):
    if players > '5':
        print('too many players Still in development **cough cough covid i"m dying gn..') 
    if players == '1':
        p1 = input('enter metamask ethereum address , ily its all fun and games')
        web3.eth.p1 = p1
        p1_key =input('enter private key from metamask')
        bid = float(input('how much ethereum would you like to bid'))
        stake = float(input('how much ethereum would you like to stake from the bid ?'))
        returns = defaultdict(list)
        for i_episode in range(1,num_episodes+1):
            if i_episode % 1000 == 0:
                print('\rEpiseode {}/{}'.format(i_episode,num_episodes), end=' ')
                sys.stdout.flush()

            num_episodes = 10000
            for i_episode in range(num_episodes):
                done = False
                s_0 = env.reset()
                gen_ep = []

                state = [s_0]
                while done == False:
                    # Implement policy
                    if s_0[0] >= 18:
                        s_1, reward, done, info = env.step(0)
                    else:
                        s_1, reward,done, info = env.step(1)

                    gen_ep.append((reward*gamma,state))
                    returns.get(reward)
                    print(gen_ep,'\n')
                    state.append((s_0,reward))

                    if done == True and reward>0*gamma:
                        print('done \n',reward*gamma)
                        print('PLAYER WON THE GAME ^____^ \n')
                        print('PAYING OUT TO PLAYERS')
                        print('Current Balance {}'.format(shares1))
                        nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
                        tx = {
                            'nonce': nonce,
                            'to': p1,
                            'value': web3.toWei((bid+stake)+.03, 'ether'),
                            'gas': 21000,
                            'gasPrice': web3.toWei('45', 'gwei'),               
                        }
                        print(tx)
                        signed_tx = web3.eth.account.signTransaction(tx, private_key)
                        print(signed_tx)            
                        tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)                       
                        

                    else:
                        print('agent won:',reward*gamma)
                        print('paying out stakes to dAIsy')
                        nonce = web3.eth.getTransactionCount(web3.eth.p1)             
                        tx = {
                            'nonce': nonce,
                            'to': account,
                            'value': web3.toWei(stake+.03, 'ether'),
                            'gas': 21000,
                            'gasPrice': web3.toWei('45', 'gwei'),               
                        }
                        print(tx)
                        signed_tx = web3.eth.account.signTransaction(tx, p1_key)
                        print(signed_tx)            
                        tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
                        break
                        continue
                        for s_i, s in i_episode(state[:-1]):
                            Vs = np.average(gen_ep[s_i:])*gamma
                            print('Vs:',Vs)
                            print('Current Balance {}'.format(shares1))
                        break
                    
            return        

    if players == '2':
        p1 =input('enter ethereum address , ily its all fun and games')  
        web3.eth.p1 = p1
        p1_key = input('player 1, enter metamask private key')
        p1_bid = float(input('p1 how much ethereum would you like to bid'))
        p1_stake = float(input('how much ethereum would you like to stake from the bid ?'))
        p2 =input('enter ethereum address , ily its all fun and games')
        web3.eth.p2 = p2
        p2_key = input('player 2, enter metamask private key')
        p2_bid = float(input('p2 how much ethereum would you like to bid'))
        p2_stake = float(input('p2 how much ethereum would you like to stake from the bid ?'))        
        returns = defaultdict(list)
        for i_episode in range(1,num_episodes+1):
            if i_episode % 1000 == 0:
                print('\rEpiseode {}/{}'.format(i_episode,num_episodes), end=' ')
                sys.stdout.flush()

            num_episodes = 10000
            for i_episode in range(num_episodes):
                done = False
                s_0 = env.reset()
                gen_ep = []

                state = [s_0]
                while done == False:
                    # Implement policy
                    if s_0[0] >= 18:
                        s_1, reward, done, info = env.step(0)
                    else:
                        s_1, reward,done, info = env.step(1)

                    gen_ep.append((reward*gamma,state))
                    returns.get(reward)
                    print(gen_ep,'\n')
                    state.append((s_0,reward))

                    if done == True and reward>0*gamma:
                        print('done \n',reward*gamma)
                        print('PLAYER WON THE GAME ^____^ \n')
                        print('PAYING OUT TO PLAYERS')
                        print('Current Balance {}'.format(shares1))
                        nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
                        tx = {
                            'nonce': nonce+7**4+1,
                            'to': p1,
                            'value': web3.toWei((p1_bid+p1_stake)+.003, 'ether'),
                            'gas': 500000,
                            'gasPrice': web3.toWei('50', 'gwei'),               
                        }
                        print(tx)
                        signed_tx = web3.eth.account.signTransaction(tx, private_key)
                        print(signed_tx)            
                        tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
                        
                        nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
                        tx = {
                            'nonce': nonce+7**4+2+1,
                            'to': p2,
                            'value': web3.toWei((p2_bid+p2_stake)+.003, 'ether'),
                            'gas': 500000,
                            'gasPrice': web3.toWei('50', 'gwei'),               
                        }
                        print(tx)
                        signed_tx = web3.eth.account.signTransaction(tx, p2_key)
                        print(signed_tx)            
                        tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
                        print('Current Balance {}'.format(shares1))
                        print('THE GAME IS OVER !', reward)
                            
                    else:
                        print('agent won:',reward*gamma)
                        nonce = web3.eth.getTransactionCount(web3.eth.p1)             
                        tx = {
                            'nonce': nonce+7**4+3+1,
                            'to': account,
                            'value': web3.toWei((p1_bid+p1_stake)+.001, 'ether'),
                            'gas': 500000,
                            'gasPrice': web3.toWei('35', 'gwei'),               
                        }
                        print(tx)
                        signed_tx = web3.eth.account.signTransaction(tx, p1_key)
                        print(signed_tx)            
                        tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
                        print('Current Balance {}'.format(shares1))
                        nonce = web3.eth.getTransactionCount(web3.eth.p2)             
                        tx = {
                            'nonce': nonce+7**4+4+1,
                            'to': account,
                            'value': web3.toWei((p2_bid+p2_stake)+.001, 'ether'),
                            'gas': 500000,
                            'gasPrice': web3.toWei('35', 'gwei'),               
                        }
                        print(tx)
                        signed_tx = web3.eth.account.signTransaction(tx, p2_key)
                        print(signed_tx)            
                        tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction) 
#                         print('Current Balance {}'.format(shares1))
                        break
                        continue
                        for s_i, s in i_episode(state[:-1]):
                            Vs = np.average(gen_ep[s_i:])*gamma
                            print('Vs:',Vs)
                            print('Current Balance {}'.format(shares1))
                        break
                    
            return        
        
        
#     if players == '3':
#         p1 =input('enter ethereum address , ily its all fun and games')
#         p2 =input('enter ethereum address , ily its all fun and games')
#         p3 =input('enter ethereum address , ily its all fun and games')
#         returns = defaultdict(list)
#         for i_episode in range(1,num_episodes+1):
#             if i_episode % 1000 == 0:
#                 print('\rEpiseode {}/{}'.format(i_episode,num_episodes), end=' ')
#                 sys.stdout.flush()

#             num_episodes = 10000
#             for i_episode in range(num_episodes):
#                 done = False
#                 s_0 = env.reset()
#                 gen_ep = []

#                 state = [s_0]
#                 while done == False:
#                     # Implement policy
#                     if s_0[0] >= 18:
#                         s_1, reward, done, info = env.step(0)
#                     else:
#                         s_1, reward,done, info = env.step(1)

#                     gen_ep.append((reward*gamma,state))
#                     returns.get(reward)
#                     print(gen_ep,'\n')
#                     state.append((s_0,reward))

#                     if done == True and reward>0*gamma: 
#                         print('done \n',reward*gamma)
#                         print('PLAYER WON THE GAME ^____^ \n')
#                         print('PAYING OUT TO PLAYERS')
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)   
#                         print('Current Balance {}'.format(shares1))
#                         print('THE GAME IS OVER !', reward)
#                     else:
#                         print('agent won:',reward*gamma)
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p1,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p2,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p3,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)  
#                         print('Current Balance {}'.format(shares1))
#                         break
#                         continue
#                         for s_i, s in i_episode(state[:-1]):
#                             Vs = np.average(gen_ep[s_i:])*gamma
#                             print('Vs:',Vs)
#                         break
                    
#             return                
        
#     if players == '4':
#         p1 =input('enter ethereum address , ily its all fun and games')
#         p2 =input('enter ethereum address , ily its all fun and games')
#         p3 =input('enter ethereum address , ily its all fun and games')
#         p4 =input('enter ethereum address , ily its all fun and games')
#         returns = defaultdict(list)
#         for i_episode in range(1,num_episodes+1):
#             if i_episode % 1000 == 0:
#                 print('\rEpiseode {}/{}'.format(i_episode,num_episodes), end=' ')
#                 sys.stdout.flush()



#             num_episodes = 10000
#             for i_episode in range(num_episodes):
#                 done = False
#                 s_0 = env.reset()
#                 gen_ep = []

#                 state = [s_0]
#                 while done == False:
#                     # Implement policy
#                     if s_0[0] >= 18:
#                         s_1, reward, done, info = env.step(0)
#                     else:
#                         s_1, reward,done, info = env.step(1)

#                     gen_ep.append((reward*gamma,state))
#                     returns.get(reward)
#                     print(gen_ep,'\n')
#                     state.append((s_0,reward))

#                     if done == True and reward>0*gamma:
#                         print('done \n',reward*gamma)
#                         print('PLAYER WON THE GAME ^____^ \n')
#                         print('PAYING OUT TO PLAYERS')
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)  
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print('Current Balance {}'.format(shares1))
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)   
#                         print('THE GAME IS OVER !', reward)
#                     else:
#                         print('agent won:',reward*gamma)
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p1,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p2,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p3,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)  
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p4,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)  
#                         print('Current Balance {}'.format(shares1))
#                         break
#                         continue
#                         for s_i, s in i_episode(state[:-1]):
#                             Vs = np.average(gen_ep[s_i:])*gamma
#                             print('Vs:',Vs)
#                             print('Current Balance {}'.format(shares1))
#                         break
                    
#             return        
        
        
#     if players == '5':
#         p1 =input('enter ethereum address , ily its all fun and games')  
#         p2 =input('enter ethereum address , ily its all fun and games')
#         p3 =input('enter ethereum address , ily its all fun and games')
#         p4 =input('enter ethereum address , ily its all fun and games')
#         p5 =input('enter ethereum address , ily its all fun and games')
#         returns = defaultdict(list)
#         for i_episode in range(1,num_episodes+1):
#             if i_episode % 1000 == 0:
#                 print('\rEpiseode {}/{}'.format(i_episode,num_episodes), end=' ')
#                 sys.stdout.flush()



#             num_episodes = 10000
#             for i_episode in range(num_episodes):
#                 done = False
#                 s_0 = env.reset()
#                 gen_ep = []

#                 state = [s_0]
#                 while done == False:
#                     # Implement policy
#                     if s_0[0] >= 18:
#                         s_1, reward, done, info = env.step(0)
#                     else:
#                         s_1, reward,done, info = env.step(1)

#                     gen_ep.append((reward*gamma,state))
#                     returns.get(reward)
#                     print(gen_ep,'\n')
#                     state.append((s_0,reward))

#                     if done == True and reward>0*gamma:
#                         print('done \n',reward*gamma)
#                         print('PLAYER WON THE GAME ^____^ \n')
#                         print('PAYING OUT TO PLAYERS')
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)   
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)   
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)   
#                         tx = {
#                             'nonce': nonce,
#                             'to': web3.eth.defaultAccount,
#                             'value': web3.toWei(.0003, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)  
#                         print('Current Balance {}'.format(shares1))
#                         print('THE GAME IS OVER !', reward)
#                     else:
#                         print('agent won:',reward*gamma)
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p1,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p2,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p3,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction) 
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p4,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction) 
#                         print('Current Balance {}'.format(shares1))
#                         nonce = web3.eth.getTransactionCount(web3.eth.defaultAccount)             
#                         tx = {
#                             'nonce': nonce,
#                             'to': p5,
#                             'value': web3.toWei(.00045, 'ether'),
#                             'gas': 2000000,
#                             'gasPrice': web3.toWei('50', 'gwei'),               
#                         }
#                         print(tx)
#                         signed_tx = web3.eth.account.signTransaction(tx, private_key)
#                         print(signed_tx)            
#                         tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)  
#                         print('Current Balance {}'.format(shares1))
#                         break
#                         continue
#                         for s_i, s in i_episode(state[:-1]):
#                             Vs = np.average(gen_ep[s_i:])*gamma
#                             print('Vs:',Vs)
#                             print('Current Balance {}'.format(shares1))
#                         break
                    
#             return 
                

print(monte(env,1000,gen_ep=generate_episode_from_limit))
