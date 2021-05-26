#import libraries
import os
import numpy as np
import pandas as pd
import random

import gym
from gym import spaces


from collections import deque
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env

import utility

#custom bitcoin env class
class BitcoinEnv(gym.Env):
  def __init__(self, dataset_norm, dataset, initial_balance , lookback_window_size , env_steps_size):
    #initialize
    super(BitcoinEnv, self).__init__()
    self.df = dataset_norm #normalized dataaset
    self.df_original = dataset #non-normalized data for visualization
    self.df_total_steps = len(self.df)-1
    self.initial_balance = initial_balance
    self.lookback_window_size = lookback_window_size
    self.env_steps_size = env_steps_size
    self.commission = 0.00001 # commission fees
    self.columns = list(self.df.columns[:-1])
    observation_length = len(self.columns)+ 5 
   
    #define action and observation space
    self.action_space = spaces.MultiDiscrete([3, 11])
    
    self.observation_space = spaces.Box(low =-1 , high =1,shape = (self.lookback_window_size, observation_length), dtype = np.float32)

    # Orders history contains btc transactions history for the last lookback_window_size steps
    self.orders_history = deque(maxlen=self.lookback_window_size)
        
    # Market history contains the OHCL values for the last lookback_window_size prices
    self.market_history = deque(maxlen=self.lookback_window_size)
  
  
  #reset function
  def reset(self):
    self.visualization = utility.TradingGraph(Render_range=self.df_total_steps) # initialize visualization i.e trading graph
    self.trades = [] # trades list for visualization
    self.balance = self.initial_balance
    self.net_worth = self.initial_balance
    self.last_price = 0
    self.btc_held = 0
    self.btc_sold = 0
    self.btc_bought = 0
    self.last_balance = self.initial_balance
    self.last_held = 0
    #start and end step for train and test 
    if self.env_steps_size > 0: # used for training dataset
        self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - self.env_steps_size)
        self.end_step = self.start_step + self.env_steps_size
    else: # used for testing dataset
        self.start_step = self.lookback_window_size
        self.end_step = self.df_total_steps

    self.current_step = self.start_step
    
    #get data for lookback window 
    for i in reversed(range(self.lookback_window_size)):
      current_step = self.current_step - i
      #since orders history not norminalized we divide by 10000
      self.orders_history.append([self.balance/10000, self.net_worth/10000, self.btc_bought/10000, self.btc_sold/10000, self.btc_held/10000])
      self.market_history.append([self.df.loc[self.current_step, column] for column in self.columns
                                  ])
    #concatenate market and orders history which becomes state
    state = np.concatenate((self.market_history, self.orders_history), axis=1)
    return state
  
  #step function  
  def step(self, action):
    #if current step > env end step(env_step size) or networth less or = o set done true
    done = self.current_step == self.end_step or self.net_worth <= 0
    
    Date = self.df_original.loc[self.current_step, 'Date'] # for visualization
    High = self.df_original.loc[self.current_step, 'High'] # for visualization
    Low = self.df_original.loc[self.current_step, 'Low'] # for visualization
        
    self.btc_bought = 0
    self.btc_sold = 0
    
    #get action type and amount
    action_type = action[0]
    amount = (action[1]*10)/100

    # Set the current price to a weighted price
    current_price = self.df_original.loc[self.current_step, "Weighted_Price"]
    
    reward = 0#set reward to 0
       
    #if action type hold or amount 0(hold) 
    if action_type == 0 or amount ==0:
      self.balance = self.last_balance
      self.btc_held = self.last_held
      self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': 0, 'percentage':amount, 'type': "hold"})
      reward = (self.balance+(self.btc_held*current_price))-(self.last_balance+(self.last_held*self.last_price))#reward function
    #else calculate transaction btc bought,sold, balance, held etc
    elif (action_type == 1 and self.balance > 0) and amount>0:
      self.btc_bought = self.balance / current_price * amount
      self.balance -=  self.btc_bought * current_price * (1 + self.commission)
      self.btc_held += self.btc_bought
      self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.btc_bought, 'percentage':amount, 'type': "buy"})
      reward = (self.last_balance-self.balance+(self.last_held*current_price))-(self.last_balance+self.balance+(self.last_held*current_price))
    elif (action_type == 2 and self.btc_held > 0) and amount>0:
      self.btc_sold = self.btc_held * amount
      self.balance += self.btc_sold * current_price * (1-self.commission)
      self.btc_held -= self.btc_sold
      self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.btc_sold, 'percentage':amount, 'type': "sell"})
      reward = (self.last_balance+self.balance+(self.last_held*current_price))-(self.last_balance-self.balance+(self.last_held*current_price))

    else:#else if we have less or equal to 0 btc or balance -> done
      done = self.btc_held<= 0 or self.balance<=0
    
    self.net_worth = self.balance + (self.btc_held * current_price)#calculate networth
    
    #append orders history for next step
    self.orders_history.append([self.balance/10000, self.net_worth/10000, self.btc_bought/10000, self.btc_sold/10000, self.btc_held/10000])
   
    obs = self._next_observation()#get next observation ptss
    
    self.past_step = self.current_step
    
    #increment step
    self.current_step += 1
    
    self.last_price = current_price
    
    self.last_balance = self.balance
    
    self.last_held = self.btc_held

    return obs, reward, done, {}

  # Get the data points for next step
  def _next_observation(self):
    self.market_history.append([self.df.loc[self.current_step, column] for column in self.columns
                                  ])
    obs = np.concatenate((self.market_history, self.orders_history), axis=1)
    return obs

  # render environment
  def render(self, mode = "live"):
    if mode == "live":
      Date = self.df_original.loc[self.past_step, 'Date']
      Open = self.df_original.loc[self.past_step, 'Open']
      Close = self.df_original.loc[self.past_step, 'Close']
      High = self.df_original.loc[self.past_step, 'High']
      Low = self.df_original.loc[self.past_step, 'Low']
      # Render the environment to the screen
      self.visualization.render(Date, Open, High, Low, Close, self.net_worth, self.trades)

#normalize the data
def Normalizing(df_original):
    df = df_original.copy()
    column_names = df.columns.tolist()
    for column in column_names[:-1]:
        # Logging and Differencing
        test = np.log(df[column]) - np.log(df[column].shift(1))
        if test[1:].isnull().any():
            df[column] = df[column] - df[column].shift(1)
        else:
            df[column] = np.log(df[column]) - np.log(df[column].shift(1))
        # Min Max Scaler implemented
        Min = df[column].min()
        Max = df[column].max()
        df[column] = (df[column] - Min) / (Max - Min)
    return df
	
	
#import dataset
dataset = pd.read_csv("./datasets/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")	  

#portion of data for training and testing
dataset = dataset[4000000:4800000]
dataset = dataset.reset_index(drop = True)

#convert timestamp to datetime and drop unwanted columns
dataset['Date'] = [datetime.fromtimestamp(x) for x in dataset['Timestamp']]
dataset = dataset.drop([ "Volume_(Currency)", "Timestamp"], axis=1)


#darop na
dataset = dataset.dropna()

dataset_norm = Normalizing(dataset)
  
	  
#remove first row due to Nan in norminalized data
dataset = dataset[1:].reset_index()
dataset_norm = dataset_norm[1:].reset_index()
#slice dataset for train and test and drop irrelevant columns 
slice_point = int(len(dataset_norm) * (99.981/100))
train_df_norm = dataset_norm[:slice_point].drop(["index"], axis =1)
test_df_norm = dataset_norm[slice_point:].reset_index().drop([ "level_0","index"], axis =1)
train_df = dataset[:slice_point].drop(["index"], axis =1)
test_df = dataset[slice_point:].reset_index().drop([ "level_0","index"], axis =1)
	  
# It will check your custom environment and output additional warnings if needed
env = BitcoinEnv(train_df_norm, train_df, initial_balance = 1000, lookback_window_size = 31,env_steps_size = 500)
check_env(env)	  
	  
#create dummy vec env for train and test df	  
train_env = DummyVecEnv([lambda: BitcoinEnv(train_df_norm, train_df, 
                         initial_balance = 5000, lookback_window_size = 60,env_steps_size = 1500)])
test_env = DummyVecEnv([lambda: BitcoinEnv(test_df_norm, test_df, 
                        initial_balance = 5000, lookback_window_size = 60,env_steps_size = 0)])	  
	  
	  
#create instance of model for learning
model = PPO("MlpPolicy",
             train_env,
             #verbose=1, 
             #tensorboard_log="./tensorboard/"
             )
model.learn(total_timesteps=200000)#train model	  


#test model on test dataset
obs = test_env.reset()
len_test_df = len(test_df)
for i in range(len_test_df-60):
  action, _states = model.predict(obs)
  obs, rewards, done, info = test_env.step(action)
  test_env.render(mode = "live")
  
  
  
