#import libraries
import os
import numpy as np
import pandas as pd
import random


import plotly as py
import plotly.graph_objects as go
import plotly.offline as ply
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly.subplots import make_subplots


#trading graph class
class TradingGraph:
  def __init__(self, Render_range):
    #initialize varaibles
    self.Open = []
    self.Close = []
    self.High = []
    self.Low = []
    self.Date = []
    self.net_worth = []
    self.trades = []
    self.Render_range = Render_range
    
    
  def render(self, Date, Open, High, Low, Close, net_worth, trades):
    #append data to initialized variables
    self.Open.append(Open)
    self.Close.append(Close)
    self.High.append(High)
    self.Low.append(Low)
    self.Date.append(Date)
    self.net_worth.append(net_worth)
    self.trades.append(trades)
    #render environment after process completion
    if len(self.net_worth)==(self.Render_range-60):
        #make subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        #plot a candlestick graph
        fig.add_trace(go.Candlestick(x=self.Date,
                open=self.Open, high=self.High, low=self.Low,
                close=self.Close, name="Open and Close Price"),secondary_y=True, 
                     )
        #plot networth
        fig.add_trace(go.Scatter(x=self.Date, y=self.net_worth, marker=dict(color='#A2D5F2'), name="Net worth"),
               secondary_y=False)
        #initialize variables 
        buy = []
        buy_date = []
        sell = []
        sell_date = []
        hold = []
        hold_date = []
        text_sell = []
        text_buy = []
        text_hold = []
        #get action type, date and percentage
        for tradess in self.trades:
          for trade in tradess:
            if trade['type'] == 'buy':
                high_low = trade['Low']-500
                date_high_low = trade['Date']
                text = trade['percentage']
                buy.append(high_low)
                buy_date.append(date_high_low)
                text_buy.append(text)
            elif trade['type'] == 'sell':
                high_low = trade['High']+500
                date_high_low = trade['Date']
                text = trade['percentage']
                sell.append(high_low)
                sell_date.append(date_high_low)
                text_sell.append(text)
            elif trade['type'] == 'hold':
                high_low = trade['High']+500
                date_high_low = trade['Date']
                text = trade['percentage']
                hold.append(high_low)
                hold_date.append(date_high_low)
                text_hold.append(text)
        #plot action types, percentage       
        fig.add_trace(go.Scatter(x=buy_date, y=buy, mode="markers", marker=dict(color="green", size=5), marker_symbol="triangle-up", text = text_buy, name="buy"),
                   secondary_y=True)
        fig.add_trace(go.Scatter(x=sell_date, y=sell, mode="markers", marker=dict(color="red", size=5), marker_symbol="triangle-down", text = text_sell, name="sell"),
                   secondary_y=True)
        fig.add_trace(go.Scatter(x=hold_date, y=hold, mode="markers", marker=dict(color="brown", size=5), marker_symbol="circle", text = text_hold, name="hold"),
                   secondary_y=True)
        #update axis titles
        fig.update_layout(title="Bitcoin Trading Bot",
                  xaxis_title="Date",
                  yaxis_title="Balance")
        fig.update_yaxes(title="Price",secondary_y=True)
        fig.show()