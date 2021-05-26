# RL_Bitcoin_Bot

This is an attempt at building a bitcoin trading bot. For this experiment i utilized Open Gym AI custom environments and stable baselines 3 module to create the bot.

## Trading Graph

Below is a preview of the bot result after being tested on unseen data
![trading graph](/images/newplot.png)

## Dataset
The dataset used for training can be found [here](https://www.kaggle.com/mczielinski/bitcoin-historical-data). The dataset is a minute by minute historical data of bitcoin.

## Run
```python
python Bitcoin_bot.py
```

## References
[Reinforcement learning Bitcoin trading bot by Rokas Balsys](https://pylessons.com/RL-BTC-BOT-backbone/) 

## Improvements
-Better reward function
-Complete dataset
