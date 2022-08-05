from gym import Env
from gym.envs.registration import EnvSpec
from gym.spaces import Box
from math import inf, log
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import torch

from actions import actions
from indicators import rsi, stoch, hvp, trend
from position import Position

CUTOFF = 109

class Market(Env):
  metadata = {'render.modes': ['human']}
  spec = EnvSpec("MarketEnv-v0")

  def __init__(self, batch_length, num_inputs, num_outputs):
    self.action_space = Box(low=0.0, high=1.0, shape=(num_outputs,), dtype=np.float64)
    self.observation_space = Box(low=-inf, high=inf, shape=(num_inputs,), dtype=np.float64)

    self.batch_length = batch_length
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs

    self.symbol_index = -1
    self.position = Position()

  def _random_symbols(self):
    today = pd.Timestamp.today().date()
    two_weeks = pd.Timedelta(days=14)
    four_years = pd.Timedelta(weeks=4*52)
    
    def is_active(data):
      return today - data.iloc[-1].name.date() < two_weeks
    def is_mature(data):
      return today - data.iloc[0].name.date() > four_years

    symbols = pdr.data.get_nasdaq_symbols()
    sample = []
    while len(sample) < self.batch_length:
      symbol = symbols.sample().index[0]
      try:
        data = pdr.get_data_yahoo(symbol)
        if not is_active(data):
          print(f'skipping {symbol} (inactive, last={data.iloc[-1].name.date()})', end=' ')
        elif not is_mature(data):
          print(f'skipping {symbol} (immature, len={len(data)})', end=' ')
        else:
          sample.append((symbol, data))
          print(f'added {symbol} ({len(sample)}/{self.batch_length})', end=' ')
      except Exception as e:
        print(f'skipping {symbol} (exception)', end=' ')
    print('\n')
    return sample

  def reset(self):
    self.symbol_index = (1 + self.symbol_index) % self.batch_length
    if self.symbol_index == 0:
      self.symbols, self.data = zip(*self._random_symbols())
      for df in self.data: # data is [pd.DataFrame]
        del df['Adj Close']
        df.loc[:, 'RSI']   = rsi(df['Close'])
        df.loc[:, 'Stoch'] = stoch(df['Close'])
        df.loc[:, 'HVP']   = hvp(df['Close'])
        df.loc[:, 'Trend'] = trend(df['High'], df['Low'], df['Close'])
    
    self.df = self.data[self.symbol_index][CUTOFF:]
    win_len = self.num_inputs // self.df.shape[1]
    t = torch.Tensor(self.df.values).unfold(0, win_len, 1)#.transpose(1, 2)
    self.observations = torch.flatten(t, 1).reshape((len(t), 1, self.num_inputs))
    self.obs_index = 0
    self.position.reset()
    return self.observations[0]

  def step(self, policy):
    price = self.df.iloc[self.obs_index]['Close']
    
    if not np.any(np.isnan(policy)):
      policy = policy / np.sum(policy)
      action_index = np.random.choice(self.num_outputs, p=policy)
      action = actions[action_index]

      size = action.win/action.stop_loss - (1-action.win)/action.take_profit # kelly criterion
      if not self.position.is_open():
        self.position.open(price, size, action.stop_loss, action.take_profit, step=self.obs_index)
      # elif self.position.size > 0.0 and (size < 0.0 or (price < self.position.stop_loss or price > self.position.take_profit)) \
      #   or self.position.size < 0.0 and (size > 0.0 or (price > self.position.stop_loss or price < self.position.take_profit)):
      elif self.position.size > 0.0 and (price < self.position.stop_loss or price > self.position.take_profit) \
        or self.position.size < 0.0 and (price > self.position.stop_loss or price < self.position.take_profit):
        self.position.close(price, step=self.obs_index)

    self.obs_index += 1
    observation = self.observations[self.obs_index]
    reward = log(1+self.position.balance+self.position.total_pnl(price))

    is_last_obs = self.obs_index == len(self.observations) - 1
    terminated = is_last_obs or self.position.has_blown_up()
    if terminated and self.position.is_open():
      self.position.close(price, step=self.obs_index)

    return observation, reward, terminated, {'max_steps': len(self.df),}

  def render(self):
    pass

  def close(self):
    pass
