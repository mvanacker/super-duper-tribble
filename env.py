from gym import Env
from gym.envs.registration import EnvSpec
from gym.spaces import Box
from math import inf, log
import numpy as np
import torch

from actions import actions
from indicators import rsi, stoch, hvp, trend
from position import Position
from symbol_sampler import sample

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

  def reset(self):
    self.symbol_index = (1 + self.symbol_index) % self.batch_length
    if self.symbol_index == 0:
      self.symbols, self.data = zip(*sample(self.batch_length))
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

    is_last_obs = self.obs_index == len(self.observations) - 1
    terminated = is_last_obs or self.position.has_blown_up()
    if terminated and self.position.is_open():
      self.position.close(price, step=self.obs_index)

    margin = 1+self.position.balance+self.position.total_pnl(price)
    reward = log(max(1, margin))

    return observation, reward, terminated, {
      'max_steps': len(self.df),
      'position': self.position,
    }

  def render(self):
    pass

  def close(self):
    pass
