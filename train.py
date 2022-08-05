#!/usr/bin/env python3

# silence TF warnings for now
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from actions import actions
from env import Market

LEARNING_RATE = 0.001
LENGTH_BATCH  = 100
NUM_INPUTS    = 9*1
NUM_HIDDEN    = 512
NUM_OUTPUTS   = len(actions)

BOUND_MIN     = 0.0
PERCENTILE    = 70

class Model(nn.Module):
  def __init__(self, num_inputs, num_hidden, num_outputs):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(num_inputs, num_hidden),
      nn.ReLU(),
      nn.Linear(num_hidden, num_outputs)
    )

  def forward(self, x):
    return self.net(x)

def batches(env, net, length):
  batch = []
  reward = 0.0
  steps = []
  obs_t = env.reset()
  sm = nn.Softmax(dim=1)
  while True:
    policy = sm(net(obs_t)).data[0].numpy()
    steps.append((obs_t[0].numpy(), policy))
    
    obs_t, reward_step, terminated, info = env.step(policy)
    reward += reward_step * len(steps) / info['max_steps']

    if terminated:
      batch.append((reward, steps))
      print(f'Finished episode {len(batch)}, reward={reward} len={len(steps)}\n')

      if len(batch) == length:
        yield batch
        batch = []
      reward = 0.0
      steps = []
      obs_t = env.reset()

def main(net):
  # torch.device("cpu")
  env = Market(LENGTH_BATCH, NUM_INPUTS, NUM_OUTPUTS)
  # env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
  loss_func = nn.CrossEntropyLoss()
  optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

  print(f'Starting...')
  with SummaryWriter(comment="-indicator") as writer:
    for b, batch in enumerate(batches(env, net, LENGTH_BATCH)):
      rewards, _ = zip(*batch)
      mean = float(np.mean(rewards))

      # bound = max(BOUND_MIN, np.percentile(rewards, PERCENTILE))
      bound = np.percentile(rewards, PERCENTILE)
      train_in = []
      train_out = []
      for reward, steps in batch:
        if reward >= bound:
          obs, act = zip(*steps)
          train_in.append(obs)
          train_out.append(act)

      if len(train_in):
        train_in = np.concatenate(train_in)
        train_out = np.concatenate(train_out)

        if not (np.any(np.isnan(train_in)) or np.any(np.isnan(train_out))):
          train_in_t = torch.FloatTensor(train_in)
          train_out_t = torch.FloatTensor(train_out)

          optimizer.zero_grad()
          output_t = net(train_in_t)
          loss_t = loss_func(output_t, train_out_t)
          loss_t.backward()
          optimizer.step()
          loss = loss_t.item()

          # target = 0.0
          # for df in env.data:
          #   prices = df['Close']
          #   min_price = np.min(prices)
          #   target += (np.max(prices) - min_price) / min_price
          # target *= 0.25

          # print(f"[BATCH {b}] loss={loss:.3f}, µ={mean:.1f}, bound={bound:.1f}, target={target}\n")
          print(f"[BATCH {b}] loss={loss:.3f}, µ={mean:.1f}, bound={bound:.1f}\n")
          writer.add_scalar("loss", loss, b)
          writer.add_scalar("bound", bound, b)
          writer.add_scalar("mean", mean, b)
          writer.flush()

          # if mean > target:
          #   break

def save(model):
  t = datetime.now()
  path = f'wnb-{t.year}-{t.month:0>2}-{t.day:0>2}-{t.hour:0>2}-{t.minute:0>2}'
  torch.save(model.state_dict(), path)

if __name__ == "__main__":
  try:
    model = Model(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
    main(model)
  except:
    save(model)
    exit(0)
