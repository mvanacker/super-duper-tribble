#!/usr/bin/env python3

# silence TF warnings for now
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorboardX import SummaryWriter
from time import sleep
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from actions import actions
from env import Market
from model import Model

LEARNING_RATE = 0.001
LENGTH_BATCH  = 100
NUM_INPUTS    = 9*1
NUM_HIDDEN    = 512
NUM_OUTPUTS   = len(actions)

BOUND_MIN     = 400.0
BEST_MAX      = 500
PERCENTILE    = 80

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
    reward += reward_step

    if terminated:
      reward *= len(steps) / info['max_steps']
      batch.append((reward, steps))
      print(f'Finished episode {len(batch):0>3}, len={len(steps):0>4}, reward={reward:.2f} (pnl={info["position"].pnl:.2f})')

      if len(batch) == length:
        yield batch
        batch = []
      reward = 0.0
      steps = []
      obs_t = env.reset()

def train(net):
  # torch.device("cpu")
  env = Market(LENGTH_BATCH, NUM_INPUTS, NUM_OUTPUTS)
  # env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
  loss_func = nn.CrossEntropyLoss()
  optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

  print(f'Starting...')
  best = []
  with SummaryWriter(comment="-indicator") as writer:
    for b, batch in enumerate(batches(env, net, LENGTH_BATCH)):
      rewards, _ = zip(*batch)
      mean = float(np.mean(rewards))

      bound = max(BOUND_MIN, np.percentile(rewards, PERCENTILE))
      # bound = np.percentile(rewards, PERCENTILE)
      train_in = []
      train_out = []
      for reward, steps in batch:
        if reward >= bound:
          obs, act = zip(*steps)
          if not (np.any(np.isnan(obs)) or np.any(np.isnan(act))):
            best.append((obs, act))
            train_in.append(obs)
            train_out.append(act)

      for obs, act in best:
        train_in.append(obs)
        train_out.append(act)
      best = best[-BEST_MAX:]

      train_in = np.concatenate(train_in)
      train_out = np.concatenate(train_out)

      train_in_t = torch.FloatTensor(train_in)
      train_out_t = torch.FloatTensor(train_out)

      fed_backward = False
      while not fed_backward:
        try:
          optimizer.zero_grad()
          output_t = net(train_in_t)
          loss_t = loss_func(output_t, train_out_t)
          loss_t.backward()
          optimizer.step()
          loss = loss_t.item()
          fed_backward = True
        except RuntimeError as e:
          print(e)
          sleep(5)

      print(f"[BATCH {b}] loss={loss:.3f}, µ={mean:.1f}, bound={bound:.1f}\n")
      writer.add_scalar("loss", loss, b)
      writer.add_scalar("bound", bound, b)
      writer.add_scalar("mean", mean, b)
      writer.flush()
      
      model.save('wnb')

if __name__ == "__main__":
  model = Model(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
  if len(sys.argv) > 1:
    model.load(sys.argv[1])
  train(model)
