import numpy as np
import pandas as pd
from math import inf

def rsi(src, length=14):
  diff = src.diff()
  gain = diff.clip(0, +inf).ewm(com=length-1).mean()
  loss = diff.clip(-inf, 0).ewm(com=length-1).mean()
  return 1.0/(1-loss/gain)

def stoch(src, length=14, smooth=3):
  r = src.rolling(length)
  m = r.min()
  return ((src-m)/(r.max()-m)).rolling(smooth).mean()

def hvp(src, std_len=10, pct_len=100):
  return np.log(src).diff().rolling(std_len).std().rolling(pct_len).rank()

def tr(high, low, close):
  prev = close.shift(1)
  return pd.DataFrame((high-low, abs(high-prev), abs(low-prev))).max()

def trend(high, low, close, length=14, smooth=3, tolerance=.25):
  dh   = high.diff()
  dl   = -low.diff()
  dh_  = dh.copy()
  dh_.loc[(dh < 0) | (dh <= dl)] = 0
  dl .loc[(dl < 0) | (dl <= dh)] = 0
  dh   = dh_

  atr  = tr(high, low, close).ewm(com=length-1).mean()
  up   = (dh.ewm(com=length-1).mean() / atr).rolling(smooth).mean()
  down = (dl.ewm(com=length-1).mean() / atr).rolling(smooth).mean()
  d    = up-down
  ad   = abs(d)
  str  = ((ad/(up+down)) - tolerance).clip(0, inf)
  return (d/ad*str).rolling(smooth).mean()
