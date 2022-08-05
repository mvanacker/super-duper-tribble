from collections import namedtuple
import numpy as np

Action = namedtuple('Action', ['take_profit', 'stop_loss', 'win'])

SL_MIN  = 10
SL_MAX  = 50
SL_RES  = 10
TP_MIN  = 10
TP_MAX  = 50
TP_RES  = 10
WIN_MIN = 45
WIN_MAX = 55
WIN_RES = 1

sl_levels  = np.arange(SL_MIN,  1+SL_MAX,  SL_RES)
tp_levels  = np.arange(TP_MIN,  1+TP_MAX,  TP_RES)
win_levels = np.arange(WIN_MIN, 1+WIN_MAX, WIN_RES)

Policy = namedtuple('Policy', [f'sl_{s}_tp_{t}_win_{w}' for s in sl_levels for t in tp_levels for w in win_levels])

sl_levels  = sl_levels  / 100
tp_levels  = tp_levels  / 100
win_levels = win_levels / 100

actions = [Action(s, t, w) for s in sl_levels for t in tp_levels for w in win_levels]
