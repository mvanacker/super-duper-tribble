import pandas as pd

__symbols = pd.read_csv(f'symbols.csv')['Symbol']

__today = pd.Timestamp.today().date()
__max_inactivity = pd.Timedelta(days=14)
__min_age = pd.Timedelta(weeks=4*52)

def is_active(data, max_inactivity=__max_inactivity):
  return __today - pd.Timestamp(data.iloc[-1].name).date() < max_inactivity
def is_mature(data, min_age=__min_age):
  return __today - pd.Timestamp(data.iloc[0].name).date() > min_age

def sample(size, max_inactivity=__max_inactivity, min_age=__min_age):
  data = []
  while len(data) < size:
    s = __symbols.sample().iloc[0]
    try:
      df = pd.read_csv(f'symbols\\{s}.csv', index_col='Date')
      if is_active(df, max_inactivity) and is_mature(df, min_age):
        data.append((s, df))
    except FileNotFoundError:
      pass
  return data
