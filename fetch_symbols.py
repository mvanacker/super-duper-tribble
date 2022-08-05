import pandas as pd
import pandas_datareader as pdr
s = pdr.get_nasdaq_symbols()
for symbol in s.index:
  try:
    pdr.get_data_yahoo(symbol).to_csv(f'symbols\\{symbol}.csv')
  except:
    pass
