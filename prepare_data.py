import stocks
from pathlib import Path
import datetime as dt
import multiprocessing as mp
import pandas as pd

# Set data directory here
data_dir = Path.cwd() / 'data'
if data_dir.exists():
    pass
else:
    Path.mkdir(data_dir)

today = dt.date.today().strftime("%m-%d-%Y")
num_proc = mp.cpu_count() - 1

"""
Get results for the day. Run after market close.
"""

tickers = stocks.gather_wsb_tickers(data_dir, today)
prices = stocks.gather_multi(tickers,period="1d",key='all')
result = stocks.gather_results(prices,tickers)
stocks.append_to_table(data_dir,data=result,date_str=today)

# test = pd.read_csv('data/data_03-22-2021.csv',index_col=0)
# test2 = stocks.normalize(test)
# test3 = test.mean()
# test4 = test.std()