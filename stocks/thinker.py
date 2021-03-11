import datetime
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import os
import re
from matplotlib.dates import DateFormatter
import multiprocessing as mp
from pathlib import Path
from datetime import date


def thinker(ticker, date, stock_data, options_data, scenario, *args):
    """ get insight into specific option.
      ticker = str of ticker
      date = str of date (YYYY-MM-DD)
      scenario = nominal, bullish, bearish
      insight_date = date when news anticipated to affect stock price
      duration = how long it will have an effect, multiple of weeks
      """
    if args:
        insight_date = args[0]
        duration = args[1]

    today = datetime.date.today()
    end_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    latest_price = stock_data[ticker]['end_price']
    weekly_std = stock_data[ticker]["weekly"]['std']
    daily_mu = stock_data[ticker]["daily"]['mean']
    daily_std = stock_data[ticker]["daily"]['std']

    # TODO determine which distribution fits data best? then determine probability of price point.

    if scenario == 'nominal':
        # all days
        # dt = int((end_date - today).days)
        # business days only
        dt = np.busday_count(today, datetime.datetime.strptime(date, "%Y-%m-%d").date())
        # conservative outlook, use 75% of avg and BD only.
        mu = daily_mu * 0.75
        # mu = daily_mu
        x_f = latest_price * (1 + mu) ** dt

    # bullish news
    elif scenario == 'bullish':

        try:

            # calc price up until insight date.
            insight_date = datetime.datetime.strptime(insight_date, "%Y-%m-%d").date()
            # dt_1 = int((insight_date - today).days)
            dt_1 = np.busday_count(today, insight_date)
            mu = 0.75 * daily_mu
            x_1 = latest_price * (1 + mu) ** dt_1
            x_2 = x_1 * (1 + weekly_std) ** duration

            # dt_2 = int((end_date - (insight_date + datetime.timedelta(weeks=duration))).days)
            dt_2 = np.busday_count((insight_date + datetime.timedelta(weeks=duration)), end_date)
            x_f = x_2 * (1 + mu) ** dt_2


        except:

            print("Error. Make sure correct arguments entered.")

        # x_1 = stock_data[ticker]['end_price'] * (1 + weekly_std)
        # dt = int((datetime.datetime.strptime(date, "%Y-%m-%d").date() - (today + datetime.timedelta(days=7))).days)
        # x_f =  x_1 * (1 + daily_mu)**dt

    # bearish news.
    elif scenario == 'bearish':

        pass

    else:

        print("Error. This capability has not been added yet.")

    total_change = (x_f - latest_price) / latest_price * 100
    print(f"scenario = {scenario}")
    print('current price:', latest_price)
    print(f'predicted price on {date}:', x_f)
    print(f'total % change: {total_change}')
    print(f'weekly_std = {weekly_std} | daily_mu = {daily_mu} | daily_std = {daily_std}')
    print(f'expiration date is {date}')
    print(options_data[date][['strike', 'lastPrice', 'volume', 'impliedVolatility']])

    return None


def optimize(syms, data, start_val, gen_plot=False):
    """
    Optimize for maximum Sharpe ratio.
    """
    x = np.array([1 / float(len(syms)) for symbol in syms])
    normed = data.divide(data.iloc[0])

    def sharpe(x):
        rf = 0.0
        k = np.sqrt(252)
        alloced = normed.mul(x)
        pos_vals = alloced.mul(start_val)
        port_val = pos_vals.sum(axis=1)
        daily_returns = (port_val / port_val.shift(1)) - 1
        daily_returns = daily_returns[1:]

        # Modified Sharpe Ratio to remove dependence of std
        return -1 * k * daily_returns.sub(rf).mean() / (daily_returns.std())

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    result = spo.minimize(sharpe, x, method='SLSQP', bounds=tuple((0.0, 1) for symbol in syms),
                          constraints=constraints)
    sr = result.fun * -1
    allocs = result.x
    alloced = normed.mul(allocs)
    pos_vals = alloced.mul(start_val)
    port_val = pos_vals.sum(axis=1)
    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()

    vals = allocs * start_val
    vals = dict(zip(syms, vals))

    if gen_plot:
        port_norm = port_val.divide(port_val.iloc[0])
        port_norm = pd.DataFrame(port_norm, columns=['Portfolio'])
        df_temp = normed.join(port_norm)
        ax = df_temp.plot()
        ax.set_ylabel("Value")
        date_form = DateFormatter("%b %Y")
        ax.xaxis.set_major_formatter(date_form)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    return allocs, cr, adr, sddr, sr, vals


def calc_portfolio(tickers, allocs, total, buy_amount):
    new_total = total + buy_amount
    vals = np.array(allocs) * new_total
    vals = dict(zip(tickers, vals))

    return vals


def calc_large_movers(tickers, data):
    for x in tickers:
        high = yf.Ticker(x).history(period="1mo")[['High']]
        open = yf.Ticker(x).history(period="1mo")[['Open']]
        change = (high - open.values) / open.values
        if (change > 0.1).any()[0]:
            data[x] = change
        else:
            pass
    return data


def calc_wsb_daily_change(dir):
    data_dir = dir / 'data'
    files = []
    today = date.today().strftime("%m-%d-%Y")

    for file in os.listdir(data_dir):

        match = re.search(r'\d\d-\d\d-\d\d\d\d', file)

        if match and any(x in file for x in ['wsb', 'sentiment']):
            files.append(file)
            # df = pd.read_csv(data_dir / file, index_col=0, dtype={"mentions": np.int32})

    files = sorted(files, key=lambda x: datetime.datetime.strptime(
        re.search(r'\d\d-\d\d-\d\d\d\d', x)[0], "%m-%d-%Y"), reverse=True)

    df1 = pd.read_csv(data_dir / files[0], index_col=0, dtype={"mentions": np.int32})
    df2 = pd.read_csv(data_dir / files[1], index_col=0, dtype={"mentions": np.int32})

    df1.insert(0, column='rank', value=np.arange(1, len(df1) + 1))
    df2['rank'] = np.arange(1, len(df2) + 1)

    temp = df2['rank'] - df1['rank']
    temp = temp.dropna()

    df1['change'] = temp
    df1 = df1.fillna(0)
    df1 = df1.astype({'change': 'int32'})

    if Path(data_dir / ("wsb_sentiment_" + today + ".csv")).is_file():
        overwrite = input('Scrape already exists for today in /data dir.\nOverwrite (Y/N)? ')

        if overwrite == 'Y' or overwrite == 'y':
            df1.to_csv(data_dir / ('wsb_sentiment_' + today + '.csv'))
        elif overwrite == 'N' or overwrite == 'n':
            pass
        else:
            print('Invalid input.')

    else:
        df1.to_csv(data_dir / ('wsb_sentiment_' + today + '.csv'))

    return None


def find_hot_stocks(tickers=None):
    if not tickers:
        tickers1 = pd.read_csv('data/nasdaq_stocks.csv', sep=',')
        tickers2 = pd.read_csv('data/nyse_stocks.csv', sep=',')
        tickers = pd.concat([tickers1, tickers2])
        tickers = tickers['Symbol']

    # today = datetime.date.today()
    num_proc = mp.cpu_count() - 1
    tickers = tickers.to_numpy()
    tickers_split = np.array_split(tickers, num_proc)

    # Initialize empty df with correct row date indexes
    data = yf.Ticker('AAPL').history(period='1mo')
    data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], inplace=True)
    i = 1

    cases = [(stocks, data) for stocks in tickers_split]
    pool = mp.Pool(num_proc)
    results = pool.starmap(calc_large_movers, cases)
    pool.close()
    pool.join()

    results = pd.concat(results, axis=1)

    return results


def append_to_table(data_dir, short_interest, date_str):
    today = date.today().strftime("%m-%d-%Y")

    data_df = pd.read_csv(data_dir / ('wsb_sentiment_' + date_str + '.csv'), header=0, index_col=0)
    data_df['short_interest'] = np.zeros(len(data_df))
    tickers = data_df.index.values
    shorted_filtered = short_interest[short_interest['Ticker'].isin(tickers)]

    index = shorted_filtered.index.values
    short_intst = shorted_filtered.ShortInt.values
    short_intst = short_intst.astype('str')
    short_intst = np.char.strip(short_intst, chars='%')
    short_intst = short_intst.astype(np.float32)
    short_intst = short_intst / 100
    short_intst_tickers = shorted_filtered.Ticker.values

    for i in range(len(index)):
        if short_intst_tickers[i] in tickers:
            data_df.loc[short_intst_tickers[i], 'short_interest'] = short_intst[i]

    save_path = Path(data_dir / ("data_" + today + ".csv"))

    if save_path.is_file():
        overwrite = input('Data file already exists for today. Overwrite (Y/N)? ')

        if overwrite == 'Y' or overwrite == 'y':
            data_df.to_csv(save_path)
        elif overwrite == 'N' or overwrite == 'n':
            pass
        else:
            print('Invalid input.')
    else:
        data_df.to_csv(save_path)

    return None
