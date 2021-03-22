import stocks
import datetime as dt
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd


def main():
    num_proc = mp.cpu_count() - 1
    data_dir = Path.cwd() / 'data'
    today = dt.date.today().strftime("%m-%d-%Y")

    """ Sentiment analysis

    Use the below functions for daily scrape of r/wallstreetbets and calculation of sentiment + change
    """

    stocks.scrape_wsb(data_dir)
    stocks.calc_wsb_daily_change(data_dir)

    # Gather data for puts and calls
    tickers = stocks.gather_wsb_tickers(data_dir, today)
    tickers_split = np.array_split(np.asarray(tickers), num_proc)

    """ Options
    
    Code below will contain all functions for calculating anything related to options for given tickers.
    """
    #
    # tickers = ['AAPL', 'AMD', 'MSFT', 'SQ', 'AMAT']
    # ticker = tickers[0]
    # days_from_today = [60, 90, 150, 180, 240, 480]
    # # Get stock data
    # stock_data = stocks.gather_stock_data(tickers, time_span="1y", interval="1d")
    # # Get options data
    # options_data = stocks.gather_options_ticker(ticker, days_from_today, type="calls")
    # viable_dates = list(options_data.keys())
    # date = viable_dates[2]
    # # stocks.thinker(ticker, date, stock_data, options_data, 'bullish', '2020-06-01', 1)
    # stocks.thinker(ticker, date, stock_data, options_data, 'nominal')

    # plot_types = ['histogram', 'percent_returns']
    # stocks.plot(tickers, plot_types, stock_data)

    # data = stocks.gatherHotStocks()
    # data.to_csv("data/hot_stocks.csv")

    pool = mp.Pool(num_proc)
    with pool:
        data = pool.map(stocks.get_call_put_ratio, tickers_split)
    data = pd.concat(data, axis=0)
    stocks.append_to_table(data_dir,data,today)

    pool = mp.Pool(num_proc)
    with pool:
        data = pool.map(stocks.get_put_call_magnitude, tickers_split)
    data = pd.concat(data, axis=0)
    stocks.append_to_table(data_dir,data,today)
    """ Short interest
    
    Calculate short interest for given ticker or given wsb sheet

    """
    # desired_date = '03-11-2021'
    short_interest = stocks.gather_short_interest(data_dir)
    stocks.append_to_table(data_dir, data=short_interest, date_str=today)
    """ Price fluctuations
    
    Get max rise in past 30 days and days since last large increase (>7%)

    # """
    intraday_change = stocks.find_intraday_change(tickers)
    max = pd.DataFrame(intraday_change.max(),columns=['max_intraday_change_1mo'])
    stocks.append_to_table(data_dir, data=max, date_str=today)

    days_since_max = stocks.days_since_max_spike(intraday_change, tickers)
    stocks.append_to_table(data_dir, data=days_since_max, date_str=today)
    days_since_last = stocks.days_since_last_spike(intraday_change, tickers)
    stocks.append_to_table(data_dir, data=days_since_last, date_str=today)
    """ Technical Indicators

    RSI, BB to start.

    """
    window = 14
    prices = stocks.gather_multi(tickers, period="1mo")
    rsi = stocks.calc_RSI(tickers, prices, window)
    formatted_rsi = stocks.format_data(rsi,tickers,name="rsi")
    stocks.append_to_table(data_dir,data=rsi,date_str=today)

    sma = stocks.calc_SMA(prices, window)
    rstd = stocks.calc_rolling_std(prices, window)
    bb_value = stocks.get_BB(prices, sma, rstd)
    formatted_bb = stocks.format_data(bb_value,tickers,name="bb_val")
    stocks.append_to_table(data_dir,data=formatted_bb,date_str=today)

    MACD = stocks.get_MACD(prices)
    formatted_MACD = stocks.format_data(MACD,tickers,name="macd")
    stocks.append_to_table(data_dir,data=formatted_MACD,date_str=today)

    prices = stocks.gather_multi(tickers, period="3mo")
    ichi = stocks.get_ichimoku(prices)
    formatted_ichi = stocks.format_data(ichi,tickers,name="macd")
    stocks.append_to_table(data_dir,data=formatted_ichi,date_str=today)

    """ Sharpe ratio
        
    Use the below functions to feed a list of tickers and start date + end date to a basic optimizer 
    that will maximize the Sharpe ratio for a portfolio of those tickers given the timeframe.
    """
    # tickers = ['VOO','AAPL', 'AMD', 'MSFT', 'DIS','SWKS','AMAT','SQ','TEAM','AMZN','NVDA',
    #            'INTC','GOOGL','SPCE','CRM','MU','SHOP','NFLX','TSLA','NIO','PLTR','WKHS','ARKK']

    # sd = dt.datetime(2020, 9, 1)
    # ed = dt.datetime(2021, 2, 26)
    # start_val = 2000

    # df = stocks.gatherMulti(start_date=sd, end_date=ed, syms=tickers)
    # tickers = list(df)
    # allocations, cr, adr, sddr, sr, vals = stocks.optimize(tickers,data=df, start_val=start_val, gen_plot=True)

    # Print statistics
    # print(f"Start Amount: {start_val}")
    # print(f"Start Date: {sd}")
    # print(f"End Date: {ed}")
    # print(f"Sharpe Ratio: {sr}")
    # print(f"Cumulative Return: {cr}")
    # print(f"Average Daily Return: {adr}")
    # print(f"Allocations: {vals}")

    # desired = {tickers[i]:allocations[i] for i in range(len(allocations)) if allocations[i] > 0.001}
    # sharpe_tickers = list(desired.keys())
    # allocs = list(desired.values())
    # print(f"Optimal allocations: {desired}")
    # owned_tickers = ['MSFT','SQ','AAPL']
    # shares = [5.802718,10.601169,14.78506]
    # holdings = stocks.get_portfolio(owned_tickers,shares)
    # total = holdings.iloc[1].sum()
    # new_tickers = ['MSFT','SQ','AAPL','NIO']
    # new_allocs = [0.20,0.30,0.25,0.25]
    # buy_amount = 0
    # buys = stocks.calc_portfolio(new_tickers,new_allocs,total,buy_amount)
    # print(buys,sum(buys.values()))

    # TODO
    """ Argparse code
    
    Once finished this will enable use of main.py via command line arguments
    """
    # parser = argparse.ArgumentParser(prog='tendies maker')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')
    # parser.add_argument('--optimize', help='Pass a list of tickers to optimize for max Sharpe ratio')
    # args = parser.parse_args()
    # print(f'{sys.argv[1]} blah b lah')





if __name__ == "__main__":
    main()
