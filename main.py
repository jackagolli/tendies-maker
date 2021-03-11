import stocks
import datetime as dt
import argparse
import numpy as np
import sys
import os
import yfinance as yf
from pathlib import Path


def main():
    data_dir = Path.cwd() / 'data'
    """ Options
    
    Code below will contain all functions for calculating anything related to options for given tickers.
    """
    #
    # tickers = ['AAPL', 'AMD', 'MSFT', 'SQ', 'AMAT']
    # ticker = tickers[0]
    # days_from_today = [60, 90, 150, 180, 240, 480]
    # # get stock data
    # stock_data = stocks.gatherStockData(tickers, time_span="1y", interval="1d")
    # # get options data
    # options_data = stocks.gatherOptionsData(ticker, days_from_today, type="calls")
    # viable_dates = list(options_data.keys())
    # date = viable_dates[2]
    # # stocks.thinker(ticker, date, stock_data, options_data, 'bullish', '2020-06-01', 1)
    # stocks.thinker(ticker, date, stock_data, options_data, 'nominal')

    # plot_types = ['histogram', 'percent_returns']
    # stocks.plot(tickers, plot_types, stock_data)

    # data = stocks.gatherHotStocks()
    # data.to_csv("data/hot_stocks.csv")
    """ Sentiment analysis

    Use the below functions for daily scrape of r/wallstreetbets and calculation of sentiment + change
    """

    # stocks.scrape_wsb(data_dir)
    # stocks.calc_wsb_daily_change(data_dir)

    """ Short interest
    
    Calculate short interest for given ticker or given wsb sheet

    """
    desired_date = '03-11-2021'
    short_interest = stocks.gather_short_interest(data_dir)
    stocks.append_to_table(data_dir, short_interest, desired_date)

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
