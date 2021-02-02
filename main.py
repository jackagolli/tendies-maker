import stocks
import datetime as dt
import argparse
import numpy as np
import sys


def main():
    # Options stuff
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

    test = stocks.gatherMemeStocks()

    # Sharpe ratio stuff

    # tickers = ['VOO','AAPL', 'AMD', 'MSFT', 'DIS','SWKS','AMAT','SQ','TEAM','AMZN','NVDA',
    #            'INTC','GOOGL','SPCE','CRM','MU','SHOP','NFLX','TSLA']
    #
    # sd = dt.datetime(2020, 10, 1)
    # ed = dt.datetime(2021, 1, 19)
    # start_val = 1000
    #
    # df = stocks.gatherMulti(start_date=sd, end_date=ed, syms=tickers)
    # tickers = list(df)
    # allocations, cr, adr, sddr, sr, vals = stocks.optimize(tickers,data=df, start_val=start_val, gen_plot=True)
    #
    # # Print statistics
    # # print(f"Start Amount: {start_val}")
    # print(f"Start Date: {sd}")
    # print(f"End Date: {ed}")
    # print(f"Sharpe Ratio: {sr}")
    # print(f"Volatility (stdev of daily returns): {sddr}")
    # print(f"Cumulative Return: {cr}")
    # # print(f"Average Daily Return: {adr}")
    # # print(f"Allocations: {vals}")
    # desired = {tickers[i]:allocations[i] for i in range(len(allocations)) if allocations[i] > 0.001}
    # sharpe_tickers = list(desired.keys())
    # allocs = list(desired.values())
    # print(f"Optimal allocations: {desired}")
    #
    # owned_tickers = ['AAPL','MSFT','SQ','SHOP']
    # shares = [3.684302,5.773128,10.601169,1.052782]
    # holdings = stocks.getPortolfio(owned_tickers,shares)
    # total = holdings.iloc[1].sum()
    # new_tickers = ['AAPL','MSFT','SQ','SHOP']
    # new_allocs = [0.30,0.20,0.30,0.20]
    # buy_amount = 0
    # buys = stocks.calcPortfolio(new_tickers,new_allocs,total,buy_amount)
    # print(buys,sum(buys.values()))

    # Arguments stuff

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
