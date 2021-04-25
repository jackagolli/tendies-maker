import stocks
import datetime as dt

""" Sharpe ratio

Use the below functions to feed a list of tickers and start date + end date to a basic optimizer
that will maximize the Sharpe ratio for a portfolio of those tickers given the timeframe.
"""
tickers = ['MSFT','AAPL', 'AMD','QCOM','ADBE', 'ACN', 'LRCX','ISRG','TMO','ILMN','MSFT','DIS','SWKS',
           'AMAT','SQ','TEAM','AMZN','NVDA','INTC','GOOGL','SPCE','CRM','MU','SHOP','NFLX','TSLA','NIO',
           'WKHS','ARKK','PYPL','PLUG','PLTR']

sd = dt.datetime(2016, 4, 13)
ed = dt.datetime(2021, 4, 13)
start_val = 1000

df = stocks.gather_multi(start_date=sd, end_date=ed, syms=tickers)
columns = list(df.columns)
tickers =[x[1] for x in columns]
df.columns = tickers
allocations, cr, adr, sddr, sr, vals = stocks.optimize(tickers,data=df, start_val=start_val, gen_plot=True)

# Print statisticss
print(f"Start Amount: {start_val}")
print(f"Start Date: {sd}")
print(f"End Date: {ed}")
print(f"Sharpe Ratio: {sr}")
print(f"Cumulative Return: {cr}")
print(f"Average Daily Return: {adr}")
# print(f"Allocations: {vals}")

desired = {tickers[i]:allocations[i] for i in range(len(allocations)) if allocations[i] > 0.001}
sharpe_tickers = list(desired.keys())
allocs = list(desired.values())
print(f"Optimal allocations: {desired}")

# This code is to see what you have to buy given your current share holdings
# owned_tickers = ['MSFT','SQ','AAPL']
# shares = [5.802718,10.601169,14.78506]
# holdings = stocks.get_portfolio(owned_tickers,shares)
# total = holdings.iloc[1].sum()

# Specify buy amount
total = 1000
new_tickers = list(desired.keys())
manual_tickers = ['SQ','TSLA','NIO','WKHS']
manual_allocs = [0.48,0.27,0.20,0.05]
buy_amount = 0
manual_buys = stocks.calc_portfolio(manual_tickers,manual_allocs,total,buy_amount)
optimal_buys = stocks.calc_portfolio(new_tickers,allocs,total,buy_amount)
print(f'Manual buys are {manual_buys} and total cost is ${sum(manual_buys.values())}')
print(f'Optimal buys are {optimal_buys} and total cost is ${sum(optimal_buys.values())}')
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
# # stocks.thinker(ticker, date, stock_data, options_data, 'b  ullish', '2020-06-01', 1)
# stocks.thinker(ticker, date, stock_data, options_data, 'nominal')

# plot_types = ['histogram', 'percent_returns']
# stocks.plot(tickers, plot_types, stock_data)