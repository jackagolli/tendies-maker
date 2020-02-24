import stocks


def main():

    # initialize
    tickers = ['SPY','AMAT','AAPL','DIS','MSFT']
    # pick one ticker for looking at closer (options).
    ticker = tickers[1]
    # pick rough desired options dates. should go medium-long term (2-3 mo, 6 mo, or 1 yr) to
    # bypass short-term fluctuations
    days_from_today = [90,140,480]

    # get stock data
    stock_data = stocks.gatherStockData(tickers, time_span="1y", interval="1d")
    # get options data
    options_data = stocks.gatherOptionsData(ticker, days_from_today, type="calls")
    viable_dates = list(options_data.keys())
    date = viable_dates[0]
    stocks.thinker(ticker, date, stock_data, options_data, 'bullish', '2020-06-01', 1)
    stocks.thinker(ticker, date, stock_data, options_data, 'nominal')

    #plot_types = ['histogram', 'percent_returns']
    #stocks.plot(tickers, plot_types, "weekly", stock_data)


if __name__ == "__main__":
    main()