import stocks


def main():

    # initialize
    tickers = ['SPY','AAPL','AMD','MSFT','DIS']
    ticker = tickers[0]
    days_from_today = [60,180,480]

    # get data
    stock_data = stocks.gatherStockData(tickers, time_span="1y", interval="1d")
    options_data = stocks.gatherOptionsData(ticker, days_from_today, type="calls")
    viable_dates = list(options_data.keys())
    date = viable_dates[0]

    #stocks.thinker(ticker, date, stock_data, options_data, 'bullish', '2020-04-01', 2)
    stocks.thinker(ticker, date, stock_data, options_data, 'nominal')

    # plot_types = ['histogram', 'percent_returns']
    # stocks.plot(tickers, plot_types, "daily", stock_data)




if __name__ == "__main__":
    main()