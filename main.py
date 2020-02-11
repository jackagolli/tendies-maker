import stocks

def main():

    tickers = ['SPY','AAPL','AMD','MSFT','DIS']
    stock_data = stocks.gatherData(tickers, time_span="1y", interval="1d")
    plot_types = ['histogram', 'percent_returns']
    stocks.plot([tickers[0]], plot_types,stock_data)
    print(stock_data[tickers[0]]["daily"]["pct_change"].describe())
    print(stock_data[tickers[0]]["monthly"]["pct_change"].describe())

if __name__ == "__main__":
    main()