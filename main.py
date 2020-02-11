import stocks
import datetime

def main():

    tickers = ['SPY','AAPL','AMD','MSFT','DIS']
    #stock_data = stocks.gatherStockData(tickers, time_span="1y", interval="1d")
    ticker = tickers[1]
    #print(stock_data[ticker]['daily']['mean'])
    days_from_today = [60,180,480]
    options_data = stocks.gatherOptionsData(ticker, days_from_today, type="calls")
    print(options_data)
    plot_types = ['histogram', 'percent_returns']
    #stocks.plot(tickers, plot_types,stock_data)


if __name__ == "__main__":
    main()