import datetime
import yfinance as yf


def gatherData(tickers, time_span, interval):

    data_dict = {}
    today = datetime.date.today()

    for ticker in tickers:
        data_dict[ticker] = {}
        data_dict[ticker]["monthly"] = {}
        data_dict[ticker]["daily"] = {}
        company = yf.Ticker(ticker)
        data = company.history(period=time_span, interval=interval)
        data_dict[ticker]["start_price"] = data["Close"][0]
        data_dict[ticker]["end_price"] = data["Close"][-1]
        daily_close = data['Close']
        daily_pct_change = daily_close.pct_change()
        cum_daily_return = ((1 + daily_pct_change).cumprod() - 1) * 100
        monthly = data.resample('BM').apply(lambda x: x[-1])
        monthly_pct_change = monthly.pct_change()

        # daily_pct_change.name = "Daily % Change"
        # print(daily_pct_change.describe())
        # sns.distplot(daily_pct_change,kde=False, bins=50)
        # plt.title(ticker)
        # plt.ylabel('Frequency')
        # plt.show()
        #
        # cum_daily_return.plot(figsize=(12,8))
        # plt.title(ticker)
        # plt.ylabel('Returns (%)')
        # plt.show()

        data_dict[ticker]["daily"]['pct_change'] = daily_pct_change
        data_dict[ticker]["daily"]["cum_return"] = cum_daily_return
        data_dict[ticker]["monthly"]['pct_change'] = monthly_pct_change
        data_dict[ticker]["monthly"]["cum_return"] = cum_daily_return.resample("M").mean()

        # data = yf.download('AAPL','2016-01-01','2018-01-01')
        # data.Close.plot()
        # plt.show()

    return data_dict
