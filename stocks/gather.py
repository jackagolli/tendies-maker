import yfinance as yf
import datetime

def gatherStockData(tickers, time_span, interval):

    data_dict = {}

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
        monthly_pct_change = monthly['Close'].pct_change()

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
        data_dict[ticker]["daily"]['mean'] = daily_pct_change.describe()["mean"]
        data_dict[ticker]["daily"]['std'] = daily_pct_change.describe()["std"]
        data_dict[ticker]["monthly"]['pct_change'] = monthly_pct_change
        data_dict[ticker]["monthly"]['mean'] = monthly_pct_change.describe()["mean"]
        data_dict[ticker]["monthly"]['std'] = monthly_pct_change.describe()["std"]
        data_dict[ticker]["monthly"]["cum_return"] = cum_daily_return.resample("M").mean()

        # data = yf.download('AAPL','2016-01-01','2018-01-01')
        # data.Close.plot()
        # plt.show()

    return data_dict

def gatherOptionsData(ticker, days_from_today, type):

    data_dict = {}
    today = datetime.date.today()
    days_from_today[:] = [today + datetime.timedelta(days=dt) for dt in days_from_today]
    company = yf.Ticker(ticker)
    available_opts = company.options
    format_avail_opts = []
    for date in available_opts:
        format_avail_opts.append(datetime.datetime.strptime(date, "%Y-%m-%d").date())

    matched_option_dates = [min(format_avail_opts, key=lambda x: abs(x - date)) for date in days_from_today]

    if type == "calls":
        for date in matched_option_dates:
            date_str = date.strftime("%Y-%m-%d")
            data_dict[date_str] = company.option_chain(date_str).calls

    return data_dict

