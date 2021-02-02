import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from multiprocessing import Pool, cpu_count




def gatherStockData(tickers, time_span, interval):
    data_dict = {}

    for ticker in tickers:
        data_dict[ticker] = {}
        data_dict[ticker]["monthly"] = {}
        data_dict[ticker]["weekly"] = {}
        data_dict[ticker]["daily"] = {}
        company = yf.Ticker(ticker)
        data = company.history(period=time_span, interval=interval)
        data_dict[ticker]["start_price"] = data["Close"][0]
        data_dict[ticker]["end_price"] = data["Close"][-1]
        daily_close = data['Close']
        data_dict[ticker]['daily']['close'] = daily_close
        daily_pct_change = daily_close.pct_change()
        cum_daily_return = ((1 + daily_pct_change).cumprod() - 1) * 100
        weekly = data.resample('W-Mon').mean()
        weekly_pct_change = weekly['Close'].pct_change()
        monthly = data.resample('M').mean()
        monthly_pct_change = monthly['Close'].pct_change()

        data_dict[ticker]["daily"]['pct_change'] = daily_pct_change
        data_dict[ticker]["daily"]["cum_return"] = cum_daily_return
        data_dict[ticker]["daily"]['mean'] = daily_pct_change.describe()["mean"]
        data_dict[ticker]["daily"]['std'] = daily_pct_change.describe()["std"]

        data_dict[ticker]["weekly"]['close'] = weekly['Close']
        data_dict[ticker]["weekly"]['pct_change'] = weekly_pct_change
        data_dict[ticker]["weekly"]['mean'] = weekly_pct_change.describe()["mean"]
        data_dict[ticker]["weekly"]['std'] = weekly_pct_change.describe()["std"]

        data_dict[ticker]['monthly']['close'] = monthly['Close']
        data_dict[ticker]["monthly"]['pct_change'] = monthly_pct_change
        data_dict[ticker]["monthly"]['mean'] = monthly_pct_change.describe()["mean"]
        data_dict[ticker]["monthly"]['std'] = monthly_pct_change.describe()["std"]
        data_dict[ticker]["monthly"]["cum_return"] = cum_daily_return.resample("M").mean()

        # data = yf.download('AAPL','2016-01-01','2018-01-01')

    return data_dict


def gatherOptionsData(ticker, days_from_today, type):
    """ data stored in yfinance.options:
    ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume',
    'openInterest', 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency']
    type = calls or puts
    """
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


# This function searches
def gatherMemeStocks():
    today = datetime.date.today()
    all_stocks = pd.read_csv('data/nasdaq_stocks_filtered.csv', sep=',')
    all_stocks = all_stocks['Symbol']

    all_stocks = all_stocks.to_numpy()
    # initialize empty df with correct row indexes
    data = yf.Ticker('AAPL').history(period="1mo")
    data.drop(columns=['Open','High','Low','Close','Volume','Dividends','Stock Splits'], inplace=True)
    i = 1

    def calc_change(tickers, data):
        for x in tickers:
            high = yf.Ticker(x).history(period="1mo")[['High']]
            open = yf.Ticker(x).history(period="1mo")[['Open']]
            change = (high - open.values) / open.values
            if (change > 0.1).any()[0]:
                data[x] = change
            else:
                pass
        return data

    data = calc_change(all_stocks, data)
    # for x in all_stocks:
    #     high = yf.Ticker(x).history(period="1mo")[['High']]
    #     open = yf.Ticker(x).history(period="1mo")[['Open']]
    #     change = (high - open.values) / open.values
    #     data[x] = change
    #     print(f'{len(all_stocks) - i} are left')
    #     i+=1
    # all_stocks = all_stocks.tolist()


        # daily_returns = (prices / prices.shift(1)) - 1
        # daily_returns = daily_returns[1:]
# change by 100 for pct


    return data

def gatherMulti(start_date, end_date, syms):
    data = yf.download(" ".join(syms), start=start_date, end=end_date)
    df = data['Close']

    return df


def getPortolfio(tickers, shares):
    """
    Get current portfolio given allocations
    """
    data = yf.download(" ".join(tickers), period="1d", interval="1m")
    df = data['Close'].iloc[[-5]]
    df = df.append(pd.DataFrame(np.array([shares[i] * df[val][0] for i, val in enumerate(tickers)]).reshape(1, 4),
                                index=["values"], columns=tickers))
    total = df.sum(axis=1)[0]
    df = df.append(pd.DataFrame(np.array([df[val][0] / total for i, val in enumerate(tickers)]).reshape(1, 4),
                                index=["allocs"], columns=tickers))

    df = df.rename(index={f'{df.index.values[0]}': "current price"})

    return df
