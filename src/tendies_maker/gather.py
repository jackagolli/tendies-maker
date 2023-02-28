import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
import re
import multiprocessing as mp
from datetime import date, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from bs4 import BeautifulSoup
from tqdm import tqdm

num_proc = mp.cpu_count() - 1


def gather_stock_data(tickers, time_span, interval):
    """
    This will be deprecated

    :param tickers:
    :param time_span:
    :param interval:
    :return:
    """
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


def gather_options_ticker(ticker, days_from_today, type):
    """ data stored in yfinance.options:
    ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume',
    'openInterest', 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency']
    type = calls or puts

    This will be deprecated
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


def gather_multi(syms, **kwargs):
    """
    Gather stock data for multiple tickers

    :param syms: String of symbols with whitespace between each
    :param kwargs: Eather provide a start_date and end_date datetime object, or pass a str to period
    :return: dataframe of price history for given time period
    """
    start_date = kwargs.get('start_date', None)
    end_date = kwargs.get('end_date', None)
    period = kwargs.get('period', None)
    key = kwargs.get('key', 'Adj Close')

    if start_date and end_date:
        df = yf.download(" ".join(syms), start=start_date, end=end_date)
    else:
        df = yf.download(" ".join(syms), period=period)

    if key == 'all':
        pass
    else:
        df = df.filter(like=key, axis=1)

    return df


def gather_DTE(tickers):
    df = pd.DataFrame(index=tickers, columns=['DTE'])

    for ticker in tqdm(tickers):
        yf_ticker = yf.Ticker(ticker)
        dates = yf_ticker.earnings_dates
        delta = 0
        if dates is not None:
            future_dates = [x for x in dates.index.to_pydatetime().tolist() if x > datetime.datetime.now(timezone.utc)]
            if future_dates:
                next_date = future_dates[-1].date()
                delta = next_date - date.today()
                delta = int(delta.days)

        df.loc[ticker, 'DTE'] = delta

    return df



def gather_single_prices(ticker, period="1mo"):
    ticker = yf.Ticker(ticker)
    data = ticker.history(period=period)

    return data


def get_call_put_ratio(tickers):
    df = pd.DataFrame(index=tickers, columns=['Put Call Volume Ratio'])
    for ticker in tqdm(tickers):
        total_put_vol = 0
        total_call_vol = 0
        single = yf.Ticker(ticker)

        try:
            dates = single.options
            for date in dates:
                options = single.option_chain(date)
                calls = options[0]
                puts = options[1]
                total_call_vol += calls['volume'].sum()
                total_put_vol += puts['volume'].sum()

            df.loc[ticker, 'Put Call Volume Ratio'] = total_put_vol / total_call_vol

        except:

            df.loc[ticker, 'Put Call Volume Ratio'] = 0

            continue
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    return df


def get_put_call_magnitude(tickers):
    df = pd.DataFrame(index=tickers, columns=['Put Call Value Ratio'])
    for ticker in tqdm(tickers):
        total_put_val = 0
        total_call_val = 0
        single = yf.Ticker(ticker)

        try:
            dates = single.options
            for date in dates:
                options = single.option_chain(date)
                calls = options[0].fillna(0)
                puts = options[1].fillna(0)
                total_call_val += np.sum((calls['volume'].values * calls['lastPrice'].values))
                total_put_val += np.sum((puts['volume'].values * puts['lastPrice'].values))

            df.loc[ticker, 'Put Call Value Ratio'] = total_put_val / total_call_val

        except:

            df.loc[ticker, 'Put Call Value Ratio'] = 0

            continue
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    return df


def get_portfolio(tickers, shares):
    """
    Get current portfolio given num of shares and tickers

    :param list tickers: stock tickers, list of str
    :param list shares: num of shares, list of float
    :return: dataframe of current portfolio holdings
    """
    n = len(tickers)
    data = yf.download(" ".join(tickers), period="1d", interval="1m")
    df = data['Close'].iloc[[-5]]
    df = df.append(pd.DataFrame(np.array([shares[i] * df[val][0] for i, val in enumerate(tickers)]).reshape(1, n),
                                index=["values"], columns=tickers))
    total = df.sum(axis=1)[0]
    df = df.append(pd.DataFrame(np.array([df[val][0] / total for i, val in enumerate(tickers)]).reshape(1, n),
                                index=["allocs"], columns=tickers))

    df = df.rename(index={f'{df.index.values[0]}': "current price"})

    return df


def scrape_wsb(data_dir):
    res = requests.get('https://stocks.comment.ai/trending.html')
    wsb_data = pd.DataFrame(columns=["mentions", "sentiment"])
    tickers = []
    today = date.today().strftime("%m-%d-%Y")

    for line in res.text.splitlines():

        m = re.findall(r'(?<=<td>)\d+(?=<\/td>)|(?<=<\/svg>\s)\d+(?=\s<br\/>)|(?<=center;">)\w+(?=<\/td>)', line)

        if m:

            total = int(m[0])
            pos = int(m[1])
            neutral = int(m[2])
            neg = int(m[3])
            ticker = m[4]

            if total == (pos + neutral + neg):
                # Append only if values add up.
                # 0 = neg, 0.5 = neutral, 1 = positive
                tickers.append(m[4])
                normalized_sentiment = (0 * neg + 0.5 * neutral + 1 * pos) / total
                wsb_data.loc[ticker] = [total, normalized_sentiment]

    if Path(data_dir / ("wsb_sentiment_" + today + ".csv")).is_file():
        overwrite = input('Scrape already generated for today. Overwrite (Y/N)? ')

        if overwrite == 'Y' or overwrite == 'y':
            wsb_data.to_csv("data/wsb_sentiment_" + today + ".csv")
        elif overwrite == 'N' or overwrite == 'n':
            pass
        else:
            print('Invalid input.')
    else:
        wsb_data.to_csv("data/wsb_sentiment_" + today + ".csv")

    return None


def gather_short_interest(data_dir):
    scraped_tables = pd.read_html('https://www.highshortinterest.com/', header=0)
    df = scraped_tables[2]
    df = df[~df['Ticker'].str.contains("google_ad_client", na=False)]
    df = df.dropna()
    # df = df.drop(columns=['Exchange'])
    df = df.set_index('Ticker')
    df.index.name = None
    df = df[['ShortInt']]
    df.columns = ['short_interest']

    short_interest = df.short_interest.values
    short_interest = short_interest.astype('str')
    short_interest = np.char.strip(short_interest, chars='%')
    short_interest = short_interest.astype(np.float32)
    short_interest = short_interest / 100

    df['short_interest'] = short_interest

    return df


def gather_wsb_tickers(data_dir, date_str):
    try:
        df = pd.read_csv(data_dir / ('wsb_sentiment_' + date_str + '.csv'), header=0)
    except:
        df = pd.read_csv(data_dir / ('data_' + date_str + '.csv'), header=0)
    tickers = df.iloc[:, 0].tolist()

    return tickers


def gather_results(prices, tickers):
    change = pd.DataFrame(index=tickers, columns=['Y'])

    for x in tickers:
        changes = []
        if prices.shape[0] == 1:
            for index, row in prices.iterrows():
                high = prices.High.loc[index, x]
                open = prices.Open.loc[index, x]
                delta = (high - open) / open
                changes.append(delta)

            change.loc[x, 'Y'] = max(changes)
        else:
            high = prices[('High', x)]
            open = prices[('Open', x)]
            close = prices[('Close', x)]
            delta = (high[1] - open[1]) / open[1]
            # delta = (high.values - open.values) / open.values
            changes.append(delta)
            delta = (high[1] - close[0]) / close[0]
            changes.append(delta)
            change.loc[x, 'Y'] = max(changes)

    change.fillna(0, inplace=True)
    change[change < 0.06] = 0
    change[change != 0] = 1

    return change


def scrape_news_sentiment(tickers=None):
    # tickers = ['GME','AMC']
    nltk.download('vader_lexicon')
    df = pd.DataFrame(index=tickers, columns=['news_sentiment'])
    base_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        url = base_url + ticker
        try:
            req = Request(url=url, headers={"User-Agent": "Chrome"})
            response = urlopen(req)
            html = BeautifulSoup(response, "html.parser")
            news_table = html.find(id='news-table')
        except:
            news_table = ''
        news_tables[ticker] = news_table

    news_headlines = {}

    for ticker, news_table in news_tables.items():
        news_headlines[ticker] = []
        dates = []
        try:

            for i in news_table.findAll('tr'):
                # Strictly get more recent sentiment
                if len(dates) == 4:
                    break
                text = i.a.get_text()
                date_scrape = i.td.text.split()

                if len(date_scrape) != 1:
                    date = date_scrape[0]
                    dates.append(date)

                news_headlines[ticker].append(text)

        except:
            news_headlines[ticker].append('')

    vader = SentimentIntensityAnalyzer()

    for ticker, value in news_headlines.items():
        # This is the avg score between -1 and 1 of the all the news headlines
        news_df = pd.DataFrame(news_headlines[ticker], columns=['headline'])
        scores = news_df['headline'].apply(vader.polarity_scores).tolist()
        scores_df = pd.DataFrame(scores)
        score = scores_df['compound'].mean()
        df.loc[ticker, 'news_sentiment'] = score
        # news_df = news_df.join(scores_df, rsuffix='_right')
        # news_df['date'] = pd.to_datetime(news_df.date).dt.date

    return df
