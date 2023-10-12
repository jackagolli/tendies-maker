from datetime import date
import os
from pathlib import Path
import re
import requests

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from tqdm import tqdm
import yfinance as yf

from src.tendies_maker.db import DB

num_proc = mp.cpu_count() - 1
db = DB()
params = {'apiKey': os.environ["POLYGON_API_KEY"]}


def get_macro_econ_data():
    today = datetime.datetime.today()
    labels = {'PCE': 'PCE',
              'UNRATE': 'Unemployment',
              'MICH': 'Inflation Expectation',
              'JTSJOL': 'Job Openings',
              'FEDFUNDS': 'Fed Funds Rate',
              'M2REAL': 'Real M2',
              'GDPC1': 'Real GDP',
              'RSXFS': 'Retail Sales',
              'EXHOSLUSM495S': 'Existing Home Sales'
              }
    data = pdr.data.DataReader(list(labels.keys()), 'fred', today - datetime.timedelta(
        days=365), today)
    # Calculate the last % change for each macroindicator
    latest_pct_changes = data.apply(lambda x: x.dropna().pct_change().iloc[-1] if x.dropna().shape[0] > 1 else np.nan)
    latest_pct_changes.rename(labels, inplace=True)

    # Get the last valid index (date of last non-null value) for each macroindicator
    last_update_dates = data.apply(lambda x: x.last_valid_index())
    last_update_dates.rename({k: v for k, v in labels.items()}, inplace=True)

    # Combine the latest % changes and last update dates into a single DataFrame
    combined_data = pd.DataFrame({
        'Latest % Change': latest_pct_changes,
        'Last Update Date': last_update_dates
    })

    return combined_data


def get_news(ticker, asof_date):
    full_url = f'https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.lte={asof_date}&limit=50'
    response = requests.get(full_url, params=params)

    if response.status_code == 200:
        response_data = response.json()
        data = response_data.get('results', [])

        if data:
            return pd.DataFrame(data)
        else:
            return None

    else:
        print(f"Error: Received status code {response.status_code}")


def get_market_holidays():
    full_url = 'https://api.polygon.io/v1/marketstatus/upcoming'
    response = requests.get(full_url, params=params)

    if response.status_code == 200:
        data = response.json()
        closed_dates = [entry['date'] for entry in data if entry['status'] == 'closed']
        return list(set(closed_dates))
    else:
        return None


def get_nearest_open_market_date(input_date_str):
    input_date = datetime.datetime.strptime(input_date_str, '%Y-%m-%d').date()
    closed_dates = get_market_holidays()

    if closed_dates is None:
        return None

    while True:
        # Advance to the next weekday if it's a weekend
        while input_date.weekday() >= 5:
            input_date += datetime.timedelta(days=1)

        # Check if it's a market holiday
        if input_date.strftime('%Y-%m-%d') not in closed_dates:
            return input_date.strftime('%Y-%m-%d')

        # Otherwise, go to the next day
        input_date += datetime.timedelta(days=1)


def get_options_data(ticker, asof_date):
    # Generate the URL based on whether you want expired options or not
    full_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{asof_date}/" \
               f"{asof_date}?adjusted=true&sort=desc&limit=50000"
    response = requests.get(full_url, params=params)

    if response.status_code == 200:
        response_data = response.json()
        data = response_data.get('results', [])

        if data:
            data[0].pop('t', None)
            data[0]['ticker'] = ticker
            df = pd.DataFrame(data)
            return df
        else:
            return None

    else:
        print(f"Error: Received status code {response.status_code}")

def get_options_snapshot(ticker):
    # Generate the URL based on whether you want expired options or not
    full_url = f"https://api.polygon.io/v3/snapshot/options/{ticker}?limit=250"
    next_url = full_url  # Start with the first URL
    all_flattened_data = []
    while next_url:
        response = requests.get(next_url, params=params)
        if response.status_code == 200:
            response_data = response.json()
            data = response_data.get('results', [])

            # Flatten the data
            flattened_data = []
            for record in data:
                flat_record = {}
                for key, value in record.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flat_record[f"{key}_{sub_key}"] = sub_value
                    else:
                        flat_record[key] = value
                flattened_data.append(flat_record)

            all_flattened_data.extend(flattened_data)

            # Get the next URL for pagination, if available
            next_url = response_data.get('next_url', None)
        else:
            print(f"Error: Received status code {response.status_code}")
            break

    result = pd.DataFrame.from_records(all_flattened_data)
    result.dropna(inplace=True)
    return result
def get_options_chain(ticker, asof_date):
    all_flattened_data = []

    for expired in [False]:
        # Generate the URL based on whether you want expired options or not
        full_url = f"https://api.polygon.io/v3/reference/options/contracts?" \
                   f"underlying_ticker={ticker}&expired={str(expired).lower()}&order=desc&limit=250&sort=" \
                   f"expiration_date&as_of={asof_date}"

        next_url = full_url  # Start with the first URL

        while next_url:
            response = requests.get(next_url, params=params)
            if response.status_code == 200:
                response_data = response.json()
                data = response_data.get('results', [])

                # Flatten the data
                flattened_data = []
                for record in data:
                    flat_record = {}
                    for key, value in record.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                flat_record[f"{key}_{sub_key}"] = sub_value
                        else:
                            flat_record[key] = value
                    flattened_data.append(flat_record)

                all_flattened_data.extend(flattened_data)

                # Get the next URL for pagination, if available
                next_url = response_data.get('next_url', None)
            else:
                print(f"Error: Received status code {response.status_code}")
                break

    return pd.DataFrame.from_records(all_flattened_data)


def get_price_history(tickers, delta=180):
    stock_client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    start_date = datetime.datetime.today() - datetime.timedelta(days=delta)
    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start_date
    )
    bars = stock_client.get_stock_bars(request_params)
    return bars.df


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


def gather_dte(tickers):
    df = pd.DataFrame(index=tickers, columns=['DTE'])
    i = 0

    for ticker in tqdm(tickers):
        yf_ticker = yf.Ticker(ticker)
        dates = yf_ticker.earnings_dates
        delta = 0
        if dates is not None:
            future_dates = [x for x in dates.index.tz_localize(None).to_pydatetime().tolist() if
                            x > datetime.datetime.now()]
            if future_dates:
                next_date = future_dates[-1].date()
                delta = next_date - date.today()
                delta = int(delta.days)
        i += 1
        df.loc[ticker, 'DTE'] = delta

    return df


def get_options_stats(tickers, today, write_to_file=False):
    data_col = ['Exp Date', 'Days Left', 'Call Volume', 'Put Volume', 'Call Open Interest', 'Put Open Interest',
                'Call IV', 'Put IV', 'Volume Ratio', 'Open Interest Ratio',
                'Vol Weighted Avg Call Price', 'Vol Weighted Avg Put Price',
                'Open Interest Weighted Avg Call Price', 'Open Interest Weighted Avg Put Price']

    for ticker in tqdm(tickers):

        data_lst = []

        current_ticker = yf.Ticker(ticker)
        options_date = current_ticker.options

        for current_date in tqdm(options_date):
            current_date_dt = datetime.datetime.strptime(current_date, '%Y-%m-%d')
            days_left = (current_date_dt - today).days + 1

            call_data = current_ticker.option_chain(current_date).calls.fillna(0)
            put_data = current_ticker.option_chain(current_date).puts.fillna(0)

            call_volume = call_data['volume'].sum()
            put_volume = put_data['volume'].sum()

            call_open_interest = call_data['openInterest'].sum()
            put_open_interest = put_data['openInterest'].sum()

            call_iv = call_data['impliedVolatility'].mean()
            put_iv = call_data['impliedVolatility'].mean()

            volume_ratio = put_volume / call_volume
            open_interest_ratio = put_open_interest / call_open_interest

            call_volume_weight = call_data['volume'] / call_volume
            put_volume_weight = put_data['volume'] / put_volume

            call_open_interest_weight = call_data['openInterest'] / call_open_interest
            put_open_interest_weight = put_data['openInterest'] / put_open_interest

            if call_volume > 0:
                weighted_avg_call_vol = np.average(call_data['strike'], weights=call_volume_weight)
            else:
                weighted_avg_call_vol = 0
            if put_volume > 0:
                weighted_avg_put_vol = np.average(put_data['strike'], weights=put_volume_weight)
            else:
                weighted_avg_put_vol = 0

            if call_open_interest > 0:
                weighted_avg_call_oi = np.average(call_data['strike'], weights=call_open_interest_weight)
            else:
                weighted_avg_call_oi = 0
            if put_open_interest > 0:
                weighted_avg_put_oi = np.average(put_data['strike'], weights=put_open_interest_weight)
            else:
                weighted_avg_put_oi = 0

            current_data_lst = [current_date, days_left, call_volume, put_volume, call_open_interest,
                                put_open_interest, call_iv, put_iv, volume_ratio, open_interest_ratio,
                                weighted_avg_call_vol, weighted_avg_put_vol, weighted_avg_call_oi,
                                weighted_avg_put_oi]

            data_lst.append(current_data_lst)

        data_df = pd.DataFrame(data_lst, columns=data_col)
        data_file_dir = Path(__file__).parent.parent.parent.parent / 'Shoop Magic'
        data_time = today.strftime('%Y-%m-%d')
        if write_to_file:
            try:
                writer = pd.ExcelWriter(f'{data_file_dir}/options_{data_time}.xlsx', if_sheet_exists='replace',
                                        engine='openpyxl', mode='a')
            except FileNotFoundError:
                writer = pd.ExcelWriter(f'{data_file_dir}/options_{data_time}.xlsx', engine='openpyxl')

            data_df.to_excel(writer, sheet_name=ticker, index=False)
            writer.close()

    return


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
