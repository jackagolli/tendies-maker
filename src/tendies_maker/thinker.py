import numpy as np
import scipy.optimize as spo
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
import os
import re
import multiprocessing as mp
from pathlib import Path
from datetime import date
import datetime
import math

from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator, VolumeWeightedAveragePrice
from transformers import pipeline

num_proc = mp.cpu_count() - 2


def news_sentiment_analysis(news):
    nlp = pipeline("text-classification", model="ProsusAI/finbert")
    total_sentiment_score = 0
    total_articles = 0
    lambda_rate = 1e-6
    title_weight = 0.3
    description_weight = 0.7

    # Sentiment Label mapping
    sentiment_map = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }

    for _, article in news.iterrows():

        title = article.get('title')
        description = article.get('description')

        if pd.isna(title) or pd.isna(description):
            continue

        description = description[:512]
        published_time = datetime.datetime.fromisoformat(article['published_utc'].replace('Z', '+00:00'))
        time_delta = (datetime.datetime.now(datetime.timezone.utc) - published_time).total_seconds()

        # Time Decay Factor
        decay_factor = math.exp(-lambda_rate * time_delta)

        # Sentiment analysis
        title_result = nlp(title)
        description_result = nlp(description)

        # Calculate the weighted sentiment and confidence scores
        weighted_sentiment = (title_weight * sentiment_map[title_result[0]['label']] * title_result[0]['score'] +
                              description_weight * sentiment_map[description_result[0]['label']] *
                              description_result[0]['score']) * decay_factor

        total_sentiment_score += weighted_sentiment
        total_articles += 1

    average_sentiment = total_sentiment_score / total_articles if total_articles else 0

    return average_sentiment


def append_technical_indicators(price_history, sma_windows=None, ema_windows=None):
    if sma_windows is None:
        sma_windows = [50]

    if ema_windows is None:
        ema_windows = [12, 26]

    # EMA/SMA
    # for window in sma_windows:
    #     price_history = pd.concat(
    #         [price_history, SMAIndicator(close=price_history["close"], window=window,
    #                                      fillna=False).sma_indicator()], axis=1)
    #
    # for window in ema_windows:
    #     price_history = pd.concat(
    #         [price_history, EMAIndicator(close=price_history["close"], window=window,
    #                                      fillna=False).ema_indicator()], axis=1)

    price_history = pd.concat([price_history, RSIIndicator(close=price_history["close"], window=14,
                                                           fillna=False).rsi()], axis=1)

    price_history = pd.concat([price_history, BollingerBands(close=price_history["close"], window=20,
                                                             window_dev=2).bollinger_pband()], axis=1)

    price_history = pd.concat([price_history, MACD(close=price_history["close"]).macd()], axis=1)

    ichi_ind = IchimokuIndicator(high=price_history["high"], low=price_history["low"])
    price_history["ichi"] = ichi_ind.ichimoku_a() - ichi_ind.ichimoku_b()

    price_history = pd.concat([price_history,
                               MFIIndicator(high=price_history["high"],
                                            low=price_history["low"],
                                            close=price_history["close"],
                                            volume=price_history["volume"]).money_flow_index()], axis=1)

    return price_history


def append_options_metrics(options_df):
    # Filter puts and calls into their own DataFrames
    puts_df = options_df[options_df['details_contract_type'] == 'put']
    calls_df = options_df[options_df['details_contract_type'] == 'call']
    put_call_ratio = len(puts_df) / len(calls_df) if len(calls_df) != 0 else 0

    # Sum up the volumes for puts and calls
    total_put_volume = puts_df['day_volume'].sum()
    total_call_volume = calls_df['day_volume'].sum()

    # Calculate the put-call volume ratio
    put_call_volume_ratio = total_put_volume / total_call_volume if total_call_volume != 0 else 0

    # Avoid division by zero by replacing zero deltas with NaN
    options_df['theta_per_delta'] = options_df['greeks_theta'] / options_df['greeks_delta']
    options_df.loc[options_df['greeks_delta'] == 0, 'theta_per_delta'] = 0
    options_df['volume_to_open_interest'] = options_df['day_volume'] / options_df['open_interest']
    options_df.loc[options_df['open_interest'] == 0, 'volume_to_open_interest'] = 0
    # For call options
    options_df.loc[options_df['details_contract_type'] == 'call', 'moneyness'] = \
        options_df['day_close'] - options_df['details_strike_price']

    # For put options
    options_df.loc[options_df['details_contract_type'] == 'put', 'moneyness'] = \
        options_df['details_strike_price'] - options_df['day_close']

    # Assuming 'details_expiration_date' is in YYYY-MM-DD format and 'day_last_updated' is a timestamp in nanoseconds
    options_df['details_expiration_date'] = pd.to_datetime(options_df['details_expiration_date'])
    options_df['day_last_updated_date'] = pd.to_datetime(options_df['day_last_updated'], unit='ns')

    # Calculate days to expiration
    options_df['days_to_expiration'] = (
            options_df['details_expiration_date'] - options_df['day_last_updated_date']).dt.days

    weighted_avg_vwap = (options_df['day_vwap'] * options_df['day_volume']).sum() / options_df['day_volume'].sum()

    metrics = {
        'put_call_ratio': put_call_ratio,
        'put_call_volume_ratio': put_call_volume_ratio,
        'weighted_avg_vwap': weighted_avg_vwap,
        'avg_vwap': options_df['day_vwap'].mean(),
    }
    return options_df, metrics


def append_fluctuations(price_history):
    # Max intraday change since tgt_days
    price_history['intraday_change'] = (price_history['high'] - price_history['open']) / \
                                price_history['open']

    price_history['day_change'] = (price_history['close'] - price_history['open']) / price_history['open']

    # Calculate max intraday change for each ticker (level 0)
    # Find the date of the max intraday change for each ticker
    date_of_max_intraday = price_history['intraday_change'].groupby(level=0).idxmax()

    # Calculate days since the last max intraday change
    today = datetime.date.today()
    days_since_last_spike = {ticker: (today - pd.Timestamp(date).date()).days for ticker, (_, date) in
                             date_of_max_intraday.items()}

    # Also get the value of the last max intraday spike
    value_of_max_intraday = {ticker: price_history.loc[idx, 'intraday_change'] for
                             ticker, idx in date_of_max_intraday.items()}

    results = {
        'days_since_last_spike': days_since_last_spike,
        'value_of_max_intraday': value_of_max_intraday
    }
    return price_history, results


def thinker(ticker, date, stock_data, options_data, scenario, *args):
    """ get insight into specific option.
      ticker = str of ticker
      date = str of date (YYYY-MM-DD)
      scenario = nominal, bullish, bearish
      insight_date = date when news anticipated to affect stock price
      duration = how long it will have an effect, multiple of weeks
      """
    if args:
        insight_date = args[0]
        duration = args[1]

    today = datetime.date.today()
    end_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    latest_price = stock_data[ticker]['end_price']
    weekly_std = stock_data[ticker]["weekly"]['std']
    daily_mu = stock_data[ticker]["daily"]['mean']
    daily_std = stock_data[ticker]["daily"]['std']

    # TODO determine which distribution fits data best? then determine probability of price point.

    if scenario == 'nominal':
        # all days
        # dt = int((end_date - today).days)
        # business days only
        dt = np.busday_count(today, datetime.datetime.strptime(date, "%Y-%m-%d").date())
        # conservative outlook, use 75% of avg and BD only.
        mu = daily_mu * 0.75
        # mu = daily_mu
        x_f = latest_price * (1 + mu) ** dt

    # bullish news
    elif scenario == 'bullish':

        try:

            # calc price up until insight date.
            insight_date = datetime.datetime.strptime(insight_date, "%Y-%m-%d").date()
            # dt_1 = int((insight_date - today).days)
            dt_1 = np.busday_count(today, insight_date)
            mu = 0.75 * daily_mu
            x_1 = latest_price * (1 + mu) ** dt_1
            x_2 = x_1 * (1 + weekly_std) ** duration

            # dt_2 = int((end_date - (insight_date + datetime.timedelta(weeks=duration))).days)
            dt_2 = np.busday_count((insight_date + datetime.timedelta(weeks=duration)), end_date)
            x_f = x_2 * (1 + mu) ** dt_2


        except:

            print("Error. Make sure correct arguments entered.")

        # x_1 = stock_data[ticker]['end_price'] * (1 + weekly_std)
        # dt = int((datetime.datetime.strptime(date, "%Y-%m-%d").date() - (today + datetime.timedelta(days=7))).days)
        # x_f =  x_1 * (1 + daily_mu)**dt

    # bearish news.
    elif scenario == 'bearish':

        pass

    else:

        print("Error. This capability has not been added yet.")

    total_change = (x_f - latest_price) / latest_price * 100
    print(f"scenario = {scenario}")
    print('current price:', latest_price)
    print(f'predicted price on {date}:', x_f)
    print(f'total % change: {total_change}')
    print(f'weekly_std = {weekly_std} | daily_mu = {daily_mu} | daily_std = {daily_std}')
    print(f'expiration date is {date}')
    print(options_data[date][['strike', 'lastPrice', 'volume', 'impliedVolatility']])

    return None


def optimize(syms, data, start_val, gen_plot=False):
    """
    Optimize for maximum Sharpe ratio.
    """
    x = np.array([1 / float(len(syms)) for symbol in syms])
    normed = data.divide(data.iloc[0])

    def sharpe(x):
        rf = 0.0
        k = np.sqrt(252)
        alloced = normed.mul(x)
        pos_vals = alloced.mul(start_val)
        port_val = pos_vals.sum(axis=1)
        daily_returns = (port_val / port_val.shift(1)) - 1
        daily_returns = daily_returns[1:]

        # Modified Sharpe Ratio to remove dependence of std
        return -1 * k * daily_returns.sub(rf).mean() / (daily_returns.std())

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    result = spo.minimize(sharpe, x, method='SLSQP', bounds=tuple((0.0, 1) for symbol in syms),
                          constraints=constraints)
    sr = result.fun * -1
    allocs = result.x
    alloced = normed.mul(allocs)
    pos_vals = alloced.mul(start_val)
    port_val = pos_vals.sum(axis=1)
    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()

    vals = allocs * start_val
    vals = dict(zip(syms, vals))

    if gen_plot:
        pd.options.plotting.backend = "plotly"
        port_norm = port_val.divide(port_val.iloc[0])
        port_norm = pd.DataFrame(port_norm, columns=['Portfolio'])
        df_temp = normed.join(port_norm)
        # df_temp['date'] = df_temp.index
        fig = go.Figure()
        for col in df_temp.columns:
            fig.add_trace(go.Scatter(x=df_temp.index, y=df_temp[col], name=col))
        fig.show()

        # matplotlib stuff
        # ax = df_temp.plot()
        # ax.set_ylabel("Value")
        # date_form = DateFormatter("%b %Y")
        # ax.xaxis.set_major_formatter(date_form)
        # plt.legend(loc="upper left")
        # plt.grid()
        # plt.show()

    return allocs, cr, adr, sddr, sr, vals


def calc_portfolio(tickers, allocs, total, buy_amount):
    new_total = total + buy_amount
    vals = np.array(allocs) * new_total
    vals = dict(zip(tickers, vals))

    return vals


def calc_intraday_change(tickers, data):
    for x in tickers:
        high = yf.Ticker(x).history(period="1mo")[['High']]
        open = yf.Ticker(x).history(period="1mo")[['Open']]
        try:
            change = (high - open.values) / open.values
        except:
            change = 0

        data[x] = change
        # if (change > 0.07).any()[0]:
        #     data[x] = change
        # else:
        #     pass
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)
    return data


def calc_wsb_daily_change(data_dir, overwrite=False):
    files = []
    today = date.today().strftime("%m-%d-%Y")
    print('Calculating change since previous file...')
    for file in os.listdir(data_dir):

        match = re.search(r'\d\d-\d\d-\d\d\d\d', file)

        if match and any(x in file for x in ['wsb', 'sentiment']):
            files.append(file)
            # df = pd.read_csv(data_dir / file, index_col=0, dtype={"mentions": np.int32})

    files = sorted(files, key=lambda x: datetime.datetime.strptime(
        re.search(r'\d\d-\d\d-\d\d\d\d', x)[0], "%m-%d-%Y"), reverse=True)

    try:
        df1 = pd.read_csv(data_dir / files[0], index_col=0, dtype={"mentions": np.int32})
        df2 = pd.read_csv(data_dir / files[1], index_col=0, dtype={"mentions": np.int32})
        df1.insert(0, column='rank', value=np.arange(1, len(df1) + 1))
        df2['rank'] = np.arange(1, len(df2) + 1)

        temp = df2['rank'] - df1['rank']
        temp = temp.dropna()

        df1['change'] = temp
        df1 = df1.fillna(0)
        df1 = df1.astype({'change': 'int32'})

        if Path(data_dir / ("wsb_sentiment_" + today + ".csv")).is_file() and not overwrite:
            overwrite = input('WSB sentiment data already exists for today in /data dir. '
                              'Overwrite with change since most recent file? (Y/N)? ')

            if overwrite == 'Y' or overwrite == 'y':
                df1.to_csv(data_dir / ('wsb_sentiment_' + today + '.csv'))
            elif overwrite == 'N' or overwrite == 'n':
                pass
            else:
                print('Invalid input.')

        else:
            df1.to_csv(data_dir / ('wsb_sentiment_' + today + '.csv'))

    except:
        print('No scrapes found from previous days.')

    return None


def find_intraday_change(tickers=None):
    if not tickers:
        tickers1 = pd.read_csv('data/nasdaq_stocks_filtered.csv', sep=',')
        tickers2 = pd.read_csv('data/nyse_stocks.csv', sep=',')
        tickers = pd.concat([tickers1, tickers2])
        tickers = tickers['Symbol']

    # today = datetime.date.today()

    try:
        dtype = tickers.dtype
        tickers = tickers.to_numpy()

    except:
        tickers = np.asarray(tickers, dtype='object')

    tickers_split = np.array_split(tickers, num_proc)

    # Initialize empty df with correct row date indexes
    data = yf.Ticker('AAPL').history(period='1mo')
    data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], inplace=True)

    cases = [(stocks, data) for stocks in tickers_split]
    pool = mp.Pool(num_proc)
    results = pool.starmap(calc_intraday_change, cases)
    pool.close()
    pool.join()

    results = pd.concat(results, axis=1)

    return results


def days_since_max_spike(intraday_change, tickers):
    today = np.datetime64(date.today())
    df = pd.DataFrame(intraday_change.idxmax(axis=0), index=tickers, columns=['days_since_max_rise'])
    df['days_since_max_rise'] = today - df['days_since_max_rise']
    df['days_since_max_rise'] = df['days_since_max_rise'].dt.days.astype('int32')

    return df


def days_since_last_spike(intraday_change, tickers):
    today = np.datetime64(date.today())
    intraday_change[intraday_change < 0.06] = 0

    df = pd.DataFrame(index=tickers, columns=['days_since_last_rise'])

    for ticker in tickers:
        for item in np.flip(intraday_change[ticker].values):
            if item != 0:
                try:
                    last_date = intraday_change[ticker][intraday_change[ticker] == item].index[0]
                    delta = today - last_date
                    delta = int(delta.days)
                    df.loc[ticker, 'days_since_last_rise'] = delta
                    break
                except:
                    continue

    df = df.fillna(0)

    return df


def append_to_table(data_dir, data, date_str, name="", overwrite=False):
    """
    Append a formatted df to the main data table.

    :param data_dir: Path() object leading to data subdirectory
    :param data: The actual dataframe
    :param date_str: A %m-%d-%Y formatted str date for saving the new file
    :param name: Name to show in print statement for data type
    :param overwrite: Flag whether to skip manual input for overwriting files
    :return: none, saves .csv file
    """
    data = data.fillna(0)

    label = data.columns.values[0]
    save_path = Path(data_dir / ("data_" + date_str + ".csv"))

    try:
        data_df = pd.read_csv(save_path, header=0, index_col=0)

    except:

        data_df = pd.read_csv(data_dir / ('wsb_sentiment_' + date_str + '.csv'), header=0, index_col=0)

    data_df[label] = np.zeros(len(data_df))

    tickers = data_df.index.values
    data_filtered = data[data.index.isin(tickers)]
    data_tickers = data_filtered.index.values
    data_filtered = data_filtered[label].values

    for i in range(len(data_tickers)):
        if data_tickers[i] in tickers:
            data_df.loc[data_tickers[i], label] = data_filtered[i]

    save_path = Path(data_dir / ("data_" + date_str + ".csv"))

    if save_path.is_file() and not overwrite:
        overwrite = input(f'Data file already exists for {date_str}. Overwrite with {name} data? (Y/N) ')

        if overwrite == 'Y' or overwrite == 'y':

            data_df.to_csv(save_path)
        elif overwrite == 'N' or overwrite == 'n':
            pass
        else:
            print('Invalid input.')
    else:
        data_df.to_csv(save_path)

    return None


def format_data(data, tickers, name):
    df = pd.DataFrame(index=tickers, columns=[name])
    for ticker in tickers:
        for item in np.flip(data[('Adj Close', ticker)].values):
            if item != 0:
                df.loc[ticker, name] = item
                break

    return df


def calc_RSI(tickers, prices, window):
    delta = prices.diff()
    up = delta[delta > 0]
    up = up.fillna(0)
    down = delta[delta < 0].abs()
    down = down.fillna(0)

    avg_up = up.rolling(window).mean()
    avg_down = down.rolling(window).mean()

    RS = avg_up / avg_down

    RSI = 100 - (100 / (1 + RS))

    RSI = RSI.fillna(0)

    return RSI


def calc_SMA(prices, window):
    """Return rolling mean of given values, using specified window size."""
    return prices.rolling(window).mean()


def calc_rolling_std(prices, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return prices.rolling(window).std()


def calc_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band


def get_BB(values, upper_band, lower_band):
    BB = (values - lower_band) / (upper_band - lower_band)
    return BB


def get_MACD(prices):
    # Get EMA using pandas
    ema_12 = prices.ewm(span=12).mean()
    ema_26 = prices.ewm(span=26).mean()
    MACD_line = ema_12 - ema_26
    signal_line = MACD_line.ewm(span=9).mean()
    MACD_hist = MACD_line - signal_line

    return MACD_hist


def get_ichimoku(prices):
    high_kenkan = prices.rolling(9).max()
    low_kenkan = prices.rolling(9).min()
    high_kijun = prices.rolling(26).max()
    low_kijun = prices.rolling(26).min()
    high_senkou_B = prices.rolling(52).max()
    low_senkou_B = prices.rolling(52).min()

    conversion_line = (high_kenkan + low_kenkan) / 2
    base_line = (high_kijun + low_kijun) / 2

    # Cloud formed by A - B. If A > B, cloud is green and bullish. If A < B, cloud is red and bearish.
    # So if ichimoku val is negative its bearish if positive it's bullish
    leading_span_A = (conversion_line + base_line) / 2
    leading_span_B = (high_senkou_B + low_senkou_B) / 2

    diff = leading_span_A - leading_span_B

    return diff


def standard_score_normalize(df, ignored_columns=None):
    initial_df = df

    try:
        df = initial_df.drop(columns=ignored_columns)

    except:
        print("One or more of the specified columns don't exist.")

    diff = initial_df.columns.difference(df.columns)
    dropped_cols = initial_df[diff]

    df = (df - df.mean()) / df.std()

    df = pd.concat([df, dropped_cols], axis=1)

    return df


def min_max_normalize(df, ignored_columns=None):
    initial_df = df
    initial_index = df.index

    try:
        df = initial_df.drop(columns=ignored_columns)

    except:
        print("One or more of the specified columns don't exist.")

    diff = initial_df.columns.difference(df.columns)
    dropped_cols = initial_df[diff]
    dropped_col_names = list(dropped_cols.columns.values)

    if 'Y' in ignored_columns:
        dropped_col_names.pop(dropped_col_names.index('Y'))
        dropped_cols = dropped_cols[dropped_col_names + ['Y']]

    new_cols = df.columns

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), index=initial_index, columns=new_cols)

    df = pd.concat([df, dropped_cols], axis=1)

    return df
