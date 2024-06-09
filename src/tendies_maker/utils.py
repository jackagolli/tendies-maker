import numpy as np
import requests
import pandas as pd
from datetime import datetime
import pytz

polygon_api = "https://api.polygon.io"


def pct_to_numeric(series):
    pct = series.values.astype('str')
    pct = np.char.replace(pct, ',', '')
    pct = np.char.strip(pct, chars='%')
    pct = pct.astype(np.float32)
    pct = pct / 100

    return pct


def is_market_open(time):
    open_time = datetime.strptime('09:30:00', '%H:%M:%S').time()
    close_time = datetime.strptime('16:00:00', '%H:%M:%S').time()

    # Parse the given time string
    current_time = datetime.strptime(time, '%H:%M:%S').time()

    # Check if the given time is within the period
    return open_time <= current_time <= close_time


def convert_timestamp(time):
    timestamp_sec = time / 1000
    ts = pd.Timestamp(timestamp_sec, unit='s', tz='US/Eastern')

    return ts.strftime('%Y-%m-%d %X')


def create_mock_raw_data(api_key, ticker, timespan, multiplier, start_date, end_date):
    query = f"{polygon_api}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}" \
            f"?adjusted=true&sort=asc&apiKey={api_key}"

    return requests.get(query)


def parse_mock_raw_dat(data, include_after_hour, save_to_file, save_dir):
    output_df = pd.DataFrame(columns=["time", "open", "close", "volume", "vw"])

    for candle in data:
        current_date = convert_timestamp(candle['t'])
        if include_after_hour:
            output_df.loc[len(output_df)] = [current_date, candle['o'], candle['c'],
                                             candle['v'], candle['vw']]
        else:
            _, current_time = current_date.split()
            if is_market_open(current_time):
                output_df.loc[len(output_df)] = [current_date, candle['o'], candle['c'],
                                                 candle['v'], candle['vw']]

    if save_to_file:
        output_df.to_parquet(save_dir)

    return output_df


