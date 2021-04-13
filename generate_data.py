import stocks
import datetime as dt
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd
import sys

"""
Gather relevant data for ML model. Try to run before market open, 8:15 am CST at the latest.
"""

# Set data directory here
data_dir = Path.cwd() / 'data'
if data_dir.exists():
    pass
else:
    Path.mkdir(data_dir)

today = dt.date.today().strftime("%m-%d-%Y")
num_proc = mp.cpu_count() - 1

parser = argparse.ArgumentParser()
# parser.add_argument("--save", help="save results to file",
#                     action="store_true")
parser.add_argument("-wsb", help="generates WSB sentiment, this is required for other arguments",
                    action="store_true")
parser.add_argument("-shorts", help="scrapes short interest of most shorted stocks",
                    action="store_true")
parser.add_argument("-indicators",
                    help="calculates embedded technical indicators; BB value, ichimoku cloud, MACD, and RSI  ",
                    action="store_true")
parser.add_argument("-news", help="generates natural language sentiment analysis for recent news headlines",
                    action="store_true")
parser.add_argument("-earnings", help="generates days to upcoming earnings for given tickers",
                    action="store_true")
parser.add_argument("-changes", help="generates data for intraday price changes",
                    action="store_true")
parser.add_argument("-options", help="generates data for call/put ratio and magnitudes",
                    action="store_true")
parser.add_argument("-overwrite", help="flag to not have manual request to overwrite files and overwrite all",
                    action="store_true")
args = parser.parse_args()

if __name__ == '__main__':
    mp.freeze_support()
    if args.overwrite:
        bool = True
    else:
        bool = False
    if args.wsb:
        """ Sentiment analysis
    
        Use the below functions for daily scrape of r/wallstreetbets and calculation of sentiment + change
        """

        stocks.scrape_wsb(data_dir)
        stocks.calc_wsb_daily_change(data_dir,overwrite=bool)

    try:
        tickers = stocks.gather_wsb_tickers(data_dir, today)
        tickers_split = np.array_split(np.asarray(tickers), num_proc)
    except:
        # TODO functionality for custom tickers list
        print("Please generate WSB scrape to determine tickers.")
        sys.exit()

    if args.shorts:
        """ Short interest
    
        Calculate short interest for given ticker or given wsb sheet
    
        """
        # desired_date = '03-11-2021'
        short_interest = stocks.gather_short_interest(data_dir)
        stocks.append_to_table(data_dir, data=short_interest, date_str=today, name="short interest",overwrite=bool)

    if args.changes:
        """ Price fluctuations--
    
        Get max rise in past 30 days and days since last large increase (>7%)
    
        # """
        intraday_change = stocks.find_intraday_change(tickers)
        max_change = pd.DataFrame(intraday_change.max(), columns=['max_intraday_change_1mo'])
        stocks.append_to_table(data_dir, data=max_change, date_str=today, name="max intraday change",overwrite=bool)

        days_since_max = stocks.days_since_max_spike(intraday_change, tickers)
        stocks.append_to_table(data_dir, data=days_since_max, date_str=today, name="days since max spike",overwrite=bool)
        days_since_last = stocks.days_since_last_spike(intraday_change, tickers)
        stocks.append_to_table(data_dir, data=days_since_last, date_str=today, name="days since most recent spike",
                                overwrite=bool)

    if args.indicators:
        """ Technical Indicators
    
        RSI, BB value, MACD, Ichimoku clouds
    
        """
        window = 14
        prices = stocks.gather_multi(tickers, period="1mo")
        rsi = stocks.calc_RSI(tickers, prices, window)
        formatted_rsi = stocks.format_data(rsi, tickers, name="rsi")
        stocks.append_to_table(data_dir, data=formatted_rsi, date_str=today, name="RSI",overwrite=bool)

        sma = stocks.calc_SMA(prices, window=10)
        rstd = stocks.calc_rolling_std(prices, window=10)
        upper_band, lower_band = stocks.calc_bollinger_bands(sma, rstd)
        bb_value = stocks.get_BB(prices, upper_band, lower_band)
        formatted_bb = stocks.format_data(bb_value, tickers, name="bb_val")
        stocks.append_to_table(data_dir, data=formatted_bb, date_str=today, name="BB value",overwrite=bool)

        MACD = stocks.get_MACD(prices)
        formatted_MACD = stocks.format_data(MACD, tickers, name="macd")
        stocks.append_to_table(data_dir, data=formatted_MACD, date_str=today, name="MACD",overwrite=bool)

        prices = stocks.gather_multi(tickers, period="3mo")
        ichi = stocks.get_ichimoku(prices)
        formatted_ichi = stocks.format_data(ichi, tickers, name="ichimoku")
        stocks.append_to_table(data_dir, data=formatted_ichi, date_str=today, name="Ichimoku cloudd",overwrite=bool)

    if args.news:
        # News sentiment
        news_sentiment = stocks.scrape_news_sentiment(tickers)
        stocks.append_to_table(data_dir, data=news_sentiment, date_str=today, name="news sentiment",overwrite=bool)

    if args.options:
        pool = mp.Pool(num_proc)
        print("Creating new Pool for parallelizing runs...")
        with pool:
            print("Pool created...")
            data = pool.map(stocks.get_call_put_ratio, tickers_split)
        data = pd.concat(data, axis=0)
        stocks.append_to_table(data_dir, data, today, name="call-put ratio",overwrite=bool)

        print("Creating new Pool for parallelizing runs...")
        with pool:
            print("Pool created...")
            data = pool.map(stocks.get_put_call_magnitude, tickers_split)
        data = pd.concat(data, axis=0)
        stocks.append_to_table(data_dir, data, today, name="call-put value ratio",overwrite=bool)

    if args.earnings:
        dte = stocks.gather_DTE(tickers)
        stocks.append_to_table(data_dir, dte, today, name="days to earnings",overwrite=bool)
