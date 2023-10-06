import datetime
from email.message import EmailMessage
import io
import os
from pathlib import Path
import requests
import smtplib
import time

import boto3
import bs4
import dateutil.parser
import dotenv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import pandas_datareader as pdr
from pydantic import BaseModel
from sqlalchemy import text
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator
from tqdm import tqdm

from src.tendies_maker.utils import pct_to_numeric
from src.tendies_maker.gather import gather_dte, get_put_call_magnitude, get_call_put_ratio, get_price_history
from src.tendies_maker.db import DB

db = DB()


class TrainingData(BaseModel):
    raw_data: pd.DataFrame
    price_history: pd.DataFrame
    tickers: list[str]
    tgt_pct: float
    tgt_days: int
    normalized_data: pd.DataFrame = None
    date: datetime.datetime = datetime.datetime.today()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        basedir = Path(__file__).resolve().parents[2]
        dotenv_file = os.path.join(basedir, ".env")

        if os.path.isfile(dotenv_file):
            dotenv.load_dotenv(dotenv_file)

        wsb_tickers, df = self.scrape_wsb_tickers()
        if "tickers" not in data.keys():
            data["tickers"] = wsb_tickers
            data["raw_data"] = df
        else:
            data["raw_data"] = pd.DataFrame(index=data["tickers"])
        price_history = get_price_history(data["tickers"])
        data["price_history"] = price_history
        super().__init__(**data)

    @staticmethod
    def scrape_wsb_tickers():
        res = requests.get('https://apewisdom.io')
        df = pd.read_html(res.text)[0]
        df = df[['#', 'Symbol', 'Mentions', '24h', 'Upvotes']]
        df.rename(columns={'#': 'Rank'}, inplace=True)
        df.dropna(inplace=True)
        df = df[~df['Rank'].apply(lambda x: isinstance(x, str) and "Close Ad" in x)]
        df["24h"] = pct_to_numeric(df["24h"])
        df.set_index("Symbol", inplace=True)
        df.drop(columns=["Rank"], inplace=True)
        return df.index.tolist(), df

    @staticmethod
    def scrape_fomc_calendar():
        response = requests.get('https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm')
        html = bs4.BeautifulSoup(response.text, "html.parser")
        latest_year = html.find('div', {'class': 'panel panel-default'})
        year = latest_year.a.text.split()[0]
        day_results = [div.text.rstrip('*').split('-')[0] for div in
                       latest_year.findAll('div', {'class': 'fomc-meeting__date'})]
        month_results = [div.text.rstrip('*').split('/')[0] for div in
                         latest_year.findAll('div', {'class': 'fomc-meeting__month'})]

        upcoming_dates = [dateutil.parser.parse(f'{month} {day} {year}') for month, day in
                          zip(month_results, day_results) if dateutil.parser.parse(f'{month} {day} {year}') >
                          datetime.datetime.today()]

        return upcoming_dates

    @staticmethod
    def scrape_news_sentiment(tickers):
        nltk.download('vader_lexicon')
        df = pd.DataFrame(index=tickers, columns=['News Sentiment'])
        base_url = 'https://finviz.com/quote.ashx?t='
        news_tables = {}

        for ticker in tqdm(tickers):
            url = base_url + ticker
            time.sleep(0.01)
            response = requests.get(url, headers={"User-Agent": "Chrome"})
            html = bs4.BeautifulSoup(response.text, "html.parser")
            news_table = html.find(id='news-table')
            if news_table:
                news_tables[ticker] = news_table

        news_headlines = {}

        for ticker, news_table in tqdm(news_tables.items()):
            news_headlines[ticker] = []
            dates = []

            for i in news_table.findAll('tr'):
                # Strictly get more recent sentiment
                if len(dates) > 4:
                    break

                try:
                    text = i.find('div', {'class': 'news-link-left'}).text
                    date_scrape = i.find('td', {'align': 'right'}).text.split()
                except AttributeError:
                    continue

                if len(date_scrape) != 1:
                    date = date_scrape[0]
                    dates.append(date)

                news_headlines[ticker].append(text)

        vader = SentimentIntensityAnalyzer()

        for ticker, value in tqdm(news_headlines.items()):
            # This is the avg score between -1 and 1 of the all the news headlines
            news_df = pd.DataFrame(news_headlines[ticker], columns=['headline'])
            scores = news_df['headline'].apply(vader.polarity_scores).tolist()
            scores_df = pd.DataFrame(scores)
            score = scores_df['compound'].mean()
            df.loc[ticker, 'News Sentiment'] = score

        df.fillna(0, inplace=True)

        return df

    @staticmethod
    def query_all_data():
        sql = """select * from public.raw_data rd"""
        with db.engine.begin() as conn:
            data = pd.read_sql(text(sql), conn)
        return data

    def calculate_data(self):
        data = self.query_all_data()
        data['Date'] = pd.to_datetime(data['Date']).dt.normalize()

        unique_tickers = data['Symbol'].unique().tolist()
        price_history = get_price_history(unique_tickers).reset_index(names=['Symbol', 'Date'])
        price_history['Date'] = price_history['Date'].dt.tz_localize(None).dt.normalize()

        # Initialize 'Result' column to 'NO'
        data['Result'] = 'NO'

        # Parameters
        window = 7
        pct = 5.0

        def check_price_increase(group):
            ticker = group['Symbol'].iloc[0]
            for idx, row in group.iterrows():
                start_date = row['Date']
                end_date = start_date + pd.Timedelta(days=window)

                # Filter Alpaca data for this ticker within the window
                mask = (price_history['Symbol'] == ticker) & (price_history['Date'] > start_date) & (
                            price_history['Date'] <= end_date)
                window_data = price_history[mask]

                if not window_data.empty:
                    start_price = window_data['close'].iloc[0]
                    max_price = window_data['close'].max()

                    # Check if the stock went up by the given percentage
                    if max_price >= start_price * (1 + pct / 100):
                        data.loc[idx, 'Result'] = 'YES'

        # Apply the function to each group of rows with the same ticker
        data.groupby('Symbol').apply(check_price_increase)
        return data

    @staticmethod
    def email_report(data=None, date=None):
        if data is None and date is None:
            sql = """
            select * from public.raw_data rd inner join
            (select max("Date") as date from raw_data) md on rd."Date" = md.date
            """
            with db.engine.begin() as conn:
                data = pd.read_sql(text(sql), conn, index_col='Symbol')
                date = data['Date'].dt.to_pydatetime()[0].strftime("%m-%d-%Y")
                data.drop(columns=['Date'], inplace=True)
        s3 = boto3.resource('s3',
                            aws_access_key_id=os.environ['ACCESS_KEY'],
                            aws_secret_access_key=os.environ['SECRET_KEY'])
        bucket = s3.Bucket('tendies-maker')
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer)
        filename = f'data-{date}.csv'
        bucket.put_object(Key=filename, Body=csv_buffer.getvalue())

        sender_email = os.environ['FROM_EMAIL']
        password = os.environ['EMAIL_SECRET']

        msg = EmailMessage()
        msg['From'] = "jagolli192@gmail.com"
        msg['To'] = ', '.join(["jagolli192@gmail.com", "rhehdgus10@gmail.com"])
        msg['Subject'] = 'TendiesMaker Daily Report'

        msg.add_attachment(csv_buffer.getvalue(), filename=filename)

        # Create secure connection with server and send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)

        return None

    def write_data(self):
        self.raw_data['Date'] = self.date
        with db.engine.begin() as conn:
            try:
                # this will fail if there is a new column
                self.raw_data.to_sql("raw_data", con=conn, if_exists="append", index=True)
                success = True
            except:
                success = False

        if not success:
            with db.engine.begin() as conn:
                data = pd.read_sql(text("""SELECT * FROM public.raw_data"""), conn)
                data = pd.concat([data, self.raw_data])
                data.to_sql(name='raw_data', con=conn, if_exists='replace', index=False)

        return None

    def append_days_to_fomc(self):
        next_fomc_date = self.scrape_fomc_calendar()[0]
        self.raw_data['Days to FOMC'] = (next_fomc_date - self.date).days

        return None

    def append_macro_econ_data(self):
        today = datetime.datetime.today()
        labels = {'PCE': 'PCE', 'UNRATE': 'Unemployment', 'MICH': 'Inflation Expectation', 'JTSJOL': 'Job Openings'}
        inflation = pdr.data.DataReader(list(labels.keys()), 'fred', today - datetime.timedelta(
            days=180), today)
        inflation = inflation.apply(lambda x: x.shift(x.isnull().sum()), axis=0)
        inflation.rename(columns=labels, inplace=True)
        records = inflation.pct_change().iloc[-1:].to_dict('records')[0]
        self.raw_data = self.raw_data.assign(**records)
        return None

    def append_all(self):
        self.append_ta()
        self.append_macro_econ_data()
        self.append_days_to_fomc()
        self.append_short_interest()
        self.append_fluctuations()
        self.append_news_sentiment()
        self.append_dte()
        self.append_options_data()
        return None

    def append_news_sentiment(self):
        df = self.scrape_news_sentiment(self.tickers)
        self.raw_data = self.raw_data.merge(df, how="left", left_index=True, right_index=True)

    def append_short_interest(self):
        scraped_tables = pd.read_html('https://www.highshortinterest.com/', header=0)
        df = scraped_tables[2]
        df = df[~df['Ticker'].str.contains("google_ad_client", na=False)]
        df.dropna(inplace=True)
        df.set_index('Ticker', inplace=True)
        df = df[['ShortInt']]
        df.rename({'ShortInt': 'Short Interest'}, inplace=True, axis=1)
        df['Short Interest'] = pct_to_numeric(df)

        self.raw_data = self.raw_data.merge(df, how="left", left_index=True, right_index=True)
        self.raw_data.fillna(0, inplace=True)

        return None

    def append_ta(self, sma_windows=None, ema_windows=None):

        if sma_windows is None:
            sma_windows = [50]

        if ema_windows is None:
            ema_windows = [12, 26]

        self.price_history = pd.concat([self.price_history, RSIIndicator(close=self.price_history["close"], window=14,
                                                                         fillna=False).rsi()], axis=1)

        for window in sma_windows:
            self.price_history = pd.concat(
                [self.price_history, SMAIndicator(close=self.price_history["close"], window=window,
                                                  fillna=False).sma_indicator()], axis=1)

        for window in ema_windows:
            self.price_history = pd.concat(
                [self.price_history, EMAIndicator(close=self.price_history["close"], window=window,
                                                  fillna=False).ema_indicator()], axis=1)

        self.price_history = pd.concat([self.price_history, BollingerBands(close=self.price_history["close"], window=20,
                                                                           window_dev=2).bollinger_pband()], axis=1)

        self.price_history = pd.concat([self.price_history, MACD(close=self.price_history["close"]).macd()], axis=1)

        ichi_ind = IchimokuIndicator(high=self.price_history["high"], low=self.price_history["low"])
        self.price_history["ichi"] = ichi_ind.ichimoku_a() - ichi_ind.ichimoku_b()

        self.price_history = pd.concat([self.price_history,
                                        MFIIndicator(high=self.price_history["high"], low=self.price_history["low"],
                                                     close=self.price_history["close"],
                                                     volume=self.price_history["volume"]).money_flow_index()], axis=1)

        for col in self.price_history.columns[7:]:
            df = self.price_history[[col]].groupby(level=0).tail(1).reset_index(level=1, drop=True)
            self.raw_data = self.raw_data.merge(df, how="left", left_index=True, right_index=True)

        return None

    def append_dte(self):
        dte = gather_dte(self.tickers)
        self.raw_data = self.raw_data.merge(dte, how="left", left_index=True, right_index=True)

    def append_fluctuations(self):
        # Max intraday change since tgt_days
        self.price_history['intraday'] = (self.price_history['high'] - self.price_history['open']) / \
                                         self.price_history['open']

        self.price_history['day_change'] = (self.price_history['close'] - self.price_history['open']) / \
                                           self.price_history['open']

        self.price_history['change_since_close'] = self.price_history['close'].diff() / self.price_history[
            'close'].shift(1)

        self.raw_data['Daily Change (Open to Close)'] = self.price_history['day_change'].groupby(level=0).tail(
            1).droplevel(1).fillna(0)
        self.raw_data['Daily Change (Close to Close)'] = self.price_history['change_since_close'].groupby(
            level=0).tail(1).droplevel(1).fillna(0)

        one_month_history = self.price_history[self.price_history.index.get_level_values(1).date >
                                               (datetime.date.today() - datetime.timedelta(days=self.tgt_days))]
        df = one_month_history[['intraday']].max(level=0).rename(
            columns={'intraday': 'Max Intraday Change (1mo)'})

        self.raw_data = self.raw_data.merge(df, how="left", left_index=True, right_index=True)

        # Days since last spike greater than tgt_pct
        df = pd.DataFrame(index=self.tickers, columns=['Days Since Last Spike'])
        days_since_spike_df = self.price_history[self.price_history['intraday'] > self.tgt_pct].groupby(level=0).tail(
            1).reset_index(level=1)[['timestamp']]
        df['Days Since Last Spike'] = (datetime.date.today() - days_since_spike_df['timestamp'].dt.date).dt.days
        df.fillna(0, inplace=True)

        self.raw_data = self.raw_data.merge(df, how="left", left_index=True, right_index=True)
        self.raw_data.fillna(0, inplace=True)
        return None

    def append_options_data(self):
        df = get_call_put_ratio(self.tickers)
        self.raw_data = self.raw_data.merge(df, how="left", left_index=True, right_index=True)

        df = get_put_call_magnitude(self.tickers)
        self.raw_data = self.raw_data.merge(df, how="left", left_index=True, right_index=True)

        return None
