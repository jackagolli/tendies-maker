import datetime
import os
from pathlib import Path
import time
import requests

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import bs4
import dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import create_engine
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator
from tqdm import tqdm

from src.tendies_maker.utils import pct_to_numeric
from src.tendies_maker.gather import gather_DTE, get_put_call_magnitude, get_call_put_ratio


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

        tickers, df = self.scrape_wsb_tickers()
        price_history = self.get_price_history(tickers)
        data["tickers"] = tickers
        data["raw_data"] = df
        data["price_history"] = price_history
        super().__init__(**data)

    @staticmethod
    def scrape_wsb_tickers():
        res = requests.get('https://apewisdom.io')
        df = pd.read_html(res.text)[0]
        df = df[['#', 'Symbol', 'Mentions', '24h', 'Upvotes']]
        df.rename(columns={'#': 'Rank'}, inplace=True)
        df.dropna(inplace=True)
        df = df[~df["Rank"].str.contains("Close Ad")]
        df["24h"] = pct_to_numeric(df["24h"])
        df.set_index("Symbol", inplace=True)
        return df.index.tolist(), df

    @staticmethod
    def scrape_news_sentiment(tickers):
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
    def get_price_history(tickers):
        stock_client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
        start_date = datetime.datetime.today() - datetime.timedelta(days=180)
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_date
        )
        bars = stock_client.get_stock_bars(request_params)
        return bars.df

    def append_all(self):
        self.append_ta()
        self.append_short_interest()
        self.append_news_sentiment()
        self.append_dte()
        self.append_fluctuations()
        self.append_options_data()
        return None

    def write_data(self):
        engine = create_engine(os.environ["SQLALCHEMY_DATABASE_URI"], echo=True)
        self.raw_data['Date'] = self.date
        self.raw_data.to_sql("raw_data", con=engine, if_exists="append", index=True)
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
        dte = gather_DTE(self.tickers)
        self.raw_data = self.raw_data.merge(dte, how="left", left_index=True, right_index=True)

    def append_fluctuations(self):
        # Max intraday change since tgt_days
        self.price_history['intraday'] = (self.price_history['high'] - self.price_history['open']) / \
                                         self.price_history['open']
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

        return None

    def append_options_data(self):
        df = get_call_put_ratio(self.tickers)
        self.raw_data = self.raw_data.merge(df, how="left", left_index=True, right_index=True)

        df = get_put_call_magnitude(self.tickers)
        self.raw_data = self.raw_data.merge(df, how="left", left_index=True, right_index=True)

        return None