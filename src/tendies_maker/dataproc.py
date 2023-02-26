import datetime
import os
from pathlib import Path
import requests

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import dotenv
import pandas as pd
from pydantic import BaseModel

from utils import pct_to_numeric


class TrainingData(BaseModel):
    raw_data: pd.DataFrame
    price_history = pd.DataFrame
    tickers: list[str]
    normalized_data: pd.DataFrame = None
    date = datetime.datetime.today().strftime("%m-%d-%Y")

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
    def get_price_history(tickers):
        stock_client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
        start_date = datetime.datetime.today() - datetime.timedelta(days=90)
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_date
        )
        bars = stock_client.get_stock_bars(request_params)
        return bars.df

    def add_short_interest(self):
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
