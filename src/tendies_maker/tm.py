from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI
from modal import Stub, Volume, Image, Mount, Secret, Cron, asgi_app, web_endpoint

stub = Stub("tm")
image = (
    Image.debian_slim()
    .apt_install('libpq-dev')
    .pip_install("pandas", "plotly", "keras", "keras", "tensorflow", "pyyaml", "h5py", "requests",
                 "alpaca-py", "beautifulsoup4", "pandas-datareader", "yfinance", "tqdm", "python-dotenv", "psycopg2",
                 "pyarrow", "SQLAlchemy", "fastparquet", "duckdb", "boto3", "pretty-html-table", "nltk", "loguru",
                 "ta", "diffusers[torch]", "transformers", "ftfy", "accelerate", "scikit-learn", "pydantic", "fastapi",
                 "joblib")
)
VOLUME_DIR = "/tm-data"
stub.volume = Volume.persisted('tm-data-vol')
web_app = FastAPI()


class Prediction(BaseModel):
    volume: float
    trade_count: float
    vwap: float
    rsi: float
    bbipband: float
    MACD_12_26: float
    ichi: float
    tenkan_kijun_cross: int
    price_vs_senkou_a: float
    price_vs_senkou_b: float
    mfi_14: float
    intraday_change: float
    days_since_last_spike: int
    PCE: float
    unemployment: float
    inflation_expectation: float
    job_openings: float
    fed_funds_rate: float
    real_m2: float
    real_gdp: float
    retail_sales: float
    existing_home_sales: float
    days_to_fomc: int
    pc_ratio_volume: float
    day_of_week: str
    days_to_next_holiday: int
    dividend_yield: float
    time_since_last_dividend: int
    news_sentiment: float


def _preprocess_data(df):
    import pandas as pd
    day_of_week = pd.get_dummies(df['day_of_week'], prefix='day_of_week', dtype=float)
    df = pd.concat([df, day_of_week], axis=1).drop('day_of_week', axis=1)
    # Ensure all day columns are present
    all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    for day in all_days:
        column_name = f'day_of_week_{day}'
        if column_name not in df.columns:
            df[column_name] = 0

    return df


@stub.function(image=image, volumes={VOLUME_DIR: stub.volume},
               mounts=[Mount.from_local_python_packages("gather", "db", "config", "utils")],
               secret=Secret.from_name("tm-secrets"),
               timeout=3600)
def options_data():
    import pandas as pd
    from tqdm import tqdm
    from gather import get_price_history, get_options_chain, get_options_data
    from config import logger

    ticker = 'SPY'
    logger.info(f'Fetching price history for {ticker}')
    price_history = get_price_history([ticker], None)

    min_date = price_history.index.get_level_values('timestamp').min().strftime("%Y-%m-%d")
    max_date = price_history.index.get_level_values('timestamp').max().strftime('%Y-%m-%d')

    logger.info(f'Fetching {ticker} options data for dates {min_date} to {max_date}')
    earliest_options = get_options_chain(ticker, min_date)
    latest_options = get_options_chain(ticker, max_date)
    all_relevant_options = pd.concat([earliest_options, latest_options]).drop_duplicates()

    all_options_data = []

    for option_ticker in tqdm(all_relevant_options['ticker']):
        option_data = get_options_data(option_ticker, min_date, max_date)
        all_options_data.append(option_data)

    full_options_data_df = pd.concat(all_options_data)
    full_options_data_df = full_options_data_df.merge(all_relevant_options[['ticker', 'contract_type']],
                                                      left_on='ticker', right_on='ticker', how='left')
    logger.info(f'Saving options data to parquet')
    full_options_data_df.to_parquet(Path(VOLUME_DIR, "options_data.parquet"))
    stub.volume.commit()


@stub.function(image=image, volumes={VOLUME_DIR: stub.volume},
               mounts=[Mount.from_local_python_packages("gather", "db", "thinker", "datamodel", "config",
                                                        "utils")],
               secret=Secret.from_name("tm-secrets"),
               timeout=3600)
def training_data():
    import pandas as pd
    from tqdm import tqdm
    tqdm.pandas()

    from config import logger
    from datamodel import TrainingData as td
    from gather import get_price_history, get_news, get_dividends, get_market_holidays, get_macro_econ_data
    from thinker import append_sentiment_analysis, append_days_to_holiday, append_dividend_yield, \
        append_technical_indicators, append_fluctuations

    ticker = 'SPY'

    logger.info(f'Fetching price history for {ticker}')
    price_history = get_price_history([ticker], None)
    price_history = price_history.sort_index(level=1)

    logger.info(f'Appending {ticker} technical indicators')
    price_history = append_technical_indicators(price_history)

    logger.info(f'Appending {ticker} price fluctuations')
    price_history, fluctuations = append_fluctuations(price_history)
    timestamps = price_history.index.get_level_values('timestamp').normalize().tz_localize(None)

    logger.info(f'Appending macroeconomic indicators')
    econ = get_macro_econ_data(False)
    econ = econ.resample('D').ffill().bfill()
    price_history['date_for_merge'] = timestamps
    price_history = price_history.merge(econ, left_on='date_for_merge', right_index=True, how='left')

    price_history.drop(columns=['date_for_merge'], inplace=True)
    price_history.fillna(method='ffill', inplace=True)

    logger.info(f'Appending days to FOMC')
    fomc_dates = pd.to_datetime(td.scrape_fomc_calendar(False))
    next_fomc_indices = fomc_dates.searchsorted(timestamps)
    days_to_fomc = [(fomc_dates[next_fomc_indices[i]] - timestamps[i]).days
                    for i in range(len(timestamps))]
    is_fomc_date = timestamps.isin(fomc_dates)
    price_history['days_to_fomc'] = days_to_fomc
    price_history.loc[is_fomc_date, 'days_to_fomc'] = 0

    logger.info(f'Appending latest options data p/c ratio')
    options_df = pd.read_parquet(Path(VOLUME_DIR, "options_data.parquet"))

    agg_data = options_df.groupby(['date', 'contract_type']).agg(
        total_volume=('v', 'sum'),
    ).unstack()
    agg_data['pc_ratio_volume'] = agg_data[('total_volume', 'put')] / agg_data[('total_volume', 'call')]
    agg_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_data.columns.values]

    agg_data.reset_index(inplace=True)
    price_history.reset_index(inplace=True)
    agg_data['date'] = pd.to_datetime(agg_data['date']).dt.tz_localize('UTC')
    price_history = price_history.merge(agg_data[['pc_ratio_volume_', 'date']], left_on=['timestamp'],
                                        right_on=['date'],
                                        how='left')
    price_history.drop(columns=['date'], inplace=True)
    price_history.set_index(['symbol', 'timestamp'], inplace=True)
    price_history.dropna(inplace=True)

    price_history['day_of_week'] = price_history.index.get_level_values('timestamp').day_name()

    logger.info(f'Appending days to holiday')
    closed_dates = get_market_holidays()
    price_history = append_days_to_holiday(price_history, closed_dates)

    logger.info(f'Appending {ticker} dividend yield')
    dividends_df = get_dividends(ticker)
    price_history = append_dividend_yield(price_history, dividends_df)

    logger.info(f'Appending news sentiment')
    news = get_news(ticker)
    price_history['news_sentiment'] = price_history.progress_apply(lambda row: append_sentiment_analysis(row, news),
                                                                   axis=1)
    price_history.to_parquet(Path(VOLUME_DIR, "training_data.parquet"))
    stub.volume.commit()


@stub.function(image=image, volumes={VOLUME_DIR: stub.volume})
@web_endpoint(method="POST")
def predict(prediction: Prediction):
    import joblib
    import numpy as np
    import pandas as pd
    from tensorflow import keras

    df = pd.DataFrame([prediction.dict()])
    df = _preprocess_data(df)
    scaler = joblib.load(Path(VOLUME_DIR,'tm_scaler.joblib'))
    X = scaler.transform(df)
    dr = joblib.load(Path(VOLUME_DIR,'tm_pca.joblib'))
    X_transformed = dr.transform(X)
    model = keras.models.load_model(Path(VOLUME_DIR, 'tm_basic_nn.keras'))
    pred = model.predict(X_transformed)
    pred = (pred > 0.5)
    result = 'BUY' if pred[0][0] else 'NOT BUY'
    return {'signal': result}


@stub.function(image=image, volumes={VOLUME_DIR: stub.volume},
               secret=Secret.from_name("tm-secrets"), schedule=Cron("0 13 * * 1-5"),
               mounts=[Mount.from_local_python_packages("gather", "db", "thinker", "datamodel", "config",
                                                        "utils")],
               )
def email_snapshot():
    import datetime
    from email.message import EmailMessage
    import os
    import requests
    import smtplib

    from alpaca.data import StockHistoricalDataClient, StockLatestBarRequest
    import pandas as pd
    from pretty_html_table import build_table

    from datamodel import TrainingData as td
    from gather import (get_dividends, get_news, get_market_holidays, get_options_snapshot, get_macro_econ_data,
                        get_price_history)
    from thinker import news_sentiment_analysis, append_technical_indicators, append_fluctuations

    ticker = 'SPY'

    options_df = get_options_snapshot(ticker)
    put_options = options_df[options_df['details_contract_type'] == 'put']
    call_options = options_df[options_df['details_contract_type'] == 'call']
    pc_volume_ratio = put_options['day_volume'].sum() / call_options['day_volume'].sum()
    pc_open_interest_ratio = put_options['open_interest'].sum() / call_options['open_interest'].sum()
    net_delta = call_options['greeks_delta'].sum() + put_options['greeks_delta'].sum()


    stock_client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    bars = stock_client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols='SPY'))
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples([(ticker, bars['SPY'].timestamp)],
                                                      names=['symbol', 'timestamp']))
    df['close'] = bars['SPY'].close
    df['open'] = bars['SPY'].open
    df['high'] = bars['SPY'].high
    df['low'] = bars['SPY'].low
    df['volume'] = bars['SPY'].volume
    df['trade_count'] = bars['SPY'].trade_count
    df['vwap'] = bars['SPY'].vwap

    price_history = get_price_history([ticker], 90)
    price_history = pd.concat([price_history, df])
    price_history = append_technical_indicators(price_history)
    price_history, result = append_fluctuations(price_history)
    current = price_history.iloc[-1].to_dict()


    dividends_df = get_dividends(ticker).iloc[0]
    dividend_yield = (dividends_df['cash_amount'] * dividends_df['frequency'] / bars[ticker].close) * 100
    days_since_last_dividend = (datetime.datetime.today() - datetime.datetime.strptime(
        dividends_df['ex_dividend_date'], "%Y-%m-%d")).days

    econ = get_macro_econ_data(False)
    econ = econ.resample('D').ffill().bfill().ffill()
    econ = econ.dropna().iloc[-1].to_dict()
    fomc_dates = pd.to_datetime(td.scrape_fomc_calendar(True))
    days_to_fomc = (fomc_dates[0] - datetime.datetime.today()).days + 1
    today = datetime.datetime.now()
    if today.weekday() == 5:
        last_weekday = today - datetime.timedelta(days=1)
    elif today.weekday() == 6:
        last_weekday = today - datetime.timedelta(days=2)
    else:
        last_weekday = today
    day_of_week = last_weekday.strftime('%A')

    closed_dates = get_market_holidays()
    future_dates = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in closed_dates]
    future_dates.sort()
    for date in future_dates:
        if date > datetime.datetime.today().date():
            days_to_holiday = (date - datetime.datetime.today().date()).days
            break
    news = get_news(ticker)
    news_sentiment = news_sentiment_analysis(news, datetime.datetime.now())

    prediction = Prediction(
        volume=current['volume'],
        trade_count=current['trade_count'],
        vwap=current['vwap'],
        rsi=current['rsi'],
        bbipband=current['bbipband'],
        MACD_12_26=current['MACD_12_26'],
        ichi=current['ichi'],
        tenkan_kijun_cross=current['tenkan_kijun_cross'],
        price_vs_senkou_a=current['price_vs_senkou_a'],
        price_vs_senkou_b=current['price_vs_senkou_b'],
        mfi_14=current['mfi_14'],
        intraday_change=current['intraday_change'],
        days_since_last_spike=current['days_since_last_spike'],
        PCE=econ['PCE'],
        unemployment=econ['unemployment'],
        inflation_expectation=econ['inflation_expectation'],
        job_openings=econ['job_openings'],
        fed_funds_rate=econ['fed_funds_rate'],
        real_m2=econ['real_m2'],
        real_gdp=econ['real_gdp'],
        retail_sales=econ['retail_sales'],
        existing_home_sales=econ['existing_home_sales'],
        days_to_fomc=days_to_fomc,
        pc_ratio_volume=pc_volume_ratio,
        days_to_next_holiday=days_to_holiday,
        dividend_yield=dividend_yield,
        time_since_last_dividend=days_since_last_dividend,
        news_sentiment=news_sentiment,
        day_of_week=day_of_week,
    )

    response = requests.post("https://jackagolli--tm-predict.modal.run/", json=prediction.dict(), timeout=20.0).json()
    signal = response['signal']
    df = td.tail(2).transpose()
    final_data_html = build_table(df, 'blue_light', index=True)
    sender_email = os.environ['FROM_EMAIL']
    password = os.environ['EMAIL_SECRET']

    msg = EmailMessage()
    msg['From'] = "jagolli192@gmail.com"
    msg['To'] = ', '.join(["jagolli192@gmail.com", "rhehdgus10@gmail.com"])
    # msg['To'] = ', '.join(["jagolli192@gmail.com"])

    if signal == "BUY":
        signal_style = "color: green; font-weight: bold;"
    else:
        signal_style = "color: black; font-weight: bold;"

    msg['Subject'] = 'TendiesMaker Report'
    html = f"""
    <html>
      <head>
        <style>
          table {{
            width: 100%;
            border-collapse: collapse;
          }}
          th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
          }}
          th {{
            background-color: #f2f2f2;
          }}
        </style>
      </head>
      <body>
        <h1>SPY Summary</h1>
        Signal: <div style="{signal_style}">{signal}</div>
        <table>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
          <tr>
            <td>Put/Call Volume Ratio</td>
            <td>{pc_volume_ratio:.2f}</td>
          </tr>
          <tr>
            <td>Put/Call Open Interest Ratio</td>
            <td>{pc_open_interest_ratio:.2f}</td>
          </tr>
          <tr>
            <td>Net Delta Positioning</td>
            <td>{net_delta:.2f}</td>
          </tr>
        </table>
        {final_data_html}
      </body>
    </html>
    """

    # csv_buffer = io.StringIO()
    # msg.add_attachment(csv_buffer.getvalue(), filename=filename)
    msg.set_content(html, subtype='html')
    # Create secure connection with server and send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.send_message(msg)


@stub.function(image=image, volumes={VOLUME_DIR: stub.volume},
               secret=Secret.from_name("tm-secrets"))
def train():
    import joblib
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.regularizers import l1, l2
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler


    df = pd.read_parquet(Path(VOLUME_DIR,"training_data.parquet"))
    df = _preprocess_data(df)

    # define target
    look_forward_days = 3  # You can change this to 2 or 3 as per your strategy
    price_increase_threshold = 0.01  # Define what you consider as a significant increase, e.g., 1%
    df_reversed = df.iloc[::-1]

    # Apply the rolling function on the reversed DataFrame
    df_reversed['max_future_high'] = df_reversed['high'].rolling(window=look_forward_days, min_periods=1).max()

    # Reverse the DataFrame back to original order
    df['max_future_high'] = df_reversed['max_future_high'].iloc[::-1]

    # Calculate the target
    df['target'] = ((df['max_future_high'] - df['open']) / df['open']) >= price_increase_threshold
    df['target'] = df['target'].astype(int)
    df = df.iloc[:-1]

    Y = df['target'].to_numpy()
    # drop last row
    df.drop(columns=['target', 'max_future_high', 'close', 'open', 'high', 'low'], inplace=True)
    X = df.to_numpy()
    # normalize
    scaler = StandardScaler()
    scaler.fit(X)
    X_std = scaler.transform(X)

    joblib.dump(scaler, Path(VOLUME_DIR,'tm_scaler.joblib'))

    # Compute PCA
    pca = PCA()
    pca.fit(X_std)
    # Determine the number of components that explain at least 95% of the variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(cum_var >= 0.95) + 1

    # Fit PCA with optimal components
    dr_algorithm = PCA(n_components=optimal_components)
    X_transformed = dr_algorithm.fit_transform(X_std)
    joblib.dump(dr_algorithm, Path(VOLUME_DIR,'tm_pca.joblib'))

    def create_nn_model(optimizer='adam', l1_value=0.01, l2_value=0.01, dropout_rate=0.5):
        model = Sequential()
        model.add(Dense(32, activation='relu', kernel_regularizer=l1(l1_value), activity_regularizer=l2(l2_value)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    model = create_nn_model()

    early_stopping = EarlyStopping(monitor='loss', patience=3)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001)

    model.fit(X_transformed, Y, epochs=20, batch_size=12, callbacks=[early_stopping, reduce_lr])

    model.save(Path(VOLUME_DIR, 'tm_basic_nn.keras'))
    stub.volume.commit()


@stub.function(image=image)
@asgi_app()
def fastapi_app():
    return web_app


@stub.function(schedule=Cron("30 23 * * 1-5"), timeout=3600)
def daily_data_run():
    options_data.remote()
    training_data.remote()
    train.remote()


@stub.local_entrypoint()
def main():
    train.remote()
