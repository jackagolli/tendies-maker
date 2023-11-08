from pathlib import Path

from modal import Stub, Volume, Image, Mount, Secret, Period

stub = Stub("options-data")
historical_data_image = (
    Image.debian_slim()
    .apt_install('libpq-dev')
    .pip_install("pandas")
    .pip_install("requests")
    .pip_install("alpaca-py")
    .pip_install("beautifulsoup4")
    .pip_install("pandas-datareader")
    .pip_install("yfinance")
    .pip_install("tqdm")
    .pip_install("python-dotenv")
    .pip_install("SQLAlchemy")
    .pip_install("psycopg2")
    .pip_install("pyarrow")
    .pip_install("fastparquet")
    .pip_install("duckdb")
)
VOLUME_DIR = "/tm-data"
stub.volume = Volume.persisted('tm-data-vol')


@stub.function(image=historical_data_image, volumes={VOLUME_DIR: stub.volume},
               mounts=[Mount.from_local_python_packages("gather", "db")],
               secret=Secret.from_name("tm-secrets"))
def options_data():
    import pandas as pd
    from tqdm import tqdm
    from gather import get_price_history, get_options_chain, get_options_data

    ticker = 'SPY'
    price_history = get_price_history([ticker], None)

    min_date = price_history.index.get_level_values('timestamp').min().strftime("%Y-%m-%d")
    max_date = price_history.index.get_level_values('timestamp').max().strftime('%Y-%m-%d')

    earliest_options = get_options_chain(ticker, min_date)
    latest_options = get_options_chain(ticker, max_date)
    all_relevant_options = pd.concat([earliest_options, latest_options]).drop_duplicates()

    all_options_data = []

    for option_ticker in tqdm(all_relevant_options['ticker']):
        option_data = get_options_data(option_ticker, min_date, max_date)
        all_options_data.append(option_data)

    # Concatenate all the fetched data
    full_options_data_df = pd.concat(all_options_data)
    full_options_data_df.to_parquet(Path(VOLUME_DIR, "options_data.parquet"))
    stub.volume.commit()


@stub.function(image=historical_data_image, volumes={VOLUME_DIR: stub.volume},
               mounts=[Mount.from_local_python_packages("gather", "db", "thinker", "datamodel")],
               secret=Secret.from_name("tm-secrets"))
def training_data():
    import datetime
    import pytz

    import pandas as pd
    from tqdm import tqdm
    tqdm.pandas()

    from datamodel import TrainingData as td
    from gather import get_price_history, get_news, get_dividends, get_market_holidays, get_macro_econ_data
    from thinker import append_sentiment_analysis, append_days_to_holiday, append_dividend_yield, \
        append_technical_indicators, append_fluctuations

    ticker = 'SPY'

    price_history = get_price_history([ticker], None)
    price_history = price_history.sort_index(level=1)
    price_history = append_technical_indicators(price_history)
    price_history, fluctuations = append_fluctuations(price_history)
    timestamps = price_history.index.get_level_values('timestamp').normalize().tz_localize(None)

    # Macro econ data
    econ = get_macro_econ_data(False)
    econ = econ.resample('D').ffill().bfill()
    price_history['date_for_merge'] = timestamps
    price_history = price_history.merge(econ, left_on='date_for_merge', right_index=True, how='left')

    price_history.drop(columns=['date_for_merge'], inplace=True)
    price_history.fillna(method='ffill', inplace=True)

    # FOMC dates
    fomc_dates = pd.to_datetime(td.scrape_fomc_calendar(False))
    next_fomc_indices = fomc_dates.searchsorted(timestamps)
    days_to_fomc = [(fomc_dates[next_fomc_indices[i]] - timestamps[i]).days
                    for i in range(len(timestamps))]
    is_fomc_date = timestamps.isin(fomc_dates)
    price_history['days_to_fomc'] = days_to_fomc
    price_history.loc[is_fomc_date, 'days_to_fomc'] = 0

    # Get and merge latest options data
    options_df = pd.read_parquet("options_data.parquet")

    agg_data = options_df.groupby(['date', 'contract_type']).agg(
        total_volume=('v', 'sum'),
    ).unstack()
    agg_data['pc_ratio_volume'] = agg_data[('total_volume', 'put')] / agg_data[('total_volume', 'call')]
    agg_data.columns = ['_'.join(col).strip() for col in agg_data.columns.values]

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

    closed_dates = get_market_holidays()
    price_history = append_days_to_holiday(price_history, closed_dates)

    dividends_df = get_dividends(ticker)
    price_history = append_dividend_yield(price_history, dividends_df)

    news = get_news(ticker)
    price_history['news_sentiment'] = price_history.progress_apply(lambda row: append_sentiment_analysis(row, news),
                                                                   axis=1)
    price_history.to_parquet(Path(VOLUME_DIR, "training_data.parquet"))
    stub.volume.commit()


@stub.function(schedule=Period(days=1), timeout=3600)
def daily_data_run():
    options_data.remote()
    training_data.remote()


@stub.function(schedule=Period(days=1), timeout=3600)
def email_snapshot():
    options_data.remote()
    training_data.remote()


@stub.local_entrypoint()
def main():
    training_data.local()
