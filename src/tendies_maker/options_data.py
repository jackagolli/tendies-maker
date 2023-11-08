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


@stub.function(schedule=Period(days=1))
def run_daily():
    options_data.remote()
