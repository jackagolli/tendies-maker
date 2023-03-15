from src.tendies_maker.datamodel import TrainingData
from src.tendies_maker.gather import get_options_stats
from datetime import datetime

# get today's date

# td = TrainingData(tgt_pct=0.05, tgt_days=30)

today = datetime.today()
options_tickers = ['SPY', 'AAPL', 'TSLA', 'META', 'GOOG', 'MSFT', 'AMZN', 'NVDA', 'BRK-B', 'XOM']
earning_tickers = ['FIVE', 'LAC', 'LICY', 'GME']

tickers = options_tickers + earning_tickers

test_options = get_options_stats(tickers, today, write_to_file=True)
