from src.tendies_maker.datamodel import TrainingData
from src.tendies_maker.gather import get_options_stats
from datetime import datetime

# get today's date

# td = TrainingData(tgt_pct=0.05, tgt_days=30)

today = datetime.today()
test_options = get_options_stats(['SPY', 'AAPL'], today, write_to_file=True)
