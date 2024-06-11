from polygon import RESTClient
import pandas as pd

from thinker import append_technical_indicators

client = RESTClient(api_key="VcmqzRfUqWkFDNWwxVosKglCf9LyltUN")

start_date = "2023-01-01"
end_date = "2023-12-31"

# List Aggregates (Bars)
aggs = client.get_aggs(ticker='SPY', multiplier=5, timespan="minute", from_="2022-01-01", to="2023-12-31", limit=50000)

data = [a.__dict__ for a in aggs]
raw_data = pd.DataFrame(data)
raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], unit='ms')

raw_data['date'] = raw_data['timestamp'].dt.date

# Group by date and calculate required aggregates
train_data = raw_data.groupby('date').agg(
    open=('open', 'first'),
    close=('close', 'last'),
    high=('high', 'max'),
    low=('low', 'min'),
    five_min_avg_high=('high', 'mean'),
    five_min_avg_low=('low', 'mean'),
    five_min_avg_volume=('volume', 'mean'),
    five_min_max_volume=('volume', 'max'),
    five_min_min_volume=('volume', 'min'),
    volume=('volume', 'sum')
)

# Calculating Day VWAP
train_data['day_vwap'] = ((raw_data['close'] * raw_data['volume']).groupby(raw_data['date']).sum() /
                          raw_data.groupby('date')['volume'].sum())

# Reset index to make 'date' a column again if needed
train_data.reset_index(inplace=True)
train_data = append_technical_indicators(train_data)

breakpoint()