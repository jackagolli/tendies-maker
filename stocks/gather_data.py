import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

today = datetime.date.today()
companyTicker = 'AAPL'
company = yf.Ticker(companyTicker)
data = company.history(period="1y", interval="1d")
start_price = data["Close"][0]
end_price = data["Close"][-1]


daily_close = data['Close']
daily_pct_change = daily_close.pct_change()
print(daily_pct_change)
# Replace NA values with 0
daily_pct_change.fillna(0, inplace=True)

daily_pct_change.hist(bins=50)
plt.show()
print(daily_pct_change.describe())


# Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)

monthly = data.resample('BM').apply(lambda x: x[-1])

cum_daily_return = (1 + daily_pct_change).cumprod()
cum_daily_return.plot(figsize=(12,8))

cum_monthly_return = cum_daily_return.resample("M").mean()
print(cum_monthly_return)
# Show the plot
plt.show()

# data = yf.download('AAPL','2016-01-01','2018-01-01')
# data.Close.plot()
# plt.show()
