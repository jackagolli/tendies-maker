import datetime
import yfinance as yf
import matplotlib.pyplot as plt



companyTicker = 'AMD'
company = yf.Ticker(companyTicker)
historicalData = company.history(period="1yr")

data = yf.download('AAPL','2016-01-01','2018-01-01')
data.Close.plot()
plt.show()
