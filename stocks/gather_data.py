import datetime
import yfinance as yf




companyTicker = 'AMD'
company = yf.Ticker(companyTicker)
historicalData = company.history(period="1yr")
print(company.info)

